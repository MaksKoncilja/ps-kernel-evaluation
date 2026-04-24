from __future__ import annotations
from turtle import position
from typing import Any, NamedTuple
import time
import os

# METAL seems to be problematic on my laptop, so keeping it on CPU is a good idea
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.persistent_sampling as persistent_sampling
import blackjax.smc.solver as smc_solver
import blackjax.mcmc.diffusions as diffusions
from blackjax.smc import extend_params

from targets import Target, make_gaussian_mixture_target


class InnerAdaptationInfo(NamedTuple):
    update_info: Any
    adapted_parameter: Any


def _scalarize_positive_parameter(x: Any) -> jnp.ndarray:
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    return jnp.asarray(arr.reshape(-1)[0], dtype=arr.dtype)


def make_rm_update_strategy(adapt_one_step_fn):
    """
    Returns a BlackJAX-compatible update_strategy.

    adapt_one_step_fn must have signature:
        (current_param, acceptance_value, t) -> new_param
    """

    def update_strategy(mcmc_init_fn, logposterior_fn, mcmc_step_fn, num_mcmc_steps, n_particles):
        def mcmc_kernel(rng_key, position, mcmc_parameters):
            states = jax.vmap(lambda x: mcmc_init_fn(x, logposterior_fn))(position)
            current_parameters = dict(mcmc_parameters)
            last_update_info = None

            keys = rng_key  # already shape (num_particles, 2)
            for mcmc_iter in range(num_mcmc_steps):
                keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
                subkeys = keys[:, 0]
                keys = keys[:, 1]

                states, update_info = mcmc_step_fn(
                    subkeys,
                    states,
                    logposterior_fn,
                    **current_parameters,
                )
                last_update_info = update_info

                acceptance_value = _safe_mean_acceptance(update_info)
                current_parameters = adapt_one_step_fn(
                    current_parameters,
                    acceptance_value,
                    mcmc_iter + 1,
                )

            positions = states.position if hasattr(states, "position") else states
            info = InnerAdaptationInfo(
                update_info=last_update_info,
                adapted_parameter=current_parameters,
            )
            return positions, info

        return mcmc_kernel, n_particles

    return update_strategy


def make_bisection_update_strategy(target_acceptance_rate: float, tolerance: float = 0.03):
    """
    BlackJAX-compatible update_strategy for HMC/NUTS.

    Improvements kept deliberately simple:
    - adapt only every max(1, num_mcmc_steps // 5) inner steps
    - stop adapting after bisection reports convergence
    - keep Python loop (no scan) for robustness
    """

    def update_strategy(mcmc_init_fn, logposterior_fn, mcmc_step_fn, num_mcmc_steps, n_particles):
        bisection_update = blackjax.adaptation.step_size.bisection_monotonic_fn(
            acc_prob_wanted=target_acceptance_rate,
            tolerance=tolerance,
        )
        adapt_every = max(1, num_mcmc_steps // 5)

        def mcmc_kernel(rng_key, position, mcmc_parameters):
            states = jax.vmap(lambda x: mcmc_init_fn(x, logposterior_fn))(position)
            current_parameters = dict(mcmc_parameters)
            last_update_info = None

            # reset bisection state each tempering iteration
            bisection_state = (
                jnp.array([-jnp.inf, jnp.inf], dtype=jnp.float32),
                False,
            )

            keys = rng_key  # already shape (num_particles, 2)
            for mcmc_iter in range(num_mcmc_steps):
                keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
                subkeys = keys[:, 0]
                keys = keys[:, 1]

                states, update_info = mcmc_step_fn(
                    subkeys,
                    states,
                    logposterior_fn,
                    **current_parameters,
                )
                last_update_info = update_info

                should_attempt_adapt = ((mcmc_iter + 1) % adapt_every) == 0
                if should_attempt_adapt and (not bool(bisection_state[1])):
                    acceptance_value = _safe_mean_acceptance(update_info)
                    if bool(jnp.isfinite(acceptance_value)):
                        current_step_size = jnp.asarray(
                            current_parameters["step_size"][0], dtype=jnp.float32
                        )
                        bisection_state, new_step_size = bisection_update(
                            bisection_state,
                            current_step_size,
                            acceptance_value,
                        )
                        current_parameters["step_size"] = jnp.full_like(
                            current_parameters["step_size"],
                            jnp.asarray(new_step_size, dtype=jnp.float32),
                        )

            positions = states.position if hasattr(states, "position") else states
            info = InnerAdaptationInfo(
                update_info=last_update_info,
                adapted_parameter=current_parameters,
            )
            return positions, info

        return mcmc_kernel, n_particles

    return update_strategy


def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted, dtype=jnp.float32))
    return jnp.nan


def _empirical_covariance(
    particles: jnp.ndarray,
    ridge: float = 1e-6,
    diagonal_only: bool = False,
) -> jnp.ndarray:
    x = jnp.asarray(particles)
    mean = jnp.mean(x, axis=0, keepdims=True)
    xc = x - mean
    n = x.shape[0]
    denom = jnp.maximum(n - 1, 1)
    cov = (xc.T @ xc) / denom
    if diagonal_only:
        cov = jnp.diag(jnp.diag(cov))
    d = cov.shape[0]
    cov = cov + ridge * jnp.eye(d)
    return cov


def _proposal_sqrt_from_cov(cov: jnp.ndarray, scale: float) -> jnp.ndarray:
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(
    t: int,
    c: float = 1.0,
    t0: float = 10.0,
    kappa: float = 0.6,
) -> float:
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(
    rw_scale: jnp.ndarray,
    acceptance_value: float,
    target_acceptance_rate: float,
    t: int,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> jnp.ndarray:
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


def _update_step_size_robbins_monro(
    step_size: jnp.ndarray,
    acceptance_value: float,
    target_acceptance_rate: float,
    t: int,
    rm_c: float = 1.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> jnp.ndarray:
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_step_size = jnp.log(step_size)
    log_step_size = log_step_size + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_step_size)


def _compute_posterior_moments_from_particles(
    particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def _normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    w = jnp.asarray(weights, dtype=jnp.float32)
    return w / jnp.sum(w)


def _variance_log_weights(weights: jnp.ndarray) -> float:
    w = _normalize_weights(weights)
    eps = 1e-32
    logw = jnp.log(w + eps)
    return float(jnp.var(logw))


def _weight_entropy(weights: jnp.ndarray) -> float:
    w = _normalize_weights(weights)
    eps = 1e-32
    return float(-jnp.sum(w * jnp.log(w + eps)))


def _population_esjd(before_particles: jnp.ndarray, after_particles: jnp.ndarray) -> float:
    dx = jnp.asarray(after_particles) - jnp.asarray(before_particles)
    sqdist = jnp.sum(dx ** 2, axis=1)
    return float(jnp.mean(sqdist))


def _diagonal_inverse_mass_matrix(particles: jnp.ndarray, ridge: float = 1e-6) -> jnp.ndarray:
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0, ddof=1)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    var = var + ridge
    return 1.0 / var


def _dimension_scaled_ula_step_size_from_particles(
    particles: jnp.ndarray,
    tempering_param: float,
    adaptation_scale: float = 1.0,
    min_step_size: float = 1e-6,
    max_step_size: float = 1.0,
    variance_ridge: float = 1e-8,
    tempering_floor: float = 1e-3,
) -> jnp.ndarray:
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    mean_var = jnp.mean(var) + variance_ridge

    beta_factor = jnp.sqrt(
        jnp.maximum(
            1.0 - jnp.asarray(tempering_param, dtype=jnp.float32),
            jnp.asarray(tempering_floor, dtype=jnp.float32),
        )
    )

    eps = adaptation_scale * beta_factor / jnp.sqrt(mean_var)

    return jnp.clip(
        jnp.asarray(eps, dtype=jnp.float32),
        jnp.asarray(min_step_size, dtype=jnp.float32),
        jnp.asarray(max_step_size, dtype=jnp.float32),
    )


def _find_reasonable_initial_step_size(
    *,
    rng_key: jax.Array,
    kernel_generator,
    reference_state,
    initial_step_size: float | jnp.ndarray,
    target_acceptance_rate: float,
) -> jnp.ndarray:
    try:
        step_size = blackjax.adaptation.step_size.find_reasonable_step_size(
            rng_key=rng_key,
            kernel_generator=kernel_generator,
            reference_state=reference_state,
            initial_step_size=jnp.asarray(initial_step_size, dtype=jnp.float32),
            target_accept=target_acceptance_rate,
        )
        return jnp.asarray(step_size, dtype=jnp.float32)
    except Exception:
        return jnp.asarray(initial_step_size, dtype=jnp.float32)


def _run_ps_generic_once(
    *,
    target: Target,
    num_particles: int,
    seed: int,
    max_iterations: int,
    alpha: float,
    num_mcmc_steps: int,
    build_ps_kernel_fn,
    init_kernel_params_fn,
    adapt_kernel_params_fn,
    gradient_eval_increment_fn,
    kernel_name: str,
) -> dict[str, Any]:
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    kernel_params = init_kernel_params_fn(initial_particles)

    kernel = build_ps_kernel_fn(
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        kernel_params=kernel_params,
    )

    state = kernel.init(initial_particles)

    tempering_path, logZ_path, ess_path, acceptance_path, step_size_path = [], [], [], [], []
    elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path = [], [], [], []

    start = time.perf_counter()
    n_iter, gradient_eval_count, resampling_steps = 0, 0, 0

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, subkey = jax.random.split(loop_key)

        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        kernel = build_ps_kernel_fn(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            max_iterations=max_iterations,
            alpha=alpha,
            num_mcmc_steps=num_mcmc_steps,
            kernel_params=kernel_params,
        )

        state, info = kernel.step(subkey, state)
        outer_update_info = getattr(info, "update_info", None)

        adapted_parameter = getattr(outer_update_info, "adapted_parameter", None)
        if adapted_parameter is not None:
            adapted_parameter = dict(adapted_parameter)
            if "step_size" in adapted_parameter:
                adapted_parameter["step_size"] = _scalarize_positive_parameter(
                    adapted_parameter["step_size"]
                )
            kernel_params = {**kernel_params, **adapted_parameter}

        update_info = getattr(outer_update_info, "update_info", outer_update_info)

        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.persistent_weights)

        ess_value = float(1.0 / jnp.sum(_normalize_weights(weights) ** 2))
        acceptance_value = float(_safe_mean_acceptance(update_info))
        elapsed = time.perf_counter() - start

        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z if hasattr(state, "log_Z") else state.persistent_log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)

        if "step_size" in kernel_params:
            step_size_path.append(float(_scalarize_positive_parameter(kernel_params["step_size"])))
        elif "rw_scale" in kernel_params:
            step_size_path.append(float(_scalarize_positive_parameter(kernel_params["rw_scale"])))
        else:
            step_size_path.append(np.nan)

        resampling_steps += 1
        nuts_num_steps = getattr(update_info, "num_integration_steps", None)
        if nuts_num_steps is not None:
            gradient_eval_count += int(jnp.sum(jnp.asarray(nuts_num_steps)))
        else:
            gradient_eval_count += int(
                gradient_eval_increment_fn(
                    num_particles=num_particles,
                    num_mcmc_steps=num_mcmc_steps,
                    kernel_params=kernel_params,
                )
            )

        kernel_params = adapt_kernel_params_fn(
            kernel_params=kernel_params,
            particles=particles,
            acceptance_value=acceptance_value,
            tempering_param=float(state.tempering_param),
            t=n_iter + 1,
        )
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles(final_particles)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_array = np.asarray(acceptance_path, dtype=float)
    finite_acceptance = acceptance_array[np.isfinite(acceptance_array)]
    acceptance_rate_last = float(finite_acceptance[-1]) if finite_acceptance.size > 0 else np.nan
    acceptance_rate_mean = float(finite_acceptance.mean()) if finite_acceptance.size > 0 else np.nan

    return {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": kernel_name,
        "seed": int(seed),
        "dimension": int(target.dimension),
        "num_particles": int(num_particles),
        "logZ": float(
            final_state.log_Z
            if hasattr(final_state, "log_Z")
            else state.log_Z
            if hasattr(state, "log_Z")
            else state.persistent_log_Z
        ),
        "posterior_mean": posterior_mean,
        "posterior_second_moment": posterior_second_moment,
        "final_ess": final_ess,
        "acceptance_rate_mean": acceptance_rate_mean,
        "acceptance_rate_last": acceptance_rate_last,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": np.asarray(acceptance_path, dtype=float),
        "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
        "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
        "esjd_path": np.asarray(esjd_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
    }


def _run_ps_generic_once_with_inner_adaptation(
    *,
    target: Target,
    num_particles: int,
    seed: int,
    max_iterations: int,
    alpha: float,
    num_mcmc_steps: int,
    init_kernel_params_fn,
    build_single_step_fns,
    adapt_kernel_params_inner_fn,
    adapt_kernel_params_outer_fn,
    gradient_eval_increment_fn,
    kernel_name: str,
) -> dict[str, Any]:
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    kernel_params = init_kernel_params_fn(initial_particles)
    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    tempering_path, logZ_path, ess_path, acceptance_path, step_size_path = [], [], [], [], []
    elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path = [], [], [], []

    start = time.perf_counter()
    n_iter, gradient_eval_count, resampling_steps = 0, 0, 0
    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    def calculate_lambda(ps_state):
        n_particles_local = ps_state.persistent_weights.shape[1]
        target_val = jnp.log(n_particles_local * alpha)
        max_delta = 1 - ps_state.tempering_schedule[ps_state.iteration]

        def fun_to_solve(delta):
            log_weights, _ = persistent_sampling.compute_log_persistent_weights(
                ps_state.persistent_log_likelihoods,
                ps_state.persistent_log_Z,
                ps_state.tempering_schedule.at[ps_state.iteration + 1].set(
                    ps_state.tempering_schedule[ps_state.iteration] + delta
                ),
                ps_state.iteration + 1,
                normalize_to_one=True,
            )
            ess_val = jnp.log(persistent_sampling.compute_persistent_ess(log_weights))
            return ess_val - target_val

        delta = jnp.nan_to_num(smc_solver.dichotomy(fun_to_solve, 0.0, max_delta))
        return ps_state.tempering_schedule[ps_state.iteration] + jnp.clip(delta, 0.0, max_delta)

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, resample_key, step_key = jax.random.split(loop_key, 3)

        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        next_lambda = calculate_lambda(state)
        iteration = state.iteration + 1
        tempering_schedule = state.tempering_schedule.at[iteration].set(next_lambda)

        log_persistent_weights, log_Z_t = persistent_sampling.compute_log_persistent_weights(
            state.persistent_log_likelihoods,
            state.persistent_log_Z,
            tempering_schedule,
            iteration,
            normalize_to_one=True,
        )
        persistent_weights = jnp.exp(log_persistent_weights)

        resampled_particles, _ = persistent_sampling.resample_from_persistent(
            resample_key,
            state.persistent_particles,
            persistent_weights,
            resampling.systematic,
        )

        init_fn, single_step_fn = build_single_step_fns()

        def logposterior_fn(x):
            return log_prior_fn(x) + next_lambda * log_likelihood_fn(x)

        mcmc_states = jax.vmap(lambda position: init_fn(position, logposterior_fn))(resampled_particles)

        last_acceptance_value = np.nan
        last_update_info = None
        for mcmc_iter in range(num_mcmc_steps):
            step_key, one_key = jax.random.split(step_key)
            keys = jax.random.split(one_key, num_particles)
            mcmc_states, update_info = single_step_fn(keys, mcmc_states, logposterior_fn, kernel_params)
            last_update_info = update_info
            last_acceptance_value = float(_safe_mean_acceptance(update_info))
            positions = mcmc_states.position if hasattr(mcmc_states, "position") else mcmc_states
            kernel_params = adapt_kernel_params_inner_fn(
                kernel_params=kernel_params,
                particles=positions,
                acceptance_value=last_acceptance_value,
                t=mcmc_iter + 1,
            )

        iteration_particles = mcmc_states.position if hasattr(mcmc_states, "position") else mcmc_states
        iteration_log_likelihoods = vmapped_log_likelihood_fn(iteration_particles)

        persistent_particles = jax.tree.map(
            lambda persistent, iteration_p: persistent.at[iteration].set(iteration_p),
            state.persistent_particles,
            iteration_particles,
        )
        persistent_log_Z = state.persistent_log_Z.at[iteration].set(log_Z_t)
        persistent_log_likelihoods = state.persistent_log_likelihoods.at[iteration].set(iteration_log_likelihoods)
        state = persistent_sampling.PersistentSMCState(
            persistent_particles=persistent_particles,
            persistent_log_likelihoods=persistent_log_likelihoods,
            persistent_log_Z=persistent_log_Z,
            tempering_schedule=tempering_schedule,
            iteration=iteration,
        )

        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.persistent_weights)

        ess_value = float(1.0 / jnp.sum(_normalize_weights(weights) ** 2))
        elapsed = time.perf_counter() - start

        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(last_acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)

        if "step_size" in kernel_params:
            step_size_path.append(float(_scalarize_positive_parameter(kernel_params["step_size"])))
        elif "rw_scale" in kernel_params:
            step_size_path.append(float(_scalarize_positive_parameter(kernel_params["rw_scale"])))
        else:
            step_size_path.append(np.nan)

        resampling_steps += 1
        nuts_num_steps = getattr(last_update_info, "num_integration_steps", None)
        if nuts_num_steps is not None:
            gradient_eval_count += int(jnp.sum(jnp.asarray(nuts_num_steps)))
        else:
            gradient_eval_count += int(
                gradient_eval_increment_fn(
                    num_particles=num_particles,
                    num_mcmc_steps=num_mcmc_steps,
                    kernel_params=kernel_params,
                )
            )

        kernel_params = adapt_kernel_params_outer_fn(
            kernel_params=kernel_params,
            particles=particles,
            acceptance_value=last_acceptance_value,
            t=n_iter + 1,
        )
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles(final_particles)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_array = np.asarray(acceptance_path, dtype=float)
    finite_acceptance = acceptance_array[np.isfinite(acceptance_array)]
    acceptance_rate_last = float(finite_acceptance[-1]) if finite_acceptance.size > 0 else np.nan
    acceptance_rate_mean = float(finite_acceptance.mean()) if finite_acceptance.size > 0 else np.nan

    return {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": kernel_name,
        "seed": int(seed),
        "dimension": int(target.dimension),
        "num_particles": int(num_particles),
        "logZ": float(final_state.log_Z),
        "posterior_mean": posterior_mean,
        "posterior_second_moment": posterior_second_moment,
        "final_ess": final_ess,
        "acceptance_rate_mean": acceptance_rate_mean,
        "acceptance_rate_last": acceptance_rate_last,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": np.asarray(acceptance_path, dtype=float),
        "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
        "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
        "esjd_path": np.asarray(esjd_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
    }


# =========================
# PS + RW
def run_ps_rwm_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    rw_step_size: float = 1.0,
    target_acceptance_rate: float = 0.234,
    covariance_ridge: float = 1e-6,
    diagonal_only_covariance: bool = False,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> dict[str, Any]:
    if target is None:
        target = make_gaussian_mixture_target(dimension)
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def init_kernel_params_fn(initial_particles):
        return {
            "rw_scale": jnp.asarray(rw_step_size, dtype=jnp.float32),
            "proposal_cov": _empirical_covariance(
                initial_particles,
                ridge=covariance_ridge,
                diagonal_only=diagonal_only_covariance,
            ),
        }

    def build_single_step_fns():
        def init_fn(position, logdensity_fn):
            return blackjax.rmh.init(position, logdensity_fn)

        def step_fn(keys, states, logdensity_fn, kernel_params):
            def one(key, state):
                transition_generator = blackjax.mcmc.random_walk.normal(
                    _proposal_sqrt_from_cov(kernel_params["proposal_cov"], kernel_params["rw_scale"])
                )
                return base_rmh_kernel(key, state, logdensity_fn, transition_generator=transition_generator)
            return jax.vmap(one)(keys, states)

        return init_fn, step_fn

    def adapt_inner(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)
        proposed_rw_scale = _update_rw_scale_robbins_monro(
            rw_scale=new_params["rw_scale"],
            acceptance_value=acceptance_value,
            target_acceptance_rate=target_acceptance_rate,
            t=t,
            rm_c=rm_c,
            rm_t0=rm_t0,
            rm_kappa=rm_kappa,
        )
        new_params["rw_scale"] = jnp.where(
            jnp.isfinite(acceptance_value),
            proposed_rw_scale,
            new_params["rw_scale"],
        )
        return new_params

    def adapt_outer(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)
        new_params["proposal_cov"] = _empirical_covariance(
            particles,
            ridge=covariance_ridge,
            diagonal_only=diagonal_only_covariance,
        )
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return 0

    return _run_ps_generic_once_with_inner_adaptation(
        target=target,
        num_particles=num_particles,
        seed=seed,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        build_single_step_fns=build_single_step_fns,
        init_kernel_params_fn=init_kernel_params_fn,
        adapt_kernel_params_inner_fn=adapt_inner,
        adapt_kernel_params_outer_fn=adapt_outer,
        gradient_eval_increment_fn=gradient_eval_increment_fn,
        kernel_name="rwm",
    )


# =========================
# PS + HMC
def run_ps_hmc_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 0.1,
    num_integration_steps: int = 10,
    target_acceptance_rate: float = 0.651,
    mass_matrix_ridge: float = 1e-6,
) -> dict[str, Any]:
    if target is None:
        target = make_gaussian_mixture_target(dimension)

    base_hmc_kernel = blackjax.hmc.build_kernel()

    def init_kernel_params_fn(initial_particles):
        inverse_mass_matrix = _diagonal_inverse_mass_matrix(
            initial_particles,
            ridge=mass_matrix_ridge,
        )
        reference_position = jnp.asarray(initial_particles[0])
        reference_state = blackjax.hmc.init(reference_position, target.log_prior_fn)
        warmup_key = jax.random.fold_in(jax.random.PRNGKey(seed), 101)

        def kernel_generator(eps):
            def kernel(key, state):
                return base_hmc_kernel(
                    key,
                    state,
                    target.log_prior_fn,
                    step_size=eps,
                    inverse_mass_matrix=inverse_mass_matrix,
                    num_integration_steps=int(num_integration_steps),
                )
            return kernel

        initial_step = _find_reasonable_initial_step_size(
            rng_key=warmup_key,
            kernel_generator=kernel_generator,
            reference_state=reference_state,
            initial_step_size=step_size,
            target_acceptance_rate=target_acceptance_rate,
        )

        return {
            "step_size": initial_step,
            "inverse_mass_matrix": inverse_mass_matrix,
            "num_integration_steps": int(num_integration_steps),
        }

    def vmapped_hmc_step_fn(
        keys,
        states,
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
    ):
        def one(key, state, eps):
            return base_hmc_kernel(
                key,
                state,
                logdensity_fn,
                step_size=eps,
                inverse_mass_matrix=inverse_mass_matrix,
                num_integration_steps=int(num_integration_steps),
            )
        return jax.vmap(one)(keys, states, step_size)

    hmc_update_strategy = make_bisection_update_strategy(
        target_acceptance_rate=target_acceptance_rate,
    )

    def build_ps_kernel_fn(
        *,
        log_prior_fn,
        log_likelihood_fn,
        max_iterations,
        alpha,
        num_mcmc_steps,
        kernel_params,
    ):
        hmc_parameters = {
            "step_size": jnp.full(
                (num_particles,),
                jnp.asarray(kernel_params["step_size"], dtype=jnp.float32),
            ),
            "inverse_mass_matrix": jnp.asarray(
                [kernel_params["inverse_mass_matrix"]],
                dtype=jnp.float32,
            ),
            "num_integration_steps": jnp.asarray(
                [kernel_params["num_integration_steps"]],
                dtype=jnp.int32,
            ),
        }

        return blackjax.adaptive_persistent_sampling_smc(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=vmapped_hmc_step_fn,
            mcmc_init_fn=blackjax.hmc.init,
            mcmc_parameters=hmc_parameters,
            resampling_fn=resampling.systematic,
            target_ess=alpha,
            num_mcmc_steps=num_mcmc_steps,
            update_strategy=hmc_update_strategy,
        )

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, tempering_param, t):
        new_params = dict(kernel_params)
        new_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(
            particles,
            ridge=mass_matrix_ridge,
        )
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return (
            int(num_particles)
            * int(num_mcmc_steps)
            * int(kernel_params["num_integration_steps"])
        )

    return _run_ps_generic_once(
        target=target,
        num_particles=num_particles,
        seed=seed,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        build_ps_kernel_fn=build_ps_kernel_fn,
        init_kernel_params_fn=init_kernel_params_fn,
        adapt_kernel_params_fn=adapt_kernel_params_fn,
        gradient_eval_increment_fn=gradient_eval_increment_fn,
        kernel_name="hmc",
    )


class _ULAState(NamedTuple):
    position: Any
    logdensity: float
    logdensity_grad: Any


# ===================
# PS + ULA
def run_ps_ula_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 5e-3,
    ula_adaptation_scale: float = 1.0,
    ula_min_step_size: float = 1e-6,
    ula_max_step_size: float = 1.0,
) -> dict[str, Any]:
    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def ula_init_fn(position, logdensity_fn):
        grad_fn = jax.value_and_grad(logdensity_fn)
        logdensity, logdensity_grad = grad_fn(position)
        return _ULAState(position, logdensity, logdensity_grad)

    def ula_step_fn(rng_key, state, logdensity_fn, step_size):
        grad_fn = jax.value_and_grad(logdensity_fn)
        one_step = diffusions.overdamped_langevin(grad_fn)
        new_state = one_step(rng_key, state, step_size)
        return _ULAState(*new_state), None

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32)}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        ula_parameters = extend_params({"step_size": jnp.asarray(kernel_params["step_size"])})

        return blackjax.adaptive_persistent_sampling_smc(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=ula_step_fn,
            mcmc_init_fn=ula_init_fn,
            mcmc_parameters=ula_parameters,
            resampling_fn=resampling.systematic,
            target_ess=alpha,
            num_mcmc_steps=num_mcmc_steps,
        )

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, tempering_param, t):
        new_params = dict(kernel_params)
        new_params["step_size"] = _dimension_scaled_ula_step_size_from_particles(
            particles,
            tempering_param=tempering_param,
            adaptation_scale=ula_adaptation_scale,
            min_step_size=ula_min_step_size,
            max_step_size=ula_max_step_size,
        )
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return int(num_particles) * int(num_mcmc_steps)

    return _run_ps_generic_once(
        target=target,
        num_particles=num_particles,
        seed=seed,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        build_ps_kernel_fn=build_ps_kernel_fn,
        init_kernel_params_fn=init_kernel_params_fn,
        adapt_kernel_params_fn=adapt_kernel_params_fn,
        gradient_eval_increment_fn=gradient_eval_increment_fn,
        kernel_name="ula",
    )

# =================
# PS + MALA
def run_ps_mala_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 0.02,
    target_acceptance_rate: float = 0.574,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> dict[str, Any]:
    if target is None:
        target = make_gaussian_mixture_target(dimension)

    base_mala_kernel = blackjax.mala.build_kernel()

    def init_kernel_params_fn(initial_particles):
        return {
            "step_size": jnp.asarray(step_size, dtype=jnp.float32),
        }

    def vmapped_mala_step_fn(keys, states, logdensity_fn, step_size):
        # step_size is unshared: shape (num_particles,)
        def one(key, state, eps):
            return base_mala_kernel(
                key,
                state,
                logdensity_fn,
                step_size=eps,
            )
        return jax.vmap(one)(keys, states, step_size)

    def adapt_one_step(current_parameters, acceptance_value, t):
        new_parameters = dict(current_parameters)

        proposed_step_size = _update_step_size_robbins_monro(
            step_size=new_parameters["step_size"],
            acceptance_value=acceptance_value,
            target_acceptance_rate=target_acceptance_rate,
            t=t,
            rm_c=rm_c,
            rm_t0=rm_t0,
            rm_kappa=rm_kappa,
        )

        new_parameters["step_size"] = jnp.where(
            jnp.isfinite(acceptance_value),
            proposed_step_size,
            new_parameters["step_size"],
        )
        return new_parameters

    mala_update_strategy = make_rm_update_strategy(adapt_one_step)

    def build_ps_kernel_fn(
        *,
        log_prior_fn,
        log_likelihood_fn,
        max_iterations,
        alpha,
        num_mcmc_steps,
        kernel_params,):
        mala_parameters = {"step_size": jnp.full((num_particles,), jnp.asarray(kernel_params["step_size"], dtype=jnp.float32),),}
        
        return blackjax.adaptive_persistent_sampling_smc(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            max_iterations=max_iterations,
            mcmc_step_fn=vmapped_mala_step_fn,
            mcmc_init_fn=blackjax.mala.init,
            mcmc_parameters=mala_parameters,
            resampling_fn=resampling.systematic,
            target_ess=alpha,
            num_mcmc_steps=num_mcmc_steps,
            update_strategy=mala_update_strategy,
        )

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, tempering_param, t):
        # inner RM already updated step_size
        return dict(kernel_params)

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return int(num_particles) * int(num_mcmc_steps)

    return _run_ps_generic_once(
        target=target,
        num_particles=num_particles,
        seed=seed,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        build_ps_kernel_fn=build_ps_kernel_fn,
        init_kernel_params_fn=init_kernel_params_fn,
        adapt_kernel_params_fn=adapt_kernel_params_fn,
        gradient_eval_increment_fn=gradient_eval_increment_fn,
        kernel_name="mala",
    )

# =================
# PS + NUTS
def run_ps_nuts_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 0.1,
                    max_num_doublings: int = 10,
                    mass_matrix_ridge: float = 1e-6,
                    target_acceptance_rate: float = 0.8,
                    rm_c: float = 3.5,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    # BlackJAX's NUTS with diagonal inverse mass matrix.
    # We keep step size fixed in this first version and only adapt geometry.

    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),
                "inverse_mass_matrix": _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge,),
                "max_num_doublings": int(max_num_doublings),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        nuts_parameters = {"step_size": jnp.asarray(kernel_params["step_size"]),
                           "inverse_mass_matrix": jnp.asarray(kernel_params["inverse_mass_matrix"]),
                           "max_num_doublings": int(kernel_params["max_num_doublings"]),}

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=blackjax.nuts.build_kernel(),
                                                        mcmc_init_fn=blackjax.nuts.init,
                                                        mcmc_parameters=extend_params(nuts_parameters),
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, tempering_param, t):
        new_params = dict(kernel_params)
        if not np.isnan(acceptance_value):
            new_params["step_size"] = _update_step_size_robbins_monro(step_size=new_params["step_size"], acceptance_value=acceptance_value,
                                                                      target_acceptance_rate=target_acceptance_rate, t=t, rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)

        new_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(particles, ridge=mass_matrix_ridge,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        # fallback only; actual variable integration steps are handled in _run_ps_generic_once
        return 0

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="nuts",)
# =================
# PS + unadjusted MCLMC with capped-history adaptation built on BlackJAX
# Uses BlackJAX MCLMC kernel and BlackJAX internal adaptation, while keeping
# adaptation cost bounded in high dimensions / high particle counts.

def _flatten_persistent_particles_up_to_iteration(
    persistent_particles: jnp.ndarray,
    iteration: int,
) -> jnp.ndarray:
    return jnp.asarray(persistent_particles[: iteration + 1]).reshape(-1, persistent_particles.shape[-1])


def _capped_adaptation_particles(
    *,
    rng_key: jax.Array,
    current_particles: jnp.ndarray,
    persistent_particles: jnp.ndarray,
    iteration: int,
    adaptation_multiple: float,
    adaptation_max_particles: int,
    geometry_min_particles: int,
) -> jnp.ndarray:
    current_particles = jnp.asarray(current_particles)
    n_current = int(current_particles.shape[0])
    target_size = int(min(max(geometry_min_particles, n_current), max(1, adaptation_max_particles)))

    if n_current >= target_size or iteration <= 0:
        return current_particles

    persistent_flat = _flatten_persistent_particles_up_to_iteration(persistent_particles, iteration)
    total_available = int(persistent_flat.shape[0])
    capped_total = int(min(total_available, max(1, int(adaptation_multiple * n_current)), adaptation_max_particles))

    if capped_total <= n_current:
        return current_particles

    num_extra = capped_total - n_current
    idx = jax.random.choice(rng_key, total_available, shape=(num_extra,), replace=False)
    extra_particles = persistent_flat[idx]
    return jnp.concatenate([current_particles, extra_particles], axis=0)


def _choose_probe_particles(
    *,
    rng_key: jax.Array,
    current_particles: jnp.ndarray,
    persistent_particles: jnp.ndarray,
    iteration: int,
    num_probe_particles: int,
    probe_from_persistent: bool,
) -> jnp.ndarray:
    current_particles = jnp.asarray(current_particles)
    n_current = int(current_particles.shape[0])
    num_probe = max(1, min(int(num_probe_particles), n_current))

    if (not probe_from_persistent) or iteration <= 0:
        idx = jax.random.choice(rng_key, n_current, shape=(num_probe,), replace=False)
        return current_particles[idx]

    persistent_flat = _flatten_persistent_particles_up_to_iteration(persistent_particles, iteration)
    total_available = int(persistent_flat.shape[0])
    if total_available <= num_probe:
        return persistent_flat

    idx = jax.random.choice(rng_key, total_available, shape=(num_probe,), replace=False)
    return persistent_flat[idx]


def _median_positive(values: list[float], fallback: float) -> float:
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if len(finite) == 0:
        return float(fallback)
    return float(np.median(np.asarray(finite, dtype=float)))


def _smooth_positive_update(old_value: float, proposed_value: float, smoothing: float) -> float:
    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    if not np.isfinite(proposed_value) or proposed_value <= 0.0:
        return float(old_value)
    return float((1.0 - smoothing) * float(old_value) + smoothing * float(proposed_value))


def _clipped_step_size_update(
    old_step_size: float,
    proposed_step_size: float,
    smoothing: float,
    min_step_size: float,
    max_step_size: float,
    max_decrease_factor: float,
    max_increase_factor: float,
) -> float:
    updated = _smooth_positive_update(old_step_size, proposed_step_size, smoothing)
    lower = float(old_step_size) * float(max_decrease_factor)
    upper = float(old_step_size) * float(max_increase_factor)
    updated = float(np.clip(updated, lower, upper))
    updated = float(np.clip(updated, min_step_size, max_step_size))
    return updated


def _run_blackjax_mclmc_adaptation_probe(
    *,
    probe_particles: jnp.ndarray,
    logdensity_fn,
    inverse_mass_matrix: jnp.ndarray,
    step_size: float,
    L: float,
    num_steps: int,
    rng_key: jax.Array,
    desired_energy_var: float,
    trust_in_estimate: float,
    num_effective_samples: int,
    frac_tune1: float,
    frac_tune2: float,
    frac_tune3: float,
    l_factor: float,
):
    import blackjax.adaptation.mclmc_adaptation as mclmc_adaptation

    tuned_kernel = lambda inv_mass: blackjax.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inv_mass,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    probe_particles = jnp.asarray(probe_particles)
    num_probe = int(probe_particles.shape[0])
    probe_keys = jax.random.split(rng_key, max(1, num_probe))

    proposed_step_sizes = []
    proposed_Ls = []
    total_tuning_integrator_steps = 0

    for idx in range(num_probe):
        adaptation_state = blackjax.mclmc.init(
            probe_particles[idx],
            logdensity_fn,
            rng_key=probe_keys[idx],
        )

        adaptation_params = mclmc_adaptation.MCLMCAdaptationState(
            L=float(L),
            step_size=float(step_size),
            inverse_mass_matrix=jnp.asarray(inverse_mass_matrix),
        )

        try:
            _, adapted_params, tuning_integrator_steps = mclmc_adaptation.mclmc_find_L_and_step_size(
                mclmc_kernel=tuned_kernel,
                num_steps=int(num_steps),
                state=adaptation_state,
                rng_key=probe_keys[idx],
                frac_tune1=frac_tune1,
                frac_tune2=frac_tune2,
                frac_tune3=frac_tune3,
                desired_energy_var=desired_energy_var,
                trust_in_estimate=trust_in_estimate,
                num_effective_samples=num_effective_samples,
                diagonal_preconditioning=False,
                params=adaptation_params,
                l_factor=l_factor,
            )
        except TypeError:
            _, adapted_params, tuning_integrator_steps = mclmc_adaptation.mclmc_find_L_and_step_size(
                mclmc_kernel=tuned_kernel,
                num_steps=int(num_steps),
                state=adaptation_state,
                rng_key=probe_keys[idx],
                frac_tune1=frac_tune1,
                frac_tune2=frac_tune2,
                frac_tune3=frac_tune3,
                desired_energy_var=desired_energy_var,
                trust_in_estimate=trust_in_estimate,
                num_effective_samples=num_effective_samples,
                diagonal_preconditioning=False,
                params=adaptation_params,
                Lfactor=l_factor,
            )

        proposed_step_sizes.append(float(adapted_params.step_size))
        proposed_Ls.append(float(adapted_params.L))
        total_tuning_integrator_steps += int(tuning_integrator_steps)

    return {
        "step_size": _median_positive(proposed_step_sizes, step_size),
        "L": _median_positive(proposed_Ls, L),
        "tuning_integrator_steps": int(total_tuning_integrator_steps),
        "num_probe_particles": int(num_probe),
    }


# =================
# PS + unadjusted MCLMC with capped-history BlackJAX adaptation
# Fast design:
# - diagonal inverse mass matrix only
# - geometry updated once per PS iteration
# - step size corrected inside move every num_mcmc_steps // 5 steps
# - L adapted once per PS iteration from a small probe set
# - capped history used only when current N is too small for stable geometry

def run_ps_mclmc_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 1e-2,
    L: float = 1.0,
    mass_matrix_ridge: float = 1e-6,
    desired_energy_var: float = 5e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: int = 150,
    frac_tune1: float = 0.1,
    frac_tune2: float = 0.1,
    frac_tune3: float = 0.1,
    l_factor: float = 0.4,
    adaptation_multiple: float = 3.0,
    adaptation_max_particles: int = 512,
    geometry_min_particles: int = 64,
    num_probe_particles: int = 4,
    probe_num_steps: int | None = None,
    epsilon_smoothing: float = 0.3,
    L_smoothing: float = 0.1,
    min_step_size: float = 1e-6,
    max_step_size: float = 1.0,
    max_step_size_decrease_factor: float = 0.7,
    max_step_size_increase_factor: float = 1.15,
    probe_from_persistent: bool = False,
) -> dict[str, Any]:
    if target is None:
        target = make_gaussian_mixture_target(dimension)

    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn

    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    probe_num_steps = int(num_mcmc_steps if probe_num_steps is None else probe_num_steps)
    adapt_every_eps = max(1, num_mcmc_steps // 5)
    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    kernel_params = {
        "step_size": float(step_size),
        "L": float(L),
        "inverse_mass_matrix": _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge),
        "step_size_reference": float(step_size),
    }

    tempering_path, logZ_path, ess_path, acceptance_path, step_size_path = [], [], [], [], []
    elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path, L_path = [], [], [], [], []
    adaptation_set_size_path, probe_count_path = [], []

    start = time.perf_counter()
    n_iter, gradient_eval_count, resampling_steps = 0, 0, 0

    def calculate_lambda(ps_state):
        n_particles_local = ps_state.persistent_weights.shape[1]
        target_val = jnp.log(n_particles_local * alpha)
        max_delta = 1 - ps_state.tempering_schedule[ps_state.iteration]

        def fun_to_solve(delta):
            log_weights, _ = persistent_sampling.compute_log_persistent_weights(
                ps_state.persistent_log_likelihoods,
                ps_state.persistent_log_Z,
                ps_state.tempering_schedule.at[ps_state.iteration + 1].set(
                    ps_state.tempering_schedule[ps_state.iteration] + delta
                ),
                ps_state.iteration + 1,
                normalize_to_one=True,
            )
            ess_val = jnp.log(persistent_sampling.compute_persistent_ess(log_weights))
            return ess_val - target_val

        delta = jnp.nan_to_num(smc_solver.dichotomy(fun_to_solve, 0.0, max_delta))
        return ps_state.tempering_schedule[ps_state.iteration] + jnp.clip(delta, 0.0, max_delta)

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, resample_key, move_key, geom_key, probe_key, adapt_key = jax.random.split(loop_key, 6)

        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        next_lambda = calculate_lambda(state)
        iteration = state.iteration + 1
        tempering_schedule = state.tempering_schedule.at[iteration].set(next_lambda)

        log_persistent_weights, log_Z_t = persistent_sampling.compute_log_persistent_weights(
            state.persistent_log_likelihoods,
            state.persistent_log_Z,
            tempering_schedule,
            iteration,
            normalize_to_one=True,
        )
        persistent_weights = jnp.exp(log_persistent_weights)

        resampled_particles, _ = persistent_sampling.resample_from_persistent(
            resample_key,
            state.persistent_particles,
            persistent_weights,
            resampling.systematic,
        )

        geometry_particles = _capped_adaptation_particles(
            rng_key=geom_key,
            current_particles=resampled_particles,
            persistent_particles=state.persistent_particles,
            iteration=int(state.iteration),
            adaptation_multiple=adaptation_multiple,
            adaptation_max_particles=adaptation_max_particles,
            geometry_min_particles=geometry_min_particles,
        )
        kernel_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(
            geometry_particles,
            ridge=mass_matrix_ridge,
        )

        adaptation_set_size_path.append(int(geometry_particles.shape[0]))

        def logposterior_fn(x):
            return log_prior_fn(x) + next_lambda * log_likelihood_fn(x)

        mclmc_kernel = blackjax.mclmc.build_kernel(
            logdensity_fn=logposterior_fn,
            inverse_mass_matrix=kernel_params["inverse_mass_matrix"],
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

        init_keys = jax.random.split(move_key, num_particles + 1)
        mcmc_states = jax.vmap(
            lambda position, k: blackjax.mclmc.init(position, logposterior_fn, rng_key=k)
        )(resampled_particles, init_keys[:-1])
        inner_key = init_keys[-1]

        last_acceptance_value = np.nan
        last_update_info = None

        for mcmc_iter in range(num_mcmc_steps):
            inner_key, one_key = jax.random.split(inner_key)
            keys = jax.random.split(one_key, num_particles)

            def one_step(key, state_):
                return mclmc_kernel(
                    key,
                    state_,
                    float(kernel_params["L"]),
                    float(kernel_params["step_size"]),
                )

            mcmc_states, update_info = jax.vmap(one_step)(keys, mcmc_states)
            last_update_info = update_info
            last_acceptance_value = float(_safe_mean_acceptance(update_info))

            if ((mcmc_iter + 1) % adapt_every_eps) == 0:
                kernel_params["step_size"] = _clipped_step_size_update(
                    old_step_size=float(kernel_params["step_size"]),
                    proposed_step_size=float(kernel_params["step_size_reference"]),
                    smoothing=epsilon_smoothing,
                    min_step_size=min_step_size,
                    max_step_size=max_step_size,
                    max_decrease_factor=max_step_size_decrease_factor,
                    max_increase_factor=max_step_size_increase_factor,
                )

        iteration_particles = mcmc_states.position if hasattr(mcmc_states, "position") else mcmc_states
        iteration_log_likelihoods = vmapped_log_likelihood_fn(iteration_particles)

        probe_positions = _choose_probe_particles(
            rng_key=probe_key,
            current_particles=iteration_particles,
            persistent_particles=state.persistent_particles,
            iteration=int(state.iteration),
            num_probe_particles=num_probe_particles,
            probe_from_persistent=probe_from_persistent,
        )
        probe_count_path.append(int(probe_positions.shape[0]))

        probe_result = _run_blackjax_mclmc_adaptation_probe(
            probe_particles=probe_positions,
            logdensity_fn=logposterior_fn,
            inverse_mass_matrix=kernel_params["inverse_mass_matrix"],
            step_size=float(kernel_params["step_size"]),
            L=float(kernel_params["L"]),
            num_steps=probe_num_steps,
            rng_key=adapt_key,
            desired_energy_var=desired_energy_var,
            trust_in_estimate=trust_in_estimate,
            num_effective_samples=num_effective_samples,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            frac_tune3=frac_tune3,
            l_factor=l_factor,
        )
        kernel_params["step_size_reference"] = float(probe_result["step_size"])
        kernel_params["step_size"] = _clipped_step_size_update(
            old_step_size=float(kernel_params["step_size"]),
            proposed_step_size=float(probe_result["step_size"]),
            smoothing=epsilon_smoothing,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
            max_decrease_factor=max_step_size_decrease_factor,
            max_increase_factor=max_step_size_increase_factor,
        )
        kernel_params["L"] = _smooth_positive_update(
            float(kernel_params["L"]),
            float(probe_result["L"]),
            L_smoothing,
        )
        gradient_eval_count += int(probe_result["tuning_integrator_steps"])

        persistent_particles = jax.tree.map(
            lambda persistent, iteration_p: persistent.at[iteration].set(iteration_p),
            state.persistent_particles,
            iteration_particles,
        )
        persistent_log_Z = state.persistent_log_Z.at[iteration].set(log_Z_t)
        persistent_log_likelihoods = state.persistent_log_likelihoods.at[iteration].set(iteration_log_likelihoods)
        state = persistent_sampling.PersistentSMCState(
            persistent_particles=persistent_particles,
            persistent_log_likelihoods=persistent_log_likelihoods,
            persistent_log_Z=persistent_log_Z,
            tempering_schedule=tempering_schedule,
            iteration=iteration,
        )

        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.persistent_weights)

        ess_value = float(1.0 / jnp.sum(_normalize_weights(weights) ** 2))
        elapsed = time.perf_counter() - start
        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(last_acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)
        step_size_path.append(float(kernel_params["step_size"]))
        L_path.append(float(kernel_params["L"]))

        resampling_steps += 1
        gradient_eval_count += int(num_particles) * int(num_mcmc_steps)
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles(final_particles)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_array = np.asarray(acceptance_path, dtype=float)
    finite_acceptance = acceptance_array[np.isfinite(acceptance_array)]
    acceptance_rate_last = float(finite_acceptance[-1]) if finite_acceptance.size > 0 else np.nan
    acceptance_rate_mean = float(finite_acceptance.mean()) if finite_acceptance.size > 0 else np.nan

    return {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": "mclmc",
        "seed": int(seed),
        "dimension": int(target.dimension),
        "num_particles": int(num_particles),
        "logZ": float(final_state.log_Z),
        "posterior_mean": posterior_mean,
        "posterior_second_moment": posterior_second_moment,
        "final_ess": final_ess,
        "acceptance_rate_mean": acceptance_rate_mean,
        "acceptance_rate_last": acceptance_rate_last,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": np.asarray(acceptance_path, dtype=float),
        "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
        "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
        "esjd_path": np.asarray(esjd_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
        "L_path": np.asarray(L_path, dtype=float),
        "adaptation_set_size_path": np.asarray(adaptation_set_size_path, dtype=int),
        "probe_count_path": np.asarray(probe_count_path, dtype=int),
    }