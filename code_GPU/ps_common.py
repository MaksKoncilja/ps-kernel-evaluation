from __future__ import annotations

from typing import Any
import time

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.persistent_sampling as persistent_sampling
import blackjax.smc.solver as smc_solver


def debug_worker_backend() -> dict[str, Any]:
    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
        "jax_platform_name_env": __import__("os").environ.get("JAX_PLATFORM_NAME"),
        "jax_platforms_env": __import__("os").environ.get("JAX_PLATFORMS"),
        "jax_enable_x64_env": __import__("os").environ.get("JAX_ENABLE_X64"),
    }


def scalarize_positive_parameter(x: Any) -> jnp.ndarray:
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    return jnp.asarray(arr.reshape(-1)[0], dtype=arr.dtype)


def safe_mean_acceptance(update_info) -> jnp.ndarray:
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted, dtype=jnp.float32))
    return jnp.nan


def empirical_covariance(
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
    return cov + ridge * jnp.eye(d, dtype=cov.dtype)


def proposal_sqrt_from_cov(cov: jnp.ndarray, scale: float | jnp.ndarray) -> jnp.ndarray:
    chol = jnp.linalg.cholesky(cov)
    return jnp.asarray(scale, dtype=chol.dtype) * chol


def robbins_monro_step_size(
    t: int,
    c: float = 1.0,
    t0: float = 10.0,
    kappa: float = 0.6,
) -> float:
    return float(c / ((t + t0) ** kappa))


def update_rw_scale_robbins_monro(
    rw_scale: jnp.ndarray,
    acceptance_value: float,
    target_acceptance_rate: float,
    t: int,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> jnp.ndarray:
    gamma_t = robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


def compute_posterior_moments_from_particles(
    particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def coordinate_specific_summaries(
    posterior_mean: np.ndarray,
    posterior_second_moment: np.ndarray,
) -> dict[str, float]:
    posterior_mean_first_coord = float(posterior_mean[0])
    posterior_second_moment_first_coord = float(posterior_second_moment[0])

    if posterior_mean.shape[0] > 1:
        posterior_mean_rest_coords_mean = float(np.mean(posterior_mean[1:]))
        posterior_second_moment_rest_coords_mean = float(np.mean(posterior_second_moment[1:]))
    else:
        posterior_mean_rest_coords_mean = np.nan
        posterior_second_moment_rest_coords_mean = np.nan

    return {
        "posterior_mean_first_coord": posterior_mean_first_coord,
        "posterior_second_moment_first_coord": posterior_second_moment_first_coord,
        "posterior_mean_rest_coords_mean": posterior_mean_rest_coords_mean,
        "posterior_second_moment_rest_coords_mean": posterior_second_moment_rest_coords_mean,
    }


def normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    w = jnp.asarray(weights, dtype=jnp.float32)
    return w / jnp.sum(w)


def variance_log_weights(weights: jnp.ndarray) -> float:
    w = normalize_weights(weights)
    eps = 1e-32
    logw = jnp.log(w + eps)
    return float(jnp.var(logw))


def weight_entropy(weights: jnp.ndarray) -> float:
    w = normalize_weights(weights)
    eps = 1e-32
    return float(-jnp.sum(w * jnp.log(w + eps)))


def population_esjd(before_particles: jnp.ndarray, after_particles: jnp.ndarray) -> float:
    dx = jnp.asarray(after_particles) - jnp.asarray(before_particles)
    sqdist = jnp.sum(dx ** 2, axis=1)
    return float(jnp.mean(sqdist))


def calculate_next_lambda(ps_state, alpha: float) -> jnp.ndarray:
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


def run_ps_with_inner_adaptation_once(
    *,
    target: Any,
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

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, resample_key, step_key = jax.random.split(loop_key, 3)

        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        next_lambda = calculate_next_lambda(state, alpha)
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
            last_acceptance_value = float(safe_mean_acceptance(update_info))
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

        ess_value = float(1.0 / jnp.sum(normalize_weights(weights) ** 2))
        elapsed = time.perf_counter() - start

        post_step_particles = np.asarray(particles)
        esjd_value = population_esjd(pre_step_particles, post_step_particles)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(last_acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(variance_log_weights(weights))
        weight_entropy_path.append(weight_entropy(weights))
        esjd_path.append(esjd_value)

        if "step_size" in kernel_params:
            step_size_path.append(float(scalarize_positive_parameter(kernel_params["step_size"])))
        elif "rw_scale" in kernel_params:
            step_size_path.append(float(scalarize_positive_parameter(kernel_params["rw_scale"])))
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
    posterior_mean, posterior_second_moment = compute_posterior_moments_from_particles(final_particles)
    coordinate_summaries = coordinate_specific_summaries(posterior_mean, posterior_second_moment)

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
        "posterior_mean_first_coord": coordinate_summaries["posterior_mean_first_coord"],
        "posterior_second_moment_first_coord": coordinate_summaries["posterior_second_moment_first_coord"],
        "posterior_mean_rest_coords_mean": coordinate_summaries["posterior_mean_rest_coords_mean"],
        "posterior_second_moment_rest_coords_mean": coordinate_summaries["posterior_second_moment_rest_coords_mean"],
        "particles": final_particles,
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
