from __future__ import annotations

from typing import Any
import os
import time

# GPU-oriented PS + HMC implementation.
# Do not force CPU. Let JAX use GPU when env_PS_GPU exposes a CUDA backend.
# Optional override: set FORCE_JAX_CPU=1 before importing this module.
if os.environ.get("FORCE_JAX_CPU", "0") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
else:
    # Avoid inheriting an accidental CPU-only setting from the shell/job.
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ.pop("JAX_PLATFORM_NAME", None)

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.persistent_sampling as persistent_sampling

from targets import Target, make_target
from ps_common import (
    debug_worker_backend,
    calculate_next_lambda,
    normalize_weights,
    variance_log_weights,
    weight_entropy,
    population_esjd,
    compute_posterior_moments_from_particles,
    coordinate_specific_summaries,
)


def _diagonal_inverse_mass_matrix(particles: jnp.ndarray, ridge: float = 1e-6) -> jnp.ndarray:
    """Diagonal inverse mass matrix estimated from the current particle cloud."""
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0, ddof=1)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    return 1.0 / (var + ridge)


def _find_reasonable_initial_step_size(
    *,
    rng_key: jax.Array,
    kernel_generator,
    reference_state,
    initial_step_size: float | jnp.ndarray,
    target_acceptance_rate: float,
) -> jnp.ndarray:
    """Use BlackJAX's reasonable-step-size heuristic when available.

    This mirrors the original HMC code: the heuristic is used only to choose the
    first epsilon; the PS iterations then adapt epsilon by the bisection scheme.
    """
    try:
        step_size = blackjax.adaptation.step_size.find_reasonable_step_size(
            rng_key=rng_key,
            kernel_generator=kernel_generator,
            reference_state=reference_state,
            initial_step_size=jnp.asarray(initial_step_size, dtype=jnp.float32),
            target_accept=target_acceptance_rate,
        )
        return jnp.asarray(step_size)
    except Exception:
        return jnp.asarray(initial_step_size)


def _safe_acceptance_rate(update_info) -> jnp.ndarray:
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted))
    return jnp.asarray(jnp.nan)


def _bisection_step_size_update(
    step_size: jnp.ndarray,
    acceptance_value: jnp.ndarray,
    lower_log_step: jnp.ndarray,
    upper_log_step: jnp.ndarray,
    is_converged: jnp.ndarray,
    *,
    target_acceptance_rate: jnp.ndarray,
    tolerance: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One monotone bisection/bracketing update for HMC epsilon.

    This is the same logic as the original BlackJAX-style bisection adaptation:
    if acceptance is too high, epsilon is increased; if acceptance is too low,
    epsilon is decreased; once both a lower and upper bracket are available, the
    new value is their midpoint on the log scale. The state is reset at every PS
    iteration.
    """
    dtype = step_size.dtype
    log_step = jnp.log(step_size)
    acceptance_value = jnp.asarray(acceptance_value, dtype=dtype)
    target = jnp.asarray(target_acceptance_rate, dtype=dtype)
    tol = jnp.asarray(tolerance, dtype=dtype)

    finite_acceptance = jnp.isfinite(acceptance_value)
    close_enough = jnp.abs(acceptance_value - target) <= tol
    should_stop = is_converged | (~finite_acceptance) | close_enough

    accept_high = acceptance_value > target

    proposed_lower = jnp.where(accept_high, log_step, lower_log_step)
    proposed_upper = jnp.where(accept_high, upper_log_step, log_step)

    has_lower = jnp.isfinite(proposed_lower)
    has_upper = jnp.isfinite(proposed_upper)

    # If only one side of the bracket is known, expand by a factor of 2.
    expanded_up = log_step + jnp.log(jnp.asarray(2.0, dtype=dtype))
    expanded_down = log_step - jnp.log(jnp.asarray(2.0, dtype=dtype))
    midpoint = 0.5 * (proposed_lower + proposed_upper)

    next_log_step_if_high = jnp.where(has_upper, midpoint, expanded_up)
    next_log_step_if_low = jnp.where(has_lower, midpoint, expanded_down)
    next_log_step = jnp.where(accept_high, next_log_step_if_high, next_log_step_if_low)

    next_step_size = jnp.exp(next_log_step)
    next_step_size = jnp.where(should_stop, step_size, next_step_size)
    next_lower = jnp.where(should_stop, lower_log_step, proposed_lower)
    next_upper = jnp.where(should_stop, upper_log_step, proposed_upper)
    next_converged = is_converged | close_enough

    return next_step_size, next_lower, next_upper, next_converged


def _make_hmc_move_k_steps(
    log_prior_fn,
    log_likelihood_fn,
    *,
    num_particles: int,
    num_mcmc_steps: int,
    num_integration_steps: int,
    target_acceptance_rate: float,
    bisection_tolerance: float,
):
    """Build one compiled k-step HMC mutation for one PS temperature.

    Algorithmically this matches the original PS+HMC structure:
      1. initialize HMC state for each resampled particle;
      2. run k HMC mutation steps with fixed trajectory length;
      3. every max(1, k // 5) steps, compute mean acceptance over particles;
      4. update the shared step size epsilon by a bisection/bracketing scheme;
      5. reset the bisection state at each new PS temperature.

    The difference is that steps 2--4 are staged into XLA using lax.scan.
    """
    base_hmc_kernel = blackjax.hmc.build_kernel()
    adapt_every = max(1, int(num_mcmc_steps) // 5)

    target_acceptance_rate_jax = jnp.asarray(target_acceptance_rate)
    bisection_tolerance_jax = jnp.asarray(bisection_tolerance)

    def tempered_logdensity(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity

    def init_one(position, beta):
        return blackjax.hmc.init(position, tempered_logdensity(beta))

    def step_one(rng_key, state, beta, step_size, inverse_mass_matrix):
        return base_hmc_kernel(
            rng_key,
            state,
            tempered_logdensity(beta),
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=int(num_integration_steps),
        )

    vmapped_init = jax.vmap(init_one, in_axes=(0, None))
    vmapped_step = jax.vmap(step_one, in_axes=(0, 0, None, None, None))

    @jax.jit
    def hmc_move_k_steps(key, particles, beta, step_size, inverse_mass_matrix):
        states = vmapped_init(particles, beta)
        dtype = step_size.dtype
        lower_log_step = jnp.asarray(-jnp.inf, dtype=dtype)
        upper_log_step = jnp.asarray(jnp.inf, dtype=dtype)
        is_converged = jnp.asarray(False)

        def scan_body(carry, mcmc_iter):
            key, states, step_size, lower_log_step, upper_log_step, is_converged = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_particles)

            states, update_info = vmapped_step(
                step_keys,
                states,
                beta,
                step_size,
                inverse_mass_matrix,
            )
            acceptance_value = _safe_acceptance_rate(update_info)

            should_adapt = ((mcmc_iter + 1) % adapt_every) == 0

            def do_update(args):
                eps, lo, hi, done = args
                return _bisection_step_size_update(
                    eps,
                    acceptance_value,
                    lo,
                    hi,
                    done,
                    target_acceptance_rate=target_acceptance_rate_jax,
                    tolerance=bisection_tolerance_jax,
                )

            def no_update(args):
                return args

            step_size, lower_log_step, upper_log_step, is_converged = jax.lax.cond(
                should_adapt,
                do_update,
                no_update,
                (step_size, lower_log_step, upper_log_step, is_converged),
            )

            return (
                key,
                states,
                step_size,
                lower_log_step,
                upper_log_step,
                is_converged,
            ), (acceptance_value, step_size)

        (key, states, step_size, _, _, _), (acceptance_path, step_size_path) = jax.lax.scan(
            scan_body,
            (key, states, step_size, lower_log_step, upper_log_step, is_converged),
            jnp.arange(num_mcmc_steps),
        )

        return key, states.position, step_size, acceptance_path, step_size_path

    return hmc_move_k_steps


def run_ps_hmc_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 500,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 0.1,
    num_integration_steps: int = 10,
    target_acceptance_rate: float = 0.651,
    bisection_tolerance: float = 0.03,
    mass_matrix_ridge: float = 1e-6,
    return_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run Persistent Sampling + adaptive HMC on GPU.

    HMC uses a diagonal inverse mass matrix estimated from the particle cloud.
    Within each PS iteration, the k-step HMC mutation adapts a shared step size
    every max(1, k // 5) MCMC steps using a bisection/bracketing scheme targeting
    the desired acceptance probability. The bisection state is reset at each PS
    temperature, matching the structure of the original samplers.py method.
    """
    if target is None:
        target = make_target(target_name, dimension, **(target_kwargs or {}))
    elif target_kwargs is not None:
        raise ValueError("Pass either target or target_kwargs, not both.")

    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn

    key = jax.random.PRNGKey(seed)
    key, init_key, warmup_key, loop_key = jax.random.split(key, 4)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    initial_particles.block_until_ready()

    inverse_mass_matrix = _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge)

    # Same initial-step heuristic as the original code: use one reference particle
    # and the prior log-density to get a reasonable initial epsilon.
    base_hmc_kernel = blackjax.hmc.build_kernel()
    reference_position = jnp.asarray(initial_particles[0])
    reference_state = blackjax.hmc.init(reference_position, target.log_prior_fn)

    def kernel_generator(eps):
        def kernel(k, state):
            return base_hmc_kernel(
                k,
                state,
                target.log_prior_fn,
                step_size=eps,
                inverse_mass_matrix=inverse_mass_matrix,
                num_integration_steps=int(num_integration_steps),
            )
        return kernel

    step_size_jax = _find_reasonable_initial_step_size(
        rng_key=warmup_key,
        kernel_generator=kernel_generator,
        reference_state=reference_state,
        initial_step_size=step_size,
        target_acceptance_rate=target_acceptance_rate,
    )
    step_size_jax = jnp.asarray(step_size_jax, dtype=initial_particles.dtype)

    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    hmc_move_k_steps = _make_hmc_move_k_steps(
        log_prior_fn,
        log_likelihood_fn,
        num_particles=int(num_particles),
        num_mcmc_steps=int(num_mcmc_steps),
        num_integration_steps=int(num_integration_steps),
        target_acceptance_rate=float(target_acceptance_rate),
        bisection_tolerance=float(bisection_tolerance),
    )

    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    tempering_path: list[float] = []
    logZ_path: list[float] = []
    ess_path: list[float] = []
    acceptance_path: list[float] = []
    step_size_path: list[float] = []
    elapsed_time_path: list[float] = []
    variance_log_weights_path: list[float] = []
    weight_entropy_path: list[float] = []
    esjd_path: list[float] = []

    start = time.perf_counter()
    n_iter = 0
    gradient_eval_count = 0
    resampling_steps = 0

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, resample_key, move_key = jax.random.split(loop_key, 3)

        if return_diagnostics:
            pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)
        else:
            pre_step_particles = None

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

        move_key_after, iteration_particles, step_size_jax, acc_path_jax, step_path_jax = hmc_move_k_steps(
            move_key,
            resampled_particles,
            next_lambda,
            step_size_jax,
            inverse_mass_matrix,
        )
        loop_key = jax.random.fold_in(loop_key, jnp.asarray(move_key_after[0], dtype=jnp.uint32))
        iteration_particles.block_until_ready()

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

        ess_value_jax = 1.0 / jnp.sum(normalize_weights(weights) ** 2)
        ess_value = float(ess_value_jax)
        elapsed = time.perf_counter() - start

        if return_diagnostics:
            post_step_particles = np.asarray(particles)
            esjd_value = population_esjd(pre_step_particles, post_step_particles)
            acceptance_path.extend(np.asarray(acc_path_jax, dtype=float).tolist())
            step_size_path.extend(np.asarray(step_path_jax, dtype=float).tolist())
            elapsed_time_path.append(float(elapsed))
            variance_log_weights_path.append(variance_log_weights(weights))
            weight_entropy_path.append(weight_entropy(weights))
            esjd_path.append(esjd_value)
        else:
            acceptance_path.append(float(jnp.mean(acc_path_jax)))
            step_size_path.append(float(step_size_jax))
            elapsed_time_path.append(float(elapsed))
            variance_log_weights_path.append(np.nan)
            weight_entropy_path.append(np.nan)
            esjd_path.append(np.nan)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)

        # Original structure: update diagonal mass matrix once per PS iteration,
        # using the current particle cloud, for use in the next PS mutation.
        inverse_mass_matrix = _diagonal_inverse_mass_matrix(particles, ridge=mass_matrix_ridge)

        resampling_steps += 1
        gradient_eval_count += int(num_particles) * int(num_mcmc_steps) * int(num_integration_steps)
        n_iter += 1

    state_unpadded = blackjax.persistent_sampling.remove_padding(state)
    state_unpadded.particles.block_until_ready()
    runtime_sec = time.perf_counter() - start

    final_particles = np.asarray(state_unpadded.particles)
    posterior_mean, posterior_second_moment = compute_posterior_moments_from_particles(final_particles)
    coordinate_summaries = coordinate_specific_summaries(posterior_mean, posterior_second_moment)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_array = np.asarray(acceptance_path, dtype=float)
    finite_acceptance = acceptance_array[np.isfinite(acceptance_array)]
    acceptance_rate_last = float(finite_acceptance[-1]) if finite_acceptance.size > 0 else np.nan
    acceptance_rate_mean = float(finite_acceptance.mean()) if finite_acceptance.size > 0 else np.nan

    out = {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": "hmc",
        "seed": int(seed),
        "dimension": int(target.dimension),
        "num_particles": int(num_particles),
        "logZ": float(state_unpadded.log_Z),
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
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "final_step_size": float(step_size_jax),
        "num_integration_steps": int(num_integration_steps),
        "adapt_every": int(max(1, int(num_mcmc_steps) // 5)),
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": acceptance_array,
        "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
        "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
        "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
        "esjd_path": np.asarray(esjd_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
    }
    return out
