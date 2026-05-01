from __future__ import annotations

from typing import Any, NamedTuple
import os
import time

# GPU-oriented PS + ULA implementation.
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
import blackjax.mcmc.diffusions as diffusions

from targets import Target, make_gaussian_mixture_target, make_target
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


class ULAState(NamedTuple):
    position: Any
    logdensity: Any
    logdensity_grad: Any


def _ula_step_size_from_particles(
    particles: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    adaptation_scale: float = 1.0,
    min_step_size: float = 1e-6,
    max_step_size: float = 1.0,
    variance_ridge: float = 1e-8,
    tempering_floor: float = 1e-3,
) -> jnp.ndarray:
    """Deterministic ULA step-size rule.

    epsilon = adaptation_scale * sqrt(max(1 - beta, tempering_floor))
              / sqrt(mean_j Var(theta_j) + variance_ridge),
    clipped to [min_step_size, max_step_size].

    This is the same rule used in the original PS+ULA implementation, but it is
    kept on-device and evaluated once per PS temperature.
    """
    x = jnp.asarray(particles)
    dtype = x.dtype

    var = jnp.var(x, axis=0)
    var = jnp.where(jnp.isfinite(var), var, jnp.asarray(0.0, dtype=dtype))
    mean_var = jnp.mean(var) + jnp.asarray(variance_ridge, dtype=dtype)

    beta = jnp.asarray(beta, dtype=dtype)
    beta_factor = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(1.0, dtype=dtype) - beta,
            jnp.asarray(tempering_floor, dtype=dtype),
        )
    )

    eps = jnp.asarray(adaptation_scale, dtype=dtype) * beta_factor / jnp.sqrt(mean_var)
    return jnp.clip(
        eps,
        jnp.asarray(min_step_size, dtype=dtype),
        jnp.asarray(max_step_size, dtype=dtype),
    )


def _make_ula_move_k_steps(
    log_prior_fn,
    log_likelihood_fn,
    *,
    num_particles: int,
    num_mcmc_steps: int,
):
    """Build one compiled k-step ULA mutation for one PS temperature.

    Algorithmically this matches PS+ULA:
      1. set the tempered log density using the current beta;
      2. initialize ULA state for each resampled particle;
      3. run k unadjusted Langevin steps for all particles.

    The inner k-step loop is represented by lax.scan so the mutation is a single
    compiled XLA program on GPU.
    """

    def tempered_logdensity(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity

    def init_one(position, beta):
        grad_fn = jax.value_and_grad(tempered_logdensity(beta))
        logdensity, logdensity_grad = grad_fn(position)
        return ULAState(position, logdensity, logdensity_grad)

    def step_one(rng_key, state, beta, step_size):
        grad_fn = jax.value_and_grad(tempered_logdensity(beta))
        one_step = diffusions.overdamped_langevin(grad_fn)
        new_state = one_step(rng_key, state, step_size)
        return ULAState(*new_state)

    vmapped_init = jax.vmap(init_one, in_axes=(0, None))
    vmapped_step = jax.vmap(step_one, in_axes=(0, 0, None, None))

    @jax.jit
    def ula_move_k_steps(key, particles, beta, step_size):
        states = vmapped_init(particles, beta)

        def scan_body(carry, _):
            key, states = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_particles)
            states = vmapped_step(step_keys, states, beta, step_size)
            return (key, states), None

        (key, states), _ = jax.lax.scan(
            scan_body,
            (key, states),
            xs=None,
            length=num_mcmc_steps,
        )

        return key, states.position

    return ula_move_k_steps


def run_ps_ula_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 500,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    ula_adaptation_scale: float = 1.0,
    ula_min_step_size: float = 1e-6,
    ula_max_step_size: float = 1.0,
    ula_variance_ridge: float = 1e-8,
    ula_tempering_floor: float = 1e-3,
    return_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run persistent sampling with a GPU-friendly ULA mutation kernel.

    ULA has no Metropolis correction, so there is no acceptance probability. The
    step size is set deterministically at every PS temperature as

        epsilon ∝ sqrt(1 - beta) / sqrt(Var(theta)),

    using the resampled particle cloud and the next tempering level.
    """
    if target is None:
        target = make_target(target_name, dimension, **(target_kwargs or {}))
    elif target_kwargs is not None:
        raise ValueError("Pass either target or target_kwargs, not both.")

    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    initial_particles.block_until_ready()

    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    ula_move_k_steps = _make_ula_move_k_steps(
        target.log_prior_fn,
        target.log_likelihood_fn,
        num_particles=int(num_particles),
        num_mcmc_steps=int(num_mcmc_steps),
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

        step_size = _ula_step_size_from_particles(
            resampled_particles,
            next_lambda,
            adaptation_scale=ula_adaptation_scale,
            min_step_size=ula_min_step_size,
            max_step_size=ula_max_step_size,
            variance_ridge=ula_variance_ridge,
            tempering_floor=ula_tempering_floor,
        )

        _, iteration_particles = ula_move_k_steps(
            move_key,
            resampled_particles,
            next_lambda,
            step_size,
        )

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

        ess_value = float(1.0 / jnp.sum(normalize_weights(weights) ** 2))
        elapsed = time.perf_counter() - start
        if return_diagnostics:
            post_step_particles = np.asarray(particles)
            esjd_value = population_esjd(pre_step_particles, post_step_particles)
        else:
            esjd_value = np.nan

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(np.nan)
        elapsed_time_path.append(float(elapsed))
        step_size_path.append(float(step_size))

        if return_diagnostics:
            variance_log_weights_path.append(variance_log_weights(weights))
            weight_entropy_path.append(weight_entropy(weights))
            esjd_path.append(esjd_value)

        resampling_steps += 1
        gradient_eval_count += int(num_particles) * int(num_mcmc_steps)
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = compute_posterior_moments_from_particles(final_particles)
    coordinate_summaries = coordinate_specific_summaries(posterior_mean, posterior_second_moment)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan

    out = {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": "ula",
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
        "acceptance_rate_mean": np.nan,
        "acceptance_rate_last": np.nan,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "final_step_size": float(step_size_path[-1]) if step_size_path else np.nan,
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": np.asarray(acceptance_path, dtype=float),
        "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
    }

    if return_diagnostics:
        out.update(
            {
                "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
                "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
                "esjd_path": np.asarray(esjd_path, dtype=float),
            }
        )

    return out


if __name__ == "__main__":
    print(debug_worker_backend())
    result = run_ps_ula_once(
        dimension=5,
        num_particles=1024,
        seed=0,
        max_iterations=200,
        alpha=0.999,
        num_mcmc_steps=10,
        return_diagnostics=False,
    )
    print({k: result[k] for k in ["logZ", "n_iter", "final_ess", "final_step_size", "runtime_sec"]})
