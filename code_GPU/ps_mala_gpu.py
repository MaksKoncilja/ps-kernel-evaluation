from __future__ import annotations

from typing import Any
import os
import time

# GPU-oriented PS + MALA implementation.
# Do not force CPU. Let JAX use GPU when env_PS_GPU exposes a CUDA backend.
# Optional override: set FORCE_JAX_CPU=1 before importing this module.
if os.environ.get("FORCE_JAX_CPU", "0") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
else:
    # Avoid inheriting an accidental CPU-only setting from the shell.
    os.environ.pop("JAX_PLATFORMS", None)

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.persistent_sampling as persistent_sampling

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


def _robbins_monro_gain_device(
    t: jnp.ndarray,
    *,
    c: jnp.ndarray,
    t0: jnp.ndarray,
    kappa: jnp.ndarray,
    dtype,
) -> jnp.ndarray:
    t = jnp.asarray(t, dtype=dtype)
    return jnp.asarray(c, dtype=dtype) / ((t + jnp.asarray(t0, dtype=dtype)) ** jnp.asarray(kappa, dtype=dtype))


def _update_step_size_robbins_monro_device(
    step_size: jnp.ndarray,
    acceptance_value: jnp.ndarray,
    *,
    target_acceptance_rate: jnp.ndarray,
    t: jnp.ndarray,
    rm_c: jnp.ndarray,
    rm_t0: jnp.ndarray,
    rm_kappa: jnp.ndarray,
) -> jnp.ndarray:
    """On-device Robbins--Monro update for MALA step size.

    log(epsilon_{t+1}) = log(epsilon_t) + gamma_t (acceptance_t - alpha*)
    """
    dtype = step_size.dtype
    gamma_t = _robbins_monro_gain_device(
        t,
        c=rm_c,
        t0=rm_t0,
        kappa=rm_kappa,
        dtype=dtype,
    )
    proposed = jnp.exp(
        jnp.log(step_size)
        + gamma_t * (acceptance_value - jnp.asarray(target_acceptance_rate, dtype=dtype))
    )
    return jnp.where(jnp.isfinite(acceptance_value), proposed, step_size)


def _make_mala_move_k_steps(
    log_prior_fn,
    log_likelihood_fn,
    *,
    num_particles: int,
    num_mcmc_steps: int,
    target_acceptance_rate: float,
    rm_c: float,
    rm_t0: float,
    rm_kappa: float,
):
    """Build one compiled k-step MALA mutation for one PS temperature.

    Algorithmically this matches the original PS+MALA structure:
      1. initialize MALA state for each resampled particle;
      2. for m = 1,...,k, run one MALA step for all particles;
      3. average acceptance over particles;
      4. update epsilon by Robbins--Monro at every MCMC step.

    The difference is that steps 2--4 are staged into XLA using lax.scan,
    which avoids Python/device synchronization inside the inner MCMC loop.
    """
    base_mala_kernel = blackjax.mala.build_kernel()

    target_acceptance_rate_jax = jnp.asarray(target_acceptance_rate)
    rm_c_jax = jnp.asarray(rm_c)
    rm_t0_jax = jnp.asarray(rm_t0)
    rm_kappa_jax = jnp.asarray(rm_kappa)

    def tempered_logdensity(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity

    def init_one(position, beta):
        return blackjax.mala.init(position, tempered_logdensity(beta))

    def step_one(rng_key, mcmc_state, beta, step_size):
        return base_mala_kernel(
            rng_key,
            mcmc_state,
            tempered_logdensity(beta),
            step_size=step_size,
        )

    vmapped_init = jax.vmap(init_one, in_axes=(0, None))
    vmapped_step = jax.vmap(step_one, in_axes=(0, 0, None, None))

    @jax.jit
    def mala_move_k_steps(key, particles, beta, step_size):
        states = vmapped_init(particles, beta)

        def scan_body(carry, mcmc_iter):
            key, states, step_size = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_particles)

            states, update_info = vmapped_step(step_keys, states, beta, step_size)
            acceptance_value = jnp.mean(jnp.asarray(update_info.acceptance_rate))

            step_size = _update_step_size_robbins_monro_device(
                step_size,
                acceptance_value,
                target_acceptance_rate=target_acceptance_rate_jax,
                t=mcmc_iter + 1,
                rm_c=rm_c_jax,
                rm_t0=rm_t0_jax,
                rm_kappa=rm_kappa_jax,
            )
            return (key, states, step_size), (acceptance_value, step_size)

        (key, states, step_size), (acceptance_path, step_size_path) = jax.lax.scan(
            scan_body,
            (key, states, step_size),
            jnp.arange(num_mcmc_steps),
        )

        return key, states.position, step_size, acceptance_path, step_size_path

    return mala_move_k_steps


def run_ps_mala_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 2000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    step_size: float = 0.02,
    target_acceptance_rate: float = 0.574,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
    return_diagnostics: bool = True,
) -> dict[str, Any]:
    """Run Persistent Sampling + adaptive MALA on GPU.

    MALA step size epsilon is adapted at every inner MCMC step via
    Robbins--Monro targeting alpha* ~= 0.574. The k-step MALA mutation is
    compiled with jax.jit + jax.lax.scan.
    """
    if target is None:
        if target_name == "gaussian_mixture" and target_kwargs is None:
            target = make_gaussian_mixture_target(dimension)
        else:
            target = make_target(target_name, dimension, **(target_kwargs or {}))

    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn

    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    initial_particles.block_until_ready()

    work_dtype = initial_particles.dtype
    step_size_jax = jnp.asarray(step_size, dtype=work_dtype)
    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    mala_move_k_steps = _make_mala_move_k_steps(
        log_prior_fn,
        log_likelihood_fn,
        num_particles=int(num_particles),
        num_mcmc_steps=int(num_mcmc_steps),
        target_acceptance_rate=float(target_acceptance_rate),
        rm_c=float(rm_c),
        rm_t0=float(rm_t0),
        rm_kappa=float(rm_kappa),
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

        move_key_after, iteration_particles, step_size_jax, acc_path_jax, step_path_jax = mala_move_k_steps(
            move_key,
            resampled_particles,
            next_lambda,
            step_size_jax,
        )
        # Fold in the consumed move key so future randomness changes if the compiled
        # move changes its internal splitting scheme.
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

        normalized_weights = normalize_weights(weights)
        ess_value = float(1.0 / jnp.sum(normalized_weights ** 2))
        elapsed = time.perf_counter() - start
        last_acceptance_value = float(acc_path_jax[-1])
        mean_acceptance_value = float(jnp.mean(acc_path_jax))
        final_step_size_value = float(step_size_jax)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        # For the per-PS-iteration acceptance path, keep the mean over the k inner MALA steps.
        acceptance_path.append(mean_acceptance_value)
        elapsed_time_path.append(float(elapsed))
        step_size_path.append(final_step_size_value)

        if return_diagnostics:
            post_step_particles = np.asarray(particles)
            esjd_path.append(population_esjd(pre_step_particles, post_step_particles))
            variance_log_weights_path.append(variance_log_weights(weights))
            weight_entropy_path.append(weight_entropy(weights))
        else:
            esjd_path.append(np.nan)
            variance_log_weights_path.append(np.nan)
            weight_entropy_path.append(np.nan)

        resampling_steps += 1
        gradient_eval_count += int(num_particles) * int(num_mcmc_steps)
        n_iter += 1

    particles.block_until_ready()
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

    out = {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": "mala",
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
        "final_step_size": final_step_size_value if n_iter > 0 else float(step_size),
    }

    return out


# Optional alias for consistency with ps_rwm_cpu.py.
def run_once(**kwargs) -> dict[str, Any]:
    return run_ps_mala_once(**kwargs)


if __name__ == "__main__":
    print(debug_worker_backend())
    out = run_ps_mala_once(
        dimension=5,
        num_particles=1024,
        seed=0,
        max_iterations=10_000,
        alpha=0.999,
        num_mcmc_steps=10,
        step_size=0.02,
        return_diagnostics=False,
    )
    print("logZ:", out["logZ"])
    print("n_iter:", out["n_iter"])
    print("final_ess:", out["final_ess"])
    print("acceptance_rate_mean:", out["acceptance_rate_mean"])
    print("final_step_size:", out["final_step_size"])
    print("runtime_sec:", out["runtime_sec"])
