from __future__ import annotations

from typing import Any
import os
import time

# GPU-oriented PS + MCLMC implementation with fewer host/device synchronizations.
# Set FORCE_JAX_CPU=1 before importing this module if you explicitly want CPU.
if os.environ.get("FORCE_JAX_CPU", "0") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
else:
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
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0, ddof=1)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    return 1.0 / (var + ridge)


def _median_positive(values: list[float], fallback: float) -> float:
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    return float(np.median(np.asarray(finite, dtype=float))) if finite else float(fallback)


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
    updated = float(np.clip(updated, old_step_size * max_decrease_factor, old_step_size * max_increase_factor))
    return float(np.clip(updated, min_step_size, max_step_size))


def _clipped_step_size_update_device(
    old_step_size: jnp.ndarray,
    proposed_step_size: jnp.ndarray,
    *,
    smoothing: jnp.ndarray,
    min_step_size: jnp.ndarray,
    max_step_size: jnp.ndarray,
    max_decrease_factor: jnp.ndarray,
    max_increase_factor: jnp.ndarray,
) -> jnp.ndarray:
    dtype = old_step_size.dtype
    proposed_step_size = jnp.asarray(proposed_step_size, dtype=dtype)
    smoothing = jnp.asarray(smoothing, dtype=dtype)
    updated = (1.0 - smoothing) * old_step_size + smoothing * proposed_step_size
    updated = jnp.where(jnp.isfinite(proposed_step_size) & (proposed_step_size > 0.0), updated, old_step_size)
    updated = jnp.clip(updated, old_step_size * max_decrease_factor, old_step_size * max_increase_factor)
    return jnp.clip(updated, min_step_size, max_step_size)


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
) -> dict[str, Any]:
    """Small-probe BlackJAX MCLMC adaptation.

    This remains Python-side because BlackJAX's adaptation helper is not a clean
    pure-JAX primitive in this usage pattern. The caller can make it cheap by
    running it periodically and with a small probe set.
    """
    import blackjax.adaptation.mclmc_adaptation as mclmc_adaptation

    tuned_kernel = lambda inv_mass: blackjax.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inv_mass,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    probe_particles = jnp.asarray(probe_particles)
    num_probe = int(probe_particles.shape[0])
    probe_keys = jax.random.split(rng_key, max(1, num_probe))
    proposed_step_sizes, proposed_Ls = [], []
    total_tuning_integrator_steps = 0

    for idx in range(num_probe):
        adaptation_state = blackjax.mclmc.init(probe_particles[idx], logdensity_fn, rng_key=probe_keys[idx])
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


def _make_one_ps_mclmc_step(
    log_prior_fn,
    log_likelihood_fn,
    *,
    num_particles: int,
    num_mcmc_steps: int,
    alpha: float,
    mass_matrix_ridge: float,
    epsilon_smoothing: float,
    min_step_size: float,
    max_step_size: float,
    max_step_size_decrease_factor: float,
    max_step_size_increase_factor: float,
):
    """Build one compiled PS iteration.

    This function performs, in one JIT boundary:
      beta selection, persistent resampling, diagonal geometry update,
      k MCLMC mutation steps via lax.scan, likelihood update, persistent state
      update, and a small metrics calculation.

    The BlackJAX L/step-size adaptation probe is deliberately outside this
    function; it is the only remaining Python-side adaptation component.
    """
    adapt_every_eps = max(1, int(num_mcmc_steps) // 5)
    alpha_jax = jnp.asarray(alpha)
    ridge_jax = jnp.asarray(mass_matrix_ridge)
    smoothing_jax = jnp.asarray(epsilon_smoothing)
    min_step_size_jax = jnp.asarray(min_step_size)
    max_step_size_jax = jnp.asarray(max_step_size)
    max_decrease_jax = jnp.asarray(max_step_size_decrease_factor)
    max_increase_jax = jnp.asarray(max_step_size_increase_factor)
    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    def tempered_logdensity(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity

    def init_one(position, rng_key, beta):
        return blackjax.mclmc.init(position, tempered_logdensity(beta), rng_key=rng_key)

    def step_one(rng_key, state, beta, inverse_mass_matrix, L, step_size):
        kernel = blackjax.mclmc.build_kernel(
            logdensity_fn=tempered_logdensity(beta),
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
        return kernel(rng_key, state, L, step_size)

    vmapped_init = jax.vmap(init_one, in_axes=(0, 0, None))
    vmapped_step = jax.vmap(step_one, in_axes=(0, 0, None, None, None, None))

    @jax.jit
    def one_ps_mclmc_step(loop_key, state, L, step_size, step_size_reference):
        loop_key, resample_key, move_key = jax.random.split(loop_key, 3)

        next_lambda = calculate_next_lambda(state, alpha_jax)
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

        # Geometry update. This uses the current resampled particle cloud.
        # For the usual N >= adaptation_max_particles case this matches the
        # original capped-history helper, which returns current particles.
        inverse_mass_matrix = _diagonal_inverse_mass_matrix(resampled_particles, ridge=ridge_jax)

        move_key, init_key = jax.random.split(move_key)
        init_keys = jax.random.split(init_key, num_particles)
        states = vmapped_init(resampled_particles, init_keys, next_lambda)

        def mclmc_scan_body(carry, mcmc_iter):
            key, states, step_size = carry
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_particles)
            states, _ = vmapped_step(step_keys, states, next_lambda, inverse_mass_matrix, L, step_size)

            should_correct = ((mcmc_iter + 1) % adapt_every_eps) == 0
            step_size = jax.lax.cond(
                should_correct,
                lambda eps: _clipped_step_size_update_device(
                    eps,
                    step_size_reference,
                    smoothing=smoothing_jax,
                    min_step_size=min_step_size_jax,
                    max_step_size=max_step_size_jax,
                    max_decrease_factor=max_decrease_jax,
                    max_increase_factor=max_increase_jax,
                ),
                lambda eps: eps,
                step_size,
            )
            return (key, states, step_size), step_size

        (move_key, states, step_size), step_path = jax.lax.scan(
            mclmc_scan_body,
            (move_key, states, step_size),
            jnp.arange(num_mcmc_steps),
        )

        iteration_particles = states.position
        iteration_log_likelihoods = vmapped_log_likelihood_fn(iteration_particles)

        persistent_particles = jax.tree.map(
            lambda persistent, iteration_p: persistent.at[iteration].set(iteration_p),
            state.persistent_particles,
            iteration_particles,
        )
        persistent_log_Z = state.persistent_log_Z.at[iteration].set(log_Z_t)
        persistent_log_likelihoods = state.persistent_log_likelihoods.at[iteration].set(iteration_log_likelihoods)

        new_state = persistent_sampling.PersistentSMCState(
            persistent_particles=persistent_particles,
            persistent_log_likelihoods=persistent_log_likelihoods,
            persistent_log_Z=persistent_log_Z,
            tempering_schedule=tempering_schedule,
            iteration=iteration,
        )

        weights = jnp.asarray(new_state.persistent_weights)
        norm_w = normalize_weights(weights)
        ess_value = 1.0 / jnp.sum(norm_w ** 2)
        current_particles = iteration_particles  # avoid remove_padding inside jit: state.iteration is dynamic

        metrics = {
            "tempering_param": new_state.tempering_param,
            "logZ": new_state.log_Z,
            "ess": ess_value,
            "step_size": step_size,
            "L": L,
        }
        return loop_key, new_state, current_particles, inverse_mass_matrix, step_size, step_path, metrics

    return one_ps_mclmc_step


def _choose_probe_particles_current_only(
    *,
    rng_key: jax.Array,
    current_particles: jnp.ndarray,
    num_probe_particles: int,
) -> jnp.ndarray:
    n_current = int(current_particles.shape[0])
    num_probe = max(1, min(int(num_probe_particles), n_current))
    idx = jax.random.choice(rng_key, n_current, shape=(num_probe,), replace=False)
    return current_particles[idx]


def run_ps_mclmc_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 500,
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
    num_probe_particles: int = 1,
    probe_num_steps: int | None = None,
    adapt_mclmc_every: int = 5,
    epsilon_smoothing: float = 0.3,
    L_smoothing: float = 0.1,
    min_step_size: float = 1e-6,
    max_step_size: float = 1.0,
    max_step_size_decrease_factor: float = 0.7,
    max_step_size_increase_factor: float = 1.15,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run PS + MCLMC with fewer Python synchronizations.

    Main difference from ps_mclmc_gpu_fast.py:
      * one full PS iteration is compiled as one JAX function;
      * Python does not touch particles/weights between resampling, mutation,
        likelihood update, persistent-state update, and ESS calculation;
      * the BlackJAX adaptation probe remains optional/periodic because it is
        not pure-JAX in this usage pattern.

    For closest original adaptation behavior use:
        adapt_mclmc_every=1, num_probe_particles=4, probe_num_steps=num_mcmc_steps
    For speed use:
        adapt_mclmc_every=5 or 10, num_probe_particles=1, probe_num_steps=max(3, k//5)
    """
    if target is None:
        target = make_target(target_name, dimension, **(target_kwargs or {}))

    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn

    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    work_dtype = initial_particles.dtype

    state = persistent_sampling.init(initial_particles, log_likelihood_fn, max_iterations)

    if probe_num_steps is None:
        probe_num_steps = max(3, int(num_mcmc_steps) // 5)
    else:
        probe_num_steps = int(probe_num_steps)
    adapt_mclmc_every = max(0, int(adapt_mclmc_every))

    one_ps_mclmc_step = _make_one_ps_mclmc_step(
        log_prior_fn,
        log_likelihood_fn,
        num_particles=int(num_particles),
        num_mcmc_steps=int(num_mcmc_steps),
        alpha=float(alpha),
        mass_matrix_ridge=float(mass_matrix_ridge),
        epsilon_smoothing=float(epsilon_smoothing),
        min_step_size=float(min_step_size),
        max_step_size=float(max_step_size),
        max_step_size_decrease_factor=float(max_step_size_decrease_factor),
        max_step_size_increase_factor=float(max_step_size_increase_factor),
    )

    step_size_jax = jnp.asarray(step_size, dtype=work_dtype)
    L_jax = jnp.asarray(L, dtype=work_dtype)
    step_size_reference_jax = jnp.asarray(step_size, dtype=work_dtype)

    tempering_path, logZ_path, ess_path = [], [], []
    elapsed_time_path, step_size_path, L_path = [], [], []
    variance_log_weights_path, weight_entropy_path, esjd_path = [], [], []
    probe_count_path = []

    start = time.perf_counter()
    n_iter = 0
    gradient_eval_count = 0
    resampling_steps = 0
    adaptation_probe_count = 0
    previous_particles_for_esjd = None

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        if return_diagnostics:
            previous_particles_for_esjd = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        loop_key, state, current_particles, inverse_mass_matrix, step_size_jax, step_path_jax, metrics = one_ps_mclmc_step(
            loop_key,
            state,
            L_jax,
            step_size_jax,
            step_size_reference_jax,
        )

        # Optional/periodic Python-side BlackJAX adaptation probe.
        should_probe = (adapt_mclmc_every > 0) and ((n_iter % adapt_mclmc_every) == 0)
        if should_probe:
            loop_key, probe_key, adapt_key = jax.random.split(loop_key, 3)
            beta_for_probe = metrics["tempering_param"]

            def logposterior_fn(x):
                return log_prior_fn(x) + beta_for_probe * log_likelihood_fn(x)

            probe_positions = _choose_probe_particles_current_only(
                rng_key=probe_key,
                current_particles=current_particles,
                num_probe_particles=num_probe_particles,
            )
            probe_result = _run_blackjax_mclmc_adaptation_probe(
                probe_particles=probe_positions,
                logdensity_fn=logposterior_fn,
                inverse_mass_matrix=inverse_mass_matrix,
                step_size=float(step_size_jax),
                L=float(L_jax),
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
            step_size_reference_jax = jnp.asarray(probe_result["step_size"], dtype=work_dtype)
            step_size_jax = jnp.asarray(
                _clipped_step_size_update(
                    old_step_size=float(step_size_jax),
                    proposed_step_size=float(probe_result["step_size"]),
                    smoothing=epsilon_smoothing,
                    min_step_size=min_step_size,
                    max_step_size=max_step_size,
                    max_decrease_factor=max_step_size_decrease_factor,
                    max_increase_factor=max_step_size_increase_factor,
                ),
                dtype=work_dtype,
            )
            L_jax = jnp.asarray(_smooth_positive_update(float(L_jax), float(probe_result["L"]), L_smoothing), dtype=work_dtype)
            gradient_eval_count += int(probe_result["tuning_integrator_steps"])
            adaptation_probe_count += 1
            if return_diagnostics:
                probe_count_path.append(int(probe_result["num_probe_particles"]))
        elif return_diagnostics:
            probe_count_path.append(0)

        # One scalar synchronization per metric we keep. This is much cheaper than
        # synchronizing after each substep or copying particles every iteration.
        tempering_path.append(float(metrics["tempering_param"]))
        logZ_path.append(float(metrics["logZ"]))
        ess_path.append(float(metrics["ess"]))
        elapsed_time_path.append(float(time.perf_counter() - start))

        if return_diagnostics:
            particles_np = np.asarray(current_particles)
            weights = jnp.asarray(state.persistent_weights)
            variance_log_weights_path.append(variance_log_weights(weights))
            weight_entropy_path.append(weight_entropy(weights))
            esjd_path.append(population_esjd(previous_particles_for_esjd, particles_np))
            step_size_path.append(float(step_size_jax))
            L_path.append(float(L_jax))

        resampling_steps += 1
        gradient_eval_count += int(num_particles) * int(num_mcmc_steps)
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = compute_posterior_moments_from_particles(final_particles)
    coordinate_summaries = coordinate_specific_summaries(posterior_mean, posterior_second_moment)

    out = {
        "target_name": target.name,
        "algorithm_name": "ps",
        "kernel_name": "mclmc",
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
        "final_ess": float(ess_path[-1]) if ess_path else np.nan,
        "acceptance_rate_mean": np.nan,
        "acceptance_rate_last": np.nan,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "gradient_eval_count": int(gradient_eval_count),
        "resampling_steps": int(resampling_steps),
        "final_step_size": float(step_size_jax),
        "final_L": float(L_jax),
        "probe_num_steps": int(probe_num_steps),
        "num_probe_particles": int(num_probe_particles),
        "adapt_mclmc_every": int(adapt_mclmc_every),
        "adaptation_probe_count": int(adaptation_probe_count),
    }

    if return_diagnostics:
        out.update({
            "tempering_path": np.asarray(tempering_path, dtype=float),
            "logZ_path": np.asarray(logZ_path, dtype=float),
            "ess_path": np.asarray(ess_path, dtype=float),
            "elapsed_time_path": np.asarray(elapsed_time_path, dtype=float),
            "variance_log_weights_path": np.asarray(variance_log_weights_path, dtype=float),
            "weight_entropy_path": np.asarray(weight_entropy_path, dtype=float),
            "esjd_path": np.asarray(esjd_path, dtype=float),
            "step_size_path": np.asarray(step_size_path, dtype=float),
            "L_path": np.asarray(L_path, dtype=float),
            "probe_count_path": np.asarray(probe_count_path, dtype=int),
        })
    return out