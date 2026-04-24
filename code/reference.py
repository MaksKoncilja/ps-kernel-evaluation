# reference.py
# paralelised version with Robbins-Monro diminsihing adaptation
# Robbins-Monro addaptation each MCMC step with dimminishing schedule for each SMC step.
# each worker: seed -> build target -> run SMC -> compute summary -> return dict
# parent procees: collect dicts -> aggregate -> save JSON
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Force CPU unless the environment already specified something else.
# This must happen before importing jax.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as smc_solver

from targets import make_target


def debug_worker_backend() -> dict[str, Any]:
    # whicih device (gpu, cpu) jax is using and which are available
    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }


@dataclass
class ReferenceStats:  # container for the final refrence results
    target_name: str
    dimension: int
    num_reference_runs: int
    num_particles_reference: int
    logZ_ref_mean: float
    logZ_ref_std: float
    f1_mean_ref: np.ndarray   # E[theta]
    f1_std_ref: np.ndarray
    f2_mean_ref: np.ndarray   # E[theta^2]
    f2_std_ref: np.ndarray
    final_ess_mean: float
    final_ess_std: float
    acceptance_rate_mean: float
    acceptance_rate_std: float
    n_iter_mean: float
    n_iter_std: float
    runtime_sec_mean: float
    runtime_sec_std: float


def _default_reference_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # assume project root is one level up form code folder
    base_dir = project_root / "data" / "results" / "reference"
    checkpoint_path = base_dir / "checkpoints" / "reference_checkpoint.json"
    save_final_path = base_dir / "summary" / "reference_stats.json"
    return checkpoint_path, save_final_path  # path for chunked results and final reference statistics


def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    # returns scalar mean acceptance rate over particles for one MCMC step
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        arr = jnp.asarray(acc)
        return jnp.mean(arr)
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        arr = jnp.asarray(is_accepted)
        return jnp.mean(arr)
    return jnp.nan


def _empirical_covariance(particles: jnp.ndarray, ridge: float = 1e-6, diagonal_only: bool = False,) -> jnp.ndarray:
    # Empirical covariance of particles with small ridge regularization.
    x = particles
    mean = jnp.mean(x, axis=0, keepdims=True)
    xc = x - mean
    n = x.shape[0]
    denom = jnp.maximum(n - 1, 1)
    cov = (xc.T @ xc) / denom
    if diagonal_only:
        cov = jnp.diag(jnp.diag(cov))
    d = cov.shape[0]
    cov = cov + ridge * jnp.eye(d, dtype=cov.dtype)
    return cov


def _proposal_sqrt_from_cov(cov: jnp.ndarray, scale: float) -> jnp.ndarray:
    # cholesky decomposition of covariance matrix multiplied by the scale factor.
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(t: int, c: float = 1.0, t0: float = 5.0, kappa: float = 0.6,) -> float:
    # diminishing Robbins-Monro schedule gamma_t = c / (t + t0)^kappa
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(rw_scale: jnp.ndarray, acceptance_value: float,target_acceptance_rate: float, 
                                   t: int, rm_c: float = 1.0, rm_t0: float = 5.0, rm_kappa: float = 0.6,
                                   rw_scale_min: float | None = None,
                                   rw_scale_max: float | None = None,) -> jnp.ndarray:
    # Robbins-Monro adaptation on the log scale
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    new_rw_scale = jnp.exp(log_rw_scale)

    if rw_scale_min is not None:
        new_rw_scale = jnp.maximum(new_rw_scale, jnp.asarray(rw_scale_min, dtype=new_rw_scale.dtype))
    if rw_scale_max is not None:
        new_rw_scale = jnp.minimum(new_rw_scale, jnp.asarray(rw_scale_max, dtype=new_rw_scale.dtype))

    return new_rw_scale


def _update_step_size_robbins_monro(step_size: jnp.ndarray, acceptance_value: float,target_acceptance_rate: float,
                                    t: int, rm_c: float = 2.0, rm_t0: float = 1.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    # Robbins-Monro adaptation on the log step-size scale
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_step_size = jnp.log(step_size)
    log_step_size = log_step_size + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_step_size)


def adaptive_loop_with_rwm_adaptation(  key,
                                        *,
                                        log_prior_fn,
                                        log_likelihood_fn,
                                        initial_particles,
                                        dimension: int,
                                        max_iterations: int,
                                        target_ess: float,
                                        num_mcmc_steps: int,
                                        initial_rw_scale: float,
                                        target_acceptance_rate: float = 0.234,
                                        covariance_ridge: float = 1e-6,
                                        diagonal_only_covariance: bool = False,
                                        rm_c: float = 1.0,
                                        rm_t0: float = 5.0,
                                        rm_kappa: float = 0.6,
                                        rw_scale_min: float | None = None,
                                        rw_scale_max: float | None = None,
                                        freeze_adaptation_beta: float | None = None,):
    state = blackjax.smc.tempered.init(initial_particles)
    work_dtype = initial_particles.dtype
    rw_scale = jnp.asarray(initial_rw_scale, dtype=work_dtype)
    proposal_cov = _empirical_covariance(initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)
    tempering_path, logZ_path, ess_path = [], [], []
    acceptance_path, rw_scale_path = [], []
    logZ, n_iter = 0.0, 0
    base_rmh_kernel = blackjax.rmh.build_kernel()
    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)
    def tempered_logdensity_fn(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity
    
    def rmh_step_fn(rng_key, mcmc_state, logdensity_fn, rw_scale, proposal_cov):
        # apply one RWMH step to every particle in parallel
        transition_generator = blackjax.mcmc.random_walk.normal(_proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel(rng_key, mcmc_state, logdensity_fn, transition_generator=transition_generator,)
    vmapped_rmh_init = jax.vmap( lambda position, logdensity_fn: blackjax.rmh.init(position, logdensity_fn), in_axes=(0, None),)
    vmapped_rmh_step = jax.vmap(rmh_step_fn, in_axes=(0, 0, None, None, None))
    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        current_beta = jnp.asarray(state.tempering_param, dtype=work_dtype)
        one = jnp.asarray(1.0, dtype=work_dtype)
        zero = jnp.asarray(0.0, dtype=work_dtype)
        target_ess_typed = jnp.asarray(target_ess, dtype=work_dtype)
        max_delta = one - current_beta

        # Choose next temperature using current particles.
        delta_beta = smc_ess.ess_solver(vmapped_log_likelihood_fn, state.particles, target_ess_typed, max_delta, smc_solver.dichotomy, )
        delta_beta = jnp.clip(delta_beta, zero, max_delta)
        next_beta = current_beta + delta_beta

        # Incremental weights for moving from current_beta to next_beta.
        current_loglik = vmapped_log_likelihood_fn(state.particles)
        incremental_log_weights = delta_beta * current_loglik
        normalized_current_weights = jax.nn.softmax(incremental_log_weights)

        ess_value = float(1.0 / jnp.sum(normalized_current_weights ** 2))
        # Standard SMC logZ update: use pre-move particle system.
        logZ += float(jax.scipy.special.logsumexp(incremental_log_weights) - jnp.log(incremental_log_weights.shape[0]))
        # Resample using the current normalized incremental weights.
        key, resample_key = jax.random.split(key)
        ancestors = resampling.systematic( resample_key, normalized_current_weights, state.particles.shape[0], )
        resampled_particles = state.particles[ancestors]
        # Move under the NEW tempered target p_t with beta = next_beta.
        next_logdensity_fn = tempered_logdensity_fn(next_beta)
        mcmc_states = vmapped_rmh_init(resampled_particles, next_logdensity_fn)

        # Robbins-Monro counter resets inside each SMC iteration: 1, ..., k.
        for mcmc_iter in range(num_mcmc_steps):
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, resampled_particles.shape[0])

            mcmc_states, update_info = vmapped_rmh_step(step_keys, mcmc_states,
                                                        next_logdensity_fn, rw_scale, proposal_cov,)
            acceptance_value = float(_safe_mean_acceptance(update_info))
            acceptance_path.append(acceptance_value)

            allow_adaptation = not np.isnan(acceptance_value)
            if freeze_adaptation_beta is not None:
                allow_adaptation = allow_adaptation and (float(next_beta) < freeze_adaptation_beta)

            if allow_adaptation:
                rw_scale = _update_rw_scale_robbins_monro(  rw_scale=rw_scale,
                                                            acceptance_value=acceptance_value,
                                                            target_acceptance_rate=target_acceptance_rate,
                                                            t=mcmc_iter + 1,
                                                            rm_c=rm_c,
                                                            rm_t0=rm_t0,
                                                            rm_kappa=rm_kappa,
                                                            rw_scale_min=rw_scale_min,
                                                            rw_scale_max=rw_scale_max,)

            rw_scale_path.append(float(rw_scale))

        moved_particles = mcmc_states.position

        # After resample + move, particles are treated as equally weighted.
        num_particles = moved_particles.shape[0]
        uniform_weights = jnp.ones((num_particles,), dtype=work_dtype) / num_particles

        state = blackjax.smc.tempered.TemperedSMCState( particles=moved_particles,
                                                        weights=uniform_weights,
                                                        tempering_param=next_beta,)

        tempering_path.append(float(next_beta))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)

        proposal_cov = _empirical_covariance(state.particles,ridge=covariance_ridge,diagonal_only=diagonal_only_covariance,)
        n_iter += 1

    diagnostics = { "tempering_path": np.asarray(tempering_path, dtype=float),
                    "logZ_path": np.asarray(logZ_path, dtype=float),
                    "ess_path": np.asarray(ess_path, dtype=float),
                    "acceptance_path": np.asarray(acceptance_path, dtype=float),
                    "rw_scale_path": np.asarray(rw_scale_path, dtype=float),
                    "final_logZ": float(logZ),
                    "final_rw_scale": float(rw_scale),}

    return n_iter, state, diagnostics


def adaptive_loop_with_mala_adaptation(key,
                                       *,
                                       log_prior_fn,
                                       log_likelihood_fn,
                                       initial_particles,
                                       dimension: int,
                                       max_iterations: int,
                                       target_ess: float,
                                       num_mcmc_steps: int,
                                       initial_step_size: float,
                                       target_acceptance_rate: float = 0.574,
                                       rm_c: float = 2.0,
                                       rm_t0: float = 1.0,
                                       rm_kappa: float = 0.6,):
    state = blackjax.smc.tempered.init(initial_particles)
    work_dtype = initial_particles.dtype
    step_size = jnp.asarray(initial_step_size, dtype=work_dtype)
    tempering_path, logZ_path, ess_path = [], [], []
    acceptance_path, step_size_path = [], []
    logZ, n_iter = 0.0, 0
    base_mala_kernel = blackjax.mala.build_kernel()
    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    def tempered_logdensity_fn(beta):
        def _logdensity(theta):
            return log_prior_fn(theta) + beta * log_likelihood_fn(theta)
        return _logdensity

    def mala_step_fn(rng_key, mcmc_state, logdensity_fn, step_size):
        # apply one MALA step to every particle in parallel
        return base_mala_kernel(rng_key, mcmc_state, logdensity_fn, step_size=step_size)

    vmapped_mala_init = jax.vmap(
        lambda position, logdensity_fn: blackjax.mala.init(position, logdensity_fn),
        in_axes=(0, None),
    )
    vmapped_mala_step = jax.vmap(mala_step_fn, in_axes=(0, 0, None, None))

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        current_beta = jnp.asarray(state.tempering_param, dtype=work_dtype)
        one = jnp.asarray(1.0, dtype=work_dtype)
        zero = jnp.asarray(0.0, dtype=work_dtype)
        target_ess_typed = jnp.asarray(target_ess, dtype=work_dtype)
        max_delta = one - current_beta

        # Choose next temperature using current particles.
        delta_beta = smc_ess.ess_solver(vmapped_log_likelihood_fn, state.particles, target_ess_typed, max_delta, smc_solver.dichotomy, )
        delta_beta = jnp.clip(delta_beta, zero, max_delta)
        next_beta = current_beta + delta_beta

        # Incremental weights for moving from current_beta to next_beta.
        current_loglik = vmapped_log_likelihood_fn(state.particles)
        incremental_log_weights = delta_beta * current_loglik
        normalized_current_weights = jax.nn.softmax(incremental_log_weights)

        ess_value = float(1.0 / jnp.sum(normalized_current_weights ** 2))
        # Standard SMC logZ update: use pre-move particle system.
        logZ += float(jax.scipy.special.logsumexp(incremental_log_weights) - jnp.log(incremental_log_weights.shape[0]))
        # Resample using the current normalized incremental weights.
        key, resample_key = jax.random.split(key)
        ancestors = resampling.systematic(resample_key, normalized_current_weights, state.particles.shape[0],)
        resampled_particles = state.particles[ancestors]
        # Move under the NEW tempered target p_t with beta = next_beta.
        next_logdensity_fn = tempered_logdensity_fn(next_beta)
        mcmc_states = vmapped_mala_init(resampled_particles, next_logdensity_fn)

        # Robbins-Monro counter resets inside each SMC iteration: 1, ..., k.
        for mcmc_iter in range(num_mcmc_steps):
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, resampled_particles.shape[0])

            mcmc_states, update_info = vmapped_mala_step(step_keys, mcmc_states,
                                                         next_logdensity_fn, step_size,)
            acceptance_value = float(_safe_mean_acceptance(update_info))
            acceptance_path.append(acceptance_value)

            if not np.isnan(acceptance_value):
                step_size = _update_step_size_robbins_monro(step_size=step_size,
                                                            acceptance_value=acceptance_value,
                                                            target_acceptance_rate=target_acceptance_rate,
                                                            t=mcmc_iter + 1,
                                                            rm_c=rm_c,
                                                            rm_t0=rm_t0,
                                                            rm_kappa=rm_kappa,)

            step_size_path.append(float(step_size))

        moved_particles = mcmc_states.position

        # After resample + move, particles are treated as equally weighted.
        num_particles = moved_particles.shape[0]
        uniform_weights = jnp.ones((num_particles,), dtype=work_dtype) / num_particles

        state = blackjax.smc.tempered.TemperedSMCState(particles=moved_particles,
                                                       weights=uniform_weights,
                                                       tempering_param=next_beta,)

        tempering_path.append(float(next_beta))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)
        n_iter += 1

    diagnostics = {"tempering_path": np.asarray(tempering_path, dtype=float),
                   "logZ_path": np.asarray(logZ_path, dtype=float),
                   "ess_path": np.asarray(ess_path, dtype=float),
                   "acceptance_path": np.asarray(acceptance_path, dtype=float),
                   "step_size_path": np.asarray(step_size_path, dtype=float),
                   "final_logZ": float(logZ),
                   "final_step_size": float(step_size),}

    return n_iter, state, diagnostics


def compute_posterior_moments_from_particles(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # first and second moment f1 = theta, f2 = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def run_reference_sampler_once( dimension: int,
                                num_particles: int,
                                seed: int,
                                target_name: str = "gaussian_mixture",
                                target_kwargs: dict[str, Any] | None = None,
                                max_iterations: int = 10_000,
                                alpha: float = 0.999,
                                num_mcmc_steps: int = 25,
                                kernel_name: str = "rwm",
                                rw_step_size: float = 0.5,
                                target_acceptance_rate: float = 0.234,
                                covariance_ridge: float = 1e-6,
                                diagonal_only_covariance: bool = False,
                                rm_c: float = 1.0,
                                rm_t0: float = 5.0,
                                rm_kappa: float = 0.6,
                                rw_scale_min: float | None = None,
                                rw_scale_max: float | None = None,
                                freeze_adaptation_beta: float | None = None,
                                mala_step_size: float = 0.02,
                                mala_target_acceptance_rate: float = 0.574,
                                mala_rm_c: float = 2.0,
                                mala_rm_t0: float = 1.0,
                                mala_rm_kappa: float = 0.6,
                                return_diagnostics: bool = True,) -> dict[str, Any]:
    # run refrence SMC for one seed

    target = make_target(target_name, dimension, **(target_kwargs or {}))
    true_dimension = int(target.dimension)

    # for sparse logistic regression use the full empirical covariance matrix.
    # this target is 51-dimensional in the paper and the reference runs use particle counts
    # large enough for a stable covariance estimate, so full covariance + cholesky is reasonable here.
    if target.name == "sparse_logistic_regression" and kernel_name == "rwm":
        diagonal_only_covariance = False
        covariance_ridge = max(covariance_ridge, 1e-4)

    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, run_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)

    start = time.perf_counter()

    if kernel_name == "rwm":
        n_iter, smc_final_state, diagnostics = adaptive_loop_with_rwm_adaptation(run_key,
                                                                                 log_prior_fn=log_prior_fn,
                                                                                 log_likelihood_fn=log_likelihood_fn,
                                                                                 initial_particles=initial_particles,
                                                                                 dimension=true_dimension,
                                                                                 max_iterations=max_iterations,
                                                                                 target_ess=alpha,
                                                                                 num_mcmc_steps=num_mcmc_steps,
                                                                                 initial_rw_scale=rw_step_size,
                                                                                 target_acceptance_rate=target_acceptance_rate,
                                                                                 covariance_ridge=covariance_ridge,
                                                                                 diagonal_only_covariance=diagonal_only_covariance,
                                                                                 rm_c=rm_c,
                                                                                 rm_t0=rm_t0,
                                                                                 rm_kappa=rm_kappa,
                                                                                 rw_scale_min=rw_scale_min,
                                                                                 rw_scale_max=rw_scale_max,
                                                                                 freeze_adaptation_beta=freeze_adaptation_beta,)
    elif kernel_name == "mala":
        n_iter, smc_final_state, diagnostics = adaptive_loop_with_mala_adaptation(run_key,
                                                                                  log_prior_fn=log_prior_fn,
                                                                                  log_likelihood_fn=log_likelihood_fn,
                                                                                  initial_particles=initial_particles,
                                                                                  dimension=true_dimension,
                                                                                  max_iterations=max_iterations,
                                                                                  target_ess=alpha,
                                                                                  num_mcmc_steps=num_mcmc_steps,
                                                                                  initial_step_size=mala_step_size,
                                                                                  target_acceptance_rate=mala_target_acceptance_rate,
                                                                                  rm_c=mala_rm_c,
                                                                                  rm_t0=mala_rm_t0,
                                                                                  rm_kappa=mala_rm_kappa,)
    else:
        raise ValueError(f"Unknown kernel_name: {kernel_name}")

    runtime_sec = time.perf_counter() - start

    particles = np.array(smc_final_state.particles)
    f1, f2 = compute_posterior_moments_from_particles(particles)
    ess_path = np.asarray(diagnostics["ess_path"], dtype=float)
    acceptance_path = np.asarray(diagnostics["acceptance_path"], dtype=float)
    final_ess = float(ess_path[-1]) if ess_path.size > 0 else np.nan
    acceptance_rate_mean = float(np.nanmean(acceptance_path)) if acceptance_path.size > 0 else np.nan

    out = {"target_name": target.name,
            "kernel_name": kernel_name,
            "dimension": true_dimension,
            "seed": int(seed),
            "logZ": float(diagnostics["final_logZ"]),
            "posterior_mean": f1,
            "posterior_second_moment": f2,
            "n_iter": int(n_iter),
            "runtime_sec": float(runtime_sec),
            "final_ess": final_ess,
            "acceptance_rate_mean": acceptance_rate_mean, }

    if kernel_name == "rwm":
        out["final_rw_scale"] = float(diagnostics["final_rw_scale"])
    elif kernel_name == "mala":
        out["final_step_size"] = float(diagnostics["final_step_size"])

    if return_diagnostics:
        out.update({"tempering_path": np.asarray(diagnostics["tempering_path"], dtype=float),
                    "logZ_path": np.asarray(diagnostics["logZ_path"], dtype=float),
                    "ess_path": ess_path,
                    "acceptance_path": acceptance_path,})

        if kernel_name == "rwm":
            out["rw_scale_path"] = np.asarray(diagnostics["rw_scale_path"], dtype=float)
        elif kernel_name == "mala":
            out["step_size_path"] = np.asarray(diagnostics["step_size_path"], dtype=float)

    return out


def _run_reference_worker(args: dict[str, Any]) -> dict[str, Any]:
    # Top-level worker wrapper for multiprocessing, where Each worker runs exactly one seed.
    return run_reference_sampler_once(**args)


def _append_reference_output(out: dict[str, Any],
                            completed_seeds,
                            logZ_runs,
                            f1_runs,
                            f2_runs,
                            final_ess_runs,
                            acceptance_rate_runs,
                            n_iter_runs,
                            runtime_sec_runs,) -> None:
    # takes the result of one run appending it to the aggregation lists

    completed_seeds.append(out["seed"])
    logZ_runs.append(out["logZ"])
    f1_runs.append(out["posterior_mean"])
    f2_runs.append(out["posterior_second_moment"])
    final_ess_runs.append(out["final_ess"])
    acceptance_rate_runs.append(out["acceptance_rate_mean"])
    n_iter_runs.append(out["n_iter"])
    runtime_sec_runs.append(out["runtime_sec"])


def _sort_runs_by_seed( completed_seeds,
                        logZ_runs,
                        f1_runs,
                        f2_runs,
                        final_ess_runs,
                        acceptance_rate_runs,
                        n_iter_runs,
                        runtime_sec_runs,):
    # for easyer reproducibility the seeds are ordered
    if len(completed_seeds) == 0:
        return (completed_seeds, logZ_runs,f1_runs,
                f2_runs, final_ess_runs,
                acceptance_rate_runs, n_iter_runs,
                runtime_sec_runs,)

    order = np.argsort(np.asarray(completed_seeds))
    completed_seeds = [completed_seeds[i] for i in order]
    logZ_runs = [logZ_runs[i] for i in order]
    f1_runs = [f1_runs[i] for i in order]
    f2_runs = [f2_runs[i] for i in order]
    final_ess_runs = [final_ess_runs[i] for i in order]
    acceptance_rate_runs = [acceptance_rate_runs[i] for i in order]
    n_iter_runs = [n_iter_runs[i] for i in order]
    runtime_sec_runs = [runtime_sec_runs[i] for i in order]

    return (completed_seeds, logZ_runs, f1_runs, f2_runs,
            final_ess_runs, acceptance_rate_runs,n_iter_runs,
            runtime_sec_runs,)


def _assemble_reference_stats(  target_name: str,
                                dimension: int,
                                num_particles_reference: int,
                                num_reference_runs: int,
                                logZ_runs,
                                f1_runs,
                                f2_runs,
                                final_ess_runs,
                                acceptance_rate_runs,
                                n_iter_runs,
                                runtime_sec_runs,) -> ReferenceStats:
    # takes the outputs of multiple runns and combines them into RefrenceSatas object
    logZ_runs = np.asarray(logZ_runs, dtype=float)
    f1_runs = np.stack(f1_runs, axis=0)
    f2_runs = np.stack(f2_runs, axis=0)
    final_ess_runs = np.asarray(final_ess_runs, dtype=float)
    acceptance_rate_runs = np.asarray(acceptance_rate_runs, dtype=float)
    n_iter_runs = np.asarray(n_iter_runs, dtype=float)
    runtime_sec_runs = np.asarray(runtime_sec_runs, dtype=float)

    logZ_ref_std = float(np.nanstd(logZ_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    f1_std_ref = np.std(f1_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f1_runs[0])
    f2_std_ref = np.std(f2_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f2_runs[0])
    final_ess_std = float(np.nanstd(final_ess_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    acceptance_rate_std = float(np.nanstd(acceptance_rate_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    n_iter_std = float(np.nanstd(n_iter_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    runtime_sec_std = float(np.nanstd(runtime_sec_runs, ddof=1)) if num_reference_runs > 1 else 0.0

    return ReferenceStats(  target_name=target_name,
                            dimension=int(dimension),
                            num_reference_runs=num_reference_runs,
                            num_particles_reference=num_particles_reference,
                            logZ_ref_mean=float(np.nanmean(logZ_runs)),
                            logZ_ref_std=logZ_ref_std,
                            f1_mean_ref=np.mean(f1_runs, axis=0),
                            f1_std_ref=f1_std_ref,
                            f2_mean_ref=np.mean(f2_runs, axis=0),
                            f2_std_ref=f2_std_ref,
                            final_ess_mean=float(np.nanmean(final_ess_runs)),
                            final_ess_std=final_ess_std,
                            acceptance_rate_mean=float(np.nanmean(acceptance_rate_runs)),
                            acceptance_rate_std=acceptance_rate_std,
                            n_iter_mean=float(np.nanmean(n_iter_runs)),
                            n_iter_std=n_iter_std,
                            runtime_sec_mean=float(np.nanmean(runtime_sec_runs)),
                            runtime_sec_std=runtime_sec_std,)


def save_reference_stats(ref: ReferenceStats, outpath: str | Path) -> None:
    # saves ReferenceStats object into JSON file.
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(ref)
    payload["f1_mean_ref"] = ref.f1_mean_ref.tolist()
    payload["f1_std_ref"] = ref.f1_std_ref.tolist()
    payload["f2_mean_ref"] = ref.f2_mean_ref.tolist()
    payload["f2_std_ref"] = ref.f2_std_ref.tolist()
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_reference_stats(path: str | Path) -> ReferenceStats:
    # loads the saved JSON file with an ReferenceStats object
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ReferenceStats(  target_name=payload["target_name"],
                            dimension=int(payload["dimension"]),
                            num_reference_runs=payload["num_reference_runs"],
                            num_particles_reference=payload["num_particles_reference"],
                            logZ_ref_mean=payload["logZ_ref_mean"],
                            logZ_ref_std=payload["logZ_ref_std"],
                            f1_mean_ref=np.asarray(payload["f1_mean_ref"], dtype=float),
                            f1_std_ref=np.asarray(payload["f1_std_ref"], dtype=float),
                            f2_mean_ref=np.asarray(payload["f2_mean_ref"], dtype=float),
                            f2_std_ref=np.asarray(payload["f2_std_ref"], dtype=float),
                            final_ess_mean=payload["final_ess_mean"],
                            final_ess_std=payload["final_ess_std"],
                            acceptance_rate_mean=payload["acceptance_rate_mean"],
                            acceptance_rate_std=payload["acceptance_rate_std"],
                            n_iter_mean=payload["n_iter_mean"],
                            n_iter_std=payload["n_iter_std"],
                            runtime_sec_mean=payload.get("runtime_sec_mean", np.nan),
                            runtime_sec_std=payload.get("runtime_sec_std", np.nan),)


def save_chunk_checkpoint(  checkpoint_path: str | Path,
                            *,
                            target_name: str,
                            dimension: int,
                            num_particles_reference: int,
                            num_reference_runs: int,
                            completed_seeds,
                            logZ_runs,
                            f1_runs,
                            f2_runs,
                            final_ess_runs,
                            acceptance_rate_runs,
                            n_iter_runs,
                            runtime_sec_runs,) -> None:
    # this saces the progres of the chunked execution
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = { "target_name": str(target_name),
                "dimension": int(dimension),
                "num_particles_reference": int(num_particles_reference),
                "num_reference_runs": int(num_reference_runs),
                "completed_seeds": list(map(int, completed_seeds)),
                "logZ_runs": np.asarray(logZ_runs, dtype=float).tolist(),
                "f1_runs": np.asarray(f1_runs, dtype=float).tolist(),
                "f2_runs": np.asarray(f2_runs, dtype=float).tolist(),
                "final_ess_runs": np.asarray(final_ess_runs, dtype=float).tolist(),
                "acceptance_rate_runs": np.asarray(acceptance_rate_runs, dtype=float).tolist(),
                "n_iter_runs": np.asarray(n_iter_runs, dtype=float).tolist(),
                "runtime_sec_runs": np.asarray(runtime_sec_runs, dtype=float).tolist(),}

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_chunk_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["completed_seeds"] = list(map(int, payload["completed_seeds"]))
    payload["logZ_runs"] = np.asarray(payload["logZ_runs"], dtype=float)
    payload["f1_runs"] = np.asarray(payload["f1_runs"], dtype=float)
    payload["f2_runs"] = np.asarray(payload["f2_runs"], dtype=float)
    payload["final_ess_runs"] = np.asarray(payload["final_ess_runs"], dtype=float)
    payload["acceptance_rate_runs"] = np.asarray(payload["acceptance_rate_runs"], dtype=float)
    payload["n_iter_runs"] = np.asarray(payload["n_iter_runs"], dtype=float)
    if "runtime_sec_runs" in payload:
        payload["runtime_sec_runs"] = np.asarray(payload["runtime_sec_runs"], dtype=float)
    else:
        payload["runtime_sec_runs"] = np.full_like(payload["n_iter_runs"], np.nan, dtype=float)
    return payload


def build_reference_stats_chunked(  dimension: int,
                                    num_particles_reference: int,
                                    num_reference_runs: int,
                                    target_name: str = "gaussian_mixture",
                                    target_kwargs: dict[str, Any] | None = None,
                                    chunk_size: int = 10,
                                    max_iterations: int = 10_000,
                                    alpha: float = 0.999,
                                    num_mcmc_steps: int = 25,
                                    kernel_name: str = "rwm",
                                    rw_step_size: float = 0.5,
                                    target_acceptance_rate: float = 0.234,
                                    covariance_ridge: float = 1e-6,
                                    diagonal_only_covariance: bool = False,
                                    rm_c: float = 1.0,
                                    rm_t0: float = 5.0,
                                    rm_kappa: float = 0.6,
                                    rw_scale_min: float | None = None,
                                    rw_scale_max: float | None = None,
                                    freeze_adaptation_beta: float | None = None,
                                    mala_step_size: float = 0.02,
                                    mala_target_acceptance_rate: float = 0.574,
                                    mala_rm_c: float = 2.0,
                                    mala_rm_t0: float = 1.0,
                                    mala_rm_kappa: float = 0.6,
                                    verbose: bool = True,
                                    checkpoint_path: str | Path | None = None,
                                    save_final_path: str | Path | None = None,
                                    parallel: bool = False,
                                    num_workers: int | None = None,
                                ) -> ReferenceStats:
    # runs multiple reference runs in chunks

    start_time = time.perf_counter()

    # determine the true target dimension once, using the actual target construction
    target_for_metadata = make_target(target_name, dimension, **(target_kwargs or {}))
    true_dimension = int(target_for_metadata.dimension)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_chunk_checkpoint(checkpoint_path)

        if (ckpt["target_name"] != target_name
            or ckpt["dimension"] != true_dimension
            or ckpt["num_particles_reference"] != num_particles_reference
            or ckpt["num_reference_runs"] != num_reference_runs):
            raise ValueError("Checkpoint metadata does not match current run settings.")

        completed_seeds = list(ckpt["completed_seeds"])
        logZ_runs = list(ckpt["logZ_runs"])
        f1_runs = list(ckpt["f1_runs"])
        f2_runs = list(ckpt["f2_runs"])
        final_ess_runs = list(ckpt["final_ess_runs"])
        acceptance_rate_runs = list(ckpt["acceptance_rate_runs"])
        n_iter_runs = list(ckpt["n_iter_runs"])
        runtime_sec_runs = list(ckpt["runtime_sec_runs"])

        if verbose:
            print(f"Loaded checkpoint with {len(completed_seeds)} completed runs.")
    else:
        completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs, runtime_sec_runs = [], [], [], [], [], [], [], []

    num_completed = len(completed_seeds)
    total_runs = num_reference_runs

    completed_seed_set = set(completed_seeds)
    remaining_seeds = [seed for seed in range(num_reference_runs) if seed not in completed_seed_set]

    for chunk_start in range(0, len(remaining_seeds), chunk_size):
        chunk = remaining_seeds[chunk_start:chunk_start + chunk_size]
        if verbose:
            print(f"Running reference chunk seeds {chunk[0]}:{chunk[-1] + 1}")

        job_args_list = []
        for seed in chunk:
            job_args_list.append({  "target_name": target_name,
                                    "dimension": true_dimension,
                                    "num_particles": num_particles_reference,
                                    "target_kwargs": target_kwargs,
                                    "seed": seed,
                                    "max_iterations": max_iterations,
                                    "alpha": alpha,
                                    "num_mcmc_steps": num_mcmc_steps,
                                    "kernel_name": kernel_name,
                                    "rw_step_size": rw_step_size,
                                    "target_acceptance_rate": target_acceptance_rate,
                                    "covariance_ridge": covariance_ridge,
                                    "diagonal_only_covariance": diagonal_only_covariance,
                                    "rm_c": rm_c,
                                    "rm_t0": rm_t0,
                                    "rm_kappa": rm_kappa,
                                    "rw_scale_min": rw_scale_min,
                                    "rw_scale_max": rw_scale_max,
                                    "freeze_adaptation_beta": freeze_adaptation_beta,
                                    "mala_step_size": mala_step_size,
                                    "mala_target_acceptance_rate": mala_target_acceptance_rate,
                                    "mala_rm_c": mala_rm_c,
                                    "mala_rm_t0": mala_rm_t0,
                                    "mala_rm_kappa": mala_rm_kappa,
                                    "return_diagnostics": False,})

        if parallel:
            workers = num_workers or min(len(chunk), os.cpu_count() or 1)
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = [executor.submit(_run_reference_worker, job_args) for job_args in job_args_list]

                for future in as_completed(futures):
                    out = future.result()
                    _append_reference_output(out, completed_seeds, logZ_runs, f1_runs, f2_runs,
                                            final_ess_runs, acceptance_rate_runs, n_iter_runs, runtime_sec_runs,)
                    num_completed += 1
                    if verbose:
                        elapsed = time.perf_counter() - start_time
                        avg_time = elapsed / max(num_completed, 1)
                        remaining = avg_time * (total_runs - num_completed)
                        print(  f"Run {num_completed}/{total_runs} completed "
                                f"(seed={out['seed']}) | "
                                f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s")
        else:
            for job_args in job_args_list:
                out = _run_reference_worker(job_args)
                _append_reference_output(
                    out,completed_seeds, logZ_runs, f1_runs,f2_runs,
                    final_ess_runs, acceptance_rate_runs, n_iter_runs, runtime_sec_runs,)
                num_completed += 1
                if verbose:
                    elapsed = time.perf_counter() - start_time
                    avg_time = elapsed / max(num_completed, 1)
                    remaining = avg_time * (total_runs - num_completed)
                    print(f"Run {num_completed}/{total_runs} completed "f"(seed={out['seed']}) | "f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s")

        (completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs,
        n_iter_runs, runtime_sec_runs) = _sort_runs_by_seed(completed_seeds, logZ_runs, f1_runs, f2_runs,
                                                            final_ess_runs, acceptance_rate_runs, n_iter_runs, runtime_sec_runs,)

        if checkpoint_path is not None:
            save_chunk_checkpoint(  checkpoint_path,
                                    target_name=target_name,
                                    dimension=true_dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    completed_seeds=completed_seeds,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,
                                    runtime_sec_runs=runtime_sec_runs,)
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    ref = _assemble_reference_stats(target_name=target_name,
                                    dimension=true_dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,
                                    runtime_sec_runs=runtime_sec_runs,)

    if save_final_path is not None:
        save_reference_stats(ref, save_final_path)
        if verbose:
            print(f"Final reference stats saved to {save_final_path}")

    return ref


def nan_report_reference_stats(ref: ReferenceStats) -> dict[str, Any]:
    # alnalyses if the saved reference containes nan values.
    return {"logZ_ref_mean_is_nan": bool(np.isnan(ref.logZ_ref_mean)),
            "logZ_ref_std_is_nan": bool(np.isnan(ref.logZ_ref_std)),
            "f1_mean_ref_nan_count": int(np.isnan(ref.f1_mean_ref).sum()),
            "f1_std_ref_nan_count": int(np.isnan(ref.f1_std_ref).sum()),
            "f2_mean_ref_nan_count": int(np.isnan(ref.f2_mean_ref).sum()),
            "f2_std_ref_nan_count": int(np.isnan(ref.f2_std_ref).sum()),
            "final_ess_mean_is_nan": bool(np.isnan(ref.final_ess_mean)),
            "final_ess_std_is_nan": bool(np.isnan(ref.final_ess_std)),
            "acceptance_rate_mean_is_nan": bool(np.isnan(ref.acceptance_rate_mean)),
            "acceptance_rate_std_is_nan": bool(np.isnan(ref.acceptance_rate_std)),
            "n_iter_mean_is_nan": bool(np.isnan(ref.n_iter_mean)),
            "n_iter_std_is_nan": bool(np.isnan(ref.n_iter_std)),
            "runtime_sec_mean_is_nan": bool(np.isnan(ref.runtime_sec_mean)),
            "runtime_sec_std_is_nan": bool(np.isnan(ref.runtime_sec_std)),}


if __name__ == "__main__":
    checkpoint_path, save_final_path = _default_reference_paths()
    ref = build_reference_stats_chunked(dimension=16,
                                        num_particles_reference=256,
                                        num_reference_runs=20,
                                        target_name="gaussian_mixture",
                                        chunk_size=8,
                                        alpha=0.999,
                                        num_mcmc_steps=25,
                                        kernel_name="rwm",
                                        parallel=True,
                                        num_workers=4,
                                        checkpoint_path=checkpoint_path,
                                        save_final_path=save_final_path,
                                        verbose=True,)

    print(ref)