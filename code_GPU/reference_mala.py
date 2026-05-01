# reference_mala_gpu.py
# GPU-oriented reference implementation for tempered SMC + MALA only.
#
# Main design choice:
#   MALA step size epsilon is adapted at EVERY inner MCMC step using
#   Robbins--Monro, targeting alpha* ~= 0.574, but the whole k-step
#   MALA move is staged into one JAX program using jit + lax.scan.
#
# This intentionally excludes RWM. Keep RWM in the old CPU reference script.

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import time
import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# JAX device selection
# ---------------------------------------------------------------------
# Do not force CPU. Let JAX use GPU if the environment provides GPU jaxlib.
# Optional override: set FORCE_JAX_CPU=1 before importing this module.
if os.environ.get("FORCE_JAX_CPU", "0") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
else:
    os.environ.pop("JAX_PLATFORMS", None)

os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as smc_solver

from targets import make_target


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def debug_worker_backend() -> dict[str, Any]:
    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
    }


@dataclass
class ReferenceStats:
    target_name: str
    dimension: int
    num_reference_runs: int
    num_particles_reference: int
    logZ_ref_mean: float
    logZ_ref_std: float
    f1_mean_ref: np.ndarray
    f1_std_ref: np.ndarray
    f2_mean_ref: np.ndarray
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
    project_root = here.parent.parent
    base_dir = project_root / "data" / "results" / "reference"
    checkpoint_path = base_dir / "checkpoints" / "reference_mala_gpu_checkpoint.json"
    save_final_path = base_dir / "summary" / "reference_mala_gpu_stats.json"
    return checkpoint_path, save_final_path


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _robbins_monro_step_size_device(
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
    """On-device Robbins--Monro update on log epsilon.

    epsilon_{t+1} = exp(log epsilon_t + gamma_t (acceptance_t - alpha*))
    """
    dtype = step_size.dtype
    gamma_t = _robbins_monro_step_size_device(
        t,
        c=rm_c,
        t0=rm_t0,
        kappa=rm_kappa,
        dtype=dtype,
    )
    proposed = jnp.exp(
        jnp.log(step_size) + gamma_t * (acceptance_value - jnp.asarray(target_acceptance_rate, dtype=dtype))
    )
    return jnp.where(jnp.isfinite(acceptance_value), proposed, step_size)


def _compute_posterior_moments_from_particles(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


# ---------------------------------------------------------------------
# GPU-friendly MALA k-step move builder
# ---------------------------------------------------------------------
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
    """Return a JIT-compiled MALA k-step move for one SMC temperature.

    This is the key GPU optimization. The algorithmic behavior is the same as
    the original MALA loop:

      for mcmc_iter = 1,...,k:
          run one MALA step for all particles
          average the acceptance over particles
          update epsilon by Robbins--Monro

    The difference is that this loop is represented by `jax.lax.scan`, so the
    per-MCMC-step acceptance and step-size updates stay on the device.
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

            # BlackJAX MALA info exposes acceptance_rate. Keep it on device.
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


# ---------------------------------------------------------------------
# Tempered SMC + MALA
# ---------------------------------------------------------------------
def adaptive_loop_with_mala_adaptation(
    key,
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
    rm_kappa: float = 0.6,
    return_diagnostics: bool = True,
):
    state = blackjax.smc.tempered.init(initial_particles)
    work_dtype = initial_particles.dtype
    step_size = jnp.asarray(initial_step_size, dtype=work_dtype)

    tempering_path, logZ_path, ess_path = [], [], []
    acceptance_path, step_size_path = [], []
    logZ, n_iter = 0.0, 0

    vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)

    mala_move_k_steps = _make_mala_move_k_steps(
        log_prior_fn,
        log_likelihood_fn,
        num_particles=int(initial_particles.shape[0]),
        num_mcmc_steps=int(num_mcmc_steps),
        target_acceptance_rate=float(target_acceptance_rate),
        rm_c=float(rm_c),
        rm_t0=float(rm_t0),
        rm_kappa=float(rm_kappa),
    )

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        current_beta = jnp.asarray(state.tempering_param, dtype=work_dtype)
        one = jnp.asarray(1.0, dtype=work_dtype)
        zero = jnp.asarray(0.0, dtype=work_dtype)
        target_ess_typed = jnp.asarray(target_ess, dtype=work_dtype)
        max_delta = one - current_beta

        # Choose next temperature using current particles.
        delta_beta = smc_ess.ess_solver(
            vmapped_log_likelihood_fn,
            state.particles,
            target_ess_typed,
            max_delta,
            smc_solver.dichotomy,
        )
        delta_beta = jnp.clip(delta_beta, zero, max_delta)
        next_beta = current_beta + delta_beta

        # Incremental weights for moving from current_beta to next_beta.
        current_loglik = vmapped_log_likelihood_fn(state.particles)
        incremental_log_weights = delta_beta * current_loglik
        normalized_current_weights = jax.nn.softmax(incremental_log_weights)

        ess_value_jax = 1.0 / jnp.sum(normalized_current_weights ** 2)
        logZ_increment_jax = (
            jax.scipy.special.logsumexp(incremental_log_weights)
            - jnp.log(incremental_log_weights.shape[0])
        )

        # One unavoidable host sync per SMC iteration for loop control/diagnostics.
        # Crucially, there is no host sync inside the k-step MALA adaptation loop.
        ess_value = float(ess_value_jax)
        logZ += float(logZ_increment_jax)

        key, resample_key, move_key = jax.random.split(key, 3)
        ancestors = resampling.systematic(
            resample_key,
            normalized_current_weights,
            state.particles.shape[0],
        )
        resampled_particles = state.particles[ancestors]

        key_after_move, moved_particles, step_size, acc_path_jax, step_path_jax = mala_move_k_steps(
            move_key,
            resampled_particles,
            next_beta,
            step_size,
        )
        key = key_after_move

        # Make timing honest for each SMC iteration and avoid deferred work piling up.
        moved_particles.block_until_ready()

        if return_diagnostics:
            acceptance_path.extend(np.asarray(acc_path_jax, dtype=float).tolist())
            step_size_path.extend(np.asarray(step_path_jax, dtype=float).tolist())
        else:
            # Keep enough information to compute acceptance_rate_mean cheaply later.
            acceptance_path.append(float(jnp.mean(acc_path_jax)))
            step_size_path.append(float(step_size))

        num_particles = moved_particles.shape[0]
        uniform_weights = jnp.ones((num_particles,), dtype=work_dtype) / num_particles
        state = blackjax.smc.tempered.TemperedSMCState(
            particles=moved_particles,
            weights=uniform_weights,
            tempering_param=next_beta,
        )

        tempering_path.append(float(next_beta))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)
        n_iter += 1

    # Ensure final particles are ready before reporting runtime upstream.
    state.particles.block_until_ready()

    diagnostics = {
        "tempering_path": np.asarray(tempering_path, dtype=float),
        "logZ_path": np.asarray(logZ_path, dtype=float),
        "ess_path": np.asarray(ess_path, dtype=float),
        "acceptance_path": np.asarray(acceptance_path, dtype=float),
        "step_size_path": np.asarray(step_size_path, dtype=float),
        "final_logZ": float(logZ),
        "final_step_size": float(step_size),
    }

    return n_iter, state, diagnostics


# ---------------------------------------------------------------------
# Public single-run API
# ---------------------------------------------------------------------
def run_reference_sampler_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    kernel_name: str = "mala",
    mala_step_size: float = 0.02,
    mala_target_acceptance_rate: float = 0.574,
    mala_rm_c: float = 2.0,
    mala_rm_t0: float = 1.0,
    mala_rm_kappa: float = 0.6,
    return_diagnostics: bool = True,
) -> dict[str, Any]:
    if kernel_name.lower() != "mala":
        raise ValueError(
            "reference_mala_gpu.py only implements kernel_name='mala'. "
            "Use your old CPU reference script for RWM."
        )

    target = make_target(target_name, dimension, **(target_kwargs or {}))
    true_dimension = int(target.dimension)

    key = jax.random.PRNGKey(seed)
    key, init_key, run_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    initial_particles.block_until_ready()

    start = time.perf_counter()
    n_iter, smc_final_state, diagnostics = adaptive_loop_with_mala_adaptation(
        run_key,
        log_prior_fn=target.log_prior_fn,
        log_likelihood_fn=target.log_likelihood_fn,
        initial_particles=initial_particles,
        dimension=true_dimension,
        max_iterations=max_iterations,
        target_ess=alpha,
        num_mcmc_steps=num_mcmc_steps,
        initial_step_size=mala_step_size,
        target_acceptance_rate=mala_target_acceptance_rate,
        rm_c=mala_rm_c,
        rm_t0=mala_rm_t0,
        rm_kappa=mala_rm_kappa,
        return_diagnostics=return_diagnostics,
    )
    smc_final_state.particles.block_until_ready()
    runtime_sec = time.perf_counter() - start

    particles = np.asarray(smc_final_state.particles)
    f1, f2 = _compute_posterior_moments_from_particles(particles)

    f1_first_coord = float(f1[0])
    f2_first_coord = float(f2[0])
    if true_dimension > 1:
        f1_rest_mean = float(np.mean(f1[1:]))
        f2_rest_mean = float(np.mean(f2[1:]))
    else:
        f1_rest_mean = np.nan
        f2_rest_mean = np.nan

    ess_path = np.asarray(diagnostics["ess_path"], dtype=float)
    acceptance_path = np.asarray(diagnostics["acceptance_path"], dtype=float)
    final_ess = float(ess_path[-1]) if ess_path.size > 0 else np.nan
    acceptance_rate_mean = float(np.nanmean(acceptance_path)) if acceptance_path.size > 0 else np.nan

    out = {
        "target_name": target.name,
        "kernel_name": "mala",
        "dimension": true_dimension,
        "seed": int(seed),
        "logZ": float(diagnostics["final_logZ"]),
        "posterior_mean": f1,
        "posterior_second_moment": f2,
        "posterior_mean_first_coord": f1_first_coord,
        "posterior_second_moment_first_coord": f2_first_coord,
        "posterior_mean_rest_coords_mean": f1_rest_mean,
        "posterior_second_moment_rest_coords_mean": f2_rest_mean,
        "n_iter": int(n_iter),
        "runtime_sec": float(runtime_sec),
        "final_ess": final_ess,
        "acceptance_rate_mean": acceptance_rate_mean,
        "final_step_size": float(diagnostics["final_step_size"]),
        "particles": particles,
    }

    if return_diagnostics:
        out.update(
            {
                "tempering_path": np.asarray(diagnostics["tempering_path"], dtype=float),
                "logZ_path": np.asarray(diagnostics["logZ_path"], dtype=float),
                "ess_path": ess_path,
                "acceptance_path": acceptance_path,
                "step_size_path": np.asarray(diagnostics["step_size_path"], dtype=float),
            }
        )

    return out


# ---------------------------------------------------------------------
# Chunked multi-run API
# ---------------------------------------------------------------------
def _run_reference_worker(args: dict[str, Any]) -> dict[str, Any]:
    return run_reference_sampler_once(**args)


def _append_reference_output(
    out: dict[str, Any],
    completed_seeds,
    logZ_runs,
    f1_runs,
    f2_runs,
    final_ess_runs,
    acceptance_rate_runs,
    n_iter_runs,
    runtime_sec_runs,
) -> None:
    completed_seeds.append(out["seed"])
    logZ_runs.append(out["logZ"])
    f1_runs.append(out["posterior_mean"])
    f2_runs.append(out["posterior_second_moment"])
    final_ess_runs.append(out["final_ess"])
    acceptance_rate_runs.append(out["acceptance_rate_mean"])
    n_iter_runs.append(out["n_iter"])
    runtime_sec_runs.append(out["runtime_sec"])


def _sort_runs_by_seed(
    completed_seeds,
    logZ_runs,
    f1_runs,
    f2_runs,
    final_ess_runs,
    acceptance_rate_runs,
    n_iter_runs,
    runtime_sec_runs,
):
    if len(completed_seeds) == 0:
        return (
            completed_seeds,
            logZ_runs,
            f1_runs,
            f2_runs,
            final_ess_runs,
            acceptance_rate_runs,
            n_iter_runs,
            runtime_sec_runs,
        )

    order = np.argsort(np.asarray(completed_seeds))
    completed_seeds = [completed_seeds[i] for i in order]
    logZ_runs = [logZ_runs[i] for i in order]
    f1_runs = [f1_runs[i] for i in order]
    f2_runs = [f2_runs[i] for i in order]
    final_ess_runs = [final_ess_runs[i] for i in order]
    acceptance_rate_runs = [acceptance_rate_runs[i] for i in order]
    n_iter_runs = [n_iter_runs[i] for i in order]
    runtime_sec_runs = [runtime_sec_runs[i] for i in order]

    return (
        completed_seeds,
        logZ_runs,
        f1_runs,
        f2_runs,
        final_ess_runs,
        acceptance_rate_runs,
        n_iter_runs,
        runtime_sec_runs,
    )


def _assemble_reference_stats(
    target_name: str,
    dimension: int,
    num_particles_reference: int,
    num_reference_runs: int,
    logZ_runs,
    f1_runs,
    f2_runs,
    final_ess_runs,
    acceptance_rate_runs,
    n_iter_runs,
    runtime_sec_runs,
) -> ReferenceStats:
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

    return ReferenceStats(
        target_name=target_name,
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
        runtime_sec_std=runtime_sec_std,
    )


def save_reference_stats(ref: ReferenceStats, outpath: str | Path) -> None:
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
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ReferenceStats(
        target_name=payload["target_name"],
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
        runtime_sec_std=payload.get("runtime_sec_std", np.nan),
    )


def save_chunk_checkpoint(
    checkpoint_path: str | Path,
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
    runtime_sec_runs,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target_name": str(target_name),
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
        "runtime_sec_runs": np.asarray(runtime_sec_runs, dtype=float).tolist(),
    }

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


def build_reference_stats_chunked(
    dimension: int,
    num_particles_reference: int,
    num_reference_runs: int,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    chunk_size: int = 10,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    kernel_name: str = "mala",
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
    if kernel_name.lower() != "mala":
        raise ValueError("reference_mala_gpu.py only implements kernel_name='mala'.")

    if parallel and jax.default_backend() != "cpu":
        warnings.warn(
            "parallel=True uses Python multiprocessing. On a single GPU this usually creates "
            "several independent JAX processes competing for the same GPU memory. For GPU runs, "
            "prefer parallel=False and use particle-level parallelism through JAX vmap/JIT.",
            RuntimeWarning,
        )

    start_time = time.perf_counter()

    target_for_metadata = make_target(target_name, dimension, **(target_kwargs or {}))
    true_dimension = int(target_for_metadata.dimension)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_chunk_checkpoint(checkpoint_path)
        if (
            ckpt["target_name"] != target_name
            or ckpt["dimension"] != true_dimension
            or ckpt["num_particles_reference"] != num_particles_reference
            or ckpt["num_reference_runs"] != num_reference_runs
        ):
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
        completed_seeds, logZ_runs, f1_runs, f2_runs = [], [], [], []
        final_ess_runs, acceptance_rate_runs, n_iter_runs, runtime_sec_runs = [], [], [], []

    num_completed = len(completed_seeds)
    total_runs = num_reference_runs
    completed_seed_set = set(completed_seeds)
    remaining_seeds = [seed for seed in range(num_reference_runs) if seed not in completed_seed_set]

    for chunk_start in range(0, len(remaining_seeds), chunk_size):
        chunk = remaining_seeds[chunk_start:chunk_start + chunk_size]
        if verbose:
            print(f"Running MALA reference chunk seeds {chunk[0]}:{chunk[-1] + 1}")

        job_args_list = []
        for seed in chunk:
            job_args_list.append(
                {
                    "target_name": target_name,
                    "dimension": true_dimension,
                    "num_particles": num_particles_reference,
                    "target_kwargs": target_kwargs,
                    "seed": seed,
                    "max_iterations": max_iterations,
                    "alpha": alpha,
                    "num_mcmc_steps": num_mcmc_steps,
                    "kernel_name": "mala",
                    "mala_step_size": mala_step_size,
                    "mala_target_acceptance_rate": mala_target_acceptance_rate,
                    "mala_rm_c": mala_rm_c,
                    "mala_rm_t0": mala_rm_t0,
                    "mala_rm_kappa": mala_rm_kappa,
                    "return_diagnostics": False,
                }
            )

        if parallel:
            workers = num_workers or min(len(chunk), os.cpu_count() or 1)
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = [executor.submit(_run_reference_worker, job_args) for job_args in job_args_list]
                for future in as_completed(futures):
                    out = future.result()
                    _append_reference_output(
                        out,
                        completed_seeds,
                        logZ_runs,
                        f1_runs,
                        f2_runs,
                        final_ess_runs,
                        acceptance_rate_runs,
                        n_iter_runs,
                        runtime_sec_runs,
                    )
                    num_completed += 1
                    if verbose:
                        elapsed = time.perf_counter() - start_time
                        avg_time = elapsed / max(num_completed, 1)
                        remaining = avg_time * (total_runs - num_completed)
                        print(
                            f"Run {num_completed}/{total_runs} completed "
                            f"(seed={out['seed']}) | elapsed={elapsed:.1f}s | ETA={remaining:.1f}s"
                        )
        else:
            for job_args in job_args_list:
                out = _run_reference_worker(job_args)
                _append_reference_output(
                    out,
                    completed_seeds,
                    logZ_runs,
                    f1_runs,
                    f2_runs,
                    final_ess_runs,
                    acceptance_rate_runs,
                    n_iter_runs,
                    runtime_sec_runs,
                )
                num_completed += 1
                if verbose:
                    elapsed = time.perf_counter() - start_time
                    avg_time = elapsed / max(num_completed, 1)
                    remaining = avg_time * (total_runs - num_completed)
                    print(
                        f"Run {num_completed}/{total_runs} completed "
                        f"(seed={out['seed']}) | elapsed={elapsed:.1f}s | ETA={remaining:.1f}s"
                    )

        (
            completed_seeds,
            logZ_runs,
            f1_runs,
            f2_runs,
            final_ess_runs,
            acceptance_rate_runs,
            n_iter_runs,
            runtime_sec_runs,
        ) = _sort_runs_by_seed(
            completed_seeds,
            logZ_runs,
            f1_runs,
            f2_runs,
            final_ess_runs,
            acceptance_rate_runs,
            n_iter_runs,
            runtime_sec_runs,
        )

        if checkpoint_path is not None:
            save_chunk_checkpoint(
                checkpoint_path,
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
                runtime_sec_runs=runtime_sec_runs,
            )
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    ref = _assemble_reference_stats(
        target_name=target_name,
        dimension=true_dimension,
        num_particles_reference=num_particles_reference,
        num_reference_runs=num_reference_runs,
        logZ_runs=logZ_runs,
        f1_runs=f1_runs,
        f2_runs=f2_runs,
        final_ess_runs=final_ess_runs,
        acceptance_rate_runs=acceptance_rate_runs,
        n_iter_runs=n_iter_runs,
        runtime_sec_runs=runtime_sec_runs,
    )

    if save_final_path is not None:
        save_reference_stats(ref, save_final_path)
        if verbose:
            print(f"Final reference stats saved to {save_final_path}")

    return ref


def nan_report_reference_stats(ref: ReferenceStats) -> dict[str, Any]:
    return {
        "logZ_ref_mean_is_nan": bool(np.isnan(ref.logZ_ref_mean)),
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
        "runtime_sec_std_is_nan": bool(np.isnan(ref.runtime_sec_std)),
    }


if __name__ == "__main__":
    checkpoint_path, save_final_path = _default_reference_paths()
    ref = build_reference_stats_chunked(
        dimension=16,
        num_particles_reference=256,
        num_reference_runs=20,
        target_name="gaussian_mixture",
        chunk_size=8,
        alpha=0.999,
        num_mcmc_steps=25,
        kernel_name="mala",
        parallel=False,
        checkpoint_path=checkpoint_path,
        save_final_path=save_final_path,
        verbose=True,
    )
    print(ref)
