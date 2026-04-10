# reference.py
#paralelised version with Robbins-Monro diminsihing adaptation
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

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from targets import make_gaussian_mixture_target

def debug_worker_backend() -> dict[str, Any]:
    import jax
    return { "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],}


@dataclass
class ReferenceStats:  # container for the final refrennce results
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

def _default_reference_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    project_root = here.parent.parent
    base_dir = project_root / "data" / "results" / "reference"
    checkpoint_path = base_dir / "checkpoints" / "reference_checkpoint.json"
    save_final_path = base_dir / "summary" / "reference_stats.json"
    return checkpoint_path, save_final_path


def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    # returns scalar mean acceptance rate
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted, dtype=jnp.float32))
    return jnp.nan


def _empirical_covariance(particles: jnp.ndarray, ridge: float = 1e-6, diagonal_only: bool = False, ) -> jnp.ndarray:
    # Empirical diagonal covariance of particles with small ridge regularization.
    # ridge is a small constant for matrix stabilisation
    x = particles
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
    # cholesky decomposition of covariance matrix multiplied by the scale factor.
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(t: int, c: float = 1.0, t0: float = 10.0, kappa: float = 0.6) -> float:
    # diminishing Robbins-Monro schedule gamma_t = c / (t + t0)^kappa
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(rw_scale: jnp.ndarray,  acceptance_value: float, target_acceptance_rate: float, t: int,
                                    rm_c: float = 1.0, rm_t0: float = 10.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    # Robbins-Monro adaptation on the log scale
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


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
                                        rm_t0: float = 10.0,
                                        rm_kappa: float = 0.6,):
    # adaptively tempered SMC, with proposal covariance adapted via Robbins-Monro diminishing adaptation.

    state = blackjax.smc.tempered.init(initial_particles)
    rw_scale = jnp.asarray(initial_rw_scale, dtype=jnp.float32)
    proposal_cov = _empirical_covariance(initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)
    tempering_path, logZ_path, ess_path, acceptance_path, rw_scale_path = [], [], [], [], []
    logZ, n_iter = 0.0, 0
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def rmh_step_fn(rng_key, state, logdensity_fn, rw_scale, proposal_cov):
        transition_generator = blackjax.mcmc.random_walk.normal(
            _proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel(rng_key, state, logdensity_fn, transition_generator=transition_generator,)

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        mcmc_parameters = extend_params( {"rw_scale": jnp.asarray(rw_scale), "proposal_cov": proposal_cov})

        kernel = blackjax.adaptive_tempered_smc(logprior_fn=log_prior_fn,
                                                loglikelihood_fn=log_likelihood_fn,
                                                mcmc_step_fn=rmh_step_fn,
                                                mcmc_init_fn=blackjax.rmh.init,
                                                mcmc_parameters=mcmc_parameters,
                                                resampling_fn=resampling.systematic,
                                                target_ess=target_ess,
                                                num_mcmc_steps=num_mcmc_steps,)

        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        logZ += float(info.log_likelihood_increment)
        ess_value = float(1.0 / jnp.sum(state.weights ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        rw_scale_path.append(float(rw_scale))

        if not np.isnan(acceptance_value):
            rw_scale = _update_rw_scale_robbins_monro(  rw_scale=rw_scale,
                                                        acceptance_value=acceptance_value,
                                                        target_acceptance_rate=target_acceptance_rate,
                                                        t=n_iter + 1,
                                                        rm_c=rm_c,
                                                        rm_t0=rm_t0,
                                                        rm_kappa=rm_kappa,)

        proposal_cov = _empirical_covariance(state.particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)

        n_iter += 1

    diagnostics = { "tempering_path": np.asarray(tempering_path, dtype=float),
                    "logZ_path": np.asarray(logZ_path, dtype=float),
                    "ess_path": np.asarray(ess_path, dtype=float),
                    "acceptance_path": np.asarray(acceptance_path, dtype=float),
                    "rw_scale_path": np.asarray(rw_scale_path, dtype=float),
                    "final_logZ": float(logZ),
                    "final_rw_scale": float(rw_scale),}

    return n_iter, state, diagnostics


def compute_posterior_moments_from_particles(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # first and second moment f1 = theta, f2 = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def run_reference_sampler_once( dimension: int,
                                num_particles: int,
                                seed: int,
                                max_iterations: int = 10_000,
                                alpha: float = 0.999,
                                num_mcmc_steps: int = 25,
                                rw_step_size: float = 0.5,
                                target_acceptance_rate: float = 0.234,
                                covariance_ridge: float = 1e-6,
                                diagonal_only_covariance: bool = False,
                                rm_c: float = 1.0,
                                rm_t0: float = 10.0,
                                rm_kappa: float = 0.6,
                                return_diagnostics: bool = True,) -> dict[str, Any]:
    # run refrence SMC for one seed

    target = make_gaussian_mixture_target(dimension)
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, run_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)

    start = time.perf_counter()
    n_iter, smc_final_state, diagnostics = adaptive_loop_with_rwm_adaptation(    run_key,
                                                                                log_prior_fn=log_prior_fn,
                                                                                log_likelihood_fn=log_likelihood_fn,
                                                                                initial_particles=initial_particles,
                                                                                dimension=dimension,
                                                                                max_iterations=max_iterations,
                                                                                target_ess=alpha,
                                                                                num_mcmc_steps=num_mcmc_steps,
                                                                                initial_rw_scale=rw_step_size,
                                                                                target_acceptance_rate=target_acceptance_rate,
                                                                                covariance_ridge=covariance_ridge,
                                                                                diagonal_only_covariance=diagonal_only_covariance,
                                                                                rm_c=rm_c,
                                                                                rm_t0=rm_t0,
                                                                                rm_kappa=rm_kappa, )
    runtime_sec = time.perf_counter() - start

    particles = np.array(smc_final_state.particles)
    f1, f2 = compute_posterior_moments_from_particles(particles)
    ess_path = np.asarray(diagnostics["ess_path"], dtype=float)
    acceptance_path = np.asarray(diagnostics["acceptance_path"], dtype=float)
    final_ess = float(ess_path[-1]) if ess_path.size > 0 else np.nan
    acceptance_rate_mean = float(np.nanmean(acceptance_path)) if acceptance_path.size > 0 else np.nan

    out = { "seed": int(seed),
            "logZ": float(diagnostics["final_logZ"]),
            "posterior_mean": f1,
            "posterior_second_moment": f2,
            "n_iter": int(n_iter),
            "runtime_sec": float(runtime_sec),
            "final_rw_scale": float(diagnostics["final_rw_scale"]),
            "final_ess": final_ess,
            "acceptance_rate_mean": acceptance_rate_mean,}

    if return_diagnostics:
        out.update({"tempering_path": np.asarray(diagnostics["tempering_path"], dtype=float),
                    "logZ_path": np.asarray(diagnostics["logZ_path"], dtype=float),
                    "ess_path": ess_path,
                    "acceptance_path": acceptance_path,
                    "rw_scale_path": np.asarray(diagnostics["rw_scale_path"], dtype=float),})

    return out


def _run_reference_worker(args: dict[str, Any]) -> dict[str, Any]:
   # Top-level worker wrapper for multiprocessing, where Each worker runs exactly one seed.
    return run_reference_sampler_once(**args)


def _append_reference_output(out: dict[str, Any], completed_seeds, logZ_runs, f1_runs, f2_runs,
                            final_ess_runs, acceptance_rate_runs, n_iter_runs,) -> None:
    #takes the result of one run appending it to the aggregation lists

    completed_seeds.append(out["seed"])
    logZ_runs.append(out["logZ"])
    f1_runs.append(out["posterior_mean"])
    f2_runs.append(out["posterior_second_moment"])
    final_ess_runs.append(out["final_ess"])
    acceptance_rate_runs.append(out["acceptance_rate_mean"])
    n_iter_runs.append(out["n_iter"])


def _sort_runs_by_seed( completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs,):
    # for easyer reproducibility the seeds are ordered
    if len(completed_seeds) == 0:
        return ( completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs, )

    order = np.argsort(np.asarray(completed_seeds))
    completed_seeds = [completed_seeds[i] for i in order]
    logZ_runs = [logZ_runs[i] for i in order]
    f1_runs = [f1_runs[i] for i in order]
    f2_runs = [f2_runs[i] for i in order]
    final_ess_runs = [final_ess_runs[i] for i in order]
    acceptance_rate_runs = [acceptance_rate_runs[i] for i in order]
    n_iter_runs = [n_iter_runs[i] for i in order]

    return ( completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs,)


def _assemble_reference_stats(  dimension: int,
                                num_particles_reference: int,
                                num_reference_runs: int,
                                logZ_runs,
                                f1_runs,
                                f2_runs,
                                final_ess_runs,
                                acceptance_rate_runs,
                                n_iter_runs, ) -> ReferenceStats:
    
    # takes the outputs of multiple runns and combines them into RefrenceSatas object
    logZ_runs = np.asarray(logZ_runs, dtype=float)
    f1_runs = np.stack(f1_runs, axis=0)
    f2_runs = np.stack(f2_runs, axis=0)
    final_ess_runs = np.asarray(final_ess_runs, dtype=float)
    acceptance_rate_runs = np.asarray(acceptance_rate_runs, dtype=float)
    n_iter_runs = np.asarray(n_iter_runs, dtype=float)

    logZ_ref_std = float(np.nanstd(logZ_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    f1_std_ref = np.std(f1_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f1_runs[0])
    f2_std_ref = np.std(f2_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f2_runs[0])
    final_ess_std = float(np.nanstd(final_ess_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    acceptance_rate_std = float(np.nanstd(acceptance_rate_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    n_iter_std = float(np.nanstd(n_iter_runs, ddof=1)) if num_reference_runs > 1 else 0.0

    return ReferenceStats(  target_name="gaussian_mixture",
                            dimension=dimension,
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
                            n_iter_std=n_iter_std,)


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
                            dimension=payload["dimension"],
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
                            n_iter_std=payload["n_iter_std"],)


def save_chunk_checkpoint(  checkpoint_path: str | Path,
                            *,
                            dimension: int,
                            num_particles_reference: int,
                            num_reference_runs: int,
                            completed_seeds,
                            logZ_runs,
                            f1_runs,
                            f2_runs,
                            final_ess_runs,
                            acceptance_rate_runs,
                            n_iter_runs,) -> None:
    # this saces the progres of the chunked execution
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = { "dimension": int(dimension),
                "num_particles_reference": int(num_particles_reference),
                "num_reference_runs": int(num_reference_runs),
                "completed_seeds": list(map(int, completed_seeds)),
                "logZ_runs": np.asarray(logZ_runs, dtype=float).tolist(),
                "f1_runs": np.asarray(f1_runs, dtype=float).tolist(),
                "f2_runs": np.asarray(f2_runs, dtype=float).tolist(),
                "final_ess_runs": np.asarray(final_ess_runs, dtype=float).tolist(),
                "acceptance_rate_runs": np.asarray(acceptance_rate_runs, dtype=float).tolist(),
                "n_iter_runs": np.asarray(n_iter_runs, dtype=float).tolist(),}

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
    return payload

def build_reference_stats_chunked(  dimension: int,
                                    num_particles_reference: int,
                                    num_reference_runs: int,
                                    chunk_size: int = 10,
                                    max_iterations: int = 10_000,
                                    alpha: float = 0.999,
                                    num_mcmc_steps: int = 25,
                                    rw_step_size: float = 0.5,
                                    target_acceptance_rate: float = 0.234,
                                    covariance_ridge: float = 1e-6,
                                    diagonal_only_covariance: bool = False,
                                    rm_c: float = 1.0,
                                    rm_t0: float = 10.0,
                                    rm_kappa: float = 0.6,
                                    verbose: bool = True,
                                    checkpoint_path: str | Path | None = None,
                                    save_final_path: str | Path | None = None,
                                    parallel: bool = False,
                                    num_workers: int | None = None,) -> ReferenceStats:
    # runs multiple reference runs in chunks

    start_time = time.perf_counter()

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_chunk_checkpoint(checkpoint_path)

        if (
            ckpt["dimension"] != dimension
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

        if verbose:
            print(f"Loaded checkpoint with {len(completed_seeds)} completed runs.")
    else:
        completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs = [], [], [], [], [], [], []

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
            job_args_list.append({
                "dimension": dimension,
                "num_particles": num_particles_reference,
                "seed": seed,
                "max_iterations": max_iterations,
                "alpha": alpha,
                "num_mcmc_steps": num_mcmc_steps,
                "rw_step_size": rw_step_size,
                "target_acceptance_rate": target_acceptance_rate,
                "covariance_ridge": covariance_ridge,
                "diagonal_only_covariance": diagonal_only_covariance,
                "rm_c": rm_c,
                "rm_t0": rm_t0,
                "rm_kappa": rm_kappa,
                "return_diagnostics": False,})

        if parallel:
            workers = num_workers or min(len(chunk), os.cpu_count() or 1)
            ctx = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = [executor.submit(_run_reference_worker, job_args) for job_args in job_args_list]

                for future in as_completed(futures):
                    out = future.result()
                    _append_reference_output(out, completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs,)
                    num_completed += 1
                    if verbose:
                        elapsed = time.perf_counter() - start_time
                        avg_time = elapsed / max(num_completed, 1)
                        remaining = avg_time * (total_runs - num_completed)
                        print(f"Run {num_completed}/{total_runs} completed "
                              f"(seed={out['seed']}) | "
                              f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s")
        else:
            for job_args in job_args_list:
                out = _run_reference_worker(job_args)

                _append_reference_output(out, completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs, )

                num_completed += 1
                if verbose:
                    elapsed = time.perf_counter() - start_time
                    avg_time = elapsed / max(num_completed, 1)
                    remaining = avg_time * (total_runs - num_completed)
                    print(f"Run {num_completed}/{total_runs} completed "
                          f"(seed={out['seed']}) | "
                          f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s")

        (completed_seeds, logZ_runs, f1_runs, f2_runs,
         final_ess_runs, acceptance_rate_runs, n_iter_runs) = _sort_runs_by_seed( completed_seeds, logZ_runs, f1_runs, f2_runs,
                                                                                final_ess_runs, acceptance_rate_runs, n_iter_runs,)

        if checkpoint_path is not None:
            save_chunk_checkpoint(checkpoint_path,
                                  dimension=dimension,
                                  num_particles_reference=num_particles_reference,
                                  num_reference_runs=num_reference_runs,
                                  completed_seeds=completed_seeds,
                                  logZ_runs=logZ_runs,
                                  f1_runs=f1_runs,
                                  f2_runs=f2_runs,
                                  final_ess_runs=final_ess_runs,
                                  acceptance_rate_runs=acceptance_rate_runs,
                                  n_iter_runs=n_iter_runs,)
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    ref = _assemble_reference_stats(dimension=dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,)

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
            "n_iter_std_is_nan": bool(np.isnan(ref.n_iter_std)),}


if __name__ == "__main__":
    checkpoint_path, save_final_path = _default_reference_paths()
    ref = build_reference_stats_chunked(dimension=16,
                                        num_particles_reference=256,
                                        num_reference_runs=20,
                                        chunk_size=8,
                                        alpha=0.999,
                                        num_mcmc_steps=25,
                                        parallel=True,
                                        num_workers=4,
                                        checkpoint_path=checkpoint_path,
                                        save_final_path=save_final_path,
                                        verbose=True,)
    print(ref)

"""
# version with Robbins-Monroconstant adaptation but not a diminishing one, this one is prebuild in BackJAX
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import time

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate

from targets import make_gaussian_mixture_target

@dataclass
class ReferenceStats: #container for the final refrennce results
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


def _safe_mean_acceptance(update_info) -> jnp.ndarray: 
    # returns scalar mean acceptance rate
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted, dtype=jnp.float32))
    return jnp.nan


def _empirical_covariance(particles: jnp.ndarray, ridge: float = 1e-6, diagonal_only: bool = False,) -> jnp.ndarray:
    # Empirical diagonal covariance of particles with small ridge regularization.
    # ridge is a small constant for matrix stabilisation
    x = particles
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
    #cholesky decomposition of covariance matrix multiplied by the scale factor. 
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


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
                                        diagonal_only_covariance: bool = False, ):
    # adaptively tempered SMC, with proposal covariance adapted via Robbins-Monro diminishing adaptation. 

    state = blackjax.smc.tempered.init(initial_particles)
    rw_scale = jnp.asarray(initial_rw_scale, dtype=jnp.float32)
    proposal_cov = _empirical_covariance(initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)

    tempering_path, logZ_path, ess_path,acceptance_path,rw_scale_path = [], [], [], [], []
    logZ, n_iter= 0.0, 0
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def rmh_step_fn(rng_key, state, logdensity_fn, rw_scale, proposal_cov):
        transition_generator = blackjax.mcmc.random_walk.normal(_proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel(rng_key, state, logdensity_fn, transition_generator=transition_generator,)

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        mcmc_parameters = extend_params({"rw_scale": jnp.asarray(rw_scale), "proposal_cov": proposal_cov, })

        kernel = blackjax.adaptive_tempered_smc(logprior_fn=log_prior_fn,
                                                loglikelihood_fn=log_likelihood_fn,
                                                mcmc_step_fn=rmh_step_fn,
                                                mcmc_init_fn=blackjax.rmh.init,
                                                mcmc_parameters=mcmc_parameters,
                                                resampling_fn=resampling.systematic,
                                                target_ess=target_ess,
                                                num_mcmc_steps=num_mcmc_steps,)

        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        logZ += float(info.log_likelihood_increment)
        ess_value = float(1.0 / jnp.sum(state.weights ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        rw_scale_path.append(float(rw_scale))
        if not np.isnan(acceptance_value):
            rw_scale = update_scale_from_acceptance_rate(scales=jnp.asarray([rw_scale]), 
                                                         acceptance_rates=jnp.asarray([acceptance_value]), 
                                                         target_acceptance_rate=target_acceptance_rate,)[0]

        proposal_cov = _empirical_covariance(state.particles,
                                            ridge=covariance_ridge,
                                            diagonal_only=diagonal_only_covariance,)

        n_iter += 1

    diagnostics = { "tempering_path": np.asarray(tempering_path, dtype=float),
                    "logZ_path": np.asarray(logZ_path, dtype=float),
                    "ess_path": np.asarray(ess_path, dtype=float),
                    "acceptance_path": np.asarray(acceptance_path, dtype=float),
                    "rw_scale_path": np.asarray(rw_scale_path, dtype=float),
                    "final_logZ": float(logZ),
                    "final_rw_scale": float(rw_scale),}

    return n_iter, state, diagnostics


def compute_posterior_moments_from_particles(particles: np.ndarray, ) -> tuple[np.ndarray, np.ndarray]:
    # first and second moment f1 = theta, f2 = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def run_reference_sampler_once( dimension: int,
                                num_particles: int,
                                seed: int,
                                max_iterations: int = 10_000,
                                alpha: float = 0.999,
                                num_mcmc_steps: int = 25,
                                rw_step_size: float = 0.5,
                                target_acceptance_rate: float = 0.234,
                                covariance_ridge: float = 1e-6,
                                diagonal_only_covariance: bool = False,) -> dict[str, Any]:
    #run refrence SMC for one seed
    #return: 
    #logZ, posterior mean, posterior second moment, number of iterations, runtime, tempering path, logZ path,
    #ess path, acceptance path, rw scale path, final rw scale, final ess, acceptance rate mean.
    
    target = make_gaussian_mixture_target(dimension)
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, run_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    start = time.perf_counter()
    n_iter, smc_final_state, diagnostics = adaptive_loop_with_rwm_adaptation(run_key,
                                                                            log_prior_fn=log_prior_fn,
                                                                            log_likelihood_fn=log_likelihood_fn,
                                                                            initial_particles=initial_particles,
                                                                            dimension=dimension,
                                                                            max_iterations=max_iterations,
                                                                            target_ess=alpha,
                                                                            num_mcmc_steps=num_mcmc_steps,
                                                                            initial_rw_scale=rw_step_size,
                                                                            target_acceptance_rate=target_acceptance_rate,
                                                                            covariance_ridge=covariance_ridge,
                                                                            diagonal_only_covariance=diagonal_only_covariance,)
    runtime_sec = time.perf_counter() - start
    particles = np.array(smc_final_state.particles)
    f1, f2 = compute_posterior_moments_from_particles(particles)
    ess_path = np.asarray(diagnostics["ess_path"], dtype=float)
    acceptance_path = np.asarray(diagnostics["acceptance_path"], dtype=float)
    final_ess = float(ess_path[-1]) if ess_path.size > 0 else np.nan
    acceptance_rate_mean = float(np.nanmean(acceptance_path)) if acceptance_path.size > 0 else np.nan

    return {"seed": int(seed),
            "logZ": float(diagnostics["final_logZ"]),
            "posterior_mean": f1,
            "posterior_second_moment": f2,
            "n_iter": int(n_iter),
            "runtime_sec": float(runtime_sec),
            "tempering_path": np.asarray(diagnostics["tempering_path"], dtype=float),
            "logZ_path": np.asarray(diagnostics["logZ_path"], dtype=float),
            "ess_path": ess_path,
            "acceptance_path": acceptance_path,
            "rw_scale_path": np.asarray(diagnostics["rw_scale_path"], dtype=float),
            "final_rw_scale": float(diagnostics["final_rw_scale"]),
            "final_ess": final_ess,
            "acceptance_rate_mean": acceptance_rate_mean,}


def _assemble_reference_stats(dimension: int, num_particles_reference: int, num_reference_runs: int, logZ_runs, f1_runs, f2_runs, 
                              final_ess_runs, acceptance_rate_runs, n_iter_runs, ) -> ReferenceStats:
    # takes the outputs of multiple runns and combines them into RefrenceSatas object
    logZ_runs = np.asarray(logZ_runs, dtype=float)
    f1_runs = np.stack(f1_runs, axis=0)
    f2_runs = np.stack(f2_runs, axis=0)
    final_ess_runs = np.asarray(final_ess_runs, dtype=float)
    acceptance_rate_runs = np.asarray(acceptance_rate_runs, dtype=float)
    n_iter_runs = np.asarray(n_iter_runs, dtype=float)
    logZ_ref_std = float(np.nanstd(logZ_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    f1_std_ref = np.std(f1_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f1_runs[0])
    f2_std_ref = np.std(f2_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f2_runs[0])
    final_ess_std = float(np.nanstd(final_ess_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    acceptance_rate_std = float(np.nanstd(acceptance_rate_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    n_iter_std = float(np.nanstd(n_iter_runs, ddof=1)) if num_reference_runs > 1 else 0.0

    return ReferenceStats(  target_name="gaussian_mixture",
                            dimension=dimension,
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
                            n_iter_std=n_iter_std,)


def save_reference_stats(ref: ReferenceStats, outpath: str | Path) -> None:
    #saves ReferenceStats object into JSON file. 
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
                            dimension=payload["dimension"],
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
                            n_iter_std=payload["n_iter_std"],)


def save_chunk_checkpoint(checkpoint_path: str | Path, *, dimension: int, num_particles_reference: int,
                          num_reference_runs: int, completed_seeds, logZ_runs, f1_runs,f2_runs, final_ess_runs, 
                          acceptance_rate_runs, n_iter_runs,) -> None:
    # this saces the progres of the chunked execution
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = { "dimension": int(dimension),
                "num_particles_reference": int(num_particles_reference),
                "num_reference_runs": int(num_reference_runs),
                "completed_seeds": list(map(int, completed_seeds)),
                "logZ_runs": np.asarray(logZ_runs, dtype=float).tolist(),
                "f1_runs": np.asarray(f1_runs, dtype=float).tolist(),
                "f2_runs": np.asarray(f2_runs, dtype=float).tolist(),
                "final_ess_runs": np.asarray(final_ess_runs, dtype=float).tolist(),
                "acceptance_rate_runs": np.asarray(acceptance_rate_runs, dtype=float).tolist(),
                "n_iter_runs": np.asarray(n_iter_runs, dtype=float).tolist(),}

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
    return payload


def build_reference_stats_chunked(  dimension: int,
                                    num_particles_reference: int,
                                    num_reference_runs: int,
                                    chunk_size: int = 10,
                                    max_iterations: int = 10_000,
                                    alpha: float = 0.999,
                                    num_mcmc_steps: int = 25,
                                    rw_step_size: float = 0.5,
                                    target_acceptance_rate: float = 0.234,
                                    covariance_ridge: float = 1e-6,
                                    diagonal_only_covariance: bool = False,
                                    verbose: bool = True,
                                    checkpoint_path: str | Path | None = None,
                                    save_final_path: str | Path | None = None, ) -> ReferenceStats:
    # runs multiple reference runs in chunks
    
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_chunk_checkpoint(checkpoint_path)

        if (ckpt["dimension"] != dimension
            or ckpt["num_particles_reference"] != num_particles_reference
            or ckpt["num_reference_runs"] != num_reference_runs ):
            raise ValueError("Checkpoint metadata does not match current run settings.")

        completed_seeds = list(ckpt["completed_seeds"])
        logZ_runs = list(ckpt["logZ_runs"])
        f1_runs = list(ckpt["f1_runs"])
        f2_runs = list(ckpt["f2_runs"])
        final_ess_runs = list(ckpt["final_ess_runs"])
        acceptance_rate_runs = list(ckpt["acceptance_rate_runs"])
        n_iter_runs = list(ckpt["n_iter_runs"])

        if verbose:
            print(f"Loaded checkpoint with {len(completed_seeds)} completed runs.")
    else:
        completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs = [], [], [], [], [], [], []
    remaining_seeds = [seed for seed in range(num_reference_runs) if seed not in set(completed_seeds)]
    for chunk_start in range(0, len(remaining_seeds), chunk_size):
        chunk = remaining_seeds[chunk_start:chunk_start + chunk_size]
        if verbose:
            print(f"Running reference chunk seeds {chunk[0]}:{chunk[-1] + 1}")
        for seed in chunk:
            out = run_reference_sampler_once(dimension=dimension,
                                            num_particles=num_particles_reference,
                                            seed=seed,
                                            max_iterations=max_iterations,
                                            alpha=alpha,
                                            num_mcmc_steps=num_mcmc_steps,
                                            rw_step_size=rw_step_size,
                                            target_acceptance_rate=target_acceptance_rate,
                                            covariance_ridge=covariance_ridge,
                                            diagonal_only_covariance=diagonal_only_covariance,)

            completed_seeds.append(seed)
            logZ_runs.append(out["logZ"])
            f1_runs.append(out["posterior_mean"])
            f2_runs.append(out["posterior_second_moment"])
            final_ess_runs.append(out["final_ess"])
            acceptance_rate_runs.append(out["acceptance_rate_mean"])
            n_iter_runs.append(out["n_iter"])

        if checkpoint_path is not None:
            save_chunk_checkpoint(  checkpoint_path,
                                    dimension=dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    completed_seeds=completed_seeds,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,)
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    ref = _assemble_reference_stats(dimension=dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,)

    if save_final_path is not None:
        save_reference_stats(ref, save_final_path)
        if verbose:
            print(f"Final reference stats saved to {save_final_path}")

    return ref


def nan_report_reference_stats(ref: ReferenceStats) -> dict[str, Any]:
    # alnalyses if the saved reference containes nan values 
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
            "n_iter_std_is_nan": bool(np.isnan(ref.n_iter_std)),}
"""

#############################

""" 
Description for the version below, so the version with Robbins-Monro diminsihing adaptation that is not paralelised
How reference py works: 
It does three things:
one run
run_reference_sampler_once(...) runs one adaptive tempered SMC chain for one seed
many runs
build_reference_stats_chunked(...) repeats that for many seeds, in chunks
aggregation + saving
after all runs, it computes means/stds and writes one final JSON file
"""

"""
#version with Robbins-Monro diminsihing adaptation 
# not paralelised

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import time

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from targets import make_gaussian_mixture_target

@dataclass
class ReferenceStats: #container for the final refrennce results
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


def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    # returns scalar mean acceptance rate
    if update_info is None:
        return jnp.nan
    acc = getattr(update_info, "acceptance_rate", None)
    if acc is not None:
        return jnp.mean(jnp.asarray(acc, dtype=jnp.float32))
    is_accepted = getattr(update_info, "is_accepted", None)
    if is_accepted is not None:
        return jnp.mean(jnp.asarray(is_accepted, dtype=jnp.float32))
    return jnp.nan


def _empirical_covariance(particles: jnp.ndarray, ridge: float = 1e-6, diagonal_only: bool = False,) -> jnp.ndarray:
    # Empirical diagonal covariance of particles with small ridge regularization.
    # ridge is a small constant for matrix stabilisation
    x = particles
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
    #cholesky decomposition of covariance matrix multiplied by the scale factor.
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(t: int, c: float = 1.0, t0: float = 10.0, kappa: float = 0.6) -> float:
    # diminishing Robbins-Monro schedule gamma_t = c / (t + t0)^kappa
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(
    rw_scale: jnp.ndarray,
    acceptance_value: float,
    target_acceptance_rate: float,
    t: int,
    rm_c: float = 1.0,
    rm_t0: float = 10.0,
    rm_kappa: float = 0.6,
) -> jnp.ndarray:
    # Robbins-Monro adaptation on the log scale
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


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
                                        rm_t0: float = 10.0,
                                        rm_kappa: float = 0.6, ):
    # adaptively tempered SMC, with proposal covariance adapted via Robbins-Monro diminishing adaptation.

    state = blackjax.smc.tempered.init(initial_particles)
    rw_scale = jnp.asarray(initial_rw_scale, dtype=jnp.float32)
    proposal_cov = _empirical_covariance(initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)

    tempering_path, logZ_path, ess_path, acceptance_path, rw_scale_path = [], [], [], [], []
    logZ, n_iter = 0.0, 0
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def rmh_step_fn(rng_key, state, logdensity_fn, rw_scale, proposal_cov):
        transition_generator = blackjax.mcmc.random_walk.normal(_proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel(rng_key, state, logdensity_fn, transition_generator=transition_generator,)

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        mcmc_parameters = extend_params({"rw_scale": jnp.asarray(rw_scale), "proposal_cov": proposal_cov, })

        kernel = blackjax.adaptive_tempered_smc(logprior_fn=log_prior_fn,
                                                loglikelihood_fn=log_likelihood_fn,
                                                mcmc_step_fn=rmh_step_fn,
                                                mcmc_init_fn=blackjax.rmh.init,
                                                mcmc_parameters=mcmc_parameters,
                                                resampling_fn=resampling.systematic,
                                                target_ess=target_ess,
                                                num_mcmc_steps=num_mcmc_steps,)

        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state)
        logZ += float(info.log_likelihood_increment)
        ess_value = float(1.0 / jnp.sum(state.weights ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(logZ))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        rw_scale_path.append(float(rw_scale))

        if not np.isnan(acceptance_value):
            rw_scale = _update_rw_scale_robbins_monro(
                rw_scale=rw_scale,
                acceptance_value=acceptance_value,
                target_acceptance_rate=target_acceptance_rate,
                t=n_iter + 1,
                rm_c=rm_c,
                rm_t0=rm_t0,
                rm_kappa=rm_kappa,
            )

        proposal_cov = _empirical_covariance(state.particles,
                                            ridge=covariance_ridge,
                                            diagonal_only=diagonal_only_covariance,)

        n_iter += 1

    diagnostics = { "tempering_path": np.asarray(tempering_path, dtype=float),
                    "logZ_path": np.asarray(logZ_path, dtype=float),
                    "ess_path": np.asarray(ess_path, dtype=float),
                    "acceptance_path": np.asarray(acceptance_path, dtype=float),
                    "rw_scale_path": np.asarray(rw_scale_path, dtype=float),
                    "final_logZ": float(logZ),
                    "final_rw_scale": float(rw_scale),}

    return n_iter, state, diagnostics


def compute_posterior_moments_from_particles(particles: np.ndarray, ) -> tuple[np.ndarray, np.ndarray]:
    # first and second moment f1 = theta, f2 = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def run_reference_sampler_once( dimension: int,
                                num_particles: int,
                                seed: int,
                                max_iterations: int = 10_000,
                                alpha: float = 0.999,
                                num_mcmc_steps: int = 25,
                                rw_step_size: float = 0.5,
                                target_acceptance_rate: float = 0.234,
                                covariance_ridge: float = 1e-6,
                                diagonal_only_covariance: bool = False,
                                rm_c: float = 1.0,
                                rm_t0: float = 10.0,
                                rm_kappa: float = 0.6,
                                return_diagnostics: bool = True,) -> dict[str, Any]:
    #run refrence SMC for one seed
    #return:
    #logZ, posterior mean, posterior second moment, number of iterations, runtime, tempering path, logZ path,
    #ess path, acceptance path, rw scale path, final rw scale, final ess, acceptance rate mean.

    target = make_gaussian_mixture_target(dimension)
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, run_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    start = time.perf_counter()
    n_iter, smc_final_state, diagnostics = adaptive_loop_with_rwm_adaptation(run_key,
                                                                            log_prior_fn=log_prior_fn,
                                                                            log_likelihood_fn=log_likelihood_fn,
                                                                            initial_particles=initial_particles,
                                                                            dimension=dimension,
                                                                            max_iterations=max_iterations,
                                                                            target_ess=alpha,
                                                                            num_mcmc_steps=num_mcmc_steps,
                                                                            initial_rw_scale=rw_step_size,
                                                                            target_acceptance_rate=target_acceptance_rate,
                                                                            covariance_ridge=covariance_ridge,
                                                                            diagonal_only_covariance=diagonal_only_covariance,
                                                                            rm_c=rm_c,
                                                                            rm_t0=rm_t0,
                                                                            rm_kappa=rm_kappa,)
    runtime_sec = time.perf_counter() - start
    particles = np.array(smc_final_state.particles)
    f1, f2 = compute_posterior_moments_from_particles(particles)
    ess_path = np.asarray(diagnostics["ess_path"], dtype=float)
    acceptance_path = np.asarray(diagnostics["acceptance_path"], dtype=float)
    final_ess = float(ess_path[-1]) if ess_path.size > 0 else np.nan
    acceptance_rate_mean = float(np.nanmean(acceptance_path)) if acceptance_path.size > 0 else np.nan

    out = {"seed": int(seed),
           "logZ": float(diagnostics["final_logZ"]),
           "posterior_mean": f1,
           "posterior_second_moment": f2,
           "n_iter": int(n_iter),
           "runtime_sec": float(runtime_sec),
           "final_rw_scale": float(diagnostics["final_rw_scale"]),
           "final_ess": final_ess,
           "acceptance_rate_mean": acceptance_rate_mean,}

    if return_diagnostics:
        out.update({"tempering_path": np.asarray(diagnostics["tempering_path"], dtype=float),
                    "logZ_path": np.asarray(diagnostics["logZ_path"], dtype=float),
                    "ess_path": ess_path,
                    "acceptance_path": acceptance_path,
                    "rw_scale_path": np.asarray(diagnostics["rw_scale_path"], dtype=float), })

    return out


def _assemble_reference_stats(dimension: int, num_particles_reference: int, num_reference_runs: int, logZ_runs, f1_runs, f2_runs,
                              final_ess_runs, acceptance_rate_runs, n_iter_runs, ) -> ReferenceStats:
    # takes the outputs of multiple runns and combines them into RefrenceSatas object
    logZ_runs = np.asarray(logZ_runs, dtype=float)
    f1_runs = np.stack(f1_runs, axis=0)
    f2_runs = np.stack(f2_runs, axis=0)
    final_ess_runs = np.asarray(final_ess_runs, dtype=float)
    acceptance_rate_runs = np.asarray(acceptance_rate_runs, dtype=float)
    n_iter_runs = np.asarray(n_iter_runs, dtype=float)
    logZ_ref_std = float(np.nanstd(logZ_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    f1_std_ref = np.std(f1_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f1_runs[0])
    f2_std_ref = np.std(f2_runs, axis=0, ddof=1) if num_reference_runs > 1 else np.zeros_like(f2_runs[0])
    final_ess_std = float(np.nanstd(final_ess_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    acceptance_rate_std = float(np.nanstd(acceptance_rate_runs, ddof=1)) if num_reference_runs > 1 else 0.0
    n_iter_std = float(np.nanstd(n_iter_runs, ddof=1)) if num_reference_runs > 1 else 0.0

    return ReferenceStats(  target_name="gaussian_mixture",
                            dimension=dimension,
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
                            n_iter_std=n_iter_std,)


def save_reference_stats(ref: ReferenceStats, outpath: str | Path) -> None:
    #saves ReferenceStats object into JSON file.
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
                            dimension=payload["dimension"],
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
                            n_iter_std=payload["n_iter_std"],)


def save_chunk_checkpoint(checkpoint_path: str | Path, *, dimension: int, num_particles_reference: int,
                          num_reference_runs: int, completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs,
                          acceptance_rate_runs, n_iter_runs,) -> None:
    # this saces the progres of the chunked execution
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = { "dimension": int(dimension),
                "num_particles_reference": int(num_particles_reference),
                "num_reference_runs": int(num_reference_runs),
                "completed_seeds": list(map(int, completed_seeds)),
                "logZ_runs": np.asarray(logZ_runs, dtype=float).tolist(),
                "f1_runs": np.asarray(f1_runs, dtype=float).tolist(),
                "f2_runs": np.asarray(f2_runs, dtype=float).tolist(),
                "final_ess_runs": np.asarray(final_ess_runs, dtype=float).tolist(),
                "acceptance_rate_runs": np.asarray(acceptance_rate_runs, dtype=float).tolist(),
                "n_iter_runs": np.asarray(n_iter_runs, dtype=float).tolist(),}

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
    return payload


def build_reference_stats_chunked(  dimension: int,
                                    num_particles_reference: int,
                                    num_reference_runs: int,
                                    chunk_size: int = 10,
                                    max_iterations: int = 10_000,
                                    alpha: float = 0.999,
                                    num_mcmc_steps: int = 25,
                                    rw_step_size: float = 0.5,
                                    target_acceptance_rate: float = 0.234,
                                    covariance_ridge: float = 1e-6,
                                    diagonal_only_covariance: bool = False,
                                    rm_c: float = 1.0,
                                    rm_t0: float = 10.0,
                                    rm_kappa: float = 0.6,
                                    verbose: bool = True,
                                    checkpoint_path: str | Path | None = None,
                                    save_final_path: str | Path | None = None, ) -> ReferenceStats:
    # runs multiple reference runs in chunks

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = load_chunk_checkpoint(checkpoint_path)

        if (ckpt["dimension"] != dimension
            or ckpt["num_particles_reference"] != num_particles_reference
            or ckpt["num_reference_runs"] != num_reference_runs ):
            raise ValueError("Checkpoint metadata does not match current run settings.")

        completed_seeds = list(ckpt["completed_seeds"])
        logZ_runs = list(ckpt["logZ_runs"])
        f1_runs = list(ckpt["f1_runs"])
        f2_runs = list(ckpt["f2_runs"])
        final_ess_runs = list(ckpt["final_ess_runs"])
        acceptance_rate_runs = list(ckpt["acceptance_rate_runs"])
        n_iter_runs = list(ckpt["n_iter_runs"])

        if verbose:
            print(f"Loaded checkpoint with {len(completed_seeds)} completed runs.")
    else:
        completed_seeds, logZ_runs, f1_runs, f2_runs, final_ess_runs, acceptance_rate_runs, n_iter_runs = [], [], [], [], [], [], []
    remaining_seeds = [seed for seed in range(num_reference_runs) if seed not in set(completed_seeds)]
    for chunk_start in range(0, len(remaining_seeds), chunk_size):
        chunk = remaining_seeds[chunk_start:chunk_start + chunk_size]
        if verbose:
            print(f"Running reference chunk seeds {chunk[0]}:{chunk[-1] + 1}")
        for seed in chunk:
            out = run_reference_sampler_once(dimension=dimension,
                                            num_particles=num_particles_reference,
                                            seed=seed,
                                            max_iterations=max_iterations,
                                            alpha=alpha,
                                            num_mcmc_steps=num_mcmc_steps,
                                            rw_step_size=rw_step_size,
                                            target_acceptance_rate=target_acceptance_rate,
                                            covariance_ridge=covariance_ridge,
                                            diagonal_only_covariance=diagonal_only_covariance,
                                            rm_c=rm_c,
                                            rm_t0=rm_t0,
                                            rm_kappa=rm_kappa,
                                            return_diagnostics=False,)

            completed_seeds.append(seed)
            logZ_runs.append(out["logZ"])
            f1_runs.append(out["posterior_mean"])
            f2_runs.append(out["posterior_second_moment"])
            final_ess_runs.append(out["final_ess"])
            acceptance_rate_runs.append(out["acceptance_rate_mean"])
            n_iter_runs.append(out["n_iter"])

        if checkpoint_path is not None:
            save_chunk_checkpoint(  checkpoint_path,
                                    dimension=dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    completed_seeds=completed_seeds,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,)
            if verbose:
                print(f"Checkpoint saved to {checkpoint_path}")

    ref = _assemble_reference_stats(dimension=dimension,
                                    num_particles_reference=num_particles_reference,
                                    num_reference_runs=num_reference_runs,
                                    logZ_runs=logZ_runs,
                                    f1_runs=f1_runs,
                                    f2_runs=f2_runs,
                                    final_ess_runs=final_ess_runs,
                                    acceptance_rate_runs=acceptance_rate_runs,
                                    n_iter_runs=n_iter_runs,)

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
            "n_iter_std_is_nan": bool(np.isnan(ref.n_iter_std)),}
"""

#############################

"""
Now for a version that is parallelized 

"""