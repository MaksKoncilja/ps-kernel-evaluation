# experiment.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax.numpy as jnp

from samplers import (
    run_ps_rwm_once,
    run_ps_hmc_once,
    run_ps_ula_once,
    run_ps_mala_once,
    run_ps_nuts_once,
)

from targets import (
    make_gaussian_mixture_target,
    make_rosenbrock_target,
    make_sparse_logistic_regression_target,
)


SCALAR_SUMMARY_KEYS = [
    "logZ",
    "final_ess",
    "acceptance_rate_mean",
    "acceptance_rate_last",
    "n_iter",
    "runtime_sec",
    "gradient_eval_count",
    "resampling_steps",
]

ARRAY_SUMMARY_KEYS = [
    "posterior_mean",
    "posterior_second_moment",
]


@dataclass
class ExperimentConfig:
    target_name: str
    algorithm_name: str
    dimensions: list[int]
    num_particles_grid: list[int]
    num_mcmc_steps_grid: list[int]
    kernels: list[str]
    seeds: list[int]
    dataset_path: str | None = None
    max_iterations: int = 10_000
    alpha: float = 0.999
    chunk_size: int = 10
    parallel: bool = True
    num_workers: int | None = None
    verbose: bool = True


def _default_results_root() -> Path:
    # default results root inside project/data/results (where I decideed to save them)
    here = Path(__file__).resolve()  # path of a current file
    project_root = here.parent.parent  # the parent file of the whole project
    return project_root / "data" / "results"  # root for all the outputs


def _kernel_defaults(kernel_name: str) -> dict[str, Any]:
    # specific kernel parameters set as defaults
    defaults = {
        "rwm": {},
        "hmc": {},
        "ula": {},
        "mala": {},
        "nuts": {},
    }
    if kernel_name not in defaults:
        raise ValueError(f"Unknown kernel '{kernel_name}'")
    return defaults[kernel_name]


def _sampler_dispatch(kernel_name: str):
    # returns the kernel function depending on a str. abbreviation
    dispatch = {
        "rwm": run_ps_rwm_once,
        "hmc": run_ps_hmc_once,
        "ula": run_ps_ula_once,
        "mala": run_ps_mala_once,
        "nuts": run_ps_nuts_once,
    }
    if kernel_name not in dispatch:
        raise ValueError(f"Unknown kernel '{kernel_name}'")
    return dispatch[kernel_name]


def _raw_output_path(
    *,
    results_root: Path,
    target_name: str,
    dimension: int,
    num_particles: int,
    num_mcmc_steps: int,
    kernel_name: str,
    seed: int,
) -> Path:
    # output path for the raw data
    return (
        results_root
        / "raw"
        / target_name
        / f"D_{dimension}"
        / f"N_{num_particles}"
        / f"MCMC_{num_mcmc_steps}"
        / kernel_name
        / f"seed_{seed}.json"
    )


def _summary_output_path(
    *,
    results_root: Path,
    target_name: str,
    dimension: int,
    num_particles: int,
    num_mcmc_steps: int,
    kernel_name: str,
) -> Path:
    # output path for the summary of each kernel
    return (
        results_root
        / "summary"
        / target_name
        / f"D_{dimension}"
        / f"N_{num_particles}"
        / f"MCMC_{num_mcmc_steps}"
        / f"{kernel_name}_summary.json"
    )


def _to_jsonable(x: Any) -> Any:
    # makes it possible to save the output data in JSON
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _save_json(payload: dict[str, Any], outpath: str | Path) -> None:
    # saves a json file
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _load_json(path: str | Path) -> dict[str, Any]:
    # loads it
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_german_numeric_dataset(path: str | Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Load german.data-numeric:
    # first 24 columns -> X
    # last column -> y_raw in {1, 2}
    path = Path(path)
    data = np.loadtxt(path, dtype=float)

    if data.ndim != 2 or data.shape[1] != 25:
        raise ValueError(
            f"Expected german.data-numeric to have shape (n, 25), got {data.shape}."
        )

    X = data[:, :24]
    y_raw = data[:, 24].astype(int)

    if not np.all(np.isin(y_raw, [1, 2])):
        raise ValueError("Expected labels in last column to be only 1 or 2.")

    # Map labels:
    # 1 = Good -> 0
    # 2 = Bad  -> 1
    y = (y_raw == 2).astype(float)

    return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)


def _build_target_for_experiment(
    *,
    target_name: str,
    dimension: int,
    dataset_path: str | None = None,
):
    # builds the mathematical target for a given benchmark
    if target_name == "gaussian_mixture":
        return make_gaussian_mixture_target(dimension)

    if target_name == "rosenbrock":
        return make_rosenbrock_target(dimension)

    if target_name == "sparse_logistic_regression":
        if dataset_path is None:
            raise ValueError("dataset_path is required for sparse_logistic_regression")

        X, y = load_german_numeric_dataset(dataset_path)

        target = make_sparse_logistic_regression_target(
            X,
            y,
            add_intercept=True,
            standardize=True,
        )

        if int(dimension) != int(target.dimension):
            raise ValueError(
                f"For sparse_logistic_regression the target dimension is {target.dimension}, "
                f"but config requested dimension={dimension}."
            )

        return target

    raise ValueError(f"Unknown target_name '{target_name}'")


def _run_experiment_worker(args: dict[str, Any]) -> dict[str, Any]:
    # top-level worker wrapper for multiprocessing
    sampler = _sampler_dispatch(args["kernel_name"])

    target = _build_target_for_experiment(
        target_name=args["target_name"],
        dimension=args["dimension"],
        dataset_path=args.get("dataset_path"),
    )

    # common sampler arguments
    sampler_kwargs = {
        "dimension": int(target.dimension),
        "num_particles": args["num_particles"],
        "seed": args["seed"],
        "target": target,
        "max_iterations": args["max_iterations"],
        "alpha": args["alpha"],
        "num_mcmc_steps": args["num_mcmc_steps"],
    }
    sampler_kwargs.update(args.get("kernel_kwargs", {}))

    out = sampler(**sampler_kwargs)  # runs the sampler, gives result dict.
    out["target_name"] = args["target_name"]
    out["num_mcmc_steps"] = int(args["num_mcmc_steps"])
    out["raw_output_path"] = str(args["raw_output_path"])
    return out


def _save_raw_result(out: dict[str, Any], raw_output_path: str | Path) -> None:
    # saves the raw result from the out dict.
    payload = dict(out)
    payload.pop("raw_output_path", None)
    _save_json(payload, raw_output_path)


def _aggregate_scalar(values: np.ndarray) -> dict[str, float]:
    # returns mean and std of finite arrays
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan}
    std = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0
    return {"mean": float(np.nanmean(finite)), "std": std}


def _aggregate_array(arrays: list[np.ndarray]) -> dict[str, Any]:
    # computes mean and standard deviation across runs for vector outputs
    stacked = np.stack(arrays, axis=0)
    if stacked.shape[0] > 1:
        std = np.std(stacked, axis=0, ddof=1)
    else:
        std = np.zeros_like(stacked[0])
    return {
        "mean": np.mean(stacked, axis=0),
        "std": std,
    }


def _summarize_runs(
    *,
    config: ExperimentConfig,
    kernel_name: str,
    dimension: int,
    num_particles: int,
    num_mcmc_steps: int,
    raw_paths: list[Path],
) -> dict[str, Any]:
    # creates summary files from raw files
    runs = [_load_json(path) for path in raw_paths]

    summary: dict[str, Any] = {
        "target_name": config.target_name,
        "algorithm_name": config.algorithm_name,
        "kernel_name": kernel_name,
        "dimension": int(dimension),
        "num_particles": int(num_particles),
        "num_mcmc_steps": int(num_mcmc_steps),
        "num_completed_runs": int(len(runs)),
        "completed_seeds": sorted(int(run["seed"]) for run in runs),
        "raw_result_paths": [str(path) for path in raw_paths],
    }

    for key in SCALAR_SUMMARY_KEYS:
        values = np.asarray([run.get(key, np.nan) for run in runs], dtype=float)
        stats = _aggregate_scalar(values)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_std"] = stats["std"]

    for key in ARRAY_SUMMARY_KEYS:
        arrays = [np.asarray(run[key], dtype=float) for run in runs]
        stats = _aggregate_array(arrays)
        summary[f"{key}_mean"] = stats["mean"]
        summary[f"{key}_std"] = stats["std"]

    return summary


def _print_config_header(
    *,
    kernel_name: str,
    dimension: int,
    num_particles: int,
    num_mcmc_steps: int,
    num_missing: int,
    total: int,
) -> None:
    # prints progress status in kernel so that I now things are working
    print(
        f"[kernel={kernel_name} | D={dimension} | N={num_particles} | MCMC={num_mcmc_steps}] "
        f"missing_runs={num_missing}/{total}",
        flush=True,
    )


def run_experiment_grid(
    config: ExperimentConfig,
    results_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    # runs PS experiments over the cartesian product of config fields
    if results_root is None:  # determines where the results are stored
        results_root = _default_results_root()
    results_root = Path(results_root)

    all_summaries: list[dict[str, Any]] = []  # output container
    total_jobs = 0

    # computes the total number of independet runs
    for dimension in config.dimensions:
        for num_particles in config.num_particles_grid:
            for num_mcmc_steps in config.num_mcmc_steps_grid:
                for kernel_name in config.kernels:
                    total_jobs += len(config.seeds)

    completed_jobs = 0
    start_time = time.perf_counter()

    # main nested loop over configurations
    for dimension in config.dimensions:
        for num_particles in config.num_particles_grid:
            for num_mcmc_steps in config.num_mcmc_steps_grid:
                for kernel_name in config.kernels:
                    # makes raw file paths, for each seed
                    raw_paths = [
                        _raw_output_path(
                            results_root=results_root,
                            target_name=config.target_name,
                            dimension=dimension,
                            num_particles=num_particles,
                            num_mcmc_steps=num_mcmc_steps,
                            kernel_name=kernel_name,
                            seed=seed,
                        )
                        for seed in config.seeds
                    ]

                    # check in the files if raw JSON file exists
                    missing_jobs = []
                    for seed, raw_path in zip(config.seeds, raw_paths):
                        if raw_path.exists():
                            completed_jobs += 1
                            continue

                        # each missing job gets its own dict
                        missing_jobs.append(
                            {
                                "target_name": config.target_name,
                                "dataset_path": config.dataset_path,
                                "kernel_name": kernel_name,
                                "dimension": int(dimension),
                                "num_particles": int(num_particles),
                                "num_mcmc_steps": int(num_mcmc_steps),
                                "seed": int(seed),
                                "max_iterations": int(config.max_iterations),
                                "alpha": float(config.alpha),
                                "kernel_kwargs": _kernel_defaults(kernel_name),
                                "raw_output_path": str(raw_path),
                            }
                        )

                    if config.verbose:  # printing configuration header
                        _print_config_header(
                            kernel_name=kernel_name,
                            dimension=dimension,
                            num_particles=num_particles,
                            num_mcmc_steps=num_mcmc_steps,
                            num_missing=len(missing_jobs),
                            total=len(config.seeds),
                        )

                    # chunking jobs
                    for chunk_start in range(0, len(missing_jobs), config.chunk_size):
                        chunk = missing_jobs[chunk_start:chunk_start + config.chunk_size]
                        if len(chunk) == 0:
                            continue

                        if config.parallel:
                            workers = config.num_workers or min(len(chunk), os.cpu_count() or 1)
                            ctx = mp.get_context("spawn")

                            # creating a process pool
                            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                                futures = [executor.submit(_run_experiment_worker, job_args) for job_args in chunk]

                                for future in as_completed(futures):  # collect completed jobs
                                    out = future.result()
                                    _save_raw_result(out, out["raw_output_path"])
                                    completed_jobs += 1

                                    if config.verbose:
                                        elapsed = time.perf_counter() - start_time
                                        avg_time = elapsed / max(completed_jobs, 1)
                                        remaining = avg_time * (total_jobs - completed_jobs)
                                        print(
                                            f"Run {completed_jobs}/{total_jobs} completed "
                                            f"(kernel={out['kernel_name']}, D={out['dimension']}, "
                                            f"N={out['num_particles']}, MCMC={out['num_mcmc_steps']}, seed={out['seed']}) | "
                                            f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s",
                                            flush=True,
                                        )
                        else:
                            for job_args in chunk:
                                out = _run_experiment_worker(job_args)
                                _save_raw_result(out, out["raw_output_path"])
                                completed_jobs += 1

                                if config.verbose:  # print in terminal info of one batch
                                    elapsed = time.perf_counter() - start_time
                                    avg_time = elapsed / max(completed_jobs, 1)
                                    remaining = avg_time * (total_jobs - completed_jobs)
                                    print(
                                        f"Run {completed_jobs}/{total_jobs} completed "
                                        f"(kernel={out['kernel_name']}, D={out['dimension']}, "
                                        f"N={out['num_particles']}, MCMC={out['num_mcmc_steps']}, seed={out['seed']}) | "
                                        f"elapsed={elapsed:.1f}s | ETA={remaining:.1f}s",
                                        flush=True,
                                    )

                    existing_raw_paths = [path for path in raw_paths if path.exists()]
                    if len(existing_raw_paths) == 0:
                        continue

                    # from the raw files calculate make the summary
                    summary = _summarize_runs(
                        config=config,
                        kernel_name=kernel_name,
                        dimension=dimension,
                        num_particles=num_particles,
                        num_mcmc_steps=num_mcmc_steps,
                        raw_paths=existing_raw_paths,
                    )

                    summary_path = _summary_output_path(
                        results_root=results_root,
                        target_name=config.target_name,
                        dimension=dimension,
                        num_particles=num_particles,
                        num_mcmc_steps=num_mcmc_steps,
                        kernel_name=kernel_name,
                    )
                    _save_json(summary, summary_path)
                    all_summaries.append(summary)

                    if config.verbose:
                        print(f"Summary saved to {summary_path}", flush=True)

    return all_summaries


if __name__ == "__main__":
    config = ExperimentConfig(
        target_name="gaussian_mixture",
        algorithm_name="ps",
        dimensions=[10],
        num_particles_grid=[16],
        num_mcmc_steps_grid=[25],
        kernels=["rwm", "hmc", "mala", "ula", "nuts"],
        seeds=list(range(5)),
        dataset_path=None,
        chunk_size=5,
        parallel=True,
        num_workers=5,
        verbose=True,
    )

    summaries = run_experiment_grid(config=config)
    print(f"Completed {len(summaries)} summary files.", flush=True)
    