from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from samplers import (
    run_ps_hmc_once,
    run_ps_mala_once,
    run_ps_nuts_once,
    run_ps_rwm_once,
    run_ps_ula_once,
)
from reference import run_reference_sampler_once
from targets import make_gaussian_mixture_target


@dataclass
class ValidationConfig:
    dimension: int = 10
    num_particles: int = 128
    max_iterations: int = 250
    alpha: float = 0.999
    num_mcmc_steps: int = 5
    seed: int = 0
    num_repeats: int = 3
    check_reference: bool = True
    reference_particles: int = 256
    reference_num_mcmc_steps: int = 10
    strict_tempering_completion: bool = False
    atol_mean: float = 1.0
    atol_second_moment: float = 2.0
    rtol_mean: float = 0.35
    rtol_second_moment: float = 0.35


SAMPLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "rwm": run_ps_rwm_once,
    "hmc": run_ps_hmc_once,
    "ula": run_ps_ula_once,
    "mala": run_ps_mala_once,
    "nuts": run_ps_nuts_once,
}


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _finite_array(x: Any) -> bool:
    arr = _to_numpy(x)
    return bool(np.all(np.isfinite(arr)))


def _path_is_nondecreasing(x: Any, atol: float = 1e-10) -> bool:
    arr = _to_numpy(x)
    if arr.size <= 1:
        return True
    return bool(np.all(np.diff(arr) >= -atol))


def _safe_mean_finite(x: Any) -> bool:
    arr = _to_numpy(x)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    return finite.size > 0


def _validate_common_structure(result: dict[str, Any], dimension: int) -> list[str]:
    errors: list[str] = []
    required_keys = [
        "target_name",
        "algorithm_name",
        "kernel_name",
        "seed",
        "dimension",
        "num_particles",
        "logZ",
        "posterior_mean",
        "posterior_second_moment",
        "final_ess",
        "acceptance_rate_mean",
        "n_iter",
        "runtime_sec",
        "tempering_path",
        "logZ_path",
        "ess_path",
        "acceptance_path",
        "elapsed_time_path",
        "gradient_eval_count",
        "resampling_steps",
        "variance_log_weights_path",
        "weight_entropy_path",
        "esjd_path",
    ]
    for key in required_keys:
        if key not in result:
            errors.append(f"missing key: {key}")

    if errors:
        return errors

    if result["algorithm_name"] != "ps":
        errors.append(f"algorithm_name should be 'ps', got {result['algorithm_name']!r}")

    if int(result["dimension"]) != int(dimension):
        errors.append(f"dimension mismatch: expected {dimension}, got {result['dimension']}")

    pm = _to_numpy(result["posterior_mean"])
    p2 = _to_numpy(result["posterior_second_moment"])
    if pm.shape != (dimension,):
        errors.append(f"posterior_mean has wrong shape: {pm.shape}")
    if p2.shape != (dimension,):
        errors.append(f"posterior_second_moment has wrong shape: {p2.shape}")

    scalar_keys = ["logZ", "final_ess", "n_iter", "runtime_sec", "gradient_eval_count", "resampling_steps"]
    for key in scalar_keys:
        if not np.isfinite(float(result[key])):
            errors.append(f"{key} is not finite: {result[key]}")

    path_keys = [
        "tempering_path",
        "logZ_path",
        "ess_path",
        "acceptance_path",
        "elapsed_time_path",
        "variance_log_weights_path",
        "weight_entropy_path",
        "esjd_path",
    ]
    lengths = {key: len(result[key]) for key in path_keys}
    lengths["n_iter"] = int(result["n_iter"])
    base_len = int(result["n_iter"])
    for key, value in lengths.items():
        if value != base_len:
            errors.append(f"length mismatch: {key} has length {value}, expected {base_len}")

    if not _path_is_nondecreasing(result["tempering_path"]):
        errors.append("tempering_path is not nondecreasing")
    if not _path_is_nondecreasing(result["elapsed_time_path"]):
        errors.append("elapsed_time_path is not nondecreasing")

    if len(result["tempering_path"]) > 0:
        last_beta = float(_to_numpy(result["tempering_path"])[-1])
        if not (0.0 < last_beta <= 1.000001):
            errors.append(f"final tempering parameter out of range: {last_beta}")

    if float(result["final_ess"]) <= 0.0:
        errors.append(f"final_ess should be positive, got {result['final_ess']}")
    if int(result["gradient_eval_count"]) < 0:
        errors.append(f"gradient_eval_count should be nonnegative, got {result['gradient_eval_count']}")
    if int(result["resampling_steps"]) != int(result["n_iter"]):
        errors.append(
            f"resampling_steps should equal n_iter in current implementation, got {result['resampling_steps']} vs {result['n_iter']}"
        )

    if not _finite_array(result["posterior_mean"]):
        errors.append("posterior_mean contains nonfinite values")
    if not _finite_array(result["posterior_second_moment"]):
        errors.append("posterior_second_moment contains nonfinite values")
    if not _finite_array(result["logZ_path"]):
        errors.append("logZ_path contains nonfinite values")
    if not _finite_array(result["ess_path"]):
        errors.append("ess_path contains nonfinite values")
    if not _finite_array(result["elapsed_time_path"]):
        errors.append("elapsed_time_path contains nonfinite values")
    if not _finite_array(result["variance_log_weights_path"]):
        errors.append("variance_log_weights_path contains nonfinite values")
    if not _finite_array(result["weight_entropy_path"]):
        errors.append("weight_entropy_path contains nonfinite values")
    if not _finite_array(result["esjd_path"]):
        errors.append("esjd_path contains nonfinite values")

    acc = _to_numpy(result["acceptance_path"])
    finite_acc = acc[np.isfinite(acc)]
    if finite_acc.size > 0:
        if np.any(finite_acc < -1e-8) or np.any(finite_acc > 1.0 + 1e-8):
            errors.append("acceptance_path has finite values outside [0, 1]")

    ent = _to_numpy(result["weight_entropy_path"])
    if np.any(ent < -1e-8):
        errors.append("weight_entropy_path has negative values")

    esjd = _to_numpy(result["esjd_path"])
    if np.any(esjd < -1e-8):
        errors.append("esjd_path has negative values")

    return errors


def _validate_against_target_moments(result: dict[str, Any], cfg: ValidationConfig) -> list[str]:
    errors: list[str] = []
    target = make_gaussian_mixture_target(cfg.dimension)

    if target.posterior_mean_exact_fn is not None:
        approx_mean = _to_numpy(target.posterior_mean_exact_fn())
        got_mean = _to_numpy(result["posterior_mean"])
        if not np.allclose(got_mean, approx_mean, atol=cfg.atol_mean, rtol=cfg.rtol_mean):
            max_abs = float(np.max(np.abs(got_mean - approx_mean)))
            errors.append(
                "posterior_mean too far from approximate sanity target "
                f"(max abs diff={max_abs:.4g}, atol={cfg.atol_mean}, rtol={cfg.rtol_mean})"
            )

    if target.posterior_second_moment_exact_fn is not None:
        approx_second = _to_numpy(target.posterior_second_moment_exact_fn())
        got_second = _to_numpy(result["posterior_second_moment"])
        if not np.allclose(got_second, approx_second, atol=cfg.atol_second_moment, rtol=cfg.rtol_second_moment):
            max_abs = float(np.max(np.abs(got_second - approx_second)))
            errors.append(
                "posterior_second_moment too far from approximate sanity target "
                f"(max abs diff={max_abs:.4g}, atol={cfg.atol_second_moment}, rtol={cfg.rtol_second_moment})"
            )

    return errors


def _validate_tempering_completion(result: dict[str, Any], cfg: ValidationConfig) -> list[str]:
    errors: list[str] = []
    tp = _to_numpy(result["tempering_path"])
    if tp.size == 0:
        errors.append("tempering_path is empty")
        return errors

    last_beta = float(tp[-1])
    reached_end = last_beta >= 1.0 - 1e-6
    hit_cap = int(result["n_iter"]) >= int(cfg.max_iterations)
    if cfg.strict_tempering_completion and not reached_end:
        errors.append(f"tempering did not reach 1.0, final beta={last_beta:.6f}")
    elif (not reached_end) and (not hit_cap):
        errors.append(
            f"tempering stopped early without reaching 1.0 and without hitting max_iterations; final beta={last_beta:.6f}"
        )
    return errors


def _compare_repeated_runs(results: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(results) < 2:
        return errors

    a = results[0]
    b = results[1]
    compare_keys = [
        "logZ",
        "posterior_mean",
        "posterior_second_moment",
        "final_ess",
        "acceptance_rate_mean",
        "n_iter",
        "tempering_path",
        "logZ_path",
        "ess_path",
        "acceptance_path",
        "gradient_eval_count",
        "resampling_steps",
        "variance_log_weights_path",
        "weight_entropy_path",
        "esjd_path",
    ]
    for key in compare_keys:
        av = _to_numpy(a[key]) if isinstance(a[key], (list, tuple, np.ndarray)) else a[key]
        bv = _to_numpy(b[key]) if isinstance(b[key], (list, tuple, np.ndarray)) else b[key]
        if isinstance(av, np.ndarray):
            if not np.allclose(av, bv, atol=1e-10, rtol=1e-10, equal_nan=True):
                errors.append(f"same-seed repeatability failed for key {key}")
        else:
            if isinstance(av, float):
                if not math.isclose(float(av), float(bv), rel_tol=1e-10, abs_tol=1e-10):
                    errors.append(f"same-seed repeatability failed for key {key}")
            elif av != bv:
                errors.append(f"same-seed repeatability failed for key {key}")
    return errors


def _check_reference_agreement(result: dict[str, Any], cfg: ValidationConfig) -> list[str]:
    errors: list[str] = []
    ref = run_reference_sampler_once(
        dimension=cfg.dimension,
        num_particles=cfg.reference_particles,
        seed=cfg.seed,
        max_iterations=cfg.max_iterations,
        alpha=cfg.alpha,
        num_mcmc_steps=cfg.reference_num_mcmc_steps,
        return_diagnostics=False,
    )
    mean_diff = np.max(np.abs(_to_numpy(result["posterior_mean"]) - _to_numpy(ref["posterior_mean"])))
    second_diff = np.max(
        np.abs(_to_numpy(result["posterior_second_moment"]) - _to_numpy(ref["posterior_second_moment"]))
    )
    if not np.isfinite(mean_diff) or mean_diff > 2.0:
        errors.append(f"posterior_mean disagrees too much with reference RWM run (max abs diff={mean_diff:.4g})")
    if not np.isfinite(second_diff) or second_diff > 4.0:
        errors.append(
            f"posterior_second_moment disagrees too much with reference RWM run (max abs diff={second_diff:.4g})"
        )
    return errors


def validate_sampler(name: str, cfg: ValidationConfig) -> dict[str, Any]:
    sampler = SAMPLERS[name]
    run_kwargs = dict(
        dimension=cfg.dimension,
        num_particles=cfg.num_particles,
        seed=cfg.seed,
        max_iterations=cfg.max_iterations,
        alpha=cfg.alpha,
        num_mcmc_steps=cfg.num_mcmc_steps,
    )

    repeated_results = [sampler(**run_kwargs) for _ in range(max(cfg.num_repeats, 1))]
    result = repeated_results[0]

    errors: list[str] = []
    errors.extend(_validate_common_structure(result, cfg.dimension))
    errors.extend(_validate_tempering_completion(result, cfg))
    errors.extend(_validate_against_target_moments(result, cfg))
    errors.extend(_compare_repeated_runs(repeated_results))
    if cfg.check_reference and name == "rwm":
        errors.extend(_check_reference_agreement(result, cfg))

    acc = _to_numpy(result["acceptance_path"])
    finite_acc = acc[np.isfinite(acc)]
    summary = {
        "kernel_name": name,
        "passed": len(errors) == 0,
        "errors": errors,
        "summary": {
            "logZ": float(result["logZ"]),
            "final_ess": float(result["final_ess"]),
            "acceptance_rate_mean": (
                float(np.mean(finite_acc)) if finite_acc.size > 0 else None
            ),
            "n_iter": int(result["n_iter"]),
            "runtime_sec": float(result["runtime_sec"]),
            "gradient_eval_count": int(result["gradient_eval_count"]),
            "resampling_steps": int(result["resampling_steps"]),
            "final_tempering_param": (
                float(_to_numpy(result["tempering_path"])[-1]) if len(result["tempering_path"]) > 0 else None
            ),
            "posterior_mean_mean": float(np.mean(_to_numpy(result["posterior_mean"]))),
            "posterior_second_moment_mean": float(np.mean(_to_numpy(result["posterior_second_moment"]))),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PS sampler wrappers on the Gaussian-mixture target.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.999)
    parser.add_argument("--num-mcmc-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--samplers", nargs="*", choices=sorted(SAMPLERS.keys()), default=sorted(SAMPLERS.keys()))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--strict-tempering-completion", action="store_true")
    parser.add_argument("--reference-particles", type=int, default=256)
    parser.add_argument("--reference-num-mcmc-steps", type=int, default=10)
    parser.add_argument("--atol-mean", type=float, default=1.0)
    parser.add_argument("--atol-second-moment", type=float, default=2.0)
    parser.add_argument("--rtol-mean", type=float, default=0.35)
    parser.add_argument("--rtol-second-moment", type=float, default=0.35)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    cfg = ValidationConfig(
        dimension=args.dimension,
        num_particles=args.num_particles,
        max_iterations=args.max_iterations,
        alpha=args.alpha,
        num_mcmc_steps=args.num_mcmc_steps,
        seed=args.seed,
        num_repeats=args.num_repeats,
        check_reference=not args.skip_reference,
        strict_tempering_completion=args.strict_tempering_completion,
        reference_particles=args.reference_particles,
        reference_num_mcmc_steps=args.reference_num_mcmc_steps,
        atol_mean=args.atol_mean,
        atol_second_moment=args.atol_second_moment,
        rtol_mean=args.rtol_mean,
        rtol_second_moment=args.rtol_second_moment,
    )

    report: dict[str, Any] = {
        "config": asdict(cfg),
        "results": {},
        "overall_passed": True,
    }

    for name in args.samplers:
        print(f"\n=== validating {name} ===")
        try:
            summary = validate_sampler(name, cfg)
        except Exception as exc:  # pragma: no cover
            summary = {
                "kernel_name": name,
                "passed": False,
                "errors": [f"runtime exception: {exc}", traceback.format_exc()],
                "summary": {},
            }
        report["results"][name] = summary
        report["overall_passed"] = bool(report["overall_passed"] and summary["passed"])

        print(json.dumps(summary, indent=2))

    if args.json_out is not None:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {outpath}")

    print("\n=== overall ===")
    print(json.dumps({"overall_passed": report["overall_passed"]}, indent=2))
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
