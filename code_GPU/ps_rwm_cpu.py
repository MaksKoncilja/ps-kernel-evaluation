from __future__ import annotations

from typing import Any
import os

# CPU-only PS + RWM reference implementation.
# This must happen before importing jax, blackjax, targets, or ps_common.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax

from targets import Target, make_gaussian_mixture_target, make_target
from ps_common import (
    debug_worker_backend,
    empirical_covariance,
    proposal_sqrt_from_cov,
    update_rw_scale_robbins_monro,
    run_ps_with_inner_adaptation_once,
)


def run_ps_rwm_once(
    dimension: int,
    num_particles: int,
    seed: int,
    target: Target | None = None,
    target_name: str = "gaussian_mixture",
    target_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 10_000,
    alpha: float = 0.999,
    num_mcmc_steps: int = 25,
    rw_step_size: float = 1.0,
    target_acceptance_rate: float = 0.234,
    covariance_ridge: float = 1e-6,
    diagonal_only_covariance: bool = False,
    rm_c: float = 2.0,
    rm_t0: float = 1.0,
    rm_kappa: float = 0.6,
) -> dict[str, Any]:
    """Run Persistent Sampling + adaptive Random-Walk Metropolis on CPU.

    This is intended as the CPU/reference kernel. It deliberately excludes MALA,
    HMC, NUTS, ULA, and MCLMC to avoid accidentally running the wrong backend.
    """
    if target is None:
        if target_name == "gaussian_mixture" and target_kwargs is None:
            target = make_gaussian_mixture_target(dimension)
        else:
            target = make_target(target_name, dimension, **(target_kwargs or {}))

    base_rmh_kernel = blackjax.rmh.build_kernel()

    def init_kernel_params_fn(initial_particles):
        return {
            "rw_scale": jnp.asarray(rw_step_size, dtype=jnp.float32),
            "proposal_cov": empirical_covariance(
                initial_particles,
                ridge=covariance_ridge,
                diagonal_only=diagonal_only_covariance,
            ),
        }

    def build_single_step_fns():
        def init_fn(position, logdensity_fn):
            return blackjax.rmh.init(position, logdensity_fn)

        def step_fn(keys, states, logdensity_fn, kernel_params):
            def one(key, state):
                transition_generator = blackjax.mcmc.random_walk.normal(
                    proposal_sqrt_from_cov(
                        kernel_params["proposal_cov"],
                        kernel_params["rw_scale"],
                    )
                )
                return base_rmh_kernel(
                    key,
                    state,
                    logdensity_fn,
                    transition_generator=transition_generator,
                )

            return jax.vmap(one)(keys, states)

        return init_fn, step_fn

    def adapt_inner(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)
        proposed_rw_scale = update_rw_scale_robbins_monro(
            rw_scale=new_params["rw_scale"],
            acceptance_value=acceptance_value,
            target_acceptance_rate=target_acceptance_rate,
            t=t,
            rm_c=rm_c,
            rm_t0=rm_t0,
            rm_kappa=rm_kappa,
        )
        new_params["rw_scale"] = jnp.where(
            jnp.isfinite(acceptance_value),
            proposed_rw_scale,
            new_params["rw_scale"],
        )
        return new_params

    def adapt_outer(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)
        new_params["proposal_cov"] = empirical_covariance(
            particles,
            ridge=covariance_ridge,
            diagonal_only=diagonal_only_covariance,
        )
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return 0

    return run_ps_with_inner_adaptation_once(
        target=target,
        num_particles=num_particles,
        seed=seed,
        max_iterations=max_iterations,
        alpha=alpha,
        num_mcmc_steps=num_mcmc_steps,
        build_single_step_fns=build_single_step_fns,
        init_kernel_params_fn=init_kernel_params_fn,
        adapt_kernel_params_inner_fn=adapt_inner,
        adapt_kernel_params_outer_fn=adapt_outer,
        gradient_eval_increment_fn=gradient_eval_increment_fn,
        kernel_name="rwm",
    )


# Optional alias for consistency with future files like ps_mala_gpu.py.
def run_once(**kwargs) -> dict[str, Any]:
    return run_ps_rwm_once(**kwargs)


if __name__ == "__main__":
    print(debug_worker_backend())
    out = run_ps_rwm_once(
        dimension=5,
        num_particles=1024,
        seed=0,
        max_iterations=10_000,
        alpha=0.999,
        num_mcmc_steps=10,
        rw_step_size=0.5,
    )
    print("logZ:", out["logZ"])
    print("n_iter:", out["n_iter"])
    print("final_ess:", out["final_ess"])
    print("acceptance_rate_mean:", out["acceptance_rate_mean"])
    print("runtime_sec:", out["runtime_sec"])
