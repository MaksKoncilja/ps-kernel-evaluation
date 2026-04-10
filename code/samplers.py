from __future__ import annotations
from typing import Any, NamedTuple
import time
import os

# METAL seems to be problematic on my laptop, so keeping it on CPU is a good idea
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.diffusions as diffusions
from blackjax.smc import extend_params

from targets import Target, make_gaussian_mixture_target


def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    # tries to get scalar mean acceptance rate from BlackJAX kernel info
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
    # particle cov, with numerical stabilisation
    x = jnp.asarray(particles)
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
    # Cholesky decomposition multiplied by scale
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(t: int,c: float = 1.0,t0: float = 10.0,kappa: float = 0.6,) -> float:
    # Robbins-Monro diminishing adaptation scale factor: gamma_t = c / (t + t0)^kappa
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(rw_scale: jnp.ndarray,acceptance_value: float, target_acceptance_rate: float, 
                                    t: int, rm_c: float = 2.0, rm_t0: float = 1.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    # Robbins-Monro update for RW scale
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


def _update_step_size_robbins_monro(step_size: jnp.ndarray, acceptance_value: float,target_acceptance_rate: float,
                                    t: int, rm_c: float = 1.0,rm_t0: float = 1.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    # Robbins-Monro update for generic positive step size
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_step_size = jnp.log(step_size)
    log_step_size = log_step_size + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_step_size)


def _compute_posterior_moments_from_particles(particles: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    # f1(theta) = theta, f2(theta) = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def _normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    # Normalize the weights
    w = jnp.asarray(weights, dtype=jnp.float32)
    return w / jnp.sum(w)


def _variance_log_weights(weights: jnp.ndarray) -> float:
    # variance of normalized log weights
    w = _normalize_weights(weights)
    eps = 1e-32
    logw = jnp.log(w + eps)
    return float(jnp.var(logw))


def _weight_entropy(weights: jnp.ndarray) -> float:
    # Entropy of normalized weights: -sum_i w_i log w_i
    w = _normalize_weights(weights)
    eps = 1e-32
    return float(-jnp.sum(w * jnp.log(w + eps)))


def _population_esjd(before_particles: jnp.ndarray, after_particles: jnp.ndarray) -> float:
    # Particle population ESJD analogue across one PS iteration
    dx = jnp.asarray(after_particles) - jnp.asarray(before_particles)
    sqdist = jnp.sum(dx ** 2, axis=1)
    return float(jnp.mean(sqdist))


def _diagonal_inverse_mass_matrix(particles: jnp.ndarray, ridge: float = 1e-6,) -> jnp.ndarray:
    # Estimate diagonal covariance from particles and return its inverse diagonal
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0, ddof=1)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    var = var + ridge
    return 1.0 / var


def _run_ps_generic_once(*,
                        target: Target,
                        num_particles: int,
                        seed: int,
                        max_iterations: int,
                        alpha: float,
                        num_mcmc_steps: int,
                        build_ps_kernel_fn,
                        init_kernel_params_fn,
                        adapt_kernel_params_fn,
                        gradient_eval_increment_fn,
                        kernel_name: str,) -> dict[str, Any]:
    
    #Shared adaptive tempered PS loop 
    
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    kernel_params = init_kernel_params_fn(initial_particles)

    kernel = build_ps_kernel_fn(log_prior_fn=log_prior_fn,
                                log_likelihood_fn=log_likelihood_fn,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                kernel_params=kernel_params,)

    state = kernel.init(initial_particles)

    tempering_path, logZ_path, ess_path, acceptance_path, step_size_path = [], [], [], [], []
    elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path = [], [], [], []

    start = time.perf_counter()
    n_iter, gradient_eval_count, resampling_steps = 0, 0, 0

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, subkey = jax.random.split(loop_key)

        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles)

        kernel = build_ps_kernel_fn(
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            max_iterations=max_iterations,
            alpha=alpha,
            num_mcmc_steps=num_mcmc_steps,
            kernel_params=kernel_params,)

        state, info = kernel.step(subkey, state)

        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.persistent_weights)

        ess_value = float(1.0 / jnp.sum(_normalize_weights(weights) ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        elapsed = time.perf_counter() - start

        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)

        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z if hasattr(state, "log_Z") else state.persistent_log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)

        if "step_size" in kernel_params:
            step_size_path.append(float(kernel_params["step_size"]))
        elif "rw_scale" in kernel_params:
            step_size_path.append(float(kernel_params["rw_scale"]))
        else:
            step_size_path.append(np.nan)

        resampling_steps += 1
        # for NUTS, num_integration_steps is variable and can be read from update_info if available
        nuts_num_steps = getattr(update_info, "num_integration_steps", None)
        if nuts_num_steps is not None:
            gradient_eval_count += int(jnp.sum(jnp.asarray(nuts_num_steps)))
        else:
            gradient_eval_count += int(gradient_eval_increment_fn(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps, kernel_params=kernel_params,))
        kernel_params = adapt_kernel_params_fn(kernel_params=kernel_params, particles=particles, acceptance_value=acceptance_value, t=n_iter + 1,)
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles(final_particles)

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_array = np.asarray(acceptance_path, dtype=float) 
    finite_acceptance = acceptance_array[np.isfinite(acceptance_array)]
    acceptance_rate_last = float(finite_acceptance[-1]) if finite_acceptance.size > 0 else np.nan
    acceptance_rate_mean = (float(finite_acceptance.mean()) if finite_acceptance.size > 0 else np.nan)

    return {"target_name": target.name,
            "algorithm_name": "ps",
            "kernel_name": kernel_name,
            "seed": int(seed),
            "dimension": int(target.dimension),
            "num_particles": int(num_particles),

            "logZ": float(final_state.log_Z if hasattr(final_state, "log_Z") else state.log_Z if hasattr(state, "log_Z") else state.persistent_log_Z),
            "posterior_mean": posterior_mean,
            "posterior_second_moment": posterior_second_moment,

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
            "step_size_path": np.asarray(step_size_path, dtype=float)}



# =========================
# PS + RW
def run_ps_rwm_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    rw_step_size: float = 1.,
                    target_acceptance_rate: float = 0.234,
                    covariance_ridge: float = 1e-6,
                    diagonal_only_covariance: bool = False,
                    rm_c: float = 2.0,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:

    if target is None:
        target = make_gaussian_mixture_target(dimension)
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def rmh_step_fn(rng_key, state, logdensity_fn, rw_scale, proposal_cov):
        transition_generator = blackjax.mcmc.random_walk.normal(_proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel( rng_key, state, logdensity_fn, transition_generator=transition_generator, )

    def init_kernel_params_fn(initial_particles):
        return {"rw_scale": jnp.asarray(rw_step_size, dtype=jnp.float32),
                "proposal_cov": _empirical_covariance( initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,), }

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        mcmc_parameters = extend_params({"rw_scale": jnp.asarray(kernel_params["rw_scale"]), "proposal_cov": kernel_params["proposal_cov"],})

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=rmh_step_fn,
                                                        mcmc_init_fn=blackjax.rmh.init,
                                                        mcmc_parameters=mcmc_parameters,
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)

        if not np.isnan(acceptance_value):
            new_params["rw_scale"] = _update_rw_scale_robbins_monro(rw_scale=new_params["rw_scale"], acceptance_value=acceptance_value,
                                                                    target_acceptance_rate=target_acceptance_rate, t=t, rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)

        new_params["proposal_cov"] = _empirical_covariance(particles,ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return 0

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="rwm",)


# =========================
# PS + HMC
def run_ps_hmc_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 0.1,
                    num_integration_steps: int = 10,
                    target_acceptance_rate: float = 0.651,
                    mass_matrix_ridge: float = 1e-6,
                    rm_c: float = 2.0,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    """
    One run of adaptive tempered PS with HMC kernel.
    If target is None, fallback to the current Gaussian-mixture target.
    """
    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),
                "inverse_mass_matrix": _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge,),
                "num_integration_steps": int(num_integration_steps),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        hmc_parameters = {"step_size": jnp.asarray(kernel_params["step_size"]),
                        "inverse_mass_matrix": jnp.asarray(kernel_params["inverse_mass_matrix"]),
                        "num_integration_steps": int(kernel_params["num_integration_steps"]),}

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=blackjax.hmc.build_kernel(),
                                                        mcmc_init_fn=blackjax.hmc.init,
                                                        mcmc_parameters=extend_params(hmc_parameters),
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)

        if not np.isnan(acceptance_value):
            new_params["step_size"] = _update_step_size_robbins_monro(step_size=new_params["step_size"], acceptance_value=acceptance_value,
                                                                    target_acceptance_rate=target_acceptance_rate, t=t,rm_c=rm_c,rm_t0=rm_t0,rm_kappa=rm_kappa,)

        new_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(particles,ridge=mass_matrix_ridge,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        return int(num_particles) * int(num_mcmc_steps) * int(kernel_params["num_integration_steps"])

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="hmc",)


class _ULAState(NamedTuple):
    position: Any
    logdensity: float
    logdensity_grad: Any

# ===================
# PS + ULA (BlackJAX overdamped Langevin diffusion), has no MC correcttion -> no accept. rate
def run_ps_ula_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 5e-3,) -> dict[str, Any]:

    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def ula_init_fn(position, logdensity_fn):
        grad_fn = jax.value_and_grad(logdensity_fn)
        logdensity, logdensity_grad = grad_fn(position)
        return _ULAState(position, logdensity, logdensity_grad)

    def ula_step_fn(rng_key, state, logdensity_fn, step_size):
        grad_fn = jax.value_and_grad(logdensity_fn)
        one_step = diffusions.overdamped_langevin(grad_fn)
        new_state = one_step(rng_key, state, step_size)
        return _ULAState(*new_state), None

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        ula_parameters = extend_params({"step_size": jnp.asarray(kernel_params["step_size"]),})

        return blackjax.adaptive_persistent_sampling_smc(   logprior_fn=log_prior_fn,
                                                            loglikelihood_fn=log_likelihood_fn,
                                                            max_iterations=max_iterations,
                                                            mcmc_step_fn=ula_step_fn,
                                                            mcmc_init_fn=ula_init_fn,
                                                            mcmc_parameters=ula_parameters,
                                                            resampling_fn=resampling.systematic,
                                                            target_ess=alpha,
                                                            num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        # ULA is unadjusted, lets keep the step size fixed 
        return dict(kernel_params)

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        # we have one gradient evaluation per ULA step, per particle
        return int(num_particles) * int(num_mcmc_steps)

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="ula",)
# =================
# PS + MALA
def run_ps_mala_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 0.02,
                    target_acceptance_rate: float = 0.574,
                    rm_c: float = 2.0,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    # BlackJAX's MALA and Robbins-Monro diminishing adaptation toward target acceptance 0.574.

    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        mala_parameters = extend_params({"step_size": jnp.asarray(kernel_params["step_size"]), })

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=blackjax.mala.build_kernel(),
                                                        mcmc_init_fn=blackjax.mala.init,
                                                        mcmc_parameters=mala_parameters,
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)

        if not np.isnan(acceptance_value):
            new_params["step_size"] = _update_step_size_robbins_monro(step_size=new_params["step_size"], acceptance_value=acceptance_value,
                                                                      target_acceptance_rate=target_acceptance_rate,
                                                                      t=t,rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        # One gradient evaluation per MALA step, per particle
        return int(num_particles) * int(num_mcmc_steps)

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="mala",)

# =================
# PS + NUTS
def run_ps_nuts_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 0.1,
                    max_num_doublings: int = 10,
                    mass_matrix_ridge: float = 1e-6,
                    target_acceptance_rate: float = 0.8,
                    rm_c: float = 3.5,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    # BlackJAX's NUTS with diagonal inverse mass matrix.
    # We keep step size fixed in this first version and only adapt geometry.

    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),
                "inverse_mass_matrix": _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge,),
                "max_num_doublings": int(max_num_doublings),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        nuts_parameters = {"step_size": jnp.asarray(kernel_params["step_size"]),
                           "inverse_mass_matrix": jnp.asarray(kernel_params["inverse_mass_matrix"]),
                           "max_num_doublings": int(kernel_params["max_num_doublings"]),}

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=blackjax.nuts.build_kernel(),
                                                        mcmc_init_fn=blackjax.nuts.init,
                                                        mcmc_parameters=extend_params(nuts_parameters),
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        new_params = dict(kernel_params)
        if not np.isnan(acceptance_value):
            new_params["step_size"] = _update_step_size_robbins_monro(step_size=new_params["step_size"], acceptance_value=acceptance_value,
                                                                      target_acceptance_rate=target_acceptance_rate, t=t, rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)

        new_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(particles, ridge=mass_matrix_ridge,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        # fallback only; actual variable integration steps are handled in _run_ps_generic_once
        return 0

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="nuts",)


# še ne razumem dobro, zato ne uporabljam.
"""
# =================
# PS + MCLMC
def run_ps_mclmc_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    target: Target | None = None,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 1e-2,
                    L: float = 1.0,
                    mass_matrix_ridge: float = 1e-6,) -> dict[str, Any]:
    # BlackJAX's MCLMC with diagonal inverse mass matrix.
    # First simple version: keep step_size and L fixed, only update diagonal geometry.

    if target is None:
        target = make_gaussian_mixture_target(dimension)

    def init_kernel_params_fn(initial_particles):
        return {"step_size": jnp.asarray(step_size, dtype=jnp.float32),
                "L": float(L),
                "inverse_mass_matrix": _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge,),}

    def build_ps_kernel_fn(*, log_prior_fn, log_likelihood_fn, max_iterations, alpha, num_mcmc_steps, kernel_params):
        mclmc_parameters = {"step_size": jnp.asarray(kernel_params["step_size"]),
                            "L": float(kernel_params["L"]),
                            "inverse_mass_matrix": jnp.asarray(kernel_params["inverse_mass_matrix"]),}

        return blackjax.adaptive_persistent_sampling_smc(logprior_fn=log_prior_fn,
                                                        loglikelihood_fn=log_likelihood_fn,
                                                        max_iterations=max_iterations,
                                                        mcmc_step_fn=blackjax.mclmc.build_kernel(),
                                                        mcmc_init_fn=blackjax.mclmc.init,
                                                        mcmc_parameters=extend_params(mclmc_parameters),
                                                        resampling_fn=resampling.systematic,
                                                        target_ess=alpha,
                                                        num_mcmc_steps=num_mcmc_steps,)

    def adapt_kernel_params_fn(*, kernel_params, particles, acceptance_value, t):
        # first MCLMC version: keep step_size and L fixed, only update diagonal geometry
        new_params = dict(kernel_params)
        new_params["inverse_mass_matrix"] = _diagonal_inverse_mass_matrix(particles, ridge=mass_matrix_ridge,)
        return new_params

    def gradient_eval_increment_fn(*, num_particles, num_mcmc_steps, kernel_params):
        # crude first counter; may be refined later if MCLMC info exposes more detailed effort stats
        return int(num_particles) * int(num_mcmc_steps)

    return _run_ps_generic_once(target=target,
                                num_particles=num_particles,
                                seed=seed,
                                max_iterations=max_iterations,
                                alpha=alpha,
                                num_mcmc_steps=num_mcmc_steps,
                                build_ps_kernel_fn=build_ps_kernel_fn,
                                init_kernel_params_fn=init_kernel_params_fn,
                                adapt_kernel_params_fn=adapt_kernel_params_fn,
                                gradient_eval_increment_fn=gradient_eval_increment_fn,
                                kernel_name="mclmc",)
"""




# first implementation of the first two combinations 
"""
# first implementation pf the Rw and HMC kernels
def _safe_mean_acceptance(update_info) -> jnp.ndarray:
    # tries to get scalar mean acceptance rate from BlackJAX kernel info
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
    #particle cov, with numerical stabilistation | we need that for Robbins-Monro adaptation
    x = jnp.asarray(particles)
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
    #Cholesky decomposition multiplied by scale | also for Robbins-Monro adaptation
    chol = jnp.linalg.cholesky(cov)
    return scale * chol


def _robbins_monro_step_size(t: int, c: float = 1.0, t0: float = 10.0, kappa: float = 0.6, ) -> float:
    # Robbins-Monro diminishing adaptation scale factor: gamma_t = c / (t + t0)^kappa
    return float(c / ((t + t0) ** kappa))


def _update_rw_scale_robbins_monro(rw_scale: jnp.ndarray, acceptance_value: float, target_acceptance_rate: float, t: int, 
                                   rm_c: float = 1.0, rm_t0: float = 1.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    # Robbins-Monro update
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_rw_scale = jnp.log(rw_scale)
    log_rw_scale = log_rw_scale + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_rw_scale)


def _compute_posterior_moments_from_particles(particles: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    # f1(theta) = theta, f2(theta) = theta^2
    f1 = particles.mean(axis=0)
    f2 = (particles ** 2).mean(axis=0)
    return f1, f2


def _normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    # Normalize the weights
    w = jnp.asarray(weights, dtype=jnp.float32)
    return w / jnp.sum(w)


def _variance_log_weights(weights: jnp.ndarray) -> float:
    # calculate the variance of log weights (normalized)
    w = _normalize_weights(weights)
    eps = 1e-32
    logw = jnp.log(w + eps)
    return float(jnp.var(logw))


def _weight_entropy(weights: jnp.ndarray) -> float:
    # Entropy of weights (normalized): -sum_i w_i log w_i
    w = _normalize_weights(weights)
    eps = 1e-32 # stabilizator
    return float(-jnp.sum(w * jnp.log(w + eps)))


def _population_esjd(before_particles: jnp.ndarray, after_particles: jnp.ndarray) -> float:
    # Particle population ESJD (analogue) across one iteration of PS: mean_i || after_i - before_i ||^2
    dx = jnp.asarray(after_particles) - jnp.asarray(before_particles)
    sqdist = jnp.sum(dx ** 2, axis=1)
    return float(jnp.mean(sqdist))


# =========================
# Result class

@dataclass
class PSSamplerRunResult:
    target_name: str
    algorithm_name: str
    kernel_name: str
    seed: int
    dimension: int
    num_particles: int
    logZ: float
    posterior_mean: np.ndarray
    posterior_second_moment: np.ndarray
    final_ess: float
    acceptance_rate_mean: float
    n_iter: int
    runtime_sec: float
    tempering_path: np.ndarray
    logZ_path: np.ndarray
    ess_path: np.ndarray
    acceptance_path: np.ndarray
    elapsed_time_path: np.ndarray
    gradient_eval_count: int
    resampling_steps: int
    variance_log_weights_path: np.ndarray
    weight_entropy_path: np.ndarray
    esjd_path: np.ndarray


# =========================
# PS + RW 
def run_ps_rwm_once(dimension: int,
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
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    # PS + RW
    target = make_gaussian_mixture_target(dimension)
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    rw_scale = jnp.asarray(rw_step_size, dtype=jnp.float32)
    proposal_cov = _empirical_covariance(initial_particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)
    base_rmh_kernel = blackjax.rmh.build_kernel()

    def rmh_step_fn(rng_key, state, logdensity_fn, rw_scale, proposal_cov):
        transition_generator = blackjax.mcmc.random_walk.normal(_proposal_sqrt_from_cov(proposal_cov, rw_scale))
        return base_rmh_kernel(rng_key, state, logdensity_fn, transition_generator=transition_generator,)

    def build_ps_kernel(current_rw_scale, current_proposal_cov):
        mcmc_parameters = extend_params({"rw_scale": jnp.asarray(current_rw_scale), "proposal_cov": current_proposal_cov,})

        return blackjax.adaptive_persistent_sampling_smc(   logprior_fn=log_prior_fn,
                                                            loglikelihood_fn=log_likelihood_fn,
                                                            max_iterations=max_iterations,
                                                            mcmc_step_fn=rmh_step_fn,
                                                            mcmc_init_fn=blackjax.rmh.init,
                                                            mcmc_parameters=mcmc_parameters,
                                                            resampling_fn=resampling.systematic,
                                                            target_ess=alpha,
                                                            num_mcmc_steps=num_mcmc_steps,)

    kernel = build_ps_kernel(rw_scale, proposal_cov)
    state = kernel.init(initial_particles)
    tempering_path, logZ_path, ess_path, acceptance_path, elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path  = [],[],[],[],[],[],[],[]
    start = time.perf_counter()
    n_iter = 0
    gradient_eval_count = 0  # RW uses no gradients
    resampling_steps = 0

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, subkey = jax.random.split(loop_key)
        pre_step_particles = np.asarray(blackjax.persistent_sampling.remove_padding(state).particles) # for ESJD approximation
        kernel = build_ps_kernel(rw_scale, proposal_cov) # rebuild kernel with current adapted parameters
        state, info = kernel.step(subkey, state)
        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.weights)
        ess_value = float(1.0 / jnp.sum(weights ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        elapsed = time.perf_counter() - start
        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)
        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)

        resampling_steps += 1 # kernel step is one resampling/move iteration.
        if not np.isnan(acceptance_value): # Robbins-Monro diminishing adaptation of RW scale
            rw_scale = _update_rw_scale_robbins_monro( rw_scale=rw_scale, acceptance_value=acceptance_value, target_acceptance_rate=target_acceptance_rate, 
                                                      t=n_iter + 1, rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)

        # update proposal covariance from current particle cloud
        proposal_cov = _empirical_covariance(particles, ridge=covariance_ridge, diagonal_only=diagonal_only_covariance,)
        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles( final_particles )

    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_rate_mean = (float(np.nanmean(np.asarray(acceptance_path, dtype=float))) if len(acceptance_path) > 0 else np.nan )

    result = PSSamplerRunResult(target_name=target.name,
                                algorithm_name="ps",
                                kernel_name="rwm",
                                seed=int(seed),
                                dimension=int(dimension),
                                num_particles=int(num_particles),
                                logZ=float(final_state.log_Z),
                                posterior_mean=posterior_mean,
                                posterior_second_moment=posterior_second_moment,
                                final_ess=final_ess,
                                acceptance_rate_mean=acceptance_rate_mean,
                                n_iter=int(n_iter),
                                runtime_sec=float(runtime_sec),
                                tempering_path=np.asarray(tempering_path, dtype=float),
                                logZ_path=np.asarray(logZ_path, dtype=float),
                                ess_path=np.asarray(ess_path, dtype=float),
                                acceptance_path=np.asarray(acceptance_path, dtype=float),
                                elapsed_time_path=np.asarray(elapsed_time_path, dtype=float),
                                gradient_eval_count=int(gradient_eval_count),
                                resampling_steps=int(resampling_steps),
                                variance_log_weights_path=np.asarray(variance_log_weights_path, dtype=float),
                                weight_entropy_path=np.asarray(weight_entropy_path, dtype=float),
                                esjd_path=np.asarray(esjd_path, dtype=float),)

    return {"target_name": result.target_name,
            "algorithm_name": result.algorithm_name,
            "kernel_name": result.kernel_name,
            "seed": result.seed,
            "dimension": result.dimension,
            "num_particles": result.num_particles,

            "logZ": result.logZ,
            "posterior_mean": result.posterior_mean,
            "posterior_second_moment": result.posterior_second_moment,

            "final_ess": result.final_ess,
            "acceptance_rate_mean": result.acceptance_rate_mean,
            "n_iter": result.n_iter,
            "runtime_sec": result.runtime_sec,

            "tempering_path": result.tempering_path,
            "logZ_path": result.logZ_path,
            "ess_path": result.ess_path,
            "acceptance_path": result.acceptance_path,
            "elapsed_time_path": result.elapsed_time_path,
            "gradient_eval_count": result.gradient_eval_count,
            "resampling_steps": result.resampling_steps,
            "variance_log_weights_path": result.variance_log_weights_path,
            "weight_entropy_path": result.weight_entropy_path,
            "esjd_path": result.esjd_path,}


def _update_step_size_robbins_monro( step_size: jnp.ndarray, acceptance_value: float, target_acceptance_rate: float,
                                     t: int, rm_c: float = 1.0, rm_t0: float = 1.0, rm_kappa: float = 0.6,) -> jnp.ndarray:
    gamma_t = _robbins_monro_step_size(t=t, c=rm_c, t0=rm_t0, kappa=rm_kappa)
    log_step_size = jnp.log(step_size)
    log_step_size = log_step_size + gamma_t * (acceptance_value - target_acceptance_rate)
    return jnp.exp(log_step_size)


def _diagonal_inverse_mass_matrix( particles: jnp.ndarray, ridge: float = 1e-6, ) -> jnp.ndarray:
    # Estimate diagonal covariance from particles and return its inverse diagonal.
    x = jnp.asarray(particles)
    var = jnp.var(x, axis=0, ddof=1)
    var = jnp.where(jnp.isfinite(var), var, 0.0)
    var = var + ridge
    return 1.0 / var

# =========================
# PS + HMC

def run_ps_hmc_once(dimension: int,
                    num_particles: int,
                    seed: int,
                    max_iterations: int = 10_000,
                    alpha: float = 0.999,
                    num_mcmc_steps: int = 25,
                    step_size: float = 1e-2,
                    num_integration_steps: int = 10,
                    target_acceptance_rate: float = 0.651,
                    mass_matrix_ridge: float = 1e-6,
                    rm_c: float = 1.0,
                    rm_t0: float = 1.0,
                    rm_kappa: float = 0.6,) -> dict[str, Any]:
    
    # Adaptive tempered PS + HMC with diagonal inverse mass matrix updated from particles, Robbins-Monro diminishing adaptation of HMC step size
    target = make_gaussian_mixture_target(dimension)
    log_prior_fn = target.log_prior_fn
    log_likelihood_fn = target.log_likelihood_fn
    key = jax.random.PRNGKey(seed)
    key, init_key, loop_key = jax.random.split(key, 3)
    initial_particles = target.sample_prior_fn(init_key, num_particles)
    step_size = jnp.asarray(step_size, dtype=jnp.float32)
    inverse_mass_matrix = _diagonal_inverse_mass_matrix(initial_particles, ridge=mass_matrix_ridge,)

    def build_ps_kernel(current_step_size, current_inverse_mass_matrix):
        hmc_parameters = { "step_size": jnp.asarray(current_step_size), "inverse_mass_matrix": jnp.asarray(current_inverse_mass_matrix), "num_integration_steps": int(num_integration_steps),}

        return blackjax.adaptive_persistent_sampling_smc(   logprior_fn=log_prior_fn,
                                                            loglikelihood_fn=log_likelihood_fn,
                                                            max_iterations=max_iterations,
                                                            mcmc_step_fn=blackjax.hmc.build_kernel(),
                                                            mcmc_init_fn=blackjax.hmc.init,
                                                            mcmc_parameters=extend_params(hmc_parameters),
                                                            resampling_fn=resampling.systematic,
                                                            target_ess=alpha,
                                                            num_mcmc_steps=num_mcmc_steps,)

    kernel = build_ps_kernel(step_size, inverse_mass_matrix)
    state = kernel.init(initial_particles)

    tempering_path, logZ_path, ess_path, acceptance_path, elapsed_time_path, variance_log_weights_path, weight_entropy_path, esjd_path = [],[],[],[],[],[],[],[]

    start = time.perf_counter()
    n_iter, gradient_eval_count, resampling_steps = 0, 0, 0

    while float(state.tempering_param) < 1.0 and n_iter < max_iterations:
        loop_key, subkey = jax.random.split(loop_key)
        state_unpadded_before = blackjax.persistent_sampling.remove_padding(state)
        pre_step_particles = np.asarray(state_unpadded_before.particles)
        kernel = build_ps_kernel(step_size, inverse_mass_matrix) # addapt the HMC parameters
        state, info = kernel.step(subkey, state)
        state_unpadded = blackjax.persistent_sampling.remove_padding(state)
        particles = jnp.asarray(state_unpadded.particles)
        weights = jnp.asarray(state.weights)
        ess_value = float(1.0 / jnp.sum(weights ** 2))
        update_info = getattr(info, "update_info", None)
        acceptance_value = float(_safe_mean_acceptance(update_info))
        elapsed = time.perf_counter() - start
        post_step_particles = np.asarray(particles)
        esjd_value = _population_esjd(pre_step_particles, post_step_particles)
        tempering_path.append(float(state.tempering_param))
        logZ_path.append(float(state.log_Z))
        ess_path.append(ess_value)
        acceptance_path.append(acceptance_value)
        elapsed_time_path.append(float(elapsed))
        variance_log_weights_path.append(_variance_log_weights(weights))
        weight_entropy_path.append(_weight_entropy(weights))
        esjd_path.append(esjd_value)

        resampling_steps += 1

        # one HMC transition ~ num_integration_steps gradient evals per particle, per move step, repeated num_mcmc_steps times
        gradient_eval_count += (int(num_particles) * int(num_mcmc_steps) * int(num_integration_steps))

        if not np.isnan(acceptance_value):
            step_size = _update_step_size_robbins_monro(step_size=step_size, acceptance_value=acceptance_value,
                                                        target_acceptance_rate=target_acceptance_rate, t=n_iter + 1, rm_c=rm_c, rm_t0=rm_t0, rm_kappa=rm_kappa,)

        inverse_mass_matrix = _diagonal_inverse_mass_matrix(particles, ridge=mass_matrix_ridge,)

        n_iter += 1

    runtime_sec = time.perf_counter() - start
    final_state = blackjax.persistent_sampling.remove_padding(state)
    final_particles = np.asarray(final_state.particles)
    posterior_mean, posterior_second_moment = _compute_posterior_moments_from_particles(final_particles)
    final_ess = float(ess_path[-1]) if len(ess_path) > 0 else np.nan
    acceptance_rate_mean = (float(np.nanmean(np.asarray(acceptance_path, dtype=float))) if len(acceptance_path) > 0 else np.nan )

    return {"target_name": target.name,
            "algorithm_name": "ps",
            "kernel_name": "hmc",
            "seed": int(seed),
            "dimension": int(dimension),
            "num_particles": int(num_particles),

            "logZ": float(final_state.log_Z),
            "posterior_mean": posterior_mean,
            "posterior_second_moment": posterior_second_moment,

            "final_ess": final_ess,
            "acceptance_rate_mean": acceptance_rate_mean,
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
            "esjd_path": np.asarray(esjd_path, dtype=float),}
"""

