# calculate target 

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp

Array = jnp.ndarray

@dataclass(frozen=True) # cannot be modefiend after creation
class Target:
    name: str
    dimension: int
    log_prior_fn: Callable[[Array], Array] # finctions take and return arrays
    log_likelihood_fn: Callable[[Array], Array]
    log_posterior_fn: Callable[[Array], Array]
    sample_prior_fn: Callable[[jax.Array, int], Array]
    posterior_mean_exact_fn: Callable[[], Array] | None = None # sometimes empty fields
    posterior_second_moment_exact_fn: Callable[[], Array] | None = None
    logZ_exact_fn: Callable[[], float] | None = None


def log_gaussian_diag(x: Array, mean: Array, var: float) -> Array:
    #Log-density of a diag. Gauss. with scalar var.
    d = x.shape[-1]
    return -0.5 * ( jnp.sum((x - mean) ** 2 / var) + d * jnp.log(2.0 * jnp.pi * var) )


def uniform_box_logpdf(theta: Array, bound: float) -> Array: 
    #Log-density of product Uniform[-bound, bound]^D function returns -inf if any coord. is outside [-bound, bound].
    inside = jnp.all((theta >= -bound) & (theta <= bound))
    log_norm_1d = -jnp.log(2.0 * bound) # density of one is 1/ (2 bound)
    log_norm = theta.shape[-1] * log_norm_1d
    return jnp.where(inside, log_norm, -jnp.inf)


def sample_uniform_box(key: jax.Array, num_samples: int, d: int, bound: float) -> Array:
    # to sample particles form the U[-bound, bound]^D
    return jax.random.uniform( key, shape=(num_samples, d), minval=-bound, maxval=bound,)


# Experiment 1: Gaussian mixture target 
def make_gaussian_mixture_target(dimension: int) -> Target:
    # Jakob proposed first test to be a Gaussian mixture with:
    # prior  : Uniform[-4 sqrt(D), 4 sqrt(D)]^D
    # likelihood: (1/3) N(0, I) + (2/3) N(sqrt(D) * 1, 0.5 I)
    d = int(dimension)
    bound = 4.0 * jnp.sqrt(float(d))
    mean1 = jnp.zeros(d)
    mean2 = jnp.sqrt(float(d)) * jnp.ones(d)
    mixture_weights = jnp.array([1.0 / 3.0, 2.0 / 3.0])

    def log_prior_fn(theta: Array) -> Array:
        # pi(theta) = U[-bound, bound]^{D}
        return uniform_box_logpdf(theta, bound)

    def log_likelihood_fn(theta: Array) -> Array:
        # seting up (1/3) N(0, I) + (2/3) N(sqrt(D) * 1, 0.5 I)
        log_prob1 = log_gaussian_diag(theta, mean1, 1.0)
        log_prob2 = log_gaussian_diag(theta, mean2, 0.5)
        return jax.scipy.special.logsumexp(jnp.array([log_prob1, log_prob2]), b=mixture_weights,)

    def log_posterior_fn(theta: Array) -> Array:
        # posterior \propto likelihood * prior
        return log_prior_fn(theta) + log_likelihood_fn(theta)

    def sample_prior_fn(key: jax.Array, num_samples: int) -> Array:
    # to sample particles form the prior U[-bound, bound]^D
        return sample_uniform_box(key, num_samples, d, bound)

    # These exact formulas are for the Gaussian mixture itself.
    # Because your prior is truncated uniform, they are approximate sanity checks,
    # not exact posterior formulas for the truncated posterior.
    def posterior_mean_exact_fn() -> Array:
        return (1.0 / 3.0) * mean1 + (2.0 / 3.0) * mean2

    def posterior_second_moment_exact_fn() -> Array:
        # E[X^2] = const * (mean^2 + var)
        second1 = jnp.ones(d) * 1.0
        second2 = jnp.ones(d) * (0.5 + float(d))
        return (1.0 / 3.0) * second1 + (2.0 / 3.0) * second2

    return Target(
        name="gaussian_mixture",
        dimension=d,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        sample_prior_fn=sample_prior_fn,
        posterior_mean_exact_fn=posterior_mean_exact_fn,
        posterior_second_moment_exact_fn=posterior_second_moment_exact_fn,
        logZ_exact_fn=None,)