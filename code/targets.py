# calculate target 
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import os
# Force CPU unless the environment already specified something else.
# This must happen before importing jax.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp

Array = jnp.ndarray #when Array is written it means JAX array

@dataclass(frozen=True) # cannot be modefiend after creation
class Target: # central container 
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
    #Log-density of a diag. Gauss. with diag. var
    d = x.shape[-1]
    return -0.5 * (jnp.sum((x - mean) ** 2 / var, axis=-1) + d * jnp.log(2.0 * jnp.pi * var))


def uniform_box_logpdf(theta: Array, bound: float) -> Array: 
    #Log-density of product Uniform[-bound, bound]^D function returns -inf if any coord. is outside [-bound, bound].
    inside = jnp.all((theta >= -bound) & (theta <= bound), axis=-1)
    log_norm_1d = -jnp.log(2.0 * bound) # density of one is 1/ (2 bound)
    log_norm = theta.shape[-1] * log_norm_1d
    return jnp.where(inside, log_norm, -jnp.inf)


def sample_uniform_box(key: jax.Array, num_samples: int, d: int, bound: float) -> Array:
    # to sample particles form the U[-bound, bound]^D
    return jax.random.uniform(key, shape=(num_samples, d), minval=-bound, maxval=bound)


def log_gaussian_isotropic(x: Array, mean: Array, std: float) -> Array:
    #same as log_gaussian_diag but for Rosenbrock prior.
    var = std ** 2
    d = x.shape[-1]
    return -0.5 * (jnp.sum((x - mean) ** 2, axis=-1) / var + d * jnp.log(2.0 * jnp.pi * var))


def sample_gaussian_isotropic(key: jax.Array, num_samples: int, d: int, std: float) -> Array:
    #sample from N(0, std^2 I)
    mean = jnp.zeros((num_samples, d))
    return mean + std * jax.random.normal(key, shape=(num_samples, d))


def log_standard_normal_vector(x: Array) -> Array:
    #log density of N(0, I), later used in logistic regression
    d = x.shape[-1]
    return -0.5 * (jnp.sum(x ** 2, axis=-1) + d * jnp.log(2.0 * jnp.pi))


def log_gamma_pdf_scalar(x: Array, shape: float, rate: float) -> Array:
    #log density of Gamma distribution with shape and rate parameters, used in logistic regression
    return jnp.where(x > 0.0, shape * jnp.log(rate) - jax.scipy.special.gammaln(shape) + (shape - 1.0) * jnp.log(x) - rate * x, -jnp.inf,)


def log_gamma_pdf_vector(x: Array, shape: float, rate: float) -> Array:
    #summed log density (assuming entries of x are independant Gamma variables).
    valid = jnp.all(x > 0.0, axis=-1)
    return jnp.where(valid, jnp.sum(shape * jnp.log(rate) - jax.scipy.special.gammaln(shape) + (shape - 1.0) * jnp.log(x) - rate * x, axis=-1), -jnp.inf,)


def standardize_columns(X: Array) -> Array:
    # data preprocessing function, to standardise colums. 
    mean = jnp.mean(X, axis=0, keepdims=True)
    std = jnp.std(X, axis=0, keepdims=True)
    std = jnp.where(std > 0.0, std, 1.0)
    return (X - mean) / std


# Experiment 1: Gaussian mixture target 
def make_gaussian_mixture_target(dimension: int) -> Target:
    # Jakob proposed first test to be a Gaussian mixture with:
    # prior  : Uniform[-4 sqrt(D), 4 sqrt(D)]^D
    # likelihood: (1/3) N(0, I) + (2/3) N(sqrt(D) * 1, 0.5 I)
    d = int(dimension)
    if d <= 0:
        raise ValueError(f"Gaussian mixture target requires a positive dimension, got dimension={dimension}.")

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
        return jax.scipy.special.logsumexp(jnp.stack([log_prob1, log_prob2], axis=-1), axis=-1, b=mixture_weights)

    def log_posterior_fn(theta: Array) -> Array:
        # posterior \propto likelihood * prior
        return log_prior_fn(theta) + log_likelihood_fn(theta)

    def sample_prior_fn(key: jax.Array, num_samples: int) -> Array:
        # to sample particles form the prior U[-bound, bound]^D
        return sample_uniform_box(key, num_samples, d, bound)

    # These exact formulas are for the Gaussian mixture itself.
    # Because your prior is truncated uniform, they are approximate sanity checks,
    def posterior_mean_reference_fn() -> Array:
        return (1.0 / 3.0) * mean1 + (2.0 / 3.0) * mean2

    def posterior_second_moment_reference_fn() -> Array:
        # E[X^2] = const * (mean^2 + var)
        second1 = jnp.ones(d) * 1.0
        second2 = jnp.ones(d) * (0.5 + float(d))
        return (1.0 / 3.0) * second1 + (2.0 / 3.0) * second2

    return Target(  name="gaussian_mixture",
                    dimension=d,
                    log_prior_fn=log_prior_fn,
                    log_likelihood_fn=log_likelihood_fn,
                    log_posterior_fn=log_posterior_fn,
                    sample_prior_fn=sample_prior_fn,
                    posterior_mean_exact_fn=posterior_mean_reference_fn,
                    posterior_second_moment_exact_fn=posterior_second_moment_reference_fn,
                    logZ_exact_fn=None,)


# Experiment 2: Rosenbrock target
def make_rosenbrock_target(dimension: int) -> Target:
    # Dimension-parameterized Rosenbrock target using adjacent coordinate pairs:
    # log L(theta) = -sum_i [10 (theta_{2i-1}^2 - theta_{2i})^2 + (theta_{2i-1} - 1)^2]
    # prior N(0, 5^2 I)
    d = int(dimension)

    if d <= 0:
        raise ValueError(f"Rosenbrock target requires a positive dimension, got dimension={dimension}.")
    if d % 2 != 0:
        raise ValueError(f"Rosenbrock target takes only an even dimension, you tried dimension={d}.")

    prior_std = 5.0
    prior_mean = jnp.zeros(d)

    def log_prior_fn(theta: Array) -> Array:
        return log_gaussian_isotropic(theta, prior_mean, prior_std)

    def log_likelihood_fn(theta: Array) -> Array:
        odd = theta[..., 0::2]   # theta_1, theta_3, ...
        even = theta[..., 1::2]  # theta_2, theta_4, ...
        terms = 10.0 * (odd ** 2 - even) ** 2 + (odd - 1.0) ** 2
        return -jnp.sum(terms, axis=-1)

    def log_posterior_fn(theta: Array) -> Array:
        return log_prior_fn(theta) + log_likelihood_fn(theta)

    def sample_prior_fn(key: jax.Array, num_samples: int) -> Array:
        return sample_gaussian_isotropic(key, num_samples, d, prior_std)

    return Target(
        name="rosenbrock",
        dimension=d,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        sample_prior_fn=sample_prior_fn,
        posterior_mean_exact_fn=None,
        posterior_second_moment_exact_fn=None,
        logZ_exact_fn=None,)


# Experiment 3: Sparse logistic regression target
def make_sparse_logistic_regression_target(X: Array, y: Array, *, add_intercept: bool = True, standardize: bool = True, ) -> Target:
    #  Likelihood:  L(y | beta, lambda, tau) = prod_i Bernoulli(y_i | sigmoid(((tau * lambda * beta)^T X_i)))
    # Priors: tau = Gamma(1/2, 1/2), beta_j = N(0, 1), lambda_j = Gamma(1/2, 1/2)
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of rows, got {X.shape[0]} and {y.shape[0]}.")
    unique_y = jnp.unique(y)
    if not jnp.all((unique_y == 0.0) | (unique_y == 1.0)):
        raise ValueError("y must contain only binary values 0 and 1.")
    if standardize:
        X = standardize_columns(X)
    if add_intercept:
        X = jnp.concatenate([jnp.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)
    _, p = X.shape
    d = 1 + p + p   # u + beta + v

    def unpack_theta(theta: Array) -> tuple[Array, Array, Array]:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        u = theta[0]
        beta = theta[1:1 + p]
        v = theta[1 + p:1 + 2 * p]
        return u, beta, v

    def constrained_parameters(theta: Array) -> tuple[Array, Array, Array]:
        u, beta, v = unpack_theta(theta)
        # light clipping for numerical stability in float64
        u = jnp.clip(u, -40.0, 40.0)
        v = jnp.clip(v, -40.0, 40.0)
        tau = jnp.exp(u)
        lam = jnp.exp(v)
        return tau, beta, lam

    def log_prior_fn(theta: Array) -> Array:
        u, beta, v = unpack_theta(theta)
        # use the same stabilized transformation here as well
        u_clip = jnp.clip(u, -40.0, 40.0)
        v_clip = jnp.clip(v, -40.0, 40.0)
        tau = jnp.exp(u_clip)
        lam = jnp.exp(v_clip)
        logp_tau = log_gamma_pdf_scalar(tau, shape=0.5, rate=0.5)
        logp_beta = log_standard_normal_vector(beta)
        logp_lam = log_gamma_pdf_vector(lam, shape=0.5, rate=0.5)
        log_jacobian = u_clip + jnp.sum(v_clip) #jacobian of the transformation from (u, beta, v) to (tau, beta, lam)
        return logp_tau + logp_beta + logp_lam + log_jacobian

    def log_likelihood_fn(theta: Array) -> Array:
        tau, beta, lam = constrained_parameters(theta)
        weights = tau * lam * beta
        logits = X @ weights

        return jnp.sum(y * logits - jnp.logaddexp(0.0, logits)) # standard stable Bernoulli log-likelihood:

    def log_posterior_fn(theta: Array) -> Array:
        return log_prior_fn(theta) + log_likelihood_fn(theta)

    def sample_prior_fn(key: jax.Array, num_samples: int) -> Array:
        key_tau, key_beta, key_lam = jax.random.split(key, 3)
        tau = jax.random.gamma(key_tau, 0.5, shape=(num_samples, 1)) / 0.5
        beta = jax.random.normal(key_beta, shape=(num_samples, p), dtype=jnp.float64)
        lam = jax.random.gamma(key_lam, 0.5, shape=(num_samples, p)) / 0.5
        tau = jnp.asarray(tau, dtype=jnp.float64)
        lam = jnp.asarray(lam, dtype=jnp.float64)
        # transform to unconstrained coordinates
        u = jnp.log(tau)
        v = jnp.log(lam)
        return jnp.concatenate([u, beta, v], axis=1)

    return Target(  name="sparse_logistic_regression",
                    dimension=d,
                    log_prior_fn=log_prior_fn,
                    log_likelihood_fn=log_likelihood_fn,
                    log_posterior_fn=log_posterior_fn,
                    sample_prior_fn=sample_prior_fn,
                    posterior_mean_exact_fn=None,
                    posterior_second_moment_exact_fn=None,
                    logZ_exact_fn=None,)


def make_target(name: str, dimension: int | None = None, **kwargs) -> Target:
    name = name.lower()
    if name == "gaussian_mixture":
        if dimension is None:
            raise ValueError("dimension is required for gaussian_mixture")
        return make_gaussian_mixture_target(dimension)
    if name == "rosenbrock":
        if dimension is None:
            raise ValueError("dimension is required for rosenbrock")
        return make_rosenbrock_target(dimension)
    if name == "sparse_logistic_regression":
        return make_sparse_logistic_regression_target(**kwargs)
    raise ValueError(f"Unknown target name: {name}")
