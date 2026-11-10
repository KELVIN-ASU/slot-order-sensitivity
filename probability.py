"""Gaussian and period-pi wrapped-normal probabilities and scoring rules."""

import math

import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import norm

PERIOD = math.pi
ORIGIN = -math.pi / 2
ALIAS_RADIUS = 7


# Return signed shortest displacements on a periodic support
def wrap(values, period=PERIOD):
    return (values + period / 2) % period - period / 2


# Compute linear residuals and shortest circular hue residuals.
def residual_matrix(mu, target):
    residual = np.asarray(mu, dtype=np.float64) - target
    residual[..., 7:] = wrap(residual[..., 7:])
    return residual


# Return the expected absolute value of a Gaussian random variable.
def absolute_normal(mean, variance):
    sigma = np.sqrt(np.maximum(variance, 1e-15))
    standardized = mean / sigma
    density = np.exp(-0.5 * standardized**2) / math.sqrt(2 * math.pi)
    return 2 * sigma * density + mean * (2 * ndtr(standardized) - 1)


# Integrate geodesic distance under a wrapped normal using aliases or harmonics.
def expected_circular_distance(center, variance):
    center, variance = np.broadcast_arrays(wrap(np.asarray(center)), variance)
    center = center.astype(np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError("Variance must be positive and finite.")
    output = np.zeros_like(center)
    small = np.sqrt(variance) < PERIOD / 2
    mean = center[small]
    sigma = np.sqrt(variance[small])
    for alias in range(-ALIAS_RADIUS, ALIAS_RADIUS + 1):
        shifted = mean + alias * PERIOD
        lower = (-PERIOD / 2 - shifted) / sigma
        middle = -shifted / sigma
        upper = (PERIOD / 2 - shifted) / sigma
        positive = shifted * (ndtr(upper) - ndtr(middle))
        positive += sigma * (norm.pdf(middle) - norm.pdf(upper))
        negative = shifted * (ndtr(middle) - ndtr(lower))
        negative += sigma * (norm.pdf(lower) - norm.pdf(middle))
        output[small] += positive - negative
    harmonics = np.arange(1, 16, 2, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    damping = np.exp(-0.5 * variance[~small, None] * frequencies**2)
    phase = center[~small, None] * frequencies
    terms = (damping * np.cos(phase)) / harmonics**2
    output[~small] = PERIOD / 4 - 2 * PERIOD / math.pi**2 * terms.sum(-1)
    return output


# Integrate the two geodesic-distance expectations independently of CRPS harmonics.
def circular_crps_reference(mu, variance, target):
    first = expected_circular_distance(mu - target, variance)
    second = expected_circular_distance(np.zeros_like(mu), 2 * variance)
    return first - 0.5 * second


# Integrate squared circular distance for standardized uncertainty diagnostics.
def expected_circular_squared_distance(center, variance):
    center, variance = np.broadcast_arrays(wrap(np.asarray(center)), variance)
    variance = np.asarray(variance, dtype=np.float64)
    output = np.zeros_like(center, dtype=np.float64)
    small = np.sqrt(variance) < PERIOD / 2
    sigma = np.sqrt(variance[small])
    for alias in range(-ALIAS_RADIUS, ALIAS_RADIUS + 1):
        mean = center[small] + alias * PERIOD
        lower = (-PERIOD / 2 - mean) / sigma
        upper = (PERIOD / 2 - mean) / sigma
        values = (mean**2 + sigma**2) * (ndtr(upper) - ndtr(lower))
        values += 2 * mean * sigma * (norm.pdf(lower) - norm.pdf(upper))
        values += sigma**2 * (lower * norm.pdf(lower) - upper * norm.pdf(upper))
        output[small] += values
    harmonics = np.arange(1, 9, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    damping = np.exp(-0.5 * variance[~small, None] * frequencies**2)
    phase = center[~small, None] * frequencies
    terms = (-1) ** harmonics * damping * np.cos(phase) / harmonics**2
    output[~small] = PERIOD**2 / 12 + PERIOD**2 / math.pi**2 * terms.sum(-1)
    return np.maximum(output, 1e-15)


# Evaluate geodesic CRPS using the odd harmonics of circular distance.
def circular_crps_fourier(mu, variance, target, terms=512):
    mu, variance, target = np.broadcast_arrays(mu, variance, target)
    if np.any(np.asarray(variance) < math.exp(-10) * (1 - 1e-6)):
        return circular_crps_reference(mu, variance, target)
    harmonics = np.arange(1, 2 * terms, 2, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    shape = mu.shape
    mu = mu.reshape(-1)
    variance = variance.reshape(-1)
    target = target.reshape(-1)
    output = np.empty_like(mu, dtype=np.float64)
    for start in range(0, len(mu), 512):
        stop = start + 512
        damping = np.exp(-0.5 * variance[start:stop, None] * frequencies**2)
        phase = wrap(mu[start:stop] - target[start:stop])[:, None] * frequencies
        series = damping * np.cos(phase) - 0.5 * damping**2
        output[start:stop] = PERIOD / 8 - 2 * PERIOD / math.pi**2 * (
            series / harmonics**2
        ).sum(-1)
    return output.reshape(shape)


# Approximate circular CRPS with all pairs of Gaussian quantile nodes.
def circular_crps_quantile(mu, variance, target, samples):
    probabilities = (np.arange(samples) + 0.5) / samples
    nodes = norm.ppf(probabilities)
    draws = mu[..., None] + np.sqrt(variance[..., None]) * nodes
    first = np.abs(wrap(draws - target[..., None])).mean(-1)
    second = np.abs(wrap(draws[..., :, None] - draws[..., None, :])).mean((-1, -2))
    return first - 0.5 * second


# Get per-image, per-factor CRPS values.
def crps_matrix(mu, variance, target, terms=512, samples=None):
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-8)
    output = np.empty_like(mu, dtype=np.float64)
    first = absolute_normal(mu[:, :7] - target[:, :7], variance[:, :7])
    output[:, :7] = first - np.sqrt(variance[:, :7] / math.pi)
    for start in range(0, len(mu), 128):
        subset = slice(start, start + 128)
        if samples is None:
            values = circular_crps_fourier(
                mu[subset, 7:], variance[subset, 7:], target[subset, 7:], terms
            )
        else:
            values = circular_crps_quantile(
                mu[subset, 7:], variance[subset, 7:], target[subset, 7:], samples
            )
        output[subset, 7:] = values
    return output


#  Compute a wrapped-normal log density with controlled alias tails.
def wrapped_log_density(delta, variance):
    delta, variance = np.broadcast_arrays(wrap(np.asarray(delta)), variance)
    small = np.sqrt(variance) < PERIOD / 2
    output = np.empty(delta.shape, dtype=np.float64)
    values = np.full(delta[small].shape, -np.inf)
    for alias in range(-ALIAS_RADIUS, ALIAS_RADIUS + 1):
        standardized = (delta[small] + alias * PERIOD) ** 2 / variance[small]
        log_density = -0.5 * (standardized + np.log(2 * math.pi * variance[small]))
        values = np.logaddexp(values, log_density)
    output[small] = values
    harmonics = np.arange(1, 9, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    damping = np.exp(-0.5 * variance[~small, None] * frequencies**2)
    series = 1 + 2 * (damping * np.cos(delta[~small, None] * frequencies)).sum(-1)
    output[~small] = np.log(series) - math.log(PERIOD)
    return output


def wrapped_cdf(target, mu, variance):
    """Return wrapped-normal CDF values anchored at minus pi over two."""
    target, mu, variance = np.broadcast_arrays(target, mu, variance)
    mu = wrap(mu)
    position = np.clip(target, ORIGIN, ORIGIN + PERIOD)
    small = np.sqrt(variance) < PERIOD / 2
    output = np.zeros_like(position, dtype=np.float64)
    sigma = np.sqrt(variance[small])
    for alias in range(-ALIAS_RADIUS, ALIAS_RADIUS + 1):
        upper = (position[small] + alias * PERIOD - mu[small]) / sigma
        lower = (ORIGIN + alias * PERIOD - mu[small]) / sigma
        output[small] += ndtr(upper) - ndtr(lower)
    harmonics = np.arange(1, 9, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    damping = np.exp(-0.5 * variance[~small, None] * frequencies**2)
    upper_phase = (position[~small] - mu[~small])[:, None] * frequencies
    lower_phase = (ORIGIN - mu[~small])[:, None] * frequencies
    series = damping * (np.sin(upper_phase) - np.sin(lower_phase))
    output[~small] = (position[~small] - ORIGIN) / PERIOD + (
        series / (math.pi * harmonics)
    ).sum(-1)
    return np.clip(output, 0, 1)


# Compute linear and circular probability integral transforms.
def pit_matrix(mu, variance, target):
    output = np.empty_like(mu, dtype=np.float64)
    output[:, :7] = ndtr((target[:, :7] - mu[:, :7]) / np.sqrt(variance[:, :7]))
    for factor in range(7, 10):
        output[:, factor] = wrapped_cdf(
            target[:, factor], mu[:, factor], variance[:, factor]
        )
    return np.clip(output, 1e-7, 1 - 1e-7)


def nll_matrix(mu, variance, target):
    """Return marginal Gaussian and wrapped-normal negative log likelihoods."""
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-8)
    output = 0.5 * ((target - mu) ** 2 / variance + np.log(2 * math.pi * variance))
    output[:, 7:] = -wrapped_log_density(target[:, 7:] - mu[:, 7:], variance[:, 7:])
    return output


def mixed_crps_torch(mu, logvar, target, config, estimator="fourier"):
    """Differentiate factor-mean Gaussian and circular geodesic CRPS."""

    variance = logvar.double().clamp(-10, 10).exp()
    sigma = variance[:, :7].sqrt()
    standardized = (target[:, :7].double() - mu[:, :7].double()) / sigma
    linear = standardized * torch.erf(standardized / math.sqrt(2))
    linear += math.sqrt(2 / math.pi) * torch.exp(-0.5 * standardized**2)
    linear = sigma * (linear - 1 / math.sqrt(math.pi))
    if estimator == "fourier":
        circular = circular_crps_torch(
            mu[:, 7:], logvar[:, 7:], target[:, 7:], config.fourier_terms
        )
    elif estimator == "quantile":
        count = config.circular_train_samples
        probabilities = (
            torch.arange(count, device=mu.device, dtype=torch.float64) + 0.5
        ) / count
        nodes = math.sqrt(2) * torch.erfinv(2 * probabilities - 1)
        draws = mu[:, 7:, None].double() + variance[:, 7:, None].sqrt() * nodes
        first = wrap(draws - target[:, 7:, None]).abs().mean(-1)
        pairs = wrap(draws[..., :, None] - draws[..., None, :]).abs().mean((-1, -2))
        circular = first - 0.5 * pairs
    else:
        raise ValueError(estimator)
    return torch.cat((linear, circular), dim=1).mean()


def circular_crps_torch(mu, logvar, target, terms=512):
    """Differentiate the wrapped-normal geodesic score through its Fourier series."""

    harmonics = torch.arange(1, 2 * terms, 2, device=mu.device, dtype=torch.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    variance = logvar.double().clamp(-10, 10).exp()
    damping = torch.exp(-0.5 * variance.unsqueeze(-1) * frequencies**2)
    phase = wrap(mu.double() - target.double()).unsqueeze(-1) * frequencies
    series = damping * phase.cos() - 0.5 * damping.square()
    return PERIOD / 8 - 2 * PERIOD / math.pi**2 * (series / harmonics**2).sum(-1)


#  Evaluate differentiable all-pairs quantile CRPS in the input dtype.
def circular_crps_quantile_torch(mu, logvar, target, samples):
    probabilities = (
        torch.arange(samples, device=mu.device, dtype=mu.dtype) + 0.5
    ) / samples
    nodes = math.sqrt(2) * torch.erfinv(2 * probabilities - 1)
    sigma = torch.exp(0.5 * logvar.clamp(-10, 10))
    draws = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * nodes
    first = wrap(draws - target.unsqueeze(-1)).abs().mean(-1)
    second = wrap(draws.unsqueeze(-1) - draws.unsqueeze(-2)).abs().mean((-1, -2))
    return first - 0.5 * second


# Evaluate Gaussian and adaptive-alias wrapped-normal NLL.
def mixed_nll_torch(mu, logvar, target, config):
    safe_logvar = logvar.clamp(-10, 10)
    variance = safe_logvar.exp()
    linear = 0.5 * (math.log(2 * math.pi) + safe_logvar[:, :7])
    linear += 0.5 * (target[:, :7] - mu[:, :7]) ** 2 / variance[:, :7]
    sigma_max = float(variance[:, 7:].max().detach().sqrt().cpu())
    aliases = math.ceil(config.wrapped_alias_coverage_sigmas * sigma_max / PERIOD) + 1
    aliases = max(config.wrapped_alias_min, aliases)
    if aliases > config.wrapped_alias_hard_limit:
        raise ValueError("Wrapped-normal alias limit exceeded.")
    shifts = (
        torch.arange(-aliases, aliases + 1, device=mu.device, dtype=mu.dtype) * PERIOD
    )
    delta = wrap(target[:, 7:] - mu[:, 7:]).unsqueeze(-1) + shifts
    values = -0.5 * (math.log(2 * math.pi) + safe_logvar[:, 7:, None])
    values = values - 0.5 * delta.square() / variance[:, 7:, None]
    circular = -torch.logsumexp(values, dim=-1)
    return torch.cat((linear, circular), dim=1).mean()
