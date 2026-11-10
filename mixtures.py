"""Predictive-distribution mixtures and their uncertainty diagnostics."""

import math

import numpy as np
from scipy.special import logsumexp, ndtr

from probability import (
    PERIOD,
    absolute_normal,
    expected_circular_distance,
    expected_circular_squared_distance,
    wrap,
    wrapped_cdf,
    wrapped_log_density,
)

SMALL_SIGMA = PERIOD / 2
ALIAS_RADIUS = 7
FOURIER_TERMS = 8


def mixture_log_density(component_means, component_variances, target, circular=False):
    """Evaluate the log density of an equal-weight Gaussian or wrapped-normal mixture."""
    if not circular:
        values = -0.5 * (
            (target[:, None] - component_means) ** 2 / component_variances
            + np.log(2 * math.pi * component_variances)
        )
    else:
        values = wrapped_log_density(
            target[:, None] - component_means, component_variances
        )
    return logsumexp(values, axis=1) - math.log(component_means.shape[1])


def mixture_joint_nll(mu, variance, target):
    """Score the equal mixture of product distributions with a shared component index."""
    component_scores = np.zeros(mu.shape[:2], dtype=np.float64)
    for factor in range(10):
        difference = target[:, factor, None] - mu[:, :, factor]
        if factor < 7:
            values = difference**2 / variance[:, :, factor]
            values += np.log(2 * math.pi * variance[:, :, factor])
            component_scores -= 0.5 * values
        else:
            component_scores += wrapped_log_density(difference, variance[:, :, factor])
    return math.log(mu.shape[1]) - logsumexp(component_scores, axis=1)


def wrapped_arc_mass(delta, var, half_width):
    """Integrate a wrapped normal over a symmetric circular arc."""
    delta, var, half_width = np.broadcast_arrays(
        wrap(np.asarray(delta, dtype=np.float64)),
        np.asarray(var, dtype=np.float64),
        np.asarray(half_width, dtype=np.float64),
    )
    out = np.empty(delta.shape)
    small = np.sqrt(var) < SMALL_SIGMA
    d, s, h = (delta[small], np.sqrt(var[small]), half_width[small])
    mass = np.zeros(d.shape)
    for alias in range(-ALIAS_RADIUS, ALIAS_RADIUS + 1):
        mass += ndtr((h - d + alias * PERIOD) / s) - ndtr((-h - d + alias * PERIOD) / s)
    out[small] = mass
    d, v, h = (delta[~small], var[~small], half_width[~small])
    n = np.arange(1, FOURIER_TERMS + 1, dtype=np.float64)
    k = 2 * math.pi * n / PERIOD
    out[~small] = 2 * h / PERIOD + np.sum(
        2
        / (math.pi * n)
        * np.exp(-0.5 * v[..., None] * k**2)
        * np.cos(d[..., None] * k)
        * np.sin(h[..., None] * k),
        axis=-1,
    )
    return out


def harmonic_limit(min_variance, exponent=46.0):
    """Choose a harmonic cutoff from the minimum variance and damping tolerance."""
    return (
        int(math.ceil(math.sqrt(2 * exponent / min_variance) * PERIOD / (2 * math.pi)))
        + 1
    )


def arc_mass(mu, var, center, half_width):
    """Average component probabilities inside a common circular arc."""
    return np.clip(
        wrapped_arc_mass(mu - center[:, None], var, half_width[:, None]).mean(1), 0, 1
    )


def mixture_interval(component_means, component_variances, center, circular, level=0.9):
    """Find equal-tailed linear intervals or centered circular probability-mass arcs."""
    if circular:
        lower, upper = (
            np.zeros(len(component_means)),
            np.full(len(component_means), PERIOD / 2),
        )
        for _ in range(36):
            middle = (lower + upper) / 2
            enough = (
                arc_mass(component_means, component_variances, center, middle) >= level
            )
            upper, lower = (
                np.where(enough, middle, upper),
                np.where(enough, lower, middle),
            )
        return (center - upper, center + upper)
    lower_bound = np.min(component_means - 10 * np.sqrt(component_variances), axis=1)
    upper_bound = np.max(component_means + 10 * np.sqrt(component_variances), axis=1)
    quantiles = []
    for probability in [(1 - level) / 2, (1 + level) / 2]:
        lower, upper = (lower_bound.copy(), upper_bound.copy())
        for _ in range(40):
            middle = (lower + upper) / 2
            enough = (
                ndtr(
                    (middle[:, None] - component_means) / np.sqrt(component_variances)
                ).mean(1)
                >= probability
            )
            upper, lower = (
                np.where(enough, middle, upper),
                np.where(enough, lower, middle),
            )
        quantiles.append((lower + upper) / 2)
    return quantiles


def mixture_point_predictions(means, variances):
    """Compute linear means and variance-weighted circular mean directions."""
    point = means.mean(1)
    smallest = variances[:, :, 7:].min(1, keepdims=True)
    weights = np.exp(-2 * (variances[:, :, 7:] - smallest))
    direction = (weights * np.exp(2j * wrap(means[:, :, 7:]))).mean(1)
    resultant = direction * np.exp(-2 * smallest[:, 0])
    point[:, 7:] = np.angle(direction) / 2
    direction_defined = np.abs(direction) / weights.mean(1) >= 1e-08
    point[:, 7:] = np.where(direction_defined, point[:, 7:], 0.0)
    if means.shape[1] == 1:
        point[:, 7:] = wrap(means[:, 0, 7:])
    return (point, resultant, direction_defined)


def circular_mixture_crps(component_means, component_variances, target):
    """Score a wrapped-normal mixture with the circular geodesic energy formula."""
    limit = harmonic_limit(component_variances.min())
    harmonics = np.arange(1, min(1023, limit) + 1, 2, dtype=np.float64)
    frequencies = 2 * math.pi * harmonics / PERIOD
    damping = np.exp(-0.5 * component_variances[..., None] * frequencies**2)
    cosine, sine = (
        (damping * np.cos(wrap(component_means)[..., None] * frequencies)).mean(1),
        (damping * np.sin(wrap(component_means)[..., None] * frequencies)).mean(1),
    )
    phase = wrap(target)[:, None] * frequencies
    score = PERIOD / 8 - 2 * PERIOD / math.pi**2 * np.sum(
        (cosine * np.cos(phase) + sine * np.sin(phase) - 0.5 * (cosine**2 + sine**2))
        / harmonics**2,
        axis=1,
    )
    if component_variances.min() < math.exp(-10):
        first = expected_circular_distance(
            component_means - target[:, None], component_variances
        ).mean(1)
        pair_mean = component_means[:, :, None] - component_means[:, None, :]
        pair_variance = (
            component_variances[:, :, None] + component_variances[:, None, :]
        )
        second = expected_circular_distance(pair_mean, pair_variance).mean((1, 2))
        score = first - 0.5 * second
    return score


def mixture_metrics(means, variances, targets, level=0.9):
    """Compute errors, distribution scores, probability intervals, and dispersion."""
    means, variances, targets = (
        np.asarray(means, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    sample_count, component_count, factor_count = means.shape
    if (
        factor_count != 10
        or variances.shape != means.shape
        or targets.shape != (sample_count, 10)
        or (not np.isfinite(means).all())
        or (not np.isfinite(variances).all())
        or np.any(variances <= 0)
    ):
        raise ValueError("Invalid predictive mixture arrays.")
    point, resultant, direction_defined = mixture_point_predictions(means, variances)
    output = {
        key: np.empty((sample_count, 10))
        for key in [
            "squared_error",
            "absolute_error",
            "nll",
            "crps",
            "coverage90",
            "width90",
            "interval_score90",
            "dispersion",
            "standardized_squared_error",
            "expected_squared_error",
            "pit",
        ]
    }
    for factor in range(10):
        component_means, component_variances, factor_target = (
            means[:, :, factor],
            variances[:, :, factor],
            targets[:, factor],
        )
        circular = factor >= 7
        residual = point[:, factor] - factor_target
        if circular:
            residual = wrap(residual)
        output["squared_error"][:, factor] = residual**2
        output["absolute_error"][:, factor] = np.abs(residual)
        output["nll"][:, factor] = -mixture_log_density(
            component_means, component_variances, factor_target, circular
        )
        if circular:
            output["crps"][:, factor] = circular_mixture_crps(
                component_means, component_variances, factor_target
            )
            expected_squared = expected_circular_squared_distance(
                component_means - point[:, factor, None], component_variances
            ).mean(1)
            output["dispersion"][:, factor] = 1 - np.abs(resultant[:, factor - 7])
            output["expected_squared_error"][:, factor] = np.maximum(
                expected_squared, 1e-15
            )
            output["pit"][:, factor] = wrapped_cdf(
                factor_target[:, None], component_means, component_variances
            ).mean(1)
        else:
            first = absolute_normal(
                component_means - factor_target[:, None], component_variances
            ).mean(1)
            second = absolute_normal(
                component_means[:, :, None] - component_means[:, None, :],
                component_variances[:, :, None] + component_variances[:, None, :],
            ).mean((1, 2))
            output["crps"][:, factor] = first - 0.5 * second
            output["dispersion"][:, factor] = np.mean(
                component_variances + (component_means - point[:, factor, None]) ** 2,
                axis=1,
            )
            output["expected_squared_error"][:, factor] = output["dispersion"][
                :, factor
            ]
            output["pit"][:, factor] = ndtr(
                (factor_target[:, None] - component_means)
                / np.sqrt(component_variances)
            ).mean(1)
        output["standardized_squared_error"][:, factor] = (
            residual**2 / output["expected_squared_error"][:, factor]
        )
        lower, upper = mixture_interval(
            component_means, component_variances, point[:, factor], circular, level
        )
        width = upper - lower
        if circular:
            miss = np.maximum(np.abs(residual) - width / 2, 0)
        else:
            miss = np.maximum(lower - factor_target, 0) + np.maximum(
                factor_target - upper, 0
            )
        output["coverage90"][:, factor] = miss <= 1e-10
        output["width90"][:, factor] = width
        output["interval_score90"][:, factor] = width + 2 * miss / (1 - level)
    output["point"] = point
    output["joint_nll"] = mixture_joint_nll(means, variances, targets)
    output["circular_direction_defined"] = direction_defined
    if component_count > 1:
        for key in ["squared_error", "absolute_error", "standardized_squared_error"]:
            output[key][:, 7:] = np.where(direction_defined, output[key][:, 7:], np.nan)
    return output
