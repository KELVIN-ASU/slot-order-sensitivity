"""Temperature, isotonic CDF, and split-conformal calibration."""

import math

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from probability import (
    ORIGIN,
    PERIOD,
    mixed_nll_torch,
    pit_matrix,
    residual_matrix,
    wrap,
    wrapped_cdf,
)


def optimize_temperature(mu, variance, target, per_factor, config):
    """Fit positive standard-deviation multipliers by minimizing NLL with Adam."""

    means = torch.as_tensor(mu, device=config.device, dtype=torch.float32)
    log_variance = torch.as_tensor(
        variance, device=config.device, dtype=torch.float32
    ).log()
    targets = torch.as_tensor(target, device=config.device, dtype=torch.float32)
    shape = ()
    if per_factor:
        shape = (mu.shape[1],)
    initial = math.log(math.expm1(1.0))
    parameter = torch.full(shape, initial, device=config.device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=config.temperature_lr)
    for step in range(config.temperature_steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = functional.softplus(parameter) + 1e-6
        scaled = log_variance + 2 * temperature.log()
        loss = mixed_nll_torch(means, scaled, targets, config)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite temperature objective.")
        loss.backward()
        optimizer.step()
    return (functional.softplus(parameter) + 1e-6).detach().cpu().numpy()


def interval_from_half_width(mu, half_width):
    """Build linear intervals and cap wide hue arcs at a full circle."""
    width = np.broadcast_to(half_width, mu.shape).copy()
    lower = mu - width
    upper = mu + width
    full_circle = np.zeros_like(mu, dtype=bool)
    full_circle[:, 7:] = width[:, 7:] >= PERIOD / 2
    lower[:, 7:] = wrap(lower[:, 7:])
    upper[:, 7:] = wrap(upper[:, 7:])
    return lower, upper, full_circle


def symmetric_interval(mu, variance, level):
    """Construct Gaussian-derived symmetric intervals and circular arcs."""
    half_width = norm.ppf((1 + level) / 2) * np.sqrt(np.maximum(variance, 1e-8))
    return interval_from_half_width(mu, half_width)


def interval_diagnostics(lower, upper, target, level, full_circle):
    """Compute interval widths, containment, and endpoint miss distances."""
    inside = (target >= lower) & (target <= upper)
    width = upper - lower
    miss = np.maximum(lower - target, 0) + np.maximum(target - upper, 0)
    for factor in range(7, 10):
        span = np.mod(upper[:, factor] - lower[:, factor], PERIOD)
        offset = np.mod(target[:, factor] - lower[:, factor], PERIOD)
        inside[:, factor] = (offset <= span) | full_circle[:, factor]
        width[:, factor] = span
        width[full_circle[:, factor], factor] = PERIOD
        endpoints = np.minimum(
            np.abs(wrap(target[:, factor] - lower[:, factor])),
            np.abs(wrap(target[:, factor] - upper[:, factor])),
        )
        miss[:, factor] = np.where(inside[:, factor], 0, endpoints)
    score = width + 2 * miss / (1 - level)
    return {
        "inside": inside,
        "width": width,
        "miss_distance": miss,
        "interval_score": score,
    }


def conformal_quantile(scores, alpha):
    """Select the conservative conformal rank with higher-quantile interpolation."""
    count = len(scores)
    probability = min(np.ceil((count + 1) * (1 - alpha)) / count, 1)
    return np.quantile(scores, probability, axis=0, method="higher")


def fit_isotonic(mu, variance, target):
    """Fit factorwise isotonic maps from PIT probabilities to empirical ranks."""
    probabilities = pit_matrix(mu, variance, target)
    empirical = (np.arange(len(target)) + 0.5) / len(target)
    models = []
    for factor in range(target.shape[1]):
        order = np.argsort(probabilities[:, factor], kind="mergesort")
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(probabilities[order, factor], empirical)
        models.append(model)
    return models


def isotonic_transform(probabilities, models):
    """Transform factorwise PIT values with calibration-only isotonic maps."""
    output = np.empty_like(probabilities)
    for factor, model in enumerate(models):
        output[:, factor] = model.transform(probabilities[:, factor])
    return np.clip(output, 1e-7, 1 - 1e-7)


def isotonic_inverse(probability, models, count):
    """Invert a 2001-point monotone calibration grid."""
    grid = np.linspace(0, 1, 2001)
    output = np.empty((count, len(models)))
    for factor, model in enumerate(models):
        mapped = model.transform(grid)
        values, indices = np.unique(mapped, return_index=True)
        if len(values) < 2:
            output[:, factor] = probability
        else:
            output[:, factor] = np.interp(probability, values, grid[indices])
    return np.clip(output, 1e-7, 1 - 1e-7)


def predictive_quantile(probabilities, mu, variance):
    """Compute Gaussian quantiles and anchored wrapped-normal quantiles."""
    output = mu + np.sqrt(variance) * norm.ppf(probabilities)
    for factor in range(7, 10):
        lower = np.full(len(mu), ORIGIN)
        upper = np.full(len(mu), ORIGIN + PERIOD)
        for step in range(50):
            middle = (lower + upper) / 2
            mass = wrapped_cdf(middle, mu[:, factor], variance[:, factor])
            lower = np.where(mass < probabilities[:, factor], middle, lower)
            upper = np.where(mass >= probabilities[:, factor], middle, upper)
        output[:, factor] = (lower + upper) / 2
    return output


def fit_calibration_bundle(pack, config):
    """Fit all calibration methods on the designated calibration subset."""
    mu = pack["mu"]
    variance = pack["var"]
    target = pack["y"]
    base = symmetric_interval(mu, variance, config.interval_level)
    diagnostics = interval_diagnostics(
        *base[:2], target, config.interval_level, base[2]
    )
    return {
        "scalar_temperature": optimize_temperature(mu, variance, target, False, config),
        "per_factor_temperature": optimize_temperature(
            mu, variance, target, True, config
        ),
        "isotonic": fit_isotonic(mu, variance, target),
        "q_abs": conformal_quantile(
            np.abs(residual_matrix(mu, target)), 1 - config.interval_level
        ),
        "q_gsc": conformal_quantile(
            diagnostics["miss_distance"], 1 - config.interval_level
        ),
    }


def calibrated_variances(variance, bundle):
    """Return raw, scalar-temperature, and per-factor-temperature variances."""
    return {
        "raw": variance,
        "scalar_temperature": variance * bundle["scalar_temperature"] ** 2,
        "per_factor_temperature": variance * bundle["per_factor_temperature"] ** 2,
    }


def all_intervals(mu, variance, bundle, level):
    """Construct all six interval methods from fitted calibration parameters."""
    intervals = {}
    for name, values in calibrated_variances(variance, bundle).items():
        interval_name = name
        if name == "raw":
            interval_name = "raw_gaussian"
        intervals[interval_name] = symmetric_interval(mu, values, level)
    intervals["absolute_conformal"] = interval_from_half_width(mu, bundle["q_abs"])
    base = symmetric_interval(mu, variance, level)
    diagnostics = interval_diagnostics(*base[:2], mu, level, base[2])
    half_width = diagnostics["width"] / 2 + bundle["q_gsc"]
    intervals["gaussian_split_conformal"] = interval_from_half_width(mu, half_width)
    lower_probability = isotonic_inverse((1 - level) / 2, bundle["isotonic"], len(mu))
    upper_probability = isotonic_inverse((1 + level) / 2, bundle["isotonic"], len(mu))
    lower = predictive_quantile(lower_probability, mu, variance)
    upper = predictive_quantile(upper_probability, mu, variance)
    full = np.zeros_like(mu, dtype=bool)
    full[:, 7:] = upper[:, 7:] - lower[:, 7:] >= PERIOD - 1e-6
    intervals["isotonic_cdf"] = (lower, upper, full)
    return intervals
