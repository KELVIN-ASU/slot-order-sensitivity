"""Scientific summary tables for all calibration and evaluation conditions."""

import numpy as np
from scipy.stats import kstest, norm
from tqdm import tqdm

from artifacts import atomic_csv_save, atomic_npz_save
from calibration import (
    all_intervals,
    calibrated_variances,
    interval_diagnostics,
    isotonic_transform,
    symmetric_interval,
)
from data import FACTOR_NAMES
from probability import (
    crps_matrix,
    expected_circular_squared_distance,
    nll_matrix,
    pit_matrix,
    residual_matrix,
    wrap,
)
from statistical_analysis import contribution_summary


def reg_ece_per_factor(mu, variance, target, samples, bins=15):
    """Measure regression calibration error in equal-count predictive-scale bins."""
    sigma = np.sqrt(np.maximum(variance, 1e-8))
    observed = np.abs(residual_matrix(mu, target))
    expected = np.sqrt(2 / np.pi) * sigma
    nodes = norm.ppf((np.arange(samples) + 0.5) / samples)
    draws = sigma[:, 7:, None] * nodes
    expected[:, 7:] = np.abs(wrap(draws)).mean(-1)
    errors = []
    for factor in range(10):
        edges = np.quantile(sigma[:, factor], np.linspace(0, 1, bins + 1))
        error = 0.0
        for index in range(bins):
            mask = (sigma[:, factor] >= edges[index]) & (
                sigma[:, factor] < edges[index + 1]
            )
            if index == bins - 1:
                mask = (sigma[:, factor] >= edges[index]) & (
                    sigma[:, factor] <= edges[index + 1]
                )
            if mask.any():
                error += mask.mean() * abs(
                    observed[mask, factor].mean() - expected[mask, factor].mean()
                )
        errors.append(error)
    return np.asarray(errors)


def predictive_contributions(mu, variance, target, config):
    """Compute per-image predictive scores and symmetric-interval diagnostics."""
    residual = residual_matrix(mu, target)
    intervals = symmetric_interval(mu, variance, config.interval_level)
    diagnostics = interval_diagnostics(
        *intervals[:2], target, config.interval_level, intervals[2]
    )
    scores = crps_matrix(mu, variance, target, config.fourier_terms)
    likelihood = nll_matrix(mu, variance, target)
    values = {
        "mse": (residual**2).mean(1),
        "mae": np.abs(residual).mean(1),
        "nll": likelihood.mean(1),
        "crps": scores.mean(1),
        "coverage90": diagnostics["inside"].mean(1),
        "width90": diagnostics["width"].mean(1),
        "interval_score90": diagnostics["interval_score"].mean(1),
    }
    return values, scores, likelihood, diagnostics


def pit_rows(probabilities, description):
    """Summarize factorwise PIT statistics and uniformity tests."""
    rows = []
    for factor, name in enumerate(FACTOR_NAMES):
        values = probabilities[:, factor]
        result = kstest(values, "uniform")
        histogram, edges = np.histogram(values, bins=20, range=(0, 1), density=True)
        row = dict(description)
        row.update(
            {
                "factor_index": factor,
                "factor": name,
                "n_samples": len(values),
                "pit_mean": float(values.mean()),
                "pit_variance": float(values.var()),
                "pit_mean_abs_gap_from_0_5": float(abs(values.mean() - 0.5)),
                "pit_variance_abs_gap_from_1_over_12": float(
                    abs(values.var() - 1 / 12)
                ),
                "pit_ks_statistic": float(result.statistic),
                "pit_ks_pvalue": float(result.pvalue),
                "pit_histogram_l1_from_uniform": float(np.abs(histogram - 1).mean()),
            }
        )
        rows.append(row)
    return rows


def record_prediction(pack, bundle, description, tables, config):
    """Report predictive scores, factorwise metrics, PIT, and interval diagnostics."""
    mu = pack["mu"].astype(np.float64)
    variance = pack["var"].astype(np.float64)
    target = pack["y"].astype(np.float64)
    prefix = "test"
    pit_filename = "per_factor_pit_statistics.csv"
    interval_filename = "interval_width_and_score_statistics.csv"
    if description["split"] == "audit":
        prefix = "audit"
        pit_filename = "audit_per_factor_pit.csv"
        interval_filename = "audit_intervals.csv"
    contributions = {}
    for variant, calibrated in tqdm(
        calibrated_variances(variance, bundle).items(),
        desc=f"{description['context']} predictive metrics",
        unit="variant",
        leave=False,
    ):
        values, scores, likelihood, diagnostics = predictive_contributions(
            mu, calibrated, target, config
        )
        contributions[variant] = values
        probabilities = pit_matrix(mu, calibrated, target)
        ece = reg_ece_per_factor(mu, calibrated, target, config.circular_eval_samples)
        row = dict(description)
        row["variant"] = variant
        row.update(contribution_summary(values))
        row["sharpness_variance"] = float(calibrated.mean())
        row["reg_ece"] = float(ece.mean())
        row["n_samples"] = len(target)
        tables.add(f"{prefix}_predictive_metrics.csv", row)
        if description["context"] == "slot_initialization":
            tables.add("slot_initialization_alignment_metrics.csv", dict(row))
        statistics = pit_rows(probabilities, dict(description, variant=variant))
        tables.extend(pit_filename, statistics)
        residual = residual_matrix(mu, target)
        for factor, name in enumerate(FACTOR_NAMES):
            factor_row = dict(
                description, variant=variant, factor_index=factor, factor=name
            )
            factor_row.update(
                {
                    "rmse": float(np.sqrt((residual[:, factor] ** 2).mean())),
                    "mae": float(np.abs(residual[:, factor]).mean()),
                    "nll": float(likelihood[:, factor].mean()),
                    "crps": float(scores[:, factor].mean()),
                    "mean_predicted_variance": float(calibrated[:, factor].mean()),
                    "reg_ece": float(ece[factor]),
                }
            )
            tables.add("per_factor_full_extraction.csv", factor_row)
    probabilities = isotonic_transform(
        pit_matrix(mu, variance, target), bundle["isotonic"]
    )
    tables.extend(
        pit_filename, pit_rows(probabilities, dict(description, variant="isotonic_cdf"))
    )
    for method, intervals in all_intervals(
        mu, variance, bundle, config.interval_level
    ).items():
        diagnostics = interval_diagnostics(
            *intervals[:2], target, config.interval_level, intervals[2]
        )
        for factor, name in enumerate(FACTOR_NAMES):
            row = dict(description, method=method, factor_index=factor, factor=name)
            row.update(
                {
                    "level": config.interval_level,
                    "n_samples": len(target),
                    "coverage": float(diagnostics["inside"][:, factor].mean()),
                    "coverage_gap": float(
                        diagnostics["inside"][:, factor].mean() - config.interval_level
                    ),
                    "mean_width": float(diagnostics["width"][:, factor].mean()),
                    "median_width": float(np.median(diagnostics["width"][:, factor])),
                    "mean_interval_score": float(
                        diagnostics["interval_score"][:, factor].mean()
                    ),
                }
            )
            tables.add(interval_filename, row)
    return contributions


def prediction_variation(mu, variance, scales):
    """Measure individual-image changes on a fixed 24-permutation slot orbit."""
    change_parts = []
    moment_parts = []
    for start in range(0, len(mu), 64):
        means = mu[start : start + 64].astype(np.float64)
        variances = variance[start : start + 64]
        difference = means[:, :, None] - means[:, None, :]
        difference[..., 7:] = wrap(difference[..., 7:])
        change_parts.append(np.abs(difference).max(axis=(1, 2)))
        harmonics = np.arange(1, 17)
        phase = 2j * means[:, :, 7:, None] * harmonics
        damping = -2 * variances[:, :, 7:, None] * harmonics**2
        moments = np.exp(phase + damping)
        moment_parts.append(np.abs(moments - moments[:, :1]).max(axis=(1, 2, 3)))
    changes = np.concatenate(change_parts)
    return {
        "maximum_pairwise_mean_change": changes,
        "maximum_normalized_mean_change": (changes / scales).max(1),
        "max_logvariance_change_from_identity": np.abs(
            np.log(variance) - np.log(variance[:, :1])
        ).max(axis=(1, 2)),
        "max_circular_moment_change_from_identity": np.concatenate(moment_parts),
    }


def record_variation(pack, description, scales, tables, path, config):
    """Save per-image changes and factor-specific quantiles without asserting invariance."""
    variation = prediction_variation(pack["mu"], pack["var"], scales)
    atomic_npz_save(path, indices=pack["indices"], **variation)
    mean_variance = pack["var"].astype(np.float64).mean(axis=1)
    predictive_scale = np.sqrt(np.maximum(mean_variance, 1e-15))
    for factor, name in enumerate(FACTOR_NAMES):
        values = variation["maximum_pairwise_mean_change"][:, factor]
        normalized = values / scales[factor]
        sigma_normalized = values / predictive_scale[:, factor]
        row = dict(description, factor_index=factor, factor=name)
        row.update(
            {
                "mean_change_p50": float(np.median(values)),
                "mean_change_p95": float(np.quantile(values, 0.95)),
                "mean_change_max": float(values.max()),
                "normalizing_scale": scales[factor],
                "normalized_change_p95": float(np.quantile(normalized, 0.95)),
                "fraction_above_tolerance": float(
                    np.mean(normalized > config.prediction_change_tolerance)
                ),
                "sigma_normalized_change_p50": float(np.median(sigma_normalized)),
                "sigma_normalized_change_p95": float(
                    np.quantile(sigma_normalized, 0.95)
                ),
                "fraction_above_sigma_tolerance": float(
                    np.mean(sigma_normalized > config.sigma_change_tolerance)
                ),
            }
        )
        tables.add("prediction_variation_by_factor.csv", row)
    row = dict(description)
    values = variation["maximum_normalized_mean_change"]
    row.update(
        {
            "mean_change_p50": float(np.median(values)),
            "mean_change_p95": float(np.quantile(values, 0.95)),
            "mean_change_max": float(values.max()),
            "fraction_above_tolerance": float(
                np.mean(values > config.prediction_change_tolerance)
            ),
            "logvariance_change_p95": float(
                np.quantile(variation["max_logvariance_change_from_identity"], 0.95)
            ),
            "circular_moment_change_p95": float(
                np.quantile(variation["max_circular_moment_change_from_identity"], 0.95)
            ),
        }
    )
    tables.add("prediction_variation_summary.csv", row)


def record_uncertainty_changes(pack, info, tables):
    """Compare per-image error and predictive dispersion changes on identical slots."""
    residual = residual_matrix(pack["mu"], pack["y"][:, None])
    observed = residual**2
    expected = pack["var"].astype(np.float64).copy()
    expected[:, :, 7:] = expected_circular_squared_distance(
        np.zeros_like(expected[:, :, 7:]), expected[:, :, 7:]
    )
    standardized = observed / np.maximum(expected, 1e-15)
    error_change = observed.mean(1) - observed[:, 0]
    dispersion_change = expected.mean(1) - expected[:, 0]
    for factor, name in enumerate(FACTOR_NAMES):
        correlation = np.nan
        first = error_change[:, factor]
        second = dispersion_change[:, factor]
        if first.std() > 1e-12 and second.std() > 1e-12:
            correlation = float(np.corrcoef(first, second)[0, 1])
        tables.add(
            "uncertainty_error_comparison.csv",
            dict(
                info,
                factor_index=factor,
                factor=name,
                identity_mse=float(observed[:, 0, factor].mean()),
                orbit_mse=float(observed[:, :, factor].mean()),
                identity_expected_mse=float(expected[:, 0, factor].mean()),
                orbit_expected_mse=float(expected[:, :, factor].mean()),
                identity_standardized_squared_error=float(
                    standardized[:, 0, factor].mean()
                ),
                orbit_standardized_squared_error=float(
                    standardized[:, :, factor].mean()
                ),
                error_dispersion_change_correlation=correlation,
            ),
        )


class SummaryTables:
    """Collect scientific results and publish each table atomically."""

    def __init__(self, root):
        self.root = root
        self.rows = {}

    def add(self, filename, row):
        """Append one scientific summary row."""
        if filename not in self.rows:
            self.rows[filename] = []
        self.rows[filename].append(row)

    def extend(self, filename, rows):
        """Append rows sharing one scientific table schema."""
        if filename not in self.rows:
            self.rows[filename] = []
        self.rows[filename].extend(rows)

    def flush(self):
        """Publish accumulated tables without separate run-status files."""
        for filename, rows in self.rows.items():
            atomic_csv_save(rows, self.root / filename)
