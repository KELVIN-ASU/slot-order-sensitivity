## Factorwise permutation, initialization, and corruption configuration checks. ##

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from configuration import Config, load_config, validate_config
from inference import record_draw_variation_by_factor
from reporting import SummaryTables, prediction_variation, record_variation
from result_summaries import permutation_initialization_comparison


# Check summaries against explicit reference values.
class VariationTests(unittest.TestCase):

    def test_uncertainty_scaled_permutation_change_and_circular_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables = SummaryTables(root)
            config = replace(Config(), sigma_change_tolerance=0.5)
            means = np.zeros((3, 24, 10), dtype=np.float64)
            means[:, -1, 0] = [0.4, 0.8, 1.2]
            means[:, :12, 7] = np.pi / 2 - 0.01
            means[:, 12:, 7] = -np.pi / 2 + 0.01
            variance = np.full_like(means, 0.25)
            variance[:, :12, 0] = 0.04
            variance[:, 12:, 0] = 0.36
            scales = np.ones(10)
            scales[7:] = np.pi
            pack = {"mu": means, "var": variance, "indices": np.arange(3)}
            means_before = means.copy()
            variance_before = variance.copy()
            random_state = np.random.get_state()
            record_variation(pack, {}, scales, tables, root / "change.npz", config)
            rows = tables.rows["prediction_variation_by_factor.csv"]
            expected = np.array([0.4, 0.8, 1.2]) / np.sqrt(0.2)
            self.assertAlmostEqual(
                rows[0]["sigma_normalized_change_p50"], np.median(expected)
            )
            self.assertAlmostEqual(
                rows[0]["sigma_normalized_change_p95"], np.quantile(expected, 0.95)
            )
            self.assertEqual(rows[0]["fraction_above_sigma_tolerance"], 1.0)
            self.assertAlmostEqual(rows[7]["mean_change_p50"], 0.02)
            self.assertAlmostEqual(rows[7]["sigma_normalized_change_p50"], 0.04)
            self.assertEqual(rows[7]["fraction_above_sigma_tolerance"], 0.0)
            expected_arrays = prediction_variation(means, variance, scales)
            with np.load(root / "change.npz") as saved:
                self.assertEqual(set(saved.files), {"indices", *expected_arrays})
                for name, values in expected_arrays.items():
                    np.testing.assert_array_equal(saved[name], values)
            np.testing.assert_array_equal(means, means_before)
            np.testing.assert_array_equal(variance, variance_before)
            for first, second in zip(random_state, np.random.get_state()):
                np.testing.assert_equal(first, second)

    def test_sigma_tolerance_changes_only_exceedance_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = SummaryTables(Path(directory))
            means = np.zeros((2, 24, 10))
            means[:, -1, 0] = [0.5, 1.0]
            pack = {"mu": means, "var": np.ones_like(means), "indices": [0, 1]}
            for tolerance in (0.5, 2.0):
                record_variation(
                    pack,
                    {},
                    np.ones(10),
                    tables,
                    Path(directory) / "change.npz",
                    replace(Config(), sigma_change_tolerance=tolerance),
                )
            rows = tables.rows["prediction_variation_by_factor.csv"]
            first = dict(rows[0])
            second = dict(rows[10])
            self.assertEqual(first.pop("fraction_above_sigma_tolerance"), 0.5)
            self.assertEqual(second.pop("fraction_above_sigma_tolerance"), 0.0)
            self.assertEqual(first, second)

    def test_draw_variation_preserves_factor_specific_quantiles(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = SummaryTables(Path(directory))
            changes = np.arange(30, dtype=float).reshape(3, 10) / 100
            scales = np.arange(1, 11, dtype=float)
            entry = {
                "group": "primary",
                "regime": "frozen",
                "architecture": "flatmlp",
                "seed": 45,
            }
            record_draw_variation_by_factor(changes, scales, entry, 2, tables)
            for factor, row in enumerate(tables.rows["draw_variation_by_factor.csv"]):
                expected = changes[:, factor] / scales[factor]
                self.assertEqual(row["factor_index"], factor)
                self.assertEqual(row["n_draws"], 2)
                self.assertEqual(row["status"], "available")
                self.assertEqual(row["normalized_change_p50"], np.median(expected))
                self.assertEqual(
                    row["normalized_change_p95"], np.quantile(expected, 0.95)
                )
            record_draw_variation_by_factor(changes, scales, entry, 1, tables)
            for row in tables.rows["draw_variation_by_factor.csv"][10:]:
                self.assertEqual(row["status"], "insufficient_draws")
                self.assertTrue(np.isnan(row["normalized_change_p50"]))
                self.assertTrue(np.isnan(row["normalized_change_p95"]))

    def test_comparison_matches_reference_and_averages_draws_within_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(Config(), output_root=root)
            tables = SummaryTables(root)
            for group, regime, seed, denominator, status in (
                ("primary", "frozen", 45, 2.0, "available"),
                ("primary", "frozen", 46, 4.0, "available"),
                ("primary", "joint", 45, 8.0, "available"),
                ("primary", "frozen", 47, 0.0, "available"),
                ("primary", "frozen", 48, 1e-14, "available"),
                ("primary", "frozen", 49, np.nan, "insufficient_draws"),
                ("backbone_101", "frozen", 45, None, "missing_reference"),
            ):
                for factor in (0, 7):
                    info = dict(
                        group=group,
                        regime=regime,
                        train_seed=seed,
                        factor_index=factor,
                        factor=str(factor),
                        split="working",
                    )
                    for draw, change in ((7, 2.0), (13, 6.0)):
                        tables.add(
                            "prediction_variation_by_factor.csv",
                            dict(
                                info,
                                architecture="flatmlp",
                                mode="per_example_stochastic",
                                slot_init_seed=draw,
                                normalizing_scale=2.0,
                                mean_change_p50=change,
                                mean_change_p95=change * 2,
                            ),
                        )
                    if denominator is not None:
                        tables.add(
                            "draw_variation_by_factor.csv",
                            dict(
                                info,
                                architecture="deepsets",
                                status=status,
                                n_draws=1 if status == "insufficient_draws" else 2,
                                normalized_change_p50=denominator,
                                normalized_change_p95=denominator * 2,
                            ),
                        )
            permutation_initialization_comparison(tables, config)
            frame = pd.read_csv(
                root / "evaluation/permutation_change_vs_initialization_variation.csv"
            )
            self.assertEqual(len(frame), 14)
            self.assertTrue((frame.n_permutation_draws == 2).all())
            self.assertTrue((frame.normalized_permutation_change_p50 == 2.0).all())
            self.assertTrue((frame.normalized_permutation_change_p95 == 4.0).all())
            for row in frame.itertuples():
                expected_status = "available"
                expected_ratio = 1.0
                if row.group == "backbone_101":
                    expected_status = "missing_reference"
                elif row.train_seed in (47, 48):
                    expected_status = "negligible_reference"
                elif row.train_seed == 49:
                    expected_status = "insufficient_draws"
                elif row.regime == "joint":
                    expected_ratio = 0.25
                elif row.train_seed == 46:
                    expected_ratio = 0.5
                for quantile in ("p50", "p95"):
                    self.assertEqual(
                        getattr(row, f"comparison_status_{quantile}"), expected_status
                    )
                    ratio = getattr(
                        row, f"permutation_to_initialization_ratio_{quantile}"
                    )
                    if expected_status == "available":
                        self.assertEqual(ratio, expected_ratio)
                    else:
                        self.assertTrue(np.isnan(ratio))

    def test_corruption_severities_accept_single_and_supported_subsets(self):
        for severities in ((3,), (1, 5), (5, 2, 4), (1, 2, 3, 4, 5)):
            validate_config(replace(Config(), corruption_severities=severities))
        for severities in ((), (1, 1), (0,), (6,), (1.0,), (True,), ("3",)):
            with self.subTest(severities=severities):
                with self.assertRaises(ValueError):
                    validate_config(replace(Config(), corruption_severities=severities))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                '{"corruption_severities": [3], "sigma_change_tolerance": 0.7}'
            )
            config = load_config(path)
            self.assertEqual(config.corruption_severities, (3,))
            self.assertEqual(config.sigma_change_tolerance, 0.7)
        for tolerance in (0.0, -0.5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                validate_config(replace(Config(), sigma_change_tolerance=tolerance))


if __name__ == "__main__":
    unittest.main()
