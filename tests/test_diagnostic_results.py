### Diagnostic discrepancies are recorded without terminating the experiment.

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_experiment
import torch

import diagnostics
import inference
import pipeline
from artifacts import RunManifest
from configuration import Config, load_config
from models import FlatMLPProbFactorPredictor, SlotAttentionAE
from reporting import SummaryTables


class DiagnosticResultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_score_and_gradient_discrepancies_are_returned(self):
        reference = diagnostics.score_integral_reference
        with (
            patch(
                "diagnostics.score_integral_reference",
                side_effect=lambda *args: reference(*args) + 0.1,
            ),
            patch("diagnostics.score_reference_gradients", return_value=(9.0, 9.0)),
        ):
            row = diagnostics.numerical_score_case(
                torch.float64, 0.5, 0.137, replace(Config(), device="cpu")
            )
        self.assertFalse(row["score_passed"])
        self.assertFalse(row["gradient_passed"])
        self.assertGreater(row["score_error"], row["score_tolerance"])
        self.assertGreater(row["gradient_error"], row["gradient_tolerance"])

    def test_differentiation_and_score_identities_record_failures(self):
        def score(mu, logvar, target, terms):
            return mu + logvar + target

        with (
            patch("diagnostics.circular_crps_torch", side_effect=score),
            patch("diagnostics.torch.autograd.gradcheck", return_value=False),
        ):
            rows = diagnostics.check_score_differentiation(replace(Config(), device="cpu"))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["passed"] is False for row in rows))
        self.assertEqual(
            {row["check"] for row in rows},
            {
                "autograd_gradcheck",
                "circular_score_periodicity",
                "clamped_variance_gradient",
                "circular_uniform_limit",
            },
        )

    def test_quadrature_records_each_scale_after_failure(self):
        with patch("diagnostics.quad", return_value=(0.25, 0.0)):
            rows = diagnostics.check_wrapped_quadrature()
        self.assertEqual(len(rows), 6)
        mass = [row for row in rows if row["check"] == "wrapped_normal_mass"]
        self.assertEqual(len(mass), 3)
        self.assertTrue(all(row["passed"] is False for row in mass))

    def test_deepsets_diagnostic_records_all_outputs_and_permutations(self):
        class OrderedPredictor(torch.nn.Module):
            def forward(self, slots):
                return slots[:, 0], 2 * slots[:, 0]

        with patch("diagnostics.build_predictor", return_value=OrderedPredictor()):
            rows = diagnostics.check_invariant_pooling(replace(Config(), device="cpu"))
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["permutation"] for row in rows}), 24)
        self.assertTrue(any(row["passed"] is False for row in rows))

    def test_backbone_equivariance_records_all_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = test_experiment.image_dataset(Path(directory) / "images")
            config = replace(Config(), device="cpu", num_workers=0)
            backbone = SlotAttentionAE().eval()
            forward = backbone.slot_attention.forward
            calls = 0

            def shifted(*args, **kwargs):
                nonlocal calls
                result = forward(*args, **kwargs)
                calls += 1
                if calls > 1:
                    result = result + 0.01
                return result

            with patch.object(backbone.slot_attention, "forward", side_effect=shifted):
                rows = diagnostics.backbone_equivariance(
                    backbone, dataset, np.arange(3), "primary", config
                )
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row["equivariance_passed"] is False for row in rows))
            self.assertTrue(all(row["tolerance"] == config.equivariance_atol for row in rows))

    def test_saved_failures_produce_console_and_manifest_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(Config(), output_root=root / "results", array_root=root / "arrays")
            manifest = RunManifest(config)
            tables = SummaryTables(config.output_root / "evaluation")
            row = diagnostics.diagnostic_measurement("example", 2.0, 0.0, 0.01)
            tables.add("numerical_diagnostics.csv", row)
            tables.flush()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                diagnostics.record_diagnostic_warnings([row], "numerical_diagnostics.csv", manifest)
                manifest.warning({"check": "nonfinite_measurement", "error": float("nan")})
            self.assertIn("WARNING:", output.getvalue())
            saved = json.loads(manifest.path.read_text())
            self.assertEqual(saved["warnings"][0]["failed_rows"], 1)
            self.assertEqual(saved["warnings"][1]["error"], "nan")
            frame = pd.read_csv(tables.root / "numerical_diagnostics.csv")
            self.assertEqual(frame.absolute_error.iloc[0], 2.0)
            self.assertFalse(frame.passed.iloc[0])

    def test_empirical_crps_audit_returns_discrepancies(self):
        config = replace(Config(), device="cpu", empirical_gradient_samples=1, image_batch_size=2)
        predictor = FlatMLPProbFactorPredictor().eval()
        slots = np.ones((2, 4, 64), dtype=np.float32)
        pack = {
            "mu": np.zeros((2, 10)),
            "var": np.ones((2, 10)),
            "y": np.zeros((2, 10)),
            "indices": np.arange(2),
        }
        reference = diagnostics.score_integral_reference
        with (
            patch(
                "diagnostics.score_integral_reference",
                side_effect=lambda *args: reference(*args) + 0.1,
            ),
            patch("diagnostics.score_reference_gradients", return_value=(9.0, 9.0)),
        ):
            rows, parameters = diagnostics.empirical_score_audit(predictor, pack, slots, config)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["score_passed"] is False for row in rows))
        self.assertTrue(all(row["gradient_passed"] is False for row in rows))
        self.assertTrue(np.isfinite(parameters["fourier_gradient_norm"]))

    def test_complete_pipeline_saves_failed_diagnostics_and_continues(self):
        backbone_check = diagnostics.backbone_equivariance
        numerical_check = diagnostics.numerical_checks
        empirical_check = diagnostics.empirical_score_audit
        visualize = pipeline.backbone_visualizations
        initialize = inference.evaluate_initialization_conditions

        def failed_backbone(*args):
            rows = backbone_check(*args)
            config = args[-1]
            for row in rows:
                row["equivariance_max_error"] = config.equivariance_atol * 2
                row["equivariance_passed"] = False
            return rows

        def failed_numerical(config, diagnostic_rows):
            rows = numerical_check(config, diagnostic_rows)
            rows[0]["score_error"] = 1.0
            rows[0]["score_passed"] = False
            diagnostic_rows[0]["passed"] = False
            return rows

        def failed_empirical(*args):
            rows, parameters = empirical_check(*args)
            rows[0]["score_error"] = 1.0
            rows[0]["score_passed"] = False
            return rows, parameters

        def check_backbone_saved(backbone, dataset, indices, label, config):
            frame = pd.read_csv(config.output_root / "evaluation/slot_equivariance.csv")
            self.assertEqual(len(frame[frame.backbone == label]), 24)
            self.assertFalse(frame.equivariance_passed.any())
            if label != "primary":
                self.assertEqual(set(frame.backbone_seed.dropna()), {101})
            return visualize(backbone, dataset, indices, label, config)

        def check_empirical_saved(*args, **kwargs):
            config = args[9]
            frame = pd.read_csv(config.output_root / "evaluation/empirical_crps_gradients.csv")
            self.assertTrue((~frame.score_passed).any())
            return initialize(*args, **kwargs)

        def run_and_inspect(path):
            pipeline.run_pipeline(path)
            config = load_config(path)
            manifests = list(
                config.output_root.parent.glob(
                    f"*/{config.output_root.name}/run_manifest.json"
                )
            )
            self.assertEqual(len(manifests), 1)
            saved = json.loads(manifests[0].read_text())
            self.assertEqual(saved["status"], "complete")
            checks = {row["check"] for row in saved["warnings"]}
            self.assertTrue(
                {
                    "slot_equivariance",
                    "pretraining_unit_tests.csv",
                    "numerical_diagnostics.csv",
                    "empirical_crps_gradients.csv",
                }.issubset(checks)
            )

        config = replace(Config(), corruption_severities=(3,))
        with (
            patch("test_experiment.Config", return_value=config),
            patch("test_experiment.run_pipeline", side_effect=run_and_inspect),
            patch("pipeline.backbone_equivariance", side_effect=failed_backbone),
            patch("pipeline.numerical_checks", side_effect=failed_numerical),
            patch("inference.empirical_score_audit", side_effect=failed_empirical),
            patch("pipeline.backbone_visualizations", side_effect=check_backbone_saved),
            patch(
                "inference.evaluate_initialization_conditions", side_effect=check_empirical_saved
            ),
        ):
            case = test_experiment.PipelineIntegrationTests("test_complete_cpu_pipeline")
            case.test_complete_cpu_pipeline()


if __name__ == "__main__":
    unittest.main()
