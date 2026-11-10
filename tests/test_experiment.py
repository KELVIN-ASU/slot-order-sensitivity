### Numerical, reproducibility, matching, storage, and integration checks.###

import json
import os
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from artifacts import RunManifest, atomic_json_dump, atomic_path, sha256_file
from calibration import (
    all_intervals,
    conformal_quantile,
    fit_calibration_bundle,
    interval_diagnostics,
    symmetric_interval,
)
from configuration import Config
from data import Causal3DIdentDataset, CorruptionDataset, make_loader
from diagnostics import (
    correspondence_diagnostics,
    hungarian_align_slots,
    numerical_checks,
)
from evaluation_cache import EvaluationCache
from inference import predict_condition, summarize_mixture
from mixtures import mixture_metrics
from models import FlatMLPProbFactorPredictor, SlotAttentionAE
from permutation_scoring import score_permutation, score_permutations
from pipeline import primary_training_conditions, remove_temporary_caches, run_pipeline
from probability import circular_crps_torch
from reporting import prediction_variation
from reproducibility import (
    configure_backend,
    make_device_generator,
    make_slot_init_noise,
    set_global_seed,
)
from statistical_analysis import METRICS, across_seed_tables


class ExperimentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        configure_backend()

    def test_progress_preserves_loader_order_and_generator_state(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        config = replace(Config(), device="cpu", num_workers=0)
        plain = make_loader(dataset, range(20), 4, config, True, 45)
        displayed = make_loader(dataset, range(20), 4, config, True, 45)
        expected = []
        for batch in plain:
            expected.extend(batch[0].tolist())
        actual = []
        for batch in tqdm(displayed, disable=True):
            actual.extend(batch[0].tolist())
        self.assertEqual(actual, expected)
        self.assertTrue(
            torch.equal(plain.generator.get_state(), displayed.generator.get_state())
        )

    def test_frozen_config(self):
        config = Config()
        with self.assertRaises(AttributeError):
            config.split_seed = 1

    def test_training_conditions_preserve_frozen_initialization_comparison(self):
        conditions = primary_training_conditions(Config())
        self.assertEqual(len(conditions), 35)
        self.assertFalse(
            any(
                regime == "joint" and architecture == "flatmlp_stochastic_init"
                for regime, architecture, seed in conditions
            )
        )
        self.assertEqual(
            sum(
                architecture == "flatmlp_stochastic_init"
                for regime, architecture, seed in conditions
            ),
            5,
        )
        config = replace(Config(), predictor_seeds=(45,))
        self.assertEqual(len(primary_training_conditions(config)), 7)

    def test_image_cache_preserves_pixels_and_corruption_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = image_dataset(root / "images")
            expected = [dataset[index][0].clone() for index in range(len(dataset))]
            config = replace(Config(), device="cpu")
            for shared in (True, False):
                cached = Causal3DIdentDataset(dataset.root, 64)
                cached.cache_images(root / "pixels.npy", shared)
                corrupted = CorruptionDataset(cached, "gaussian_noise", 3, config)
                for index, pixels in enumerate(expected):
                    torch.testing.assert_close(cached[index][0], pixels, rtol=0, atol=0)
                    generator = torch.Generator().manual_seed(100047 + index)
                    noise = torch.randn(pixels.shape, generator=generator)
                    target = torch.clamp(pixels + 0.09 * noise, 0, 1)
                    torch.testing.assert_close(
                        corrupted[index][0], target, rtol=0, atol=0
                    )

    def test_frozen_slot_reuse_matches_direct_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = image_dataset(root / "images")
            config = replace(
                Config(),
                device="cpu",
                num_workers=0,
                array_root=root / "arrays",
                slot_extract_batch_size=2,
            )
            backbone = SlotAttentionAE().eval()
            predictor = FlatMLPProbFactorPredictor().eval()
            checkpoint = root / "backbone.pth"
            torch.save(backbone.state_dict(), checkpoint)
            store = EvaluationCache(config)
            entry = {"regime": "frozen", "backbone_checkpoint": str(checkpoint)}
            cache = store.for_model(entry)
            self.assertIs(store.for_model(entry), cache)
            self.assertIsNone(store.for_model(dict(entry, regime="joint")))
            indices = np.array([2, 0, 1])
            for mode in config.initialization_modes:
                direct = predict_condition(
                    backbone, predictor, dataset, indices, mode, 7, config
                )[0]
                for repeat in range(2):
                    reused = predict_condition(
                        backbone,
                        predictor,
                        dataset,
                        indices,
                        mode,
                        7,
                        config,
                        cache=cache,
                    )[0]
                    for key in direct:
                        np.testing.assert_array_equal(direct[key], reused[key])
            self.assertEqual(cache.extraction_count, 3)
            self.assertEqual(cache.hit_count, 3)
            first_key = cache.condition_key(
                dataset, indices, "per_example_stochastic", 7
            )
            second_key = cache.condition_key(
                dataset, indices[::-1], "per_example_stochastic", 7
            )
            self.assertNotEqual(first_key, second_key)
            self.assertNotEqual(
                first_key,
                cache.condition_key(dataset, indices, "per_example_stochastic", 13),
            )

    def test_sequential_scoring_preserves_order_and_results(self):
        generator = np.random.default_rng(47)
        means = generator.normal(size=(4, 24, 10)).astype(np.float32)
        variance = np.exp(generator.normal(size=means.shape)).astype(np.float32)
        target = generator.normal(size=(4, 10)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(Config(), array_root=Path(directory))
            serial = score_permutations(means, variance, target, True, config)
            expected = []
            for index in range(24):
                expected.append(
                    score_permutation(means, variance, target, index, True, config)
                )
            self.assertEqual(len(serial), 24)
            for first, second in zip(serial, expected):
                for first_values, second_values in zip(first[:4], second[:4]):
                    if isinstance(first_values, dict):
                        for key in first_values:
                            np.testing.assert_array_equal(
                                first_values[key], second_values[key]
                            )
                    else:
                        np.testing.assert_array_equal(first_values, second_values)
                self.assertEqual(first[4], second[4])

    def test_random_streams_match_source_functions(self):
        config = Config()
        namespace = {
            "torch": torch,
            "np": np,
            "random": random,
            "os": os,
            "CFG": config,
        }
        path = Path(__file__).with_name("random_stream_reference.txt")
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        namespace["set_global_seed"](46)
        expected = (random.random(), np.random.random(), torch.randn(5))
        set_global_seed(46)
        actual = (random.random(), np.random.random(), torch.randn(5))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        for dtype in (torch.float32, torch.float64):
            features = torch.zeros(3, 7, 64, dtype=dtype)
            for mode in config.initialization_modes:
                expected_stream = namespace["make_device_generator"]("cpu", 7)
                actual_stream = make_device_generator("cpu", 7)
                for step in range(2):
                    expected_noise = namespace["make_slot_init_noise"](
                        features, mode, 7, expected_stream
                    )
                    actual_noise = make_slot_init_noise(
                        features, mode, 7, config, actual_stream
                    )
                    torch.testing.assert_close(
                        expected_noise, actual_noise, rtol=0, atol=0
                    )

    def test_numerical_values_and_gradients(self):
        config = replace(Config(), device="cpu")
        rows = numerical_checks(config)
        self.assertEqual(len(rows), 152)
        self.assertLess(max(row["score_error"] for row in rows), 2e-7)

    def test_periodic_score_and_clamp(self):
        mu = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0.2], dtype=torch.float64)
        for logvar_value in (-12.0, 12.0):
            logvar = torch.tensor(
                [logvar_value], dtype=torch.float64, requires_grad=True
            )
            score = circular_crps_torch(mu, logvar, target)
            shifted = circular_crps_torch(mu + np.pi, logvar, target)
            torch.testing.assert_close(score, shifted)
            gradient = torch.autograd.grad(score, logvar)[0]
            self.assertEqual(gradient.item(), 0)

    def test_cosine_assignment_does_not_use_magnitude(self):
        reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        candidate = torch.tensor([[[80.0, 60.0], [0.1, 0.0]]])
        aligned, assignments, costs, margins = hungarian_align_slots(
            reference, candidate
        )
        np.testing.assert_array_equal(assignments, [[1, 0]])
        self.assertAlmostEqual(costs[0], 0.2, places=6)
        self.assertTrue(np.all(margins > 0))

    def test_initialization_draws_and_stream_progression(self):
        config = Config()
        features = torch.zeros(3, 7, 64)
        first = make_slot_init_noise(features, "fixed_shared", 1234, config)
        second = make_slot_init_noise(features, "fixed_shared", 1234, config)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[0], first[1], rtol=0, atol=0)
        generator = make_device_generator("cpu", 7)
        first = make_slot_init_noise(
            features, "per_example_stochastic", 7, config, generator
        )
        second = make_slot_init_noise(
            features, "per_example_stochastic", 7, config, generator
        )
        self.assertFalse(torch.equal(first, second))
        generator = make_device_generator("cpu", 7)
        replay = make_slot_init_noise(
            features, "per_example_stochastic", 7, config, generator
        )
        torch.testing.assert_close(first, replay, rtol=0, atol=0)

    def test_fixed_random_assignment_reuses_vectors(self):
        config = Config()
        features = torch.zeros(3, 7, 64)
        reference = make_slot_init_noise(
            features, "fixed_shared", config.slot_init_seed, config
        )
        candidate = make_slot_init_noise(features, "fixed_random_assignment", 7, config)
        aligned, assignments, costs, margins = hungarian_align_slots(
            reference, candidate
        )
        torch.testing.assert_close(reference, aligned)

    def test_per_image_change_can_hide_in_dataset_rmse(self):
        means = np.zeros((2, 24, 10))
        means[0, :, 0] = np.tile([-1.0, 1.0], 12)
        means[1, :, 0] = -means[0, :, 0]
        aggregate = np.sqrt((means**2).mean((0, 2)))
        np.testing.assert_allclose(aggregate, aggregate[0])
        change = prediction_variation(means, np.ones_like(means), np.ones(10))
        np.testing.assert_allclose(change["maximum_pairwise_mean_change"][:, 0], 2)

    def test_full_circle_and_conformal_convention(self):
        means = np.zeros((2, 10))
        lower, upper, full = symmetric_interval(means, np.ones_like(means) * 100, 0.9)
        diagnostics = interval_diagnostics(lower, upper, means, 0.9, full)
        np.testing.assert_allclose(diagnostics["width"][:, 7:], np.pi)
        self.assertTrue(diagnostics["inside"][:, 7:].all())
        scores = np.arange(20).reshape(10, 2)
        expected = np.quantile(scores, 1, axis=0, method="higher")
        np.testing.assert_allclose(conformal_quantile(scores, 0.1), expected)

    def test_mixture_order_and_reference(self):
        generator = np.random.default_rng(47)
        means = generator.normal(size=(2, 24, 10))
        variance = np.exp(generator.uniform(-10, 10, means.shape))
        target = generator.normal(size=(2, 10))
        first = mixture_metrics(means, variance, target)
        second = mixture_metrics(means[:, ::-1], variance[:, ::-1], target)
        for key in first:
            np.testing.assert_allclose(
                first[key], second[key], atol=1e-9, equal_nan=True
            )
        for image in range(2):
            for factor in range(7, 10):
                from diagnostics import distance_integral_reference

                component_mean = means[image, :, factor]
                component_variance = variance[image, :, factor]
                first_expectation = []
                second_expectation = []
                for left, left_variance in zip(component_mean, component_variance):
                    first_expectation.append(
                        distance_integral_reference(
                            left - target[image, factor], np.sqrt(left_variance)
                        )
                    )
                    for right, right_variance in zip(
                        component_mean, component_variance
                    ):
                        second_expectation.append(
                            distance_integral_reference(
                                left - right, np.sqrt(left_variance + right_variance)
                            )
                        )
                reference = np.mean(first_expectation) - 0.5 * np.mean(
                    second_expectation
                )
                self.assertAlmostEqual(
                    first["crps"][image, factor], reference, places=8
                )

    def test_repeated_distribution_does_not_change_mixture_metrics(self):
        generator = np.random.default_rng(47)
        means = generator.normal(size=(3, 1, 10))
        variance = np.exp(generator.normal(size=means.shape))
        target = generator.normal(size=(3, 10))
        single = mixture_metrics(means, variance, target)
        repeated = mixture_metrics(
            np.repeat(means, 24, axis=1), np.repeat(variance, 24, axis=1), target
        )
        for key in single:
            np.testing.assert_allclose(
                single[key], repeated[key], atol=1e-8, equal_nan=True
            )

    def test_single_manifest_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                Config(), output_root=root / "output", array_root=root / "arrays"
            )
            manifest = RunManifest(config)
            with manifest.stage("example"):
                pass
            self.assertEqual(
                list(config.output_root.rglob("*.json")),
                [config.output_root / "run_manifest.json"],
            )
            with self.assertRaises(FileExistsError):
                RunManifest(config)

    def test_atomic_output_survives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text("retained")
            with self.assertRaises(RuntimeError):
                with atomic_path(path) as temporary:
                    temporary.write_text("incomplete")
                    raise RuntimeError("test interruption")
            self.assertEqual(path.read_text(), "retained")

    def test_cache_cleanup_closes_mappings_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(Config(), array_root=root / "arrays")
            dataset = image_dataset(root / "images")
            dataset.cache_images(
                config.array_root / "image_cache" / "pixels.npy", False
            )
            image_mapping = dataset.resized_images._mmap
            checkpoint = root / "backbone.pth"
            checkpoint.write_bytes(b"backbone identity")
            cache = EvaluationCache(config)
            frozen = cache.for_model(
                {"regime": "frozen", "backbone_checkpoint": str(checkpoint)}
            )
            path = config.array_root / "evaluation_slots" / "slots.npy"
            path.parent.mkdir(parents=True)
            np.save(path, np.zeros((3, 4, 64), dtype=np.float32))
            slots = np.load(path, mmap_mode="r")
            slot_mapping = slots._mmap
            frozen.extractions["condition"] = (slots, None, None, [])
            for name in ("slot_cache", "scoring_arrays"):
                (config.array_root / name).mkdir()
            retained = config.array_root / "prediction_change.npz"
            retained.write_bytes(b"retained result")
            remove_temporary_caches(cache, (dataset,), config)
            self.assertTrue(image_mapping.closed)
            self.assertTrue(slot_mapping.closed)
            self.assertEqual(retained.read_bytes(), b"retained result")
            self.assertTrue(checkpoint.is_file())
            for name in (
                "slot_cache",
                "evaluation_slots",
                "image_cache",
                "scoring_arrays",
            ):
                self.assertFalse((config.array_root / name).exists())

    def test_every_calibration_method_is_present(self):
        generator = np.random.default_rng(47)
        mu = generator.normal(scale=0.2, size=(12, 10))
        variance = np.ones_like(mu) * 0.3
        target = generator.uniform(-1, 1, mu.shape)
        config = replace(Config(), device="cpu", temperature_steps=2)
        bundle = fit_calibration_bundle(
            {"mu": mu, "var": variance, "y": target}, config
        )
        intervals = all_intervals(mu, variance, bundle, config.interval_level)
        self.assertEqual(
            set(intervals),
            {
                "raw_gaussian",
                "scalar_temperature",
                "per_factor_temperature",
                "isotonic_cdf",
                "absolute_conformal",
                "gaussian_split_conformal",
            },
        )
        for lower, upper, full in intervals.values():
            values = interval_diagnostics(
                lower, upper, target, config.interval_level, full
            )
            self.assertTrue(np.all(values["width"] >= 0))
            self.assertTrue(np.all(values["width"][:, 7:] <= np.pi))

    def test_initialization_contrasts_pair_training_seeds(self):
        rows = []
        for seed in (45, 46):
            for architecture, fixed, stochastic in (
                ("flatmlp", 1.0, 2.0),
                ("flatmlp_stochastic_init", 0.8, 0.9),
            ):
                for mode, error in (
                    ("fixed_shared", fixed),
                    ("per_example_stochastic", stochastic),
                ):
                    row = {
                        "group": "primary",
                        "regime": "frozen",
                        "architecture": architecture,
                        "train_seed": seed,
                        "split": "working",
                        "mode": mode,
                        "variant": "raw",
                        "readout": "identity",
                    }
                    for metric in METRICS:
                        row[metric] = error + 0.01 * (seed - 45)
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                Config(), output_root=Path(directory), seed_bootstrap_repetitions=10
            )
            across_seed_tables(rows, config)
            import pandas as pd

            values = pd.read_csv(
                config.output_root / "evaluation/initialization_2x2_contrasts.csv"
            )
            selected = values[
                (values.metric == "rmse")
                & (values.contrast == "training_evaluation_interaction")
            ]
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(selected.difference.iloc[0], -0.9)
            self.assertEqual(selected.n_seeds.iloc[0], 2)


class PipelineIntegrationTests(unittest.TestCase):

    def test_complete_cpu_pipeline(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = np.random.default_rng(47)
            for partition, count in (("trainset", 30), ("testset", 4)):
                data = root / "data" / partition
                images = data / "images_0"
                images.mkdir(parents=True)
                labels = generator.uniform(-1, 1, (count, 10)).astype(np.float32)
                np.save(data / "latents_0.npy", labels)
                for index in range(count):
                    pixels = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(images / f"{index:06d}.png")
            checkpoint = root / "input_backbone.pth"
            torch.save({"model_state_dict": SlotAttentionAE().state_dict()}, checkpoint)
            config = replace(
                Config(),
                slot_checkpoint=checkpoint,
                extract_root=root / "data",
                train_tar=root / "train.tar",
                test_tar=root / "test.tar",
                output_root=root / "results",
                array_root=root / "arrays",
                device="cpu",
                factor_schema_confirmed=True,
                predictor_seeds=(45,),
                additional_backbone_seeds=(101,),
                replication_predictor_seeds=(45,),
                evaluation_draw_seeds=(7,),
                score_comparison_seeds=(45,),
                score_comparison_epochs=1,
                predictor_epochs=2,
                backbone_epochs=1,
                backbone_batch_size=4,
                predictor_batch_size=8,
                image_batch_size=4,
                slot_extract_batch_size=4,
                metric_batch_size=2,
                temperature_steps=2,
                num_workers=0,
                symmetry_sample_size=1,
                stability_max_images=1,
                empirical_gradient_samples=1,
                permutation_bootstrap_repetitions=2,
                seed_bootstrap_repetitions=4,
            )
            path = root / "config.json"
            atomic_json_dump(asdict(config), path)
            with (
                patch(
                    "inference.correspondence_diagnostics",
                    wraps=correspondence_diagnostics,
                ) as measured,
                patch(
                    "inference.predict_condition", wraps=predict_condition
                ) as predictions,
                patch(
                    "inference.summarize_mixture", wraps=summarize_mixture
                ) as mixtures,
            ):
                run_pipeline(path)
            manifests = list(
                config.output_root.parent.glob(
                    f"*/{config.output_root.name}/run_manifest.json"
                )
            )
            self.assertEqual(len(manifests), 1)
            payload = json.loads(manifests[0].read_text())
            config = replace(
                config,
                output_root=manifests[0].parent,
                array_root=Path(payload["array_root"]),
            )
            self.assertEqual(measured.call_count, 20)
            corruption_calls = []
            for call in predictions.call_args_list:
                if isinstance(call.args[2], CorruptionDataset):
                    corruption_calls.append(call)
            corruption_count = len(config.corruption_types) * len(
                config.corruption_severities
            )
            self.assertEqual(len(corruption_calls), 8 * corruption_count)
            self.assertTrue(
                all(call.kwargs["orbit"] is False for call in corruption_calls)
            )
            self.assertEqual(mixtures.call_count, 8 * 2 * 2 * 2)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["coverage"]["evaluated_models"], 8)
            self.assertEqual(payload["coverage"]["primary_models"], 7)
            self.assertEqual(len(list(config.output_root.rglob("*.json"))), 1)
            for record in payload["checkpoints"]:
                self.assertEqual(sha256_file(record["path"]), record["sha256"])
            self.assertEqual(sha256_file(checkpoint), payload["inputs"][0]["sha256"])
            for folder in (config.array_root / "initialization_mechanism").iterdir():
                self.assertEqual(len(list(folder.glob("*_distributions.npz"))), 4)
            self.assertEqual(
                len(list((config.array_root / "audit_predictions").glob("*.npz"))), 7
            )
            import pandas as pd

            mixtures_table = pd.read_csv(
                config.output_root / "evaluation/predictive_mixture_metrics.csv"
            )
            ensemble = mixtures_table[mixtures_table.readout == "ensemble24"]
            self.assertEqual(len(ensemble), 8 * 2 * 2)
            ordinary = mixtures_table[mixtures_table.readout != "ensemble24"]
            self.assertEqual(len(ordinary), 8 * 4 * 3 * 2)
            self.assertEqual(
                set(ordinary.variant),
                {"raw", "scalar_temperature", "per_factor_temperature"},
            )
            intervals = pd.read_csv(
                config.output_root / "evaluation/distribution_interval_comparison.csv"
            )
            self.assertEqual(len(intervals), 8 * 2 * 2 * 2)
            for frame in (ensemble, intervals):
                self.assertEqual(set(frame.variant), {"raw", "per_factor_temperature"})
                self.assertEqual(
                    set(frame["mode"]), {"fixed_shared", "per_example_stochastic"}
                )
                fixed = frame[frame["mode"] == "fixed_shared"]
                self.assertEqual(set(fixed.slot_init_seed), {1234})
            permutations = pd.read_csv(
                config.output_root / "evaluation/complete_slot_permutation_table.csv"
            )
            self.assertEqual(len(permutations), 8 * 4 * 3 * 24)
            corruption = pd.read_csv(
                config.output_root / "evaluation/exact_corruption_level_metrics.csv"
            )
            self.assertEqual(set(corruption.readout), {"identity"})
            self.assertEqual(len(corruption), 8 * corruption_count * 3)
            self.assertEqual(
                set(corruption.severity), set(config.corruption_severities)
            )
            variation = pd.read_csv(
                config.output_root / "evaluation/prediction_variation_by_factor.csv"
            )
            sigma_columns = [
                "sigma_normalized_change_p50",
                "sigma_normalized_change_p95",
                "fraction_above_sigma_tolerance",
            ]
            self.assertTrue(np.isfinite(variation[sigma_columns].to_numpy()).all())
            draw_variation = pd.read_csv(
                config.output_root / "evaluation/draw_variation_by_factor.csv"
            )
            self.assertEqual(len(draw_variation), 8 * 10)
            self.assertEqual(set(draw_variation.status), {"insufficient_draws"})
            initialization_comparison = pd.read_csv(
                config.output_root
                / "evaluation/permutation_change_vs_initialization_variation.csv"
            )
            self.assertEqual(len(initialization_comparison), 8 * 10)
            primary = initialization_comparison[
                initialization_comparison.group == "primary"
            ]
            self.assertEqual(set(primary.comparison_status_p50), {"insufficient_draws"})
            replication = initialization_comparison[
                initialization_comparison.group != "primary"
            ]
            self.assertEqual(
                set(replication.comparison_status_p50), {"missing_reference"}
            )
            comparison = pd.read_csv(
                config.output_root / "evaluation/crps_reporting_changes.csv"
            )
            self.assertEqual(len(comparison), 8 * 24 * 2 * 2)
            self.assertEqual(set(comparison["mode"]), {"fixed_shared"})
            self.assertEqual(set(comparison.slot_init_seed), {1234})
            self.assertEqual(set(comparison.variant), {"raw", "per_factor_temperature"})
            checkpoints = list((config.output_root / "checkpoints").rglob("*.pth"))
            self.assertEqual(len(checkpoints), 11)
            self.assertTrue(all(path.name == "best.pth" for path in checkpoints))
            for entry in payload["models"]:
                directory = config.array_root / "validation_predictions" / entry["id"]
                predictions = list(directory.glob("*.npz"))
                if entry["group"].startswith("score_"):
                    self.assertEqual(
                        [path.name for path in predictions],
                        [f"epoch_{entry['selected_epoch']:04d}.npz"],
                    )
                else:
                    self.assertEqual(predictions, [])
                history = pd.read_csv(
                    config.output_root
                    / "evaluation/training"
                    / entry["id"]
                    / "history.csv"
                )
                expected_epochs = config.predictor_epochs
                if entry["group"].startswith("score_"):
                    expected_epochs = config.score_comparison_epochs
                self.assertEqual(len(history), expected_epochs)
            for folder in (config.array_root / "raw_predictions").iterdir():
                self.assertEqual(
                    set(path.name for path in folder.glob("*.npz")),
                    {
                        "calibration_predictions.npz",
                        "fixed_shared_1234.npz",
                        "per_example_stochastic_7.npz",
                        "draw_variation.npz",
                    },
                )
                for name in ("fixed_shared_1234.npz", "per_example_stochastic_7.npz"):
                    with np.load(folder / name, allow_pickle=False) as pack:
                        self.assertEqual(
                            set(pack.files),
                            {"mu", "var", "y", "indices", "permutations"},
                        )
                        self.assertEqual(pack["mu"].shape[1], 24)
            for path in config.array_root.rglob("*.npz"):
                self.assertFalse(
                    path.name.endswith(
                        ("_mixture.npz", "_aligned.npz", "_uncertainty_change.npz")
                    )
                )
                if "permutation_evaluation" in path.parts:
                    self.assertTrue(path.name.endswith("_prediction_change.npz"))
            for name in (
                "slot_cache",
                "evaluation_slots",
                "image_cache",
                "scoring_arrays",
            ):
                self.assertFalse((config.array_root / name).exists())
            stages = list(payload["stages"])
            self.assertLess(
                stages.index("final_validation"), stages.index("cache_cleanup")
            )
            for record in payload["outputs"]:
                self.assertTrue(Path(record["path"]).is_file())

    def test_single_severity_cpu_pipeline(self):
        config = replace(Config(), corruption_severities=(3,))
        with patch(f"{__name__}.Config", return_value=config):
            self.test_complete_cpu_pipeline()


def image_dataset(root):
    """Create differently sized RGB images with paired factor labels."""
    images = root / "images_0"
    images.mkdir(parents=True)
    generator = np.random.default_rng(47)
    np.save(
        root / "latents_0.npy", generator.uniform(-1, 1, (3, 10)).astype(np.float32)
    )
    for index in range(3):
        pixels = generator.integers(0, 256, (71 + index, 83, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{index:06d}.png")
    return Causal3DIdentDataset(root, 64)


if __name__ == "__main__":
    unittest.main()
