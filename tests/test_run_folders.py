## Run identity, directory reservation, and manifest checks.

import json
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from artifacts import RunManifest, configuration_digest, create_run_config
from configuration import Config, load_config

#  Check output isolation without changing the configuration.
class RunFolderTests(unittest.TestCase):

    def test_hash_excludes_output_paths(self):
        config = Config()
        moved = replace(
            config, output_root=Path("other/results"), array_root=Path("other/arrays")
        )
        self.assertEqual(configuration_digest(config), configuration_digest(moved))
        changed = replace(config, predictor_epochs=config.predictor_epochs + 1)
        self.assertNotEqual(configuration_digest(config), configuration_digest(changed))

    def test_matching_parent_names_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                Config(),
                output_root=root / "results" / "flatmlp_mechanism",
                array_root=root / "arrays" / "flatmlp_mechanism",
            )
            config.output_root.mkdir(parents=True)
            retained = config.output_root / "measurement.csv"
            retained.write_text("value\n1\n")
            resolved, information = create_run_config(config)
            self.assertEqual(resolved.output_root.parent.name, information["id"])
            self.assertEqual(resolved.array_root.parent.name, information["id"])
            self.assertEqual(resolved.output_root.name, config.output_root.name)
            self.assertEqual(resolved.array_root.name, config.array_root.name)
            self.assertRegex(information["id"], r"^\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$")
            self.assertEqual(configuration_digest(resolved), configuration_digest(config))
            self.assertEqual(
                replace(resolved, output_root=config.output_root, array_root=config.array_root),
                config,
            )
            manifest = RunManifest(resolved, information)
            saved = json.loads(manifest.path.read_text())
            self.assertEqual(saved["run"], information)
            self.assertEqual(saved["config"]["output_root"], str(resolved.output_root))
            self.assertEqual(saved["config"]["array_root"], str(resolved.array_root))
            self.assertEqual(saved["array_root"], str(resolved.array_root))
            self.assertEqual(retained.read_text(), "value\n1\n")
            self.assertEqual(list(resolved.output_root.rglob("*.json")), [manifest.path])

    def test_common_parent_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(Config(), output_root=root / "results", array_root=root / "arrays")
            started = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
            with patch("artifacts.datetime") as clock:
                clock.now.return_value = started
                resolved, information = create_run_config(config)
                self.assertEqual(resolved.output_root.parent, resolved.array_root.parent)
                retained = resolved.output_root.parent / "measurement.csv"
                retained.write_text("value\n1\n")
                with self.assertRaises(FileExistsError):
                    create_run_config(config)
            self.assertEqual(retained.read_text(), "value\n1\n")
            self.assertEqual(resolved.output_root.parent.name, information["id"])

    def test_array_parent_collision_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                Config(), output_root=root / "results" / "study", array_root=root / "arrays" / "study"
            )
            started = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
            timestamp = started.astimezone().strftime("%Y%m%d_%H%M%S_%f")
            run_id = f"{timestamp}_{configuration_digest(config)[:8]}"
            existing = config.array_root.parent / run_id
            existing.mkdir(parents=True)
            retained = existing / "measurement.csv"
            retained.write_text("value\n1\n")
            with patch("artifacts.datetime") as clock:
                clock.now.return_value = started
                with self.assertRaises(FileExistsError):
                    create_run_config(config)
            self.assertEqual(retained.read_text(), "value\n1\n")
            self.assertFalse((config.output_root.parent / run_id).exists())

    def test_directory_creation_preserves_random_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(Config(), output_root=root / "results", array_root=root / "arrays")
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state().clone()
            create_run_config(config)
            self.assertEqual(random.getstate(), python_state)
            after = np.random.get_state()
            self.assertEqual(after[0], numpy_state[0])
            np.testing.assert_array_equal(after[1], numpy_state[1])
            self.assertEqual(after[2:], numpy_state[2:])
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_package_configuration(self):
        config = load_config(Path(__file__).parents[1] / "config.json")
        self.assertEqual(config.predictor_seeds, (45, 46, 47, 48, 49))
        self.assertEqual(config.additional_backbone_seeds, (101, 202, 303))
        self.assertEqual(config.backbone_epochs, 50)
        self.assertEqual(config.predictor_epochs, 20)
        self.assertEqual(config.evaluation_draw_seeds, (7, 13, 99, 2026))
        self.assertEqual(config.replication_predictor_seeds, (45,))
        self.assertEqual(config.score_comparison_seeds, (45, 46))
        self.assertEqual(config.score_comparison_epochs, 5)


if __name__ == "__main__":
    unittest.main()
