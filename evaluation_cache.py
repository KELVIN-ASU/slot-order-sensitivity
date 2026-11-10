#### Shared frozen-backbone slot extractions and correspondence measurements.###

import hashlib
import time

import numpy as np
import torch
from tqdm import tqdm

from artifacts import atomic_path, sha256_file
from data import CorruptionDataset, make_loader
from reproducibility import make_device_generator
from training import slots_forward


def synchronize(device):
    # Wait for accelerator work before recording extraction timings.
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class EvaluationCache:
    # Share frozen-backbone computations within one experiment run.

    def __init__(self, config):
        self.config = config
        self.backbones = {}

    def for_model(self, entry):
        """Return shared storage for a frozen backbone and none for joint models."""
        if entry["regime"] != "frozen":
            return None
        path = entry["backbone_checkpoint"]
        if path not in self.backbones:
            digest = sha256_file(path)
            self.backbones[path] = FrozenBackboneCache(digest, self.config)
        return self.backbones[path]


class FrozenBackboneCache:
     # Cache slots and mask diagnostics for a single immutable backbone.

    def __init__(self, digest, config):
        self.config = config
        self.root = config.array_root / "evaluation_slots" / digest
        self.datasets = {}
        self.extractions = {}
        self.correspondence = {}
        self.extraction_count = 0
        self.hit_count = 0

    def condition_key(self, dataset, indices, mode, draw):
        # Identify images, preprocessing, noise streams, and extraction batches.
        base = dataset
        corruption = None
        if isinstance(dataset, CorruptionDataset):
            base = dataset.dataset
            corruption = (
                dataset.corruption,
                dataset.severity,
                dataset.config.corruption_noise_base_seed,
            )
        if base not in self.datasets:
            self.datasets[base] = base.signature()
        settings = (
            self.datasets[base],
            corruption,
            mode,
            draw,
            self.config.slot_init_seed,
            self.config.slot_extract_batch_size,
            self.config.image_size,
            self.config.device,
            "float32",
        )
        digest = hashlib.sha256(repr(settings).encode())
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def get_slots(self, backbone, dataset, indices, mode, draw):
        # Load matching slots or extract them once using the configured stream.
        key = self.condition_key(dataset, indices, mode, draw)
        if key in self.extractions:
            self.hit_count += 1
            return self.extractions[key]
        config = self.config
        path = self.root / f"{key}.npy"
        loader = make_loader(dataset, indices, config.slot_extract_batch_size, config)
        generator = make_device_generator(config.device, draw)
        timings = []
        targets = []
        image_indices = []
        shape = (len(indices), config.num_slots, config.slot_dim)
        with atomic_path(path) as temporary:
            values = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=shape
            )
            offset = 0
            with torch.no_grad():
                for images, target, selected in tqdm(
                    loader,
                    desc=f"Cached slot extraction {mode} draw {draw}",
                    unit="batch",
                    leave=False,
                ):
                    images = images.to(config.device, non_blocking=True)
                    synchronize(config.device)
                    started = time.perf_counter()
                    slots = slots_forward(
                        backbone, images, mode, draw, config, generator
                    )
                    synchronize(config.device)
                    timings.append(time.perf_counter() - started)
                    values[offset : offset + len(images)] = slots.cpu().numpy()
                    targets.append(target.numpy())
                    image_indices.append(selected.numpy())
                    offset += len(images)
            values.flush()
            del values
        record = (
            np.load(path, mmap_mode="r"),
            np.concatenate(targets),
            np.concatenate(image_indices),
            timings,
        )
        self.extractions[key] = record
        self.extraction_count += 1
        return record
