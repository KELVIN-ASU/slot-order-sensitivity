"""Causal3DIdent archives, split indices, and corruption definitions."""

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from artifacts import atomic_csv_save, atomic_npz_save, atomic_path, sha256_file
from reproducibility import make_generator, seed_worker

FACTOR_NAMES = (
    "object_position_x",
    "object_position_y",
    "object_position_z",
    "object_rotation_alpha",
    "object_rotation_beta",
    "object_rotation_gamma",
    "spotlight_position",
    "object_hue",
    "spotlight_hue",
    "background_hue",
)
BLUR_SIGMA = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}
BRIGHTNESS_DELTA = {1: 0.15, 2: 0.30, 3: 0.45, 4: 0.60, 5: 0.75}
NOISE_STD = {1: 0.03, 2: 0.06, 3: 0.09, 4: 0.12, 5: 0.15}


def extract_archive(path, destination):
    """Extract an archive after rejecting links and paths outside its destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path outside destination: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported archive member: {member.name}")
        archive.extractall(destination, filter="data")


def load_datasets(config):
    """Load paired shards in lexical order, extracting archives when necessary."""
    roots = []
    pairs = (("trainset", config.train_tar), ("testset", config.test_tar))
    for name, archive in pairs:
        root = config.extract_root / name
        if not list(root.glob("latents_*.npy")):
            extract_archive(archive, config.extract_root)
        roots.append(Causal3DIdentDataset(root, config.image_size))
    return roots[0], roots[1]


def split_indices(train_count, test_count, config):
    """Partition training and test indices using separate seeded generators."""
    order = torch.randperm(train_count, generator=make_generator(config.split_seed))
    validation_size = int(config.validation_fraction * train_count)
    training_size = train_count - validation_size
    selection_size = int(config.val_cal_fraction * validation_size)
    test_order = np.random.default_rng(config.audit_split_seed).permutation(test_count)
    splits = {
        "train": order[:training_size].numpy(),
        "validation": order[training_size : training_size + selection_size].numpy(),
        "calibration": order[training_size + selection_size :].numpy(),
        "working": np.sort(test_order[test_count // 2 :]),
        "audit": np.sort(test_order[: test_count // 2]),
    }
    for names, total in (
        (("train", "validation", "calibration"), train_count),
        (("working", "audit"), test_count),
    ):
        members = []
        for name in names:
            if len(splits[name]) == 0:
                raise ValueError(f"Empty split: {name}")
            members.append(splits[name])
        joined = np.concatenate(members)
        if len(joined) != total or len(np.unique(joined)) != total:
            raise ValueError("Dataset splits overlap or do not cover the archive.")
    return splits


def make_loader(dataset, indices, batch_size, config, shuffle=False, seed=None):
    """Construct a loader with explicit worker and generator seeds."""
    if seed is None:
        seed = config.split_seed
    selected = []
    for index in indices:
        selected.append(int(index))
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=config.num_workers > 0,
        generator=make_generator(seed),
    )


def image_cache_budget(config):
    """Bound shared image storage by configured, available, and shared memory."""
    budget = int(config.image_cache_max_gib * 1024**3)
    memory_information = Path("/proc/meminfo")
    if memory_information.is_file():
        for line in memory_information.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                budget = min(budget, available // 2)
                break
    shared_directory = Path("/dev/shm")
    if os.name == "posix" and shared_directory.is_dir():
        budget = min(budget, shutil.disk_usage(shared_directory).free // 2)
    return budget


def cache_dataset_images(train, test, config):
    """Decode and resize images once, sharing compact pixels across loader workers."""
    required = (len(train) + len(test)) * config.image_size**2 * 3
    shared = required <= image_cache_budget(config)
    for name, dataset in (("train", train), ("test", test)):
        dataset.cache_images(config.array_root / "image_cache" / f"{name}.npy", shared)
    storage = "shared RAM"
    if not shared:
        storage = "memory-mapped disk arrays"
    print(f"Resized image cache: {required / 1024**3:.2f} GiB, {storage}", flush=True)


def audit_data(train, test, splits, config, manifest):
    """Check factor ranges, reference split indices, image mapping, and duplicate bytes."""

    if not config.factor_schema_confirmed:
        raise ValueError(
            "Confirm factor semantics and image-label mapping in config.json."
        )
    if config.original_split_file is not None:
        prior = json.loads(config.original_split_file.read_text())
        manifest.input("test_split_record", config.original_split_file)
        if prior.get("full_test_size", len(test)) != len(test):
            raise ValueError("The original split record has a different test-set size.")
        if config.test_tar.is_file() and "test_archive_sha256" in prior:
            if sha256_file(config.test_tar) != prior["test_archive_sha256"]:
                raise ValueError(
                    "The test archive differs from the original split record."
                )
        for field, split in (
            ("audit_indices", "audit"),
            ("working_test_indices", "working"),
        ):
            if field not in prior or not np.array_equal(prior[field], splits[split]):
                raise ValueError(f"Original split differs: {field}")
    atomic_npz_save(config.array_root / "splits.npz", **splits)
    rows = []
    hashes = {}
    duplicates = []
    hash_file = config.array_root / "image_hashes.csv"
    with atomic_path(hash_file) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("partition,image_id,sha256\n")
            for partition, dataset in (("train", train), ("test", test)):
                for index, path in enumerate(dataset.image_paths):
                    digest = sha256_file(path)
                    if digest in hashes and hashes[digest][0] != partition:
                        duplicates.append((hashes[digest], (partition, index)))
                    hashes.setdefault(digest, (partition, index))
                    handle.write(f"{partition},{dataset.image_ids[index]},{digest}\n")
                for factor, name in enumerate(FACTOR_NAMES):
                    column = dataset.latents[:, factor]
                    quantiles = np.quantile(column, [0.001, 0.01, 0.5, 0.99, 0.999])
                    row = {
                        "dataset": partition,
                        "index": factor,
                        "factor": name,
                        "min": float(column.min()),
                        "max": float(column.max()),
                        "q0001": float(quantiles[0]),
                        "q01": float(quantiles[1]),
                        "median": float(quantiles[2]),
                        "q99": float(quantiles[3]),
                        "q0999": float(quantiles[4]),
                        "unique": len(np.unique(column)),
                        "geometry": "linear",
                        "period": None,
                    }
                    if factor >= 7:
                        row["geometry"] = "circular"
                        row["period"] = np.pi
                        span = quantiles[-1] - quantiles[0]
                        distance = abs((span + np.pi / 2) % np.pi - np.pi / 2)
                        row["boundary_ratio"] = float(span / max(distance, 1e-12))
                    rows.append(row)
    evaluation = config.output_root / "evaluation"
    atomic_csv_save(rows, evaluation / "raw_column_statistics.csv")
    atomic_csv_save(rows, evaluation / "factor_encoding_audit.csv")
    range_rows = []
    for partition, dataset in (("train", train), ("test", test)):
        spans = np.ptp(dataset.latents, axis=0)
        ratio = spans[6] / max(float(spans[3:6].mean()), 1e-12)
        range_rows.append(
            {"dataset": partition, "spotlight_to_rotation_range_ratio": ratio}
        )
    atomic_csv_save(range_rows, evaluation / "spotlight_range_ratio_check.csv")
    if duplicates:
        raise ValueError(
            "Byte-identical images occur in both training and test archives."
        )
    counts = []
    for name, indices in splits.items():
        counts.append({"split": name, "n_samples": len(indices)})
    atomic_csv_save(counts, evaluation / "dataset_split_counts.csv")
    manifest.payload["data"] = {
        "train_signature": train.signature(),
        "test_signature": test.signature(),
        "byte_duplicate_check": "passed",
        "factor_geometry": "linear 0-6; circular 7-9, period pi",
        "image_label_order": "lexical paired shards; numeric filename order checked",
        "limitation": "Byte hashes do not detect transformed near-duplicates.",
    }
    manifest.write()


class Causal3DIdentDataset(Dataset):
    # Read paired image shards and ten continuous factors in lexical order.

    def __init__(self, root, image_size):
        self.root = Path(root)
        self.image_size = image_size
        self.resized_images = None
        self.image_cache_path = None
        self.image_paths = []
        labels = []
        latent_paths = sorted(self.root.glob("latents_*.npy"))
        expected_image_folders = set()
        for latent_path in latent_paths:
            expected_image_folders.add("images_" + latent_path.stem[len("latents_") :])
        actual_image_folders = set()
        for image_folder in self.root.glob("images_*"):
            if image_folder.is_dir():
                actual_image_folders.add(image_folder.name)
        if expected_image_folders != actual_image_folders:
            raise ValueError("Unpaired image or latent shards.")
        for latent_path in latent_paths:
            shard = latent_path.stem[len("latents_") :]
            paths = sorted((self.root / f"images_{shard}").glob("*.png"))
            values = np.load(latent_path, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != 10 or len(paths) != len(values):
                raise ValueError(f"Image-label mismatch: {latent_path}")
            stems = []
            for path in paths:
                stems.append(path.stem)
            if all(stem.isdigit() for stem in stems):
                if stems != sorted(stems, key=int):
                    raise ValueError("Lexical and numeric image order disagree.")
            self.image_paths.extend(paths)
            labels.append(values.astype(np.float32))
        if not labels:
            raise FileNotFoundError(f"No paired shards under {root}")
        self.latents = np.concatenate(labels)
        self.image_ids = []
        for path in self.image_paths:
            self.image_ids.append(path.relative_to(self.root).as_posix())
        self.image_ids = np.asarray(self.image_ids)
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )
        if not np.isfinite(self.latents).all():
            raise ValueError("Nonfinite latent factors.")
        if np.any(np.abs(self.latents[:, 7:]) > np.pi / 2 + 1e-5):
            raise ValueError("Hue factors outside the configured support.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.resized_images is None and self.image_cache_path is not None:
            self.resized_images = np.load(self.image_cache_path, mmap_mode="r")
        if self.resized_images is None:
            with Image.open(self.image_paths[index]) as image:
                tensor = self.transform(image.convert("RGB"))
        else:
            pixels = self.resized_images[index]
            if isinstance(pixels, torch.Tensor):
                pixels = pixels.numpy()
            else:
                pixels = np.array(pixels, copy=True)
            tensor = transforms.functional.to_tensor(pixels)
        return tensor, torch.from_numpy(self.latents[index]), index

    def cache_images(self, path, shared):
        """Store resized RGB bytes in shared RAM or an atomic disk-backed array."""
        shape = (len(self), self.image_size, self.image_size, 3)
        if shared:
            pixels = torch.empty(shape, dtype=torch.uint8).share_memory_()
            self.fill_image_cache(pixels)
            self.resized_images = pixels
            return
        with atomic_path(path) as temporary:
            pixels = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.uint8, shape=shape
            )
            self.fill_image_cache(pixels)
            pixels.flush()
            del pixels
        self.image_cache_path = path
        self.resized_images = np.load(path, mmap_mode="r")

    def fill_image_cache(self, destination):
        """Apply the PIL resize convention before converting pixels to tensors."""
        resize = transforms.Resize((self.image_size, self.image_size))
        for index, path in enumerate(
            tqdm(self.image_paths, desc="Image cache", unit="image", leave=False)
        ):
            with Image.open(path) as image:
                pixels = np.array(resize(image.convert("RGB")), copy=True)
            if isinstance(destination, torch.Tensor):
                destination[index].copy_(torch.from_numpy(pixels))
            else:
                destination[index] = pixels

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.image_cache_path is not None:
            state["resized_images"] = None
        return state

    def signature(self):
        """Hash the paired label values and ordered file identities."""
        digest = hashlib.sha256(self.latents.tobytes())
        for image_id in self.image_ids:
            digest.update(image_id.encode())
        return digest.hexdigest()


class CorruptionDataset(Dataset):
    """Apply blur, brightness changes, or deterministically seeded per-image noise."""

    def __init__(self, dataset, corruption, severity, config):
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, index = self.dataset[index]
        if self.corruption == "gaussian_blur":
            transform = transforms.GaussianBlur(5, sigma=BLUR_SIGMA[self.severity])
            image = transform(image)
        elif self.corruption == "brightness":
            image = torch.clamp(image * (1 + BRIGHTNESS_DELTA[self.severity]), 0, 1)
        elif self.corruption == "gaussian_noise":
            generator = make_generator(self.config.corruption_noise_base_seed + index)
            noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
            image = torch.clamp(image + noise * NOISE_STD[self.severity], 0, 1)
        else:
            raise ValueError(self.corruption)
        return image, target, index
