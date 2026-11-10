"""Atomic output writing and the single run manifest."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def configuration_digest(config):
    """Hash the effective configuration excluding generated output locations."""
    values = asdict(config)
    values.pop("output_root")
    values.pop("array_root")
    encoded = json.dumps(values, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def create_run_config(config):
    """Reserve matching dated parent folders for results and scientific arrays."""
    started = datetime.now().astimezone()
    digest = configuration_digest(config)
    run_id = f"{started:%Y%m%d_%H%M%S_%f}_{digest[:8]}"
    output_root = config.output_root.parent / run_id / config.output_root.name
    array_root = config.array_root.parent / run_id / config.array_root.name
    parents = []
    for root in (output_root, array_root):
        if root.parent not in parents:
            parents.append(root.parent)
    created = []
    try:
        for parent in parents:
            parent.mkdir(parents=True, exist_ok=False)
            created.append(parent)
    except OSError:
        for parent in reversed(created):
            parent.rmdir()
        raise
    resolved = replace(config, output_root=output_root, array_root=array_root)
    information = {
        "id": run_id,
        "started_at": started.isoformat(),
        "configuration_sha256": digest,
    }
    return resolved, information


def sha256_file(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_path(path):
    """Publish a completed file with an atomic rename on its destination filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".write-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload, path):
    """Atomically write finite, human-readable JSON."""
    with atomic_path(path) as temporary:
        text = json.dumps(payload, indent=2, default=str, allow_nan=False)
        temporary.write_text(text + "\n", encoding="utf-8")


def atomic_csv_save(rows, path):
    """Atomically write a summary table."""
    table = rows
    if not isinstance(rows, pd.DataFrame):
        table = pd.DataFrame(rows)
    with atomic_path(path) as temporary:
        table.to_csv(temporary, index=False)


def atomic_npz_save(path, **arrays):
    """Atomically write compressed scientific arrays without object serialization."""
    for value in arrays.values():
        if np.asarray(value).dtype.hasobject:
            raise ValueError("Raw arrays must not contain Python objects.")
    with atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)


def atomic_torch_save(payload, path):
    """Atomically save model and optimizer tensors."""

    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def source_digest():
    """Hash the executable Python source tree."""
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_information():
    """Collect the code identity and dependency versions for the run manifest."""
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "Pillow",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            check=True,
            text=True,
        )
        commit = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            capture_output=True,
            check=True,
            text=True,
        )
        dirty = bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    accelerator = {
        "cuda_runtime": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
    }
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(index))
    accelerator["devices"] = devices
    accelerator["deterministic_algorithms"] = (
        torch.are_deterministic_algorithms_enabled()
    )
    accelerator["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    accelerator["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    accelerator["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    accelerator["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    return {
        "accelerator": accelerator,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "library_versions": versions,
    }


class RunManifest:
    """Own all run configuration, provenance, stage status, and checkpoint records."""

    def __init__(self, config, run_information=None):
        self.config = config
        self.path = config.output_root / "run_manifest.json"
        for root in (config.output_root, config.array_root):
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"Use an empty experiment folder: {root}")
        config.output_root.mkdir(parents=True, exist_ok=True)
        config.array_root.mkdir(parents=True, exist_ok=True)
        for root in (config.output_root, config.array_root):
            descriptor = os.open(
                root / ".run_lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            os.close(descriptor)
        self.payload = {
            "config": asdict(config),
            "environment": environment_information(),
            "status": "running",
            "stages": {},
            "checkpoints": [],
            "models": [],
            "inputs": [],
            "outputs": [],
            "warnings": [],
            "array_root": str(config.array_root),
            "interpretation": {
                "audit_subset": "Previously used released-archive subset; not a new holdout.",
                "alignment": "Cosine-distance Hungarian assignment.",
                "primary_backbone": "Referenced input checkpoint; independent replicas trained anew.",
            },
        }
        if run_information is not None:
            self.payload["run"] = run_information
        self.write()

    def write(self):
        """Persist the single authoritative run record."""
        atomic_json_dump(self.payload, self.path)

    @contextmanager
    def stage(self, name):
        """Record a stage's start, completion, duration, and any failure."""
        started = time.time()
        record = {"status": "running", "started_unix": started}
        self.payload["stages"][name] = record
        self.write()
        print(f"Stage started: {name}", flush=True)
        try:
            yield
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            self.payload["status"] = "failed"
            raise
        else:
            record["status"] = "complete"
        finally:
            record["duration_seconds"] = time.time() - started
            self.write()
            minutes = record["duration_seconds"] / 60
            print(
                f"Stage {record['status']}: {name} ({minutes:.2f} minutes)",
                flush=True,
            )

    def warning(self, record):
        """Record a failed scientific check that does not stop the run."""
        record = dict(record)
        for name, value in record.items():
            if isinstance(value, float) and not np.isfinite(value):
                record[name] = str(value)
        self.payload["warnings"].append(record)
        self.write()
        print(f"WARNING: {record}", flush=True)

    def input(self, role, path):
        """Record an immutable input without copying it."""
        self.payload["inputs"].append(
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
        )
        self.write()

    def checkpoint(self, path, group, regime, architecture, seed, epoch, role):
        """Record every saved checkpoint's location and content hash."""
        record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "group": group,
            "regime": regime,
            "architecture": architecture,
            "seed": seed,
            "epoch": epoch,
            "role": role,
        }
        kept = []
        for previous in self.payload["checkpoints"]:
            if previous["path"] != str(path):
                kept.append(previous)
        kept.append(record)
        self.payload["checkpoints"] = kept
        self.write()

    def model(self, entry):
        """Register a selected trained predictor in the run manifest."""
        self.payload["models"].append(entry)
        self.write()

    def finish(self):
        """Record final output hashes after every stage has succeeded."""
        records = []
        for root in (self.config.output_root, self.config.array_root):
            for path in sorted(root.rglob("*")):
                if path.is_file() and path != self.path:
                    records.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
        self.payload["outputs"] = records
        self.payload["status"] = "complete"
        self.write()
