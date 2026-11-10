## Random streams and deterministic backend settings.

import os
import random

import numpy as np
import torch


# Configure TF32, cuDNN, and deterministic algorithms
def configure_backend():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Seed Python, NumPy, and Torch and configure deterministic execution.
def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True, warn_only=False)


# Create a seeded CPU generator.
def make_generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


# Create a seeded generator on the device.
def make_device_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


# Generate fixed, stochastic, or reassigned slot noise with explicit streams.
def make_slot_init_noise(features, mode, draw_seed, config, stream_generator=None):
    if mode not in config.initialization_modes:
        raise ValueError(f"Unknown initialization mode: {mode}")
    batch = features.shape[0]
    dim = features.shape[-1]
    device = features.device
    dtype = features.dtype
    if mode == "fixed_shared":
        generator = make_device_generator(device, draw_seed)
        base = torch.randn(
            1, config.num_slots, dim, generator=generator, device=device, dtype=dtype
        )
        return base.expand(batch, -1, -1).contiguous()
    if stream_generator is None:
        stream_generator = make_device_generator(device, draw_seed)
    if mode == "per_example_stochastic":
        return torch.randn(
            batch,
            config.num_slots,
            dim,
            generator=stream_generator,
            device=device,
            dtype=dtype,
        )
    base_generator = make_device_generator(device, config.slot_init_seed)
    base = torch.randn(
        1, config.num_slots, dim, generator=base_generator, device=device, dtype=dtype
    )
    base = base.expand(batch, -1, -1)
    random_scores = torch.rand(
        batch, config.num_slots, generator=stream_generator, device=device, dtype=dtype
    )
    permutations = torch.argsort(random_scores, dim=1)
    gather_index = permutations.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(base, 1, gather_index).contiguous()


# Apply independent uniformly distributed slot permutations to a batch.
def independently_permute_slots(slots, generator):
    random_scores = torch.rand(
        slots.shape[0],
        slots.shape[1],
        device=slots.device,
        generator=generator,
        dtype=slots.dtype,
    )
    permutations = random_scores.argsort(dim=1)
    indices = permutations.unsqueeze(-1).expand_as(slots)
    return slots.gather(1, indices)
