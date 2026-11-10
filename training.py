# Reconstruction training, readout training, and validation selection.

import gc

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset
from tqdm import tqdm

from artifacts import atomic_csv_save, atomic_npz_save, atomic_path, atomic_torch_save
from data import make_loader
from models import (
    DeepSetsProbFactorPredictor,
    FlatMLPProbFactorPredictor,
    SlotAttentionAE,
)

from probability import crps_matrix, mixed_crps_torch, mixed_nll_torch, nll_matrix

from reproducibility import (
    independently_permute_slots,
    make_device_generator,
    make_slot_init_noise,
    set_global_seed,
)


def build_predictor(architecture, config):
    # Construct a probabilistic readout for the configured architecture.
    if architecture == "deepsets":
        return DeepSetsProbFactorPredictor(
            slot_dim=config.slot_dim, out_dim=config.out_dim
        )
    if architecture in config.predictor_architectures:
        return FlatMLPProbFactorPredictor(
            num_slots=config.num_slots, slot_dim=config.slot_dim, out_dim=config.out_dim
        )
    raise ValueError(architecture)


def load_backbone(path, config):
    # Load the reconstruction checkpoint
    model = SlotAttentionAE(config.num_slots, config.slot_dim, config.image_size)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    model = model.to(config.device)
    model.eval()
    return model


def load_trained_model(entry, config):
    # Load a selected predictor and its corresponding trained or frozen backbone.
    backbone = load_backbone(entry["backbone_checkpoint"], config)
    predictor = build_predictor(entry["architecture"], config).to(config.device)
    payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    if entry["regime"] == "joint":
        backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
    backbone.eval()
    predictor.eval()
    return backbone, predictor


def slots_forward(
    backbone, images, mode, draw_seed, config, stream=None, attention=False
):
    # Encode images using an explicit initialization stream.
    features = backbone.project(backbone.encoder(images))
    noise = make_slot_init_noise(features, mode, draw_seed, config, stream)
    return backbone.slot_attention(
        features, init_noise=noise, return_attention=attention
    )


# Release Python references and unused accelerator allocations.
def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(payload, path, entry, epoch, role, manifest):
    atomic_torch_save(payload, path)
    manifest.checkpoint(
        path,
        entry["group"],
        entry["regime"],
        entry["architecture"],
        entry["seed"],
        epoch,
        role,
    )


# Train an independent reconstruction backbone with stochastic slot initialization.
def train_backbone(seed, train, splits, config, manifest):
    set_global_seed(seed)
    model = SlotAttentionAE(config.num_slots, config.slot_dim, config.image_size)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.backbone_lr)
    training_loader = make_loader(
        train, splits["train"], config.backbone_batch_size, config, True, seed
    )
    validation_indices = np.concatenate((splits["validation"], splits["calibration"]))
    validation_loader = make_loader(
        train, validation_indices, config.backbone_batch_size, config, False, seed
    )
    directory = config.output_root / "checkpoints" / "backbones" / f"seed_{seed}"
    entry = {
        "group": "backbones",
        "regime": "reconstruction",
        "architecture": "slot_attention",
        "seed": seed,
    }
    history = []
    best_score = float("inf")
    for epoch in tqdm(
        range(1, config.backbone_epochs + 1), desc=f"Backbone seed {seed}", unit="epoch"
    ):
        train_score = reconstruction_epoch(model, training_loader, config, optimizer)
        validation_score = reconstruction_epoch(model, validation_loader, config)
        history.append(
            {"epoch": epoch, "train_loss": train_score, "val_loss": validation_score}
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if validation_score < best_score:
            best_score = validation_score
            save_checkpoint(
                payload, directory / "best.pth", entry, epoch, "selected", manifest
            )
        path = (
            config.output_root
            / "evaluation"
            / "backbones"
            / f"seed_{seed}"
            / "history.csv"
        )
        atomic_csv_save(history, path)
    del model, optimizer, training_loader, validation_loader
    release_models()
    return directory / "best.pth"


# Run one reconstruction epoch using Gaussian slot-initialization noise.
def reconstruction_epoch(model, loader, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for images, target, indices in tqdm(
        loader, desc="Reconstruction batches", unit="batch", leave=False
    ):
        images = images.to(config.device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, slots, masks = model(images)
            loss = functional.mse_loss(reconstruction, images)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss.")
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item() * len(images)
        count += len(images)
    return total / count


# Cache fixed-initialization slots for frozen readout training.
def cache_fixed_slots(backbone_path, group, train, config):
    path = config.array_root / "slot_cache" / group / "train_slots.npy"
    if path.exists():
        return SlotDataset(path, train)
    backbone = load_backbone(backbone_path, config)
    loader = make_loader(
        train, np.arange(len(train)), config.slot_extract_batch_size, config
    )
    with atomic_path(path) as temporary:
        slots_array = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(train), config.num_slots, config.slot_dim),
        )
        with torch.no_grad():
            for images, target, indices in tqdm(
                loader, desc="Training slot extraction", unit="batch", leave=False
            ):
                slots = slots_forward(
                    backbone,
                    images.to(config.device),
                    "fixed_shared",
                    config.slot_init_seed,
                    config,
                )
                slots_array[indices.numpy()] = slots.cpu().numpy()
        slots_array.flush()
        del slots_array
    del backbone
    release_models()
    return SlotDataset(path, train)


def train_readout(
    entry, train, splits, config, manifest, estimator="fourier", epochs=None
):
    """Train a readout and optional joint backbone with validation-only selection."""
    if epochs is None:
        epochs = config.predictor_epochs
    seed = entry["seed"]
    architecture = entry["architecture"]
    regime = entry["regime"]
    stochastic = architecture == "flatmlp_stochastic_init"
    training_dataset = train
    if regime == "frozen" and not stochastic:
        training_dataset = cache_fixed_slots(
            entry["backbone_checkpoint"], entry["group"], train, config
        )
    set_global_seed(seed)
    if regime == "joint":
        backbone = load_backbone(entry["backbone_checkpoint"], config)
        predictor = build_predictor(architecture, config).to(config.device)
    else:
        predictor = build_predictor(architecture, config).to(config.device)
        backbone = load_backbone(entry["backbone_checkpoint"], config)
    for parameter in backbone.parameters():
        parameter.requires_grad_(regime == "joint")
    groups = []
    if regime == "joint":
        groups.append({"params": backbone.parameters(), "lr": config.joint_backbone_lr})
    groups.append({"params": predictor.parameters(), "lr": config.predictor_lr})
    optimizer = torch.optim.Adam(groups)
    batch_size = config.predictor_batch_size
    if regime == "joint":
        batch_size = config.image_batch_size
    training_loader = make_loader(
        training_dataset, splits["train"], batch_size, config, True, seed
    )
    validation_loader = make_loader(
        training_dataset, splits["validation"], batch_size, config, False, seed
    )
    training_stream = make_device_generator(
        config.device, config.training_stream_base_seed + seed
    )
    permutation_stream = None
    if architecture == "flatmlp_permaug":
        permutation_stream = make_device_generator(config.device, seed)
    directory = config.output_root / "checkpoints" / entry["group"]
    directory = directory / f"{regime}_{architecture}" / f"seed_{seed}"
    history = []
    best_score = float("inf")
    selected_validation = None
    for epoch in tqdm(range(1, epochs + 1), desc=entry["id"], unit="epoch"):
        update_backbone = regime == "joint" and epoch > config.warmup_epochs
        for parameter in backbone.parameters():
            parameter.requires_grad_(update_backbone)
        train_metrics = readout_epoch(
            backbone,
            predictor,
            training_loader,
            entry,
            config,
            estimator,
            training_stream,
            optimizer,
            permutation_stream,
        )
        validation_stream = make_device_generator(
            config.device, config.validation_stream_seed
        )
        validation = predict_identity(
            backbone, predictor, validation_loader, entry, config, validation_stream
        )
        values = validation_scores(validation, config)
        selection = values[estimator]
        history.append(
            {
                "epoch": epoch,
                "train_crps_mixed": train_metrics["crps"],
                "train_nll_mixed": train_metrics["nll"],
                "train_reconstruction": train_metrics["reconstruction"],
                "val_crps_mixed": values["fourier"],
                "val_crps_quantile": values["quantile"],
                "selection_score": selection,
                "val_nll_mixed": values["nll"],
            }
        )
        payload = {
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if regime == "joint":
            payload["backbone_state_dict"] = backbone.state_dict()
        if selection < best_score:
            best_score = selection
            entry["selected_epoch"] = epoch
            save_checkpoint(
                payload, directory / "best.pth", entry, epoch, "selected", manifest
            )
            if entry["group"].startswith("score_"):
                selected_validation = dict(validation)
        history_path = (
            config.output_root / "evaluation" / "training" / entry["id"] / "history.csv"
        )
        atomic_csv_save(history, history_path)
        tqdm.write(f"{entry['id']} epoch {epoch}: validation CRPS {selection:.6f}")
    if selected_validation is not None:
        atomic_npz_save(
            config.array_root
            / "validation_predictions"
            / entry["id"]
            / f"epoch_{entry['selected_epoch']:04d}.npz",
            **selected_validation,
        )
    entry["checkpoint"] = str(directory / "best.pth")
    entry["estimator"] = estimator
    manifest.model(dict(entry))
    del backbone, predictor, optimizer, training_loader, validation_loader
    release_models()
    return entry


def readout_epoch(
    backbone,
    predictor,
    loader,
    entry,
    config,
    estimator,
    stream,
    optimizer,
    permutation_stream,
):
    """Train or validate a readout using the configured initialization and objective."""
    predictor.train()
    backbone.train(entry["regime"] == "joint")
    total_score = 0.0
    total_nll = 0.0
    total_reconstruction = 0.0
    count = 0
    mode = "fixed_shared"
    if entry["architecture"] == "flatmlp_stochastic_init":
        mode = "per_example_stochastic"
    for inputs, target, indices in tqdm(
        loader, desc="Readout batches", unit="batch", leave=False
    ):
        inputs = inputs.to(config.device, non_blocking=True)
        target = target.to(config.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(loader.dataset.dataset, SlotDataset):
            slots = inputs
        else:
            with torch.set_grad_enabled(entry["regime"] == "joint"):
                slots = slots_forward(
                    backbone, inputs, mode, config.slot_init_seed, config, stream
                )
        reconstruction_loss = torch.zeros((), device=config.device)
        if entry["regime"] == "joint":
            reconstruction, masks = backbone.decoder(slots)
            reconstruction_loss = functional.mse_loss(reconstruction, inputs)
        if permutation_stream is not None:
            slots = independently_permute_slots(slots, permutation_stream)
        mu, logvar = predictor(slots)
        score = mixed_crps_torch(mu, logvar, target, config, estimator)
        nll = mixed_nll_torch(mu, logvar, target, config)
        loss = score + config.reconstruction_weight * reconstruction_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite readout training objective.")
        loss.backward()
        parameters = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    parameters.append(parameter)
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        optimizer.step()
        total_score += score.item() * len(target)
        total_nll += nll.item() * len(target)
        total_reconstruction += reconstruction_loss.item() * len(target)
        count += len(target)
    return {
        "crps": total_score / count,
        "nll": total_nll / count,
        "reconstruction": total_reconstruction / count,
    }


def predict_identity(backbone, predictor, loader, entry, config, stream=None):
    # Collect unpermuted predictions using a fixed validation initialization stream.
    backbone.eval()
    predictor.eval()
    mode = "fixed_shared"
    if entry["architecture"] == "flatmlp_stochastic_init":
        mode = "per_example_stochastic"
    parts = {"mu": [], "var": [], "y": [], "indices": []}
    with torch.no_grad():
        for inputs, target, indices in tqdm(
            loader, desc="Readout batches", unit="batch", leave=False
        ):
            inputs = inputs.to(config.device, non_blocking=True)
            if isinstance(loader.dataset.dataset, SlotDataset):
                slots = inputs
            else:
                slots = slots_forward(
                    backbone, inputs, mode, config.slot_init_seed, config, stream
                )
            mu, logvar = predictor(slots)
            parts["mu"].append(mu.cpu().numpy())
            parts["var"].append(logvar.clamp(-10, 10).exp().cpu().numpy())
            parts["y"].append(target.numpy())
            parts["indices"].append(indices.numpy())
    output = {}
    for key, arrays in parts.items():
        output[key] = np.concatenate(arrays)
    return output


def validation_scores(pack, config):
    # Score each validation epoch with both numerical CRPS estimators.

    mu = pack["mu"].astype(np.float64)
    variance = pack["var"].astype(np.float64)
    target = pack["y"].astype(np.float64)
    scores = {
        "fourier": float(
            crps_matrix(mu, variance, target, config.fourier_terms).mean()
        ),
        "quantile": float(
            crps_matrix(
                mu, variance, target, samples=config.circular_train_samples
            ).mean()
        ),
        "nll": float(nll_matrix(mu, variance, target).mean()),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite validation scores.")
    return scores


class SlotDataset(Dataset):
    # Expose immutable cached slots with aligned labels and image indices.

    def __init__(self, path, dataset):
        self.slots = np.load(path, mmap_mode="r", allow_pickle=False)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        slots = torch.from_numpy(np.array(self.slots[index], copy=True))
        target = torch.from_numpy(self.dataset.latents[index])
        return slots, target, index
