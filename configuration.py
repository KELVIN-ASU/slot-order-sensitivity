"""Experiment configuration."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Define the complete experiment in one immutable configuration."""

    slot_checkpoint: Path = Path(
        "../SlotAttentionCheckpoints/slot_attention_epoch_50.pth"
    )
    train_tar: Path = Path("../causal3DIdent_dataset/trainset.tar")
    test_tar: Path = Path("../causal3DIdent_dataset/testset.tar")
    extract_root: Path = Path("../causal3DIdent_final")
    output_root: Path = Path("../results/flatmlp_mechanism")
    array_root: Path = Path("../experiment_arrays/flatmlp_mechanism")
    original_split_file: Path | None = None
    historical_results: Path | None = None
    device: str = "cuda:0"
    factor_schema_confirmed: bool = False
    split_seed: int = 47
    audit_split_seed: int = 999
    additional_backbone_seeds: tuple[int, ...] = (101, 202, 303)
    predictor_seeds: tuple[int, ...] = (45, 46, 47, 48, 49)
    replication_predictor_seeds: tuple[int, ...] = (45,)
    initialization_modes: tuple[str, ...] = (
        "fixed_shared",
        "per_example_stochastic",
        "fixed_random_assignment",
    )
    predictor_architectures: tuple[str, ...] = (
        "flatmlp",
        "flatmlp_stochastic_init",
        "flatmlp_permaug",
        "deepsets",
    )
    regimes: tuple[str, ...] = ("frozen", "joint")
    slot_init_seed: int = 1234
    evaluation_draw_seeds: tuple[int, ...] = (7, 13, 99, 2026)
    calibration_draw_seed: int = 5001
    training_stream_base_seed: int = 100000
    validation_stream_seed: int = 200000
    corruption_noise_base_seed: int = 100047
    num_slots: int = 4
    slot_dim: int = 64
    out_dim: int = 10
    image_size: int = 64
    backbone_epochs: int = 50
    backbone_batch_size: int = 512
    backbone_lr: float = 0.0003
    backbone_checkpoint_every: int = 5
    predictor_epochs: int = 20
    predictor_batch_size: int = 1024
    image_batch_size: int = 512
    predictor_lr: float = 0.001
    joint_backbone_lr: float = 0.00001
    warmup_epochs: int = 1
    reconstruction_weight: float = 1.0
    interval_level: float = 0.90
    val_cal_fraction: float = 0.50
    slot_extract_batch_size: int = 512
    temperature_lr: float = 0.01
    temperature_steps: int = 300
    num_workers: int = 4
    image_cache_max_gib: float = 24.0
    grad_clip: float = 1.0
    validation_fraction: float = 0.10
    circular_train_samples: int = 16
    circular_eval_samples: int = 64
    fourier_terms: int = 512
    wrapped_alias_min: int = 5
    wrapped_alias_coverage_sigmas: float = 6.0
    wrapped_alias_hard_limit: int = 4096
    permutation_bootstrap_repetitions: int = 1000
    permutation_bootstrap_seed: int = 47001
    seed_bootstrap_repetitions: int = 10000
    stability_max_images: int = 2000
    symmetry_sample_size: int = 256
    empirical_gradient_samples: int = 64
    prediction_change_tolerance: float = 0.01
    sigma_change_tolerance: float = 0.5
    equivariance_atol: float = 0.0001
    metric_batch_size: int = 64
    corruption_types: tuple[str, ...] = (
        "brightness",
        "gaussian_blur",
        "gaussian_noise",
    )
    corruption_severities: tuple[int, ...] = (1, 3, 5)
    score_comparison_seeds: tuple[int, ...] = (45, 46)
    score_comparison_epochs: int = 5
    stability_noise_std: float = 0.05


PATH_FIELDS = (
    "slot_checkpoint",
    "train_tar",
    "test_tar",
    "extract_root",
    "output_root",
    "array_root",
    "original_split_file",
    "historical_results",
)
TUPLE_FIELDS = (
    "additional_backbone_seeds",
    "predictor_seeds",
    "replication_predictor_seeds",
    "initialization_modes",
    "predictor_architectures",
    "regimes",
    "evaluation_draw_seeds",
    "corruption_types",
    "corruption_severities",
    "score_comparison_seeds",
)


def load_config(path):
    """Load and validate a configuration relative to its JSON file."""
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    allowed = set()
    for field in fields(Config):
        allowed.add(field.name)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    defaults = asdict(Config())
    defaults.update(values)
    for name in PATH_FIELDS:
        value = defaults[name]
        if value is not None:
            defaults[name] = (path.parent / value).resolve()
    for name in TUPLE_FIELDS:
        defaults[name] = tuple(defaults[name])
    config = Config(**defaults)
    validate_config(config)
    return config


def validate_config(config):
    """Reject incompatible settings before any experiment outputs are written."""
    if (config.num_slots, config.slot_dim, config.out_dim) != (4, 64, 10):
        raise ValueError(
            "This experiment requires four 64-dimensional slots and ten factors."
        )
    if config.image_size != 64:
        raise ValueError(
            "The supplied checkpoint architecture requires 64-pixel images."
        )
    positive = (
        "backbone_epochs",
        "predictor_epochs",
        "backbone_batch_size",
        "predictor_batch_size",
        "image_batch_size",
        "slot_extract_batch_size",
        "metric_batch_size",
        "fourier_terms",
        "temperature_steps",
        "permutation_bootstrap_repetitions",
        "seed_bootstrap_repetitions",
        "empirical_gradient_samples",
        "symmetry_sample_size",
        "stability_max_images",
        "backbone_checkpoint_every",
        "score_comparison_epochs",
    )
    for name in positive:
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(config.image_cache_max_gib) or config.image_cache_max_gib <= 0:
        raise ValueError("image_cache_max_gib must be positive and finite.")
    if config.warmup_epochs < 0 or config.predictor_epochs <= config.warmup_epochs:
        raise ValueError("predictor_epochs must exceed warmup_epochs.")
    if config.num_workers < 0 or config.prediction_change_tolerance <= 0:
        raise ValueError("Worker count and prediction tolerance are invalid.")
    if (
        not math.isfinite(config.sigma_change_tolerance)
        or config.sigma_change_tolerance <= 0
    ):
        raise ValueError("sigma_change_tolerance must be positive and finite.")
    if config.interval_level != 0.90:
        raise ValueError(
            "The retained interval metric schema reports 90 percent intervals."
        )
    for name in ("validation_fraction", "val_cal_fraction", "interval_level"):
        if not 0 < getattr(config, name) < 1:
            raise ValueError(f"{name} must lie strictly between zero and one.")
    for name in (
        "predictor_seeds",
        "additional_backbone_seeds",
        "evaluation_draw_seeds",
        "replication_predictor_seeds",
        "score_comparison_seeds",
    ):
        seeds = getattr(config, name)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{name} must contain distinct seeds.")
        for seed in seeds:
            if type(seed) is not int:
                raise ValueError(f"{name} must contain integers.")
    for name in (
        "initialization_modes",
        "predictor_architectures",
        "regimes",
        "corruption_types",
    ):
        if len(set(getattr(config, name))) != len(getattr(config, name)):
            raise ValueError(f"{name} must not contain duplicate conditions.")
        if set(getattr(config, name)) != set(getattr(Config(), name)):
            raise ValueError(
                f"{name} must retain every configured scientific condition."
            )
    if not config.corruption_severities:
        raise ValueError("corruption_severities must not be empty.")
    for severity in config.corruption_severities:
        if type(severity) is not int or severity not in (1, 2, 3, 4, 5):
            raise ValueError("Corruption severities must be integers from one to five.")
    if len(set(config.corruption_severities)) != len(config.corruption_severities):
        raise ValueError("corruption_severities must not contain duplicates.")
    if config.slot_init_seed in config.evaluation_draw_seeds:
        raise ValueError(
            "Alternative draw seeds must differ from the reference initialization seed."
        )
    for name in (
        "predictor_lr",
        "backbone_lr",
        "joint_backbone_lr",
        "temperature_lr",
        "grad_clip",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    protected = (config.extract_root, config.slot_checkpoint)
    for destination in (config.output_root, config.array_root):
        for source in protected:
            if (
                destination == source
                or destination.is_relative_to(source)
                or source.is_relative_to(destination)
            ):
                raise ValueError(
                    "Outputs must be separate from input data and pretrained checkpoints."
                )
    if config.output_root == config.array_root:
        raise ValueError("Use separate output and raw-array folders.")
    if config.array_root.is_relative_to(
        config.output_root
    ) or config.output_root.is_relative_to(config.array_root):
        raise ValueError("array_root must be outside output_root.")
