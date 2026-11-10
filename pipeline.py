"""Complete training and evaluation workflow."""

import gc
import shutil

import numpy as np
import pandas as pd

from artifacts import (
    RunManifest,
    atomic_csv_save,
    create_run_config,
    sha256_file,
    source_digest,
)
from configuration import load_config
from data import audit_data, cache_dataset_images, load_datasets, split_indices
from diagnostics import (
    backbone_equivariance,
    equivariance_warning,
    numerical_checks,
    record_diagnostic_warnings,
)
from evaluation_cache import EvaluationCache
from inference import evaluate_model
from reporting import SummaryTables
from reproducibility import configure_backend, set_global_seed
from result_summaries import (
    factor_permutation_summaries,
    jacobian_error_correlations,
    permutation_initialization_comparison,
    prediction_change_summaries,
    score_training_summaries,
    stochastic_initialization_summaries,
)
from statistical_analysis import across_seed_tables
from training import load_backbone, release_models, train_backbone, train_readout
from visualization import backbone_visualizations, result_figures


def model_entry(group, regime, architecture, seed, backbone_path):
    """Describe a training condition before its checkpoint has been produced."""
    return {
        "id": f"{group}_{regime}_{architecture}_seed_{seed}",
        "group": group,
        "regime": regime,
        "architecture": architecture,
        "seed": seed,
        "backbone_checkpoint": str(backbone_path),
    }


def primary_training_conditions(config):
    """List the readout regimes, architectures, and seeds to train."""
    conditions = []
    for regime in config.regimes:
        for architecture in config.predictor_architectures:
            if regime == "joint" and architecture == "flatmlp_stochastic_init":
                continue
            for seed in config.predictor_seeds:
                conditions.append((regime, architecture, seed))
    return conditions


def score_selection_tables(manifest, tables, config):
    """Compare both scoring rules across every saved validation epoch."""
    for entry in manifest.payload["models"]:
        path = (
            config.output_root / "evaluation" / "training" / entry["id"] / "history.csv"
        )
        frame = pd.read_csv(path)
        fourier = frame.loc[frame.val_crps_mixed.idxmin()]
        quantile = frame.loc[frame.val_crps_quantile.idxmin()]
        tables.add(
            "checkpoint_selection_comparison.csv",
            {
                "group": entry["group"],
                "regime": entry["regime"],
                "architecture": entry["architecture"],
                "train_seed": entry["seed"],
                "training_estimator": entry["estimator"],
                "n_epochs": len(frame),
                "fourier_selected_epoch": int(fourier.epoch),
                "quantile_selected_epoch": int(quantile.epoch),
                "selection_differs": int(fourier.epoch) != int(quantile.epoch),
                "fourier_score_at_fourier_selection": float(fourier.val_crps_mixed),
                "fourier_score_at_quantile_selection": float(quantile.val_crps_mixed),
            },
        )


def aggregate_experiment_tables(tables, config):
    """Aggregate backbone replication, orbit risk, and test-versus-audit comparisons."""
    risk = pd.DataFrame(tables.rows["orbit_risk_summary.csv"])
    risk = risk[
        (risk["mode"] == "fixed_shared")
        & (risk.slot_init_seed == config.slot_init_seed)
    ]
    keys = ["group", "regime", "architecture", "mode", "variant"]
    values = [
        "expected_relative_degradation",
        "worst_case_relative_degradation",
        "prob_exceeds_10pct",
        "prob_exceeds_50pct",
        "prob_exceeds_100pct",
    ]
    seed = risk.groupby(keys + ["train_seed"])[values].mean().reset_index()
    aggregate = seed.groupby(keys)[values].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    atomic_csv_save(
        aggregate.reset_index(),
        config.output_root / "evaluation" / "orbit_risk_summary_aggregated.csv",
    )
    predictions = pd.DataFrame(tables.rows["test_predictive_metrics.csv"])
    replications = predictions[predictions.group.str.startswith("backbone_")]
    if len(replications):
        keys = ["mode", "aligned", "variant"]
        within = (
            replications.groupby(["group"] + keys)[
                ["rmse", "nll", "crps", "coverage90", "width90", "interval_score90"]
            ]
            .mean()
            .reset_index()
        )
        grouped = within.groupby(keys)[
            ["rmse", "nll", "crps", "coverage90", "width90", "interval_score90"]
        ].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(column) for column in grouped.columns]
        atomic_csv_save(
            grouped.reset_index(),
            config.output_root
            / "evaluation"
            / "replication_backbone_group_summary.csv",
        )
    audit = pd.DataFrame(tables.rows["audit_predictive_metrics.csv"])
    reference = predictions[
        (predictions["mode"] == "fixed_shared")
        & (predictions.slot_init_seed == config.slot_init_seed)
        & (~predictions.aligned)
        & (predictions.context == "slot_initialization")
    ]
    keys = ["group", "regime", "architecture", "train_seed", "variant"]
    paired = reference.merge(audit, on=keys, suffixes=("_working", "_audit"))
    paired["relative_rmse_difference"] = (
        paired.rmse_audit - paired.rmse_working
    ) / paired.rmse_working
    paired["nll_difference"] = paired.nll_audit - paired.nll_working
    atomic_csv_save(
        paired[keys + ["relative_rmse_difference", "nll_difference"]],
        config.output_root / "evaluation" / "audit_comparison.csv",
    )


def import_reference_summaries(config, manifest):
    """Import selected reference summary tables without loading trained models."""
    if config.historical_results is None:
        return
    directory = config.historical_results
    candidates = (
        "slot_permutation_summary.csv",
        "replication_backbone_group_summary.csv",
        "test_predictive_metrics.csv",
        "pretraining_unit_tests.csv",
    )
    for filename in candidates:
        for source in sorted(directory.rglob(filename)):
            manifest.input("historical_summary", source)
            table = pd.read_csv(source)
            relative = source.relative_to(directory)
            destination = (
                config.output_root / "evaluation" / "historical_results" / relative
            )
            atomic_csv_save(table, destination)


def record_equivariance_warning(manifest, rows, label, config):
    """Add an equivariance tolerance failure to the run manifest."""
    warning = equivariance_warning(rows, label, config)
    if warning is not None:
        manifest.warning(warning)


def run_pipeline(config_path):
    """Execute training and evaluation with one run-provenance manifest."""
    config = load_config(config_path)
    if not config.factor_schema_confirmed:
        raise ValueError(
            "Confirm the factor schema and image-label mapping in config.json."
        )
    config, run_information = create_run_config(config)
    configure_backend()
    set_global_seed(config.split_seed)
    manifest = RunManifest(config, run_information)
    print(f"Results: {config.output_root}", flush=True)
    print(f"Arrays: {config.array_root}", flush=True)
    tables = SummaryTables(config.output_root / "evaluation")
    model_rows = []
    with manifest.stage("input_files"):
        manifest.input("pretrained_backbone", config.slot_checkpoint)
        manifest.checkpoint(
            config.slot_checkpoint,
            "primary",
            "reconstruction",
            "slot_attention",
            None,
            None,
            "input",
        )
        for role, path in (
            ("train_archive", config.train_tar),
            ("test_archive", config.test_tar),
        ):
            if path.is_file():
                manifest.input(role, path)
    with manifest.stage("numerical_checks"):
        diagnostic_rows = []
        rows = numerical_checks(config, diagnostic_rows)
        tables.extend("pretraining_unit_tests.csv", rows)
        tables.extend("numerical_diagnostics.csv", diagnostic_rows)
        tables.flush()
        record_diagnostic_warnings(rows, "pretraining_unit_tests.csv", manifest)
        record_diagnostic_warnings(diagnostic_rows, "numerical_diagnostics.csv", manifest)
    with manifest.stage("dataset_checks"):
        train, test = load_datasets(config)
        splits = split_indices(len(train), len(test), config)
        audit_data(train, test, splits, config, manifest)
        import_reference_summaries(config, manifest)
    with manifest.stage("image_cache"):
        cache_dataset_images(train, test, config)
    with manifest.stage("primary_backbone_diagnostics"):
        backbone = load_backbone(config.slot_checkpoint, config)
        rows = backbone_equivariance(backbone, test, splits["working"], "primary", config)
        tables.extend("slot_equivariance.csv", rows)
        tables.flush()
        record_equivariance_warning(manifest, rows, "primary", config)
        rows = backbone_visualizations(backbone, test, splits["working"], "primary", config)
        tables.extend("latent_stability.csv", rows)
        tables.flush()
        del backbone
        release_models()
    backbones = {"primary": config.slot_checkpoint}
    for seed in config.additional_backbone_seeds:
        with manifest.stage(f"backbone_training/seed_{seed}"):
            path = train_backbone(seed, train, splits, config, manifest)
            backbones[f"backbone_{seed}"] = path
        with manifest.stage(f"backbone_diagnostics/seed_{seed}"):
            backbone = load_backbone(path, config)
            rows = backbone_equivariance(
                backbone, test, splits["working"], f"backbone_{seed}", config
            )
            for row in rows:
                tables.add("slot_equivariance.csv", dict(row, backbone_seed=seed))
            tables.flush()
            record_equivariance_warning(manifest, rows, f"backbone_{seed}", config)
            rows = backbone_visualizations(
                backbone, test, splits["working"], f"backbone_{seed}", config
            )
            tables.extend("latent_stability.csv", rows)
            tables.flush()
            del backbone
            release_models()
    entries = []
    for regime, architecture, seed in primary_training_conditions(config):
        entry = model_entry(
            "primary", regime, architecture, seed, config.slot_checkpoint
        )
        with manifest.stage(f"readout_training/{entry['id']}"):
            train_readout(entry, train, splits, config, manifest)
        entries.append(entry)
    for group, path in backbones.items():
        if group == "primary":
            continue
        for seed in config.replication_predictor_seeds:
            entry = model_entry(group, "frozen", "flatmlp", seed, path)
            with manifest.stage(f"readout_training/{entry['id']}"):
                train_readout(entry, train, splits, config, manifest)
            entries.append(entry)
    for estimator in ("fourier", "quantile"):
        for seed in config.score_comparison_seeds:
            entry = model_entry(
                f"score_{estimator}", "frozen", "flatmlp", seed, config.slot_checkpoint
            )
            with manifest.stage(f"score_comparison/{entry['id']}"):
                train_readout(
                    entry,
                    train,
                    splits,
                    config,
                    manifest,
                    estimator,
                    config.score_comparison_epochs,
                )
    evaluation_cache = EvaluationCache(config)
    for entry in entries:
        with manifest.stage(f"inference/{entry['id']}"):
            evaluate_model(
                entry,
                train,
                test,
                splits,
                tables,
                model_rows,
                config,
                manifest,
                evaluation_cache,
            )
            tables.flush()
    with manifest.stage("statistical_analysis"):
        score_selection_tables(manifest, tables, config)
        tables.flush()
        aggregate_experiment_tables(tables, config)
        across_seed_tables(model_rows, config)
        stochastic_initialization_summaries(tables, config)
        factor_permutation_summaries(tables, config)
        jacobian_error_correlations(tables, config)
        prediction_change_summaries(tables, config)
        permutation_initialization_comparison(tables, config)
        score_training_summaries(manifest, config)
        result_figures(tables, config)
    with manifest.stage("final_validation"):
        validate_outputs(manifest, tables, entries, config)
    with manifest.stage("cache_cleanup"):
        remove_temporary_caches(evaluation_cache, (train, test), config)
    manifest.finish()
    print(f"Experiment complete: {config.output_root}", flush=True)


def remove_temporary_caches(evaluation_cache, datasets, config):
    """Release shared resources and remove four temporary cache directories."""
    for cache in evaluation_cache.backbones.values():
        for slots, target, indices, timings in cache.extractions.values():
            if isinstance(slots, np.memmap):
                slots._mmap.close()
        cache.extractions.clear()
        cache.correspondence.clear()
        cache.datasets.clear()
    evaluation_cache.backbones.clear()
    for dataset in datasets:
        pixels = dataset.resized_images
        if isinstance(pixels, np.memmap):
            pixels._mmap.close()
        dataset.resized_images = None
        dataset.image_cache_path = None
    gc.collect()
    for name in ("slot_cache", "evaluation_slots", "image_cache", "scoring_arrays"):
        path = config.array_root / name
        if path.is_dir():
            shutil.rmtree(path)


def validate_outputs(manifest, tables, entries, config):
    """Verify experiment coverage and every recorded checkpoint's content hash."""
    required = (
        "pretraining_unit_tests.csv",
        "numerical_diagnostics.csv",
        "slot_permutation_summary.csv",
        "complete_slot_permutation_table.csv",
        "slot_permutation_paired_bootstrap_ci.csv",
        "orbit_risk_summary.csv",
        "test_predictive_metrics.csv",
        "audit_predictive_metrics.csv",
        "per_factor_pit_statistics.csv",
        "interval_width_and_score_statistics.csv",
        "mask_role_diagnostics.csv",
        "cosine_margins.csv",
        "jacobian_permutation_symmetry.csv",
        "weight_block_symmetry.csv",
        "exact_corruption_level_metrics.csv",
        "prediction_variation_summary.csv",
        "prediction_variation_by_factor.csv",
        "draw_variation_by_factor.csv",
        "permutation_change_vs_initialization_variation.csv",
        "empirical_crps_gradients.csv",
        "predictor_gradient_comparison.csv",
        "predictive_mixture_metrics.csv",
        "checkpoint_selection_comparison.csv",
        "error_conditioned_uncertainty.csv",
        "latent_stability.csv",
    )
    for filename in required:
        path = config.output_root / "evaluation" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing scientific output: {filename}")
    if source_digest() != manifest.payload["environment"]["source_sha256"]:
        raise AssertionError("Executable source changed during the experiment.")
    for filename in (
        "test_predictive_metrics.csv",
        "audit_predictive_metrics.csv",
        "complete_slot_permutation_table.csv",
    ):
        frame = pd.DataFrame(tables.rows[filename])
        columns = [
            "rmse",
            "mae",
            "nll",
            "crps",
            "coverage90",
            "width90",
            "interval_score90",
        ]
        if not np.isfinite(frame[columns].to_numpy()).all():
            raise AssertionError(f"Nonfinite predictive summary: {filename}")
    for checkpoint in manifest.payload["checkpoints"]:
        if sha256_file(checkpoint["path"]) != checkpoint["sha256"]:
            raise AssertionError(f"Checkpoint content changed: {checkpoint['path']}")
    primary_count = len(primary_training_conditions(config))
    expected = primary_count
    expected += len(config.additional_backbone_seeds) * len(
        config.replication_predictor_seeds
    )
    if len(entries) != expected:
        raise AssertionError("The trained predictor coverage is incomplete.")
    manifest.payload["coverage"] = {
        "evaluated_models": len(entries),
        "primary_models": primary_count,
        "independent_backbones": len(config.additional_backbone_seeds),
        "calibration_methods": [
            "raw_gaussian",
            "scalar_temperature",
            "per_factor_temperature",
            "isotonic_cdf",
            "absolute_conformal",
            "gaussian_split_conformal",
        ],
        "permutations": 24,
        "matching_distance": "cosine",
    }
    manifest.write()
