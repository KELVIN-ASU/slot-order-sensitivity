# Factor sensitivity, stochastic initialization, and symmetry summary tables.

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from artifacts import atomic_csv_save

NORMALIZED_CHANGE_EPSILON = 1e-12


# Summarize stochastic-training permutation measurements across seeds.
def stochastic_initialization_summaries(tables, config):
    frame = pd.DataFrame(tables.rows["complete_slot_permutation_table.csv"])
    selected = frame[
        (frame.group == "primary")
        & (frame.regime == "frozen")
        & (frame.architecture == "flatmlp_stochastic_init")
        & (frame["mode"] == "per_example_stochastic")
        & (frame["variant"] == "raw")
    ]
    root = config.output_root / "evaluation" / "stochastic_init_mechanism"
    atomic_csv_save(selected, root / "complete_permutation_table.csv")
    rows = []
    for seed, seed_frame in selected.groupby("train_seed"):
        atomic_csv_save(seed_frame, root / f"permutation_table_seed_{seed}.csv")
        summaries = []
        for draw, subset in seed_frame.groupby("slot_init_seed"):
            identity = subset[subset.is_identity].iloc[0]
            relative = (subset.rmse.to_numpy() - identity.rmse) / max(
                identity.rmse, 1e-15
            )
            summaries.append(
                {
                    "train_seed": seed,
                    "slot_init_seed": draw,
                    "identity_rmse": identity.rmse,
                    "orbit_mean_rmse": float(subset.rmse.mean()),
                    "orbit_to_identity_ratio": float(
                        subset.rmse.mean() / max(identity.rmse, 1e-15)
                    ),
                    "worst_permutation_rmse": float(subset.rmse.max()),
                    "max_relative_degradation": float(relative.max()),
                    "fraction_exceeding_10pct": float(np.mean(relative > 0.1)),
                    "fraction_exceeding_50pct": float(np.mean(relative > 0.5)),
                    "fraction_exceeding_100pct": float(np.mean(relative > 1)),
                }
            )
        atomic_csv_save(
            summaries, root / f"permutation_variation_summary_seed_{seed}.csv"
        )
        summary = (
            pd.DataFrame(summaries).drop(columns=["slot_init_seed"]).mean().to_dict()
        )
        summary["train_seed"] = int(seed)
        rows.append(summary)
        identity = seed_frame[seed_frame.is_identity]
        atomic_csv_save(identity, root / f"identity_variation_summary_seed_{seed}.csv")
    seed_summary = pd.DataFrame(rows)
    atomic_csv_save(seed_summary, root / "seed_level_summary.csv")
    aggregate = []
    for metric in seed_summary.columns:
        if metric == "train_seed":
            continue
        aggregate.append(
            {
                "metric": metric,
                "mean": seed_summary[metric].mean(),
                "std": seed_summary[metric].std(ddof=1),
            }
        )
    atomic_csv_save(aggregate, root / "seed_aggregate_summary.csv")


# Report factorwise orbit-to-reference RMSE ratios.
def factor_permutation_summaries(tables, config):
    frame = pd.DataFrame(tables.rows["per_factor_full_extraction.csv"])
    selected = frame[
        (frame.context == "slot_permutation") & (frame["variant"] == "raw")
    ]
    keys = ["group", "regime", "architecture", "train_seed", "factor_index", "factor"]
    rows = []
    for key, subset in selected.groupby(keys):
        identity = subset[subset.is_identity].rmse.iloc[0]
        row = dict(zip(keys, key))
        row.update(
            {
                "identity_rmse": identity,
                "orbit_mean_rmse": subset.rmse.mean(),
                "orbit_to_identity_ratio": subset.rmse.mean() / max(identity, 1e-15),
            }
        )
        rows.append(row)
    atomic_csv_save(
        rows, config.output_root / "evaluation" / "factor_permutation_sensitivity.csv"
    )


#  Measure associations between derivative mismatch and RMSE changes.
def jacobian_error_correlations(tables, config):
    jacobian = pd.DataFrame(tables.rows["jacobian_permutation_symmetry.csv"])
    errors = pd.DataFrame(tables.rows["complete_slot_permutation_table.csv"])
    errors = errors[
        (errors["variant"] == "raw")
        & (errors["mode"] == "fixed_shared")
        & (errors.slot_init_seed == config.slot_init_seed)
        & (~errors.is_identity)
    ]
    keys = ["group", "regime", "architecture", "train_seed", "permutation"]
    merged = errors.merge(jacobian[keys + ["relative_mismatch_mean"]], on=keys)
    rows = []
    for key, subset in merged.groupby(keys[:-1]):
        first = subset.relative_mismatch_mean.to_numpy()
        second = subset.rmse.to_numpy()
        pearson = np.nan
        spearman = np.nan
        if first.std() > 1e-12 and second.std() > 1e-12:
            pearson = pearsonr(first, second).statistic
            spearman = spearmanr(first, second).statistic
        rows.append(dict(zip(keys[:-1], key), pearson=pearson, spearman=spearman))
    atomic_csv_save(
        rows,
        config.output_root / "evaluation" / "jacobian_permutation_correlations.csv",
    )


# Average initialization-draw statistics within seed before across-seed summaries.
def prediction_change_summaries(tables, config):
    frame = pd.DataFrame(tables.rows["prediction_variation_summary.csv"])
    conditions = []
    for row in frame.to_dict("records"):
        condition = row["mode"]
        if condition == "fixed_shared":
            condition = "alternative_fixed_shared"
            if row["slot_init_seed"] == config.slot_init_seed:
                condition = "reference_fixed_shared"
        conditions.append(condition)
    frame["initialization_condition"] = conditions
    keys = ["group", "regime", "architecture", "train_seed", "initialization_condition"]
    metrics = [
        "mean_change_p50",
        "mean_change_p95",
        "mean_change_max",
        "fraction_above_tolerance",
        "logvariance_change_p95",
        "circular_moment_change_p95",
    ]
    seed_frame = frame.groupby(keys)[metrics].mean().reset_index()
    destination = config.output_root / "evaluation"
    atomic_csv_save(seed_frame, destination / "prediction_variation_by_seed.csv")
    keys.remove("train_seed")
    grouped = seed_frame.groupby(keys)[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    atomic_csv_save(
        grouped.reset_index(), destination / "prediction_variation_across_seeds.csv"
    )


# Compare permutation changes with matched DeepSets initialization variability.
def permutation_initialization_comparison(tables, config):
    frame = pd.DataFrame(tables.rows["prediction_variation_by_factor.csv"])
    selected = frame[
        (frame["mode"] == "per_example_stochastic") & (frame.split == "working")
    ]
    reference_keys = [
        "group",
        "regime",
        "train_seed",
        "factor_index",
        "factor",
        "split",
    ]
    references = {}
    for row in tables.rows["draw_variation_by_factor.csv"]:
        if row["architecture"] == "deepsets":
            key = tuple(row[name] for name in reference_keys)
            references[key] = row
    rows = []
    keys = reference_keys + ["architecture"]
    for key, subset in selected.groupby(keys):
        row = dict(zip(keys, key))
        row["n_permutation_draws"] = int(subset.slot_init_seed.nunique())
        reference_key = tuple(row[name] for name in reference_keys)
        reference = references.get(reference_key)
        row["reference_n_draws"] = 0
        if reference is not None:
            row["reference_n_draws"] = reference["n_draws"]
        for quantile in ("p50", "p95"):
            normalized = subset[f"mean_change_{quantile}"] / subset.normalizing_scale
            permutation_change = float(normalized.mean())
            initialization_change = np.nan
            ratio = np.nan
            status = "missing_reference"
            if reference is not None:
                initialization_change = reference[f"normalized_change_{quantile}"]
                status = reference["status"]
                if status == "available":
                    if initialization_change <= NORMALIZED_CHANGE_EPSILON:
                        status = "negligible_reference"
                    else:
                        ratio = permutation_change / initialization_change
            row[f"normalized_permutation_change_{quantile}"] = permutation_change
            row[f"normalized_initialization_change_{quantile}"] = initialization_change
            row[f"permutation_to_initialization_ratio_{quantile}"] = ratio
            row[f"comparison_status_{quantile}"] = status
        rows.append(row)
    atomic_csv_save(
        rows,
        config.output_root
        / "evaluation"
        / "permutation_change_vs_initialization_variation.csv",
    )

# Report matched scoring-rule training arms on validation images only.
def score_training_summaries(manifest, config):
    from reporting import predictive_contributions
    from statistical_analysis import contribution_summary

    rows = []
    for entry in manifest.payload["models"]:
        if not entry["group"].startswith("score_"):
            continue
        epoch = entry["selected_epoch"]
        path = (
            config.array_root
            / "validation_predictions"
            / entry["id"]
            / f"epoch_{epoch:04d}.npz"
        )
        with np.load(path, allow_pickle=False) as pack:
            contributions, scores, likelihood, intervals = predictive_contributions(
                pack["mu"].astype(np.float64),
                pack["var"].astype(np.float64),
                pack["y"].astype(np.float64),
                config,
            )
        row = {
            "train_seed": entry["seed"],
            "training_estimator": entry["estimator"],
            "selected_epoch": epoch,
            "n_epochs": config.score_comparison_epochs,
        }
        row.update(contribution_summary(contributions))
        rows.append(row)
    atomic_csv_save(
        rows, config.output_root / "evaluation" / "score_training_comparison.csv"
    )
