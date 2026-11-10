## Paired-image and seed-level statistical comparisons.

import itertools

import numpy as np
import pandas as pd

from artifacts import atomic_csv_save

METRICS = ("rmse", "mae", "nll", "crps", "coverage90", "width90", "interval_score90")


# Summarize paired per-image contributions without pooling seeds.
def contribution_summary(contributions):
    summary = {"rmse": float(np.sqrt(np.nanmean(contributions["mse"])))}
    for metric in METRICS[1:]:
        summary[metric] = float(np.nanmean(contributions[metric]))
    return summary

# Estimate paired image-bootstrap intervals using an explicit random seed.
def paired_bootstrap_ci(identity, candidate, repetitions, seed):
    sample_count = len(identity["mse"])
    generator = np.random.default_rng(seed)
    draws = {}
    for metric in METRICS:
        draws[metric] = []
    for start in range(0, repetitions, 50):
        size = min(50, repetitions - start)
        indices = generator.integers(
            0, sample_count, size=(size, sample_count), endpoint=False
        )
        first = identity["mse"][indices].mean(1)
        second = candidate["mse"][indices].mean(1)
        draws["rmse"].extend((np.sqrt(second) - np.sqrt(first)).tolist())
        for metric in METRICS[1:]:
            differences = candidate[metric][indices].mean(1) - identity[metric][
                indices
            ].mean(1)
            draws[metric].extend(differences.tolist())
    first = contribution_summary(identity)
    second = contribution_summary(candidate)
    row = {}
    for metric in METRICS:
        row[f"delta_{metric}"] = second[metric] - first[metric]
        row[f"delta_{metric}_ci95_lo"] = float(np.quantile(draws[metric], 0.025))
        row[f"delta_{metric}_ci95_hi"] = float(np.quantile(draws[metric], 0.975))
    return row


# Summarize degradation over the uniform permutation orbit at fixed thresholds.
def orbit_risk(values):
    relative = (np.asarray(values) - values[0]) / max(values[0], 1e-15)
    row = {
        "n_orbit": len(values),
        "expected_relative_degradation": float(relative.mean()),
        "worst_case_relative_degradation": float(relative.max()),
    }
    for threshold in (10, 50, 100):
        row[f"prob_exceeds_{threshold}pct"] = float(np.mean(relative > threshold / 100))
    return row


# Bootstrap paired differences over independent training seeds.
def seed_difference_interval(values, config):
    values = np.asarray(values)
    generator = np.random.default_rng(config.permutation_bootstrap_seed)
    indices = generator.integers(
        0, len(values), size=(config.seed_bootstrap_repetitions, len(values))
    )
    draws = values[indices].mean(1)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    null_statistics = np.abs((signs * values).mean(1))
    return {
        "difference": float(values.mean()),
        "ci95_lo": float(np.quantile(draws, 0.025)),
        "ci95_hi": float(np.quantile(draws, 0.975)),
        "n_seeds": len(values),
        "sign_flip_pvalue": float(
            np.mean(null_statistics >= abs(values.mean()) - 1e-15)
        ),
    }


#  Average draws within seed before model comparisons and initialization contrasts.
def across_seed_tables(model_rows, config):
    frame = pd.DataFrame(model_rows)
    if frame.empty:
        raise ValueError("No evaluated model metrics are available.")
    keys = [
        "group",
        "regime",
        "architecture",
        "train_seed",
        "split",
        "mode",
        "variant",
        "readout",
    ]
    metric_columns = list(METRICS)
    seed_frame = frame.groupby(keys, dropna=False)[metric_columns].mean().reset_index()
    destination = config.output_root / "evaluation"
    atomic_csv_save(seed_frame, destination / "seed_level_metrics.csv")
    group_keys = [name for name in keys if name != "train_seed"]
    aggregated = seed_frame.groupby(group_keys, dropna=False)[metric_columns].agg(
        ["mean", "std", "count"]
    )
    aggregated.columns = ["_".join(column) for column in aggregated.columns]
    atomic_csv_save(aggregated.reset_index(), destination / "across_seed_metrics.csv")
    contrasts = []
    paired_rows = []
    contexts = ["group", "regime", "split", "mode", "variant"]
    for context, subset in seed_frame.groupby(contexts):
        choices = subset[["architecture", "readout"]].drop_duplicates().values.tolist()
        for left, right in itertools.combinations(choices, 2):
            first = subset[
                (subset.architecture == left[0]) & (subset.readout == left[1])
            ]
            second = subset[
                (subset.architecture == right[0]) & (subset.readout == right[1])
            ]
            paired = first.merge(second, on="train_seed", suffixes=("_a", "_b"))
            if len(paired) < 2:
                continue
            description = dict(zip(contexts, context))
            for metric in METRICS:
                values = paired[f"{metric}_a"] - paired[f"{metric}_b"]
                row = dict(description)
                row.update(
                    {"a": "/".join(left), "b": "/".join(right), "metric": metric}
                )
                row.update(seed_difference_interval(values.to_numpy(), config))
                paired_rows.append(row)
    if paired_rows:
        table = pd.DataFrame(paired_rows)
        table["bh_qvalue"] = np.nan
        families = contexts + ["metric"]
        for family, indices in table.groupby(families).groups.items():
            probabilities = table.loc[indices, "sign_flip_pvalue"].to_numpy()
            order = np.argsort(probabilities)
            ordered = probabilities[order] * len(order) / np.arange(1, len(order) + 1)
            adjusted = np.minimum.accumulate(ordered[::-1])[::-1]
            original_order = np.empty_like(adjusted)
            original_order[order] = np.minimum(adjusted, 1)
            table.loc[indices, "bh_qvalue"] = original_order
        atomic_csv_save(table, destination / "paired_comparisons.csv")
    mechanism = seed_frame[
        seed_frame.architecture.isin(("flatmlp", "flatmlp_stochastic_init"))
        & (seed_frame.readout == "identity")
    ]
    atomic_csv_save(mechanism, destination / "initialization_2x2_by_seed.csv")
    context_keys = ["group", "regime", "split", "variant"]
    for context, subset in mechanism.groupby(context_keys):
        for metric in METRICS:
            pivot = subset.pivot(
                index="train_seed", columns=["architecture", "mode"], values=metric
            )
            columns = [
                ("flatmlp", "fixed_shared"),
                ("flatmlp", "per_example_stochastic"),
                ("flatmlp_stochastic_init", "fixed_shared"),
                ("flatmlp_stochastic_init", "per_example_stochastic"),
            ]
            if not all(column in pivot.columns for column in columns):
                continue
            pivot = pivot.dropna(subset=columns)
            if len(pivot) < 2:
                continue
            first = pivot[columns[1]] - pivot[columns[0]]
            second = pivot[columns[3]] - pivot[columns[2]]
            for name, values in (
                ("fixed_trained_stochastic_minus_fixed", first),
                ("stochastic_trained_stochastic_minus_fixed", second),
                ("training_evaluation_interaction", second - first),
            ):
                row = dict(zip(context_keys, context))
                row.update({"metric": metric, "contrast": name})
                row.update(seed_difference_interval(values.to_numpy(), config))
                contrasts.append(row)
    atomic_csv_save(contrasts, destination / "initialization_2x2_contrasts.csv")
