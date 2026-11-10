"""Inference, initialization interventions, permutations, corruptions, and audit data."""

import time

import joblib
import numpy as np
import torch
from tqdm import tqdm

from artifacts import atomic_npz_save, atomic_path
from calibration import calibrated_variances, fit_calibration_bundle
from data import FACTOR_NAMES, CorruptionDataset, make_loader
from diagnostics import (
    PERMUTATIONS,
    correspondence_diagnostics,
    empirical_score_audit,
    hungarian_align_slots,
    record_diagnostic_warnings,
    symmetry_diagnostics,
)
from mixtures import mixture_metrics
from permutation_scoring import score_permutations
from probability import residual_matrix, wrap
from reporting import (
    record_prediction,
    record_uncertainty_changes,
    record_variation,
)
from reproducibility import make_device_generator
from statistical_analysis import contribution_summary, orbit_risk, paired_bootstrap_ci
from training import load_trained_model, release_models, slots_forward


def synchronize(device):
    """Synchronize CUDA before and after recorded inference timings."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def description(entry, split, mode, draw, context, aligned=False):
    """Describe scientific conditions without repeating run configuration."""
    return {
        "group": entry["group"],
        "regime": entry["regime"],
        "architecture": entry["architecture"],
        "train_seed": entry["seed"],
        "split": split,
        "mode": mode,
        "slot_init_seed": draw,
        "context": context,
        "aligned": aligned,
    }


def slot_batches(backbone, dataset, indices, mode, draw, config, cache=None):
    """Yield ordered slot batches with measured backbone computation times."""
    if cache is not None:
        slots, target, selected, timings = cache.get_slots(
            backbone, dataset, indices, mode, draw
        )
        for batch, start in enumerate(
            range(0, len(indices), config.slot_extract_batch_size)
        ):
            subset = slice(start, start + config.slot_extract_batch_size)
            yield (
                torch.tensor(slots[subset], device=config.device),
                torch.from_numpy(target[subset]),
                torch.from_numpy(selected[subset]),
                timings[batch],
            )
        return
    loader = make_loader(dataset, indices, config.slot_extract_batch_size, config)
    generator = make_device_generator(config.device, draw)
    for images, target, selected in tqdm(
        loader, desc=f"Slot extraction {mode} draw {draw}", unit="batch", leave=False
    ):
        images = images.to(config.device, non_blocking=True)
        synchronize(config.device)
        started = time.perf_counter()
        slots = slots_forward(backbone, images, mode, draw, config, generator)
        synchronize(config.device)
        elapsed = time.perf_counter() - started
        yield slots, target, selected, elapsed


def predict_condition(
    backbone,
    predictor,
    dataset,
    indices,
    mode,
    draw,
    config,
    orbit=True,
    reference=None,
    cache=None,
):
    """Extract each image's slots once, then reuse them for every requested ordering."""
    parts = {"mu": [], "var": [], "y": [], "indices": [], "slots": []}
    aligned_parts = {"mu": [], "var": [], "y": [], "indices": []}
    matching = []
    timings = []
    permutations = PERMUTATIONS
    if not orbit:
        permutations = PERMUTATIONS[:1]
    offset = 0
    with torch.no_grad():
        batches = slot_batches(backbone, dataset, indices, mode, draw, config, cache)
        total_batches = (
            len(indices) + config.slot_extract_batch_size - 1
        ) // config.slot_extract_batch_size
        for batch, (slots, target, image_indices, backbone_seconds) in enumerate(
            tqdm(
                batches,
                total=total_batches,
                desc=f"Predictions {mode} draw {draw}",
                unit="batch",
                leave=False,
            )
        ):
            predictor(slots)
            synchronize(config.device)
            means = []
            variances = []
            head_seconds = []
            for permutation in permutations:
                started = time.perf_counter()
                mu, logvar = predictor(slots[:, permutation])
                synchronize(config.device)
                head_seconds.append(time.perf_counter() - started)
                if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                    raise ValueError("Nonfinite inference predictions.")
                means.append(mu.cpu().numpy())
                variances.append(logvar.clamp(-10, 10).exp().cpu().numpy())
            parts["mu"].append(np.stack(means, axis=1))
            parts["var"].append(np.stack(variances, axis=1))
            parts["y"].append(target.numpy())
            parts["indices"].append(image_indices.numpy())
            parts["slots"].append(slots.cpu().numpy())
            if reference is not None:
                current_reference = torch.from_numpy(
                    reference[offset : offset + len(slots)]
                )
                aligned, assignments, costs, margins = hungarian_align_slots(
                    current_reference, slots
                )
                mu, logvar = predictor(aligned.to(config.device))
                aligned_parts["mu"].append(mu.cpu().numpy())
                aligned_parts["var"].append(logvar.clamp(-10, 10).exp().cpu().numpy())
                aligned_parts["y"].append(target.numpy())
                aligned_parts["indices"].append(image_indices.numpy())
                for index, image_index in enumerate(image_indices):
                    matching.append(
                        {
                            "image_index": int(image_index),
                            "assignment": assignments[index].copy(),
                            "alignment_cost": float(costs[index]),
                            "cosine_margins": margins[index].copy(),
                        }
                    )
            timings.append(
                {
                    "batch": batch,
                    "n_images": len(slots),
                    "warmup_batch": batch == 0,
                    "backbone_seconds": backbone_seconds,
                    "identity_head_seconds": head_seconds[0],
                    "all24_head_seconds": sum(head_seconds),
                    "total_identity_seconds": backbone_seconds + head_seconds[0],
                    "total_ensemble_seconds": backbone_seconds + sum(head_seconds),
                }
            )
            offset += len(slots)
    pack = {}
    for key, arrays in parts.items():
        pack[key] = np.concatenate(arrays)
    pack["permutations"] = np.asarray(permutations)
    aligned_pack = None
    if reference is not None:
        aligned_pack = {}
        for key, arrays in aligned_parts.items():
            aligned_pack[key] = np.concatenate(arrays)
        aligned_pack["assignments"] = np.stack([row["assignment"] for row in matching])
        aligned_pack["alignment_costs"] = np.array(
            [row["alignment_cost"] for row in matching]
        )
        aligned_pack["cosine_margins"] = np.stack(
            [row["cosine_margins"] for row in matching]
        )
    return pack, aligned_pack, matching, timings


def identity_pack(pack):
    """Select the reference ordering without changing image or slot identities."""
    return {
        "mu": pack["mu"][:, 0],
        "var": pack["var"][:, 0],
        "y": pack["y"],
        "indices": pack["indices"],
    }


def save_prediction_arrays(path, pack):
    """Save predictive parameters, labels, image indices, and permutation order."""
    atomic_npz_save(
        path,
        mu=pack["mu"],
        var=pack["var"],
        y=pack["y"],
        indices=pack["indices"],
        permutations=pack["permutations"],
    )


def summarize_mixture(pack, variance, config):
    """Score the full equal-weight mixture using bounded memory batches."""
    pieces = {}
    for start in tqdm(
        range(0, len(pack["y"]), config.metric_batch_size),
        desc="Mixture scoring",
        unit="batch",
        leave=False,
    ):
        subset = slice(start, start + config.metric_batch_size)
        values = mixture_metrics(
            pack["mu"][subset],
            variance[subset],
            pack["y"][subset],
            config.interval_level,
        )
        for key, value in values.items():
            if key not in pieces:
                pieces[key] = []
            pieces[key].append(value)
    values = {}
    for key, arrays in pieces.items():
        values[key] = np.concatenate(arrays)
    contributions = {"mse": np.nanmean(values["squared_error"], axis=1)}
    names = {
        "mae": "absolute_error",
        "nll": "nll",
        "crps": "crps",
        "coverage90": "coverage90",
        "width90": "width90",
        "interval_score90": "interval_score90",
    }
    for name, key in names.items():
        contributions[name] = np.nanmean(values[key], axis=1)
    return contributions, values


def evaluate_orbit(pack, bundle, info, entry, tables, model_rows, scales, config):
    """Measure permutation risk, individual prediction changes, and mixture performance."""
    directory = config.array_root / "permutation_evaluation" / entry["id"]
    name = f"{info['mode']}_{info['slot_init_seed']}"
    record_variation(
        pack, info, scales, tables, directory / f"{name}_prediction_change.npz", config
    )
    reference_condition = (
        info["mode"] == "fixed_shared"
        and info["slot_init_seed"] == config.slot_init_seed
    )
    bootstrap_condition = reference_condition and info["split"] == "working"
    record_uncertainty_changes(pack, info, tables)
    mode_values = calibrated_variances(pack["var"], bundle)
    for variant_index, (variant, variance) in enumerate(mode_values.items()):
        contributions = []
        errors = []
        widths = []
        coverage = []
        compare_quantiles = reference_condition and variant in (
            "raw",
            "per_factor_temperature",
        )
        results = score_permutations(
            pack["mu"], variance, pack["y"], compare_quantiles, config
        )
        for index, permutation in enumerate(PERMUTATIONS):
            values, scores, likelihood, diagnostics, comparisons = results[index]
            contributions.append(values)
            if reference_condition:
                residual = residual_matrix(pack["mu"][:, index], pack["y"])
                for factor in range(10):
                    tables.add(
                        "per_factor_full_extraction.csv",
                        dict(
                            info,
                            context="slot_permutation",
                            variant=variant,
                            permutation="-".join(map(str, permutation)),
                            is_identity=index == 0,
                            factor_index=factor,
                            factor=FACTOR_NAMES[factor],
                            rmse=float(np.sqrt((residual[:, factor] ** 2).mean())),
                            mae=float(np.abs(residual[:, factor]).mean()),
                            nll=float(likelihood[:, factor].mean()),
                            crps=float(scores[:, factor].mean()),
                            mean_predicted_variance=float(
                                variance[:, index, factor].mean()
                            ),
                        ),
                    )

            row = dict(
                info,
                variant=variant,
                permutation="-".join(map(str, permutation)),
                is_identity=index == 0,
            )
            row.update(contribution_summary(values))
            tables.add("complete_slot_permutation_table.csv", row)
            if variant == "raw":
                errors.append(np.abs(residual_matrix(pack["mu"][:, index], pack["y"])))
                widths.append(diagnostics["width"])
                coverage.append(diagnostics["inside"])
            for comparison in comparisons:
                tables.add("crps_reporting_changes.csv", dict(row, **comparison))
        rmse = []
        orbit_values = {}
        for values in contributions:
            rmse.append(contribution_summary(values)["rmse"])
            for metric, contribution in values.items():
                orbit_values.setdefault(metric, np.zeros_like(contribution))
                orbit_values[metric] += contribution / len(PERMUTATIONS)
        summary = dict(
            info,
            variant=variant,
            identity_rmse_raw=rmse[0],
            mean_24_rmse_raw=float(np.mean(rmse)),
            minimum_rmse_raw=float(np.min(rmse)),
            maximum_rmse_raw=float(np.max(rmse)),
            orbit_to_identity_ratio=float(np.mean(rmse) / max(rmse[0], 1e-15)),
        )
        tables.add("slot_permutation_summary.csv", summary)
        tables.add(
            "orbit_risk_summary.csv", dict(info, variant=variant, **orbit_risk(rmse))
        )
        if bootstrap_condition:
            for index, values in enumerate(contributions):
                row = dict(
                    info,
                    variance_variant=variant,
                    permutation="-".join(map(str, PERMUTATIONS[index])),
                    is_identity=index == 0,
                )
                seed = (
                    config.permutation_bootstrap_seed
                    + 10000 * variant_index
                    + 1000 * entry["seed"]
                    + index
                )
                if index == 0:
                    for metric in contribution_summary(values):
                        row[f"delta_{metric}"] = 0.0
                        row[f"delta_{metric}_ci95_lo"] = 0.0
                        row[f"delta_{metric}_ci95_hi"] = 0.0
                else:
                    row.update(
                        paired_bootstrap_ci(
                            contributions[0],
                            values,
                            config.permutation_bootstrap_repetitions,
                            seed,
                        )
                    )
                tables.add("slot_permutation_paired_bootstrap_ci.csv", row)
        readouts = [("identity", contributions[0]), ("orbit", orbit_values)]
        mixture_condition = (
            reference_condition or info["mode"] == "per_example_stochastic"
        )
        mixture_variant = variant in ("raw", "per_factor_temperature")
        if mixture_condition and mixture_variant:
            mixture, distribution = summarize_mixture(pack, variance, config)
            identity_distribution_pack = {
                "mu": pack["mu"][:, :1],
                "var": variance[:, :1],
                "y": pack["y"],
            }
            identity_distribution, identity_values = summarize_mixture(
                identity_distribution_pack, variance[:, :1], config
            )
            for readout, values in (
                ("identity", identity_distribution),
                ("ensemble24", mixture),
            ):
                row = dict(info, variant=variant, readout=readout)
                row.update(contribution_summary(values))
                tables.add("distribution_interval_comparison.csv", row)
            readouts.append(("ensemble24", mixture))
        for readout, values in readouts:
            row = dict(info, variant=variant, readout=readout)
            row.update(contribution_summary(values))
            if readout == "orbit":
                row["rmse"] = float(np.mean(rmse))
            if readout == "ensemble24":
                row["undefined_circular_direction_fraction"] = float(
                    1 - distribution["circular_direction_defined"].mean()
                )
                row["joint_nll"] = float(distribution["joint_nll"].mean())
            tables.add("predictive_mixture_metrics.csv", row)
            if reference_condition or info["mode"] == "per_example_stochastic":
                model_rows.append(row)
        if variant == "raw":
            error_conditioned_tables(errors, widths, coverage, info, tables)


def error_conditioned_tables(errors, widths, coverage, info, tables):
    """Compare uncertainty at common observed-error bins without a causal interpretation."""
    for factor in range(10):
        first_error = errors[0][:, factor]
        first_width = widths[0][:, factor]
        first_coverage = coverage[0][:, factor]
        all_error = np.concatenate([values[:, factor] for values in errors])
        all_width = np.concatenate([values[:, factor] for values in widths])
        all_coverage = np.concatenate([values[:, factor] for values in coverage])
        edges = np.unique(
            np.quantile(np.r_[first_error, all_error], np.linspace(0, 1, 11))
        )
        for index in range(len(edges) - 1):
            first = (first_error >= edges[index]) & (first_error < edges[index + 1])
            second = (all_error >= edges[index]) & (all_error < edges[index + 1])
            if index == len(edges) - 2:
                first = (first_error >= edges[index]) & (
                    first_error <= edges[index + 1]
                )
                second = (all_error >= edges[index]) & (all_error <= edges[index + 1])
            if not first.any() or not second.any():
                continue
            tables.add(
                "error_conditioned_uncertainty.csv",
                dict(
                    info,
                    factor_index=factor,
                    bin_low=edges[index],
                    bin_high=edges[index + 1],
                    identity_n=int(first.sum()),
                    orbit_n=int(second.sum()),
                    identity_coverage=float(first_coverage[first].mean()),
                    orbit_coverage=float(all_coverage[second].mean()),
                    identity_width=float(first_width[first].mean()),
                    orbit_width=float(all_width[second].mean()),
                ),
            )


def evaluate_model(
    entry, train, test, splits, tables, model_rows, config, manifest, evaluation_cache
):
    """Execute every inference stage for one selected model."""
    backbone, predictor = load_trained_model(entry, config)
    cache = evaluation_cache.for_model(entry)
    with manifest.stage(f"calibration/{entry['id']}"):
        bundle = calibrate_and_audit_model(
            backbone, predictor, entry, train, splits, tables, config, cache, manifest
        )
    with manifest.stage(f"initialization_and_permutations/{entry['id']}"):
        reference = evaluate_initialization_conditions(
            backbone,
            predictor,
            entry,
            test,
            train,
            splits,
            bundle,
            tables,
            model_rows,
            config,
            cache,
        )
    with manifest.stage(f"slot_symmetry/{entry['id']}"):
        evaluate_slot_symmetry(predictor, reference["slots"], entry, tables, config)
        tables.flush()
    with manifest.stage(f"corruptions/{entry['id']}"):
        evaluate_corruptions(
            backbone, predictor, entry, test, splits, bundle, tables, config, cache
        )
    with manifest.stage(f"audit_subset/{entry['id']}"):
        evaluate_audit_subset(
            backbone, predictor, entry, test, splits, bundle, tables, config, cache
        )
    parameter_count = sum(parameter.numel() for parameter in predictor.parameters())
    backbone_count = sum(parameter.numel() for parameter in backbone.parameters())
    tables.add(
        "model_parameter_counts.csv",
        dict(
            group=entry["group"],
            regime=entry["regime"],
            architecture=entry["architecture"],
            train_seed=entry["seed"],
            predictor_parameters=parameter_count,
            backbone_parameters=backbone_count,
        ),
    )
    del backbone, predictor, reference
    release_models()


def calibrate_and_audit_model(
    backbone, predictor, entry, train, splits, tables, config, cache=None, manifest=None
):
    """Fit native-initialization calibration and audit actual predicted scales."""
    mode = "fixed_shared"
    calibration_draw = config.slot_init_seed
    validation_draw = config.slot_init_seed
    if entry["architecture"] == "flatmlp_stochastic_init":
        mode = "per_example_stochastic"
        calibration_draw = config.calibration_draw_seed
        validation_draw = config.validation_stream_seed
    calibration, aligned, matching, timings = predict_condition(
        backbone,
        predictor,
        train,
        splits["calibration"],
        mode,
        calibration_draw,
        config,
        orbit=False,
        cache=cache,
    )
    bundle = fit_calibration_bundle(identity_pack(calibration), config)
    raw_directory = config.array_root / "raw_predictions" / entry["id"]
    calibration_path = (
        config.array_root
        / "calibration"
        / entry["group"]
        / entry["regime"]
        / entry["architecture"]
        / f"seed_{entry['seed']}"
        / "calibration_bundle.joblib"
    )
    with atomic_path(calibration_path) as temporary:
        joblib.dump(bundle, temporary)
    save_prediction_arrays(raw_directory / "calibration_predictions.npz", calibration)
    validation, aligned, matching, timings = predict_condition(
        backbone,
        predictor,
        train,
        splits["validation"],
        mode,
        validation_draw,
        config,
        orbit=False,
        cache=cache,
    )
    score_rows, parameter_row = empirical_score_audit(
        predictor, identity_pack(validation), validation["slots"], config
    )
    info = description(
        entry, "validation", mode, validation_draw, "numerical_score_audit"
    )
    for row in score_rows:
        tables.add("empirical_crps_gradients.csv", dict(info, **row))
    tables.add("predictor_gradient_comparison.csv", dict(info, **parameter_row))
    tables.flush()
    if manifest is not None:
        record_diagnostic_warnings(score_rows, "empirical_crps_gradients.csv", manifest, **info)
    return bundle


def evaluate_initialization_conditions(
    backbone,
    predictor,
    entry,
    test,
    train,
    splits,
    bundle,
    tables,
    model_rows,
    config,
    cache=None,
):
    """Evaluate fixed, stochastic, and reassigned slot initializations."""
    raw_directory = config.array_root / "raw_predictions" / entry["id"]
    scales = np.ones(10) * np.pi
    scales[:7] = np.maximum(train.latents[splits["train"], :7].std(0), 1e-8)
    reference, aligned, matching, timings = predict_condition(
        backbone,
        predictor,
        test,
        splits["working"],
        "fixed_shared",
        config.slot_init_seed,
        config,
        cache=cache,
    )
    reference_slots = reference["slots"]
    conditions = [("fixed_shared", config.slot_init_seed)]
    for current_mode in config.initialization_modes:
        for draw in config.evaluation_draw_seeds:
            conditions.append((current_mode, draw))
    stochastic_means = []
    for current_mode, draw in tqdm(
        conditions, desc=f"{entry['id']} initialization conditions", unit="condition"
    ):
        tqdm.write(f"Evaluating {entry['id']}: {current_mode}, draw {draw}")
        if current_mode == "fixed_shared" and draw == config.slot_init_seed:
            pack = reference
            aligned_pack = identity_pack(reference)
            matching = []
        else:
            pack, aligned_pack, matching, timings = predict_condition(
                backbone,
                predictor,
                test,
                splits["working"],
                current_mode,
                draw,
                config,
                reference=reference_slots,
                cache=cache,
            )
        info = description(entry, "working", current_mode, draw, "slot_initialization")
        name = f"{current_mode}_{draw}"
        if (
            current_mode == "fixed_shared" and draw == config.slot_init_seed
        ) or current_mode == "per_example_stochastic":
            save_prediction_arrays(raw_directory / f"{name}.npz", pack)
        record_prediction(identity_pack(pack), bundle, info, tables, config)
        record_prediction(
            aligned_pack, bundle, dict(info, aligned=True), tables, config
        )
        if matching:
            costs = aligned_pack["alignment_costs"]
            margins = aligned_pack["cosine_margins"]
            tables.add(
                "cosine_margins.csv",
                dict(
                    info,
                    alignment_cost_mean=float(costs.mean()),
                    cosine_margin_mean=float(margins.mean()),
                    cosine_margin_minimum=float(margins.min()),
                ),
            )
        retained = timings[1:]
        if not retained:
            retained = timings
        total_images = sum(row["n_images"] for row in retained)
        runtime = dict(info)
        runtime["backbone_timing_shared"] = cache is not None
        for metric_name in (
            "backbone_seconds",
            "identity_head_seconds",
            "all24_head_seconds",
            "total_identity_seconds",
            "total_ensemble_seconds",
        ):
            runtime[metric_name + "_per_image"] = (
                sum(row[metric_name] for row in retained) / total_images
            )
        tables.add("runtime_summary.csv", runtime)
        evaluate_orbit(pack, bundle, info, entry, tables, model_rows, scales, config)
        diagnostic_key = None
        if cache is not None:
            diagnostic_key = cache.condition_key(
                test, splits["working"], current_mode, draw
            )
        if cache is not None and diagnostic_key in cache.correspondence:
            correspondence, distributions = cache.correspondence[diagnostic_key]
        else:
            correspondence, distributions = correspondence_diagnostics(
                backbone, test, splits["working"], current_mode, draw, config
            )
            if cache is not None:
                cache.correspondence[diagnostic_key] = (correspondence, distributions)
        tables.add("mask_role_diagnostics.csv", dict(info, **correspondence))
        atomic_npz_save(
            config.array_root
            / "initialization_mechanism"
            / entry["id"]
            / f"{name}_distributions.npz",
            **distributions,
        )
        if current_mode == "per_example_stochastic":
            stochastic_means.append(pack["mu"][:, 0])
    draw_change = np.stack(stochastic_means, axis=1)
    difference = draw_change[:, :, None] - draw_change[:, None, :]
    difference[..., 7:] = wrap(difference[..., 7:])
    individual = np.abs(difference).max(axis=(1, 2))
    atomic_npz_save(
        raw_directory / "draw_variation.npz",
        indices=splits["working"],
        maximum_pairwise_mean_change=individual,
    )
    tables.add(
        "draw_variation_summary.csv",
        dict(
            description(
                entry, "working", "per_example_stochastic", -1, "initialization_draws"
            ),
            normalized_change_p50=float(np.median((individual / scales).max(1))),
            normalized_change_p95=float(
                np.quantile((individual / scales).max(1), 0.95)
            ),
            normalized_change_max=float((individual / scales).max()),
        ),
    )
    record_draw_variation_by_factor(
        individual, scales, entry, len(stochastic_means), tables
    )
    return reference


def record_draw_variation_by_factor(changes, scales, entry, n_draws, tables):
    """Report identity-order initialization changes separately for each factor."""
    info = description(
        entry, "working", "per_example_stochastic", -1, "initialization_draws"
    )
    for factor, name in enumerate(FACTOR_NAMES):
        normalized = changes[:, factor] / scales[factor]
        median = np.nan
        upper_quantile = np.nan
        status = "insufficient_draws"
        if n_draws >= 2:
            median = float(np.median(normalized))
            upper_quantile = float(np.quantile(normalized, 0.95))
            status = "available"
        tables.add(
            "draw_variation_by_factor.csv",
            dict(
                info,
                factor_index=factor,
                factor=name,
                normalizing_scale=scales[factor],
                n_draws=n_draws,
                normalized_change_p50=median,
                normalized_change_p95=upper_quantile,
                status=status,
            ),
        )


def evaluate_slot_symmetry(predictor, reference_slots, entry, tables, config):
    """Save Jacobian and first-layer weight statistics for the reference slot sets."""
    jacobian, weight_row, distributions = symmetry_diagnostics(
        predictor, reference_slots, config
    )
    info = description(
        entry, "working", "fixed_shared", config.slot_init_seed, "slot_symmetry"
    )
    for row in jacobian:
        tables.add("jacobian_permutation_symmetry.csv", dict(info, **row))
    tables.add("weight_block_symmetry.csv", dict(info, **weight_row))
    atomic_npz_save(
        config.array_root / "symmetry_diagnostics" / f"{entry['id']}.npz",
        **distributions,
    )


def evaluate_corruptions(
    backbone, predictor, entry, test, splits, bundle, tables, config, cache=None
):
    """Evaluate image corruptions in identity slot order."""
    mode = "fixed_shared"
    if entry["architecture"] == "flatmlp_stochastic_init":
        mode = "per_example_stochastic"
    for corruption in config.corruption_types:
        for severity in tqdm(
            config.corruption_severities,
            desc=f"{entry['id']} {corruption}",
            unit="severity",
        ):
            tqdm.write(f"Evaluating {entry['id']}: {corruption}, severity {severity}")
            dataset = CorruptionDataset(test, corruption, severity, config)
            corruption_draw = config.slot_init_seed
            if mode == "per_example_stochastic":
                corruption_draw = config.evaluation_draw_seeds[0]
            pack, aligned, matching, timings = predict_condition(
                backbone,
                predictor,
                dataset,
                splits["working"],
                mode,
                corruption_draw,
                config,
                orbit=False,
                cache=cache,
            )
            info = dict(
                description(
                    entry, "working", mode, corruption_draw, "image_corruption"
                ),
                corruption=corruption,
                severity=severity,
            )
            values = record_prediction(
                identity_pack(pack), bundle, info, tables, config
            )
            for variant, contribution in values.items():
                tables.add(
                    "exact_corruption_level_metrics.csv",
                    dict(
                        info,
                        variant=variant,
                        readout="identity",
                        **contribution_summary(contribution),
                    ),
                )


def evaluate_audit_subset(
    backbone, predictor, entry, test, splits, bundle, tables, config, cache=None
):
    """Evaluate the previously used audit subset without recalibration."""
    if entry["architecture"] == "flatmlp_stochastic_init":
        return
    audit, aligned, matching, timings = predict_condition(
        backbone,
        predictor,
        test,
        splits["audit"],
        "fixed_shared",
        config.slot_init_seed,
        config,
        orbit=False,
        cache=cache,
    )
    info = description(entry, "audit", "fixed_shared", config.slot_init_seed, "audit")
    record_prediction(identity_pack(audit), bundle, info, tables, config)
    save_prediction_arrays(
        config.array_root / "audit_predictions" / f"{entry['id']}.npz", audit
    )
