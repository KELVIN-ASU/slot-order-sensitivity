### Cosine correspondence, slot symmetry, and numerical scoring diagnostics.###

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment
from scipy.special import ndtr
from scipy.stats import norm
from tqdm import tqdm

from data import make_loader
from probability import (
    PERIOD,
    circular_crps_quantile,
    circular_crps_quantile_torch,
    circular_crps_torch,
    mixed_crps_torch,
    wrap,
)
from reproducibility import make_device_generator, set_global_seed
from training import build_predictor, slots_forward

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def diagnostic_measurement(check, observed, reference, tolerance, **context):
    """Record a measured discrepancy and its tolerance without interrupting execution."""
    error = abs(observed - reference)
    return dict(
        context,
        check=check,
        observed=observed,
        reference=reference,
        absolute_error=error,
        tolerance=tolerance,
        passed=math.isfinite(error) and error <= tolerance,
    )


def record_diagnostic_warnings(rows, filename, manifest, **context):
    """Record a warning when saved diagnostic rows contain failed comparisons."""
    failed = 0
    for row in rows:
        flags = []
        for name, value in row.items():
            if name == "passed" or name.endswith("_passed"):
                flags.append(value)
        if any(value is False for value in flags):
            failed += 1
    if failed:
        manifest.warning(
            dict(
                context,
                check=filename,
                failed_rows=failed,
                total_rows=len(rows),
                details=filename,
            )
        )


def hungarian_align_slots(reference_slots, candidate_slots):
    """Align slot vectors with cosine-distance Hungarian assignment for each image."""
    reference = reference_slots.detach().cpu().float()
    candidate = candidate_slots.detach().cpu().float()
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("Expected matching (images, slots, dimensions) tensors.")
    reference_normalized = functional.normalize(reference, p=2, dim=-1, eps=1e-12)
    candidate_normalized = functional.normalize(candidate, p=2, dim=-1, eps=1e-12)
    similarity = torch.einsum(
        "bkd,bjd->bkj", reference_normalized, candidate_normalized
    )
    cost = (1 - similarity).numpy()
    aligned = torch.empty_like(candidate)
    assignments = np.empty(reference.shape[:2], dtype=np.int64)
    costs = np.empty(len(reference), dtype=np.float64)
    margins = np.empty(reference.shape[:2], dtype=np.float64)
    for image in range(len(reference)):
        rows, columns = linear_sum_assignment(cost[image])
        assignment = np.empty(reference.shape[1], dtype=np.int64)
        assignment[rows] = columns
        aligned[image] = candidate[image, torch.as_tensor(assignment)]
        assignments[image] = assignment
        costs[image] = cost[image, rows, columns].mean()
        for slot in range(reference.shape[1]):
            alternative = np.delete(cost[image, slot], assignment[slot])
            margins[image, slot] = (
                alternative.min() - cost[image, slot, assignment[slot]]
            )
    return aligned, assignments, costs, margins


def correspondence_diagnostics(backbone, dataset, indices, mode, draw, config):
    """Measure mask overlap, attention, centroid, and slot-assignment statistics."""
    selected = indices[: config.stability_max_images]
    loader = make_loader(dataset, selected, config.slot_extract_batch_size, config)
    stream = make_device_generator(config.device, draw)
    counts = np.zeros((config.num_slots, config.num_slots), dtype=np.int64)
    masks_iou = []
    centroid_shift = []
    attention_overlap = []
    margins = []
    costs = []
    rows, columns = np.meshgrid(
        np.arange(config.image_size), np.arange(config.image_size), indexing="ij"
    )
    with torch.no_grad():
        for images, target, image_indices in tqdm(
            loader, desc="Backbone equivariance", unit="batch", leave=False
        ):
            images = images.to(config.device)
            reference, reference_attention = slots_forward(
                backbone,
                images,
                "fixed_shared",
                config.slot_init_seed,
                config,
                attention=True,
            )
            candidate, candidate_attention = slots_forward(
                backbone, images, mode, draw, config, stream, attention=True
            )
            aligned, assignments, batch_costs, batch_margins = hungarian_align_slots(
                reference, candidate
            )
            reference_reconstruction, reference_masks = backbone.decoder(reference)
            candidate_reconstruction, candidate_masks = backbone.decoder(candidate)
            first = reference_masks[:, :, 0].cpu().numpy()
            second = candidate_masks[:, :, 0].cpu().numpy()
            batch_index = np.arange(len(images))[:, None]
            second = second[batch_index, assignments]
            second_attention = candidate_attention.cpu().numpy()[
                batch_index, assignments
            ]
            first_attention = reference_attention.cpu().numpy()
            intersection = np.minimum(first, second).sum((-1, -2))
            union = np.maximum(first, second).sum((-1, -2))
            masks_iou.append(intersection / (union + 1e-8))
            first_mass = first.sum((-1, -2)) + 1e-8
            second_mass = second.sum((-1, -2)) + 1e-8
            row_difference = (first * rows).sum((-1, -2)) / first_mass
            row_difference -= (second * rows).sum((-1, -2)) / second_mass
            column_difference = (first * columns).sum((-1, -2)) / first_mass
            column_difference -= (second * columns).sum((-1, -2)) / second_mass
            centroid_shift.append(np.hypot(row_difference, column_difference))
            attention_overlap.append(
                np.minimum(first_attention, second_attention).sum(-1)
            )
            margins.append(batch_margins)
            costs.append(batch_costs)
            for slot in range(config.num_slots):
                counts[slot] += np.bincount(
                    assignments[:, slot], minlength=config.num_slots
                )
    distributions = {
        "mask_iou": np.concatenate(masks_iou),
        "centroid_shift": np.concatenate(centroid_shift),
        "attention_overlap": np.concatenate(attention_overlap),
        "cosine_margin": np.concatenate(margins),
        "alignment_cost": np.concatenate(costs),
        "assignment_counts": counts,
        "sample_indices": selected,
    }
    proportions = counts / counts.sum(axis=1, keepdims=True)
    summary = {
        "mode": mode,
        "draw_seed": draw,
        "n_images": len(selected),
        "dominant_assignment_proportion_mean": float(proportions.max(1).mean()),
        "assignment_entropy_mean": float(
            -(proportions * np.log(proportions + 1e-12)).sum(1).mean()
        ),
        "identity_assignment_rate_mean": float(np.diag(proportions).mean()),
    }
    for name in (
        "mask_iou",
        "centroid_shift",
        "attention_overlap",
        "cosine_margin",
        "alignment_cost",
    ):
        summary[f"{name}_mean"] = float(distributions[name].mean())
        summary[f"{name}_std"] = float(distributions[name].std())
    return summary, distributions


def input_jacobian(predictor, slots):
    """Differentiate predictive means with respect to every input slot coordinate."""
    values = slots.detach().requires_grad_(True)
    mu, logvar = predictor(values)
    derivatives = []
    for factor in range(mu.shape[1]):
        derivative = torch.autograd.grad(
            mu[:, factor].sum(), values, retain_graph=True
        )[0]
        derivatives.append(derivative.detach().cpu().numpy())
    return np.stack(derivatives, axis=1)


def symmetry_diagnostics(predictor, slots, config):
    """Measure mean-Jacobian and first-layer slot-block permutation symmetry."""
    predictor.eval()
    sample = torch.as_tensor(slots[: config.symmetry_sample_size], device=config.device)
    base = input_jacobian(predictor, sample)
    rows = []
    distributions = {}
    for permutation in PERMUTATIONS:
        current = input_jacobian(predictor, sample[:, permutation])
        expected = base[:, :, permutation, :]
        numerator = np.linalg.norm(
            (current - expected).reshape(len(sample), -1), axis=1
        )
        denominator = np.linalg.norm(expected.reshape(len(sample), -1), axis=1) + 1e-12
        mismatch = numerator / denominator
        name = "-".join(map(str, permutation))
        distributions[name] = mismatch
        rows.append(
            {
                "permutation": name,
                "is_identity": permutation == PERMUTATIONS[0],
                "relative_mismatch_mean": float(mismatch.mean()),
                "relative_mismatch_std": float(mismatch.std()),
                "relative_mismatch_max": float(mismatch.max()),
                "n_examples": len(sample),
            }
        )
    weight_row = {"relative_asymmetry": np.nan, "mean_pairwise_cosine": np.nan}
    if hasattr(predictor, "encoder"):
        weights = predictor.encoder[0].weight.detach().cpu().numpy()
        blocks = np.stack(np.split(weights, config.num_slots, axis=1))
        average = blocks.mean(0)
        weight_row["relative_asymmetry"] = float(
            np.linalg.norm(blocks - average) / (np.linalg.norm(blocks) + 1e-12)
        )
        similarities = []
        for left, right in itertools.combinations(blocks, 2):
            similarities.append(
                np.sum(left * right)
                / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12)
            )
        weight_row["mean_pairwise_cosine"] = float(np.mean(similarities))
    return rows, weight_row, distributions


def distance_integral_reference(center, sigma):
    """Integrate circular distance with normal CDFs and densities, without harmonics."""
    center = float(wrap(center))
    aliases = int(math.ceil(10 * sigma / PERIOD)) + 3
    means = center + np.arange(-aliases, aliases + 1) * PERIOD

    def integral(lower, upper):
        left = (lower - means) / sigma
        right = (upper - means) / sigma
        value = means * (ndtr(right) - ndtr(left))
        value += sigma * (norm.pdf(left) - norm.pdf(right))
        return value

    return float(np.sum(integral(0, PERIOD / 2) - integral(-PERIOD / 2, 0)))


def score_integral_reference(mu, logvar, target):
    """Evaluate the score using independent alias-integrated expectations."""
    sigma = math.exp(logvar / 2)
    first = distance_integral_reference(mu - target, sigma)
    second = distance_integral_reference(0, math.sqrt(2) * sigma)
    return first - 0.5 * second


def score_reference_gradients(mu, logvar, target):
    """Use centered finite differences of the independent integral reference."""
    step = 1e-5
    mean_gradient = score_integral_reference(mu + step, logvar, target)
    mean_gradient -= score_integral_reference(mu - step, logvar, target)
    variance_gradient = score_integral_reference(mu, logvar + step, target)
    variance_gradient -= score_integral_reference(mu, logvar - step, target)
    return mean_gradient / (2 * step), variance_gradient / (2 * step)


def numerical_score_case(dtype, sigma, fraction, config):
    """Compare one circular score and its derivatives with the integral reference."""
    mu = torch.tensor([0.0], device=config.device, dtype=dtype, requires_grad=True)
    logvar = torch.tensor(
        [2 * math.log(sigma)], device=config.device, dtype=dtype, requires_grad=True
    )
    target = torch.tensor([fraction * PERIOD], device=config.device, dtype=dtype)
    score = circular_crps_torch(mu, logvar, target, config.fourier_terms)
    gradients = torch.autograd.grad(score.sum(), (mu, logvar))
    variance_value = float(logvar.item())
    reference = score_integral_reference(
        0, max(-10, min(10, variance_value)), target.item()
    )
    reference_gradients = score_reference_gradients(0, variance_value, target.item())
    gradient_error = max(
        abs(gradients[0].item() - reference_gradients[0]),
        abs(gradients[1].item() - reference_gradients[1]),
    )
    interior = -9.999 < variance_value < 9.999
    error = abs(score.item() - reference)
    gradient_status = gradient_comparison(gradient_error, interior)
    return {
        "dtype": str(dtype),
        "sigma": sigma,
        "target_fraction": fraction,
        "score": score.item(),
        "reference": reference,
        "score_error": error,
        "d_mu": gradients[0].item(),
        "d_logvar": gradients[1].item(),
        "gradient_error": gradient_error,
        "score_tolerance": 2e-7,
        "score_passed": math.isfinite(error) and error <= 2e-7,
        "gradient_tolerance": 2e-5,
        **gradient_status,
        "quantile16": float(
            circular_crps_quantile(
                np.array([0.0]), np.array([sigma**2]), np.array([target.item()]), 16
            )[0]
        ),
        "quantile64": float(
            circular_crps_quantile(
                np.array([0.0]), np.array([sigma**2]), np.array([target.item()]), 64
            )[0]
        ),
    }


def gradient_comparison(error, checked):
    """Mark gradients outside the variance clamp as unchecked."""
    passed = None
    if checked:
        passed = math.isfinite(error) and error <= 2e-5
    return {"gradient_checked": checked, "gradient_passed": passed}


def check_score_differentiation(config):
    """Check automatic derivatives and periodic score identities."""
    mu = torch.tensor(
        [0.137], dtype=torch.float64, device=config.device, requires_grad=True
    )
    logvar = torch.tensor(
        [-1.3], dtype=torch.float64, device=config.device, requires_grad=True
    )
    target = torch.tensor([0.371], dtype=torch.float64, device=config.device)
    rows = []
    gradient_passed = torch.autograd.gradcheck(
        lambda m, v: circular_crps_torch(m, v, target, config.fourier_terms),
        (mu, logvar),
        raise_exception=False,
    )
    rows.append(diagnostic_measurement("autograd_gradcheck", int(gradient_passed), 1, 0))
    score = circular_crps_torch(mu, logvar, target, config.fourier_terms)
    shifted = circular_crps_torch(mu + PERIOD, logvar, target, config.fourier_terms)
    rows.append(
        diagnostic_measurement(
            "circular_score_periodicity",
            shifted.item(),
            score.item(),
            1e-12,
        )
    )
    for value in (-12.0, 12.0):
        boundary = torch.tensor(
            [value], device=config.device, dtype=torch.float64, requires_grad=True
        )
        score = circular_crps_torch(mu, boundary, target, config.fourier_terms)
        derivative = torch.autograd.grad(score.sum(), boundary)[0]
        rows.append(
            diagnostic_measurement(
                "clamped_variance_gradient",
                derivative.item(),
                0.0,
                0.0,
                logvar=value,
            )
        )
    uniform_logvar = torch.full_like(logvar, 2 * math.log(10 * PERIOD))
    uniform_score = circular_crps_torch(
        mu, uniform_logvar, target, config.fourier_terms
    )
    rows.append(
        diagnostic_measurement(
            "circular_uniform_limit",
            uniform_score.item(),
            PERIOD / 8,
            1e-10,
        )
    )
    return rows


def check_wrapped_quadrature():
    """Check normalization and circular distance with independent quadrature."""
    rows = []
    for sigma in (0.01 * PERIOD, 0.3 * PERIOD, PERIOD):
        center = 0.234 * PERIOD
        aliases = np.arange(-30, 31) * PERIOD

        def density(value, center=center, aliases=aliases, sigma=sigma):
            standardized = (value - center + aliases) / sigma
            return np.exp(-0.5 * standardized**2).sum() / (sigma * math.sqrt(2 * math.pi))

        mass = quad(density, -PERIOD / 2, PERIOD / 2, epsabs=1e-10)[0]
        distance = quad(
            lambda value: abs(value) * density(value),
            -PERIOD / 2,
            PERIOD / 2,
            points=[0, center],
            epsabs=1e-10,
        )[0]
        rows.append(
            diagnostic_measurement(
                "wrapped_normal_mass",
                mass,
                1.0,
                1e-8,
                sigma=sigma,
            )
        )
        reference = distance_integral_reference(center, sigma)
        rows.append(
            diagnostic_measurement(
                "wrapped_normal_distance",
                distance,
                reference,
                1e-8,
                sigma=sigma,
            )
        )
    return rows


def check_invariant_pooling(config):
    """Verify DeepSets mean and variance invariance on every permutation."""
    sample = torch.randn(3, config.num_slots, config.slot_dim, device=config.device)
    predictor = build_predictor("deepsets", config).to(config.device).eval()
    rows = []
    with torch.no_grad():
        base = predictor(sample)
        for permutation in PERMUTATIONS:
            current = predictor(sample[:, permutation])
            for index in (0, 1):
                error = (base[index] - current[index]).abs().max().item()
                rows.append(
                    diagnostic_measurement(
                        "deepsets_invariance",
                        error,
                        0.0,
                        1e-5,
                        permutation="-".join(map(str, permutation)),
                        output_index=index,
                    )
                )
    return rows


def numerical_checks(config, diagnostic_rows=None):
    """Validate scoring values, derivatives, integration, and invariant pooling."""
    set_global_seed(config.split_seed)
    rows = []
    scales = np.unique(
        np.r_[np.geomspace(math.exp(-5), math.exp(5), 17), PERIOD, 10 * PERIOD]
    )
    for dtype in (torch.float32, torch.float64):
        for sigma in scales:
            for fraction in (0, 0.137, 0.4, 0.5):
                rows.append(numerical_score_case(dtype, sigma, fraction, config))
    if diagnostic_rows is None:
        diagnostic_rows = []
    diagnostic_rows.extend(check_score_differentiation(config))
    diagnostic_rows.extend(check_wrapped_quadrature())
    diagnostic_rows.extend(check_invariant_pooling(config))
    return rows


def backbone_equivariance(backbone, dataset, indices, label, config):
    """Measure initial-vector permutation equivariance and cosine-alignment recovery."""
    loader = make_loader(dataset, indices[:8], 8, config)
    images, target, selected = next(iter(loader))
    images = images.to(config.device)
    generator = make_device_generator(config.device, config.slot_init_seed)
    rows = []
    with torch.no_grad():
        features = backbone.project(backbone.encoder(images))
        noise = torch.randn(
            len(images),
            config.num_slots,
            config.slot_dim,
            device=config.device,
            generator=generator,
        )
        reference = backbone.slot_attention(features, init_noise=noise)
        for permutation in PERMUTATIONS:
            current = backbone.slot_attention(features, init_noise=noise[:, permutation])
            expected = reference[:, permutation]
            error = (current - expected).abs().max().item()
            aligned, assignments, costs, margins = hungarian_align_slots(reference, current)
            alignment_error = (aligned - reference.cpu()).abs().max().item()
            rows.append(
                {
                    "backbone": label,
                    "permutation": "-".join(map(str, permutation)),
                    "equivariance_max_error": error,
                    "aligned_max_error": alignment_error,
                    "tolerance": config.equivariance_atol,
                    "equivariance_passed": error <= config.equivariance_atol,
                    "alignment_passed": alignment_error <= config.equivariance_atol,
                }
            )
    return rows


def equivariance_warning(rows, label, config):
    """Summarize permutations that exceed the equivariance tolerance, if any."""
    failed = [
        row
        for row in rows
        if not (row["equivariance_passed"] and row["alignment_passed"])
    ]
    if not failed:
        return None
    return {
        "check": "slot_equivariance",
        "backbone": label,
        "tolerance": config.equivariance_atol,
        "failed_permutations": len(failed),
        "total_permutations": len(rows),
        "max_equivariance_error": max(row["equivariance_max_error"] for row in rows),
        "max_aligned_error": max(row["aligned_max_error"] for row in rows),
        "details": "slot_equivariance.csv",
    }


def empirical_score_audit(predictor, pack, slots, config):
    """Audit actual predicted scales, score derivatives, and predictor parameter gradients."""
    scales = np.sqrt(pack["var"][:, 7:])
    selected = []
    for factor in range(3):
        positions = np.linspace(
            0, len(scales) - 1, min(config.empirical_gradient_samples, len(scales))
        ).astype(int)
        selected.extend(np.argsort(scales[:, factor])[positions].tolist())
    rows = []
    for index in sorted(set(selected)):
        for factor in range(7, 10):
            mean_value = float(pack["mu"][index, factor])
            logvar_value = float(np.log(pack["var"][index, factor]))
            target_value = float(pack["y"][index, factor])
            mu = torch.tensor(
                [mean_value],
                device=config.device,
                dtype=torch.float64,
                requires_grad=True,
            )
            logvar = torch.tensor(
                [logvar_value],
                device=config.device,
                dtype=torch.float64,
                requires_grad=True,
            )
            target = torch.tensor([target_value], device=config.device, dtype=torch.float64)
            score = circular_crps_torch(mu, logvar, target, config.fourier_terms)
            gradients = torch.autograd.grad(score.sum(), (mu, logvar))
            reference = score_integral_reference(
                mean_value, max(-10, min(10, logvar_value)), target_value
            )
            reference_gradients = score_reference_gradients(mean_value, logvar_value, target_value)
            gradient_error = max(
                abs(gradients[0].item() - reference_gradients[0]),
                abs(gradients[1].item() - reference_gradients[1]),
            )
            score_error = abs(score.item() - reference)
            interior = -9.999 < logvar_value < 9.999
            for samples in (
                config.circular_train_samples,
                config.circular_eval_samples,
            ):
                approximation = circular_crps_quantile(
                    np.array([mean_value]),
                    np.array([math.exp(logvar_value)]),
                    np.array([target_value]),
                    samples,
                )
                approximation_torch = circular_crps_quantile_torch(mu, logvar, target, samples)
                approximation_gradients = torch.autograd.grad(
                    approximation_torch.sum(), (mu, logvar)
                )
                rows.append(
                    {
                        "image_index": int(pack["indices"][index]),
                        "factor_index": factor,
                        "sigma": math.exp(logvar_value / 2),
                        "samples": samples,
                        "score": score.item(),
                        "reference": reference,
                        "quantile_score": float(approximation[0]),
                        "quantile_d_mu": approximation_gradients[0].item(),
                        "quantile_d_logvar": approximation_gradients[1].item(),
                        "score_difference": float(approximation[0] - reference),
                        "d_mu": gradients[0].item(),
                        "d_logvar": gradients[1].item(),
                        "gradient_error": gradient_error,
                        "score_error": score_error,
                        "score_tolerance": 2e-7,
                        "score_passed": math.isfinite(score_error) and score_error <= 2e-7,
                        "gradient_tolerance": 2e-5,
                        **gradient_comparison(gradient_error, interior),
                    }
                )
    selected_slots = torch.as_tensor(
        slots[: config.image_batch_size], device=config.device
    )
    targets = torch.as_tensor(pack["y"][: len(selected_slots)], device=config.device)
    mu, logvar = predictor(selected_slots)
    first = mixed_crps_torch(mu, logvar, targets, config, "fourier")
    second = mixed_crps_torch(mu, logvar, targets, config, "quantile")
    parameters = tuple(predictor.parameters())
    first_gradients = torch.autograd.grad(first, parameters, retain_graph=True)
    second_gradients = torch.autograd.grad(second, parameters)
    left = torch.cat([value.flatten() for value in first_gradients])
    right = torch.cat([value.flatten() for value in second_gradients])
    parameter_row = {
        "fourier_gradient_norm": left.norm().item(),
        "quantile_gradient_norm": right.norm().item(),
        "relative_gradient_difference": (
            (left - right).norm() / left.norm().clamp_min(1e-15)
        ).item(),
        "gradient_cosine": functional.cosine_similarity(left[None], right[None]).item(),
        "n_images": len(selected_slots),
    }
    return rows, parameter_row
