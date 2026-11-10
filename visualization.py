### Reconstruction, slot stability, and result figures. ###

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from artifacts import atomic_path
from data import make_loader
from diagnostics import hungarian_align_slots
from reproducibility import make_device_generator, make_generator
from training import slots_forward


# Save reconstruction masks and measure initialization and noise sensitivity.
def backbone_visualizations(backbone, dataset, indices, name, config):

    plt.switch_backend("Agg")

    loader = make_loader(
        dataset,
        indices[: config.stability_max_images],
        config.slot_extract_batch_size,
        config,
    )
    first_stream = make_device_generator(config.device, config.evaluation_draw_seeds[0])
    second_stream = make_device_generator(
        config.device, config.evaluation_draw_seeds[-1] + 1
    )
    rows = []
    first_batch = True
    with torch.no_grad():
        for images, target, image_indices in tqdm(
            loader, desc="Backbone visualizations", unit="batch", leave=False
        ):
            noisy_images = []
            for index, image in zip(image_indices, images):
                generator = make_generator(
                    config.corruption_noise_base_seed + int(index)
                )
                noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
                noisy_images.append(
                    torch.clamp(image + config.stability_noise_std * noise, 0, 1)
                )
            images = images.to(config.device)
            noisy_images = torch.stack(noisy_images).to(config.device)
            first = slots_forward(
                backbone,
                images,
                "per_example_stochastic",
                config.evaluation_draw_seeds[0],
                config,
                first_stream,
            )
            second = slots_forward(
                backbone,
                images,
                "per_example_stochastic",
                config.evaluation_draw_seeds[-1] + 1,
                config,
                second_stream,
            )
            reference = slots_forward(
                backbone, images, "fixed_shared", config.slot_init_seed, config
            )
            perturbed = slots_forward(
                backbone, noisy_images, "fixed_shared", config.slot_init_seed, config
            )
            for condition, candidate, anchor in (
                ("initialization", second, first),
                ("gaussian_noise", perturbed, reference),
            ):
                aligned, assignments, costs, margins = hungarian_align_slots(
                    anchor, candidate
                )
                anchor = anchor.cpu()
                distance = (anchor - aligned).norm(dim=-1).mean(-1).numpy()
                denominator = anchor.norm(dim=-1).mean(-1).numpy() + 1e-12
                rows.append(
                    {
                        "backbone": name,
                        "condition": condition,
                        "n_images": len(images),
                        "mean_matched_distance": float(distance.mean()),
                        "mean_relative_matched_change": float(
                            (distance / denominator).mean()
                        ),
                        "mean_cosine_cost": float(costs.mean()),
                    }
                )
            if first_batch:
                reconstruction, masks = backbone.decoder(reference)
                count = min(3, len(images))
                figure, axes = plt.subplots(
                    count, 2 + config.num_slots, figsize=(12, 2 * count), squeeze=False
                )
                for image in range(count):
                    axes[image, 0].imshow(images[image].cpu().permute(1, 2, 0).numpy())
                    axes[image, 1].imshow(
                        reconstruction[image].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    )
                    for slot in range(config.num_slots):
                        axes[image, slot + 2].imshow(
                            masks[image, slot, 0].cpu().numpy(), vmin=0, vmax=1
                        )
                    for axis in axes[image]:
                        axis.axis("off")
                figure.tight_layout()
                path = (
                    config.output_root
                    / "evaluation"
                    / "figures"
                    / f"{name}_reconstructions.png"
                )
                with atomic_path(path) as temporary:
                    figure.savefig(temporary, dpi=150)
                plt.close(figure)
                first_batch = False
    return rows


def result_figures(tables, config):

    plt.switch_backend("Agg")

    data = pd.DataFrame(tables.rows["complete_slot_permutation_table.csv"])
    selected = data[
        (data.group == "primary")
        & (data["variant"] == "raw")
        & (data["mode"] == "fixed_shared")
        & (data.slot_init_seed == config.slot_init_seed)
        & (data.train_seed == config.predictor_seeds[0])
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
    for row, regime in enumerate(config.regimes):
        for column, architecture in enumerate(config.predictor_architectures):
            subset = selected[
                (selected.regime == regime) & (selected.architecture == architecture)
            ]
            axes[row, column].plot(
                np.arange(1, len(subset) + 1), subset.rmse.to_numpy(), marker="."
            )
            axes[row, column].set_title(f"{architecture}, {regime}")
            axes[row, column].set_xlabel("Slot permutation")
            axes[row, column].set_ylabel("RMSE")
    figure.tight_layout()
    with atomic_path(
        config.output_root / "evaluation" / "figures" / "slot_permutation_rmse.png"
    ) as temporary:
        figure.savefig(temporary, dpi=180)
    plt.close(figure)
    data = pd.DataFrame(tables.rows["exact_corruption_level_metrics.csv"])
    data = data[
        (data["variant"] == "raw")
        & (data.readout == "identity")
        & (data.group == "primary")
    ]
    grouped = (
        data.groupby(["train_seed", "corruption", "severity"])[["rmse", "coverage90"]]
        .mean()
        .reset_index()
    )
    summary = grouped.groupby(["corruption", "severity"])[["rmse", "coverage90"]].agg(
        ["mean", "std"]
    )
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
    for column, corruption in enumerate(config.corruption_types):
        subset = summary.loc[corruption]
        for row, metric in enumerate(("coverage90", "rmse")):
            axes[row, column].errorbar(
                subset.index,
                subset[(metric, "mean")],
                yerr=subset[(metric, "std")].fillna(0),
                marker="o",
            )
            axes[row, column].set_xlabel("Severity")
            axes[row, column].set_ylabel(metric)
            axes[row, column].set_title(corruption)
            if metric == "coverage90":
                axes[row, column].axhline(
                    config.interval_level, linestyle="--", color="black"
                )
    figure.tight_layout()
    with atomic_path(
        config.output_root / "evaluation" / "figures" / "corruption_coverage_rmse.png"
    ) as temporary:
        figure.savefig(temporary, dpi=180)
    plt.close(figure)
