"""Sequential CPU scoring of slot permutations."""

import numpy as np
from tqdm import tqdm

from probability import crps_matrix
from reporting import predictive_contributions


def score_permutation(mu, variance, target, index, compare_quantiles, config):
    """Compute predictive contributions and requested quantile-node comparisons."""
    mean = mu[:, index].astype(np.float64)
    current_variance = variance[:, index].astype(np.float64)
    target = target.astype(np.float64)
    values, scores, likelihood, diagnostics = predictive_contributions(
        mean, current_variance, target, config
    )
    comparisons = []
    if compare_quantiles:
        for samples in (config.circular_train_samples, config.circular_eval_samples):
            approximate = np.empty_like(scores)
            for start in range(0, len(target), config.metric_batch_size):
                subset = slice(start, start + config.metric_batch_size)
                approximate[subset] = crps_matrix(
                    mean[subset],
                    current_variance[subset],
                    target[subset],
                    samples=samples,
                )
            difference = approximate[:, 7:] - scores[:, 7:]
            comparisons.append(
                {
                    "samples": samples,
                    "mean_circular_difference": float(difference.mean()),
                    "mean_absolute_circular_difference": float(
                        np.abs(difference).mean()
                    ),
                    "p99_absolute_difference": float(
                        np.quantile(np.abs(difference), 0.99)
                    ),
                    "maximum_absolute_difference": float(np.abs(difference).max()),
                }
            )
    return values, scores, likelihood, diagnostics, comparisons


def score_permutations(mu, variance, target, compare_quantiles, config):
    """Score each permutation sequentially in its specified order."""
    results = []
    for index in tqdm(
        range(mu.shape[1]), desc="Permutation scoring", unit="permutation", leave=False
    ):
        result = score_permutation(
            mu, variance, target, index, compare_quantiles, config
        )
        results.append(result)
    return results
