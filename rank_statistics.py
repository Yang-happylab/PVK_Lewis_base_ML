"""Two-sided molecular rank permutation statistics."""
from typing import Any, Sequence
import itertools
import numpy as np
from scipy import stats

EXACT_PERMUTATION_MAX_N = 9
MONTE_CARLO_PERMUTATIONS = 100_000
PERMUTATION_RANDOM_STATE = 42


def rank_correlations_with_permutation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> dict[str, Any]:
    """Calculate rank coefficients and two-sided permutation p-values."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Rank-correlation inputs must be aligned one-dimensional arrays")
    if len(x) < 3:
        raise ValueError("At least three molecules are required for rank correlation")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Rank-correlation inputs must be finite")

    x_ranks = stats.rankdata(x, method="average")
    y_ranks = stats.rankdata(y, method="average")
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    spearman_denominator = float(
        np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    )
    if spearman_denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant variable")
    observed_spearman = float(
        np.sum(x_centered * y_centered) / spearman_denominator
    )

    pair_i, pair_j = np.triu_indices(len(x), k=1)
    x_signs = np.sign(x[pair_i] - x[pair_j])
    y_signs = np.sign(y[pair_i] - y[pair_j])
    kendall_denominator = float(
        np.sqrt(np.sum(x_signs != 0) * np.sum(y_signs != 0))
    )
    if kendall_denominator == 0:
        raise ValueError("Kendall tau-b is undefined for a constant variable")
    observed_kendall = float(np.sum(x_signs * y_signs) / kendall_denominator)

    if len(x) <= EXACT_PERMUTATION_MAX_N:
        permutations = np.asarray(
            list(itertools.permutations(range(len(x)))), dtype=int
        )
        method = "exhaustive exact permutation"
        random_seed: int | None = None
    else:
        rng = np.random.default_rng(PERMUTATION_RANDOM_STATE)
        base = np.arange(len(x), dtype=int)
        permutations = np.asarray(
            [rng.permutation(base) for _ in range(MONTE_CARLO_PERMUTATIONS)],
            dtype=int,
        )
        method = "Monte Carlo permutation with plus-one correction"
        random_seed = PERMUTATION_RANDOM_STATE

    permuted_y_centered = y_ranks[permutations] - np.mean(y_ranks)
    permuted_spearman = (
        permuted_y_centered @ x_centered / spearman_denominator
    )
    permuted_y = y[permutations]
    permuted_y_signs = np.sign(
        permuted_y[:, pair_i] - permuted_y[:, pair_j]
    )
    permuted_kendall = permuted_y_signs @ x_signs / kendall_denominator
    tolerance = 1e-12
    spearman_extreme = int(
        np.sum(np.abs(permuted_spearman) >= abs(observed_spearman) - tolerance)
    )
    kendall_extreme = int(
        np.sum(np.abs(permuted_kendall) >= abs(observed_kendall) - tolerance)
    )
    permutation_count = int(len(permutations))
    if random_seed is None:
        spearman_p = spearman_extreme / permutation_count
        kendall_p = kendall_extreme / permutation_count
    else:
        spearman_p = (spearman_extreme + 1) / (permutation_count + 1)
        kendall_p = (kendall_extreme + 1) / (permutation_count + 1)
    return {
        "Spearman_rho": observed_spearman,
        "Spearman_p_two_sided": float(spearman_p),
        "Kendall_tau_b": observed_kendall,
        "Kendall_p_two_sided": float(kendall_p),
        "P_value_method": method,
        "Permutation_count": permutation_count,
        "Random_seed": random_seed,
    }
