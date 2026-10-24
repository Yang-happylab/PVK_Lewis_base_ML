"""Shared validation, DUPLEX splitting, and plots for structure-model training."""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


def validate_data(X, y, description="data"):
    """Validate data quality."""
    issues = []

    if np.any(np.isnan(X)):
        issues.append(f"{description}: X contains NaN values")
    if np.any(np.isinf(X)):
        issues.append(f"{description}: X contains Inf values")
    if np.any(np.isnan(y)):
        issues.append(f"{description}: y contains NaN values")
    if np.any(np.isinf(y)):
        issues.append(f"{description}: y contains Inf values")

    X_std = np.std(X, axis=0)
    constant_features = np.where(X_std == 0)[0]
    if len(constant_features) > 0:
        issues.append(f"{description}: {len(constant_features)} constant features detected at indices {constant_features}")

    if np.std(y) == 0:
        issues.append(f"{description}: y is constant (no variance)")

    return issues


def stratified_duplex_split(
    X,
    y,
    test_size=0.2,
    n_strata=4,
    metric='mahalanobis',
    return_indices=False,
):
    """
    Stratified DUPLEX data splitting:
    1. Stratify by target variable to ensure distribution consistency.
    2. Apply DUPLEX within each stratum for representative train/test splits.
    3. Avoid data leakage from overly similar samples.
    """
    X_numeric = X.to_numpy(dtype=float)

    try:
        y_bins = pd.qcut(y, q=n_strata, duplicates='drop', labels=False)
        actual_strata = len(np.unique(y_bins))
        logging.info(f"Duplex split using {actual_strata} strata (requested {n_strata})")
    except ValueError as e:
        logging.warning(f"Failed to create {n_strata} bins, using 3: {e}")
        y_bins = pd.qcut(y, q=3, duplicates='drop', labels=False)

    train_idx_list = []
    test_idx_list = []

    for stratum in np.unique(y_bins):
        stratum_mask = (y_bins == stratum)
        stratum_indices = np.where(stratum_mask)[0]
        X_stratum = X_numeric[stratum_indices]
        y_stratum = y.iloc[stratum_indices]

        n_stratum_samples = len(stratum_indices)
        n_stratum_test = max(1, int(n_stratum_samples * test_size))
        n_stratum_train = n_stratum_samples - n_stratum_test

        logging.info(f"Stratum {stratum}: Y range [{y_stratum.min():.4f}, {y_stratum.max():.4f}], "
                    f"n={n_stratum_samples} ({n_stratum_train} train, {n_stratum_test} test)")

        scaler_temp = StandardScaler()
        X_scaled = scaler_temp.fit_transform(X_stratum)

        if metric == 'mahalanobis' and n_stratum_samples > X_stratum.shape[1]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message=(
                            r'The covariance matrix associated to your dataset '
                            r'is not full rank'
                        ),
                        category=UserWarning,
                        module=r'sklearn\.covariance\._robust_covariance',
                    )
                    cov = MinCovDet(support_fraction=1).fit(X_scaled).covariance_
                cov_inv = np.linalg.pinv(cov)
                dist_matrix = cdist(X_scaled, X_scaled, 'mahalanobis', VI=cov_inv)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                logging.warning("Mahalanobis distance failed for stratum %s, using Euclidean: %s",
                                stratum, error)
                dist_matrix = cdist(X_scaled, X_scaled, 'euclidean')
        else:
            dist_matrix = cdist(X_scaled, X_scaled, 'euclidean')

        if n_stratum_samples < 2:
            train_idx_list.extend(stratum_indices)
            logging.warning(f"Stratum {stratum} has only 1 sample, assigned to training set")
            continue

        np.fill_diagonal(dist_matrix, -np.inf)
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

        local_train = [max_dist_idx[0]]
        local_test = [max_dist_idx[1]]
        remaining = list(set(range(n_stratum_samples)) - set(local_train) - set(local_test))

        while len(local_train) < n_stratum_train and len(local_test) < n_stratum_test and remaining:
            if len(local_train) < n_stratum_train and remaining:
                min_dists = np.min(dist_matrix[remaining][:, local_train], axis=1)
                next_train_local = remaining[np.argmax(min_dists)]
                local_train.append(next_train_local)
                remaining.remove(next_train_local)

            if len(local_test) < n_stratum_test and remaining:
                min_dists = np.min(dist_matrix[remaining][:, local_test], axis=1)
                next_test_local = remaining[np.argmax(min_dists)]
                local_test.append(next_test_local)
                remaining.remove(next_test_local)

        local_train.extend(remaining)

        train_idx_list.extend(stratum_indices[local_train])
        test_idx_list.extend(stratum_indices[local_test])

    train_idx = np.array(train_idx_list, dtype=int)
    test_idx = np.array(test_idx_list, dtype=int)
    if not np.array_equal(np.sort(np.concatenate([train_idx, test_idx])),
                          np.arange(len(y))):
        raise ValueError('DUPLEX must assign every row exactly once across train and test')

    logging.info(f"Stratified DUPLEX split completed: {len(train_idx)} train, {len(test_idx)} test")

    split = (
        X_numeric[train_idx],
        X_numeric[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx]
    )
    if return_indices:
        return (*split, train_idx, test_idx)
    return split


def plot_feature_importance(importances, title, save_path, *, dpi=600):
    """Plot feature importance bar chart."""
    plt.figure(figsize=(10, 8), dpi=dpi)
    importances.sort_values(by='Importance', ascending=True).plot(
        kind='barh', x='Feature', y='Importance', legend=False)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(title, fontsize=15, pad=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_target_distribution(y_train, y_test, target_name, save_dir, *, dpi=600,
                             train_color='red', test_color='blue', min_bins=10, max_bins=20):
    """Plot target variable distribution for train and test sets."""
    plt.figure(figsize=(12, 8), dpi=dpi)

    all_y = np.concatenate([y_train, y_test])
    q75, q25 = np.percentile(all_y, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(all_y) ** (1/3))
    n_bins = int((all_y.max() - all_y.min()) / bin_width) if bin_width > 0 else 30
    n_bins = min(max(n_bins, min_bins), max_bins)

    bins = np.linspace(all_y.min(), all_y.max(), n_bins + 1)
    plt.hist(y_train, bins=bins, alpha=0.4, color=train_color,
             label=f'Training Set (n={len(y_train)})', edgecolor='darkred', linewidth=1.5)
    plt.hist(y_test, bins=bins, alpha=0.7, color=test_color,
             label=f'Test Set (n={len(y_test)})', edgecolor='darkblue', linewidth=1.5)
    plt.xlabel(f'{target_name} Value', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='upper left')
    plt.title(f'{target_name} Distribution', fontsize=24, pad=10)
    plt.grid(alpha=0.3, axis='y')

    save_path = os.path.join(save_dir, f'{target_name}_distribution.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
