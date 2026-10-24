"""
XGBoost regression modeling pipeline.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
import re
import importlib.metadata
import shap
import warnings
from time import perf_counter
from datetime import datetime
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.covariance import MinCovDet
from sklearn.ensemble import VotingRegressor
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from data_preparation import (
    expand_coordination_atom,
    file_sha256,
    load_scored_physicochemical_data,
    write_score_provenance,
)
from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from project_paths import XGB_OUTPUT_DIR

# Basic configuration
RANDOM_STATE = 42
MODEL_SEEDS = (8, 18, 188, 1888, 8888)
BASE_OUTPUT_DIR = XGB_OUTPUT_DIR

# Target variable configuration
ALL_POSSIBLE_TARGETS = ['Binding_energy', 'Transferred_charges', 'Vbur', 'Score']
TARGETS = ['Score']

# Data splitting configuration
TEST_SIZE = 0.2
N_STRATA = 4
DUPLEX_METRIC = 'mahalanobis'

# Feature engineering configuration
CORRELATION_THRESHOLD = 0.85
FEATURE_COUNTS = (4, 6, 8, 10)

# Cross-validation configuration
CV_N_SPLITS = 4
CV_N_REPEATS = 3
CV_N_STRATA = 4

# Hyperparameter search configuration
N_SEARCH_CANDIDATES = 58888
REFINEMENT_REPEATS = 5
RANKING_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                     n_jobs=1)

# Search candidates allocate 60% to focused sampling and 40% to exploration.
FOCUSED_SPACE = {
    'n_estimators': [100, 200, 300, 400, 500, 700],
    'max_depth': [1, 2, 3, 4, 6, 8],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
    'min_child_weight': [1, 2, 3, 4, 6],
    'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
    'reg_alpha': [0, 0.01, 0.05, 0.1, 0.3],
    'reg_lambda': [0.1, 0.5, 1, 2, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.5, 0.6, 0.7, 0.85, 1.0],
    'max_delta_step': [0, 1],
}

EXPLORATION_SPACE = {
    'n_estimators': [50, 100, 200, 300, 400, 500, 800, 1200],
    'max_depth': [1, 2, 3, 4, 6, 8, 10, 12],
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'min_child_weight': [1, 2, 3, 4, 5, 6, 8, 12],
    'gamma': [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
    'reg_alpha': [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5],
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'max_delta_step': [0, 1, 3, 5],
}

# Visualization configuration
PLOT_DPI = 600
PLOT_FONT_FAMILY = 'Arial'
TRAIN_COLOR = 'red'
TEST_COLOR = 'blue'
TRAIN_ALPHA = 0.3
TEST_ALPHA = 0.6
SCATTER_SIZE = 100
FEATURE_SCORE_SCATTER_COLOR = '#2878B5'

# SHAP analysis configuration
SHAP_N_BOOTSTRAP = 30
SHAP_SAMPLE_SIZE = 50
SHAP_N_STRATA = 4

# Histogram configuration
HIST_MIN_BINS = 10
HIST_MAX_BINS = 20

plt.rcParams['font.family'] = PLOT_FONT_FAMILY


def ensemble_metadata():
    return {
        'model_kind': 'xgboost_seed_average',
        'model_seeds': list(MODEL_SEEDS),
        'model_aggregation': 'arithmetic_mean',
    }


def create_xgb_ensemble(params):
    """Fit identical configurations across seeds; average predictions before scoring."""
    member_params = {
        key: value for key, value in params.items()
        if key not in ('random_state', 'n_jobs')
    }
    return VotingRegressor(
        estimators=[
            (f'seed_{seed}', XGBRegressor(
                **member_params, random_state=seed, n_jobs=1,
            ))
            for seed in MODEL_SEEDS
        ],
        n_jobs=1,
    )


def ensemble_shap_values(model, X):
    """Average signed SHAP and base values for the same averaged predictor."""
    member_values = []
    member_expected = []
    for member in model.estimators_:
        explainer = shap.TreeExplainer(
            member, feature_perturbation="tree_path_dependent",
        )
        member_values.append(np.asarray(explainer.shap_values(X), dtype=float))
        member_expected.append(float(np.asarray(explainer.expected_value).reshape(-1)[0]))
    member_values = np.asarray(member_values)
    member_expected = np.asarray(member_expected)
    return (
        member_values.mean(axis=0),
        float(member_expected.mean()),
        member_values,
        member_expected,
    )


# Data splitting module
def stratified_duplex_split(X, y, test_size=TEST_SIZE, n_strata=N_STRATA, 
                           metric=DUPLEX_METRIC):
    """
    Stratified DUPLEX data splitting ensuring target distribution consistency
    and representative train/test sets within each stratum.
    """
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
        
        if isinstance(X, pd.DataFrame):
            X_stratum = X.iloc[stratum_indices].values
        else:
            X_stratum = X[stratum_indices]
            
        if isinstance(y, pd.Series):
            y_stratum = y.iloc[stratum_indices]
        else:
            y_stratum = y[stratum_indices]
        
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
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                logging.warning(f"Mahalanobis distance failed for stratum {stratum}, using Euclidean: {exc}")
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
    
    return (
        X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx],
        X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx],
        y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx],
        y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
    )

# Cross-validation module
def create_cv(X, y, n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, 
             n_strata=CV_N_STRATA, random_state=RANDOM_STATE):
    try:
        y_bins = pd.qcut(y, q=n_strata, duplicates='drop', labels=False)
        logging.info(f"Created {len(np.unique(y_bins))} strata for cross-validation")
    except ValueError as e:
        logging.warning(f"Failed to create {n_strata} bins, trying with 3: {e}")
        y_bins = pd.qcut(y, q=3, duplicates='drop', labels=False)
    
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    return cv.split(X, y_bins)

# Feature preprocessing module
def process_coordination_atom(df):
    return expand_coordination_atom(df)

# Feature selection module
def remove_high_correlation_features(X, threshold=CORRELATION_THRESHOLD):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = set()
    
    high_corr_indices = np.where(upper > threshold)
    for i, j in zip(*high_corr_indices):
        to_drop.add(corr_matrix.columns[i])
    
    return X.drop(columns=to_drop), list(to_drop)


# Visualization module
def plot_results(y_train, y_pred_train, y_test, y_pred_test, title, test_r2, save_path):
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    
    plt.scatter(y_train, y_pred_train, alpha=TRAIN_ALPHA, color=TRAIN_COLOR, 
               s=SCATTER_SIZE, label='Training Set')
    plt.scatter(y_test, y_pred_test, alpha=TEST_ALPHA, color=TEST_COLOR, 
               s=SCATTER_SIZE, label='Test Set')
    
    all_actual = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val = min(all_actual.min(), all_pred.min())
    max_val = max(all_actual.max(), all_pred.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', 
            linewidth=3, label='Ideal')
    
    plt.xlabel('Calculated Value', fontsize=24)
    plt.ylabel('Predicted Value', fontsize=24)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='upper left')
    
    plt.text(0.95, 0.05, f'Test R² = {test_r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=20, 
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title(title, fontsize=24, pad=10)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_residuals(y_train, y_pred_train, y_test, y_pred_test, title, save_path):
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    
    plt.scatter(y_pred_train, residuals_train, alpha=TRAIN_ALPHA, color=TRAIN_COLOR, 
               s=SCATTER_SIZE, label='Training Set')
    plt.scatter(y_pred_test, residuals_test, alpha=TEST_ALPHA, color=TEST_COLOR, 
               s=SCATTER_SIZE, label='Test Set')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
    
    plt.xlabel('Predicted Value', fontsize=20)
    plt.ylabel('Residual', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=16, loc='best')
    plt.title(title, fontsize=24, pad=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_target_distribution(y_train, y_test, target_name, save_path):
    plt.figure(figsize=(12, 8), dpi=PLOT_DPI)
    
    all_y = np.concatenate([y_train, y_test])
    q75, q25 = np.percentile(all_y, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(all_y) ** (1/3))
    n_bins = int((all_y.max() - all_y.min()) / bin_width) if bin_width > 0 else 30
    n_bins = min(max(n_bins, HIST_MIN_BINS), HIST_MAX_BINS)
    
    bins = np.linspace(all_y.min(), all_y.max(), n_bins + 1)
    
    plt.hist(y_train, bins=bins, alpha=0.4, color=TRAIN_COLOR, 
             label=f'Training Set (n={len(y_train)})', edgecolor='darkred', linewidth=1.5)
    plt.hist(y_test, bins=bins, alpha=0.7, color=TEST_COLOR, 
             label=f'Test Set (n={len(y_test)})', edgecolor='darkblue', linewidth=1.5)
    
    plt.xlabel(f'{target_name} Value', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='upper left')
    plt.title(f'{target_name} Distribution', fontsize=24, pad=10)
    plt.grid(alpha=0.3, axis='y')
    
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importances, title, save_path):
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    importances.sort_values(by='Importance', ascending=True).plot(
        kind='barh', x='Feature', y='Importance', legend=False)
    plt.xlabel('Training mean(|SHAP|)', fontsize=16)
    plt.ylabel('Feature', fontsize=16)
    plt.title(title, fontsize=16, pad=10)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI)
    plt.close()


def _feature_filename(value):
    """Return a portable filename component while preserving readable names."""
    filename = re.sub(r'[^\w.-]+', '_', str(value), flags=re.UNICODE).strip('_.')
    return filename or 'feature'


def plot_top_feature_score_scatters(
    X,
    y,
    importances,
    target_name,
    output_dir,
    n_features=None,
):
    """Save one feature-versus-target scatter plot for each top-ranked feature."""
    if n_features is None:
        n_features = len(importances)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for named feature plots")
    required_importance_columns = {'Feature', 'Importance'}
    missing_columns = required_importance_columns.difference(importances.columns)
    if missing_columns:
        raise ValueError(
            "Feature importance table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    ranked = (
        importances.loc[:, ['Feature', 'Importance']]
        .sort_values('Importance', ascending=False, kind='stable')
        .head(n_features)
        .reset_index(drop=True)
    )
    if len(ranked) != n_features:
        raise ValueError(
            f"Expected {n_features} ranked features, found {len(ranked)}"
        )

    missing_features = sorted(set(ranked['Feature']).difference(X.columns))
    if missing_features:
        raise ValueError(
            "Top-ranked features are missing from the plotting data: "
            + ", ".join(missing_features)
        )

    if isinstance(y, pd.Series):
        target_values = y.reindex(X.index)
    else:
        target_values = pd.Series(np.asarray(y), index=X.index, name=target_name)
    if target_values.isna().any():
        raise ValueError("Target values could not be aligned with the feature rows")

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for rank, row in ranked.iterrows():
        feature = row['Feature']
        importance = float(row['Importance'])
        feature_values = pd.to_numeric(X[feature], errors='raise')

        figure, axis = plt.subplots(figsize=(6.2, 4.6), dpi=PLOT_DPI)
        axis.scatter(
            feature_values,
            target_values,
            s=48,
            color=FEATURE_SCORE_SCATTER_COLOR,
            alpha=0.68,
            edgecolors='white',
            linewidths=0.4,
        )
        axis.set_xlabel(str(feature), fontsize=16)
        axis.set_ylabel(str(target_name), fontsize=16)
        axis.set_title(
            f'{target_name} vs {feature}',
            fontsize=17,
            pad=10,
        )
        axis.text(
            0.03,
            0.96,
            f'SHAP rank: {rank + 1}\nTraining mean(|SHAP|): {importance:.4f}',
            transform=axis.transAxes,
            ha='left',
            va='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.82, edgecolor='0.75'),
        )
        axis.tick_params(axis='both', labelsize=12)
        axis.grid(alpha=0.18)
        figure.tight_layout()

        filename = (
            f'{rank + 1:02d}_{_feature_filename(feature)}_vs_'
            f'{_feature_filename(target_name)}.png'
        )
        output_path = os.path.join(output_dir, filename)
        figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(figure)
        output_paths.append(output_path)
        logging.info(
            f"Feature-Score scatter saved (rank {rank + 1}, {feature}): {output_path}"
        )

    return output_paths

# SHAP analysis module
def shap_analysis(model, X_train, X_test, y_train, y_test, feature_names, output_dir, round_name):
    """
    SHAP analysis with bootstrap computation, deduplicated visualization,
    and dynamic figure sizing.
    """
    original_rcParams = plt.rcParams.copy()
    try:

        if isinstance(X_train, pd.DataFrame):
            X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        else:
            X_full = np.vstack([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
        
        logging.info(f"SHAP analysis: using full dataset (n={len(X_full)}), "
                    f"performing {SHAP_N_BOOTSTRAP} stratified samplings")
        
        try:
            y_bins = pd.qcut(y_full, q=SHAP_N_STRATA, duplicates='drop', labels=False)
        except ValueError as e:
            logging.warning(f"Failed to create {SHAP_N_STRATA} bins, using 3: {e}")
            y_bins = pd.qcut(y_full, q=3, duplicates='drop', labels=False)
        
        full_shap_values, _, _, _ = ensemble_shap_values(model, X_full)
        
        all_shap_values = []
        bootstrap_indices_list = []
        
        for bootstrap_idx in range(SHAP_N_BOOTSTRAP):
            sampled_indices = []
            for stratum in np.unique(y_bins):
                stratum_indices = np.where(y_bins == stratum)[0]
                n_stratum_samples = max(1, int(len(stratum_indices) * SHAP_SAMPLE_SIZE / len(X_full)))
                sampled = np.random.choice(stratum_indices, size=n_stratum_samples, replace=True)
                sampled_indices.extend(sampled)
            
            if len(sampled_indices) > SHAP_SAMPLE_SIZE:
                sampled_indices = np.random.choice(sampled_indices, size=SHAP_SAMPLE_SIZE, replace=False)
            
            bootstrap_indices_list.append(sampled_indices)
            
            shap_values_sample = full_shap_values[sampled_indices]
            all_shap_values.append(shap_values_sample)
            
            if (bootstrap_idx + 1) % 10 == 0:
                logging.info(f"Completed {bootstrap_idx + 1}/{SHAP_N_BOOTSTRAP} SHAP samplings")
        
        sample_shap_dict = {}
        
        for boot_idx, indices in enumerate(bootstrap_indices_list):
            for i, sample_idx in enumerate(indices):
                if sample_idx not in sample_shap_dict:
                    sample_shap_dict[sample_idx] = []
                sample_shap_dict[sample_idx].append(all_shap_values[boot_idx][i])
        
        unique_indices = sorted(sample_shap_dict.keys())
        unique_shap_values = []
        
        for idx in unique_indices:
            shap_list = sample_shap_dict[idx]
            avg_shap = np.mean(shap_list, axis=0)
            unique_shap_values.append(avg_shap)
        
        unique_shap_values = np.array(unique_shap_values)
        
        if isinstance(X_full, pd.DataFrame):
            unique_samples = X_full.iloc[unique_indices].values
        else:
            unique_samples = X_full[unique_indices]
        
        logging.info(f"Unique samples for plotting after deduplication: {len(unique_indices)} (original total: {len(X_full)})")
        logging.info(f"Average sampling times per sample: {len(bootstrap_indices_list) * SHAP_SAMPLE_SIZE / len(unique_indices):.1f}")
        
        all_shap_values = np.array(all_shap_values)
        
        shap_stats = pd.DataFrame({
            'Feature': feature_names,
            'Mean_|SHAP|': np.mean(np.abs(all_shap_values), axis=(0, 1)),
            'Std_|SHAP|': np.std(np.abs(all_shap_values), axis=(0, 1)),
            'Min_SHAP': np.min(all_shap_values, axis=(0, 1)),
            'Max_SHAP': np.max(all_shap_values, axis=(0, 1))
        }).sort_values('Mean_|SHAP|', ascending=False)
        
        n_features = len(feature_names)
        fig_width = 10
        fig_height = 8
        
        logging.info(f"SHAP summary plot size: {fig_width}x{fig_height} (n_features={n_features})")
        
        plt.figure(figsize=(fig_width, fig_height), dpi=PLOT_DPI)
        shap.summary_plot(unique_shap_values, unique_samples, 
                         feature_names=feature_names, show=False)
        
        ax = plt.gca()
        ax.set_xlabel('SHAP value', fontsize=22)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        
        if hasattr(ax, 'collections') and len(ax.collections) > 0:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label(cbar.ax.get_ylabel(), fontsize=14)
        
        plt.title(f'SHAP Summary - {round_name}\n'
                 f'(n_unique={len(unique_indices)}, n_bootstrap={SHAP_N_BOOTSTRAP})', 
                 fontsize=18, pad=15)
        
        plt.subplots_adjust(left=0.25, right=0.90, top=0.90, bottom=0.12)
        
        plt.savefig(os.path.join(output_dir, f'shap_summary_{round_name}.png'), 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        fig_width_bar = 10
        fig_height_bar = 8
        
        logging.info(f"SHAP importance plot size: {fig_width_bar}x{fig_height_bar} (n_features={n_features})")
        
        plt.figure(figsize=(fig_width_bar, fig_height_bar), dpi=PLOT_DPI)
        shap.summary_plot(unique_shap_values, unique_samples, 
                         feature_names=feature_names, plot_type="bar", show=False)
        
        ax = plt.gca()
        ax.set_xlabel('mean(|SHAP value|)', fontsize=22)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        
        plt.title(f'SHAP Feature Importance - {round_name}\n'
                 f'(n_unique={len(unique_indices)}, n_bootstrap={SHAP_N_BOOTSTRAP})', 
                 fontsize=18, pad=15)
        
        plt.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.12)
        
        plt.savefig(os.path.join(output_dir, f'shap_importance_{round_name}.png'), 
                   dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        shap_stats.to_csv(os.path.join(output_dir, f'shap_statistics_{round_name}.csv'), 
                         index=False)
        
        sample_frequency = pd.DataFrame({
            'Sample_Index': unique_indices,
            'Times_Sampled': [len(sample_shap_dict[idx]) for idx in unique_indices]
        }).sort_values('Times_Sampled', ascending=False)
        sample_frequency.to_csv(os.path.join(output_dir, f'shap_sample_frequency_{round_name}.csv'), 
                               index=False)
        
        logging.info(f"SHAP analysis completed - unique samples: {len(unique_indices)}, "
                    f"total samplings: {sum(sample_frequency['Times_Sampled'])}")
        
    finally:
        plt.rcParams.update(original_rcParams)


def cross_validated_oof_shap(
    X_train,
    y_train,
    sample_names,
    feature_names,
    best_params,
    output_dir,
    round_name,
    score_version,
    campaign_round,
):
    """Create cross-fitted OOF SHAP artifacts on the outer training set only."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("OOF SHAP requires a pandas DataFrame with named features")

    feature_names = [str(feature) for feature in feature_names]
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("OOF SHAP feature names must be nonempty and unique")
    missing_features = sorted(set(feature_names).difference(X_train.columns))
    if missing_features:
        raise ValueError(f"OOF SHAP features are missing: {missing_features}")

    X_frame = X_train.loc[:, feature_names].reset_index(drop=True).copy()
    y_series = pd.Series(y_train).reset_index(drop=True).astype(float)
    name_series = pd.Series(sample_names).reset_index(drop=True).astype(str)
    n_samples = len(X_frame)
    n_features = len(feature_names)
    if len(y_series) != n_samples or len(name_series) != n_samples:
        raise ValueError("OOF SHAP features, targets, and names are not aligned")
    if name_series.str.strip().eq('').any() or name_series.duplicated().any():
        raise ValueError("OOF SHAP sample names must be nonblank and unique")
    if not np.isfinite(X_frame.to_numpy(dtype=float)).all():
        raise ValueError("OOF SHAP features contain non-finite values")
    if not np.isfinite(y_series.to_numpy(dtype=float)).all():
        raise ValueError("OOF SHAP target contains non-finite values")

    expected_fold_count = CV_N_SPLITS * CV_N_REPEATS
    cv_splits = list(create_cv(X_frame, y_series))
    if len(cv_splits) != expected_fold_count:
        raise ValueError(
            f"Expected {expected_fold_count} OOF folds, found {len(cv_splits)}"
        )

    os.makedirs(output_dir, exist_ok=True)
    shap_by_repeat = np.full(
        (CV_N_REPEATS, n_samples, n_features),
        np.nan,
        dtype=float,
    )
    prediction_by_repeat = np.full(
        (CV_N_REPEATS, n_samples),
        np.nan,
        dtype=float,
    )
    expected_value_by_repeat = np.full(
        (CV_N_REPEATS, n_samples),
        np.nan,
        dtype=float,
    )
    fold_by_repeat = np.full(
        (CV_N_REPEATS, n_samples),
        -1,
        dtype=int,
    )
    fold_metrics = []
    fold_importance_rows = []
    seed_rows = []
    seed_shap_by_repeat = np.full(
        (len(MODEL_SEEDS), CV_N_REPEATS, n_samples, n_features), np.nan,
    )

    model_params = dict(best_params)
    X_values = X_frame.to_numpy(dtype=float)
    y_values = y_series.to_numpy(dtype=float)

    logging.info(
        "Starting fixed-feature OOF SHAP: %d samples, %d features, %d folds",
        n_samples,
        n_features,
        expected_fold_count,
    )
    for split_number, (fit_indices, validation_indices) in enumerate(cv_splits, 1):
        repeat_number = (split_number - 1) // CV_N_SPLITS + 1
        fold_number = (split_number - 1) % CV_N_SPLITS + 1
        repeat_index = repeat_number - 1

        if np.isfinite(
            prediction_by_repeat[repeat_index, validation_indices]
        ).any():
            raise ValueError(
                f"OOF validation rows overlap within repeat {repeat_number}"
            )

        fold_model = create_xgb_ensemble(model_params)
        fold_model.fit(X_frame.iloc[fit_indices], y_values[fit_indices])
        validation_values = X_frame.iloc[validation_indices]
        validation_target = y_values[validation_indices]
        validation_prediction = fold_model.predict(validation_values)

        fold_shap, expected_value, seed_shap, seed_expected = ensemble_shap_values(
            fold_model, validation_values,
        )
        if fold_shap.shape != (len(validation_indices), n_features):
            raise ValueError(
                f"Unexpected OOF SHAP shape for split {split_number}: "
                f"{fold_shap.shape}"
            )
        if not np.isfinite(fold_shap).all():
            raise ValueError(f"OOF SHAP split {split_number} contains non-finite values")

        reconstructed = expected_value + fold_shap.sum(axis=1)
        max_additivity_error = float(
            np.max(np.abs(reconstructed - validation_prediction))
        )
        additivity_tolerance = 1e-4 + 1e-4 * float(
            np.max(np.abs(validation_prediction))
        )
        if max_additivity_error > additivity_tolerance:
            raise ValueError(
                f"OOF SHAP additivity check failed for split {split_number}: "
                f"{max_additivity_error:.6g}"
            )

        shap_by_repeat[repeat_index, validation_indices, :] = fold_shap
        prediction_by_repeat[repeat_index, validation_indices] = validation_prediction
        expected_value_by_repeat[repeat_index, validation_indices] = expected_value
        fold_by_repeat[repeat_index, validation_indices] = fold_number
        for seed_index, seed in enumerate(MODEL_SEEDS):
            seed_shap_by_repeat[seed_index, repeat_index, validation_indices] = seed_shap[seed_index]
            member_prediction = fold_model.estimators_[seed_index].predict(validation_values)
            for local_index, sample_index in enumerate(validation_indices):
                row = {
                    'Model_Seed': seed,
                    'Name': name_series.iloc[sample_index],
                    'Sample_Index': int(sample_index),
                    'Repeat': repeat_number,
                    'Fold': fold_number,
                    'Observed_Score': float(y_values[sample_index]),
                    'Predicted_Score': float(member_prediction[local_index]),
                    'Expected_Value': float(seed_expected[seed_index]),
                }
                for feature_index, feature in enumerate(feature_names):
                    row[f'Value_{feature}'] = float(X_values[sample_index, feature_index])
                    row[f'SHAP_{feature}'] = float(seed_shap[seed_index, local_index, feature_index])
                seed_rows.append(row)

        mean_abs_shap = np.mean(np.abs(fold_shap), axis=0)
        rank_order = np.argsort(-mean_abs_shap, kind='stable')
        ranks = np.empty(n_features, dtype=int)
        ranks[rank_order] = np.arange(1, n_features + 1)
        fold_metrics.append({
            'Split': split_number,
            'Repeat': repeat_number,
            'Fold': fold_number,
            'Train_Size': len(fit_indices),
            'Validation_Size': len(validation_indices),
            'R2': float(r2_score(validation_target, validation_prediction)),
            'MSE': float(mean_squared_error(validation_target, validation_prediction)),
            'MAE': float(mean_absolute_error(validation_target, validation_prediction)),
            'Max_SHAP_Additivity_Error': max_additivity_error,
        })
        for feature_index, feature in enumerate(feature_names):
            fold_importance_rows.append({
                'Split': split_number,
                'Repeat': repeat_number,
                'Fold': fold_number,
                'Feature': feature,
                'Mean_Abs_SHAP': float(mean_abs_shap[feature_index]),
                'Rank': int(ranks[feature_index]),
            })

    if (
        np.isnan(shap_by_repeat).any()
        or np.isnan(prediction_by_repeat).any()
        or np.isnan(expected_value_by_repeat).any()
        or (fold_by_repeat < 1).any()
    ):
        raise ValueError("OOF SHAP did not cover every training sample in every repeat")

    raw_rows = []
    for repeat_index in range(CV_N_REPEATS):
        for sample_index in range(n_samples):
            row = {
                'Name': name_series.iloc[sample_index],
                'Sample_Index': sample_index,
                'Repeat': repeat_index + 1,
                'Fold': int(fold_by_repeat[repeat_index, sample_index]),
                'Observed_Score': float(y_values[sample_index]),
                'Predicted_Score': float(
                    prediction_by_repeat[repeat_index, sample_index]
                ),
                'Residual': float(
                    y_values[sample_index]
                    - prediction_by_repeat[repeat_index, sample_index]
                ),
                'Expected_Value': float(
                    expected_value_by_repeat[repeat_index, sample_index]
                ),
            }
            for feature_index, feature in enumerate(feature_names):
                row[f'Value_{feature}'] = float(
                    X_values[sample_index, feature_index]
                )
                row[f'SHAP_{feature}'] = float(
                    shap_by_repeat[repeat_index, sample_index, feature_index]
                )
            raw_rows.append(row)
    raw_oof = pd.DataFrame(raw_rows).sort_values(
        ['Sample_Index', 'Repeat'],
        kind='stable',
    )

    consensus_shap = np.mean(shap_by_repeat, axis=0)
    consensus_shap_sd = np.std(shap_by_repeat, axis=0, ddof=0)
    consensus_prediction = np.mean(prediction_by_repeat, axis=0)
    consensus_prediction_sd = np.std(prediction_by_repeat, axis=0, ddof=0)
    consensus_expected_value = np.mean(expected_value_by_repeat, axis=0)
    consensus_reconstructed = consensus_expected_value + consensus_shap.sum(axis=1)
    max_consensus_additivity_error = float(
        np.max(np.abs(consensus_reconstructed - consensus_prediction))
    )
    if max_consensus_additivity_error > 1e-4:
        raise ValueError(
            "Consensus OOF SHAP additivity check failed: "
            f"{max_consensus_additivity_error:.6g}"
        )
    consensus_rows = []
    for sample_index in range(n_samples):
        row = {
            'Name': name_series.iloc[sample_index],
            'Sample_Index': sample_index,
            'Observed_Score': float(y_values[sample_index]),
            'Mean_OOF_Predicted_Score': float(consensus_prediction[sample_index]),
            'SD_OOF_Predicted_Score': float(consensus_prediction_sd[sample_index]),
            'Mean_OOF_Residual': float(
                y_values[sample_index] - consensus_prediction[sample_index]
            ),
            'Mean_Expected_Value': float(consensus_expected_value[sample_index]),
        }
        for feature_index, feature in enumerate(feature_names):
            row[f'Value_{feature}'] = float(X_values[sample_index, feature_index])
            row[f'Mean_SHAP_{feature}'] = float(
                consensus_shap[sample_index, feature_index]
            )
            row[f'SD_SHAP_{feature}'] = float(
                consensus_shap_sd[sample_index, feature_index]
            )
        consensus_rows.append(row)
    consensus = pd.DataFrame(consensus_rows)

    seed_importance_rows = []
    for seed_index, seed in enumerate(MODEL_SEEDS):
        seed_consensus = seed_shap_by_repeat[seed_index].mean(axis=0)
        importance = np.abs(seed_consensus).mean(axis=0)
        seed_ranks = np.empty(n_features, dtype=int)
        seed_ranks[np.argsort(-importance, kind='stable')] = np.arange(1, n_features + 1)
        for feature_index, feature in enumerate(feature_names):
            values = pd.Series(X_values[:, feature_index])
            contributions = pd.Series(seed_consensus[:, feature_index])
            direction = (
                values.corr(contributions, method='spearman')
                if values.nunique() > 1 and contributions.nunique() > 1 else np.nan
            )
            seed_importance_rows.append({
                'Model_Seed': seed,
                'Feature': feature,
                'MeanAbs_SHAP': float(importance[feature_index]),
                'Rank': int(seed_ranks[feature_index]),
                'Direction_Spearman': float(direction),
            })
    seed_importance = pd.DataFrame(seed_importance_rows)
    fold_importance = pd.DataFrame(fold_importance_rows)
    feature_summary_rows = []
    for feature_index, feature in enumerate(feature_names):
        feature_folds = fold_importance.loc[
            fold_importance['Feature'].eq(feature)
        ]
        fold_values = feature_folds['Mean_Abs_SHAP'].to_numpy(dtype=float)
        fold_ranks = feature_folds['Rank'].to_numpy(dtype=int)
        seed_ranks = seed_importance.loc[
            seed_importance['Feature'].eq(feature), 'Rank'
        ].to_numpy(dtype=int)
        feature_summary_rows.append({
            'Feature': feature,
            'Consensus_Mean_Abs_SHAP': float(
                np.mean(np.abs(consensus_shap[:, feature_index]))
            ),
            'Raw_OOF_Mean_Abs_SHAP': float(
                np.mean(np.abs(shap_by_repeat[:, :, feature_index]))
            ),
            'Mean_Within_Sample_SHAP_SD': float(
                np.mean(consensus_shap_sd[:, feature_index])
            ),
            'Fold_Mean_Abs_SHAP_Mean': float(np.mean(fold_values)),
            'Fold_Mean_Abs_SHAP_SD': float(np.std(fold_values, ddof=0)),
            'Fold_Mean_Abs_SHAP_IQR': float(
                np.percentile(fold_values, 75) - np.percentile(fold_values, 25)
            ),
            'Median_Rank': float(np.median(fold_ranks)),
            'Min_Rank': int(np.min(fold_ranks)),
            'Max_Rank': int(np.max(fold_ranks)),
            'Top1_Count': int(np.sum(fold_ranks == 1)),
            'Top3_Count': int(np.sum(fold_ranks <= min(3, n_features))),
            'Top3_Frequency': float(
                np.mean(fold_ranks <= min(3, n_features))
            ),
            'Seed_Median_Rank': float(np.median(seed_ranks)),
            'Seed_Min_Rank': int(seed_ranks.min()),
            'Seed_Max_Rank': int(seed_ranks.max()),
            'Seed_Top3_Frequency': float(np.mean(seed_ranks <= min(3, n_features))),
        })
    feature_summary = pd.DataFrame(feature_summary_rows).sort_values(
        'Consensus_Mean_Abs_SHAP',
        ascending=False,
        kind='stable',
    ).reset_index(drop=True)
    feature_summary.insert(
        0,
        'Consensus_Rank',
        np.arange(1, len(feature_summary) + 1),
    )

    artifact_suffix = _feature_filename(round_name)
    artifact_paths = {
        'raw_values': os.path.join(
            output_dir, f'oof_shap_values_{artifact_suffix}.csv'
        ),
        'consensus_values': os.path.join(
            output_dir, f'oof_shap_consensus_{artifact_suffix}.csv'
        ),
        'feature_stability': os.path.join(
            output_dir, f'oof_shap_feature_stability_{artifact_suffix}.csv'
        ),
        'fold_importance': os.path.join(
            output_dir, f'oof_shap_fold_importance_{artifact_suffix}.csv'
        ),
        'fold_metrics': os.path.join(
            output_dir, f'oof_shap_fold_metrics_{artifact_suffix}.csv'
        ),
        'seed_values': os.path.join(
            output_dir, f'oof_shap_seed_values_{artifact_suffix}.csv'
        ),
        'seed_importance': os.path.join(
            output_dir, f'oof_shap_seed_importance_{artifact_suffix}.csv'
        ),
        'summary_plot': os.path.join(
            output_dir, f'oof_shap_summary_{artifact_suffix}.png'
        ),
        'importance_plot': os.path.join(
            output_dir, f'oof_shap_importance_{artifact_suffix}.png'
        ),
        'manifest': os.path.join(
            output_dir, f'oof_shap_manifest_{artifact_suffix}.json'
        ),
    }
    raw_oof.to_csv(artifact_paths['raw_values'], index=False)
    consensus.to_csv(artifact_paths['consensus_values'], index=False)
    feature_summary.to_csv(artifact_paths['feature_stability'], index=False)
    fold_importance.to_csv(artifact_paths['fold_importance'], index=False)
    fold_metrics_frame = pd.DataFrame(fold_metrics)
    fold_metrics_frame.to_csv(artifact_paths['fold_metrics'], index=False)
    pd.DataFrame(seed_rows).sort_values(
        ['Model_Seed', 'Sample_Index', 'Repeat'], kind='stable',
    ).to_csv(artifact_paths['seed_values'], index=False)
    seed_importance.to_csv(artifact_paths['seed_importance'], index=False)

    original_rc_params = plt.rcParams.copy()
    try:
        ordered_features = feature_summary['Feature'].tolist()
        ordered_indices = [feature_names.index(feature) for feature in ordered_features]
        ordered_shap = consensus_shap[:, ordered_indices]
        ordered_values = X_frame.loc[:, ordered_features].to_numpy(dtype=float)

        plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
        shap.summary_plot(
            ordered_shap,
            ordered_values,
            feature_names=ordered_features,
            show=False,
            sort=True,
            max_display=len(ordered_features),
        )
        axis = plt.gca()
        axis.set_xlabel('Consensus OOF SHAP value', fontsize=22)
        axis.tick_params(axis='both', labelsize=18)
        plt.title(
            'Cross-fitted OOF SHAP Summary\n'
            f'(outer training set only; {CV_N_SPLITS} folds x '
            f'{CV_N_REPEATS} repeats; n={n_samples})',
            fontsize=18,
            pad=15,
        )
        plt.subplots_adjust(left=0.25, right=0.90, top=0.88, bottom=0.12)
        plt.savefig(
            artifact_paths['summary_plot'],
            dpi=PLOT_DPI,
            bbox_inches='tight',
        )
        plt.close()

        plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
        shap.summary_plot(
            ordered_shap,
            ordered_values,
            feature_names=ordered_features,
            plot_type='bar',
            show=False,
            sort=True,
            max_display=len(ordered_features),
        )
        axis = plt.gca()
        axis.set_xlabel('Mean |consensus OOF SHAP value|', fontsize=22)
        axis.tick_params(axis='both', labelsize=18)
        plt.title(
            'Cross-fitted OOF SHAP Feature Importance\n'
            f'(outer training set only; {CV_N_SPLITS} folds x '
            f'{CV_N_REPEATS} repeats; n={n_samples})',
            fontsize=18,
            pad=15,
        )
        plt.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.12)
        plt.savefig(
            artifact_paths['importance_plot'],
            dpi=PLOT_DPI,
            bbox_inches='tight',
        )
        plt.close()
    finally:
        plt.rcParams.update(original_rc_params)

    metadata = {
        **ensemble_metadata(),
        'analysis': 'cross-fitted-oof-shap',
        'model_stage': str(round_name),
        'campaign_round': str(campaign_round),
        'score_version': str(score_version),
        'outer_split': 'single stratified DUPLEX split',
        'outer_training_samples': n_samples,
        'outer_test_used_for_oof_shap': False,
        'feature_selection': (
            'fixed selected-feature set chosen once on the outer training set; '
            'no per-fold feature selection'
        ),
        'hyperparameters': (
            'fixed selected-feature-stage best parameters; no per-fold search'
        ),
        'best_params': model_params,
        'cv': {
            'method': 'RepeatedStratifiedKFold',
            'n_splits': CV_N_SPLITS,
            'n_repeats': CV_N_REPEATS,
            'n_strata': CV_N_STRATA,
            'random_state': RANDOM_STATE,
            'fold_count': expected_fold_count,
            'member_fit_count': expected_fold_count * len(MODEL_SEEDS),
        },
        'explainer': 'TreeExplainer(tree_path_dependent)',
        'sample_aggregation': (
            'mean signed member SHAP and base values across model seeds, '
            'then mean across repeats for each unique Name'
        ),
        'seed_stability': (
            'per-seed importance and feature-value/SHAP Spearman direction '
            'computed after averaging repeats by Name; descriptive diagnostics only'
        ),
        'global_importance': (
            'mean absolute consensus OOF SHAP across unique outer-training samples'
        ),
        'feature_order': feature_summary['Feature'].tolist(),
        'diagnostic_consensus_oof_metrics': {
            'r2': float(r2_score(y_values, consensus_prediction)),
            'mse': float(mean_squared_error(y_values, consensus_prediction)),
            'mae': float(mean_absolute_error(y_values, consensus_prediction)),
        },
        'performance_role': (
            'diagnostic only; formal performance is reported on the outer '
            'stratified DUPLEX test set'
        ),
        'max_consensus_shap_additivity_error': max_consensus_additivity_error,
        'artifacts': {
            key: os.path.basename(path)
            for key, path in artifact_paths.items()
            if key != 'manifest'
        },
    }
    with open(artifact_paths['manifest'], 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    logging.info(
        "OOF SHAP completed; consensus ranking: %s",
        " > ".join(feature_summary['Feature']),
    )
    return {
        'output_dir': output_dir,
        'feature_summary': feature_summary.to_dict(orient='records'),
        'fold_r2_mean': float(fold_metrics_frame['R2'].mean()),
        'fold_r2_std': float(fold_metrics_frame['R2'].std(ddof=0)),
        'outer_training_samples': n_samples,
        'score_version': str(score_version),
        'artifact_paths': artifact_paths,
    }

def bind_oof_prediction_interval_calibration(
    model_directory,
    oof_shap_result,
    campaign_round,
    score_version,
):
    """Bind unique-sample consensus OOF residuals to the selected model contract."""
    model_directory = Path(model_directory).resolve()
    contract_path = model_directory / 'model_contract_selected_features.json'
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Selected-feature model contract not found: {contract_path}"
        )
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if contract.get('model_stage') != 'selected_features':
        raise ValueError('OOF interval calibration requires the selected-feature model')
    if str(contract.get('campaign_round')) != str(campaign_round):
        raise ValueError('OOF interval calibration campaign does not match model')

    artifact_paths = oof_shap_result.get('artifact_paths', {})
    calibration_path = Path(artifact_paths.get('consensus_values', '')).resolve()
    oof_manifest_path = Path(artifact_paths.get('manifest', '')).resolve()
    for path, label in (
        (calibration_path, 'consensus OOF calibration'),
        (oof_manifest_path, 'OOF manifest'),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        try:
            path.relative_to(model_directory)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the model directory") from exc

    oof_manifest = json.loads(oof_manifest_path.read_text(encoding='utf-8'))
    if oof_manifest.get('analysis') != 'cross-fitted-oof-shap':
        raise ValueError('OOF manifest does not describe cross-fitted OOF SHAP')
    if oof_manifest.get('model_stage') != 'selected_features':
        raise ValueError('OOF manifest model stage does not match selected model')
    if str(oof_manifest.get('campaign_round')) != str(campaign_round):
        raise ValueError('OOF manifest campaign does not match selected model')
    if str(oof_manifest.get('score_version')) != str(score_version):
        raise ValueError('OOF manifest Score version does not match selected model')
    if oof_manifest.get('outer_test_used_for_oof_shap') is not False:
        raise ValueError('OOF interval calibration must exclude the outer test set')
    procedure = {}
    if contract.get('schema_version', 0) >= 3:
        procedure = ensemble_metadata()
        if any(
            contract.get(key) != value or oof_manifest.get(key) != value
            for key, value in procedure.items()
        ):
            raise ValueError('Model and OOF ensemble procedures do not match')

    calibration_frame = pd.read_csv(calibration_path, encoding='utf-8')
    required_columns = {'Name', 'Mean_OOF_Residual'}
    if not required_columns.issubset(calibration_frame.columns):
        raise ValueError('Consensus OOF calibration columns are incomplete')
    if calibration_frame['Name'].astype(str).duplicated().any():
        raise ValueError('Consensus OOF calibration contains duplicate Names')
    residuals = pd.to_numeric(
        calibration_frame['Mean_OOF_Residual'], errors='raise'
    ).to_numpy(dtype=float)
    if not np.isfinite(residuals).all():
        raise ValueError('Consensus OOF residuals must be finite')
    expected_samples = int(oof_shap_result['outer_training_samples'])
    if len(calibration_frame) != expected_samples:
        raise ValueError('Consensus OOF calibration sample count mismatch')
    if int(oof_manifest.get('outer_training_samples', -1)) != expected_samples:
        raise ValueError('OOF manifest outer-training sample count mismatch')
    manifest_artifacts = oof_manifest.get('artifacts', {})
    if manifest_artifacts.get('consensus_values') != calibration_path.name:
        raise ValueError('OOF manifest points to a different consensus table')

    contract['prediction_interval_calibration'] = {
        **procedure,
        'schema_version': 1,
        'method': 'absolute-consensus-oof-residual-conformal',
        'campaign_round': str(campaign_round),
        'score_version': str(score_version),
        'calibration_file': calibration_path.relative_to(model_directory).as_posix(),
        'calibration_sha256': file_sha256(calibration_path),
        'calibration_manifest_file': (
            oof_manifest_path.relative_to(model_directory).as_posix()
        ),
        'calibration_manifest_sha256': file_sha256(oof_manifest_path),
        'residual_column': 'Mean_OOF_Residual',
        'sample_count': expected_samples,
        'default_confidence_level': 0.95,
        'coverage_note': (
            'Empirical interval calibrated on unique outer-training OOF residuals; '
            'nominal coverage is not guaranteed for out-of-distribution molecules.'
        ),
        'created_at': datetime.now().astimezone().isoformat(),
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    logging.info(
        'Bound %d unique OOF residuals to %s for empirical Score intervals',
        expected_samples,
        contract_path,
    )
    return contract_path


def sample_search_candidates(n_candidates, random_state=42):
    """Return unique parameter dictionaries, shuffled together, and sampling metadata."""
    focused_fraction = 0.6
    focused_count = int(n_candidates * focused_fraction)
    counts = {'focused': focused_count, 'exploration': n_candidates - focused_count}
    spaces = {'focused': FOCUSED_SPACE, 'exploration': EXPLORATION_SPACE}
    rng = np.random.RandomState(random_state)
    seen = set()
    candidates = []
    branches = []
    for branch, count in counts.items():
        space = spaces[branch]
        collected = 0
        while collected < count:
            remaining = count - collected
            # Draw individual values to avoid Windows' 32-bit grid-size limit.
            batch = (
                {name: values[rng.randint(len(values))] for name, values in sorted(space.items())}
                for _ in range(remaining * 2)
            )
            for params in batch:
                key = tuple(sorted(params.items()))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(params)
                branches.append(branch)
                collected += 1
                if collected == count:
                    break
    order = rng.permutation(n_candidates)
    return [candidates[i] for i in order], {
        'requested_candidates': n_candidates,
        'actual_candidates': len(candidates),
        'random_state': random_state,
        'focused_fraction': focused_fraction,
        'branch_counts': counts,
        'spaces': spaces,
        'candidate_branches': [branches[i] for i in order],
        'sampling_rule': 'unique candidates within and across branches; shuffled evaluation order',
    }


def _write_search_text(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_search_diagnostics(
    results, output_dir, *, best_index, cv_splits, feature_names,
    campaign_round, model_stage, target_name, sampling_metadata,
):
    """Atomically save the completed training-side search diagnostics."""
    output_dir = Path(output_dir)
    n_splits = sampling_metadata["cv_n_splits"]
    best_result = results.loc[best_index]
    results = results.copy()
    candidate_count = len(results)
    branches = sampling_metadata["candidate_branches"]
    test_columns = [f"split{i}_test_score" for i in range(len(cv_splits))]
    split_scores = results[test_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    means = pd.to_numeric(results["mean_test_score"], errors="coerce").to_numpy()
    valid = np.isfinite(split_scores).all(axis=1) & np.isfinite(means)
    results["params"] = results["params"].map(
        lambda params: json.dumps(params, sort_keys=True, separators=(",", ":")))
    results.insert(0, "sampling_branch", branches)
    results.insert(0, "candidate_id", np.arange(1, candidate_count + 1))

    progress, best_score, best_id = [], None, None
    for index, score in enumerate(means):
        if valid[index] and (best_score is None or score > best_score):
            best_score, best_id = float(score), index + 1
        progress.append({"candidate_id": index + 1, "candidate_valid": bool(valid[index]),
                         "best_so_far_cv_r2": best_score,
                         "best_so_far_candidate_id": best_id})
    tables = {"cv_results.csv": results, "search_progress.csv": pd.DataFrame(progress)}
    for filename, table in tables.items():
        _write_search_text(output_dir / filename, table.to_csv(index=False))

    summary = {
        **ensemble_metadata(),
        "schema_version": 1,
        "diagnostic_scope": "parameter_selection_cv_not_independent_validation",
        "campaign_round": campaign_round, "model_stage": model_stage, "target": target_name,
        "feature_names": list(feature_names), "cv_n_splits": n_splits,
        "cv_n_repeats": sampling_metadata["cv_n_repeats"],
        "requested_candidate_budget": sampling_metadata["requested_candidates"], "actual_candidate_budget": candidate_count,
        "best_index": int(best_index), "best_candidate_id": int(best_index) + 1,
        "best_params": {"n_features": int(best_result.n_features), "params": best_result.params},
        "best_cv_r2": float(best_result.mean_test_score),
        "candidate_status_counts": {"fully_finite": int(valid.sum()),
                                    "invalid_nonfinite": int((~valid).sum())},
        "sampling_metadata": {k: v for k, v in sampling_metadata.items()
                              if k != "candidate_branches"},
        "software_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                              "scikit-learn": importlib.metadata.version("scikit-learn"),
                              "xgboost": importlib.metadata.version("xgboost")},
    }
    _write_search_text(output_dir / "search_summary.json",
                  json.dumps(summary, indent=2, sort_keys=True) + "\n")


# Shared feature paths are fitted only on the corresponding training rows.
def shap_importance(model, X):
    values, _, _, _ = ensemble_shap_values(model, X)
    return np.abs(values).mean(axis=0)


def feature_path(X, y):
    model = create_xgb_ensemble(RANKING_PARAMS).fit(X, y)
    order = np.argsort(-shap_importance(model, X), kind='stable')
    return np.argsort(order) + 1


def evaluate_candidate(candidate, folds):
    count, params = candidate['n_features'], candidate['params']
    row = {'n_features': count, 'params': params}
    fit_times = []
    for i, (train, validation, y_train, y_validation, ranking) in enumerate(folds):
        mask = ranking <= count
        model = create_xgb_ensemble(params)
        start = perf_counter()
        model.fit(train[:, mask], y_train)
        fit_times.append(perf_counter() - start)
        row[f'split{i}_test_score'] = r2_score(y_validation, model.predict(validation[:, mask]))
        row[f'split{i}_train_score'] = r2_score(y_train, model.predict(train[:, mask]))
    for role in ('test', 'train'):
        scores = [row[f'split{i}_{role}_score'] for i in range(len(folds))]
        row[f'mean_{role}_score'] = float(np.mean(scores))
        row[f'std_{role}_score'] = float(np.std(scores))
    row['mean_fit_time'] = float(np.mean(fit_times))
    return row


def load_historical_candidates(output_dir, campaign_round, target_name, feature_names,
                               counts, search_assignments, review_assignments, path_rows):
    """Read compatible finalist parameters; their saved scores are never reused."""
    output_dir = Path(output_dir)
    run_dir = output_dir.parents[1]
    data_file, provenance_file = run_dir / 'data_encoded.csv', run_dir / 'score_provenance.json'
    if not data_file.is_file() or not provenance_file.is_file():
        return [], []
    data_hash = file_sha256(data_file)
    score_version = json.loads(provenance_file.read_text(encoding='utf-8'))['score_version']
    expected_tables = {
        'cv_split_assignments.csv': pd.DataFrame(search_assignments),
        'refinement_split_assignments.csv': pd.DataFrame(review_assignments),
        'feature_paths.csv': pd.DataFrame(path_rows),
    }
    candidates, sources, seen = [], [], set()
    for history_dir in sorted(run_dir.parent.glob(f'*/{target_name}/joint_search')):
        if history_dir.resolve() == output_dir.resolve():
            continue
        source_run = history_dir.parents[1]
        record = {'source_run': source_run.name}
        try:
            if file_sha256(source_run / 'data_encoded.csv') != data_hash:
                raise ValueError('data_encoded.csv differs')
            provenance = json.loads((source_run / 'score_provenance.json').read_text(encoding='utf-8'))
            if provenance['score_version'] != score_version:
                raise ValueError('Score version differs')
            summary = json.loads((history_dir / 'search_summary.json').read_text(encoding='utf-8'))
            expected = {'campaign_round': campaign_round, 'target': target_name,
                        'model_stage': 'joint_search', 'feature_names': list(feature_names),
                        'cv_n_splits': CV_N_SPLITS, 'cv_n_repeats': CV_N_REPEATS}
            if any(summary[key] != value for key, value in expected.items()):
                raise ValueError('campaign, target, ordered features or CV settings differ')
            metadata = summary['sampling_metadata']
            if any(metadata.get(key) != value for key, value in ensemble_metadata().items()):
                raise ValueError('model seed ensemble differs')
            if (metadata['feature_path_estimator'] != RANKING_PARAMS or
                    metadata['feature_ranking'] != 'mean_absolute_seed_averaged_tree_shap_on_fold_training_rows'):
                raise ValueError('feature-ranking settings differ')
            selection = json.loads((history_dir / 'selection.json').read_text(encoding='utf-8'))
            if any(selection.get(key) != value for key, value in ensemble_metadata().items()):
                raise ValueError('refinement model seed ensemble differs')
            if selection['refinement_seed'] != 2026 or selection['refinement_repeats'] != REFINEMENT_REPEATS:
                raise ValueError('refinement settings differ')
            for filename, expected_table in expected_tables.items():
                table = pd.read_csv(history_dir / filename)
                if filename == 'feature_paths.csv':
                    table = table[table['phase'].isin(['search', 'refinement'])]
                if not table[expected_table.columns].reset_index(drop=True).equals(
                        expected_table.reset_index(drop=True)):
                    raise ValueError(f'{filename} differs')
            finalist_file = history_dir / 'refinement_results.csv'
            finalists = pd.read_csv(finalist_file)
            loaded = []
            unsupported = 0
            for row in finalists.to_dict('records'):
                params = json.loads(row['params'])
                supported = isinstance(params, dict) and any(
                    params.keys() == space.keys() and
                    all(not isinstance(value, bool) and value in space[key]
                        for key, value in params.items())
                    for space in (FOCUSED_SPACE, EXPLORATION_SPACE))
                if row['n_features'] not in counts or not supported:
                    unsupported += 1
                    continue
                loaded.append({'n_features': int(row['n_features']), 'params': params,
                               'source_run': source_run.name,
                               'source_candidate_id': int(row['candidate_id'])})
            if not loaded:
                raise ValueError('no finalists supported by the current feature counts and parameter spaces')
            record.update(status='accepted', reason='matching data, Score, splits and feature paths',
                          supported_candidates=len(loaded), unsupported_candidates=unsupported,
                          data_sha256=data_hash, score_version=score_version,
                          refinement_sha256=file_sha256(finalist_file))
            for candidate in loaded:
                key = (candidate['n_features'], tuple(sorted(candidate['params'].items())))
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            record.update(status='skipped', reason=str(exc))
            logging.warning('Skipping historical XGBoost run %s: %s', source_run.name, exc)
        sources.append(record)
    return candidates, sources


def select_simple_one_se_candidate(review):
    """Prefer fewer features, shallower trees and fewer trees within one SE."""
    best = review.loc[review.mean_test_score.idxmax()]
    threshold = float(best.mean_test_score - best.repeat_se)
    eligible = review[review.mean_test_score >= threshold].copy()
    eligible['_depth'] = eligible.params.map(lambda params: params['max_depth'])
    eligible['_trees'] = eligible.params.map(lambda params: params['n_estimators'])
    selected = eligible.sort_values(
        ['n_features', '_depth', '_trees', 'mean_test_score', 'candidate_id'],
        ascending=[True, True, True, False, True]).iloc[0]
    return selected.drop(labels=['_depth', '_trees']), threshold


def joint_feature_search(X, y, sample_names, output_dir, campaign_round, target_name,
                         n_candidates=N_SEARCH_CANDIDATES, n_jobs=-1):
    """Compete feature counts, then apply one-SE to repeated-CV finalists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(sample_names) != len(X) or len(set(sample_names)) != len(X):
        raise ValueError('CV diagnostics require one unique Name per training row')
    counts = sorted({count for count in FEATURE_COUNTS if count < X.shape[1]} | {X.shape[1]})
    values, targets = np.asarray(X), np.asarray(y)
    path_rows = []

    def prepare(splits, phase):
        folds, assignments = [], []
        for i, (train, validation) in enumerate(splits):
            ranking = feature_path(values[train], targets[train])
            folds.append((values[train], values[validation], targets[train],
                          targets[validation], ranking))
            path_rows.extend({'phase': phase, 'split': i + 1, 'feature': feature,
                              'rank': int(rank)} for feature, rank in zip(X.columns, ranking))
            assignments.extend({'split': i + 1, 'repeat': i // CV_N_SPLITS + 1,
                                'fold': i % CV_N_SPLITS + 1, 'Name': str(sample_names[p]),
                                'role': role, 'train_position': int(p)}
                               for role, positions in (('train', train), ('validation', validation))
                               for p in positions)
        return folds, assignments

    candidates, branches = [], []
    for i, count in enumerate(counts):
        budget = n_candidates // len(counts) + (i < n_candidates % len(counts))
        parameters, metadata = sample_search_candidates(budget, RANDOM_STATE + i)
        candidates.extend({'n_features': count, 'params': params} for params in parameters)
        branches.extend(metadata['candidate_branches'])
    order = np.random.RandomState(RANDOM_STATE).permutation(len(candidates))
    candidates, branches = [candidates[i] for i in order], [branches[i] for i in order]
    splits = list(create_cv(X, y))
    folds, assignments = prepare(splits, 'search')
    assignment_table = pd.DataFrame(assignments).loc[
        :, ['split', 'repeat', 'fold', 'train_position', 'Name', 'role']]
    _write_search_text(output_dir / 'cv_split_assignments.csv', assignment_table.to_csv(index=False))
    _write_search_text(output_dir / 'feature_paths.csv', pd.DataFrame(path_rows).to_csv(index=False))
    logging.info('Joint search: %s candidates across feature counts %s', len(candidates), counts)
    logging.info('Each fold averages model seeds %s before scoring', MODEL_SEEDS)
    with joblib.parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
        rows = joblib.Parallel(verbose=10)(
            joblib.delayed(evaluate_candidate)(candidate, folds) for candidate in candidates)
    results = pd.DataFrame(rows)
    results['rank_test_score'] = results.mean_test_score.rank(method='min', ascending=False).astype(int)
    best = int(results.mean_test_score.idxmax())
    metadata.update(**ensemble_metadata(),
                    requested_candidates=n_candidates, actual_candidates=len(candidates),
                    random_state=RANDOM_STATE, candidate_branches=branches,
                    branch_counts={b: branches.count(b) for b in set(branches)},
                    cv_n_splits=CV_N_SPLITS, cv_n_repeats=CV_N_REPEATS,
                    feature_path_estimator=RANKING_PARAMS,
                    feature_ranking='mean_absolute_seed_averaged_tree_shap_on_fold_training_rows',
                    feature_count_seeds={str(k): RANDOM_STATE + i for i, k in enumerate(counts)},
                    feature_count_budgets={str(k): sum(c['n_features'] == k for c in candidates) for k in counts})
    save_search_diagnostics(results, output_dir, best_index=best, cv_splits=splits,
                            feature_names=X.columns, campaign_round=campaign_round,
                            model_stage='joint_search', target_name=target_name, sampling_metadata=metadata)
    shortlist = results.sort_values('mean_test_score', ascending=False, kind='stable').groupby('n_features').head(2).index
    review_splits = list(create_cv(X, y, n_repeats=REFINEMENT_REPEATS, random_state=2026))
    review_folds, assignments = prepare(review_splits, 'refinement')
    history, history_sources = load_historical_candidates(
        output_dir, campaign_round, target_name, X.columns, counts,
        assignment_table, assignments, path_rows)
    review_candidates = [candidates[i] for i in shortlist]
    review_sources = [{'candidate_id': int(i) + 1, 'candidate_source': 'current',
                       'source_run': output_dir.parents[1].name,
                       'source_candidate_id': int(i) + 1} for i in shortlist]
    candidate_ids = {(c['n_features'], tuple(sorted(c['params'].items()))): i + 1
                     for i, c in enumerate(candidates)}
    seen = {(c['n_features'], tuple(sorted(c['params'].items()))) for c in review_candidates}
    for candidate in history:
        key = (candidate['n_features'], tuple(sorted(candidate['params'].items())))
        if key in seen:
            continue
        candidate_id = candidate_ids.setdefault(key, len(candidate_ids) + 1)
        review_candidates.append({'n_features': candidate['n_features'], 'params': candidate['params']})
        review_sources.append({'candidate_id': candidate_id, 'candidate_source': 'historical',
                               'source_run': candidate['source_run'],
                               'source_candidate_id': candidate['source_candidate_id']})
        seen.add(key)
    logging.info('Joint refinement: %s current finalists plus %s historical configurations',
                 len(shortlist), len(review_candidates) - len(shortlist))
    with joblib.parallel_config(backend='loky', n_jobs=min(joblib.effective_n_jobs(n_jobs), len(review_candidates)),
                                inner_max_num_threads=1):
        review = pd.DataFrame(joblib.Parallel()(
            joblib.delayed(evaluate_candidate)(candidate, review_folds) for candidate in review_candidates))
    review = pd.concat([pd.DataFrame(review_sources), review], axis=1)
    scores = review[[f'split{i}_test_score' for i in range(len(review_splits))]].to_numpy()
    repeat_means = scores.reshape(len(review), REFINEMENT_REPEATS, CV_N_SPLITS).mean(axis=2)
    review['repeat_se'] = repeat_means.std(axis=1, ddof=1) / np.sqrt(REFINEMENT_REPEATS)
    for i in range(REFINEMENT_REPEATS):
        review[f'repeat{i + 1}_mean_r2'] = repeat_means[:, i]
    selected, threshold = select_simple_one_se_candidate(review)
    review['within_one_se'] = review.mean_test_score >= threshold
    logging.info('Selected %s features: review CV R2=%.4f; one-SE threshold=%.4f',
                 int(selected.n_features), selected.mean_test_score, threshold)
    full, full_threshold = select_simple_one_se_candidate(review[review.n_features == X.shape[1]])
    ranking = feature_path(values, targets)
    path_rows.extend({'phase': 'final', 'split': 0, 'feature': feature, 'rank': int(rank)}
                     for feature, rank in zip(X.columns, ranking))
    selection = {**ensemble_metadata(),
                 'diagnostic_scope': 'training_selection_not_independent_validation',
                 'one_se_threshold': threshold, 'full_feature_one_se_threshold': full_threshold,
                 'selection_rule': 'within_one_se_then_fewer_features_shallower_trees_fewer_trees_higher_cv',
                 'history_sources': history_sources,
                 'historical_candidate_count': len(review_candidates) - len(shortlist),
                 'refinement_seed': 2026,
                 'refinement_repeats': REFINEMENT_REPEATS, 'choices': {}}
    for stage, row in (('full_features', full), ('selected_features', selected)):
        features = X.columns[ranking <= int(row.n_features)].tolist()
        selection['choices'][stage] = {'candidate_id': int(row.candidate_id),
            'features': features, 'params': row.params, 'cv_r2': float(row.mean_test_score)}
    review['params'] = review.params.map(lambda p: json.dumps(p, sort_keys=True))
    for name, table in [('refinement_results.csv', review),
                        ('refinement_split_assignments.csv', pd.DataFrame(assignments)),
                        ('feature_paths.csv', pd.DataFrame(path_rows))]:
        _write_search_text(output_dir / name, table.to_csv(index=False))
    _write_search_text(output_dir / 'selection.json', json.dumps(selection, indent=2) + '\n')
    return selection['choices']


# Fit and report the already selected configurations; no test-driven selection.
def train_model_round(X_train, y_train, X_test, y_test,
                     round_name, output_dir, choice, target_name,
                     campaign_round, feature_names=None):
    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        else:
            raise ValueError("Feature names not provided")
    
    best_model = create_xgb_ensemble(choice['params'])
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    
    metrics = {
        'best_params': choice['params'],
        'best_score': choice['cv_r2'],
        'r2': r2_score(y_test, y_pred_test),
        'mse': mean_squared_error(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'predictions_test': y_pred_test,
        'predictions_train': y_pred_train
    }
    
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_importance(best_model, X_train)
    })
    
    model_path = os.path.join(output_dir, f'best_model_{round_name}.pkl')
    joblib.dump(best_model, model_path)
    model_contract = {
        'schema_version': 3,
        **ensemble_metadata(),
        'model_file': os.path.basename(model_path),
        'model_sha256': file_sha256(model_path),
        'model_stage': round_name,
        'campaign_round': campaign_round,
        'target': target_name,
        'feature_names': [str(feature) for feature in feature_names],
        'created_at': datetime.now().astimezone().isoformat(),
    }
    contract_path = os.path.join(
        output_dir,
        f'model_contract_{round_name}.json',
    )
    with open(contract_path, 'w', encoding='utf-8') as handle:
        json.dump(model_contract, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    importances.to_csv(os.path.join(output_dir, f'feature_importance_{round_name}.csv'),
                      index=False)
    
    try:
        shap_analysis(
            best_model, 
            X_train,
            X_test,
            y_train,  
            y_test,   
            feature_names,
            output_dir,
            round_name
        )
    except Exception as e:
        logging.warning(f"Interpretability analysis skipped: {str(e)}", exc_info=True)
    
    return metrics, importances


def write_final_report(report_path, report_items):
    """Write the XGBoost report atomically from unambiguous stage records."""
    report_path = Path(report_path)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in report_items:
                full_metrics = item['full_feature_metrics']
                selected_metrics = item['selected_feature_metrics']
                selected_features = item['selected_features']

                handle.write(f"\n{'-'*20} {item['target']} {'-'*20}\n")
                handle.write(
                    f"Model seeds: {list(MODEL_SEEDS)}; equal-weight prediction average; "
                    "signed SHAP and base values averaged across the same members\n"
                )
                handle.write("\n[Full-feature Stage Results]\n")
                handle.write(f"Best Parameters: {full_metrics['best_params']}\n")
                handle.write(f"Selection CV R^2 (4 folds x {REFINEMENT_REPEATS} repeats): {full_metrics['best_score']:.4f}\n")
                handle.write(f"Test R^2: {full_metrics['r2']:.4f}\n")
                handle.write(f"MSE: {full_metrics['mse']:.4f}\n")
                handle.write(f"MAE: {full_metrics['mae']:.4f}\n")

                handle.write("\n[Selected-feature Stage Results]\n")
                handle.write(
                    f"Selected Features ({len(selected_features)}): \n- "
                    + '\n- '.join(selected_features)
                    + '\n'
                )
                handle.write(f"Best Parameters: {selected_metrics['best_params']}\n")
                handle.write(
                    f"Selection CV R^2 (4 folds x {REFINEMENT_REPEATS} repeats): {selected_metrics['best_score']:.4f}\n"
                )
                handle.write(f"Test R^2: {selected_metrics['r2']:.4f}\n")
                handle.write(f"MSE: {selected_metrics['mse']:.4f}\n")
                handle.write(f"MAE: {selected_metrics['mae']:.4f}\n")

                handle.write("\n[Selected-feature Cross-fitted OOF SHAP]\n")
                handle.write(
                    f"CV: {CV_N_SPLITS} folds x {CV_N_REPEATS} repeats; "
                    "fixed features and fixed selected-stage best parameters; "
                    "outer DUPLEX test excluded\n"
                )
                handle.write(
                    "Role: feature-attribution stability only; formal model "
                    "performance is the outer DUPLEX test result above\n"
                )
                handle.write("Consensus SHAP Ranking:\n")
                for record in item['oof_shap']['feature_summary']:
                    handle.write(
                        f"- {record['Feature']}: "
                        f"mean_abs={record['Consensus_Mean_Abs_SHAP']:.6f}, "
                        f"median_fold_rank={record['Median_Rank']:.1f}, "
                        f"fold_rank_range={record['Min_Rank']}-"
                        f"{record['Max_Rank']}\n"
                    )
                handle.write("="*60 + "\n")
        temporary_path.replace(report_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return report_path

# Main workflow
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and interpret the XGBoost Score model for one campaign."
    )
    add_campaign_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    campaign_round = resolve_campaign_round(args.campaign_round)
    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        campaign_directory_name(campaign_round),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Script started at {datetime.now()}")
    logging.info(f"Campaign round: {campaign_round}")

    try:
        data, score_info = load_scored_physicochemical_data(campaign_round)
        write_score_provenance(output_dir, score_info)
        logging.info(f"Score version: {score_info.get('score_version', 'unknown')}")
        logging.info("Input data: physicochemical source merged with generated Score by Name")
        data_encoded = process_coordination_atom(data)
        data_encoded_path = os.path.join(output_dir, 'data_encoded.csv')
        data_encoded.to_csv(data_encoded_path, index=False)
        data_clean = data_encoded.drop(columns=['Name'])
        
        os.makedirs(os.path.join(output_dir, "EDA"), exist_ok=True)
        full_correlation_matrix = data_clean.drop(columns=ALL_POSSIBLE_TARGETS).corr()
        
        plt.figure(figsize=(10, 10), dpi=PLOT_DPI)
        sns.heatmap(full_correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt=".2f",
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                    linewidths=0.5,
                    square=True,
                    annot_kws={"size": 10})
        
        plt.title('Feature Correlation Matrix Heatmap', fontsize=24, pad=20)
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label("Correlation Coefficient", fontsize=16)  
        cbar.ax.tick_params(labelsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "EDA", "full_correlation_matrix.png"))
        plt.close()
        
        X = data_clean.drop(columns=ALL_POSSIBLE_TARGETS)
        y = {target: data_clean[target] for target in TARGETS}
        
        logging.info(f"Number of features: {X.shape[1]}")
        logging.info(f"Excluded target variables: {ALL_POSSIBLE_TARGETS}")
        logging.info(f"Current target variables: {TARGETS}")

        final_report = []
        
        for target in TARGETS:
            logging.info(f"Processing target: {target}")
            target_dir = os.path.join(output_dir, target)
            os.makedirs(target_dir, exist_ok=True)
            
            X_train, X_test, y_train, y_test = stratified_duplex_split(
                X, y[target]
            )
            X_train, dropped_features = remove_high_correlation_features(X_train)
            X_test = X_test.loc[:, X_train.columns]
            X_filtered = X.loc[:, X_train.columns]
            logging.info(f"Removed highly correlated features using training rows: {dropped_features}")
            
            logging.info(f"Plotting target distribution for {target}...")
            plot_target_distribution(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                y_test.values if isinstance(y_test, pd.Series) else y_test,
                target,
                os.path.join(target_dir, f'{target}_distribution.png')
            )        

            choices = joint_feature_search(
                X_train, y_train, data_encoded.loc[X_train.index, 'Name'].tolist(),
                Path(target_dir) / 'joint_search', campaign_round, target,
            )

            full_features_dir = os.path.join(target_dir, 'full_features')
            os.makedirs(full_features_dir, exist_ok=True)
            
            metrics_full_features, importances_full_features = train_model_round(
                X_train, y_train, X_test, y_test,
                'full_features', full_features_dir,
                choices['full_features'],
                target_name=target,
                campaign_round=campaign_round,
            )
            
            plot_results(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                metrics_full_features['predictions_train'],
                y_test.values if isinstance(y_test, pd.Series) else y_test,
                metrics_full_features['predictions_test'],
                f'{target} - Full-feature stage: Calculated vs Predicted',
                metrics_full_features['r2'],
                os.path.join(full_features_dir, 'prediction_plot.png')
            )
            
            plot_residuals(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                metrics_full_features['predictions_train'],
                y_test.values if isinstance(y_test, pd.Series) else y_test,
                metrics_full_features['predictions_test'],
                f'{target} - Full-feature stage: Residual Analysis',
                os.path.join(full_features_dir, 'residual_plot.png')
            )
            
            plot_feature_importance(
                importances_full_features,
                f'{target} - Full-feature stage: Feature Importance',
                os.path.join(full_features_dir, 'feature_importance.png')
            )
            
            selected_features = pd.Index(choices['selected_features']['features'])
            X_train_selected = X_train.loc[:, selected_features]
            X_test_selected = X_test.loc[:, selected_features]

            selected_features_dir = os.path.join(target_dir, 'selected_features')
            os.makedirs(selected_features_dir, exist_ok=True)
            
            metrics_selected_features, importances_selected_features = train_model_round(
                X_train_selected, y_train,
                X_test_selected, y_test,
                'selected_features', selected_features_dir,
                choices['selected_features'],
                target_name=target,
                campaign_round=campaign_round,
                feature_names=selected_features,
            )

            selected_feature_frame = X_train.loc[:, selected_features].copy()
            outer_train_names = data_encoded.loc[
                X_train.index,
                'Name',
            ].reset_index(drop=True)
            oof_shap_dir = os.path.join(selected_features_dir, 'oof_shap')
            oof_shap_result = cross_validated_oof_shap(
                X_train=selected_feature_frame.reset_index(drop=True),
                y_train=y_train.reset_index(drop=True),
                sample_names=outer_train_names,
                feature_names=selected_features.tolist(),
                best_params=metrics_selected_features['best_params'],
                output_dir=oof_shap_dir,
                round_name='selected_features',
                score_version=score_info.get('score_version', 'unknown'),
                campaign_round=campaign_round,
            )
            bind_oof_prediction_interval_calibration(
                selected_features_dir,
                oof_shap_result,
                campaign_round,
                score_info.get('score_version', 'unknown'),
            )

            plot_results(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                metrics_selected_features['predictions_train'],
                y_test.values if isinstance(y_test, pd.Series) else y_test,
                metrics_selected_features['predictions_test'],
                f'{target} - Selected-feature stage: Calculated vs Predicted',
                metrics_selected_features['r2'],
                os.path.join(selected_features_dir, 'prediction_plot.png')
            )
            
            plot_residuals(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                metrics_selected_features['predictions_train'],
                y_test.values if isinstance(y_test, pd.Series) else y_test,
                metrics_selected_features['predictions_test'],
                f'{target} - Selected-feature stage: Residual Analysis',
                os.path.join(selected_features_dir, 'residual_plot.png')
            )
            
            plot_feature_importance(
                importances_selected_features,
                f'{target} - Selected-feature stage: Feature Importance',
                os.path.join(selected_features_dir, 'feature_importance.png')
            )

            feature_score_scatter_dir = os.path.join(
                selected_features_dir,
                'top_feature_score_scatters',
            )
            plot_top_feature_score_scatters(
                X_filtered,
                y[target],
                importances_selected_features,
                target,
                feature_score_scatter_dir,
                n_features=len(selected_features),
            )
            
            final_report.append({
                'target': target,
                'full_feature_metrics': metrics_full_features,
                'selected_feature_metrics': metrics_selected_features,
                'selected_features': selected_features.tolist(),
                'oof_shap': oof_shap_result,
            })

        report_path = os.path.join(output_dir, 'final_report.txt')
        write_final_report(report_path, final_report)

        logging.info("All targets processed successfully")

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        start_time = datetime.now()
        main()
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Script completed in {duration}")
    finally:
        logging.info("Script execution finished")
