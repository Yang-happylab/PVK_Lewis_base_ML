"""
Integrated regression modeling pipeline with ElasticNet, KRR, and BayesianRidge.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
import os
import warnings
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from scipy.stats import pearsonr
from structure_selection import CV_CONTRACT, nested_select
from sklearn.inspection import permutation_importance
from data_preparation import (
    build_split_manifest,
    load_scored_structure_data,
    write_model_feature_config,
    write_score_provenance,
)
from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from project_paths import STRUCTURE_MODELS_OUTPUT_DIR
from structure_features import FEATURE_COLUMNS

KRR_CHILD_WARNING_FILTER = (
    'ignore:Singular matrix in solving dual problem:'
    'UserWarning:sklearn.linear_model._ridge'
)


def _configure_parallel_warning_filters():
    """Pass exact warning filters to joblib's spawned Python workers."""
    existing_options = [
        option.strip()
        for option in os.environ.get('PYTHONWARNINGS', '').split(',')
        if option.strip()
    ]
    if KRR_CHILD_WARNING_FILTER not in existing_options:
        os.environ['PYTHONWARNINGS'] = ','.join(
            [*existing_options, KRR_CHILD_WARNING_FILTER]
        )


_configure_parallel_warning_filters()
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings(
    'ignore',
    message=r'Singular matrix in solving dual problem',
    category=UserWarning,
    module=r'sklearn\.linear_model\._ridge',
)

# Basic configuration
RANDOM_STATE = 42
BASE_OUTPUT_DIR = STRUCTURE_MODELS_OUTPUT_DIR
MODEL_FAMILY_DIRECTORY = 'en_krr_br'
ENCODED_OUTPUT_FILENAME = 'data-structure.csv'

# Target variable configuration
TARGETS = ['Score']

# Data splitting configuration
TEST_SIZE = 0.2
N_STRATA = 4
DISTANCE_METRIC = 'mahalanobis'

# Cross-validation configuration
CV_N_SPLITS = 4
CV_N_REPEATS = 3

# Random search configuration
N_ITER_ELASTICNET = 988
N_ITER_KRR = 788
N_ITER_BAYESIANRIDGE = 988

# ElasticNet hyperparameter search space
ELASTICNET_PARAM_GRID = {
    'model__alpha': np.concatenate([
        np.logspace(-3, -1, 40),
        np.logspace(-1, 0.5, 50),
        np.logspace(0.5, 1.5, 30),
    ]),
    'model__l1_ratio': np.concatenate([
        [0.01, 0.05, 1.0],
        np.linspace(0.1, 0.5, 40),
        np.linspace(0.5, 0.8, 50),
        np.linspace(0.8, 0.95, 30),
    ]),
    'model__max_iter': [50000],
    'model__tol': [1e-5],
    'model__selection': ['cyclic'],
    'model__random_state': [RANDOM_STATE],
}

# KernelRidge hyperparameter search space
KRR_PARAM_GRID = [
    {
        'model__kernel': ['rbf'],
        'model__alpha': np.concatenate([
            np.linspace(0.5, 3.0, 35),
            np.linspace(3.0, 10.0, 40),
            np.linspace(10.0, 50.0, 25)
        ]),
        'model__gamma': np.concatenate([
            np.logspace(-5, -3, 40),
            np.logspace(-3, -1.5, 35),
            np.logspace(-1.5, -0.5, 25)
        ])
    },
    {
        'model__kernel': ['poly'],
        'model__alpha': np.concatenate([
            np.linspace(1.0, 5.0, 40),
            np.linspace(5.0, 15.0, 35),
            np.linspace(15.0, 50.0, 25)
        ]),
        'model__gamma': np.concatenate([
            np.logspace(-4, -2.5, 40),
            np.logspace(-2.5, -1.5, 30),
            np.linspace(0.01, 0.05, 30)
        ]),
        'model__degree': [2, 3, 4],
        'model__coef0': np.concatenate([
            np.linspace(0, 1, 30),
            np.linspace(1, 3, 40),
            np.linspace(3, 5, 20)
        ])
    },
    {
        'model__kernel': ['laplacian'],
        'model__alpha': np.concatenate([
            np.linspace(0.5, 5.0, 40),
            np.linspace(5.0, 20.0, 35),
            np.linspace(20.0, 50.0, 25)
        ]),
        'model__gamma': np.logspace(-4, -0.5, 50)
    },
    {
        'model__kernel': ['sigmoid'],
        'model__alpha': np.concatenate([
            np.linspace(1.0, 5.0, 30),
            np.linspace(5.0, 20.0, 40),
            np.linspace(20.0, 50.0, 30)
        ]),
        'model__gamma': np.logspace(-4, -0.5, 40),
        'model__coef0': np.linspace(-1, 1, 20)
    }
]

# BayesianRidge hyperparameter search space
BAYESIANRIDGE_PARAM_GRID = {
    'model__alpha_1': np.concatenate([
        np.logspace(-8, -6, 30),
        np.logspace(-6, -4, 40),
        np.logspace(-4, -2, 30)
    ]),
    'model__alpha_2': np.concatenate([
        np.logspace(-10, -8, 30),
        np.logspace(-8, -6, 40),
        np.logspace(-6, -4, 30)
    ]),
    'model__lambda_1': np.concatenate([
        np.logspace(-4, -2, 30),
        np.logspace(-2, 0, 40),
        np.logspace(0, 1.5, 30)
    ]),
    'model__lambda_2': np.concatenate([
        np.logspace(-4, -2, 30),
        np.logspace(-2, 0, 40),
        np.logspace(0, 1.5, 30)
    ]),
    'model__max_iter': [50000],
    'model__tol': [1e-5],
    'model__compute_score': [True],
    'model__fit_intercept': [True]
}

HUBER_DELTA = 1.0
PERMUTATION_N_REPEATS = 100
PLOT_DPI = 600
PLOT_FONT_FAMILY = 'Arial'
PLOT_TRAIN_COLOR = 'red'
PLOT_TEST_COLOR = 'blue'
PLOT_TRAIN_ALPHA = 0.3
PLOT_TEST_ALPHA = 0.6
PLOT_MARKER_SIZE = 100
HIST_MIN_BINS = 10
HIST_MAX_BINS = 20
LOG_FILENAME = 'training.log'
LOG_LEVEL = logging.INFO

plt.rcParams['font.family'] = PLOT_FONT_FAMILY

MODELS = {
    'ElasticNet': {
        'estimator': ElasticNet,
        'params': ELASTICNET_PARAM_GRID
    },
    'KRR': {
        'estimator': KernelRidge,
        'params': KRR_PARAM_GRID
    },
    'BayesianRidge': {
        'estimator': BayesianRidge,
        'params': BAYESIANRIDGE_PARAM_GRID
    }
}

n_iter_map = {
    'ElasticNet': N_ITER_ELASTICNET,
    'KRR': N_ITER_KRR,
    'BayesianRidge': N_ITER_BAYESIANRIDGE
}


def huber_loss(y_true, y_pred, delta=HUBER_DELTA):
    """Huber Loss"""
    error = y_true - y_pred
    return np.mean(np.where(np.abs(error) < delta,
                         0.5 * error**2,
                         delta * (np.abs(error) - 0.5 * delta)))


def safe_pearson(y_true, y_pred):
    """Pearson correlation with safe handling for constant inputs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return 0.0

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            corr, _ = pearsonr(y_true, y_pred)

        if np.isnan(corr) or np.isinf(corr):
            return 0.0
        return corr
    except:
        return 0.0


SCORING = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mape': make_scorer(lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred)/y_true)) if np.all(y_true != 0) else 0.0),
    'max_error': 'max_error',
    'explained_variance': 'explained_variance',
    'pearson': make_scorer(safe_pearson),
    'huber': make_scorer(huber_loss, delta=HUBER_DELTA)
}


# Data validation module
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


# Data splitting module
def stratified_duplex_split(
    X,
    y,
    test_size=TEST_SIZE,
    n_strata=N_STRATA,
    metric=DISTANCE_METRIC,
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
            except:
                logging.warning(f"Mahalanobis distance failed for stratum {stratum}, using Euclidean")
                dist_matrix = cdist(X_scaled, X_scaled, 'euclidean')
        else:
            dist_matrix = cdist(X_scaled, X_scaled, 'euclidean')

        if n_stratum_samples < 2:
            train_idx_list.extend(stratum_indices)
            logging.warning(f"Stratum {stratum} has only 1 sample, assigned to training set")
            continue

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

    train_idx = np.array(train_idx_list)
    test_idx = np.array(test_idx_list)

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


# Visualization module
def plot_results(y_train, y_pred_train, y_test, y_pred_test, model_type, test_r2, save_dir):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    plt.scatter(y_train, y_pred_train, alpha=PLOT_TRAIN_ALPHA, color=PLOT_TRAIN_COLOR,
                s=PLOT_MARKER_SIZE, label='Training Set')
    plt.scatter(y_test, y_pred_test, alpha=PLOT_TEST_ALPHA, color=PLOT_TEST_COLOR,
                s=PLOT_MARKER_SIZE, label='Test Set')

    all_actual = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val = min(all_actual.min(), all_pred.min())
    max_val = max(all_actual.max(), all_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=3, label='Ideal')
    plt.xlabel('Calculated Value', fontsize=22)
    plt.ylabel('Predicted Value', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='upper left')
    plt.text(0.95, 0.05, f'Test R² = {test_r2:.4f}',
             transform=plt.gca().transAxes,
             fontsize=20,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title(f'{model_type} Prediction Results', fontsize=24, pad=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f'{model_type}_prediction_plot.png')
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_residuals(y_train, y_pred_train, y_test, y_pred_test, model_type, save_dir):
    """Plot residual distribution."""
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    plt.scatter(y_pred_train, residuals_train, alpha=PLOT_TRAIN_ALPHA, color=PLOT_TRAIN_COLOR,
                s=PLOT_MARKER_SIZE, label='Training Set')
    plt.scatter(y_pred_test, residuals_test, alpha=PLOT_TEST_ALPHA, color=PLOT_TEST_COLOR,
                s=PLOT_MARKER_SIZE, label='Test Set')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=3)

    plt.xlabel('Predicted Value', fontsize=22)
    plt.ylabel('Residual', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='best')
    plt.title(f'{model_type} Residual Results', fontsize=24, pad=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f'{model_type}_residual_plot.png')
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, title, save_path):
    """Plot feature importance bar chart."""
    plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    importances.sort_values(by='Importance', ascending=True).plot(
        kind='barh', x='Feature', y='Importance', legend=False)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(title, fontsize=15, pad=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='x', labelsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI)
    plt.close()


def plot_target_distribution(y_train, y_test, target_name, save_dir):
    """Plot target variable distribution for train and test sets."""
    plt.figure(figsize=(12, 8), dpi=PLOT_DPI)

    all_y = np.concatenate([y_train, y_test])
    q75, q25 = np.percentile(all_y, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(all_y) ** (1/3))
    n_bins = int((all_y.max() - all_y.min()) / bin_width) if bin_width > 0 else 30
    n_bins = min(max(n_bins, HIST_MIN_BINS), HIST_MAX_BINS)

    bins = np.linspace(all_y.min(), all_y.max(), n_bins + 1)
    plt.hist(y_train, bins=bins, alpha=0.4, color=PLOT_TRAIN_COLOR,
             label=f'Training Set (n={len(y_train)})', edgecolor='darkred', linewidth=1.5)
    plt.hist(y_test, bins=bins, alpha=0.7, color=PLOT_TEST_COLOR,
             label=f'Test Set (n={len(y_test)})', edgecolor='darkblue', linewidth=1.5)
    plt.xlabel(f'{target_name} Value', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='upper left')
    plt.title(f'{target_name} Distribution', fontsize=24, pad=10)
    plt.grid(alpha=0.3, axis='y')

    save_path = os.path.join(save_dir, f'{target_name}_distribution.png')
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


# Model training core module
def train_model(
    model_type,
    target,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    output_dir,
    score_info,
    split_manifest,
    cv_manifest,
    train_names,
    test_names,
):
    """Execute single model training pipeline."""
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    train_issues = validate_data(X_train, y_train, f"{model_type} training data")
    test_issues = validate_data(X_test, y_test, f"{model_type} test data")

    if train_issues or test_issues:
        for issue in train_issues + test_issues:
            logging.warning(issue)

    full_feature_names = list(feature_names)
    config = MODELS[model_type]
    base_estimator = config['estimator'](warm_start=True) if model_type == 'ElasticNet' else config['estimator']()
    if model_type == 'KRR':
        logging.info("Repeated singular-matrix warnings are suppressed during the parallel "
                     "KRR search; scikit-learn still applies its least-squares fallback.")
    selection = nested_select(
        X_train, y_train, feature_names, train_names, base_estimator,
        config['params'], n_iter_map[model_type], model_dir,
        method='backward' if model_type == 'KRR' else 'rfe',
    )
    feature_mask = selection['support']
    feature_names = np.asarray(feature_names)[feature_mask].tolist()
    X_train, X_test = X_train[:, feature_mask], X_test[:, feature_mask]
    best_model = selection['model']
    y_pred = best_model.predict(X_test)
    n_valid = n_total = selection['n_candidates']
    logging.info('%s: retained %d/%d features', model_type, len(feature_names), len(full_feature_names))

    evaluation = {
        'cv_r2_mean': float(selection['cv_r2_mean']),
        'cv_r2_std': float(selection['cv_r2_std']),
        'cv_mae_mean': float(selection['cv_mae_mean']),
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_mse': float(mean_squared_error(y_test, y_pred)),
        'test_mae': float(mean_absolute_error(y_test, y_pred)),
        'test_huber': float(huber_loss(y_test, y_pred)),
    }

    model_path = os.path.join(model_dir, f'{model_type}_best_model.pkl')
    joblib.dump(best_model, model_path)

    try:
        model = best_model.named_steps['model']

        if isinstance(model, KernelRidge):
            logging.info("Computing descriptive training-set permutation importance for KRR...")
            result = permutation_importance(
                best_model,
                X_train,
                y_train,
                n_repeats=PERMUTATION_N_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': result.importances_mean
            })
            importance_method = 'permutation'

        elif hasattr(model, 'coef_'):
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(model.coef_)
            })
            importance_method = 'coef'
        else:
            raise ValueError(f"Model {type(model).__name__} has no coef_ attribute")

        importance_path = os.path.join(model_dir, f'{model_type}_feature_importance_{importance_method}.csv')
        importances.to_csv(importance_path, index=False)
        logging.info(f"Saved feature importance ({importance_method})")

    except Exception as e:
        logging.error(f"Error computing feature importance: {str(e)}", exc_info=True)
        raise

    y_pred_train = best_model.predict(X_train)
    stacking_predictions = pd.DataFrame({
        'Name': np.concatenate([train_names, test_names]),
        'Split': ['Train'] * len(train_names) + ['Test'] * len(test_names),
        'Score': np.concatenate([y_train, y_test]),
        'Prediction': np.concatenate([y_pred_train, y_pred]),
    })
    if stacking_predictions['Name'].duplicated().any():
        raise ValueError('Stacking predictions require unique molecule names')
    stacking_predictions.to_csv(
        os.path.join(model_dir, 'stacking_base_predictions.csv'), index=False,
    )
    plot_results(y_train, y_pred_train, y_test, y_pred,
                model_type, r2_score(y_test, y_pred), model_dir)
    plot_residuals(y_train, y_pred_train, y_test, y_pred,
                  model_type, model_dir)
    plot_feature_importance(importances,
                        f'{model_type} Feature Importance',
                        os.path.join(model_dir, f'{model_type}_feature_importance.png'))

    feature_config_path = os.path.join(
        model_dir,
        f'{model_type}_feature_config.json',
    )
    write_model_feature_config(
        feature_config_path,
        model_path=model_path,
        model_type=model_type,
        target=target,
        full_feature_names=full_feature_names,
        retained_feature_names=feature_names,
        score_info=score_info,
        split_manifest=split_manifest,
        cv_manifest=cv_manifest,
        evaluation=evaluation,
        training_script=os.path.basename(__file__),
    )

    return {
        'best_params': selection['best_params'],
        'selected_features': feature_names,
        'cv_metrics': {
            'r2_mean': evaluation['cv_r2_mean'],
            'r2_std': evaluation['cv_r2_std'],
            'mae_mean': evaluation['cv_mae_mean'],
            'valid_combinations': f"{n_valid}/{n_total}"
        },
        'test_metrics': {
            'r2': evaluation['test_r2'],
            'mse': evaluation['test_mse'],
            'mae': evaluation['test_mae'],
            'huber': evaluation['test_huber']
        }
    }


# Main workflow control
def build_parser():
    parser = argparse.ArgumentParser(
        description="Train ElasticNet, KRR, and BayesianRidge for one campaign."
    )
    add_campaign_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    campaign_round = resolve_campaign_round(args.campaign_round)
    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        campaign_directory_name(campaign_round),
        MODEL_FAMILY_DIRECTORY,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_dir, LOG_FILENAME),
        level=LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    logging.getLogger('').addHandler(console_handler)

    logging.info(f"Script started at {datetime.now()}")
    logging.info(f"Campaign round: {campaign_round}")
    logging.info(f"Configuration: TEST_SIZE={TEST_SIZE}, N_STRATA={N_STRATA}, "
                f"CV_SPLITS={CV_N_SPLITS}, CV_REPEATS={CV_N_REPEATS}")
    logging.info(f"Parameter candidates per tuning stage: ElasticNet={N_ITER_ELASTICNET}, "
                f"KRR={N_ITER_KRR}, BayesianRidge={N_ITER_BAYESIANRIDGE}")

    try:
        data, score_info = load_scored_structure_data(campaign_round)
        write_score_provenance(output_dir, score_info)
        logging.info(
            "Structure data joined with generated Score by Name "
            f"(score_version={score_info.get('score_version', 'unknown')})"
        )
        logging.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

        feature_cols = list(FEATURE_COLUMNS)

        missing_cols = [col for col in feature_cols + TARGETS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        logging.info(f"Using {len(feature_cols)} ordered numerical structure features")

        final_report = {}
        training_failures = []
        for target in TARGETS:
            logging.info(f"Processing target: {target}")
            target_dir = os.path.join(output_dir, target)
            os.makedirs(target_dir, exist_ok=True)

            X = data[feature_cols]
            y = data[target]

            logging.info(f"Target {target} - Mean: {y.mean():.4f}, Std: {y.std():.4f}, Range: [{y.min():.4f}, {y.max():.4f}]")

            logging.info("Performing numerical-feature data splitting...")
            (
                X_train,
                X_test,
                y_train,
                y_test,
                train_indices,
                test_indices,
            ) = stratified_duplex_split(
                X,
                y,
                test_size=TEST_SIZE,
                n_strata=N_STRATA,
                metric=DISTANCE_METRIC,
                return_indices=True,
            )
            split_manifest = build_split_manifest(
                data['Name'],
                train_indices,
                test_indices,
                test_fraction=TEST_SIZE,
                requested_strata=N_STRATA,
                distance_metric=DISTANCE_METRIC,
            )
            cv_manifest = dict(CV_CONTRACT)

            feature_names = list(feature_cols)
            logging.info(f"Number of structure features: {len(feature_names)}")
            logging.info(f"Plotting target distribution for {target}...")
            plot_target_distribution(y_train, y_test, target, target_dir)

            with open(os.path.join(target_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
                f.write("Structure Feature Names:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"{i+1}. {name}\n")

            if target == TARGETS[0]:
                X_all = data[feature_cols]
                X_all_numeric = X_all.to_numpy(dtype=float)

                structure_df = pd.DataFrame(
                    X_all_numeric,
                    columns=feature_names
                )
                structure_df[TARGETS] = data[TARGETS].values

                encoded_path = os.path.join(output_dir, ENCODED_OUTPUT_FILENAME)
                structure_df.to_csv(encoded_path, index=False)
                logging.info(f"Numerical structure data saved to {ENCODED_OUTPUT_FILENAME}")

            model_results = {}
            for model_type in ['ElasticNet', 'KRR', 'BayesianRidge']:
                logging.info(f"Training {model_type} for {target}")
                try:
                    model_metrics = train_model(
                        model_type=model_type,
                        target=target,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_names=feature_names,
                        output_dir=target_dir,
                        score_info=score_info,
                        split_manifest=split_manifest,
                        cv_manifest=cv_manifest,
                        train_names=data['Name'].iloc[train_indices].to_numpy(),
                        test_names=data['Name'].iloc[test_indices].to_numpy(),
                    )
                    model_results[model_type] = model_metrics
                except Exception as e:
                    logging.error(f"Failed to train {model_type}: {str(e)}")
                    training_failures.append(f"{target}/{model_type}: {str(e)}")
                    continue

            final_report[target] = model_results

        report_path = os.path.join(output_dir, 'final_report.txt')
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("MODEL TRAINING FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("Configuration:\n")
            f.write(f"  - Test Size: {TEST_SIZE}\n")
            f.write(f"  - Stratification: {N_STRATA} strata\n")
            f.write(f"  - Cross-validation: {CV_N_SPLITS}-fold x {CV_N_REPEATS} repeats\n")
            f.write(f"  - Distance Metric: {DISTANCE_METRIC}\n")
            f.write(f"  - Random State: {RANDOM_STATE}\n\n")

            for target, results in final_report.items():
                f.write(f"\n{'-'*30} {target} {'-'*30}\n")
                for model_type, metrics in results.items():
                    f.write(f"\n[{model_type} Results]\n")
                    f.write(f"Best Parameters: {metrics['best_params']}\n")
                    f.write(f"Nested CV R2: {metrics['cv_metrics']['r2_mean']:.4f} +- {metrics['cv_metrics']['r2_std']:.4f}\n")
                    f.write(f"Test R2: {metrics['test_metrics']['r2']:.4f}\n")
                    f.write(f"Test MSE: {metrics['test_metrics']['mse']:.4f}\n")
                    f.write(f"Test MAE: {metrics['test_metrics']['mae']:.4f}\n")
                    f.write(f"Test Huber: {metrics['test_metrics']['huber']:.4f}\n")
                    f.write(f"Valid Combinations: {metrics['cv_metrics']['valid_combinations']}\n")

        if training_failures:
            raise RuntimeError(
                "One or more models failed: " + "; ".join(training_failures)
            )

        logging.info("All training completed successfully")
        print(f"\nTraining completed. Results saved to: {output_dir}")

    except Exception as e:
        logging.error(f"Main process failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    start_time = datetime.now()
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    finally:
        print(f"\nTotal execution time: {datetime.now() - start_time}")
