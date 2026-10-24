"""
Integrated regression modeling pipeline with RandomForest and XGBoost.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from structure_selection import CV_CONTRACT, nested_select
from structure_training import (
    validate_data,
    stratified_duplex_split,
    plot_feature_importance as _plot_feature_importance,
    plot_target_distribution as _plot_target_distribution,
)
from xgboost import XGBRegressor
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


# Basic configuration
RANDOM_STATE = 42
BASE_OUTPUT_DIR = STRUCTURE_MODELS_OUTPUT_DIR
MODEL_FAMILY_DIRECTORY = 'rf_xgb'
ENCODED_OUTPUT_FILENAME = 'data-structure.csv'

# Target variable configuration
TARGETS = ['Score']

# Data splitting configuration
TEST_SIZE = 0.2
N_STRATA = 4
DISTANCE_METRIC = 'mahalanobis'

# Random search configuration
N_ITER_RF = 588
N_ITER_XGB = 1288

# RandomForest hyperparameter search space
RF_PARAM_GRID = {
    'model__n_estimators': [100, 200, 400],
    'model__max_depth': [3, 5, 8, 12, None],
    'model__min_samples_split': [2, 4, 8, 12],
    'model__min_samples_leaf': [1, 2, 3, 5, 8],
    'model__max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'model__max_leaf_nodes': [16, 32, 64, None],
    'model__min_impurity_decrease': [0.0, 0.005, 0.01, 0.02],
    'model__max_samples': [0.5, 0.6, 0.8, None],
    'model__bootstrap': [True],
    'model__oob_score': [False],
    'model__criterion': ['friedman_mse', 'squared_error']
}

# XGBoost hyperparameter search space
XGB_PARAM_GRID = {
    'model__n_estimators': [200, 400, 800, 1200],
    'model__max_depth': [2, 3, 4, 6, 8],
    'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
    'model__min_child_weight': [1, 2, 5, 10],
    'model__subsample': [0.5, 0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.5, 0.6, 0.8, 1.0],
    'model__colsample_bylevel': [0.5, 0.6, 0.8, 1.0],
    'model__gamma': [0.0, 0.01, 0.1, 0.5],
    'model__reg_alpha': [0.0, 0.001, 0.01, 0.1, 1.0, 5.0],
    'model__reg_lambda': [0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
}

HUBER_DELTA = 1.0
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
    'RF': {
        'estimator': RandomForestRegressor,
        'params': RF_PARAM_GRID
    },
    'XGB': {
        'estimator': XGBRegressor,
        'params': XGB_PARAM_GRID
    }
}

n_iter_map = {
    'RF': N_ITER_RF,
    'XGB': N_ITER_XGB,
}


def huber_loss(y_true, y_pred, delta=HUBER_DELTA):
    """Huber Loss"""
    error = y_true - y_pred
    return np.mean(np.where(np.abs(error) < delta,
                         0.5 * error**2,
                         delta * (np.abs(error) - 0.5 * delta)))


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
    plt.xlabel('Calculated Value', fontsize=20)
    plt.ylabel('Predicted Value', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=16, loc='upper left')
    plt.text(0.95, 0.05, f'Test R² = {test_r2:.4f}',
             transform=plt.gca().transAxes,
             fontsize=18,
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

    plt.xlabel('Predicted Value', fontsize=20)
    plt.ylabel('Residual', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=16, loc='best')
    plt.title(f'{model_type} Residual Results', fontsize=24, pad=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(save_dir, f'{model_type}_residual_plot.png')
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, title, save_path):
    return _plot_feature_importance(importances, title, save_path, dpi=PLOT_DPI)


def plot_target_distribution(y_train, y_test, target_name, save_dir):
    return _plot_target_distribution(
        y_train, y_test, target_name, save_dir, dpi=PLOT_DPI,
        train_color=PLOT_TRAIN_COLOR, test_color=PLOT_TEST_COLOR,
        min_bins=HIST_MIN_BINS, max_bins=HIST_MAX_BINS,
    )


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
    base_estimator = config['estimator'](random_state=RANDOM_STATE, n_jobs=1, **({'verbosity': 0} if model_type == 'XGB' else {}))
    selection = nested_select(
        X_train, y_train, feature_names, train_names, base_estimator,
        config['params'], n_iter_map[model_type], model_dir,
        method='backward' if model_type == 'KRR' else 'rfe',
        outer_splits=cv_manifest['n_splits'], repeats=cv_manifest['n_repeats'],
        inner_splits=cv_manifest['inner_splits'], inner_repeats=cv_manifest['inner_repeats'],
        seed=cv_manifest['random_state'], max_paths=cv_manifest['max_selection_paths'],
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

        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            importance_method = 'feature_importances'
        elif hasattr(model, 'coef_'):
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(model.coef_)
            })
            importance_method = 'coef'
        else:
            raise ValueError(f"Model {type(model).__name__} has no feature_importances_ or coef_ attribute")

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
        description="Train RandomForest and XGBoost structure models for one campaign."
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
                f"CV_SPLITS={CV_CONTRACT['n_splits']}, CV_REPEATS={CV_CONTRACT['n_repeats']}")
    logging.info(f"Parameter candidates per tuning stage: RF={N_ITER_RF}, XGB={N_ITER_XGB}")

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
            for model_type in ['RF', 'XGB']:
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
            f.write(f"  - Cross-validation: {CV_CONTRACT['n_splits']}-fold x {CV_CONTRACT['n_repeats']} repeats\n")
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
