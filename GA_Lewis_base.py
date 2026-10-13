import argparse
import logging
import numpy as np
from joblib import load
import os
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from data_preparation import (
    MODEL_CV_MANIFEST_KEYS,
    MODEL_EVALUATION_KEYS,
    MODEL_FEATURE_CONFIG_SCHEMA_VERSION,
    MODEL_SPLIT_MANIFEST_KEYS,
    MODEL_TRAINING_DATA_KEYS,
    file_sha256,
    write_score_provenance,
)
from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from project_paths import (
    GA_OUTPUT_DIR,
    PROJECT_ROOT,
    STRUCTURE_MODELS_OUTPUT_DIR,
    find_latest_model,
)
from molecular_design import (
    ARM_NAMES,
    CENTER_TEMPLATES,
    assemble,
    generate_valid_sample as generate_design_sample,
    is_valid as is_valid_design,
)

from structure_features import FEATURE_COLUMNS, EXTRACTION_SETTINGS, describe_structure
from structure_stacking import train_stacking
from structure_selection import evaluation_cv_contract

# These valid molecular graphs have donor environments outside the feature contract.
EXCLUDED_STRUCTURES = {'N': 'NH3', 'O': 'H2O', 'S': 'H2S', 'Cl': 'HCl'}


def is_valid(sample):
    return is_valid_design(sample) and assemble(sample) not in EXCLUDED_STRUCTURES


def generate_valid_sample():
    while True:
        sample = generate_design_sample()
        if is_valid(sample):
            return sample

BASE_OUTPUT_DIR = GA_OUTPUT_DIR
MODEL_OUTPUT_DIR = STRUCTURE_MODELS_OUTPUT_DIR
RANDOM_SEED = 42
PLOT_FONT = 'Arial'

SCORE_PROVENANCE_KEYS = (
    'score_version',
    'campaign_round',
    'score_output_dir',
    'scores_file',
    'manifest_path',
    'data_sha256',
    'calibration_sha256',
    'calibration_subset_sha256',
    'algorithm_version',
)

# Model file names
MODEL_FILES = {
    'KRR': 'KRR_best_model.pkl',
    'RF': 'RF_best_model.pkl',
    'XGB': 'XGB_best_model.pkl',
    'EN': 'ElasticNet_best_model.pkl',
    'BR': 'BayesianRidge_best_model.pkl'
}

# Model type names stored by the two training workflows
MODEL_TYPE_BY_ALIAS = {
    'KRR': 'KRR',
    'RF': 'RF',
    'XGB': 'XGB',
    'EN': 'ElasticNet',
    'BR': 'BayesianRidge',
}

# Automatic model eligibility configuration
MODEL_SELECTION_CONFIG = {
    'gate_metric': 'cv_r2_mean',
    'min_cv_r2_mean': 0.70,
    'min_cv_r2_lcb': 0.65,
    'min_enabled_models': 2,
    'force_exclude': [],
}

# Genetic algorithm configuration
GA_CONFIG = {
    'pop_size': 100,
    'generations': 50,
    'mutation_rate': 0.3,
    'elite_ratio': 0.025,
    'early_stop': 5,
    'max_init_attempts': 5,
    'crossover_perturbation': 0.05,
}

# Visualization configuration
PLOT_CONFIG = {
    'figure_size': (10, 3),
    'dpi': 600,
    'line_width': 2,
    'marker_size': 8,
    'title_fontsize': 24,
    'label_fontsize': 16,
    'tick_fontsize': 16,
    'legend_fontsize': 16,
    'colors': {
        'best': 'blue',
        'average': 'red'
    },
    'ylim_padding': {
        'lower': 0.85,
        'upper': 1.5
    }
}

# Logging configuration
LOG_CONFIG = {
    'encoding': 'utf-8',
    'level': logging.INFO,
    'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    'console_format': "%(asctime)s - %(levelname)s - %(message)s"
}

# Output configuration
OUTPUT_CONFIG = {
    'save_history': True,
    'save_plot': True,
    'save_readable_format': True,
    'top_n_results': 10
}

np.random.seed(RANDOM_SEED)
plt.rcParams['font.family'] = PLOT_FONT


def setup_logger(dir):
    """Setup logging system."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(dir, f"run_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding=LOG_CONFIG['encoding'])
    file_handler.setLevel(LOG_CONFIG['level'])
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_CONFIG['level'])
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG['console_format']))

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_CONFIG['level'])
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class CategoryEncoder:
    """Map molecular-design samples to ordered numerical structure features."""

    def __init__(self):
        self.full_feature_names = list(FEATURE_COLUMNS)
        self.model_n_features = {}
        self.model_feature_indices = {}

    def encode_sample(self, sample):
        """Describe the molecule assembled from one template sample."""
        return describe_structure(assemble(sample))

    def encode_batch(self, samples):
        """Batch encoding."""
        return np.array([self.encode_sample(s) for s in samples])

    def align_features(self, X_full, model_name):
        """Align feature dimensions based on model's feature index mapping."""
        if model_name not in self.model_feature_indices:
            return X_full

        feature_indices = self.model_feature_indices[model_name]
        return X_full[:, feature_indices]

    def extract_feature_config_from_model(self, model, model_name, retained_feature_names):
        """Validate and apply the retained features recorded beside a model."""
        logger = logging.getLogger(__name__)

        n_features_expected = None
        if hasattr(model, 'named_steps'):
            if 'scaler' in model.named_steps:
                scaler = model.named_steps['scaler']
                if hasattr(scaler, 'n_features_in_'):
                    n_features_expected = scaler.n_features_in_
            if n_features_expected is None and 'model' in model.named_steps:
                estimator = model.named_steps['model']
                if hasattr(estimator, 'n_features_in_'):
                    n_features_expected = estimator.n_features_in_
        elif hasattr(model, 'n_features_in_'):
            n_features_expected = model.n_features_in_

        if n_features_expected is None:
            raise ValueError(
                f"Cannot determine the trained feature count for {model_name}"
            )
        if not isinstance(retained_feature_names, list) or not retained_feature_names:
            raise ValueError(
                f"Feature configuration for {model_name} must contain "
                "retained_feature_names"
            )
        if len(retained_feature_names) != len(set(retained_feature_names)):
            raise ValueError(
                f"Feature configuration for {model_name} contains duplicate names"
            )

        unknown_names = sorted(
            set(retained_feature_names) - set(self.full_feature_names)
        )
        if unknown_names:
            raise ValueError(
                f"Unknown retained features for {model_name}: {unknown_names}"
            )

        retained_indices = np.array([
            self.full_feature_names.index(name)
            for name in retained_feature_names
        ])
        if len(retained_indices) != n_features_expected:
            raise ValueError(
                f"Feature configuration for {model_name} contains "
                f"{len(retained_indices)} features, but the model expects "
                f"{n_features_expected}"
            )

        self.model_n_features[model_name] = n_features_expected
        self.model_feature_indices[model_name] = retained_indices
        logger.info(
            f"Model {model_name} uses its required feature configuration "
            f"({n_features_expected} features)"
        )
        return n_features_expected


def score_info_from_feature_config(config):
    """Extract score provenance from a model feature sidecar."""
    nested_manifest = config.get('score_manifest')
    if not isinstance(nested_manifest, dict):
        raise ValueError("Model feature configuration is missing score_manifest")

    missing = [
        key for key in SCORE_PROVENANCE_KEYS
        if key not in nested_manifest
        or nested_manifest[key] is None
        or (isinstance(nested_manifest[key], str) and not nested_manifest[key].strip())
    ]
    if missing:
        raise ValueError(
            "Model score_manifest is incomplete: " + ", ".join(missing)
        )
    if str(config.get('score_version', '')).strip() != str(
        nested_manifest['score_version']
    ).strip():
        raise ValueError(
            "Model feature configuration has inconsistent Score versions"
        )

    score_info = {}
    for key in SCORE_PROVENANCE_KEYS:
        value = nested_manifest[key]
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str) and key in {
            'score_output_dir', 'scores_file', 'manifest_path'
        }:
            path = Path(value)
            if not path.is_absolute():
                value = str(PROJECT_ROOT / path)
        score_info[key] = value
    return score_info


def evaluation_from_feature_config(config, model_name):
    """Validate and return the structured evaluation bound to one model."""
    if config.get('structure_feature_extraction') != EXTRACTION_SETTINGS:
        raise ValueError(f"Model {model_name} requires retraining with current structure features")
    if config.get('cv_manifest', {}).get('evaluation_scope') != 'outer_training_nested_cv':
        raise ValueError(f"Model {model_name} requires retraining with nested feature selection")
    schema_version = config.get('feature_config_schema_version')
    if schema_version != MODEL_FEATURE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Model {model_name} requires feature-config schema "
            f"{MODEL_FEATURE_CONFIG_SCHEMA_VERSION}, found {schema_version!r}"
        )

    expected_model_type = MODEL_TYPE_BY_ALIAS[model_name]
    if config.get('model_type') != expected_model_type:
        raise ValueError(
            f"Model {model_name} sidecar identifies model_type="
            f"{config.get('model_type')!r}, expected {expected_model_type!r}"
        )
    if config.get('target') != 'Score':
        raise ValueError(
            f"Model {model_name} was not evaluated for the Score target"
        )

    evaluation = config.get('evaluation')
    if not isinstance(evaluation, dict):
        raise ValueError(f"Model {model_name} has no structured evaluation")
    required_keys = set(MODEL_EVALUATION_KEYS)
    missing_keys = sorted(required_keys.difference(evaluation))
    if missing_keys:
        raise ValueError(
            f"Model {model_name} evaluation is missing: {missing_keys}"
        )
    numeric_evaluation = {
        key: float(evaluation[key])
        for key in sorted(required_keys)
    }
    if not all(np.isfinite(value) for value in numeric_evaluation.values()):
        raise ValueError(f"Model {model_name} evaluation is not finite")
    if numeric_evaluation['cv_r2_std'] < 0:
        raise ValueError(f"Model {model_name} has a negative CV R2 standard deviation")
    return numeric_evaluation


def _canonical_contract(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def validate_model_training_contracts(model_artifacts):
    """Require every candidate metric to come from the same data and split."""
    if not model_artifacts:
        raise ValueError("No model artifacts were loaded")

    required_split_keys = set(MODEL_SPLIT_MANIFEST_KEYS)
    required_cv_keys = set(MODEL_CV_MANIFEST_KEYS)
    required_training_data_keys = set(MODEL_TRAINING_DATA_KEYS)

    for model_name, artifact in model_artifacts.items():
        for field_name, required_keys in (
            ('split_manifest', required_split_keys),
            ('cv_manifest', required_cv_keys),
            ('training_data', required_training_data_keys),
        ):
            value = artifact.get(field_name)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Model {model_name} is missing {field_name}"
                )
            missing_keys = sorted(required_keys.difference(value))
            if missing_keys:
                raise ValueError(
                    f"Model {model_name} {field_name} is missing: {missing_keys}"
                )

    for field_name in ('split_manifest', 'cv_manifest', 'training_data'):
        contracts = {
            _canonical_contract(evaluation_cv_contract(artifact[field_name])
                                if field_name == 'cv_manifest' else artifact[field_name]): model_name
            for model_name, artifact in model_artifacts.items()
        }
        if len(contracts) != 1:
            raise ValueError(
                f"Candidate models do not share one {field_name} contract"
            )


def _portable_project_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def select_models_by_performance(models, model_artifacts, config):
    """Apply the declared CV mean and lower-bound gates to all candidates."""
    if set(models) != set(model_artifacts):
        raise ValueError("Loaded models and model artifacts do not match")

    gate_metric = str(config['gate_metric'])
    min_mean = float(config['min_cv_r2_mean'])
    min_lcb = float(config['min_cv_r2_lcb'])
    min_models = int(config['min_enabled_models'])
    force_exclude = set(config.get('force_exclude', []))
    threshold_tolerance = 1e-12
    if gate_metric != 'cv_r2_mean':
        raise ValueError("Automatic model eligibility must use cv_r2_mean")
    if not np.isfinite(min_mean) or not np.isfinite(min_lcb):
        raise ValueError("Model-selection thresholds must be finite")
    unknown_exclusions = sorted(force_exclude.difference(models))
    if unknown_exclusions:
        raise ValueError(f"Unknown force_exclude models: {unknown_exclusions}")
    if min_models < 2:
        raise ValueError("min_enabled_models must be at least 2")

    selected_models = {}
    selected_r2_scores = {}
    selection_records = []
    for model_name in MODEL_FILES:
        evaluation = model_artifacts[model_name]['evaluation']
        if gate_metric not in evaluation:
            raise ValueError(
                f"Gate metric {gate_metric!r} is unavailable for {model_name}"
            )
        cv_mean = evaluation['cv_r2_mean']
        cv_std = evaluation['cv_r2_std']
        cv_lcb = cv_mean - cv_std
        reasons = []
        if cv_mean < min_mean - threshold_tolerance:
            reasons.append(f"cv_r2_mean<{min_mean:.4f}")
        if cv_lcb < min_lcb - threshold_tolerance:
            reasons.append(f"cv_r2_lcb<{min_lcb:.4f}")
        if model_name in force_exclude:
            reasons.append('force_exclude')

        selected = not reasons
        metric_value = float(evaluation[gate_metric])
        if selected:
            if not np.isfinite(metric_value) or metric_value <= 0:
                raise ValueError(
                    f"Selected model {model_name} has invalid {gate_metric}: "
                    f"{metric_value}"
                )
            selected_models[model_name] = models[model_name]
            selected_r2_scores[model_name] = metric_value

        artifact = model_artifacts[model_name]
        selection_records.append({
            'Model': model_name,
            'CV_R2_Mean': cv_mean,
            'CV_R2_Std': cv_std,
            'CV_R2_LCB': cv_lcb,
            'Test_R2': evaluation['test_r2'],
            'Gate_Metric': gate_metric,
            'Gate_Metric_Value': metric_value,
            'Selected': selected,
            'Reason': 'selected' if selected else ';'.join(reasons),
            'Model_Path': _portable_project_path(artifact['model_path']),
            'Feature_Config_Path': _portable_project_path(
                artifact['feature_config_path']
            ),
            'Score_Version': artifact['score_version'],
            'Campaign_Round': artifact['campaign_round'],
            'Split_SHA256': artifact['split_manifest']['split_sha256'],
        })

    return selected_models, selected_r2_scores, selection_records, min_models


def save_model_selection(selection_records, output_dir):
    """Persist every pass/fail decision before the GA eligibility gate."""
    destination = Path(output_dir) / 'model_selection.csv'
    pd.DataFrame(selection_records).to_csv(destination, index=False, encoding='utf-8')
    return destination


def validate_model_score_versions(model_score_info):
    """Require one explicit Score version and campaign across all models."""
    logger = logging.getLogger(__name__)
    version_models = {}
    campaign_models = {}

    for model_name, score_info in model_score_info.items():
        version = score_info.get('score_version')
        if version is None or str(version).strip() == '':
            raise ValueError(f"Model {model_name} has no Score version")
        version_models.setdefault(str(version), []).append(model_name)
        campaign_round = score_info.get('campaign_round')
        if campaign_round is None or str(campaign_round).strip() == '':
            raise ValueError(f"Model {model_name} has no campaign round")
        campaign_models.setdefault(str(campaign_round), []).append(model_name)

    if len(version_models) > 1:
        details = '; '.join(
            f"{version}: {sorted(model_names)}"
            for version, model_names in sorted(version_models.items())
        )
        raise ValueError(
            "Refusing to mix models trained with different Score versions: "
            f"{details}"
        )

    if not version_models:
        raise ValueError("No model Score provenance was loaded")
    if len(campaign_models) > 1:
        details = '; '.join(
            f"{campaign}: {sorted(model_names)}"
            for campaign, model_names in sorted(campaign_models.items())
        )
        raise ValueError(
            "Refusing to mix models trained for different campaigns: " + details
        )

    score_version = next(iter(version_models))
    source_model = version_models[score_version][0]
    logger.info("All models use Score version %s", score_version)
    return model_score_info[source_model].copy()


def load_models(base_dir, model_files, encoder):
    """Load every candidate model and its complete, hash-bound sidecar."""
    logger = logging.getLogger(__name__)
    logger.info("Starting to load candidate models...")

    models = {}
    model_score_info = {}
    model_artifacts = {}

    try:
        for model_name, file_name in model_files.items():
            model_path = find_latest_model(base_dir, file_name)

            config_path = Path(model_path).with_name(
                Path(model_path).stem.replace('_best_model', '_feature_config') + '.json'
            )
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Required feature configuration not found: {config_path}"
                )

            with config_path.open('r', encoding='utf-8') as f:
                feature_config = json.load(f)
            evaluation = evaluation_from_feature_config(
                feature_config,
                model_name,
            )
            if feature_config.get('model_file') != Path(model_path).name:
                raise ValueError(
                    f"Model filename does not match its sidecar: {config_path}"
                )
            expected_model_sha256 = feature_config.get('model_sha256')
            actual_model_sha256 = file_sha256(model_path)
            if expected_model_sha256 != actual_model_sha256:
                raise ValueError(
                    f"Model hash does not match its sidecar: {model_path}"
                )

            full_feature_names = feature_config.get('full_feature_names')
            if full_feature_names != encoder.full_feature_names:
                raise ValueError(
                    f"Full feature order in {config_path} does not match the GA encoder"
                )
            retained_feature_names = feature_config.get('retained_feature_names')
            model_score_info[model_name] = score_info_from_feature_config(
                feature_config
            )
            models[model_name] = load(model_path)
            model_artifacts[model_name] = {
                'model_path': Path(model_path),
                'feature_config_path': config_path,
                'model_sha256': actual_model_sha256,
                'score_version': model_score_info[model_name]['score_version'],
                'campaign_round': model_score_info[model_name]['campaign_round'],
                'evaluation': evaluation,
                'split_manifest': feature_config.get('split_manifest'),
                'cv_manifest': feature_config.get('cv_manifest'),
                'training_data': feature_config.get('training_data'),
                'stacking_data': feature_config.get('stacking_data'),
            }
            logger.info(f"[OK] Loaded model: {model_name} ({model_path})")
            logger.info(
                "[OK] Model %s Score version: %s",
                model_name,
                model_score_info[model_name]['score_version'],
            )
            logger.info(f"[OK] Loaded feature configuration: {config_path}")

            encoder.extract_feature_config_from_model(
                models[model_name], model_name, retained_feature_names
            )

        logger.info(f"Successfully loaded {len(models)}/{len(model_files)} models")
        score_info = validate_model_score_versions(model_score_info)
        validate_model_training_contracts(model_artifacts)
        logger.info("All candidate models share one data, split, and CV contract")
        return models, score_info, model_artifacts

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise


class GeneticOptimizer:
    """Genetic algorithm optimizer."""

    def __init__(self, models, stacking_model, encoder, output_dir=None):
        self.models = models
        self.stacking_model = stacking_model
        self.encoder = encoder
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.history = []

    def fitness(self, samples):
        """Predict Score through the fitted Ridge stack in its input-model order."""
        X = self.encoder.encode_batch(samples)
        predictions = np.column_stack([
            model.predict(self.encoder.align_features(X, name))
            for name, model in self.models.items()
        ])
        scores = self.stacking_model.predict(predictions)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Stacking produced non-finite candidate scores")
        return scores

    def _initialize_population(self, pop_size):
        """Initialize a SMILES-unique population from valid template samples."""
        population = []
        seen_smiles = set()
        max_attempts = pop_size * GA_CONFIG['max_init_attempts']

        for _ in range(max_attempts):
            candidate = generate_valid_sample()
            smiles = assemble(candidate)
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                population.append(candidate)
            if len(population) >= pop_size:
                break

        return population

    def _crossover(self, parent1, parent2):
        """Combine one parent's center with the other's positional arms."""
        center_source, arm_source = (
            (parent1, parent2) if np.random.rand() < 0.5 else (parent2, parent1)
        )
        child = [center_source[0], 'None', 'None', 'None', 'None']
        arity = int(CENTER_TEMPLATES[child[0]]['arity'])
        for position in range(1, arity + 1):
            arm = arm_source[position]
            if arm == 'None':
                arm = center_source[position]
            if arm == 'None':
                arm = random.choice(ARM_NAMES)
            child[position] = arm

        if arity and np.random.rand() < GA_CONFIG['crossover_perturbation']:
            position = np.random.randint(1, arity + 1)
            alternatives = [arm for arm in ARM_NAMES if arm != child[position]]
            child[position] = random.choice(alternatives)

        return child if is_valid(child) else generate_valid_sample()

    def _mutate(self, sample):
        """Replace either the center template or one active arm."""
        original = sample.copy()
        mutated = original.copy()
        arity = int(CENTER_TEMPLATES[mutated[0]]['arity'])

        if np.random.rand() < 0.5:
            centers = [center for center in CENTER_TEMPLATES if center != mutated[0]]
            mutated[0] = random.choice(centers)
            new_arity = int(CENTER_TEMPLATES[mutated[0]]['arity'])
            for position in range(1, 5):
                if position <= new_arity:
                    if mutated[position] == 'None':
                        mutated[position] = random.choice(ARM_NAMES)
                else:
                    mutated[position] = 'None'
        elif arity:
            position = np.random.randint(1, arity + 1)
            alternatives = [arm for arm in ARM_NAMES if arm != mutated[position]]
            mutated[position] = random.choice(alternatives)

        return mutated if is_valid(mutated) else original

    def _select_parents(self, population, scores):
        """Roulette wheel selection."""
        scores = np.array(scores)

        if np.all(scores == scores[0]):
            indices = np.random.randint(0, len(population), size=len(population))
        else:
            norm_scores = scores - scores.min() + 1e-8
            probs = norm_scores / norm_scores.sum()
            indices = np.random.choice(len(population), size=len(population), p=probs)

        return [population[i].copy() for i in indices]

    def optimize(self, pop_size=None, generations=None, mutation_rate=None,
                 elite_ratio=None, early_stop=None):
        """Run genetic algorithm optimization."""
        pop_size = pop_size or GA_CONFIG['pop_size']
        generations = generations or GA_CONFIG['generations']
        mutation_rate = mutation_rate or GA_CONFIG['mutation_rate']
        elite_ratio = elite_ratio or GA_CONFIG['elite_ratio']
        early_stop = early_stop or GA_CONFIG['early_stop']

        self.logger.info(f"""
        === Genetic Algorithm Parameters ===
        Population size: {pop_size}
        Generations: {generations}
        Mutation rate: {mutation_rate}
        Elite ratio: {elite_ratio}
        Early stop: {early_stop}
        === Model Configuration ===
        Number of models: {len(self.models)}
        Fitness: Ridge Stacking predicted Score
        """)

        for model_name in self.models.keys():
            total_features = len(self.encoder.full_feature_names)
            n_features = self.encoder.model_n_features.get(model_name, total_features)
            self.logger.info(
                f"  - {model_name}: "
                f"(using {n_features}/{total_features} features)"
            )

        population = self._initialize_population(pop_size)
        if len(population) < pop_size:
            raise ValueError(f"Only generated {len(population)}/{pop_size} valid initial samples")

        best_score = -np.inf
        no_improve_max = 0
        avg_history = []
        candidate_archive = {}

        for gen in range(generations):
            scores = self.fitness(population)
            for sample in population:
                candidate_archive.setdefault(assemble(sample), sample.copy())
            current_max = np.max(scores)
            current_avg = np.mean(scores)
            avg_history.append(current_avg)

            self.history.append({
                'generation': gen+1,
                'best': current_max,
                'average': current_avg,
                'min': np.min(scores)
            })

            if current_max > best_score:
                best_score = current_max
                no_improve_max = 0
            else:
                no_improve_max += 1

            current_delta = 0
            if len(avg_history) >= 2:
                current_delta = avg_history[-1] - avg_history[-2]

            avg_condition = False
            if len(avg_history) >= 4:
                recent_deltas = [
                    avg_history[-1] - avg_history[-2],
                    avg_history[-2] - avg_history[-3],
                    avg_history[-3] - avg_history[-4]
                ]
                avg_condition = all(d < 0.01 for d in recent_deltas)

            self.logger.info(
                f"Generation {gen+1}/{generations} | "
                f"Max: {current_max:.4f}({no_improve_max}/{early_stop}) | "
                f"Avg: {current_avg:.4f} | "
                f"Delta: {current_delta:.4f}"
            )

            top_10_indices = np.argsort(scores)[-10:][::-1]
            self.logger.info(f"  Top 10 structures in generation {gen+1}:")
            for rank, idx in enumerate(top_10_indices, 1):
                sample = population[idx]
                score = scores[idx]
                self.logger.info(
                    f"    Rank {rank}: SMILES {assemble(sample)} | "
                    f"Center {sample[0]} | Arms {sample[1:]} | Score: {score:.4f}"
                )

            bottom_10_indices = np.argsort(scores)[:10]
            self.logger.info(f"  Bottom 10 structures in generation {gen+1}:")
            for rank, idx in enumerate(bottom_10_indices, 1):
                sample = population[idx]
                score = scores[idx]
                self.logger.info(
                    f"    Rank {rank}: SMILES {assemble(sample)} | "
                    f"Center {sample[0]} | Arms {sample[1:]} | Score: {score:.4f}"
                )

            if no_improve_max >= early_stop and avg_condition:
                self.logger.info(
                    f"Early stopping triggered:\n"
                    f"- Max stagnation: {no_improve_max} generations\n"
                    f"- Recent avg changes: {recent_deltas}\n"
                    f"Current best score: {best_score:.4f}"
                )
                break

            elite_size = int(pop_size * elite_ratio)
            elite_indices = np.argsort(scores)[-elite_size:]
            elites = [population[i].copy() for i in elite_indices]

            parents = self._select_parents(population, scores)

            children = []
            while len(children) < pop_size - elite_size:
                if len(parents) >= 2:
                    p1, p2 = random.sample(parents, 2)
                    child = self._crossover(p1, p2)
                else:
                    child = parents[0].copy()

                if np.random.rand() < mutation_rate:
                    child = self._mutate(child)

                if is_valid(child):
                    children.append(child)

            new_population = elites + children[:pop_size - elite_size]
            unique_pop = []
            seen_pop = set()
            for ind in new_population:
                smiles = assemble(ind)
                if smiles not in seen_pop:
                    seen_pop.add(smiles)
                    unique_pop.append(ind)
            while len(unique_pop) < pop_size:
                candidate = generate_valid_sample()
                smiles = assemble(candidate)
                if smiles not in seen_pop:
                    seen_pop.add(smiles)
                    unique_pop.append(candidate)
            population = unique_pop[:pop_size]

        if self.output_dir and OUTPUT_CONFIG['save_history']:
            history_file = os.path.join(self.output_dir, "optimization_history.csv")
            with open(history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Generation', 'Best', 'Average', 'Min'])
                for record in self.history:
                    writer.writerow([
                        record['generation'],
                        record['best'],
                        record['average'],
                        record['min']
                    ])

        # Include the last offspring population when the generation limit is hit.
        for sample in population:
            candidate_archive.setdefault(assemble(sample), sample.copy())
        archived_items = sorted(candidate_archive.items())
        archived_samples = [sample for _, sample in archived_items]
        final_scores = self.fitness(archived_samples)
        scored_samples = sorted(
            (
                (score, smiles, sample)
                for score, (smiles, sample) in zip(final_scores, archived_items)
            ),
            key=lambda x: (-x[0], x[1])
        )

        if self.output_dir:
            archive_file = Path(self.output_dir) / 'candidate_archive.csv'
            temporary_file = archive_file.with_suffix('.csv.tmp')
            with temporary_file.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    'Rank', 'SMILES', 'Center', 'Arm1', 'Arm2', 'Arm3', 'Arm4',
                    'PredictedScore',
                ])
                for rank, (score, smiles, sample) in enumerate(scored_samples, 1):
                    writer.writerow([rank, smiles, *sample, float(score)])
            temporary_file.replace(archive_file)
        self.logger.info(
            "Selecting top candidates from %d unique historical candidates",
            len(scored_samples),
        )

        unique_samples = []
        seen = set()
        top_n = OUTPUT_CONFIG['top_n_results']

        for score, smiles, sample in scored_samples:
            if smiles not in seen:
                seen.add(smiles)
                unique_samples.append((score, sample))
            if len(unique_samples) >= top_n:
                break

        return [s[1] for s in unique_samples], [s[0] for s in unique_samples]


def plot_optimization_history(history, save_path):
    """Plot optimization process."""
    generations = [r['generation'] for r in history]
    best_scores = [r['best'] for r in history]
    avg_scores = [r['average'] for r in history]

    fig_size = PLOT_CONFIG['figure_size']
    dpi = PLOT_CONFIG['dpi']
    lw = PLOT_CONFIG['line_width']
    ms = PLOT_CONFIG['marker_size']

    plt.figure(figsize=fig_size)

    plt.plot(generations, best_scores,
             marker='o', markersize=ms, linestyle='-',
             color=PLOT_CONFIG['colors']['best'], linewidth=lw, alpha=0.6,
             label='Best Score')

    plt.plot(generations, avg_scores,
             marker='^', markersize=ms, linestyle='--',
             color=PLOT_CONFIG['colors']['average'], linewidth=lw, alpha=0.6,
             label='Average Score')

    plt.title("Optimization Process", fontsize=PLOT_CONFIG['title_fontsize'], pad=10)
    plt.xlabel("Generation", fontsize=PLOT_CONFIG['label_fontsize'])
    plt.ylabel("Score", fontsize=PLOT_CONFIG['label_fontsize'])
    plt.tick_params(axis='x', labelsize=PLOT_CONFIG['tick_fontsize'])
    plt.tick_params(axis='y', labelsize=PLOT_CONFIG['tick_fontsize'])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=PLOT_CONFIG['legend_fontsize'],
               frameon=True, fancybox=False)

    all_scores = best_scores + avg_scores
    y_min, y_max = min(all_scores), max(all_scores)
    y_range = y_max - y_min
    if y_range < 1e-6:
        y_range = 0.1
    padding = 0.05 * y_range
    plt.ylim(y_min - padding, y_max + padding)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_best_score_10gen(history, save_path):
    """Plot best score evolution for first 10 generations."""
    generations = [r['generation'] for r in history if r['generation'] <= 10]
    best_scores = [r['best'] for r in history if r['generation'] <= 10]

    fig_size = (10, 3.5)
    dpi = PLOT_CONFIG['dpi']
    lw = PLOT_CONFIG['line_width']
    ms = PLOT_CONFIG['marker_size']

    plt.figure(figsize=fig_size)

    plt.plot(generations, best_scores,
             marker='o', markersize=ms, linestyle='-',
             color=PLOT_CONFIG['colors']['best'], linewidth=lw, alpha=0.6,
             label='Best Score')

    plt.xlabel("Generation", fontsize=PLOT_CONFIG['label_fontsize'])
    plt.ylabel("Score", fontsize=PLOT_CONFIG['label_fontsize'])
    plt.tick_params(axis='x', labelsize=PLOT_CONFIG['tick_fontsize'])
    plt.tick_params(axis='y', labelsize=PLOT_CONFIG['tick_fontsize'])
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', fontsize=PLOT_CONFIG['legend_fontsize'],
               frameon=True, fancybox=False)

    plt.xlim(0, 10)

    if best_scores:
        y_min = min(best_scores)
        y_max = max(best_scores)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.25 * y_range, y_max + 0.25 * y_range)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_top_structures(output_dir):
    """Draw the first ten saved candidates in CSV rank order."""
    table = pd.read_csv(Path(output_dir) / 'optimal_samples.csv').head(10)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in table['SMILES']]
    legends = [f"Top {row.Rank} | Score: {row.PredictedScore:.4f}"
               for row in table.itertuples(index=False)]
    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.legendFontSize = 26
    options.legendFraction = 0.12
    png = Draw.MolsToGridImage(
        molecules, molsPerRow=2, subImgSize=(800, 480), legends=legends,
        drawOptions=options, useSVG=False, returnPNG=True,
    )
    path = Path(output_dir) / 'top10_structures.png'
    path.write_bytes(png)
    logging.info('Saved Top 10 structures: %s', path)


def save_results(samples, scores, output_dir):
    """Save optimization results to CSV."""
    result_file = os.path.join(output_dir, "optimal_samples.csv")

    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'SMILES', 'Center', 'Arm1', 'Arm2', 'Arm3', 'Arm4',
            'PredictedScore'
        ])

        for i, (sample, score) in enumerate(zip(samples, scores), 1):
            writer.writerow([i, assemble(sample), *sample, float(score)])

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Run campaign-bound ensemble genetic optimization."
    )
    add_campaign_arguments(parser)
    args = parser.parse_args()
    campaign_round = resolve_campaign_round(args.campaign_round)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            BASE_OUTPUT_DIR,
            campaign_directory_name(campaign_round),
            f"GA_{timestamp}",
        )
        os.makedirs(output_dir, exist_ok=True)

        logger = setup_logger(output_dir)
        logger.info("=== Program Started ===")
        logger.info("Campaign round: %s", campaign_round)
        pd.DataFrame([
            {'SMILES': smiles, 'Formula': formula,
             'Reason': 'Donor environment unsupported by the 32-feature contract'}
            for smiles, formula in EXCLUDED_STRUCTURES.items()
        ]).to_csv(Path(output_dir) / 'excluded_structures.csv', index=False)

        logger.info("\n=== Configuration Summary ===")
        logger.info(f"Random seed: {RANDOM_SEED}")
        logger.info(f"Population size: {GA_CONFIG['pop_size']}")
        logger.info(f"Generations: {GA_CONFIG['generations']}")
        logger.info(f"Mutation rate: {GA_CONFIG['mutation_rate']}")
        logger.info(
            "Automatic model gate: CV R2 >= %.4f and CV R2-Std >= %.4f",
            MODEL_SELECTION_CONFIG['min_cv_r2_mean'],
            MODEL_SELECTION_CONFIG['min_cv_r2_lcb'],
        )
        logger.info(
            "Gate metric: %s; minimum selected models: %d",
            MODEL_SELECTION_CONFIG['gate_metric'],
            MODEL_SELECTION_CONFIG['min_enabled_models'],
        )

        encoder = CategoryEncoder()
        logger.info("\n=== Feature Configuration ===")
        logger.info(f"Full feature dimensions: {len(encoder.full_feature_names)}")

        candidate_models, score_info, candidate_artifacts = load_models(
            MODEL_OUTPUT_DIR / campaign_directory_name(campaign_round),
            MODEL_FILES,
            encoder,
        )

        if score_info['campaign_round'] != campaign_round:
            raise ValueError(
                f"Loaded models are bound to {score_info['campaign_round']}, "
                f"not {campaign_round}"
            )

        write_score_provenance(output_dir, score_info)
        logger.info(
            "Recorded model Score provenance "
            f"(score_version={score_info['score_version']})"
        )

        (
            models,
            r2_scores,
            selection_records,
            min_enabled_models,
        ) = select_models_by_performance(
            candidate_models,
            candidate_artifacts,
            MODEL_SELECTION_CONFIG,
        )
        selection_path = save_model_selection(selection_records, output_dir)
        logger.info("Model selection decisions saved to %s", selection_path)
        logger.info("\n=== Automatic Model Selection ===")
        for record in selection_records:
            logger.info(
                "%s: CV R2=%.4f, CV R2-Std=%.4f, Test R2=%.4f, "
                "selected=%s (%s)",
                record['Model'],
                record['CV_R2_Mean'],
                record['CV_R2_LCB'],
                record['Test_R2'],
                record['Selected'],
                record['Reason'],
            )

        if len(models) < min_enabled_models:
            raise ValueError(
                f"Only {len(models)} models passed the automatic gate; "
                f"at least {min_enabled_models} are required. See {selection_path}."
            )

        model_artifacts = {
            model_name: candidate_artifacts[model_name]
            for model_name in models
        }
        stacking_model = train_stacking(models, model_artifacts, encoder, output_dir)
        optimizer = GeneticOptimizer(
            models=models, stacking_model=stacking_model,
            encoder=encoder, output_dir=output_dir,
        )

        best_samples, final_scores = optimizer.optimize()

        save_results(best_samples, final_scores, output_dir)

        plot_top_structures(output_dir)

        if OUTPUT_CONFIG['save_plot']:
            plot_optimization_history(
                optimizer.history,
                os.path.join(output_dir, 'GA_plot.png')
            )
            plot_best_score_10gen(
                optimizer.history,
                os.path.join(output_dir, 'GA_best_score_10gen.png')
            )

        logger.info(f"\n=== Top {len(best_samples)} Optimal Solutions ===")
        for i, (sample, score) in enumerate(zip(best_samples, final_scores), 1):
            logger.info(
                f"Top {i}: "
                f"SMILES {assemble(sample)} | "
                f"Center {sample[0]} | "
                f"Arms {sample[1:]} | "
                f"Score: {score:.4f}"
            )

        logger.info("\n=== Program Completed Successfully ===")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Models used: {list(models.keys())}")

    except Exception:
        logger.exception("Program terminated with error!")
        raise
