import argparse
import gc
import hashlib
import math
import multiprocessing
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import PackageNotFoundError, version as package_version

from sr_efficiency import (
    ChunkEarlyStopper,
    append_json_line,
    atomic_write_json,
    conservative_fold_prune,
    file_sha256,
    feasible_frontier_stats,
    frontier_improved,
    payload_sha256,
    resolve_fold_execution_layout,
    resolve_julia_threads,
    sample_unique_configs,
    shutdown_process_pool_bounded,
    used_features,
    update_refinement_frontier,
    validation_frontier_improved,
    validation_frontier_stats,
)

# Personal-PC runtime configuration.  PySR 1.5.9 starts Julia through
# JuliaCall, whose command-line thread option takes precedence over
# JULIA_NUM_THREADS.  Set both values before importing PySR.  The default is
# 32 logical threads for the project's i9-13900KF and is capped by the logical
# processors visible to native Windows or WSL.  Users can override it without
# editing the file by setting SR_JULIA_THREADS.  BLAS helpers remain single
# threaded, so PySR owns the machine-level parallelism.
def _environment_flag(name, default=False):
    """Read a strict boolean environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; "
        f"received {raw_value!r}"
    )


def _positive_environment_integer(name, raw_value, *, maximum):
    """Parse a bounded positive workstation thread setting."""
    try:
        value = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; received {raw_value!r}") from exc
    if value < 1 or value > maximum:
        raise ValueError(
            f"{name} must be between 1 and {maximum}; received {raw_value!r}"
        )
    return value


LOGICAL_PROCESSORS = os.cpu_count() or 1
STRICT_DETERMINISTIC_SEARCH = _environment_flag("SR_DETERMINISTIC", default=False)
JULIA_THREADS = resolve_julia_threads(
    requested=("1" if STRICT_DETERMINISTIC_SEARCH else os.environ.get("SR_JULIA_THREADS")),
    logical_processors=LOGICAL_PROCESSORS,
)
os.environ["PYTHON_JULIACALL_THREADS"] = str(JULIA_THREADS)
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ["JULIA_NUM_THREADS"] = str(JULIA_THREADS)

# PySR owns the machine-level parallelism.  Keeping BLAS helpers single
# threaded prevents every Julia/Python worker from creating another thread
# pool on a personal workstation.  SR_BLAS_THREADS remains an explicit escape
# hatch for controlled thread-layout measurements.
BLAS_THREADS = _positive_environment_integer(
    "SR_BLAS_THREADS",
    os.environ.get("SR_BLAS_THREADS", "1"),
    maximum=LOGICAL_PROCESSORS,
)
for _thread_env in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = str(BLAS_THREADS)

from sr_validation import (
    inherit_cv_winner_candidates,
    analyze_term_budget,
    build_prediction_records,
    build_split_assignments,
    evaluate_pareto_candidates,
    prepare_fold_scaled_data,
    review_final_candidates,
    select_config_one_standard_error,
    select_training_candidate,
    select_validation_candidate,
    stratified_duplex_indices,
)
from sr_reporting import (
    bootstrap_fixed_structure_coefficients,
    evaluate_expression,
    parameterize_floating_constants,
    parse_source_expression,
    standardized_to_raw_expression,
    write_candidate_comparison,
    export_candidate_results,
)


import pandas as pd
import numpy as np
import sympy as sp
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import traceback
import pickle
from datetime import datetime
from pathlib import Path
from time import monotonic, process_time
import logging
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_preparation import (
    prepare_symbolic_regression_data,
    write_score_provenance,
)
from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from project_paths import (
    SYMBOLIC_REGRESSION_OUTPUT_DIR,
)

# One PySR fit uses Julia multithreading.  Independent CV folds can additionally
# use isolated spawned processes, with a bounded aggregate thread budget.
N_CPUS = JULIA_THREADS
PARALLELISM = 'serial' if STRICT_DETERMINISTIC_SEARCH else 'multithreading'
N_PROCS = None
FOLD_EXECUTION_LAYOUT = resolve_fold_execution_layout(
    logical_processors=LOGICAL_PROCESSORS,
    julia_threads=JULIA_THREADS,
    requested_workers=os.environ.get('SR_FOLD_WORKERS'),
    requested_threads_per_worker=os.environ.get('SR_FOLD_WORKER_THREADS'),
    deterministic=STRICT_DETERMINISTIC_SEARCH,
)
FOLD_WORKERS = FOLD_EXECUTION_LAYOUT.worker_count
FOLD_WORKER_THREADS = FOLD_EXECUTION_LAYOUT.threads_per_worker

try:
    _PYSR_VERSION = package_version('pysr')
except PackageNotFoundError:
    _PYSR_VERSION = 'not-installed'
_ACTUAL_JULIA_THREADS = None


def initialize_fold_worker_runtime(threads_per_worker):
    """Configure a spawned fold worker before its first lazy PySR import."""
    global JULIA_THREADS, N_CPUS, _ACTUAL_JULIA_THREADS

    resolved = int(threads_per_worker)
    if resolved < 1:
        raise ValueError('threads_per_worker must be positive')
    JULIA_THREADS = resolved
    N_CPUS = resolved
    _ACTUAL_JULIA_THREADS = None
    os.environ['SR_JULIA_THREADS'] = str(resolved)
    os.environ['PYTHON_JULIACALL_THREADS'] = str(resolved)
    os.environ['JULIA_NUM_THREADS'] = str(resolved)
    for thread_env in (
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[thread_env] = str(BLAS_THREADS)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    )


def create_pysr_regressor(**configuration):
    """Import PySR lazily and verify Julia's real thread count once."""
    global _PYSR_VERSION, _ACTUAL_JULIA_THREADS

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(
                r"PYTHON_JULIACALL_THREADS environment variable is set to "
                r"something other than 'auto', so PySR was not able to set it\..*"
            ),
            category=UserWarning,
            module=r'pysr\.julia_import',
        )
        import pysr

    _PYSR_VERSION = getattr(pysr, '__version__', 'unknown')
    if _ACTUAL_JULIA_THREADS is None:
        _ACTUAL_JULIA_THREADS = int(pysr.jl.seval("Threads.nthreads()"))
        logger = logging.getLogger(__name__)
        logger.info(
            "Julia runtime threads: %d (configured: %d)",
            _ACTUAL_JULIA_THREADS,
            JULIA_THREADS,
        )
        if _ACTUAL_JULIA_THREADS != JULIA_THREADS:
            logger.warning(
                "Julia runtime thread count does not match the configured value; "
                "restart Python after checking JuliaCall environment variables"
            )
    return pysr.PySRRegressor(**configuration)


def fit_pysr_regressor(model, X, y, *, variable_names):
    """Fit PySR while suppressing only known repeated notices."""
    with warnings.catch_warnings():
        if not STRICT_DETERMINISTIC_SEARCH:
            warnings.filterwarnings(
                'ignore',
                message=(
                    r"Note: Setting `random_state` without also setting "
                    r"`deterministic=True` and `parallelism='serial'` will "
                    r"result in non-deterministic searches\."
                ),
                category=UserWarning,
                module=r'pysr\.sr',
            )
        warnings.filterwarnings(
            'ignore',
            message=(
                r"Note: Using a large maxsize for the equation search will be "
                r"exponentially slower and use significant memory\."
            ),
            category=UserWarning,
            module=r'pysr\.sr',
        )
        return model.fit(X, y, variable_names=variable_names)


def symbolic_regression_runtime_metadata():
    """Return the resolved workstation runtime contract for provenance."""
    return {
        'logical_processors': LOGICAL_PROCESSORS,
        'configured_julia_threads': JULIA_THREADS,
        'actual_julia_threads': _ACTUAL_JULIA_THREADS,
        'blas_threads': BLAS_THREADS,
        'parallelism': PARALLELISM,
        'strict_deterministic': STRICT_DETERMINISTIC_SEARCH,
        'default_thread_policy': 'all_visible_logical_processors_up_to_32',
        'fold_worker_count': FOLD_WORKERS,
        'threads_per_fold_worker': FOLD_WORKER_THREADS,
        'aggregate_fold_worker_threads': FOLD_EXECUTION_LAYOUT.aggregate_threads,
        'fold_worker_layout_source': FOLD_EXECUTION_LAYOUT.source,
        'random_state': RANDOM_STATE,
        'pysr_version': _PYSR_VERSION,
        'custom_objective_parameter': 'loss_function',
        'power_plan_management': 'not_modified_by_pipeline',
    }


# File path configuration
BASE_OUTPUT_DIR = SYMBOLIC_REGRESSION_OUTPUT_DIR
DERIVED_DATA_FILENAME = 'data_encoded.csv'

# Data and feature configuration
FEATURE_COLUMNS = [
    'atom_N', 'Band_gap', 'fai_min', 'Dipol_moment', 'HOMO', 'TPSA',
]
REQUIRED_FEATURE_COUNT = 6
MAX_EXCESS_COMPLEXITY = 5
DESCRIPTOR_COLUMNS = {
    'MolWt': 'Molecular_weight', 'N_lig': 'Denticity_of_ligand',
    'RingNum': 'Ring_number', 'gap': 'Band_gap', 'μ': 'Dipol_moment',
    'MolVol': 'Molecular_Volume',
    **{name: name for name in (
        'HOMO', 'LUMO', 'fai_max', 'fai_min', 'TPSA',
        'atom_N', 'atom_O', 'atom_S', 'atom_Cl',
    )},
}
TARGET_COLUMN = 'Score'


def configure_feature_pool(features=None, selected_count=REQUIRED_FEATURE_COUNT):
    """Normalize descriptor aliases and validate the requested formula size."""
    names = [
        DESCRIPTOR_COLUMNS.get(name, name)
        for name in (FEATURE_COLUMNS if features is None else features)
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError('SR candidate descriptors must be nonempty and unique')
    unsupported = set(names) - set(DESCRIPTOR_COLUMNS.values())
    if unsupported:
        raise ValueError(f'Unsupported SR descriptors: {sorted(unsupported)}')
    if type(selected_count) is not int or not 1 <= selected_count <= len(names):
        raise ValueError('selected-count must be an integer between 1 and the candidate count')
    return names

# Raw descriptor units confirmed against the original workbook's Sheet1 main
# table.  The SR search itself uses dimensionless Z-scores; these source units
# apply when evaluating the de-standardized raw-feature equation.
FEATURE_METADATA = {
    'Denticity_of_ligand': {
        'quantity': 'ligand denticity count',
        'source_unit': '1 (count)',
    },
    'atom_N': {
        'quantity': 'number of coordinating nitrogen donor atoms',
        'source_unit': '1 (count)',
    },
    'fai_min': {
        'quantity': 'minimum molecular electrostatic potential (phi_min)',
        'source_unit': 'kcal mol^-1',
    },
    'Band_gap': {
        'quantity': 'HOMO-LUMO band gap',
        'source_unit': 'eV',
    },
    'Dipol_moment': {
        'quantity': 'dipole moment',
        'source_unit': 'Debye',
    },
    'HOMO': {
        'quantity': 'HOMO energy',
        'source_unit': 'eV',
    },
    'Ring_number': {
        'quantity': 'number of simultaneously coordinating heterocycles',
        'source_unit': '1 (count)',
    },
    'Molecular_weight': {'quantity': 'molecular weight', 'source_unit': 'g mol^-1'},
    'Molecular_Volume': {'quantity': 'molecular volume', 'source_unit': 'Bohr^3'},
    'LUMO': {'quantity': 'LUMO energy', 'source_unit': 'eV'},
    'fai_max': {
        'quantity': 'maximum molecular electrostatic potential (phi_max)',
        'source_unit': 'kcal mol^-1',
    },
    'TPSA': {'quantity': 'topological polar surface area', 'source_unit': 'angstrom^2'},
    'atom_O': {'quantity': 'number of coordinating O donor atoms', 'source_unit': '1 (count)'},
    'atom_S': {'quantity': 'number of coordinating S donor atoms', 'source_unit': '1 (count)'},
    'atom_Cl': {'quantity': 'number of coordinating Cl donor atoms', 'source_unit': '1 (count)'},
}


TARGET_METADATA = {
    'quantity': 'calibrated Score = z_PLQY - z_XRD',
    'source_unit': '1 (dimensionless)',
    'standardized_again_by_symbolic_regression': False,
}
METRIC_DEFINITIONS = {
    'r2': '1 - sum((y-y_hat)^2) / sum((y-mean(y))^2); larger is better',
    'rmse': 'sqrt(mean((y-y_hat)^2)); reported in Score units; smaller is better',
    'mae': 'mean(abs(y-y_hat)); reported in Score units; smaller is better',
    'mape': 'not reported because Score crosses zero',
    'cv_scope': 'top-level cv_* metrics describe configuration search; '
                'final_candidate_cv describes conditional fixed-structure selection',
}

# Data splitting configuration
TEST_SIZE = 0.2
SPLIT_METRIC = 'mahalanobis'
SPLIT_N_STRATA = 4

# Cross-validation configuration
CV_METHOD = 'repeated_stratified'
CV_N_SPLITS = 4
CV_N_REPEATS = 2
CV_N_STRATA = 4
CV_SHUFFLE = True
RIDGE_BASELINE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Symbolic regression hyperparameter search space
RANDOM_STATE = 42
N_RANDOM_SEARCHES = 16
FINAL_REVIEW_N_REPEATS = 5
FINAL_REVIEW_RANDOM_STATE = 2026


def _bounded_environment_integer(name, default, minimum, maximum):
    """Read an explicitly bounded positive integer runtime setting."""
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw_value!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, received {value}"
        )
    return value


COEFFICIENT_UNCERTAINTY_CONFIG = {
    'bootstrap_samples': _bounded_environment_integer(
        'SR_COEFFICIENT_BOOTSTRAPS', 200, 20, 2000
    ),
    'confidence_level': 0.95,
    'random_state': RANDOM_STATE + 30000,
    'max_nfev_per_sample': 300,
    'max_wall_seconds': _bounded_environment_integer(
        'SR_COEFFICIENT_BOOTSTRAP_TIMEOUT_SECONDS', 300, 10, 3600
    ),
}

param_grid = {
    'populations': [30],
    'population_size': [150],
    'niterations': [600],
    'binary_operators': [
        ['+', '-', '*'],
    ],
    'unary_operators': [
        ['square', 'sqrt', 'exp'],
    ],
    # A fully coefficiented six-linear + six-direct-square additive model is
    # approximately complexity 55 with PySR's default node accounting.  Every
    # sampled configuration therefore has enough structural room for the
    # user's per-feature linear/nonlinear allowance.
    'maxsize': [55, 60, 65, 70],
    'maxdepth': [7, 8],
    # Search loss must not penalize an otherwise useful direct quadratic term.
    # Interpretability pressure is applied later with the explicit term-budget
    # metric, not PySR's raw node count.
    'parsimony': [0],
    # Frequency balancing explores a range of tree sizes; it is not used as the
    # scientific candidate-selection score.  The latter is the explicit
    # validation/excess-term rule recorded below.
    'use_frequency': [True],
    'use_frequency_in_tournament': [True],
    # Select variables jointly with the formula; do not invoke PySR's
    # random-forest prefilter separately on every warm-start chunk.
    'constraints': [
        {
            '*': (-1, 3),
            # All three unary operators have the same bounded argument size.
            'square': 6, 'sqrt': 6, 'exp': 6,
        }
    ],
    'nested_constraints': [
        {
            'square': {'square': 1, 'sqrt': 1, 'exp': 1},
            'sqrt': {'square': 1, 'sqrt': 1, 'exp': 1},
            'exp': {'square': 1, 'sqrt': 1, 'exp': 1},
        }
    ],
    'tournament_selection_n': [10],
    'tournament_selection_p': [0.982],
    'weight_add_node': [3, 4],
    'weight_insert_node': [0.05],
    'timeout_in_seconds': [500],
}

# Intermediate expressions use at most the requested number of variables.
CUSTOM_LOSS_TEMPLATE = """
function custom_full_objective(tree, dataset::Dataset{T}, options) where {T<:Real}
__EXCESS_COMPLEXITY_CHECK__
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return T(Inf)
    end

    mse = sum((prediction .- dataset.y).^2) / length(dataset.y)

    n_features = size(dataset.X, 1)
    used_features = zeros(Bool, n_features)

    function check_node!(node)
        if node.degree == 0
            if !node.constant
                used_features[node.feature] = true
            end
        else
            check_node!(node.l)
            if node.degree == 2
                check_node!(node.r)
            end
        end
    end

    check_node!(tree)

    n_used = sum(used_features)
    if n_used > __SELECTED_COUNT__
        return T(Inf)
    end
    n_missing = __SELECTED_COUNT__ - n_used

    if n_missing > 0
        penalty_multiplier = T(10.0) ^ n_missing
        return mse * penalty_multiplier
    else
        return mse
    end
end
"""

_EXCESS_COMPLEXITY_CHECK_JULIA = """
    # Expanded sums use opaque atoms for real roots/exp and powers for squares.
    # The independent Python audit also combines equivalent symbolic forms.
    atom_features = Dict{String,Vector{Int}}()
    atom_operations = Dict{String,Int}()
    atom_direct = Dict{String,Bool}()
    atom_kind = Dict{String,String}()
    atom_argument = Dict{String,Dict{Tuple,Float64}}()
    variable_atoms = Set{String}()

    function add_term!(polynomial, factors, coefficient)
        value = get(polynomial, factors, 0.0) + coefficient
        if iszero(value)
            delete!(polynomial, factors)
        else
            polynomial[factors] = value
        end
    end

    function multiply_polynomials(left, right)
        result = Dict{Tuple,Float64}()
        for (a, ac) in left, (b, bc) in right
            factors = sort!(collect((a..., b...)))
            # sqrt(q)^2=q and exp(p)*exp(q)=exp(p+q), before budgeting.
            repeated_roots = [a for a in unique(factors) if get(atom_kind, a, "") == "sqrt" && count(==(a), factors) > 1]
            exponentials = [a for a in factors if get(atom_kind, a, "") == "exp"]
            normalized = nothing
            if !isempty(repeated_roots)
                root = first(repeated_roots)
                deleteat!(factors, findfirst(==(root), factors))
                deleteat!(factors, findfirst(==(root), factors))
                normalized = multiply_polynomials(Dict{Tuple,Float64}(Tuple(factors) => ac * bc), atom_argument[root])
                normalized === nothing && return nothing
            elseif length(exponentials) > 1
                argument = Dict{Tuple,Float64}()
                for atom in exponentials, (key, coefficient) in atom_argument[atom]
                    add_term!(argument, key, coefficient)
                end
                rest = Tuple(a for a in factors if get(atom_kind, a, "") != "exp")
                transformed = unary_terms("exp", argument)
                transformed === nothing && return nothing
                normalized = multiply_polynomials(Dict{Tuple,Float64}(rest => ac * bc), transformed)
            end
            if normalized === nothing
                add_term!(result, Tuple(factors), ac * bc)
            else
                for (key, coefficient) in normalized
                    add_term!(result, key, coefficient)
                end
            end
        end
        return result
    end

    function term_info(factors)
        atoms = unique(collect(factors))
        features = sort!(unique(vcat([atom_features[a] for a in atoms]...)))
        powers = [count(==(a), factors) for a in atoms]
        operations = sum(atom_operations[a] for a in atoms)
        operations += count(>(1), powers) + max(0, length(atoms) - 1)
        return atoms, features, powers, operations
    end

    function unary_terms(name, left)
        if all(isempty, keys(left))
            argument = get(left, (), 0.0)
            name == "sqrt" && argument < 0 && return nothing
            value = name == "sqrt" ? sqrt(argument) : exp(argument)
            isfinite(value) || return nothing
            result = Dict{Tuple,Float64}()
            add_term!(result, (), value)
            return result
        end
        coefficient = 1.0
        if name == "sqrt"
            key = first(sort!(collect(keys(left)); by=repr))
            factor = abs(left[key])
            if factor != 1
                left = Dict{Tuple,Float64}(key => value / factor for (key, value) in left)
                coefficient = sqrt(factor)
            end
        end
        features = Int[]
        operations = 1 + max(0, length(left) - 1)
        operations += Int(length(left) == 1 && first(values(left)) < 0)
        affine = true
        for (factors, value) in left
            isempty(factors) && continue
            atoms, used, powers, count_ops = term_info(factors)
            append!(features, used)
            operations += count_ops + Int(abs(value) != 1)
            affine &= length(factors) == 1 && first(factors) in variable_atoms
        end
        features = sort!(unique(features))
        signature = join(sort!([repr((factors, value)) for (factors, value) in left]), ";")
        atom = name * "(" * signature * ")"
        atom_features[atom] = features
        atom_operations[atom] = operations
        atom_direct[atom] = affine && length(features) == 1
        atom_kind[atom] = name
        atom_argument[atom] = copy(left)
        result = Dict{Tuple,Float64}()
        add_term!(result, (atom,), coefficient)
        return result
    end


    function expand_terms(node)
        if node.degree == 0
            result = Dict{Tuple,Float64}()
            if node.constant
                isfinite(node.val) || return nothing
                add_term!(result, (), Float64(node.val))
            else
                atom = "v" * string(node.feature)
                atom_features[atom] = [Int(node.feature)]
                atom_operations[atom] = 0
                atom_direct[atom] = false
                push!(variable_atoms, atom)
                add_term!(result, (atom,), 1.0)
            end
            return result
        end
        left = expand_terms(node.l)
        left === nothing && return nothing
        if node.degree == 1
            operation = options.operators.unaops[node.op]
            operation === square && return multiply_polynomials(left, left)
            name = String(nameof(operation))
            name in ("sqrt", "safe_sqrt", "exp") || return nothing
            name == "safe_sqrt" && (name = "sqrt")
            return unary_terms(name, left)
        end
        node.degree == 2 || return nothing
        right = expand_terms(node.r)
        right === nothing && return nothing
        operation = options.operators.binops[node.op]
        if operation === (*)
            return multiply_polynomials(left, right)
        elseif operation === (+) || operation === (-)
            for (factors, coefficient) in right
                add_term!(left, factors, operation === (+) ? coefficient : -coefficient)
            end
            return left
        end
        return nothing
    end

    polynomial = expand_terms(tree)
    polynomial === nothing && return T(Inf)
    excess_complexity = 0
    nonlinear_uses = zeros(Int, size(dataset.X, 1))
    for factors in keys(polynomial)
        isempty(factors) && continue
        atoms, features, powers, operation_count = term_info(factors)
        active_features = length(features)
        if length(factors) == 1 && first(factors) in variable_atoms
            continue
        end
        for feature in features
            nonlinear_uses[feature] += 1
        end
        direct_square = length(factors) == 2 && length(atoms) == 1 && first(atoms) in variable_atoms
        direct_unary = length(factors) == 1 && atom_direct[first(atoms)]
        direct_pair = length(factors) == 2 && length(atoms) == 2 && all(a -> a in variable_atoms, atoms)
        if !(direct_square || direct_unary || direct_pair)
            degree_excess = length(atoms) == 1 && first(atoms) in variable_atoms ? max(0, first(powers) - 2) : 0
            excess_complexity += max(1, operation_count) + max(0, active_features - 1) + degree_excess
        end
        excess_complexity > __EXCESS_COMPLEXITY_CEILING__ && return T(Inf)
    end
    excess_complexity += sum(max(0, uses - 1) for uses in nonlinear_uses)
    excess_complexity > __EXCESS_COMPLEXITY_CEILING__ && return T(Inf)
"""



def make_loss_function(selected_count, excess_complexity_ceiling=None):
    """Build the feature-count objective with an optional shared term budget."""
    budget_check = ''
    if excess_complexity_ceiling is not None:
        if (
            isinstance(excess_complexity_ceiling, bool)
            or int(excess_complexity_ceiling) != excess_complexity_ceiling
            or excess_complexity_ceiling < 0
        ):
            raise ValueError('excess_complexity_ceiling must be a nonnegative integer')
        budget_check = _EXCESS_COMPLEXITY_CHECK_JULIA.replace(
            '__EXCESS_COMPLEXITY_CEILING__', str(int(excess_complexity_ceiling))
        )
    return (
        CUSTOM_LOSS_TEMPLATE
        .replace('__SELECTED_COUNT__', str(selected_count))
        .replace('__EXCESS_COMPLEXITY_CHECK__', budget_check)
    )


CUSTOM_LOSS_JULIA = make_loss_function(REQUIRED_FEATURE_COUNT, MAX_EXCESS_COMPLEXITY)
param_grid['loss_function'] = [CUSTOM_LOSS_JULIA]

# Warm-start budgets and stopping tolerances.
SEARCH_EXECUTION_CONFIG = {
    'chunk_iterations': 100,
    'minimum_iterations': 300,
    'plateau_patience': 3,
    'minimum_score_improvement': 0.001,
    'minimum_loss_relative_improvement': 0.001,
    'validation_bootstrap_samples': 200,
    'enable_plateau_stop': True,
    # Mean-only racing can discard a simpler configuration eligible under one-SE.
    'enable_config_racing': False,
    'minimum_folds_before_pruning': 4,
    'pruning_safety_margin': 0.01,
}

# A JuliaCall worker is allowed to exit normally first.  If its embedded Julia
# finalizer stalls, the already-idle worker is cleaned up after a hard bound so
# the parent can continue to refinement.  This changes lifecycle handling only.
FOLD_SHUTDOWN_CONFIG = {
    'graceful_wait_seconds': _bounded_environment_integer(
        'SR_FOLD_SHUTDOWN_GRACE_SECONDS', 60, 1, 600
    ),
    'terminate_wait_seconds': 15,
    'kill_wait_seconds': 5,
}

# Refinement training configuration
ENABLE_REFINEMENT = True

REFINEMENT_CONFIG = {
    'populations': 60,
    'population_size': 300,
    'niterations': 3000,
    'timeout_in_seconds': _bounded_environment_integer(
        'SR_REFINEMENT_TIMEOUT_SECONDS', 7200, 60, 43200
    ),
    'chunk_iterations': 25,
    'minimum_iterations': 200,
    'plateau_patience': 4,
    'window_iterations': 100,
    'minimum_loss_relative_improvement': 0.0005,
    'similar_loss_relative_tolerance': 0.001,
    'enable_plateau_stop': True,
}

# Visualization configuration
plt.rcParams['font.family'] = 'Arial'
FIGURE_DPI = 600

# Core utility functions
def stratified_duplex_split(
    X,
    y,
    test_size=0.2,
    n_strata=4,
    metric='mahalanobis',
    random_state=RANDOM_STATE,
    return_indices=False,
):
    """Create the deterministic outer split and optionally return row indices."""
    X_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float)
    train_idx, test_idx = stratified_duplex_indices(
        X_values,
        y_values,
        test_size=test_size,
        n_strata=n_strata,
        metric=metric,
        random_state=random_state,
    )
    logging.info(
        "Stratified DUPLEX split completed: %d train, %d test",
        len(train_idx),
        len(test_idx),
    )
    partitions = (
        X_values[train_idx],
        X_values[test_idx],
        y_values[train_idx],
        y_values[test_idx],
    )
    if return_indices:
        return (*partitions, train_idx, test_idx)
    return partitions

def create_cv(X, y, n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
              n_strata=CV_N_STRATA, random_state=RANDOM_STATE, method=CV_METHOD):
    """Create cross-validation splitter with stratification."""
    try:
        y_bins = pd.qcut(y, q=n_strata, duplicates='drop', labels=False)
        actual_strata = len(np.unique(y_bins))
        logging.info(f"Created {actual_strata} strata for cross-validation (requested {n_strata})")
    except ValueError as e:
        logging.warning(f"Failed to create {n_strata} bins, trying with 3: {e}")
        y_bins = pd.qcut(y, q=3, duplicates='drop', labels=False)
        actual_strata = len(np.unique(y_bins))

    if method == 'repeated_stratified':
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
        logging.info(f"Using Repeated Stratified K-Fold CV: {n_splits} splits × {n_repeats} repeats = {n_splits * n_repeats} total folds")
    else:
        cv = KFold(
            n_splits=n_splits,
            shuffle=CV_SHUFFLE,
            random_state=random_state
        )
        logging.info(f"Using K-Fold CV: {n_splits} splits")

    return cv.split(X, y_bins)

class ResumeCheckpointError(RuntimeError):
    """Raised when durable SR state is missing, incompatible, or corrupt."""


class SymbolicRegressionAnalysis:
    """Symbolic regression analysis pipeline with hyperparameter search and refinement."""

    required_feature_count = REQUIRED_FEATURE_COUNT
    max_excess_complexity = MAX_EXCESS_COMPLEXITY
    param_grid = param_grid

    def __init__(self, campaign_round=None, resume_run=None, *,
                 features=None, selected_count=REQUIRED_FEATURE_COUNT,
                 max_excess_complexity=MAX_EXCESS_COMPLEXITY):
        self.feature_columns = configure_feature_pool(features, selected_count)
        self.required_feature_count = selected_count
        self.max_excess_complexity = max_excess_complexity
        self.param_grid = {
            **param_grid,
            'loss_function': [make_loss_function(selected_count, max_excess_complexity)],
        }
        self.resume_mode = resume_run is not None
        self.resume_invocation_id = (
            'r' + datetime.now().strftime('%H%M%S%f')
            if self.resume_mode
            else 'initial'
        )
        self._existing_resume_contract = None
        output_root = Path(BASE_OUTPUT_DIR).resolve()

        if self.resume_mode:
            resume_path = Path(resume_run).expanduser().resolve()
            if not resume_path.is_dir() or not resume_path.is_relative_to(output_root):
                raise ResumeCheckpointError(
                    'resume directory must be an existing run below '
                    f'{output_root}'
                )
            contract_path = resume_path / 'resume_contract.json'
            if not contract_path.is_file():
                raise ResumeCheckpointError(
                    f'resume contract is missing: {contract_path}'
                )
            try:
                self._existing_resume_contract = json.loads(
                    contract_path.read_text(encoding='utf-8')
                )
                stored_campaign = self._existing_resume_contract['contract'][
                    'campaign_round'
                ]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ResumeCheckpointError(
                    f'invalid resume contract: {contract_path}'
                ) from exc
            self.campaign_round = resolve_campaign_round(stored_campaign)
            if campaign_round is not None:
                requested_campaign = resolve_campaign_round(campaign_round)
                if requested_campaign != self.campaign_round:
                    raise ResumeCheckpointError(
                        'requested campaign does not match resume contract: '
                        f'{requested_campaign} != {self.campaign_round}'
                    )
            expected_parent = (
                output_root / campaign_directory_name(self.campaign_round)
            ).resolve()
            if resume_path.parent != expected_parent:
                raise ResumeCheckpointError(
                    'resume directory is not directly below its campaign directory'
                )
            if (resume_path / 'results.json').is_file():
                raise ResumeCheckpointError(
                    'this run already contains results.json and is complete; '
                    'refusing to overwrite it'
                )
        else:
            self.campaign_round = resolve_campaign_round(campaign_round)

        self.output_base_path = os.path.join(
            BASE_OUTPUT_DIR,
            campaign_directory_name(self.campaign_round),
        )

        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        if self.resume_mode:
            self.date_folder = resume_path.name
            self.output_path = str(resume_path)
        else:
            self.date_folder = (
                datetime.now().strftime("%Y%m%d_%H%M%S") + f"_job{job_id}"
            )
            self.output_path = os.path.join(
                self.output_base_path, self.date_folder
            )
        self.scaler = None
        os.makedirs(self.output_path, exist_ok=True)
        self.data_path = os.path.join(self.output_path, DERIVED_DATA_FILENAME)
        self.pysr_run_dir = Path(self.output_path) / 'pysr_runs'
        self.pysr_run_dir.mkdir(parents=True, exist_ok=True)
        self.configuration_checkpoint_dir = (
            Path(self.output_path) / 'configuration_results'
        )
        self.configuration_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume_contract_file = Path(self.output_path) / 'resume_contract.json'
        self.cv_selection_checkpoint_file = (
            Path(self.output_path) / 'cv_selection_checkpoint.json'
        )
        self.progress_file = Path(self.output_path) / 'search_progress.jsonl'
        self.progress_status_file = Path(self.output_path) / 'progress_status.json'
        self.progress_event_count = 0
        self._runtime_manifest_observed_written = False
        self._overall_search_progress = None
        self._search_started_monotonic = None
        self._fold_executor = None
        self._observed_fold_worker_runtime = None
        self._resume_contract_id = None
        self._fold_executor_shutdown_report = None

        self.setup_logging()

        if self.resume_mode:
            self.logger.info("Resuming symbolic-regression run: %s", self.output_path)

        self.logger.info(f"Logical processors visible to Python: {LOGICAL_PROCESSORS}")
        self.logger.info(f"Campaign round: {self.campaign_round}")
        self.logger.info(f"JuliaCall threads: {os.environ.get('PYTHON_JULIACALL_THREADS')}")
        self.logger.info(f"BLAS helper threads: {BLAS_THREADS}")
        self.logger.info(f"PySR parallelism: {PARALLELISM}")
        self.logger.info(
            "CV fold layout: %d worker process(es) x %d Julia thread(s) "
            "(aggregate=%d/%d visible logical processors)",
            FOLD_WORKERS,
            FOLD_WORKER_THREADS,
            FOLD_EXECUTION_LAYOUT.aggregate_threads,
            LOGICAL_PROCESSORS,
        )
        self.logger.info(
            "Strict deterministic search: %s",
            "enabled" if STRICT_DETERMINISTIC_SEARCH else "disabled",
        )
        if not STRICT_DETERMINISTIC_SEARCH:
            self.logger.info(
                "Random seeds are recorded, but multithreaded PySR evolution "
                "is stochastic and is not bitwise reproducible"
            )
        if 'SLURM_JOB_ID' in os.environ:
            self.logger.info(f"SLURM job ID: {os.environ['SLURM_JOB_ID']}")
            self.logger.info(f"Node: {os.environ.get('SLURM_NODELIST', 'Unknown')}")

        self.feature_selection = None
        self.target_column = TARGET_COLUMN
        self.final_candidate_position = None

        self.enable_refinement = ENABLE_REFINEMENT
        self.refinement_config = REFINEMENT_CONFIG

    def setup_logging(self):
        """Configure logging to file and console."""
        log_file = os.path.join(self.output_path, 'symbolic_regression.log')
        handlers = [logging.StreamHandler()]
        if not self.resume_mode:
            handlers.insert(0, logging.FileHandler(log_file, encoding='utf-8'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('Log file%s: %s',
                         ' (pending resume validation)' if self.resume_mode else '',
                         log_file)

    def _build_resume_contract(self, sampled_configs, artifact_hashes=None):
        """Build the strict scientific identity for durable search state."""
        split_path = Path(self.output_path) / 'split_assignments.csv'
        source_root = Path(__file__).resolve().parent
        if artifact_hashes is None:
            artifact_hashes = {
                'derived_data_sha256': file_sha256(self.data_path),
                'split_assignments_sha256': file_sha256(split_path),
            }
        implementation_files = (
            'SR_Lewis_base.py',
            'sr_efficiency.py',
            'sr_validation.py',
            'sr_reporting.py',
            'data_preparation.py',
            'score_calibration.py',
            'campaign.py',
            'project_paths.py',
        )
        contract = {
            'campaign_round': self.campaign_round,
            'score_version': self.score_info.get('score_version'),
            **artifact_hashes,
            'feature_columns': list(self.feature_columns),
            'feature_selection': getattr(self, 'feature_selection', None),
            'target_column': self.target_column,
            'required_feature_count': self.required_feature_count,
            'random_state': RANDOM_STATE,
            'sampled_configs_sha256': payload_sha256(sampled_configs),
            'scientific_search_sha256': payload_sha256({
                'parameter_grid': self.param_grid,
                'requested_unique_configurations': N_RANDOM_SEARCHES,
                'cv_method': CV_METHOD,
                'cv_n_splits': CV_N_SPLITS,
                'cv_n_repeats': CV_N_REPEATS,
                'cv_n_strata': CV_N_STRATA,
                'outer_test_size': TEST_SIZE,
                'outer_split_strata': SPLIT_N_STRATA,
                'outer_split_metric': SPLIT_METRIC,
                'outer_split_features': self.feature_columns,
                'max_excess_complexity': self.max_excess_complexity,
                'final_review_repeats': FINAL_REVIEW_N_REPEATS,
                'final_review_random_state': FINAL_REVIEW_RANDOM_STATE,
                'search_execution': SEARCH_EXECUTION_CONFIG,
                'refinement': REFINEMENT_CONFIG,
            }),
            'runtime_layout': {
                'parallelism': PARALLELISM,
                'strict_deterministic': STRICT_DETERMINISTIC_SEARCH,
                'fold_workers': FOLD_WORKERS,
                'fold_worker_threads': FOLD_WORKER_THREADS,
                'aggregate_fold_worker_threads': (
                    FOLD_EXECUTION_LAYOUT.aggregate_threads
                ),
                'blas_threads': BLAS_THREADS,
            },
            'implementation_files_sha256': {
                filename: file_sha256(source_root / filename)
                for filename in implementation_files
            },
            'versions': {
                'pysr': _PYSR_VERSION,
                'numpy': package_version('numpy'),
                'scikit_learn': package_version('scikit-learn'),
            },
            'external_test_evaluations': 0,
        }
        return {
            'schema_version': 1,
            'contract_id': payload_sha256(contract),
            'contract': contract,
        }

    def _prepare_resume_contract(self, sampled_configs, artifact_hashes=None):
        """Create or strictly validate the run contract before CV starts."""
        current = self._build_resume_contract(sampled_configs, artifact_hashes)
        if self.resume_mode:
            existing = self._existing_resume_contract
            if (
                not isinstance(existing, dict)
                or existing.get('schema_version') != current['schema_version']
                or existing.get('contract_id') != current['contract_id']
            ):
                raise ResumeCheckpointError(
                    'resume contract mismatch: data, Score, split, features, '
                    'search settings, runtime layout, or package versions changed'
                )
            if not getattr(self, '_resume_log_attached', False):
                handler = logging.FileHandler(
                    Path(self.output_path) / 'symbolic_regression.log', encoding='utf-8'
                )
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(handler)
                self._resume_log_attached = True
            self.logger.info(
                "Resume contract verified: %s", current['contract_id']
            )
        else:
            atomic_write_json(self.resume_contract_file, current)
            self.logger.info(
                "Resume contract saved: %s", current['contract_id']
            )
        self._resume_contract_id = current['contract_id']
        return current

    def _configuration_checkpoint_path(self, config_id):
        """Return one stable checkpoint path for a generated configuration ID."""
        config_id = str(config_id)
        if (
            not config_id.startswith('cfg_')
            or Path(config_id).name != config_id
            or not config_id.replace('_', '').isalnum()
        ):
            raise ResumeCheckpointError(
                f'invalid configuration ID for checkpointing: {config_id!r}'
            )
        return self.configuration_checkpoint_dir / f'{config_id}.json'

    def _write_configuration_checkpoint(self, sampled_config, result):
        """Atomically persist one terminal configuration result."""
        if self._resume_contract_id is None:
            raise ResumeCheckpointError('resume contract is not initialized')
        config_id = str(sampled_config['config_id'])
        if result.get('status') not in {
            'completed', 'conservatively_pruned', 'failed'
        }:
            raise ResumeCheckpointError(
                f'configuration {config_id} is not terminal'
            )
        payload = {
            'schema_version': 1,
            'contract_id': self._resume_contract_id,
            'saved_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'configuration': {
                'iteration': int(result['iteration']),
                'config_id': config_id,
                'config_seed': int(sampled_config['config_seed']),
                'params_sha256': payload_sha256(sampled_config['params']),
            },
            'result': result,
        }
        try:
            atomic_write_json(
                self._configuration_checkpoint_path(config_id), payload
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ResumeCheckpointError(
                f'failed to save terminal checkpoint for {config_id}'
            ) from exc
        return payload

    def _load_configuration_checkpoints(self, sampled_configs):
        """Load a validated contiguous prefix of terminal configurations."""
        expected_ids = {str(item['config_id']) for item in sampled_configs}
        unexpected = {
            path.stem
            for path in self.configuration_checkpoint_dir.glob('*.json')
            if path.stem not in expected_ids
        }
        if unexpected:
            raise ResumeCheckpointError(
                'unexpected configuration checkpoints: '
                + ', '.join(sorted(unexpected))
            )

        loaded = {}
        gap_found = False
        for iteration, sampled_config in enumerate(sampled_configs, start=1):
            config_id = str(sampled_config['config_id'])
            path = self._configuration_checkpoint_path(config_id)
            if not path.is_file():
                gap_found = True
                continue
            if gap_found:
                raise ResumeCheckpointError(
                    'configuration checkpoints are not a contiguous prefix; '
                    f'found {config_id} after a missing earlier configuration'
                )
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                configuration = payload['configuration']
                result = payload['result']
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ResumeCheckpointError(
                    f'invalid configuration checkpoint: {path}'
                ) from exc
            if (
                payload.get('schema_version') != 1
                or payload.get('contract_id') != self._resume_contract_id
                or int(configuration.get('iteration', -1)) != iteration
                or configuration.get('config_id') != config_id
                or int(configuration.get('config_seed', -1))
                != int(sampled_config['config_seed'])
                or configuration.get('params_sha256')
                != payload_sha256(sampled_config['params'])
                or result.get('config_id') != config_id
                or int(result.get('iteration', -1)) != iteration
                or result.get('status')
                not in {'completed', 'conservatively_pruned', 'failed'}
            ):
                raise ResumeCheckpointError(
                    f'configuration checkpoint does not match sampled config: {path}'
                )
            loaded[config_id] = result
        return loaded

    def _write_cv_selection_checkpoint(
        self,
        *,
        best_result,
        config_selection,
        incumbent_result,
        terminal_results,
        shutdown_report=None,
    ):
        """Persist the CV decision before any Julia worker is asked to exit."""
        statuses = {
            'completed': 0,
            'conservatively_pruned': 0,
            'failed': 0,
        }
        for result in terminal_results:
            statuses[result['status']] += 1
        payload = {
            'schema_version': 1,
            'contract_id': self._resume_contract_id,
            'status': 'cv_selection_complete',
            'saved_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'terminal_configurations': len(terminal_results),
            'terminal_status_counts': statuses,
            'selected_config_id': best_result['config_id'],
            'selected_config_seed': int(best_result['config_seed']),
            'selected_config_result': str(
                self._configuration_checkpoint_path(
                    best_result['config_id']
                ).relative_to(Path(self.output_path))
            ),
            'highest_mean_config_id': incumbent_result['config_id'],
            'selection_rule': config_selection,
            'external_test_evaluations': 0,
            'fold_worker_shutdown': shutdown_report,
        }
        if self.resume_mode and self.cv_selection_checkpoint_file.is_file():
            try:
                previous = json.loads(
                    self.cv_selection_checkpoint_file.read_text(encoding='utf-8')
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ResumeCheckpointError(
                    'existing CV selection checkpoint is unreadable'
                ) from exc
            if (
                previous.get('contract_id') != self._resume_contract_id
                or previous.get('selected_config_id')
                != best_result['config_id']
            ):
                raise ResumeCheckpointError(
                    'recomputed CV selection does not match the saved checkpoint'
                )
        atomic_write_json(self.cv_selection_checkpoint_file, payload)
        return payload

    def write_run_config(self):
        """Persist the resolved personal-PC execution limits before searching."""
        payload = {
            'schema_version': 4,
            'profile': 'publication',
            'campaign_round': self.campaign_round,
            'runtime': {
                **symbolic_regression_runtime_metadata(),
                'julia_threads': JULIA_THREADS,
                'python_juliacall_threads': os.environ.get('PYTHON_JULIACALL_THREADS'),
                'julia_num_threads': os.environ.get('JULIA_NUM_THREADS'),
                'default_thread_policy': (
                    'all_visible_logical_processors_up_to_32'
                ),
                'cv_folds_run_sequentially': FOLD_WORKERS == 1,
                'fold_worker_count': FOLD_WORKERS,
                'threads_per_fold_worker': FOLD_WORKER_THREADS,
                'aggregate_fold_worker_threads': (
                    FOLD_EXECUTION_LAYOUT.aggregate_threads
                ),
                'fold_worker_layout_source': FOLD_EXECUTION_LAYOUT.source,
                'fold_worker_runtime_observed': getattr(
                    self, '_observed_fold_worker_runtime', None
                ),
                'power_plan_management': 'not_modified_by_pipeline',
                'fold_worker_shutdown_limits': dict(FOLD_SHUTDOWN_CONFIG),
                'fold_worker_shutdown_observed': getattr(
                    self, '_fold_executor_shutdown_report', None
                ),
            },
            'search': {
                'random_state': RANDOM_STATE,
                'seed_policy': (
                    'stable config seeds; common fold seeds across configurations; '
                    'selected config seed for refinement'
                ),
                'requested_unique_configurations': N_RANDOM_SEARCHES,
                'global_configuration_early_stop': False,
                'parameter_grid': self.param_grid,
                'cv_method': CV_METHOD,
                'cv_n_splits': CV_N_SPLITS,
                'cv_n_repeats': CV_N_REPEATS,
                'execution': SEARCH_EXECUTION_CONFIG,
                'performance_policy': {
                    'pareto_predictions_each_chunk': True,
                    'paired_bootstrap_only_at_terminal_chunk': True,
                    'manifest_rewrite_after_julia_start': 'once_per_run',
                    'concise_progress_snapshot': 'progress_status.json',
                    'chunk_console_progress': True,
                    'parallel_fold_progress_directories': (
                        'fold_workers/<config_id>/fold_XX'
                    ),
                    'scientific_iteration_budget_reduced': False,
                },
                'fold_scaler': 'fit_on_fold_training_rows_only',
                'pareto_candidate_selection': {
                    'feature_policy': 'exact_count_from_candidate_pool',
                    'metric': 'raw_validation_r2',
                    'rule': (
                        'paired_bootstrap_one_standard_error_then_'
                        'excess_complexity'
                    ),
                    'bootstrap_samples': SEARCH_EXECUTION_CONFIG[
                        'validation_bootstrap_samples'
                    ],
                    'pysr_candidate_addressing': 'equations_iloc_position',
                },
                'configuration_selection': (
                    'one_standard_error_then_excess_complexity_and_stability'
                ),
                'complexity_policy': {
                    'excess_complexity_ceiling': self.max_excess_complexity,
                    'pysr_parsimony': 0,
                    'pysr_frequency_balancing': {
                        'use_frequency': True,
                        'use_frequency_in_tournament': True,
                        'role': 'search-size coverage, not candidate scoring',
                    },
                    'selection_metric': 'excess_term_budget_complexity',
                    'free_per_feature': {
                        'linear_terms': 1,
                        'direct_nonlinear_terms': 1,
                    },
                    'current_default_direct_nonlinear_operator': 'square',
                    'penalized_structures': [
                        'feature interactions',
                        'powers above two',
                        'repeated nonlinear terms for one feature',
                        'composite or nested nonlinear transforms',
                    ],
                    'pysr_structural_complexity_role': (
                        'audit field and maxsize/maxdepth resource cap only'
                    ),
                },
            },
            'outer_split': {
                'method': 'deterministic_stratified_duplex',
                'feature_columns': list(self.feature_columns),
                'test_size': TEST_SIZE,
                'n_strata': SPLIT_N_STRATA,
                'metric': SPLIT_METRIC,
                'random_state': RANDOM_STATE,
                'assignment_file': 'split_assignments.csv',
                'source_row_index': (
                    'zero-based data.csv data-row number; header excluded'
                ),
            },
            'refinement': {
                **REFINEMENT_CONFIG,
                'candidate_inheritance': 'selected_configuration_cv_winners',
                'inherited_coefficients': 'refit_on_outer_training_rows',
                'inheritance_audit_file': 'cv_winner_inheritance.json',
                'search_excess_complexity_ceiling': getattr(
                    self, 'refinement_excess_complexity_ceiling', None
                ),
            },
            'final_candidate_review': {
                'cv_n_splits': 4,
                'cv_n_repeats': FINAL_REVIEW_N_REPEATS,
                'random_state': FINAL_REVIEW_RANDOM_STATE,
                'cv_splits': 'separate_repeated_stratified_outer_training_folds',
                'structure_search': False,
                'fold_scaler': 'fit_on_fold_training_rows_only',
                'coefficient_fit': 'fold_training_only_from_neutral_initial_values',
                'selection': 'one_standard_error_then_excess_and_structural_complexity',
                'scope': 'conditional_candidate_selection_diagnostic',
                'audit_file': 'final_candidate_cv.json',
                'result': getattr(self, 'final_candidate_cv', None),
            },
            'ridge_baseline': {
                'basis_types': ['linear', 'additive_quadratic'],
                'alphas': list(RIDGE_BASELINE_ALPHAS),
                'feature_source': 'final_sr_selected_feature_names',
                'feature_names': getattr(self, 'selected_feature_names', None),
                'outer_split': 'identical_to_symbolic_regression',
                'cv_splits': 'identical_fold_indices_to_symbolic_regression',
                'fold_scaler': 'fit_on_fold_training_rows_only',
                'selection_metric': 'mean_raw_validation_r2',
            },
            'coefficient_uncertainty': dict(COEFFICIENT_UNCERTAINTY_CONFIG),
            'feature_columns': list(self.feature_columns),
            'selected_feature_names': getattr(self, 'selected_feature_names', None),
            'required_feature_count': self.required_feature_count,
            'feature_selection': getattr(self, 'feature_selection', None),
            'feature_metadata': FEATURE_METADATA,
            'all_features_required': self.required_feature_count == len(self.feature_columns),
            'target_column': self.target_column,
            'target_metadata': TARGET_METADATA,
            'metric_definitions': METRIC_DEFINITIONS,
            'score_version': self.score_info.get('score_version'),
            'resume': {
                'enabled_for_this_invocation': bool(
                    getattr(self, 'resume_mode', False)
                ),
                'contract_file': 'resume_contract.json',
                'configuration_results_directory': 'configuration_results',
                'cv_selection_checkpoint_file': 'cv_selection_checkpoint.json',
                'contract_id': getattr(self, '_resume_contract_id', None),
                'invocation_id': self.resume_invocation_id,
                'policy': (
                    'skip validated terminal configuration checkpoints; '
                    'restart an interrupted refinement as a new attempt'
                ),
            },
            'versions': {
                'pysr': _PYSR_VERSION,
                'numpy': package_version('numpy'),
                'scikit_learn': package_version('scikit-learn'),
            },
        }
        atomic_write_json(Path(self.output_path) / 'run_config.json', payload)

    @staticmethod
    def _pareto_rows(model):
        """Return JSON-ready PySR Pareto rows without relying on frame labels."""
        equations = getattr(model, 'equations_', None)
        if equations is None:
            return []
        rows = []
        for position in range(len(equations)):
            row = equations.iloc[position]
            sympy_value = row.get('sympy_format', row.get('equation', ''))
            try:
                loss = float(row.get('loss', np.inf))
                if not np.isfinite(loss):
                    loss = None
            except (TypeError, ValueError, OverflowError):
                loss = None
            try:
                complexity = int(row.get('complexity', 10**9))
            except (TypeError, ValueError, OverflowError):
                complexity = 10**9
            try:
                score = float(row.get('score', np.nan))
                if not np.isfinite(score):
                    score = None
            except (TypeError, ValueError, OverflowError):
                score = None
            rows.append({
                'candidate_position': position,
                'candidate_frame_label': str(equations.index[position]),
                'equation': str(sympy_value),
                'backend_equation': str(row.get('equation', '')),
                'loss': loss,
                'pysr_score': score,
                'complexity': complexity,
                'candidate_origin': (
                    row.get('candidate_origin')
                    if isinstance(row.get('candidate_origin'), str)
                    else 'evolution'
                ),
                'source_cv_folds': (
                    row.get('source_cv_folds')
                    if isinstance(row.get('source_cv_folds'), list) else []
                ),
            })
        return rows

    @staticmethod
    def _equation_at(model, candidate_position):
        if candidate_position is None:
            raise ValueError('an explicit PySR candidate position is required')
        row = model.equations_.iloc[int(candidate_position)]
        # PySR.sympy() refreshes the backend table and discards archived rows.
        return str(row.get('sympy_format', row['equation']))

    @staticmethod
    def _retain_chunk_candidates(model, archive):
        """Keep callable rows across chunks of the same fit and fixed scaler."""
        for _, row in model.equations_.iterrows():
            source = row.get('refit_source_equation')
            key = (str(row['equation']), source if isinstance(source, str) else None)
            previous = archive.get(key)
            if previous is None or float(row['loss']) < float(previous['loss']):
                archive[key] = row.copy()
        model.equations_ = pd.DataFrame(list(archive.values())).reset_index(drop=True)

    @staticmethod
    def _predict_at(model, X, candidate_position):
        if candidate_position is None:
            raise ValueError('an explicit PySR candidate position is required')
        prediction = np.asarray(
            model.predict(X, index=int(candidate_position)), dtype=float
        ).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError('selected PySR candidate produced non-finite predictions')
        return prediction

    def _record_progress(self, event):
        self.progress_event_count += 1
        append_json_line(
            self.progress_file,
            {
                'schema_version': 1,
                'event_index': self.progress_event_count,
                **event,
            },
        )

    def _write_progress_status(self, event):
        """Atomically publish a small, human-facing current-progress snapshot."""
        payload = {
            'schema_version': 1,
            'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            **event,
        }
        overall = getattr(self, '_overall_search_progress', None)
        if overall is not None:
            overall_snapshot = dict(overall)
            if getattr(self, '_search_started_monotonic', None) is not None:
                overall_snapshot['search_elapsed_seconds'] = (
                    monotonic() - self._search_started_monotonic
                )
            payload['overall'] = overall_snapshot
        status_file = getattr(
            self,
            'progress_status_file',
            Path(self.progress_file).with_name('progress_status.json'),
        )
        try:
            atomic_write_json(status_file, payload)
        except (OSError, TypeError, ValueError) as exc:
            # Progress reporting must never turn a scientifically valid model
            # configuration into a failed configuration.
            getattr(self, 'logger', logging.getLogger(__name__)).warning(
                "Unable to update concise progress status: %s", exc
            )

    def _mark_configuration_terminal(
        self,
        *,
        status,
        config_index,
        config_total,
        config_id,
        completed,
        pruned,
        failed,
        folds_completed,
        fold_total,
    ):
        """Record one configuration terminal state without changing search policy."""
        terminal = int(completed) + int(pruned) + int(failed)
        search_elapsed = (
            monotonic() - self._search_started_monotonic
            if getattr(self, '_search_started_monotonic', None) is not None
            else None
        )
        self._overall_search_progress = {
            'terminal_configurations': terminal,
            'total_configurations': int(config_total),
            'completed_configurations': int(completed),
            'conservatively_pruned_configurations': int(pruned),
            'failed_configurations': int(failed),
            'search_elapsed_seconds': search_elapsed,
        }
        self._write_progress_status({
            'phase': 'configuration_search',
            'status': 'configuration_terminal',
            'configuration': {
                'index': int(config_index),
                'total': int(config_total),
                'config_id': str(config_id),
            },
            'fold': {
                'completed': int(folds_completed),
                'total': int(fold_total),
            },
            'decision': str(status),
        })
        getattr(self, 'logger', logging.getLogger(__name__)).info(
            "[Overall] configurations terminal %d/%d "
            "(completed=%d, pruned=%d, failed=%d) | latest=%s [%s]",
            terminal,
            int(config_total),
            int(completed),
            int(pruned),
            int(failed),
            status,
            config_id,
        )

    def _ensure_fold_executor(self):
        """Create one persistent spawned-process pool for all CV folds."""
        if FOLD_WORKERS == 1 and FOLD_WORKER_THREADS == JULIA_THREADS:
            return None
        if self._fold_executor is None:
            self.logger.info(
                "Starting persistent CV fold pool: %d processes x %d Julia threads",
                FOLD_WORKERS,
                FOLD_WORKER_THREADS,
            )
            self._fold_executor = ProcessPoolExecutor(
                max_workers=FOLD_WORKERS,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_fold_worker_runtime,
                initargs=(FOLD_WORKER_THREADS,),
            )
        return self._fold_executor

    def _shutdown_fold_executor(self, *, require_idle=False):
        """Release fold workers with a hard bound on Julia finalization."""
        executor = getattr(self, '_fold_executor', None)
        if executor is None:
            if self._fold_executor_shutdown_report is None:
                self._fold_executor_shutdown_report = {
                    'status': 'not_created',
                    'graceful': True,
                    'forced_termination': False,
                    'remaining_pids': [],
                }
            return self._fold_executor_shutdown_report

        pending_count = len(getattr(executor, '_pending_work_items', {}) or {})
        if require_idle and pending_count:
            raise RuntimeError(
                'refusing normal CV pool shutdown while futures remain pending: '
                f'{pending_count}'
            )

        self.logger.info(
            "Stopping persistent CV fold worker pool "
            "(graceful limit=%ds)",
            FOLD_SHUTDOWN_CONFIG['graceful_wait_seconds'],
        )
        self._write_progress_status({
            'phase': 'worker_shutdown',
            'status': 'graceful_shutdown_requested',
            'configuration': None,
            'fold': None,
            'decision': 'wait_bounded',
            'shutdown_limits': dict(FOLD_SHUTDOWN_CONFIG),
        })
        report = shutdown_process_pool_bounded(
            executor,
            **FOLD_SHUTDOWN_CONFIG,
        )
        report['status'] = (
            'forced_cleanup_complete'
            if report['forced_termination']
            else 'graceful_shutdown_complete'
        )
        report['pending_futures_at_shutdown'] = int(pending_count)
        self._fold_executor = None
        self._fold_executor_shutdown_report = report

        if report['forced_termination']:
            self.logger.warning(
                "CV fold pool exceeded its graceful shutdown limit; "
                "terminated idle worker PIDs=%s, killed PIDs=%s",
                report['terminated_pids'],
                report['killed_pids'],
            )
        else:
            self.logger.info(
                "CV fold worker pool stopped gracefully in %.1fs",
                report['elapsed_seconds'],
            )
        if report['remaining_pids']:
            raise RuntimeError(
                'CV fold workers remain alive after bounded cleanup: '
                f"{report['remaining_pids']}"
            )
        if report['manager_thread_alive']:
            self.logger.warning(
                "Process-pool manager thread was still cleaning up after workers exited"
            )

        self._write_progress_status({
            'phase': 'worker_shutdown',
            'status': report['status'],
            'configuration': None,
            'fold': None,
            'decision': 'continue_to_refinement',
            'worker_shutdown': report,
        })
        if hasattr(self, 'score_info'):
            self.write_run_config()
        return report

    def _build_fold_job(
        self,
        *,
        base_config,
        train_idx,
        val_idx,
        fold_number,
        total_folds,
        config_id,
        config_index,
        config_total,
    ):
        """Prepare one leakage-safe, pickleable fold job."""
        fold_scaler, X_train_fold, X_val_fold = prepare_fold_scaled_data(
            self.X_train_raw,
            train_idx,
            val_idx,
        )
        fold_seed = RANDOM_STATE + int(fold_number) * 1000
        fold_config = dict(base_config)
        fold_config['random_state'] = fold_seed
        fold_config.pop('early_stop_condition', None)
        max_fold_iterations = int(fold_config['niterations'])
        max_fold_seconds = float(fold_config['timeout_in_seconds'])
        worker_dir = (
            Path(self.output_path)
            / 'fold_workers'
            / str(config_id)
            / f'fold_{int(fold_number):02d}'
        )
        run_id = f'{config_id}_fold_{int(fold_number):02d}'
        if self.resume_mode:
            worker_dir = worker_dir / self.resume_invocation_id
            run_id = f'{run_id}_{self.resume_invocation_id}'
        worker_dir.mkdir(parents=True, exist_ok=True)
        return {
            'model_config': fold_config,
            'X_train': np.asarray(X_train_fold, dtype=float),
            'y_train': np.asarray(self.y_train[train_idx], dtype=float),
            'X_eval': np.asarray(X_val_fold, dtype=float),
            'y_eval': np.asarray(self.y_train[val_idx], dtype=float),
            'feature_columns': list(self.feature_columns),
            'required_feature_count': self.required_feature_count,
            'fold_scaler_mean': fold_scaler.mean_.tolist(),
            'max_excess_complexity': self.max_excess_complexity,
            'fold_scaler_scale': fold_scaler.scale_.tolist(),
            'fold': int(fold_number),
            'repeat': (int(fold_number) - 1) // CV_N_SPLITS + 1,
            'fold_seed': int(fold_seed),
            'total_folds': int(total_folds),
            'config_id': str(config_id),
            'config_index': int(config_index),
            'config_total': int(config_total),
            'run_id': run_id,
            'max_iterations': max_fold_iterations,
            'chunk_iterations': int(SEARCH_EXECUTION_CONFIG['chunk_iterations']),
            'max_wall_seconds': max_fold_seconds,
            'minimum_iterations': int(SEARCH_EXECUTION_CONFIG['minimum_iterations']),
            'plateau_patience': int(SEARCH_EXECUTION_CONFIG['plateau_patience']),
            'minimum_score_improvement': float(
                SEARCH_EXECUTION_CONFIG['minimum_score_improvement']
            ),
            'minimum_loss_relative_improvement': float(
                SEARCH_EXECUTION_CONFIG['minimum_loss_relative_improvement']
            ),
            'enable_plateau_stop': bool(
                SEARCH_EXECUTION_CONFIG['enable_plateau_stop']
            ),
            'selection_bootstrap_samples': int(
                SEARCH_EXECUTION_CONFIG['validation_bootstrap_samples']
            ),
            'selection_seed': int(fold_seed),
            'worker_dir': str(worker_dir),
        }

    def _fit_fold_job(self, job):
        """Execute one prepared fold and return a JSON/pickle-safe result."""
        model, chunk_summary = self.fit_pysr_in_chunks(
            job['model_config'],
            job['X_train'],
            job['y_train'],
            run_id=job['run_id'],
            max_iterations=job['max_iterations'],
            chunk_iterations=job['chunk_iterations'],
            max_wall_seconds=job['max_wall_seconds'],
            minimum_iterations=job['minimum_iterations'],
            plateau_patience=job['plateau_patience'],
            minimum_score_improvement=job['minimum_score_improvement'],
            minimum_loss_relative_improvement=(
                job['minimum_loss_relative_improvement']
            ),
            enable_plateau_stop=job['enable_plateau_stop'],
            X_eval=job['X_eval'],
            y_eval=job['y_eval'],
            selection_bootstrap_samples=job['selection_bootstrap_samples'],
            selection_seed=job['selection_seed'],
            training_excess_complexity_ceiling=job['max_excess_complexity'],
            progress_phase='cross_validation',
            config_id=job['config_id'],
            config_index=job['config_index'],
            config_total=job['config_total'],
            fold_index=job['fold'],
            fold_total=job['total_folds'],
        )
        selected_candidate = chunk_summary['selected_candidate']
        if selected_candidate is None:
            raise ValueError(
                f"Fold {job['fold']} ended without a finite "
                f"{self.required_feature_count}-variable candidate"
            )
        candidate_position = int(selected_candidate['candidate_position'])
        y_pred_val = self._predict_at(model, job['X_eval'], candidate_position)
        val_r2 = float(r2_score(job['y_eval'], y_pred_val))
        val_rmse = float(np.sqrt(mean_squared_error(job['y_eval'], y_pred_val)))
        val_mae = float(mean_absolute_error(job['y_eval'], y_pred_val))
        fold_equation = self._equation_at(model, candidate_position)
        used_vars = used_features(fold_equation, self.feature_columns)
        if len(used_vars) != self.required_feature_count:
            raise ValueError(
                f"Fold {job['fold']} equation must contain exactly "
                f"{self.required_feature_count} candidate features after simplification"
            )
        result = {
            'fold': int(job['fold']),
            'repeat': int(job['repeat']),
            'fold_seed': int(job['fold_seed']),
            'raw_r2': val_r2,
            'rmse': val_rmse,
            'mae': val_mae,
            'equation': str(fold_equation),
            'backend_equation': str(model.equations_.iloc[candidate_position]['equation']),
            'candidate_position': candidate_position,
            'candidate_complexity': int(selected_candidate['complexity']),
            'candidate_structural_complexity': int(
                selected_candidate['structural_complexity']
            ),
            'candidate_excess_complexity': int(
                selected_candidate['excess_complexity']
            ),
            'candidate_has_direct_quadratic': bool(
                selected_candidate['has_direct_quadratic']
            ),
            'candidate_term_budget': selected_candidate['term_budget'],
            'candidate_training_loss': float(selected_candidate['loss']),
            'candidate_selection_rule': selected_candidate.get('selection_rule'),
            'candidate_best_point_position': selected_candidate.get(
                'best_point_candidate_position'
            ),
            'candidate_one_se_threshold': selected_candidate.get(
                'one_se_threshold'
            ),
            'used_variables': used_vars,
            'fold_scaler_mean': list(job['fold_scaler_mean']),
            'fold_scaler_scale': list(job['fold_scaler_scale']),
            'execution': chunk_summary,
            'worker_runtime': symbolic_regression_runtime_metadata(),
            'worker_dir': str(job['worker_dir']),
        }
        del model
        gc.collect()
        return result

    def _merge_worker_progress(self, job):
        """Merge a completed worker's append-only audit into the run audit."""
        progress_path = Path(job['worker_dir']) / 'search_progress.jsonl'
        if not progress_path.is_file():
            return
        with progress_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                event.pop('schema_version', None)
                event.pop('event_index', None)
                event['fold_worker'] = {
                    'fold': int(job['fold']),
                    'worker_dir': str(job['worker_dir']),
                    'threads': int(FOLD_WORKER_THREADS),
                }
                self._record_progress(event)

    def _run_parallel_fold_batch(self, jobs):
        """Run one bounded fold batch and surface live per-worker progress."""
        executor = self._ensure_fold_executor()
        future_to_job = {
            executor.submit(execute_fold_job_worker, job): job for job in jobs
        }
        pending = set(future_to_job)
        last_status_payload = None
        results = []
        errors = []
        while pending:
            done, pending = wait(
                pending,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            worker_statuses = []
            for future, job in future_to_job.items():
                status_path = Path(job['worker_dir']) / 'progress_status.json'
                if status_path.is_file():
                    try:
                        status = json.loads(status_path.read_text(encoding='utf-8'))
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    worker_statuses.append({
                        'fold': int(job['fold']),
                        'worker_dir': str(job['worker_dir']),
                        'status': status,
                    })
            status_payload = {
                'phase': 'cross_validation_parallel',
                'status': 'fold_batch_running' if pending else 'fold_batch_finishing',
                'configuration': {
                    'index': int(jobs[0]['config_index']),
                    'total': int(jobs[0]['config_total']),
                    'config_id': str(jobs[0]['config_id']),
                },
                'fold': {
                    'active': [int(job['fold']) for job in jobs],
                    'total': int(jobs[0]['total_folds']),
                },
                'parallel_workers': sorted(
                    worker_statuses, key=lambda row: row['fold']
                ),
                'decision': 'continue',
            }
            serialized_status = json.dumps(status_payload, sort_keys=True, default=str)
            if serialized_status != last_status_payload:
                self._write_progress_status(status_payload)
                last_status_payload = serialized_status

            for future in done:
                job = future_to_job[future]
                try:
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors.append((int(job['fold']), exc))
                    else:
                        results.append(result)
                        if self._observed_fold_worker_runtime is None:
                            self._observed_fold_worker_runtime = result.get(
                                'worker_runtime'
                            )
                            self.write_run_config()
                finally:
                    self._merge_worker_progress(job)
        if errors:
            details = '; '.join(
                f'fold {fold}: {type(exc).__name__}: {exc}'
                for fold, exc in errors
            )
            raise RuntimeError(f'parallel fold batch failed: {details}')
        return sorted(results, key=lambda row: int(row['fold']))

    def fit_pysr_in_chunks(
        self,
        model_config,
        X_train,
        y_train,
        *,
        run_id,
        max_iterations,
        chunk_iterations,
        max_wall_seconds,
        minimum_iterations,
        plateau_patience,
        minimum_score_improvement,
        minimum_loss_relative_improvement,
        enable_plateau_stop,
        X_eval=None,
        y_eval=None,
        selection_bootstrap_samples=0,
        selection_seed=RANDOM_STATE,
        training_excess_complexity_ceiling=None,
        initial_candidates=None,
        progress_phase='model_fit',
        config_id=None,
        config_index=None,
        config_total=None,
        fold_index=None,
        fold_total=None,
        plateau_window_iterations=None,
        similar_loss_relative_tolerance=0.001,
    ):
        """Fit one PySR model in bounded warm-start chunks.

        PySR 1.5.9 treats each warm-started ``fit`` budget as additional work,
        so this method tracks the total requested iterations and wall time
        externally.  Dataset, operators, objective, constraints, population
        count, and population size remain constant across chunks.
        """
        config = dict(model_config)
        if training_excess_complexity_ceiling is not None:
            config['loss_function'] = make_loss_function(
                self.required_feature_count, training_excess_complexity_ceiling
            )
        config.pop('niterations', None)
        config.pop('timeout_in_seconds', None)
        config.update({
            'niterations': min(int(chunk_iterations), int(max_iterations)),
            'timeout_in_seconds': float(max_wall_seconds),
            'warm_start': True,
            'output_directory': str(self.pysr_run_dir),
            'run_id': run_id,
        })
        base_seed = int(config.get('random_state', RANDOM_STATE))
        model = create_pysr_regressor(**config)
        # Rewrite the manifest once Julia exists so the configured and actual
        # thread counts are both persisted, not merely printed to the log.
        if not getattr(self, '_runtime_manifest_observed_written', False):
            self.write_run_config()
            self._runtime_manifest_observed_written = True
        stopper = ChunkEarlyStopper(
            min_iterations=int(minimum_iterations),
            patience=int(plateau_patience),
            min_score_improvement=float(minimum_score_improvement),
            require_score=X_eval is not None,
            window_iterations=plateau_window_iterations,
        )

        started = monotonic()
        process_started = process_time()
        requested_iterations = 0
        chunks = []
        stop_reason = 'iteration_limit'
        previous_training_frontier_points = []
        previous_validation_frontier_points = []
        refinement_history = []
        refinement_reference = []
        refinement_simplicity = {}
        candidate_archive = {}
        if initial_candidates is not None:
            for _, row in initial_candidates.iterrows():
                source = row.get('refit_source_equation')
                key = (str(row['equation']), source if isinstance(source, str) else None)
                candidate_archive[key] = row.copy()

        while requested_iterations < int(max_iterations):
            elapsed_before = monotonic() - started
            remaining_seconds = float(max_wall_seconds) - elapsed_before
            if remaining_seconds <= 0:
                stop_reason = 'wall_time_limit'
                break

            current_chunk = min(
                int(chunk_iterations),
                int(max_iterations) - requested_iterations,
            )
            chunk_number = len(chunks) + 1
            chunk_seed = base_seed + chunk_number - 1
            model.set_params(
                niterations=current_chunk,
                timeout_in_seconds=max(1.0, remaining_seconds),
                random_state=chunk_seed,
            )
            fit_wall_started = monotonic()
            fit_cpu_started = process_time()
            fit_pysr_regressor(
                model,
                X_train,
                y_train,
                variable_names=self.feature_columns,
            )
            fit_wall_seconds = monotonic() - fit_wall_started
            fit_process_cpu_seconds = process_time() - fit_cpu_started
            requested_iterations += current_chunk

            self._retain_chunk_candidates(model, candidate_archive)
            pareto_rows = self._pareto_rows(model)
            evaluation_X = X_eval if X_eval is not None else X_train
            evaluation_y = y_eval if y_eval is not None else y_train
            # Intermediate plateau checks need finite predictions and raw R²,
            # but not 200 paired-bootstrap resamples for every Pareto row.
            # The declared one-SE selection is recomputed with the full paired
            # bootstrap exactly once at the terminal chunk below.
            candidate_audits = evaluate_pareto_candidates(
                model,
                pareto_rows,
                evaluation_X,
                evaluation_y,
                self.feature_columns,
                bootstrap_samples=0,
                bootstrap_seed=int(selection_seed),
                required_feature_count=self.required_feature_count,
                excess_complexity_ceiling=training_excess_complexity_ceiling,
            )
            # Plateau progress uses the explicit *excess* term budget rather
            # than PySR's raw node count.  Adding a direct x**2 term therefore
            # remains on the same zero-excess budget as a linear-only formula.
            frontier_rows = [
                {
                    'equation': row['equation'],
                    'loss': row['loss'],
                    'complexity': row['excess_complexity'],
                }
                for row in candidate_audits
                if row.get('eligible') and (
                    training_excess_complexity_ceiling is None
                    or row['excess_complexity'] <= training_excess_complexity_ceiling
                )
            ]
            training_frontier = feasible_frontier_stats(
                frontier_rows,
                self.feature_columns,
                all_features_verified=True,
            )
            training_frontier['complexity_metric'] = 'excess_term_budget_complexity'
            training_frontier_improved = frontier_improved(
                previous_training_frontier_points,
                training_frontier['points'],
                relative_tolerance=minimum_loss_relative_improvement,
            )
            previous_training_frontier_points = list(training_frontier['points'])

            validation_frontier = None
            if X_eval is not None:
                validation_frontier = validation_frontier_stats(candidate_audits)
                validation_frontier['complexity_metric'] = (
                    'excess_term_budget_complexity'
                )
                plateau_frontier_improved = validation_frontier_improved(
                    previous_validation_frontier_points,
                    validation_frontier['points'],
                    minimum_improvement=minimum_score_improvement,
                )
                best_validation_by_complexity = {}
                for point in (
                    list(previous_validation_frontier_points)
                    + list(validation_frontier['points'])
                ):
                    complexity = int(point['complexity'])
                    best_validation_by_complexity[complexity] = max(
                        float(point['raw_r2']),
                        best_validation_by_complexity.get(complexity, -np.inf),
                    )
                previous_validation_frontier_points = [
                    {
                        'complexity': complexity,
                        'raw_r2': best_validation_by_complexity[complexity],
                    }
                    for complexity in sorted(best_validation_by_complexity)
                ]
                plateau_frontier_type = 'validation_raw_r2_upper_envelope'
            else:
                # Refinement has no held-out fold.  Its bounded plateau detector
                # therefore continues to use the outer-training loss frontier.
                plateau_frontier_improved = training_frontier_improved
                plateau_frontier_type = 'outer_training_loss_lower_envelope'
                if plateau_window_iterations is not None:
                    points = []
                    for row in candidate_audits:
                        if not row.get('eligible'):
                            continue
                        source = pareto_rows[int(row['candidate_position'])].get('refit_source_equation')
                        if not isinstance(source, str):
                            source = row.get('backend_equation') or row['equation']
                        if source not in refinement_simplicity:
                            template, parameters, _ = parameterize_floating_constants(
                                parse_source_expression(source, self.feature_columns)
                            )
                            refinement_simplicity[source] = [
                                int(row['excess_complexity']),
                                int(row['term_budget']['total_term_count']),
                                len(parameters),
                                sum(1 for _ in sp.preorder_traversal(template)),
                            ]
                        points.append({
                            'loss': float(row['loss']),
                            'simplicity': refinement_simplicity[source],
                        })
                    refinement_history, plateau_frontier_improved = update_refinement_frontier(
                        refinement_history, points,
                        relative_tolerance=minimum_loss_relative_improvement,
                        similar_loss_tolerance=similar_loss_relative_tolerance,
                        reference=refinement_reference,
                    )
                    if plateau_frontier_improved:
                        refinement_reference = list(refinement_history)
                    plateau_frontier_type = 'historical_outer_training_loss_and_simplicity'

            selected_candidate = None
            candidate_selection_error = None
            try:
                if X_eval is not None:
                    checkpoint_candidates = [
                        dict(row) for row in candidate_audits if row.get('eligible')
                    ]
                    if not checkpoint_candidates:
                        raise ValueError(
                            'no finite all-feature Pareto candidate on validation data'
                        )
                    selected_candidate = min(
                        checkpoint_candidates,
                        key=lambda row: (
                            -float(row['raw_r2']),
                            int(row['excess_complexity']),
                            float(row['loss']),
                            int(row['structural_complexity']),
                            int(row['candidate_position']),
                        ),
                    )
                    selected_candidate['selection_rule'] = (
                        'checkpoint_best_raw_validation_r2'
                    )
                else:
                    selected_candidate = select_training_candidate(
                        candidate_audits,
                        excess_complexity_ceiling=(
                            training_excess_complexity_ceiling
                        ),
                    )
            except ValueError as exc:
                candidate_selection_error = str(exc)

            selected_has_all_features = selected_candidate is not None
            plateau_raw_r2 = (
                float(selected_candidate['raw_r2'])
                if selected_candidate is not None and X_eval is not None
                else None
            )

            elapsed = monotonic() - started
            # A timeout may truncate the requested chunk; it cannot advance a window.
            plateau_iterations = requested_iterations - (
                current_chunk if elapsed >= float(max_wall_seconds) else 0
            )
            should_stop, _ = stopper.observe(
                cumulative_iterations=plateau_iterations,
                has_all_features=selected_has_all_features,
                score=plateau_raw_r2,
                frontier_improved=plateau_frontier_improved,
            )
            chunk_decision = 'continue'
            if elapsed >= float(max_wall_seconds):
                chunk_decision = 'wall_time_limit'
            elif enable_plateau_stop and should_stop:
                chunk_decision = (
                    'validated_plateau' if X_eval is not None else 'pareto_plateau'
                )
            elif requested_iterations >= int(max_iterations):
                chunk_decision = 'iteration_limit'

            terminal_selection = chunk_decision != 'continue'
            bootstrap_samples_used = 0
            if terminal_selection and X_eval is not None:
                bootstrap_samples_used = int(selection_bootstrap_samples)
                candidate_audits = evaluate_pareto_candidates(
                    model,
                    pareto_rows,
                    evaluation_X,
                    evaluation_y,
                    self.feature_columns,
                    bootstrap_samples=bootstrap_samples_used,
                    bootstrap_seed=int(selection_seed),
                    required_feature_count=self.required_feature_count,
                    excess_complexity_ceiling=training_excess_complexity_ceiling,
                )
                try:
                    selected_candidate = select_validation_candidate(
                        candidate_audits
                    )
                    candidate_selection_error = None
                except ValueError as exc:
                    selected_candidate = None
                    candidate_selection_error = str(exc)

            selected_has_all_features = selected_candidate is not None
            selected_raw_r2 = (
                float(selected_candidate['raw_r2'])
                if selected_candidate is not None and X_eval is not None
                else None
            )

            elapsed = monotonic() - started
            process_cpu_seconds = process_time() - process_started
            chunk_record = {
                'run_id': run_id,
                'chunk': chunk_number,
                'chunk_seed': chunk_seed,
                'requested_chunk_iterations': current_chunk,
                'requested_iterations_total': requested_iterations,
                'elapsed_seconds': elapsed,
                'fit_wall_seconds': fit_wall_seconds,
                'fit_process_cpu_seconds': fit_process_cpu_seconds,
                'process_cpu_seconds_total': process_cpu_seconds,
                'effective_core_equivalents': (
                    process_cpu_seconds / elapsed if elapsed > 0 else None
                ),
                'logical_cpu_utilization_fraction': (
                    process_cpu_seconds / elapsed / LOGICAL_PROCESSORS
                    if elapsed > 0 else None
                ),
                'allocated_thread_utilization_fraction': (
                    process_cpu_seconds / elapsed / JULIA_THREADS
                    if elapsed > 0 else None
                ),
                'raw_validation_r2': selected_raw_r2,
                'plateau_raw_validation_r2': plateau_raw_r2,
                'candidate_selection_context': (
                    'fold_validation' if X_eval is not None else 'outer_training'
                ),
                'selection_bootstrap_requested_samples': bootstrap_samples_used,
                'selection_bootstrap_policy': (
                    'terminal_chunk_only' if X_eval is not None else 'not_applicable'
                ),
                'selection_bootstrap_seed': int(selection_seed),
                'selected_candidate': selected_candidate,
                'candidate_selection_error': candidate_selection_error,
                'candidate_audits': candidate_audits,
                'all_features_used': selected_has_all_features,
                'feasible_frontier': training_frontier,
                'validation_frontier': validation_frontier,
                'training_pareto_frontier_improved': training_frontier_improved,
                'pareto_frontier_improved': plateau_frontier_improved,
                'plateau_frontier_type': plateau_frontier_type,
                'plateau_checks': stopper.plateau_checks,
                'plateau_window_iterations': plateau_window_iterations,
                'plateau_completed_iterations': plateau_iterations,
                'last_progress_iteration': stopper.last_progress_iteration,
                'historical_refinement_frontier': refinement_history,
                'decision': chunk_decision,
            }
            chunks.append(chunk_record)
            self._record_progress(chunk_record)

            chunk_total = int(math.ceil(int(max_iterations) / int(chunk_iterations)))
            configuration_status = None
            if config_index is not None and config_total is not None:
                configuration_status = {
                    'index': int(config_index),
                    'total': int(config_total),
                    'config_id': str(config_id or run_id),
                }
            fold_status = None
            if fold_index is not None and fold_total is not None:
                fold_status = {
                    'index': int(fold_index),
                    'total': int(fold_total),
                }
            concise_status = {
                'phase': str(progress_phase),
                'status': (
                    'fit_chunk_completed'
                    if chunk_decision == 'continue'
                    else 'fit_terminal'
                ),
                'run_id': str(run_id),
                'configuration': configuration_status,
                'fold': fold_status,
                'chunk': {
                    'index': int(chunk_number),
                    'total': chunk_total,
                },
                'iterations': {
                    'requested': int(requested_iterations),
                    'maximum': int(max_iterations),
                },
                'timing': {
                    'elapsed_seconds': float(elapsed),
                    'fit_wall_seconds': float(fit_wall_seconds),
                },
                'metrics': {
                    'raw_validation_r2': selected_raw_r2,
                    'all_features_used': bool(selected_has_all_features),
                },
                'cpu': {
                    'effective_core_equivalents': chunk_record[
                        'effective_core_equivalents'
                    ],
                    'logical_cpu_utilization_fraction': chunk_record[
                        'logical_cpu_utilization_fraction'
                    ],
                    'visible_logical_processors': int(LOGICAL_PROCESSORS),
                    'allocated_julia_threads': int(JULIA_THREADS),
                    'allocated_thread_utilization_fraction': chunk_record[
                        'allocated_thread_utilization_fraction'
                    ],
                },
                'runtime': {
                    'configured_julia_threads': int(JULIA_THREADS),
                    'actual_julia_threads': _ACTUAL_JULIA_THREADS,
                    'blas_threads': int(BLAS_THREADS),
                },
                'decision': str(chunk_decision),
            }
            self._write_progress_status(concise_status)

            progress_labels = []
            if configuration_status is not None:
                progress_labels.append(
                    f"config {configuration_status['index']}/"
                    f"{configuration_status['total']}"
                )
            elif progress_phase == 'refinement':
                progress_labels.append('refinement')
            else:
                progress_labels.append(str(progress_phase))
            if fold_status is not None:
                progress_labels.append(
                    f"fold {fold_status['index']}/{fold_status['total']}"
                )
            progress_labels.extend([
                f"chunk {chunk_number}/{chunk_total}",
                f"iterations {requested_iterations}/{int(max_iterations)}",
                f"elapsed {elapsed:.1f}s",
            ])
            r2_label = (
                f"{selected_raw_r2:.6f}"
                if selected_raw_r2 is not None
                else 'n/a'
            )
            metric_label = f"raw validation R²={r2_label}"
            if X_eval is None:
                metric_label = (
                    f"train R²={selected_candidate['raw_r2']:.6f} | "
                    f"loss={selected_candidate['loss']:.6g}"
                    if selected_candidate is not None
                    else "train R²=n/a | loss=n/a"
                )
            effective_cores = chunk_record['effective_core_equivalents']
            utilization = chunk_record['logical_cpu_utilization_fraction']
            cpu_label = (
                f"{effective_cores:.1f} effective cores "
                f"({100.0 * utilization:.1f}% of {LOGICAL_PROCESSORS} logical)"
                if effective_cores is not None and utilization is not None
                else 'n/a'
            )
            getattr(self, 'logger', logging.getLogger(__name__)).info(
                "[Progress] %s | %s | CPU=%s | decision=%s",
                ' | '.join(progress_labels),
                metric_label,
                cpu_label,
                chunk_decision,
            )

            if chunk_decision != 'continue':
                stop_reason = chunk_decision
                break

        total_elapsed = monotonic() - started
        total_process_cpu = process_time() - process_started
        summary = {
            'run_id': run_id,
            'stop_reason': stop_reason,
            'requested_iterations': requested_iterations,
            'maximum_iterations': int(max_iterations),
            'inherited_candidate_count': (
                len(initial_candidates) if initial_candidates is not None else 0
            ),
            'search_excess_complexity_ceiling': training_excess_complexity_ceiling,
            'elapsed_seconds': total_elapsed,
            'process_cpu_seconds': total_process_cpu,
            'effective_core_equivalents': (
                total_process_cpu / total_elapsed if total_elapsed > 0 else None
            ),
            'logical_cpu_utilization_fraction': (
                total_process_cpu / total_elapsed / LOGICAL_PROCESSORS
                if total_elapsed > 0 else None
            ),
            'allocated_thread_utilization_fraction': (
                total_process_cpu / total_elapsed / JULIA_THREADS
                if total_elapsed > 0 else None
            ),
            'chunks': chunks,
            'selected_candidate': (
                chunks[-1]['selected_candidate'] if chunks else None
            ),
        }
        return model, summary

    def review_final_archive(self, model, cv_splits, excess_complexity_ceiling):
        """Select an archived structure by fold-only coefficient CV, then refit."""
        self.logger.info("Reviewing archived structures with 4-fold x %d-repeat coefficient CV",
                         FINAL_REVIEW_N_REPEATS)
        selected_rows, audit = review_final_candidates(
            model.equations_, self.X_train_raw, self.y_train,
            self.feature_columns, cv_splits, self.scaler.mean_, self.scaler.scale_,
            required_feature_count=self.required_feature_count,
            excess_complexity_ceiling=excess_complexity_ceiling,
            n_splits=CV_N_SPLITS, n_repeats=FINAL_REVIEW_N_REPEATS,
            n_jobs=FOLD_WORKERS,
        )
        audit['outer_training_names'] = self.train_names.tolist()
        if not selected_rows.empty:
            audit['selected_candidate']['candidate_position'] = len(model.equations_)
            audit['comparison_files'] = write_candidate_comparison(
                audit, self.output_path, self.feature_columns,
                self.scaler.mean_, self.scaler.scale_,
            )
        atomic_write_json(Path(self.output_path) / 'final_candidate_cv.json', audit)
        if selected_rows.empty:
            raise ValueError(
                'No formula passed final candidate CV review; see final_candidate_cv.json'
            )
        position = len(model.equations_)
        model.equations_ = pd.concat([model.equations_, selected_rows], ignore_index=True)
        selected = dict(audit['selected_candidate'], candidate_position=position)
        self.final_candidate_cv = {
            'audit_file': 'final_candidate_cv.json',
            'selection_rule': audit['selection_rule'],
            'selected_candidate': selected,
            'elapsed_seconds': audit['elapsed_seconds'],
            'candidate_layers': audit.get('candidate_layers', {}),
            'comparison_files': audit.get('comparison_files', {}),
        }
        self.search_summary['final_candidate_cv'] = self.final_candidate_cv
        self.logger.info(
            "Final candidate CV review completed in %.2f seconds; selected position %d, "
            "conditional CV R²=%.6f, excess complexity=%d",
            audit['elapsed_seconds'], position, selected['cv_raw_r2_mean'],
            selected['excess_complexity'],
        )
        return selected

    def get_equations_scaled_and_raw(self):
        """
        Return two equation strings:
        - scaled_eq: formula output by the model (variables represent standardized values)
        - raw_eq: formula with z=(x-mean)/std substituted back to original features
        """
        expr_scaled = self._equation_at(self.model, self.final_candidate_position)

        expr_raw = standardized_to_raw_expression(
            expr_scaled,
            self.feature_columns,
            self.scaler.mean_,
            self.scaler.scale_,
        )

        return str(expr_scaled), str(expr_raw)

    def analyze_equation_units_and_uncertainty(
        self,
        equation_scaled,
        equation_raw,
        y_pred_train,
        y_pred_test,
    ):
        """Audit equation spaces and estimate fixed-structure coefficient CIs."""
        if not equation_scaled or not equation_raw:
            raise RuntimeError("Scaled and raw equations are required for the unit audit")

        scaled_train = evaluate_expression(
            equation_scaled, self.X_train, self.feature_columns
        )
        raw_train = evaluate_expression(
            equation_raw, self.X_train_raw, self.feature_columns
        )
        scaled_test = evaluate_expression(
            equation_scaled, self.X_test, self.feature_columns
        )
        raw_test = evaluate_expression(
            equation_raw, self.X_test_raw, self.feature_columns
        )
        scaled_full = evaluate_expression(
            equation_scaled,
            self.scaler.transform(self.X),
            self.feature_columns,
        )
        raw_full = evaluate_expression(equation_raw, self.X, self.feature_columns)

        consistency = {
            'rtol': 1e-8,
            'atol': 1e-10,
            'train_scaled_vs_model_max_abs_error': float(
                np.max(np.abs(scaled_train - y_pred_train))
            ),
            'train_raw_vs_model_max_abs_error': float(
                np.max(np.abs(raw_train - y_pred_train))
            ),
            'test_scaled_vs_model_max_abs_error': float(
                np.max(np.abs(scaled_test - y_pred_test))
            ),
            'test_raw_vs_model_max_abs_error': float(
                np.max(np.abs(raw_test - y_pred_test))
            ),
            'full_library_raw_vs_scaled_max_abs_error': float(
                np.max(np.abs(raw_full - scaled_full))
            ),
        }
        consistency['accepted'] = bool(
            np.allclose(scaled_train, y_pred_train, rtol=1e-8, atol=1e-10)
            and np.allclose(raw_train, y_pred_train, rtol=1e-8, atol=1e-10)
            and np.allclose(scaled_test, y_pred_test, rtol=1e-8, atol=1e-10)
            and np.allclose(raw_test, y_pred_test, rtol=1e-8, atol=1e-10)
            and np.allclose(raw_full, scaled_full, rtol=1e-8, atol=1e-10)
        )
        if not consistency['accepted']:
            raise RuntimeError(
                'De-standardized equation failed numerical consistency checks; '
                f'details={consistency}'
            )

        feature_rows = []
        for index, feature in enumerate(self.feature_columns):
            metadata = FEATURE_METADATA[feature]
            feature_rows.append(
                {
                    'feature': feature,
                    'quantity': metadata['quantity'],
                    'raw_source_unit': metadata['source_unit'],
                    'model_input': f'z_{feature}',
                    'model_input_unit': '1 (dimensionless)',
                    'standardization_mean': float(self.scaler.mean_[index]),
                    'standardization_scale': float(self.scaler.scale_[index]),
                    'standardization': (
                        f'z_{feature} = ({feature} - '
                        f'{self.scaler.mean_[index]:.16g}) / '
                        f'{self.scaler.scale_[index]:.16g}'
                    ),
                }
            )

        uncertainty = bootstrap_fixed_structure_coefficients(
            equation_scaled,
            self.X_train,
            self.y_train,
            self.feature_columns,
            n_bootstrap=COEFFICIENT_UNCERTAINTY_CONFIG['bootstrap_samples'],
            confidence_level=COEFFICIENT_UNCERTAINTY_CONFIG['confidence_level'],
            random_state=COEFFICIENT_UNCERTAINTY_CONFIG['random_state'],
            max_nfev=COEFFICIENT_UNCERTAINTY_CONFIG['max_nfev_per_sample'],
            max_wall_seconds=COEFFICIENT_UNCERTAINTY_CONFIG['max_wall_seconds'],
        )
        parameterized_raw = standardized_to_raw_expression(
            uncertainty['parameterized_equation_scaled'],
            self.feature_columns,
            self.scaler.mean_,
            self.scaler.scale_,
            preserved_symbol_names=uncertainty['parameter_names'],
        )
        uncertainty['parameterized_equation_raw'] = str(parameterized_raw)

        sample_rows = uncertainty.pop('sample_rows')
        term_budget = analyze_term_budget(equation_scaled, self.feature_columns)
        if term_budget['analysis_error'] is not None:
            raise RuntimeError(
                'Selected equation failed the term-budget audit: '
                f"{term_budget['analysis_error']}"
            )
        audit = {
            'schema_version': 2,
            'target': {
                'name': self.target_column,
                **TARGET_METADATA,
            },
            'normalization': {
                'feature_transform': 'outer-training StandardScaler Z-score',
                'equation_scaled_inputs_dimensionless': True,
                'target_transformed_by_sr': False,
                'raw_equation_substitution': 'z_i = (x_i - mean_i) / scale_i',
            },
            'features': feature_rows,
            'equations': {
                'equation_scaled': equation_scaled,
                'equation_raw': equation_raw,
                'parameterized_equation_scaled': uncertainty[
                    'parameterized_equation_scaled'
                ],
                'parameterized_equation_raw': uncertainty[
                    'parameterized_equation_raw'
                ],
            },
            'unit_interpretation': [
                'The searched equation uses dimensionless Z-scored feature inputs.',
                'Score is already dimensionless and is not standardized again by SR.',
                'The raw equation is an exact scaler substitution and must be evaluated '
                'with the listed source units.',
                'For nonlinear functions, the raw equation is a predictive re-expression '
                'of the dimensionless model, not an independently derived physical law.',
            ],
            'numerical_consistency': consistency,
            'complexity_policy': {
                'pysr_parsimony': 0,
                'pysr_frequency_balancing_role': (
                    'search-size coverage, not candidate scoring'
                ),
                'pysr_structural_complexity_role': (
                    'audit field and maxsize/maxdepth resource cap only'
                ),
                'selection_metric': 'excess_term_budget_complexity',
                'selected_equation_term_budget': term_budget,
            },
            'coefficient_uncertainty': uncertainty,
        }
        self.coefficient_bootstrap_samples = sample_rows
        self.equation_audit = audit

        self.logger.info("\n" + "="*80)
        self.logger.info("Equation normalization, units, and coefficient uncertainty")
        self.logger.info("="*80)
        self.logger.info(
            "Target: Score = z_PLQY - z_XRD; unit=1 (dimensionless); "
            "SR target standardization=none"
        )
        self.logger.info(
            "Complexity policy: PySR parsimony=0; one linear and one direct "
            "nonlinear term per feature are free; selected equation excess "
            "complexity=%d; direct quadratic present=%s",
            term_budget['excess_complexity'],
            term_budget['has_direct_quadratic'],
        )
        self.logger.info("Feature inputs used by PySR are Z-scores and dimensionless:")
        for row in feature_rows:
            self.logger.info(
                "  %s: raw unit=%s; mean=%.10g; scale=%.10g; %s",
                row['feature'],
                row['raw_source_unit'],
                row['standardization_mean'],
                row['standardization_scale'],
                row['standardization'],
            )
        self.logger.info("Scaled equation (dimensionless Z-score inputs):")
        self.logger.info("  %s = %s", self.target_column, equation_scaled)
        self.logger.info("Raw-feature equation (use the source units listed above):")
        self.logger.info("  %s = %s", self.target_column, equation_raw)
        self.logger.info(
            "Raw/scaled equation audit accepted on train, test, and all 208 library rows; "
            "full-library max absolute difference=%.3e",
            consistency['full_library_raw_vs_scaled_max_abs_error'],
        )
        self.logger.info(
            "Coefficient uncertainty method: fixed symbolic structure, fixed outer-training "
            "scaler, outer-training row bootstrap only; external test not used"
        )
        self.logger.info(
            "Bootstrap status=%s; successful=%d/%d; elapsed=%.2fs",
            uncertainty['status'],
            uncertainty['successful_samples'],
            uncertainty['settings']['requested_bootstrap_samples'],
            uncertainty['elapsed_seconds'],
        )
        self.logger.info(
            "Parameterized scaled equation: %s = %s",
            self.target_column,
            uncertainty['parameterized_equation_scaled'],
        )
        self.logger.info(
            "Parameterized raw equation: %s = %s",
            self.target_column,
            uncertainty['parameterized_equation_raw'],
        )
        for row in uncertainty['coefficient_rows']:
            self.logger.info(
                "  %s=%.10g; bootstrap SE=%s; %.1f%% CI=[%.10g, %.10g]",
                row['parameter'],
                row['estimate'],
                (
                    f"{row['bootstrap_standard_error']:.10g}"
                    if row['bootstrap_standard_error'] is not None
                    else 'N/A'
                ),
                row['confidence_level'] * 100.0,
                row['ci_low'],
                row['ci_high'],
            )
        for limitation in uncertainty['limitations']:
            self.logger.info("  Limitation: %s", limitation)
        return audit

    def load_data(self):
        """Generate the symbolic-regression table with the current Score version."""
        self.logger.info("Preparing symbolic-regression data from repository sources...")
        self.df, self.score_info = prepare_symbolic_regression_data(
            self.campaign_round
        )
        self.feature_selection = {
            'source': 'configured_candidate_pool',
            'feature_columns': list(self.feature_columns),
            'required_feature_count': self.required_feature_count,
            'outer_split': 'deterministic_stratified_duplex',
            'split_feature_columns': list(self.feature_columns),
        }
        self.logger.info('SR input features (%s): %s',
                         self.feature_selection['source'], self.feature_columns)
        if 'source_row_index' in self.df.columns:
            raise ValueError('derived SR data unexpectedly contains source_row_index')
        # Stable, zero-based data.csv row number (the CSV header is not counted).
        self.df.insert(1, 'source_row_index', np.arange(len(self.df), dtype=int))
        self.logger.info(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        self.logger.info(f"Columns: {list(self.df.columns)}")

        self.df.columns = self.df.columns.str.strip()
        self._validate_required_inputs()
        if not self.resume_mode:
            self.df.to_csv(self.data_path, index=False, encoding='utf-8')
            write_score_provenance(self.output_path, self.score_info)
            self.write_run_config()
        self.logger.info(f"Score version: {self.score_info.get('score_version', 'unknown')}")
        return True

    def _validate_required_inputs(self):
        """Reject invalid required values without changing the molecule set."""
        required = list(dict.fromkeys(
            self.feature_columns + [self.target_column]
        ))
        missing_cols = [col for col in required if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        numeric = self.df[required].apply(pd.to_numeric, errors='coerce')
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if invalid.any():
            details = [
                f"Name={self.df.iloc[row]['Name']}, column={required[col]}, "
                f"value={self.df.iloc[row][required[col]]!r}"
                for row, col in zip(*np.where(invalid))
            ]
            raise ValueError('Missing, nonnumeric or nonfinite SR inputs: ' + '; '.join(details))
        self.df[required] = numeric

    def preprocess_data(self):
        """Validate required values, split, and standardize."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Starting data preprocessing...")
        self.logger.info("="*80)

        self._validate_required_inputs()

        self.X = self.df[self.feature_columns].values.astype(float)
        self.y = self.df[self.target_column].values.astype(float)

        self.logger.info("\nFeature statistics:")
        stats_df = self.df[self.feature_columns].describe()
        self.logger.info(f"\n{stats_df.to_string()}")

        self.logger.info(f"\nTarget variable '{self.target_column}' statistics:")
        self.logger.info(f"  Samples: {len(self.y)}")
        self.logger.info(f"  Mean: {self.y.mean():.6f}")
        self.logger.info(f"  Std: {self.y.std():.6f}")
        self.logger.info(f"  Min: {self.y.min():.6f}")
        self.logger.info(f"  Max: {self.y.max():.6f}")
        self.logger.info(f"  Median: {np.median(self.y):.6f}")

        (
            _, _, self.y_train, self.y_test,
            self.train_indices, self.test_indices,
        ) = stratified_duplex_split(
            self.X, self.y,
            test_size=TEST_SIZE, n_strata=SPLIT_N_STRATA,
            metric=SPLIT_METRIC, random_state=RANDOM_STATE,
            return_indices=True,
        )

        self.X_train = self.X[self.train_indices]
        self.X_test = self.X[self.test_indices]

        names = self.df['Name'].astype(str).to_numpy()
        source_rows = self.df['source_row_index'].to_numpy(dtype=int)
        self.train_names = names[self.train_indices]
        self.test_names = names[self.test_indices]
        self.train_source_rows = source_rows[self.train_indices]
        self.test_source_rows = source_rows[self.test_indices]
        self.split_assignments = build_split_assignments(
            names,
            source_rows,
            self.train_indices,
            self.test_indices,
        )
        split_path = Path(self.output_path) / 'split_assignments.csv'
        if getattr(self, 'resume_mode', False):
            # Validate prospective bytes before touching the original run.
            sampled_configs = sample_unique_configs(
                self.param_grid, requested=N_RANDOM_SEARCHES, random_state=RANDOM_STATE
            )
            self._prepare_resume_contract(sampled_configs, {
                'derived_data_sha256': hashlib.sha256(
                    self.df.to_csv(index=False).encode('utf-8')
                ).hexdigest(),
                'split_assignments_sha256': hashlib.sha256(
                    pd.DataFrame(self.split_assignments).to_csv(index=False).encode('utf-8')
                ).hexdigest(),
            })
            self.df.to_csv(self.data_path, index=False, encoding='utf-8')
            write_score_provenance(self.output_path, self.score_info)
            self.write_run_config()
        pd.DataFrame(self.split_assignments).to_csv(
            split_path,
            index=False,
            encoding='utf-8',
        )

        self.logger.info('\nData split: SR stratified DUPLEX')
        self.logger.info(f"  Training set: {len(self.X_train)} ({len(self.X_train)/len(self.X)*100:.1f}%)")
        self.logger.info(f"  Test set: {len(self.X_test)} ({len(self.X_test)/len(self.X)*100:.1f}%)")
        self.logger.info(f"  Stable split assignments: {split_path}")

        self.logger.info("\nFeature standardization (based on training set statistics)...")
        self.X_train_raw = self.X_train.copy()
        self.X_test_raw = self.X_test.copy()
        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.logger.info("Standardization parameters (from training set):")
        for i, col in enumerate(self.feature_columns):
            self.logger.info(f"  {col}:")
            self.logger.info(f"    Mean = {self.scaler.mean_[i]:.6f}")
            self.logger.info(f"    Std = {self.scaler.scale_[i]:.6f}")

        self.logger.info("\nStandardized data range:")
        self.logger.info(f"  Training: [{self.X_train.min():.3f}, {self.X_train.max():.3f}]")
        self.logger.info(f"  Test: [{self.X_test.min():.3f}, {self.X_test.max():.3f}]")

        self.plot_target_distribution(self.y_train, self.y_test)

    @staticmethod
    def ridge_basis(X, quadratic=False):
        """Add direct raw squares only; scale each basis column during fitting."""
        return np.column_stack((X, np.square(X))) if quadratic else X

    def fit_ridge_baseline(self, cv_splits, *, quadratic=False):
        """Tune Ridge on the final SR descriptors and identical outer/CV rows."""
        feature_names = list(getattr(self, 'selected_feature_names', []))
        if (
            len(feature_names) != self.required_feature_count
            or len(set(feature_names)) != len(feature_names)
            or any(name not in self.feature_columns for name in feature_names)
        ):
            raise ValueError('Ridge requires the final SR selected_feature_names')
        self.ridge_feature_indices = [
            self.feature_columns.index(name) for name in feature_names
        ]
        X_selected = self.X_train_raw[:, self.ridge_feature_indices]
        X_basis = self.ridge_basis(X_selected, quadratic)
        basis_names = feature_names + ([f'{name}_squared' for name in feature_names] if quadratic else [])
        raw_terms = feature_names + ([f'{name}**2' for name in feature_names] if quadratic else [])
        prefix = 'ridge_quadratic' if quadratic else 'ridge'
        # The scaler depends on the fold, not alpha; reuse it across alpha fits.
        scaled_folds = [
            (train_idx, val_idx, *prepare_fold_scaled_data(X_basis, train_idx, val_idx))
            for train_idx, val_idx in cv_splits
        ]
        alpha_results = []
        for alpha in RIDGE_BASELINE_ALPHAS:
            fold_results = []
            for fold_number, (train_idx, val_idx, fold_scaler,
                              X_fold_train, X_fold_validation) in enumerate(scaled_folds, start=1):
                model = Ridge(alpha=float(alpha))
                model.fit(X_fold_train, self.y_train[train_idx])
                prediction = np.asarray(
                    model.predict(X_fold_validation), dtype=float
                ).reshape(-1)
                truth = self.y_train[val_idx]
                fold_results.append({
                    'fold': fold_number,
                    'repeat': (fold_number - 1) // CV_N_SPLITS + 1,
                    'raw_r2': float(r2_score(truth, prediction)),
                    'rmse': float(np.sqrt(mean_squared_error(truth, prediction))),
                    'mae': float(mean_absolute_error(truth, prediction)),
                    'fold_scaler_mean': fold_scaler.mean_.tolist(),
                    'fold_scaler_scale': fold_scaler.scale_.tolist(),
                })
            alpha_results.append({
                'alpha': float(alpha),
                'cv_raw_r2_mean': float(
                    np.mean([row['raw_r2'] for row in fold_results])
                ),
                'cv_raw_r2_std': float(
                    np.std([row['raw_r2'] for row in fold_results])
                ),
                'cv_rmse_mean': float(
                    np.mean([row['rmse'] for row in fold_results])
                ),
                'cv_rmse_std': float(
                    np.std([row['rmse'] for row in fold_results])
                ),
                'cv_mae_mean': float(
                    np.mean([row['mae'] for row in fold_results])
                ),
                'cv_mae_std': float(
                    np.std([row['mae'] for row in fold_results])
                ),
                'fold_results': fold_results,
            })

        selected = min(
            alpha_results,
            key=lambda row: (
                -float(row['cv_raw_r2_mean']),
                float(row['cv_rmse_mean']),
                float(row['alpha']),
            ),
        )
        baseline_scaler = StandardScaler()
        X_outer_train = baseline_scaler.fit_transform(X_basis)
        baseline_model = Ridge(alpha=float(selected['alpha']))
        baseline_model.fit(X_outer_train, self.y_train)
        train_prediction = np.asarray(
            baseline_model.predict(X_outer_train), dtype=float
        ).reshape(-1)

        scaled_coefficients = np.asarray(baseline_model.coef_, dtype=float)
        raw_coefficients = scaled_coefficients / baseline_scaler.scale_
        raw_intercept = float(
            baseline_model.intercept_
            - np.sum(
                scaled_coefficients
                * baseline_scaler.mean_
                / baseline_scaler.scale_
            )
        )
        scaled_equation = ' + '.join(
            [f'({float(baseline_model.intercept_):.16g})']
            + [
                f'({coefficient:.16g})*z_{feature}'
                for feature, coefficient in zip(
                    basis_names, scaled_coefficients
                )
            ]
        )
        raw_equation = ' + '.join(
            [f'({raw_intercept:.16g})']
            + [
                f'({coefficient:.16g})*{feature}'
                for feature, coefficient in zip(
                    raw_terms, raw_coefficients
                )
            ]
        )

        setattr(self, f'{prefix}_baseline_scaler', baseline_scaler)
        setattr(self, f'{prefix}_baseline_model', baseline_model)
        setattr(self, f'{prefix}_train_predictions', train_prediction)
        baseline = {
            'model': 'Ridge',
            'basis': 'linear_and_direct_raw_squares' if quadratic else 'linear',
            'basis_names': basis_names,
            'basis_scaling': 'each_raw_basis_column_standardized_on_training_rows',
            'feature_source': 'final_sr_selected_feature_names',
            'feature_names': feature_names,
            'outer_split': 'identical_to_symbolic_regression',
            'cv_scope': 'alpha_tuning_conditional_on_final_sr_features',
            'selection_metric': 'mean_raw_validation_r2',
            'selection_rule': (
                'highest_mean_raw_cv_r2_then_rmse_then_lower_alpha'
            ),
            'selected_alpha': float(selected['alpha']),
            'cv_raw_r2_mean': float(selected['cv_raw_r2_mean']),
            'cv_raw_r2_std': float(selected['cv_raw_r2_std']),
            'cv_rmse_mean': float(selected['cv_rmse_mean']),
            'cv_rmse_std': float(selected['cv_rmse_std']),
            'cv_mae_mean': float(selected['cv_mae_mean']),
            'cv_mae_std': float(selected['cv_mae_std']),
            'train_r2': float(r2_score(self.y_train, train_prediction)),
            'train_rmse': float(
                np.sqrt(mean_squared_error(self.y_train, train_prediction))
            ),
            'train_mae': float(
                mean_absolute_error(self.y_train, train_prediction)
            ),
            'equation_scaled': scaled_equation,
            'equation_raw': raw_equation,
            'scaled_intercept': float(baseline_model.intercept_),
            'scaled_coefficients': {
                feature: float(value)
                for feature, value in zip(
                    basis_names, scaled_coefficients
                )
            },
            'raw_intercept': raw_intercept,
            'raw_coefficients': {
                feature: float(value)
                for feature, value in zip(raw_terms, raw_coefficients)
            },
            'alpha_results': alpha_results,
            'external_test_evaluations': 0,
        }
        setattr(self, f'{prefix}_baseline', baseline)
        self.logger.info("\n%s baseline (final SR features, same outer split and CV folds):", prefix)
        self.logger.info("  Features: %s", feature_names)
        self.logger.info("  Selected alpha: %.6g", selected['alpha'])
        self.logger.info(
            "  Raw CV R²: %.6f ± %.6f",
            selected['cv_raw_r2_mean'],
            selected['cv_raw_r2_std'],
        )
        self.logger.info("  External test remains locked until final evaluation")

    def run_symbolic_regression(self):
        """Run the configured fixed-seed search with bounded fits and fold racing."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Starting symbolic regression analysis")
        self.logger.info("="*80)

        if not self.enable_refinement:
            error_msg = "Error: refinement training must be enabled (enable_refinement=True) to run analysis!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        param_distributions = self.param_grid
        n_random_searches = N_RANDOM_SEARCHES
        sampled_configs = sample_unique_configs(
            param_distributions,
            requested=n_random_searches,
            random_state=RANDOM_STATE,
        )
        self._prepare_resume_contract(sampled_configs)
        atomic_write_json(
            Path(self.output_path) / 'sampled_configs.json',
            {
                'schema_version': 2,
                'profile': 'publication',
                'random_state': RANDOM_STATE,
                'requested_unique_configurations': n_random_searches,
                'sampled_unique_configurations': len(sampled_configs),
                'sampling_policy': 'fixed_seed_without_replacement',
                'configurations': sampled_configs,
            },
        )
        terminal_results_by_id = self._load_configuration_checkpoints(
            sampled_configs
        )

        cv_splits = list(create_cv(
            self.X_train_raw,
            self.y_train,
            n_splits=CV_N_SPLITS,
            n_repeats=CV_N_REPEATS,
            n_strata=CV_N_STRATA,
            random_state=RANDOM_STATE,
            method=CV_METHOD
        ))

        total_folds = len(cv_splits)
        self.logger.info("\nCross-validation configuration:")
        self.logger.info(f"  Method: {CV_METHOD}")
        if CV_METHOD == 'repeated_stratified':
            self.logger.info(f"  Splits: {CV_N_SPLITS}")
            self.logger.info(f"  Repeats: {CV_N_REPEATS}")
            self.logger.info(f"  Total folds: {total_folds}")
        else:
            self.logger.info(f"  Folds: {total_folds}")
        self.logger.info(f"  Strata: {CV_N_STRATA}")

        terminal_results = sorted(
            terminal_results_by_id.values(),
            key=lambda row: int(row['iteration']),
        )
        loaded_terminal_configurations = len(terminal_results)
        all_results = [
            row for row in terminal_results if row['status'] == 'completed'
        ]
        pruned_results = [
            row
            for row in terminal_results
            if row['status'] == 'conservatively_pruned'
        ]
        failed_results = [
            row for row in terminal_results if row['status'] == 'failed'
        ]
        attempted_configs = len(terminal_results)
        completed_configs = len(all_results)
        incumbent_result = (
            max(all_results, key=lambda row: float(row['cv_raw_r2_mean']))
            if all_results
            else None
        )
        incumbent_score = (
            float(incumbent_result['cv_raw_r2_mean'])
            if incumbent_result is not None
            else -np.inf
        )
        incumbent_fold_scores = (
            [float(row['raw_r2']) for row in incumbent_result['fold_results']]
            if incumbent_result is not None
            else None
        )
        self._search_started_monotonic = monotonic()
        self._overall_search_progress = {
            'terminal_configurations': len(terminal_results),
            'total_configurations': len(sampled_configs),
            'completed_configurations': completed_configs,
            'conservatively_pruned_configurations': len(pruned_results),
            'failed_configurations': len(failed_results),
            'search_elapsed_seconds': 0.0,
        }
        self._write_progress_status({
            'phase': 'configuration_search',
            'status': 'resume_started' if self.resume_mode else 'search_started',
            'configuration': None,
            'fold': None,
            'decision': 'resume' if self.resume_mode else 'continue',
        })

        if terminal_results:
            if len(terminal_results) == len(sampled_configs):
                self.logger.info(
                    "Validated all %d terminal configuration checkpoints; "
                    "recomputing CV selection without starting fold workers",
                    len(terminal_results),
                )
            else:
                self.logger.info(
                    "Validated %d terminal configuration checkpoint(s); "
                    "continuing at configuration %d/%d",
                    len(terminal_results),
                    len(terminal_results) + 1,
                    len(sampled_configs),
                )

        self.logger.info("\nPublication search execution:")
        self.logger.info(f"  Unique configurations: {len(sampled_configs)}")
        self.logger.info("  Global target/patience/window termination: not configured")
        self.logger.info("  Every sampled configuration will reach completed, pruned, or failed status")
        self.logger.info(f"  JuliaCall threads: {os.environ.get('PYTHON_JULIACALL_THREADS')}")
        self.logger.info(
            "  CV folds: %d spawned worker process(es) x %d Julia thread(s); "
            "aggregate=%d logical threads",
            FOLD_WORKERS,
            FOLD_WORKER_THREADS,
            FOLD_EXECUTION_LAYOUT.aggregate_threads,
        )

        for i, sampled_config in enumerate(sampled_configs):
            config_id = sampled_config['config_id']
            if config_id in terminal_results_by_id:
                saved_result = terminal_results_by_id[config_id]
                self.logger.info(
                    "Resume skip %d/%d [%s]: validated terminal status=%s",
                    i + 1,
                    len(sampled_configs),
                    config_id,
                    saved_result['status'],
                )
                continue

            attempted_configs += 1
            self.logger.info(f"\n{'='*80}")
            self.logger.info(
                f"Unique search {i+1}/{len(sampled_configs)} "
                f"[{sampled_config['config_id']}]"
            )
            self.logger.info(f"{'='*80}")

            current_params = sampled_config['params'].copy()
            fold_results = []
            last_chunk_summary = None
            self._write_progress_status({
                'phase': 'configuration_search',
                'status': 'configuration_started',
                'configuration': {
                    'index': i + 1,
                    'total': len(sampled_configs),
                    'config_id': config_id,
                },
                'fold': {
                    'completed': 0,
                    'total': total_folds,
                },
                'decision': 'continue',
            })

            self.logger.info("\nCurrent hyperparameter configuration:")
            for param, value in current_params.items():
                if param == 'loss_function' and isinstance(value, str):
                    self.logger.info("  loss_function: custom objective (SHA-256 %s)", payload_sha256(value))
                else:
                    self.logger.info(f"  {param}: {value}")

            try:
                base_config = {
                    'procs': N_PROCS,
                    'parallelism': PARALLELISM,
                    'deterministic': STRICT_DETERMINISTIC_SEARCH,
                    'random_state': sampled_config['config_seed'],
                    'verbosity': 0,
                    'progress': False,
                    'turbo': True,
                    'bumper': True,
                    'temp_equation_file': False,
                    'delete_tempfiles': True,
                    # This only controls PySR's display fallback.  Every
                    # scientific prediction below uses an explicit Pareto row.
                    'model_selection': 'accuracy',
                }

                base_config.update(current_params)

                base_config.pop('early_stop_condition', None)

                if 'loss_function' in base_config and isinstance(base_config['loss_function'], str):
                    self.logger.info(
                        "  Using a custom search objective and an exact %d-variable gate",
                        self.required_feature_count,
                    )

                config_pruned = False
                prune_details = None
                fold_num = 1

                while fold_num <= total_folds:
                    batch_end = min(
                        total_folds,
                        fold_num + max(1, FOLD_WORKERS) - 1,
                    )
                    jobs = []
                    for current_fold in range(fold_num, batch_end + 1):
                        train_idx, val_idx = cv_splits[current_fold - 1]
                        jobs.append(self._build_fold_job(
                            base_config=base_config,
                            train_idx=train_idx,
                            val_idx=val_idx,
                            fold_number=current_fold,
                            total_folds=total_folds,
                            config_id=config_id,
                            config_index=i + 1,
                            config_total=len(sampled_configs),
                        ))

                    if FOLD_WORKERS > 1 or FOLD_WORKER_THREADS != JULIA_THREADS:
                        self.logger.info(
                            "\nTraining folds %s/%d in worker pool (%d workers x %d threads)...",
                            ','.join(str(job['fold']) for job in jobs),
                            total_folds,
                            FOLD_WORKERS,
                            FOLD_WORKER_THREADS,
                        )
                        batch_results = self._run_parallel_fold_batch(jobs)
                    else:
                        self.logger.info(
                            f"\nTraining fold {fold_num}/{total_folds}..."
                        )
                        batch_results = [self._fit_fold_job(jobs[0])]

                    for fold_result in batch_results:
                        current_fold = int(fold_result['fold'])
                        execution = fold_result['execution']
                        last_chunk_summary = execution
                        fold_results.append(fold_result)
                        self.logger.info(
                            "  Fold %d equation: %s",
                            current_fold,
                            fold_result['equation'],
                        )
                        self.logger.info(
                            "  Fold %d controller: %d/%d iterations, stop=%s, "
                            "elapsed=%.1fs, Julia threads=%s",
                            current_fold,
                            execution['requested_iterations'],
                            execution['maximum_iterations'],
                            execution['stop_reason'],
                            execution['elapsed_seconds'],
                            fold_result.get('worker_runtime', {}).get(
                                'actual_julia_threads'
                            ),
                        )
                        self.logger.info(
                            "  Fold %d used the required %d variables [OK]",
                            current_fold,
                            len(fold_result['used_variables']),
                        )
                        self.logger.info(
                            "  Fold %d validation R² = %.6f, RMSE = %.6f, MAE = %.6f",
                            current_fold,
                            fold_result['raw_r2'],
                            fold_result['rmse'],
                            fold_result['mae'],
                        )

                    # Parallel scheduling evaluates pruning after a complete
                    # batch, so no already-running fold is cancelled midway.
                    # This remains conservative and can only perform up to
                    # FOLD_WORKERS-1 more folds than the serial controller.
                    if SEARCH_EXECUTION_CONFIG['enable_config_racing']:
                        config_pruned, prune_details = conservative_fold_prune(
                            [float(row['raw_r2']) for row in fold_results],
                            incumbent_fold_scores,
                            minimum_folds=SEARCH_EXECUTION_CONFIG[
                                'minimum_folds_before_pruning'
                            ],
                            total_folds=total_folds,
                            safety_margin=SEARCH_EXECUTION_CONFIG[
                                'pruning_safety_margin'
                            ],
                        )
                        if config_pruned:
                            self.logger.info(
                                "  Configuration pruned after %d/%d folds: "
                                "optimistic final R²=%.6f, completed incumbent=%.6f",
                                len(fold_results),
                                total_folds,
                                prune_details['optimistic_final_mean'],
                                prune_details['incumbent_mean'],
                            )
                            break
                    fold_num = batch_end + 1

                if config_pruned:
                    pruned_result = {
                        'iteration': i + 1,
                        'config_id': config_id,
                        'config_seed': sampled_config['config_seed'],
                        'status': 'conservatively_pruned',
                        'params': current_params.copy(),
                        'completed_folds': len(fold_results),
                        'total_folds': total_folds,
                        'prune_details': prune_details,
                        'fold_results': fold_results,
                    }
                    pruned_results.append(pruned_result)
                    terminal_results_by_id[config_id] = pruned_result
                    self._write_configuration_checkpoint(
                        sampled_config, pruned_result
                    )
                    self._mark_configuration_terminal(
                        status='conservatively_pruned',
                        config_index=i + 1,
                        config_total=len(sampled_configs),
                        config_id=config_id,
                        completed=completed_configs,
                        pruned=len(pruned_results),
                        failed=len(failed_results),
                        folds_completed=len(fold_results),
                        fold_total=total_folds,
                    )
                    continue

                mean_r2 = np.mean([float(row['raw_r2']) for row in fold_results])
                std_r2 = np.std([float(row['raw_r2']) for row in fold_results])
                mean_rmse = np.mean([float(row['rmse']) for row in fold_results])
                std_rmse = np.std([float(row['rmse']) for row in fold_results])
                mean_mae = np.mean([float(row['mae']) for row in fold_results])
                std_mae = np.std([float(row['mae']) for row in fold_results])
                complexities = [
                    fold['candidate_complexity'] for fold in fold_results
                ]
                mean_complexity = float(np.mean(complexities))
                std_complexity = float(np.std(complexities))
                excess_complexities = [
                    fold['candidate_excess_complexity'] for fold in fold_results
                ]
                mean_excess_complexity = float(np.mean(excess_complexities))
                std_excess_complexity = float(np.std(excess_complexities))
                direct_quadratic_fold_frequency = float(
                    np.mean(
                        [
                            fold['candidate_has_direct_quadratic']
                            for fold in fold_results
                        ]
                    )
                )

                self.logger.info(f"\nCross-validation average performance ({total_folds} folds):")
                self.logger.info(f"  R² = {mean_r2:.6f} ± {std_r2:.6f}")
                self.logger.info(f"  RMSE = {mean_rmse:.6f} ± {std_rmse:.6f}")
                self.logger.info(f"  MAE = {mean_mae:.6f} ± {std_mae:.6f}")
                self.logger.info(
                    "  Excess complexity = "
                    f"{mean_excess_complexity:.3f} ± "
                    f"{std_excess_complexity:.3f}; folds with a direct "
                    f"quadratic = {direct_quadratic_fold_frequency:.1%}"
                )

                result = {
                    'iteration': i + 1,
                    'config_id': config_id,
                    'config_seed': sampled_config['config_seed'],
                    'status': 'completed',
                    'params': current_params.copy(),
                    'cv_raw_r2_mean': float(mean_r2),
                    'cv_raw_r2_std': float(std_r2),
                    'cv_rmse_mean': float(mean_rmse),
                    'cv_rmse_std': float(std_rmse),
                    'cv_mae_mean': float(mean_mae),
                    'cv_mae_std': float(std_mae),
                    'cv_complexity_mean': mean_complexity,
                    'cv_complexity_std': std_complexity,
                    'cv_excess_complexity_mean': mean_excess_complexity,
                    'cv_excess_complexity_std': std_excess_complexity,
                    'cv_direct_quadratic_fold_frequency': (
                        direct_quadratic_fold_frequency
                    ),
                    'selection_metric': 'mean_raw_validation_r2',
                    'selection_score': float(mean_r2),
                    'fold_results': fold_results,
                }
                all_results.append(result)
                terminal_results_by_id[config_id] = result
                completed_configs += 1
                if mean_r2 > incumbent_score:
                    incumbent_score = float(mean_r2)
                    incumbent_result = result
                    incumbent_fold_scores = [float(row['raw_r2']) for row in fold_results]
                    self.logger.info(
                        f"  New completed incumbent raw CV R²: {incumbent_score:.6f}"
                    )
                self._write_configuration_checkpoint(sampled_config, result)
                self._mark_configuration_terminal(
                    status='completed',
                    config_index=i + 1,
                    config_total=len(sampled_configs),
                    config_id=config_id,
                    completed=completed_configs,
                    pruned=len(pruned_results),
                    failed=len(failed_results),
                    folds_completed=len(fold_results),
                    fold_total=total_folds,
                )

            except ResumeCheckpointError:
                raise
            except Exception as e:
                self.logger.error(f"Iteration {i+1} failed: {str(e)}")
                failed_result = {
                    'iteration': i + 1,
                    'config_id': config_id,
                    'config_seed': sampled_config['config_seed'],
                    'status': 'failed',
                    'params': current_params.copy(),
                    'error': str(e),
                    'completed_folds': len(fold_results),
                    'fold_results': fold_results,
                    'last_fold_execution': last_chunk_summary,
                }
                failed_results.append(failed_result)
                terminal_results_by_id[config_id] = failed_result
                self._write_configuration_checkpoint(
                    sampled_config, failed_result
                )
                self._mark_configuration_terminal(
                    status='failed',
                    config_index=i + 1,
                    config_total=len(sampled_configs),
                    config_id=config_id,
                    completed=completed_configs,
                    pruned=len(pruned_results),
                    failed=len(failed_results),
                    folds_completed=len(fold_results),
                    fold_total=total_folds,
                )
                if i < 10:
                    self.logger.error(traceback.format_exc())
                continue

        terminal_results = sorted(
            terminal_results_by_id.values(),
            key=lambda row: int(row['iteration']),
        )
        all_sampled_attempted = attempted_configs == len(sampled_configs)
        terminal_total = len(terminal_results)
        if not all_sampled_attempted or terminal_total != len(sampled_configs):
            raise RuntimeError(
                'publication search did not assign every sampled configuration '
                'a terminal status'
            )
        if not all_results:
            raise ValueError("All random search iterations failed. Check data or reduce model complexity.")

        best_result, config_selection = select_config_one_standard_error(all_results)
        best_params = best_result['params'].copy()
        best_score = float(best_result['cv_raw_r2_mean'])
        refinement_excess_complexity_ceiling = self.max_excess_complexity
        self.refinement_excess_complexity_ceiling = refinement_excess_complexity_ceiling

        self.logger.info(f"\n{'='*80}")
        self.logger.info("Random search completed!")
        self.logger.info(f"{'='*80}")

        self.logger.info("\nAll sampled configurations reached a terminal status")
        self.logger.info(f"  Attempted configurations: {attempted_configs}/{n_random_searches}")
        self.logger.info(f"  Completed configurations: {completed_configs}")
        self.logger.info(f"  Pruned configurations: {len(pruned_results)}")
        self.logger.info(f"  Failed configurations: {len(failed_results)}")

        self.logger.info(
            f"\nHighest completed mean raw CV R²: {incumbent_score:.6f} "
            f"[{incumbent_result['config_id']}]"
        )
        self.logger.info(
            f"One-standard-error selected raw CV R²: {best_score:.6f} "
            f"[{best_result['config_id']}]"
        )
        self.logger.info("\nSelected hyperparameter configuration:")
        for param, value in best_params.items():
            self.logger.info(f"  {param}: {value}")

        self.best_params = best_params
        self.best_result = best_result
        self.all_search_results = all_results
        self.pruned_search_results = pruned_results
        self.failed_search_results = failed_results
        self.search_summary = {
            'schema_version': 4,
            'profile': 'publication',
            'requested_unique_configurations': n_random_searches,
            'sampled_unique_configurations': len(sampled_configs),
            'attempted_configurations': attempted_configs,
            'completed_configurations': completed_configs,
            'conservatively_pruned_configurations': len(pruned_results),
            'failed_configurations': len(failed_results),
            'all_sampled_configurations_attempted': all_sampled_attempted,
            'global_configuration_early_stop': False,
            'selection_metric': 'mean_raw_validation_r2',
            'selection_rule': config_selection,
            'selected_config_id': best_result['config_id'],
            'selected_config_seed': int(best_result['config_seed']),
            'selected_config_raw_cv_r2': best_score,
            'selected_config_excess_complexity_mean': float(
                best_result['cv_excess_complexity_mean']
            ),
            'highest_mean_config_id': incumbent_result['config_id'],
            'highest_mean_raw_cv_r2': float(incumbent_score),
            'refinement_excess_complexity_ceiling': (
                refinement_excess_complexity_ceiling
            ),
            'cv_method': CV_METHOD,
            'cv_n_splits': CV_N_SPLITS,
            'cv_n_repeats': CV_N_REPEATS if CV_METHOD == 'repeated_stratified' else None,
            'cv_total_folds': total_folds,
            'chunk_execution': SEARCH_EXECUTION_CONFIG,
            'resume': {
                'enabled': self.resume_mode,
                'contract_id': self._resume_contract_id,
                'loaded_terminal_configurations': (
                    loaded_terminal_configurations
                ),
                'checkpoint_granularity': 'terminal_configuration',
            },
        }

        self._write_cv_selection_checkpoint(
            best_result=best_result,
            config_selection=config_selection,
            incumbent_result=incumbent_result,
            terminal_results=terminal_results,
        )
        self._write_progress_status({
            'phase': 'cv_selection',
            'status': 'checkpoint_saved',
            'configuration': {
                'index': len(sampled_configs),
                'total': len(sampled_configs),
                'config_id': best_result['config_id'],
            },
            'fold': None,
            'decision': 'safe_to_shutdown_workers',
        })

        # The selected result is durable before any Julia worker is asked to
        # exit.  Refinement then runs once in the parent with the full thread
        # budget.  Only idle workers are eligible for bounded forced cleanup.
        shutdown_report = self._shutdown_fold_executor(require_idle=True)
        self.search_summary['fold_worker_shutdown'] = shutdown_report
        self._write_cv_selection_checkpoint(
            best_result=best_result,
            config_selection=config_selection,
            incumbent_result=incumbent_result,
            terminal_results=terminal_results,
            shutdown_report=shutdown_report,
        )
        self.write_run_config()

        # Refinement training
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Step 2: Refinement training (based on best hyperparameters)")
        self.logger.info(f"{'='*80}")
        self.logger.info("This is the only step to obtain the final model")
        self.logger.info("Using larger populations and more iterations for deep optimization on full training set...")

        try:
            refined_params = best_params.copy()
            refined_params['loss_function'] = make_loss_function(
                self.required_feature_count, refinement_excess_complexity_ceiling
            )
            refined_params['populations'] = self.refinement_config['populations']
            refined_params['population_size'] = self.refinement_config['population_size']
            refined_params['niterations'] = self.refinement_config['niterations']
            refined_params['timeout_in_seconds'] = self.refinement_config['timeout_in_seconds']

            if 'early_stop_condition' in refined_params:
                refined_params.pop('early_stop_condition')

            self.logger.info("\n" + "-"*80)
            self.logger.info("Refinement training full hyperparameter configuration:")
            self.logger.info("-"*80)

            self.logger.info(f"  populations          : {refined_params.get('populations', 'N/A')} ← refinement enhanced")
            self.logger.info(f"  population_size      : {refined_params.get('population_size', 'N/A')} ← refinement enhanced")
            self.logger.info(f"  niterations          : {refined_params.get('niterations', 'N/A')} ← refinement enhanced")
            self.logger.info(f"  timeout_in_seconds   : {refined_params.get('timeout_in_seconds', 'N/A')} ← explicit PC wall-time limit")
            self.logger.info(f"  chunk_iterations     : {self.refinement_config['chunk_iterations']}")
            self.logger.info(
                "  CV excess-complexity ceiling: "
                f"{refinement_excess_complexity_ceiling}"
            )
            self.logger.info(f"  binary_operators     : {refined_params.get('binary_operators', 'N/A')}")
            self.logger.info(f"  unary_operators      : {refined_params.get('unary_operators', 'N/A')}")
            self.logger.info(f"  maxsize              : {refined_params.get('maxsize', 'N/A')}")
            self.logger.info(f"  maxdepth             : {refined_params.get('maxdepth', 'N/A')}")
            self.logger.info(f"  parsimony            : {refined_params.get('parsimony', 'N/A')}")

            final_refined_config = {
                'procs': N_PROCS,
                'parallelism': PARALLELISM,
                'deterministic': STRICT_DETERMINISTIC_SEARCH,
                'random_state': int(best_result['config_seed']),
                'verbosity': 0,
                'progress': False,
                'turbo': True,
                'bumper': True,
                'model_selection': 'accuracy',
            }
            final_refined_config.update(refined_params)

            inherited_candidates, inheritance_audit = inherit_cv_winner_candidates(
                best_result, self.X_train, self.y_train, self.feature_columns,
                self.scaler.mean_, self.scaler.scale_,
                required_feature_count=self.required_feature_count,
                excess_complexity_ceiling=refinement_excess_complexity_ceiling,
            )
            self.cv_winner_inheritance = inheritance_audit
            atomic_write_json(
                Path(self.output_path) / 'cv_winner_inheritance.json',
                inheritance_audit,
            )
            if inherited_candidates.empty:
                raise ValueError('No CV winner remains eligible for refinement; see cv_winner_inheritance.json')
            self.logger.info(
                "Inherited %d CV winners after outer-training coefficient refit",
                len(inherited_candidates),
            )

            refinement_run_id = (
                f'refinement_{self.resume_invocation_id}'
                if self.resume_mode
                else 'refinement'
            )
            if self.resume_mode:
                self.logger.info(
                    "Resume starts refinement as a clean attempt: %s",
                    refinement_run_id,
                )

            final_refined_model, refinement_execution = self.fit_pysr_in_chunks(
                final_refined_config,
                self.X_train,
                self.y_train,
                run_id=refinement_run_id,
                max_iterations=self.refinement_config['niterations'],
                chunk_iterations=self.refinement_config['chunk_iterations'],
                max_wall_seconds=self.refinement_config['timeout_in_seconds'],
                minimum_iterations=self.refinement_config['minimum_iterations'],
                plateau_patience=self.refinement_config['plateau_patience'],
                minimum_score_improvement=0.0,
                minimum_loss_relative_improvement=self.refinement_config['minimum_loss_relative_improvement'],
                enable_plateau_stop=self.refinement_config['enable_plateau_stop'],
                training_excess_complexity_ceiling=(
                    refinement_excess_complexity_ceiling
                ),
                initial_candidates=inherited_candidates,
                progress_phase='refinement',
                plateau_window_iterations=self.refinement_config['window_iterations'],
                similar_loss_relative_tolerance=self.refinement_config['similar_loss_relative_tolerance'],
            )
            final_time = refinement_execution['elapsed_seconds']

            self.logger.info(f" Refinement training completed, time elapsed {final_time:.2f} seconds")

            review_splits = list(create_cv(
                self.X_train_raw, self.y_train,
                n_repeats=FINAL_REVIEW_N_REPEATS,
                random_state=FINAL_REVIEW_RANDOM_STATE,
            ))
            refined_candidate = self.review_final_archive(
                final_refined_model, review_splits, refinement_excess_complexity_ceiling
            )
            refined_candidate_position = int(refined_candidate['candidate_position'])
            refined_equation = self._equation_at(
                final_refined_model, refined_candidate_position
            )

            self.logger.info("\nRefined final equation:")
            self.logger.info(f"  {refined_equation}")

            used_vars_refined = used_features(refined_equation, self.feature_columns)

            n_used = len(used_vars_refined)
            self.logger.info(
                "\nRefined equation variables used (%d of %d candidates): %s",
                n_used, len(self.feature_columns), used_vars_refined,
            )
            if n_used != self.required_feature_count:
                raise ValueError(
                    f'Refined equation must contain exactly {self.required_feature_count} '
                    'candidate features after simplification'
                )
            self.selected_feature_names = used_vars_refined
            self.fit_ridge_baseline(cv_splits)
            self.fit_ridge_baseline(cv_splits, quadratic=True)

            y_pred_train_refined = self._predict_at(
                final_refined_model,
                self.X_train,
                refined_candidate_position,
            )

            refined_train_r2 = r2_score(self.y_train, y_pred_train_refined)
            refined_train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train_refined))
            refined_train_mae = mean_absolute_error(self.y_train, y_pred_train_refined)

            self.logger.info("\nRefined model performance:")
            self.logger.info(
                f"  Outer-train R²: {refined_train_r2:.6f}, "
                f"RMSE: {refined_train_rmse:.6f}, MAE: {refined_train_mae:.6f}"
            )
            self.logger.info(
                "  External test remains locked until the single final evaluation"
            )

            self.model = final_refined_model
            self.final_candidate_position = refined_candidate_position
            self.write_run_config()

            self.logger.info(f"\n{'='*80}")
            self.logger.info("[OK] Refinement training completed, set as final model")
            self.logger.info(f"{'='*80}")

            self.refinement_result = {
                'used': True,
                'train_r2': float(refined_train_r2),
                'train_rmse': float(refined_train_rmse),
                'train_mae': float(refined_train_mae),
                'equation': refined_equation,
                'candidate_position': refined_candidate_position,
                'candidate_origin': refined_candidate['candidate_origin'],
                'source_cv_folds': refined_candidate['source_cv_folds'],
                'candidate_complexity': int(refined_candidate['complexity']),
                'candidate_structural_complexity': int(
                    refined_candidate['structural_complexity']
                ),
                'candidate_excess_complexity': int(
                    refined_candidate['excess_complexity']
                ),
                'candidate_has_direct_quadratic': bool(
                    refined_candidate['has_direct_quadratic']
                ),
                'candidate_term_budget': refined_candidate['term_budget'],
                'candidate_training_loss': float(refined_candidate['loss']),
                'candidate_selection_rule': refined_candidate.get('selection_rule'),
                'excess_complexity_ceiling': refined_candidate.get(
                    'excess_complexity_ceiling'
                ),
                'excess_complexity_ceiling_relaxed': refined_candidate.get(
                    'excess_complexity_ceiling_relaxed'
                ),
                'training_time': final_time,
                'final_candidate_cv': self.final_candidate_cv,
                'selected_config_raw_cv_r2': float(best_score),
                'used_variables': used_vars_refined,
                'n_variables_used': n_used,
                'execution': refinement_execution,
            }

        except Exception as e:
            self.logger.error(f"Refinement training failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Refinement training failed, cannot obtain final model: {str(e)}")

        return self.model

    def analyze_results(self):
        """Analyze and evaluate final model results."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Final results analysis")
        self.logger.info("="*80)

        best_equation = self._equation_at(
            self.model, self.final_candidate_position
        )

        equation_scaled, equation_raw = self.get_equations_scaled_and_raw()

        self.logger.info("\nBest equation found:")
        self.logger.info(f"\n{self.target_column} = {best_equation}\n")

        y_pred_train = self._predict_at(
            self.model, self.X_train, self.final_candidate_position
        )
        # This is the only external-test prediction in the search workflow.
        y_pred_test = self._predict_at(
            self.model, self.X_test, self.final_candidate_position
        )

        equation_audit = self.analyze_equation_units_and_uncertainty(
            equation_scaled,
            equation_raw,
            y_pred_train,
            y_pred_test,
        )

        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)

        for prefix, quadratic in [('ridge', False), ('ridge_quadratic', True)]:
            X_basis = self.ridge_basis(
                self.X_test_raw[:, self.ridge_feature_indices], quadratic
            )
            prediction = np.asarray(
                getattr(self, f'{prefix}_baseline_model').predict(
                    getattr(self, f'{prefix}_baseline_scaler').transform(X_basis)
                ), dtype=float,
            ).reshape(-1)
            setattr(self, f'{prefix}_test_predictions', prediction)
            getattr(self, f'{prefix}_baseline').update({
                'test_r2': float(r2_score(self.y_test, prediction)),
                'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, prediction))),
                'test_mae': float(mean_absolute_error(self.y_test, prediction)),
                'external_test_evaluations': 1,
            })

        self.logger.info("Final model performance:")
        self.logger.info("\nTraining set:")
        self.logger.info(f"  R² = {train_r2:.6f}")
        self.logger.info(f"  RMSE = {train_rmse:.6f}")
        self.logger.info(f"  MAE = {train_mae:.6f}")
        self.logger.info("\nTest set:")
        self.logger.info(f"  R² = {test_r2:.6f}")
        self.logger.info(f"  RMSE = {test_rmse:.6f}")
        self.logger.info(f"  MAE = {test_mae:.6f}")
        for baseline in [self.ridge_baseline, self.ridge_quadratic_baseline]:
            self.logger.info("\nRidge %s baseline on the identical external test rows:", baseline['basis'])
            self.logger.info("  R² = %.6f; RMSE = %.6f; MAE = %.6f",
                             baseline['test_r2'], baseline['test_rmse'], baseline['test_mae'])

        if hasattr(self, 'refinement_result'):
            self.refinement_result.update({
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'external_test_evaluations': 1,
            })

        best_cv_result = self.best_result

        if best_cv_result:
            total_folds = self.search_summary['cv_total_folds']
            self.logger.info(f"\nConfiguration-search CV ({total_folds} folds):")
            self.logger.info(f"  Raw R² = {best_cv_result['cv_raw_r2_mean']:.6f} ± {best_cv_result['cv_raw_r2_std']:.6f}")
            self.logger.info(f"  RMSE = {best_cv_result['cv_rmse_mean']:.6f} ± {best_cv_result['cv_rmse_std']:.6f}")
            self.logger.info(f"  MAE = {best_cv_result['cv_mae_mean']:.6f} ± {best_cv_result['cv_mae_std']:.6f}")
            self.logger.info(
                "  Excess complexity = "
                f"{best_cv_result['cv_excess_complexity_mean']:.3f} ± "
                f"{best_cv_result['cv_excess_complexity_std']:.3f}; "
                "direct-quadratic fold frequency = "
                f"{best_cv_result['cv_direct_quadratic_fold_frequency']:.1%}"
            )

        reviewed = self.final_candidate_cv['selected_candidate']
        self.logger.info(
            "\nFinal-formula conditional CV: R² = %.6f ± %.6f",
            reviewed['cv_raw_r2_mean'], reviewed['cv_raw_r2_std'],
        )

        self.results = {
            'campaign_round': self.campaign_round,
            'score_version': self.score_info.get('score_version'),
            'runtime': symbolic_regression_runtime_metadata(),
            'equation': best_equation,
            'equation_scaled': equation_scaled,
            'equation_raw': equation_raw,
            'equation_audit': equation_audit,
            'term_budget': equation_audit['complexity_policy'][
                'selected_equation_term_budget'
            ],
            'metric_definitions': METRIC_DEFINITIONS,
            'ridge_baseline': self.ridge_baseline,
            'ridge_quadratic_baseline': self.ridge_quadratic_baseline,
            'cv_winner_inheritance': self.cv_winner_inheritance,
            'final_candidate_cv': self.final_candidate_cv,
            'feature_names': self.feature_columns,
            'selected_feature_names': list(self.selected_feature_names),
            'required_feature_count': self.required_feature_count,
            'feature_selection': getattr(self, 'feature_selection', None),
            'final_candidate_position': int(self.final_candidate_position),
            'best_params': self.best_params,
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'total_samples': len(self.X)
        }

        if best_cv_result:
            self.results.update({
                'cv_raw_r2_mean': float(best_cv_result['cv_raw_r2_mean']),
                'cv_raw_r2_std': float(best_cv_result['cv_raw_r2_std']),
                'cv_rmse_mean': float(best_cv_result['cv_rmse_mean']),
                'cv_rmse_std': float(best_cv_result['cv_rmse_std']),
                'cv_mae_mean': float(best_cv_result['cv_mae_mean']),
                'cv_mae_std': float(best_cv_result['cv_mae_std']),
                'cv_complexity_mean': float(best_cv_result['cv_complexity_mean']),
                'cv_complexity_std': float(best_cv_result['cv_complexity_std']),
                'cv_excess_complexity_mean': float(
                    best_cv_result['cv_excess_complexity_mean']
                ),
                'cv_excess_complexity_std': float(
                    best_cv_result['cv_excess_complexity_std']
                ),
                'cv_direct_quadratic_fold_frequency': float(
                    best_cv_result['cv_direct_quadratic_fold_frequency']
                ),
            })

        return y_pred_train, y_pred_test

    def save_results(self, y_pred_train, y_pred_test):
        """Save analysis results to files."""
        self.logger.info("\n" + "="*80)
        self.logger.info("Saving results...")
        self.logger.info("="*80)

        equation_audit_file = Path(self.output_path) / 'equation_audit.json'
        atomic_write_json(equation_audit_file, self.equation_audit)
        atomic_write_json(
            Path(self.output_path) / 'selected_candidate.json',
            {
                'schema_version': 2,
                'campaign_round': self.campaign_round,
                'score_version': self.score_info.get('score_version'),
                'candidate_position': int(self.final_candidate_position),
                'candidate_origin': self.refinement_result['candidate_origin'],
                'source_cv_folds': self.refinement_result['source_cv_folds'],
                'inheritance_audit_file': 'cv_winner_inheritance.json',
                'final_candidate_cv': self.final_candidate_cv,
                'selected_feature_names': list(self.selected_feature_names),
                'required_feature_count': self.required_feature_count,
                'equation_scaled': self.results['equation_scaled'],
                'equation_raw': self.results['equation_raw'],
                'term_budget': self.results['term_budget'],
                'structural_complexity': getattr(
                    self, 'refinement_result', {}
                ).get('candidate_structural_complexity'),
                'excess_complexity': self.results['term_budget'][
                    'excess_complexity'
                ],
                'has_direct_quadratic': self.results['term_budget'][
                    'has_direct_quadratic'
                ],
                'prediction_contract': (
                    'model.predict(X_scaled, index=candidate_position)'
                ),
                'selection_rule': self.final_candidate_cv['selection_rule'],
                'configuration_selection_rule': self.search_summary['selection_rule'],
            },
        )

        coefficient_rows = self.equation_audit[
            'coefficient_uncertainty'
        ]['coefficient_rows']
        coefficient_columns = [
            'parameter',
            'estimate',
            'bootstrap_mean',
            'bootstrap_median',
            'bootstrap_standard_error',
            'ci_low',
            'ci_high',
            'confidence_level',
            'successful_samples',
        ]
        pd.DataFrame(coefficient_rows, columns=coefficient_columns).to_csv(
            Path(self.output_path) / 'coefficient_uncertainty.csv',
            index=False,
            encoding='utf-8',
        )
        pd.DataFrame(self.coefficient_bootstrap_samples).to_csv(
            Path(self.output_path) / 'coefficient_bootstrap_samples.csv',
            index=False,
            encoding='utf-8',
        )
        self.logger.info(f"Equation audit saved to: {equation_audit_file}")
        self.logger.info("Coefficient uncertainty saved to: coefficient_uncertainty.csv")
        self.logger.info(
            "Coefficient bootstrap samples saved to: coefficient_bootstrap_samples.csv"
        )

        txt_file = os.path.join(self.output_path, 'results.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("                     Symbolic Regression Analysis Report\n")
            f.write("="*100 + "\n\n")

            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Campaign round: {self.campaign_round}\n")
            f.write(f"Score version: {self.score_info.get('score_version')}\n")
            f.write(f"Data file: {self.data_path}\n")
            f.write(f"Output path: {self.output_path}\n\n")

            runtime = symbolic_regression_runtime_metadata()
            f.write("Runtime contract:\n")
            f.write(f"  PySR version: {runtime['pysr_version']}\n")
            f.write(f"  Logical processors: {runtime['logical_processors']}\n")
            f.write(
                "  Julia threads configured/actual: "
                f"{runtime['configured_julia_threads']}/"
                f"{runtime['actual_julia_threads']}\n"
            )
            f.write(f"  BLAS helper threads: {runtime['blas_threads']}\n")
            f.write(f"  Parallelism: {runtime['parallelism']}\n")
            f.write(
                f"  Strict deterministic mode: {runtime['strict_deterministic']}\n"
            )
            f.write("  Operating-system power plan: not modified by this workflow\n\n")

            f.write("-"*100 + "\n")
            f.write("1. Cross-validation Configuration\n")
            f.write("-"*100 + "\n")
            f.write(f"Method: {self.search_summary['cv_method']}\n")
            if self.search_summary['cv_method'] == 'repeated_stratified':
                f.write(f"K-fold splits: {self.search_summary['cv_n_splits']}\n")
                f.write(f"Repeats: {self.search_summary['cv_n_repeats']}\n")
            f.write(f"Total folds: {self.search_summary['cv_total_folds']}\n")
            f.write("Scaler: fitted independently on each fold's training rows\n")
            f.write(f"Strata: {CV_N_STRATA}\n\n")

            f.write("-"*100 + "\n")
            f.write("2. Search Process Information\n")
            f.write("-"*100 + "\n")
            f.write(f"Profile: {self.search_summary['profile']}\n")
            f.write(
                f"Requested/sampled/attempted configurations: "
                f"{self.search_summary['requested_unique_configurations']}/"
                f"{self.search_summary['sampled_unique_configurations']}/"
                f"{self.search_summary['attempted_configurations']}\n"
            )
            f.write(
                f"Terminal states: completed={self.search_summary['completed_configurations']}, "
                f"conservatively_pruned={self.search_summary['conservatively_pruned_configurations']}, "
                f"failed={self.search_summary['failed_configurations']}\n"
            )
            f.write("Global target-R2/patience/window termination: not configured\n")
            f.write(
                "Configuration selection: one-standard-error rule on raw validation "
                "R2, then excess-term complexity and stability\n"
            )
            f.write(
                "Complexity policy: PySR parsimony=0. For each feature, one "
                "linear term and one direct nonlinear term are free; with the "
                "default operators that nonlinear term is x^2. Interactions, "
                "powers above two, repeated nonlinear terms, and nested/composite "
                "transforms add excess complexity. PySR structural complexity is "
                "reported for audit and used only as a hard resource cap.\n"
            )

            f.write("\nRefinement training status: ")
            if hasattr(self, 'refinement_result'):
                if self.refinement_result.get('used'):
                    f.write(" Enabled and adopted\n")
                    f.write("Refinement config:\n")
                    f.write(f"  Population size: {self.refinement_config['population_size']}\n")
                    f.write(f"  Iterations: {self.refinement_config['niterations']}\n")
                    f.write(f"  Training time: {self.refinement_result['training_time']:.2f} seconds\n")
                    f.write("Refinement performance:\n")
                    f.write(f"  Train R²: {self.refinement_result['train_r2']:.6f}\n")
                    f.write(f"  Test R²: {self.refinement_result['test_r2']:.6f}\n")
                elif self.refinement_result.get('error'):
                    f.write(" Attempted but failed\n")
                    f.write(f"  Error: {self.refinement_result['error']}\n")
                else:
                    f.write(" Attempted but not improved\n")
                    f.write(f"  Refined test R²: {self.refinement_result.get('test_r2', 'N/A'):.6f}\n")
            else:
                f.write("Not enabled\n")

            f.write(
                f"Selected configuration raw CV R²: "
                f"{self.search_summary['selected_config_raw_cv_r2']:.6f}\n\n"
            )
            f.write(
                f"Final formula selection: fixed-structure 4-fold x {FINAL_REVIEW_N_REPEATS}-repeat "
                "coefficient CV, one-SE then excess and structural complexity. "
                "This is a conditional candidate-selection diagnostic.\n"
                f"Review time: {self.final_candidate_cv['elapsed_seconds']:.2f} seconds; "
                "details: final_candidate_cv.json\n\n"
            )

            f.write("-"*100 + "\n")
            f.write("3. Data Information\n")
            f.write("-"*100 + "\n")
            f.write(f"Total samples: {self.results['total_samples']}\n")
            f.write(f"Training set: {self.results['train_size']} ({self.results['train_size']/self.results['total_samples']*100:.1f}%)\n")
            f.write(f"Test set: {self.results['test_size']} ({self.results['test_size']/self.results['total_samples']*100:.1f}%)\n\n")
            f.write(
                "split_assignments.csv records Name, the stable zero-based "
                "data.csv source_row_index, and Train/Test.\n\n"
            )

            f.write("-"*100 + "\n")
            f.write("4. Variable Information\n")
            f.write("-"*100 + "\n")
            f.write(f"Target variable: {self.target_column}\n\n")
            f.write(f"Candidate variables ({len(self.feature_columns)} total):\n")
            for i, col in enumerate(self.feature_columns, 1):
                f.write(f"  {i}. {col}\n")
            f.write(f"Selected variables ({self.required_feature_count}): "
                    f"{', '.join(self.selected_feature_names)}\n")
            f.write("\n")

            f.write("-"*100 + "\n")
            f.write("5. Feature Standardization Information\n")
            f.write("-"*100 + "\n")
            f.write("All features have been Z-score standardized (x_scaled = (x - mean) / std)\n\n")
            f.write("Standardization parameters:\n")
            for i, col in enumerate(self.feature_columns):
                f.write(f"  {col}:\n")
                f.write(f"    Original mean: {self.scaler.mean_[i]:.6f}\n")
                f.write(f"    Original std: {self.scaler.scale_[i]:.6f}\n")
                f.write(f"    Variance: {self.scaler.var_[i]:.6f}\n")
            f.write("\nNotes:\n")
            f.write("1) equation_scaled: variables are Z-score standardized values, standardize before substituting.\n")
            f.write("2) equation_raw   : z=(x-mean)/std has been substituted back, can use original features directly.\n\n")

            f.write("-"*100 + "\n")
            f.write("6. Selected Hyperparameter Configuration\n")
            f.write("-"*100 + "\n")
            for param, value in self.best_params.items():
                f.write(f"{param:20s}: {value}\n")
            f.write("\n")

            f.write("-"*100 + "\n")
            f.write("7. Best Equation Found\n")
            f.write("-"*100 + "\n")
            f.write(f"\nOriginal form:\n{self.target_column} = {self.results['equation']}\n\n")
            f.write("\nEquation using original features (de-standardized)\n")
            f.write(f"{self.target_column} = {self.results.get('equation_raw', 'N/A')}\n\n")
            f.write("Original formula in standardized feature space\n")
            f.write(f"{self.target_column} = {self.results.get('equation_scaled', 'N/A')}\n\n")
            term_budget = self.results['term_budget']
            f.write("Selected-equation term-budget audit:\n")
            f.write(
                f"  Excess complexity: {term_budget['excess_complexity']}\n"
            )
            f.write(
                "  Direct quadratic features: "
                f"{term_budget['direct_quadratic_features']}\n"
            )
            f.write(
                f"  Penalized terms: {term_budget['penalized_terms']}\n\n"
            )

            f.write("-"*100 + "\n")
            f.write("8. Equation Normalization, Units, and Coefficient Uncertainty\n")
            f.write("-"*100 + "\n")
            f.write(
                "Target: Score = z_PLQY - z_XRD; unit=1 (dimensionless); "
                "no additional target standardization is applied by SR.\n\n"
            )
            f.write("Feature unit and scaler contract:\n")
            for row in self.equation_audit['features']:
                f.write(
                    f"  {row['feature']}: raw unit={row['raw_source_unit']}; "
                    f"model input unit=1 (dimensionless); "
                    f"mean={row['standardization_mean']:.10g}; "
                    f"scale={row['standardization_scale']:.10g}\n"
                )
                f.write(f"    {row['standardization']}\n")
            f.write("\nScaled equation (dimensionless Z-score inputs):\n")
            f.write(
                f"{self.target_column} = "
                f"{self.equation_audit['equations']['equation_scaled']}\n\n"
            )
            f.write("Raw-feature equation (use exactly the listed source units):\n")
            f.write(
                f"{self.target_column} = "
                f"{self.equation_audit['equations']['equation_raw']}\n\n"
            )
            consistency = self.equation_audit['numerical_consistency']
            f.write(
                "Numerical equivalence audit: accepted on training, test, and all "
                f"{len(self.X)} library rows; full-library max absolute error="
                f"{consistency['full_library_raw_vs_scaled_max_abs_error']:.3e}.\n\n"
            )
            uncertainty = self.equation_audit['coefficient_uncertainty']
            f.write("Coefficient uncertainty method:\n")
            f.write(
                "  Nonparametric outer-training row bootstrap; symbolic structure "
                "and outer-training scaler fixed; external test not used.\n"
            )
            f.write(
                f"  Status: {uncertainty['status']}; successful samples: "
                f"{uncertainty['successful_samples']}/"
                f"{uncertainty['settings']['requested_bootstrap_samples']}; "
                f"elapsed: {uncertainty['elapsed_seconds']:.2f} s.\n"
            )
            f.write("  Parameterized scaled equation:\n")
            f.write(
                f"    {self.target_column} = "
                f"{uncertainty['parameterized_equation_scaled']}\n"
            )
            f.write("  Parameterized raw-feature equation:\n")
            f.write(
                f"    {self.target_column} = "
                f"{uncertainty['parameterized_equation_raw']}\n\n"
            )
            if uncertainty['coefficient_rows']:
                confidence_percent = uncertainty['settings']['confidence_level'] * 100
                f.write(
                    "  Parameter       Estimate       Bootstrap SE       "
                    f"{confidence_percent:.1f}% percentile CI\n"
                )
                for row in uncertainty['coefficient_rows']:
                    standard_error = row['bootstrap_standard_error']
                    standard_error_text = (
                        f"{standard_error:.10g}" if standard_error is not None else "N/A"
                    )
                    f.write(
                        f"  {row['parameter']:<12s} "
                        f"{row['estimate']:>13.10g} "
                        f"{standard_error_text:>18s} "
                        f"[{row['ci_low']:.10g}, {row['ci_high']:.10g}]\n"
                    )
            else:
                f.write("  The selected equation contains no fitted floating constants.\n")
            f.write("\nInterpretation limits:\n")
            for limitation in uncertainty['limitations']:
                f.write(f"  - {limitation}\n")
            f.write(
                "  - The raw equation is an exact predictive re-expression of the "
                "dimensionless model, not an independently derived physical law.\n\n"
            )

            f.write("-"*100 + "\n")
            f.write("9. Model Performance Evaluation\n")
            f.write("-"*100 + "\n\n")
            f.write("Metric definitions:\n")
            f.write(f"  R²   : {METRIC_DEFINITIONS['r2']}\n")
            f.write(f"  RMSE : {METRIC_DEFINITIONS['rmse']}\n")
            f.write(f"  MAE  : {METRIC_DEFINITIONS['mae']}\n")
            f.write(f"  MAPE : {METRIC_DEFINITIONS['mape']}\n\n")

            f.write("Training set performance:\n")
            f.write(f"  R²        = {self.results['train_r2']:.6f}\n")
            f.write(f"  RMSE      = {self.results['train_rmse']:.6f}\n")
            f.write(f"  MAE       = {self.results['train_mae']:.6f}\n\n")

            f.write("Test set performance:\n")
            f.write(f"  R²        = {self.results['test_r2']:.6f}\n")
            f.write(f"  RMSE      = {self.results['test_rmse']:.6f}\n")
            f.write(f"  MAE       = {self.results['test_mae']:.6f}\n\n")

            if 'cv_raw_r2_mean' in self.results:
                f.write("Configuration-search cross-validation performance:\n")
                f.write(f"  Raw R² = {self.results['cv_raw_r2_mean']:.6f} ± {self.results['cv_raw_r2_std']:.6f}\n")
                f.write(f"  RMSE = {self.results['cv_rmse_mean']:.6f} ± {self.results['cv_rmse_std']:.6f}\n")
                f.write(f"  MAE = {self.results['cv_mae_mean']:.6f} ± {self.results['cv_mae_std']:.6f}\n\n")

            reviewed = self.final_candidate_cv['selected_candidate']
            f.write("Final-formula conditional CV (candidate-selection diagnostic):\n")
            f.write(
                f"  Raw R² = {reviewed['cv_raw_r2_mean']:.6f} ± "
                f"{reviewed['cv_raw_r2_std']:.6f}\n\n"
            )
            comparison_files = self.final_candidate_cv.get('comparison_files', {})
            if comparison_files:
                f.write("One-SE official candidates and two-SE simpler alternatives "
                        "(at most ten per layer; no test-based ranking):\n")
                for filename in comparison_files.values():
                    f.write(f"  {filename}\n")
                f.write("\n")

            for baseline in [self.results['ridge_baseline'], self.results['ridge_quadratic_baseline']]:
                f.write(f"Ridge {baseline['basis']} baseline with final SR features on the identical outer split:\n")
                f.write(f"  Selected alpha = {baseline['selected_alpha']:.6g}\n")
                f.write(
                    f"  CV raw R² = {baseline['cv_raw_r2_mean']:.6f} ± "
                    f"{baseline['cv_raw_r2_std']:.6f}\n"
                )
                f.write(f"  Test R² = {baseline['test_r2']:.6f}\n")
                f.write(f"  Test RMSE = {baseline['test_rmse']:.6f}\n")
                f.write(f"  Test MAE = {baseline['test_mae']:.6f}\n")
                f.write(
                    "  The baseline is diagnostic only and does not change Score or "
                    "the symbolic-formula selection.\n\n"
                )

            f.write("-"*100 + "\n")
            f.write("10. Configuration Search Results\n")
            f.write("-"*100 + "\n\n")

            f.write(f"Completed configurations ({len(self.all_search_results)} total):\n\n")
            for idx, result in enumerate(self.all_search_results):
                is_selected = (
                    " [SELECTED]"
                    if result['config_id'] == self.best_result['config_id']
                    else ""
                )
                f.write(
                    f"  Configuration {result['iteration']:3d}: raw R²="
                    f"{result['cv_raw_r2_mean']:.6f}±{result['cv_raw_r2_std']:.6f}; "
                    f"mean excess complexity="
                    f"{result['cv_excess_complexity_mean']:.2f}; "
                    f"mean structural complexity={result['cv_complexity_mean']:.2f}"
                    f"{is_selected}\n"
                )
            f.write("[SELECTED] = one-standard-error selection\n\n")

            f.write("Detailed search results:\n\n")
            for result in self.all_search_results:
                f.write(f"Iteration {result['iteration']}:\n")
                f.write(f"  CV raw R²: {result['cv_raw_r2_mean']:.6f} ± {result['cv_raw_r2_std']:.6f}\n")
                f.write(f"  CV RMSE : {result['cv_rmse_mean']:.6f} ± {result['cv_rmse_std']:.6f}\n")
                f.write(f"  CV MAE  : {result['cv_mae_mean']:.6f} ± {result['cv_mae_std']:.6f}\n")
                f.write(
                    "  Mean excess complexity: "
                    f"{result['cv_excess_complexity_mean']:.3f}\n"
                )
                f.write(
                    "  Direct-quadratic fold frequency: "
                    f"{result['cv_direct_quadratic_fold_frequency']:.1%}\n"
                )
                f.write(
                    "  Mean PySR structural complexity (audit only): "
                    f"{result['cv_complexity_mean']:.3f}\n"
                )
                f.write("\n")

            f.write("Conservatively pruned configurations:\n")
            for result in self.pruned_search_results:
                f.write(
                    f"  {result['config_id']}: completed folds="
                    f"{result['completed_folds']}/{result['total_folds']}; "
                    f"optimistic raw R²={result['prune_details']['optimistic_final_mean']:.6f}\n"
                )
            f.write("Failed configurations:\n")
            for result in self.failed_search_results:
                f.write(f"  {result['config_id']}: {result['error']}\n")

            f.write("\n")
            f.write("="*100 + "\n")
            f.write("                               End of Report\n")
            f.write("="*100 + "\n")

        self.logger.info(f"TXT results saved to: {txt_file}")

        json_file = os.path.join(self.output_path, 'results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json_results = self.results.copy()
            json_results['all_search_results'] = self.all_search_results
            json_results['pruned_search_results'] = self.pruned_search_results
            json_results['failed_search_results'] = self.failed_search_results
            json_results['search_summary'] = self.search_summary
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON results saved to: {json_file}")

        atomic_write_json(
            Path(self.output_path) / 'ridge_baseline.json',
            self.ridge_baseline,
        )
        atomic_write_json(
            Path(self.output_path) / 'ridge_quadratic_baseline.json',
            self.ridge_quadratic_baseline,
        )
        prediction_records = build_prediction_records(
            self.train_names,
            self.train_source_rows,
            self.y_train,
            y_pred_train,
            split='Train',
            baseline_predicted=self.ridge_train_predictions,
        )
        prediction_records.extend(
            build_prediction_records(
                self.test_names,
                self.test_source_rows,
                self.y_test,
                y_pred_test,
                split='Test',
                baseline_predicted=self.ridge_test_predictions,
            )
        )
        combined_pred = pd.DataFrame(prediction_records)
        combined_pred['Ridge_quadratic_Predicted'] = np.concatenate([
            self.ridge_quadratic_train_predictions, self.ridge_quadratic_test_predictions,
        ])
        combined_pred.to_csv(os.path.join(self.output_path, 'predictions.csv'), index=False)

        self.logger.info(
            "Predictions saved with Name/source_row_index traceability: predictions.csv"
        )
        self.logger.info("Ridge baselines saved to: ridge_baseline.json, ridge_quadratic_baseline.json")

        self._save_model_files()
        self.plot_predictions(y_pred_train, y_pred_test)
        if self.final_candidate_cv.get('candidate_layers'):
            directory = export_candidate_results(self.output_path, dpi=FIGURE_DPI)
            self.logger.info("Per-candidate diagnostics saved to: %s", directory)

    def _save_model_files(self):
        """Persist required numerical parameters before optional pickle exports."""
        scaler_params = {
            'feature_names': self.feature_columns,
            'feature_metadata': FEATURE_METADATA,
            'target_metadata': TARGET_METADATA,
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'var': self.scaler.var_.tolist(),
        }
        scaler_json_file = Path(self.output_path) / 'scaler_params.json'
        atomic_write_json(scaler_json_file, scaler_params)
        self.logger.info("Scaler parameters saved to: %s", scaler_json_file)

        for filename, value in (
            ('pysr_model.pkl', self.model),
            ('scaler.pkl', self.scaler),
        ):
            try:
                with open(Path(self.output_path) / filename, 'wb') as handle:
                    pickle.dump(value, handle)
                self.logger.info("Saved optional pickle: %s", filename)
            except Exception as exc:
                self.logger.warning("Failed to save optional pickle %s: %s", filename, exc)

        try:
            bundled_file = os.path.join(self.output_path, 'pysr_model_with_scaler.pkl')
            bundle = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'selected_candidate_position': int(self.final_candidate_position),
                'prediction_contract': 'model.predict(X_scaled, index=selected_candidate_position)',
                'equation_scaled': self.results.get('equation_scaled'),
                'equation_raw': self.results.get('equation_raw'),
                'equation_audit': self.equation_audit,
            }
            with open(bundled_file, 'wb') as f:
                pickle.dump(bundle, f)
            self.logger.info(f"Model+scaler bundle saved to: {bundled_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save model+scaler bundle: {str(e)}")

    def plot_predictions(self, y_pred_train, y_pred_test):
        """Plot predicted vs actual values."""
        self.logger.info("\nGenerating visualization plots...")

        plt.figure(figsize=(10, 8), dpi=FIGURE_DPI)

        plt.scatter(self.y_train, y_pred_train, alpha=0.3, color='red', s=100, label='Training Set')
        plt.scatter(self.y_test, y_pred_test, alpha=0.6, color='blue', s=100, label='Test Set')

        all_actual = np.concatenate([self.y_train, self.y_test])
        all_pred = np.concatenate([y_pred_train, y_pred_test])
        min_val = min(all_actual.min(), all_pred.min())
        max_val = max(all_actual.max(), all_pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=3, label='Ideal')

        plt.xlabel('Calculated Value', fontsize=20)
        plt.ylabel('Predicted Value', fontsize=20)
        plt.tick_params(labelsize=14)
        plt.legend(fontsize=16, loc='best')

        if 'cv_raw_r2_mean' in self.results:
            title = (f'Train R²={self.results["train_r2"]:.4f}, Test R²={self.results["test_r2"]:.4f}\n'
                     f'Configuration-search CV R²={self.results["cv_raw_r2_mean"]:.4f}±{self.results["cv_raw_r2_std"]:.4f}')
        else:
            title = f'Train R²={self.results["train_r2"]:.4f}, Test R²={self.results["test_r2"]:.4f}'
        plt.title(title, fontsize=20, pad=10)

        plt.text(0.95, 0.05, f'Test R² = {self.results["test_r2"]:.4f}',
                transform=plt.gca().transAxes,
                fontsize=18,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.grid(alpha=0.3)
        plot_file = os.path.join(self.output_path, 'prediction_plot_combined.png')
        plt.savefig(plot_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Combined prediction plot saved to: {plot_file}")

        self.plot_residuals(y_pred_train, y_pred_test)

    def plot_residuals(self, y_pred_train, y_pred_test):
        """Plot residual distribution."""
        residuals_train = self.y_train - y_pred_train
        residuals_test = self.y_test - y_pred_test

        plt.figure(figsize=(10, 8), dpi=FIGURE_DPI)

        plt.scatter(y_pred_train, residuals_train, alpha=0.3, color='red', s=100, label='Training Set')
        plt.scatter(y_pred_test, residuals_test, alpha=0.6, color='blue', s=100, label='Test Set')

        all_pred = np.concatenate([y_pred_train, y_pred_test])
        x_range = [all_pred.min(), all_pred.max()]
        plt.plot(x_range, [0, 0], '--', color='black', linewidth=3, label='Zero Residual')

        plt.xlabel('Predicted Value', fontsize=20)
        plt.ylabel('Residual (Actual - Predicted)', fontsize=20)
        plt.tick_params(labelsize=14)
        plt.legend(fontsize=16, loc='best')

        train_rmse = np.sqrt(np.mean(residuals_train**2))
        test_rmse = np.sqrt(np.mean(residuals_test**2))

        if 'cv_rmse_mean' in self.results:
            title = (f'Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}\n'
                     f'Configuration-search CV RMSE={self.results["cv_rmse_mean"]:.4f}±{self.results["cv_rmse_std"]:.4f}')
        else:
            title = f'Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}'

        plt.title(title, fontsize=20, pad=10)
        plt.grid(alpha=0.3)

        residual_file_combined = os.path.join(self.output_path, 'residual_plot_combined.png')
        plt.savefig(residual_file_combined, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Combined residual plot saved to: {residual_file_combined}")

    def plot_target_distribution(self, y_train, y_test):
        """Plot target variable distribution histogram."""
        self.logger.info("\nGenerating target variable distribution plot...")

        plt.figure(figsize=(12, 8), dpi=FIGURE_DPI)

        all_y = np.concatenate([y_train, y_test])
        q75, q25 = np.percentile(all_y, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(all_y) ** (1/3))
        n_bins = int((all_y.max() - all_y.min()) / bin_width) if bin_width > 0 else 30
        n_bins = min(max(n_bins, 20), 50)

        bins = np.linspace(all_y.min(), all_y.max(), n_bins + 1)

        plt.hist(y_train, bins=bins, alpha=0.4, color='red',
                label=f'Training Set (n={len(y_train)})', edgecolor='darkred', linewidth=1.5)

        plt.hist(y_test, bins=bins, alpha=0.7, color='blue',
                label=f'Test Set (n={len(y_test)})', edgecolor='darkblue', linewidth=1.5)

        plt.xlabel(f'{self.target_column} Value', fontsize=22)
        plt.ylabel('Frequency', fontsize=22)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, loc='upper left')
        plt.title(f'{self.target_column} Distribution', fontsize=24, pad=10)
        plt.grid(alpha=0.3, axis='y')

        plot_file = os.path.join(self.output_path, f'{self.target_column}_distribution.png')
        plt.savefig(plot_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Target variable distribution plot saved to: {plot_file}")

    def run(self):
        """Run the complete symbolic regression analysis pipeline."""
        try:
            self.logger.info("\n" + "="*100)
            self.logger.info("                    Starting Symbolic Regression Analysis")
            self.logger.info("="*100)
            self.logger.info(f"Output folder: {self.output_path}\n")

            self.load_data()
            self.preprocess_data()
            self.run_symbolic_regression()
            y_pred_train, y_pred_test = self.analyze_results()
            self.save_results(y_pred_train, y_pred_test)

            self.logger.info("\n" + "="*100)
            self.logger.info("                    Analysis Complete!")
            self.logger.info("="*100)
            self.logger.info(f"\nAnalysis results saved to: {self.output_path}")
            self.logger.info("\n" + "="*100)
            self.logger.info("Best equation found:")
            self.logger.info("="*100)
            self.logger.info(f"\n{self.target_column} = {self.results['equation']}\n")

            self.logger.info(f"Test R² = {self.results['test_r2']:.6f}")
            self.logger.info(f"Test RMSE = {self.results['test_rmse']:.6f}\n")

        except Exception as e:
            self.logger.error(f"\nError during analysis: {str(e)}", exc_info=True)
            raise
        finally:
            self._shutdown_fold_executor()


def execute_fold_job_worker(job):
    """Spawn-safe entry point for one isolated PySR fold job."""
    worker_dir = Path(job['worker_dir'])
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker = SymbolicRegressionAnalysis.__new__(SymbolicRegressionAnalysis)
    worker.output_path = str(worker_dir)
    worker.pysr_run_dir = worker_dir / 'pysr_runs'
    worker.pysr_run_dir.mkdir(parents=True, exist_ok=True)
    worker.progress_file = worker_dir / 'search_progress.jsonl'
    worker.progress_status_file = worker_dir / 'progress_status.json'
    worker.progress_event_count = 0
    worker._runtime_manifest_observed_written = True
    worker._overall_search_progress = None
    worker._search_started_monotonic = None
    worker._fold_executor = None
    worker.feature_columns = list(job['feature_columns'])
    worker.required_feature_count = job['required_feature_count']
    worker.logger = logging.getLogger(
        f"sr_fold_worker.{job['config_id']}.fold_{int(job['fold']):02d}"
    )
    worker.logger.info(
        "Starting %s with %d Julia threads",
        job['run_id'],
        JULIA_THREADS,
    )
    try:
        result = worker._fit_fold_job(job)
    except Exception as exc:
        raise RuntimeError(f'{type(exc).__name__}: {exc}') from None
    worker.logger.info(
        "Completed %s: R²=%.6f, iterations=%d/%d, stop=%s",
        job['run_id'],
        result['raw_r2'],
        result['execution']['requested_iterations'],
        result['execution']['maximum_iterations'],
        result['execution']['stop_reason'],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run symbolic regression for one cumulative campaign."
    )
    add_campaign_arguments(parser)
    parser.add_argument(
        '--features', nargs='+', default=None, metavar='DESCRIPTOR',
        help='Candidate descriptors (raw or internal names); default: the configured six.',
    )
    parser.add_argument(
        '--selected-count', type=int, default=REQUIRED_FEATURE_COUNT, metavar='N',
        help='Exact number of variables in the simplified formula (default: 6).',
    )
    parser.add_argument(
        '--max-excess-complexity', type=int, default=MAX_EXCESS_COMPLEXITY, metavar='N',
        help='Shared excess-term limit for search, refinement and review (default: 5).',
    )
    parser.add_argument(
        '--resume-run',
        type=Path,
        default=None,
        help=(
            'Resume an incomplete run directory below outputs/symbolic_regression; '
            'only strictly validated terminal configuration checkpoints are reused.'
        ),
    )
    args = parser.parse_args()
    campaign_round = (
        None
        if args.resume_run is not None and args.campaign_round is None
        else resolve_campaign_round(args.campaign_round)
    )

    try:
        analyzer = SymbolicRegressionAnalysis(
            campaign_round,
            resume_run=args.resume_run,
            features=args.features,
            selected_count=args.selected_count,
            max_excess_complexity=args.max_excess_complexity,
        )
        analyzer.run()

    except Exception as e:
        print("\n" + "="*80)
        print("                    Execution Failed!")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("  1. Data file path is correct")
        print("  2. Data file format is correct (CSV or TSV)")
        print("  3. Column names match definitions in code")
        print("\nDetailed error information in log file")

        traceback.print_exc()
        raise SystemExit(1)
