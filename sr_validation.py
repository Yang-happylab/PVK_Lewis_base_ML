"""Validation helpers for the symbolic-regression workflow.

This module deliberately does not import PySR or Julia.  The small functions
can therefore be tested with fake models while the production pipeline still
uses PySR's positional candidate API.
"""

from __future__ import annotations

import math
import os
import statistics
from functools import partial
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed, parallel_config
from sklearn.covariance import MinCovDet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sr_efficiency import used_features
from sr_reporting import (
    _evaluate_compiled_expression,
    evaluate_expression,
    neutral_parameter_start,
    parameterize_floating_constants,
    parse_source_expression,
)


def analyze_term_budget(
    equation: str,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Count combined terms and shared per-variable nonlinear allowances.

    One linear term and one direct square, real root, exponential or pairwise
    interaction per feature are free. Interactions use both features' allowance.
    Additional uses and composite terms consume the shared excess budget.
    """
    feature_names = [str(name) for name in feature_names]
    symbols = {name: sp.Symbol(name) for name in feature_names}
    result: dict[str, Any] = {
        "policy": "one_linear_and_one_nonlinear_or_interaction_per_feature_free",
        "current_default_nonlinear_operator": ["square", "sqrt", "exp"],
        "linear_features": [],
        "direct_nonlinear_features": [],
        "direct_quadratic_features": [],
        "has_direct_quadratic": False,
        "free_linear_term_count": 0,
        "free_nonlinear_term_count": 0,
        "free_direct_quadratic_term_count": 0,
        "penalized_terms": [],
        "excess_complexity": 0,
        "analysis_error": None,
        "term_count": 0,
        "total_term_count": 0,
        "linear_term_count": 0,
        "nonlinear_term_count": 0,
        "interaction_term_count": 0,
        "nonlinear_uses_per_feature": {},
    }
    try:
        parsed = sp.sympify(
            equation if isinstance(equation, sp.Basic) else str(equation),
            locals={**symbols, "square": lambda value: value**2},
        )
        # A shared parsed expression can still contain PySR's square token.
        # Expand its budget without changing the separate simplified feature gate.
        expression = sp.expand(
            sp.powsimp(parsed.replace(sp.Function("square"), lambda value: value**2), combine="exp"),
            power_exp=False, power_base=False, log=False,
        )
    except Exception as exc:
        result["analysis_error"] = f"parse_failure:{type(exc).__name__}:{exc}"
        return result

    unknown_symbols = sorted(
        str(symbol)
        for symbol in expression.free_symbols
        if str(symbol) not in symbols
    )
    if unknown_symbols:
        result["analysis_error"] = (
            "unknown_symbols:" + ",".join(unknown_symbols)
        )
        return result

    linear_seen: set[str] = set()
    nonlinear_seen: set[str] = set()
    quadratic_seen: set[str] = set()
    penalized_terms: list[dict[str, Any]] = []

    def _direct_nonlinear_feature(core: sp.Expr) -> tuple[str | None, str | None]:
        if isinstance(core, sp.Pow) and core.base in symbols.values():
            if core.exp == 2:
                return str(core.base), "square"
        argument = None
        kind = None
        if isinstance(core, sp.Pow) and core.exp == sp.Rational(1, 2):
            argument, kind = core.base, "sqrt"
        elif core.func == sp.exp:
            argument, kind = core.args[0], "exp"
        if argument is not None and len(argument.free_symbols) == 1:
            try:
                if sp.Poly(argument, *symbols.values()).total_degree() <= 1:
                    return str(next(iter(argument.free_symbols))), kind
            except sp.PolynomialError:
                pass
        return None, None

    nonlinear_uses = {name: 0 for name in feature_names}
    term_count = nonlinear_count = interaction_count = 0
    for term in sp.Add.make_args(expression):
        _, core = term.as_independent(*symbols.values(), as_Add=False)
        term_symbols = sorted(
            (str(symbol) for symbol in core.free_symbols),
            key=feature_names.index,
        )
        if not term_symbols:
            continue
        term_count += 1

        if core in symbols.values():
            feature = str(core)
            if feature not in linear_seen:
                linear_seen.add(feature)
                continue

        nonlinear_feature, nonlinear_kind = _direct_nonlinear_feature(core)
        nonlinear_count += 1
        for feature in term_symbols:
            nonlinear_uses[feature] += 1
        if nonlinear_feature is not None:
            nonlinear_seen.add(nonlinear_feature)
            if nonlinear_kind == "square":
                quadratic_seen.add(nonlinear_feature)
            continue

        active_feature_count = len(set(term_symbols))
        if active_feature_count > 1:
            interaction_count += 1
        if (
            active_feature_count == 2
            and core == sp.prod(symbols[name] for name in term_symbols)
        ):
            continue
        # SymPy counts the structural 1/2 in a root as a division as well.
        # A real sqrt is one unary operation, just like square or exp.
        root_divisions = sum(
            isinstance(node, sp.Pow) and node.exp == sp.Rational(1, 2)
            for node in sp.preorder_traversal(core)
        )
        operation_count = max(1, int(sp.count_ops(core, visual=False)) - root_divisions)
        degree_excess = 0
        if isinstance(core, sp.Pow) and core.base in symbols.values():
            exponent = core.exp
            if exponent.is_integer and int(exponent) > 2:
                degree_excess = int(exponent) - 2
        penalty = operation_count + max(0, active_feature_count - 1) + degree_excess
        if nonlinear_feature is not None:
            reason = "additional_direct_nonlinear_term_for_feature"
        elif active_feature_count > 1:
            reason = "feature_interaction_or_composite_term"
        elif isinstance(core, sp.Pow) and core.base in symbols.values():
            reason = "power_above_direct_quadratic_allowance"
        else:
            reason = "nested_or_composite_nonlinear_term"
        penalized_terms.append(
            {
                "term": str(term),
                "features": term_symbols,
                "reason": reason,
                "penalty": int(penalty),
            }
        )

    for feature, uses in nonlinear_uses.items():
        if uses > 1:
            penalized_terms.append({
                "term": feature, "features": [feature],
                "reason": "additional_nonlinear_use_for_feature",
                "penalty": uses - 1,
            })

    ordered_linear = [name for name in feature_names if name in linear_seen]
    ordered_nonlinear = [name for name in feature_names if name in nonlinear_seen]
    ordered_quadratic = [name for name in feature_names if name in quadratic_seen]
    result.update(
        {
            "linear_features": ordered_linear,
            "direct_nonlinear_features": ordered_nonlinear,
            "direct_quadratic_features": ordered_quadratic,
            "has_direct_quadratic": bool(ordered_quadratic),
            "free_linear_term_count": len(ordered_linear),
            "free_nonlinear_term_count": len(ordered_nonlinear),
            "free_direct_quadratic_term_count": len(ordered_quadratic),
            "penalized_terms": penalized_terms,
            "term_count": term_count,
            "total_term_count": term_count,
            "linear_term_count": len(ordered_linear),
            "nonlinear_term_count": nonlinear_count,
            "interaction_term_count": interaction_count,
            "nonlinear_uses_per_feature": nonlinear_uses,
            "excess_complexity": int(
                sum(row["penalty"] for row in penalized_terms)
            ),
        }
    )
    return result


def inherit_cv_winner_candidates(
    best_result: Mapping[str, Any],
    X_outer_train_scaled: np.ndarray,
    y_outer_train: np.ndarray,
    feature_names: Sequence[str],
    outer_scaler_mean: Sequence[float],
    outer_scaler_scale: Sequence[float],
    *,
    required_feature_count: int,
    excess_complexity_ceiling: int,
    max_nfev: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carry one selected configuration's fold winners into a callable archive.

    The exact scaler substitution is ``z_fold = (outer_scale*z_outer +
    outer_mean - fold_mean)/fold_scale``. Only floating constants already in
    each fold formula are refitted, with at most ``max_nfev`` optimizer
    evaluations on the supplied outer-training rows. Scaler constants and
    integer structural constants remain fixed. An unsuccessful, worse, or
    ineligible refit retains the eligible transformed winner. This is candidate
    archive inheritance; it does not initialize the backend mutation population.

    ``complexity`` is an approximate SymPy preorder node count, not a backend
    tree-size measurement. Candidate selection still uses the audited excess
    term budget. Saved callables use a picklable top-level evaluator.
    """
    names = [str(name) for name in feature_names]
    symbols = [sp.Symbol(name) for name in names]
    X = np.asarray(X_outer_train_scaled, dtype=float)
    y = np.asarray(y_outer_train, dtype=float).reshape(-1)
    outer_mean = np.asarray(outer_scaler_mean, dtype=float)
    outer_scale = np.asarray(outer_scaler_scale, dtype=float)
    if (
        not names or len(set(names)) != len(names)
        or X.ndim != 2 or X.shape != (len(y), len(names)) or len(y) == 0
        or not np.isfinite(X).all() or not np.isfinite(y).all()
    ):
        raise ValueError("outer-training values must be finite and feature-aligned")
    if (
        outer_mean.shape != (len(names),) or outer_scale.shape != (len(names),)
        or not np.isfinite(outer_mean).all() or not np.isfinite(outer_scale).all()
        or np.any(outer_scale <= 0)
    ):
        raise ValueError("outer scaler must contain finite means and positive scales")
    if not isinstance(required_feature_count, (int, np.integer)) or not (
        1 <= required_feature_count <= len(names)
    ):
        raise ValueError("required_feature_count must fit the candidate feature pool")
    if (
        not isinstance(excess_complexity_ceiling, (int, np.integer))
        or excess_complexity_ceiling < 0 or int(max_nfev) < 1
    ):
        raise ValueError("excess ceiling must be nonnegative and max_nfev positive")

    config_id = str(best_result["config_id"])
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    accepted_seen: dict[tuple[str, str], int] = {}

    def checked_expression(expression):
        expression = sp.simplify(expression)
        if expression.free_symbols - set(symbols):
            raise ValueError("unknown_symbols")
        if len(expression.free_symbols) != required_feature_count:
            raise ValueError("incorrect_feature_count")
        budget = analyze_term_budget(expression, names)
        if budget["analysis_error"] is not None:
            raise ValueError("term_budget_analysis_failed")
        if budget["excess_complexity"] > excess_complexity_ceiling:
            raise ValueError("excess_complexity_ceiling_exceeded")
        return expression, budget

    for fold in best_result.get("fold_results", []):
        record = {
            "config_id": config_id,
            "fold": int(fold["fold"]),
            "repeat": int(fold.get("repeat", 1)),
            "source_candidate_position": fold.get("candidate_position"),
            "source_worker_dir": fold.get("worker_dir"),
            "source_equation": str(fold.get("backend_equation", fold.get("equation", ""))),
            "transformed_equation": None,
            "refit_equation": None,
            "status": "rejected",
            "failure_reason": None,
            "refit_status": "not_attempted",
            "refit_failure_reason": None,
            "refit_nfev": 0,
            "archive_position": None,
        }
        records.append(record)
        try:
            if fold.get("eligible") is False:
                raise ValueError("source_candidate_ineligible")
            if str(fold.get("config_id", config_id)) != config_id:
                raise ValueError("source_configuration_mismatch")
            source = parse_source_expression(record["source_equation"], names)
            source_features = used_features(source, names)
            if source.free_symbols - set(symbols):
                raise ValueError("unknown_source_symbols")
            if len(source_features) != required_feature_count:
                raise ValueError("incorrect_source_feature_count")
            fold_mean = np.asarray(fold["fold_scaler_mean"], dtype=float)
            fold_scale = np.asarray(fold["fold_scaler_scale"], dtype=float)
            if (
                fold_mean.shape != outer_mean.shape
                or fold_scale.shape != outer_scale.shape
                or not np.isfinite(fold_mean).all()
                or not np.isfinite(fold_scale).all() or np.any(fold_scale <= 0)
            ):
                raise ValueError("invalid_fold_scaler")
            record["fold_scaler_mean"] = fold_mean.tolist()
            record["fold_scaler_scale"] = fold_scale.tolist()
            substitutions = {
                symbol: (float(scale) * symbol + float(mean - fold_mu)) / float(fold_sigma)
                for symbol, mean, scale, fold_mu, fold_sigma in zip(
                    symbols, outer_mean, outer_scale, fold_mean, fold_scale
                )
            }
            transformed = source.xreplace(substitutions)
            record["transformed_equation"] = str(transformed)
            transformed, budget = checked_expression(transformed)
            record["transformed_term_budget"] = budget
            prediction = evaluate_expression(transformed, X, names)
            X_fold = (X * outer_scale + outer_mean - fold_mean) / fold_scale
            source_prediction = evaluate_expression(source, X_fold, names)
            record["scaler_conversion_max_abs_error"] = float(
                np.max(np.abs(prediction - source_prediction))
            )
            if not np.allclose(prediction, source_prediction, rtol=1e-9, atol=1e-10):
                raise ValueError("scaler_conversion_numerical_mismatch")
            loss = float(np.mean((prediction - y) ** 2))
            if not math.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            record["transformed_training_mse"] = loss
            parameterized, parameters, initial = parameterize_floating_constants(
                source, prefix="__cv_constant_"
            )
            source_template_key = sp.srepr(parameterized)
            parameterized = parameterized.xreplace(substitutions)
            record["parameterized_outer_equation"] = str(parameterized)
            record["initial_constants"] = dict(
                zip(map(str, parameters), map(float, initial))
            )
            accepted = transformed
            if parameters:
                try:
                    function = sp.lambdify(symbols + parameters, parameterized, "numpy")

                    def residuals(theta):
                        return _evaluate_compiled_expression(function, X, theta) - y

                    fit = least_squares(residuals, initial, method="trf", max_nfev=int(max_nfev))
                    record["refit_nfev"] = int(fit.nfev)
                    record["refitted_constants"] = {
                        str(parameter): float(value) if np.isfinite(value) else None
                        for parameter, value in zip(parameters, fit.x)
                    }
                    if not fit.success or not np.isfinite(fit.x).all():
                        raise ValueError("optimizer_not_converged")
                    fitted = parameterized.xreplace(
                        dict(zip(parameters, map(sp.Float, fit.x)))
                    )
                    record["refit_equation"] = str(fitted)
                    fitted, fitted_budget = checked_expression(fitted)
                    fitted_prediction = evaluate_expression(fitted, X, names)
                    fitted_loss = float(np.mean((fitted_prediction - y) ** 2))
                    record["refit_training_mse"] = fitted_loss if math.isfinite(fitted_loss) else None
                    if not math.isfinite(fitted_loss) or fitted_loss > loss:
                        raise ValueError("refit_did_not_improve_training_loss")
                    accepted, budget, loss = fitted, fitted_budget, fitted_loss
                    record["refit_status"] = "accepted"
                except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
                    record["refit_status"] = "retained_transformed_winner"
                    record["refit_failure_reason"] = f"{type(exc).__name__}:{exc}"
            else:
                record["refit_status"] = "no_floating_constants"

            record["accepted_equation"] = str(accepted)
            record["accepted_training_mse"] = loss
            record["accepted_term_budget"] = budget
            accepted_key = (sp.srepr(sp.expand(accepted)), source_template_key)
            if accepted_key in accepted_seen:
                position = accepted_seen[accepted_key]
                rows[position]["source_cv_folds"].append(record["fold"])
                record["status"] = "duplicate"
            else:
                position = len(rows)
                rows.append({
                    "complexity": sum(1 for _ in sp.preorder_traversal(accepted)),
                    "loss": loss,
                    "score": 0.0,
                    "equation": str(accepted).replace("**", "^"),
                    "sympy_format": accepted,
                    "lambda_format": partial(evaluate_expression, accepted, feature_names=names),
                    "candidate_origin": "cv_winner",
                    "source_cv_folds": [record["fold"]],
                    "refit_source_equation": record["source_equation"],
                })
                accepted_seen[accepted_key] = position
                record["status"] = "accepted"
            record["archive_position"] = position
        except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            record["failure_reason"] = f"{type(exc).__name__}:{exc}"

    audit = {
        "schema_version": 1,
        "selected_config_id": config_id,
        "source_scope": "selected_configuration_fold_winners_only",
        "inheritance_scope": "candidate_archive_only",
        "backend_mutation_population_seeded": False,
        "refit_scope": "original_floating_constants_on_outer_training_only",
        "scaler_coefficients_refitted": False,
        "external_test_used": False,
        "outer_training_row_count": len(y),
        "feature_names": names,
        "outer_scaler_mean": outer_mean.tolist(),
        "outer_scaler_scale": outer_scale.tolist(),
        "required_feature_count": int(required_feature_count),
        "excess_complexity_ceiling": int(excess_complexity_ceiling),
        "max_nfev_per_formula": int(max_nfev),
        "structural_complexity_source": "approximate_sympy_preorder_node_count",
        "source_fold_count": len(records),
        "inherited_candidate_count": len(rows),
        "rejected_fold_count": sum(record["status"] == "rejected" for record in records),
        "duplicate_fold_count": sum(record["status"] == "duplicate" for record in records),
        "folds": records,
    }
    columns = [
        "complexity", "loss", "score", "equation", "sympy_format", "lambda_format",
        "candidate_origin", "source_cv_folds", "refit_source_equation",
    ]
    return pd.DataFrame(rows, columns=columns), audit


def review_final_candidates(
    candidate_frame: pd.DataFrame,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    cv_splits: Sequence[tuple[Sequence[int], Sequence[int]]],
    outer_scaler_mean: Sequence[float],
    outer_scaler_scale: Sequence[float],
    *,
    required_feature_count: int,
    excess_complexity_ceiling: int,
    n_splits: int = 4,
    n_repeats: int = 5,
    max_nfev: int = 1000,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Review discovered structures by conditional repeated-fold CV.

    Each template is interpreted directly on freshly standardized fold inputs.
    Only its original floating constants are fitted, using deterministic +1/-1
    starts rather than discovery estimates. Inherited rows must retain their
    pre-scaler ``refit_source_equation``. The same fitting procedure refits the
    selected template on all outer-training rows, in the supplied scaler basis.

    These structures were already discovered using outer-training data. This
    review selects among them; its scores are not independent nested-SR
    validation. Training loss chooses constants within a template, never the
    final template. An empty returned frame preserves an auditable failure.
    """
    started = monotonic()
    names = [str(name) for name in feature_names]
    symbols = [sp.Symbol(name) for name in names]
    X = np.asarray(X_train_raw, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    outer_mean = np.asarray(outer_scaler_mean, dtype=float)
    outer_scale = np.asarray(outer_scaler_scale, dtype=float)
    if (
        not names or len(set(names)) != len(names)
        or X.ndim != 2 or X.shape != (len(y), len(names))
        or len(y) < 8 or not np.isfinite(X).all() or not np.isfinite(y).all()
    ):
        raise ValueError("outer-training values must be finite and feature-aligned")
    if (
        outer_mean.shape != (len(names),) or outer_scale.shape != (len(names),)
        or not np.isfinite(outer_mean).all() or not np.isfinite(outer_scale).all()
        or np.any(outer_scale <= 0)
    ):
        raise ValueError("outer scaler must contain finite means and positive scales")
    if (
        not isinstance(required_feature_count, (int, np.integer))
        or not 1 <= required_feature_count <= len(names)
        or not isinstance(excess_complexity_ceiling, (int, np.integer))
        or excess_complexity_ceiling < 0 or int(max_nfev) < 1
    ):
        raise ValueError("invalid feature count, excess ceiling or refit budget")

    if (
        not isinstance(n_splits, (int, np.integer)) or n_splits < 2
        or not isinstance(n_repeats, (int, np.integer)) or n_repeats < 2
    ):
        raise ValueError("final review requires at least two folds and two repeats")
    splits = list(cv_splits)
    if len(splits) != n_splits * n_repeats:
        raise ValueError(f"final review requires exactly {n_splits} folds x {n_repeats} repeats")
    folds = []
    for position, (train_idx, val_idx) in enumerate(splits):
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        if (
            train_idx.ndim != 1 or val_idx.ndim != 1
            or train_idx.dtype.kind not in "iu" or val_idx.dtype.kind not in "iu"
            or len(train_idx) == 0 or len(val_idx) < 2
            or not np.array_equal(np.sort(np.r_[train_idx, val_idx]), np.arange(len(y)))
        ):
            raise ValueError("each CV fold must partition outer training rows exactly once")
        scaler, X_fit, X_val = prepare_fold_scaled_data(X, train_idx, val_idx)
        folds.append((train_idx, val_idx, scaler, X_fit, X_val))
    for offset in range(0, len(folds), n_splits):
        validation_rows = np.concatenate([folds[i][1] for i in range(offset, offset + n_splits)])
        if not np.array_equal(np.sort(validation_rows), np.arange(len(y))):
            raise ValueError("each CV repeat must validate every outer-training row once")

    rule = "best_repeat_mean_one_se_then_excess_terms_parameters_structure"
    audit: dict[str, Any] = {
        "schema_version": 2,
        "status": "failed",
        "failure_reason": None,
        "scope": "fixed_structure_cv_conditional_on_previously_discovered_candidates",
        "independent_structure_validation": False,
        "external_test_used": False,
        "selection_rule": rule,
        "cv_n_splits": int(n_splits),
        "cv_n_repeats": int(n_repeats),
        "feature_names": names,
        "outer_training_row_count": len(y),
        "required_feature_count": int(required_feature_count),
        "excess_complexity_ceiling": int(excess_complexity_ceiling),
        "max_nfev_per_start": int(max_nfev),
        "constant_initialization": "two_plus_one_minus_one_starts_with_training_X_only_affine_root_offset_feasibility_adjustment",
        "scaler_scope": "each_fold_training_rows_only",
        "structural_complexity_source": "parameterized_source_template_sympy_preorder_node_count",
        "candidates": [],
        "selected_candidate": None,
        "candidate_layers": {},
        "shortlisted_candidates": [],
    }
    columns = [
        "complexity", "loss", "score", "equation", "sympy_format", "lambda_format",
        "candidate_origin", "source_cv_folds", "refit_source_equation",
        "source_candidate_positions", "source_candidate_origins",
    ]

    def finish(rows):
        audit["elapsed_seconds"] = monotonic() - started
        return pd.DataFrame(rows, columns=columns), audit

    def check(expression):
        expression = sp.simplify(expression)
        if expression.free_symbols - set(symbols):
            raise ValueError("unknown_symbols")
        if len(expression.free_symbols) != required_feature_count:
            raise ValueError("incorrect_feature_count")
        budget = analyze_term_budget(expression, names)
        if budget["analysis_error"] is not None:
            raise ValueError("term_budget_analysis_failed")
        if budget["excess_complexity"] > excess_complexity_ceiling:
            raise ValueError("excess_complexity_ceiling_exceeded")
        return expression, budget

    def fit_constants(template, parameters, function, X_fit, y_fit, trace):
        successes = []
        trace["starts"] = []
        if not parameters:
            expression, budget = check(template)
            prediction = evaluate_expression(expression, X_fit, names)
            loss = float(np.mean((prediction - y_fit) ** 2))
            if not math.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            return expression, budget, loss
        for sign in (1.0, -1.0):
            initial = np.full(len(parameters), sign)
            attempt = {"initial": initial.tolist(), "status": "failed", "failure_reason": None}
            trace["starts"].append(attempt)
            try:
                attempt["neutral_initial"] = initial.tolist()
                initial = neutral_parameter_start(template, parameters, X_fit, names, sign)
                attempt["training_root_domain_adjusted"] = not np.array_equal(initial, attempt["neutral_initial"])
                attempt["initial"] = initial.tolist()
                fit = least_squares(
                    lambda theta: _evaluate_compiled_expression(function, X_fit, theta) - y_fit,
                    initial, method="trf", max_nfev=int(max_nfev),
                )
                attempt["nfev"] = int(fit.nfev)
                attempt["coefficients"] = {
                    str(parameter): float(value) if np.isfinite(value) else None
                    for parameter, value in zip(parameters, fit.x)
                }
                if not fit.success or not np.isfinite(fit.x).all():
                    raise ValueError("optimizer_not_converged")
                expression = template.xreplace(dict(zip(parameters, map(sp.Float, fit.x))))
                expression, budget = check(expression)
                prediction = evaluate_expression(expression, X_fit, names)
                loss = float(np.mean((prediction - y_fit) ** 2))
                if not math.isfinite(loss):
                    raise ValueError("nonfinite_training_loss")
                attempt.update(status="accepted", training_mse=loss)
                successes.append((loss, expression, budget, attempt["coefficients"]))
            except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
                attempt["failure_reason"] = f"{type(exc).__name__}:{exc}"
        if not successes:
            raise ValueError("no_converged_eligible_neutral_start")
        loss, expression, budget, coefficients = min(successes, key=lambda result: result[0])
        trace["coefficients"] = coefficients
        return expression, budget, loss

    def review_template(record, template, parameters):
        try:
            function = sp.lambdify(symbols + parameters, template, "numpy")
            for number, (train_idx, val_idx, scaler, X_fit, X_val) in enumerate(folds, start=1):
                fold_trace = {
                    "fold": number, "repeat": (number - 1) // n_splits + 1,
                    "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
                    "fold_scaler_mean": scaler.mean_.tolist(), "fold_scaler_scale": scaler.scale_.tolist(),
                    "status": "failed", "failure_reason": None,
                }
                record["folds"].append(fold_trace)
                try:
                    expression, budget, loss = fit_constants(
                        template, parameters, function, X_fit, y[train_idx], fold_trace
                    )
                    prediction = evaluate_expression(expression, X_val, names)
                    raw_r2 = float(r2_score(y[val_idx], prediction))
                    if not math.isfinite(raw_r2):
                        raise ValueError("nonfinite_validation_r2")
                    fold_trace.update(
                        status="accepted", equation=str(expression), training_mse=loss,
                        raw_r2=raw_r2, rmse=float(np.sqrt(mean_squared_error(y[val_idx], prediction))),
                        mae=float(mean_absolute_error(y[val_idx], prediction)),
                        term_budget=budget,
                    )
                except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
                    fold_trace["failure_reason"] = f"{type(exc).__name__}:{exc}"
                    raise ValueError(f"fold_{number}_refit_or_prediction_failed") from exc
            repeat_means = [
                statistics.fmean(fold["raw_r2"] for fold in record["folds"][offset:offset + n_splits])
                for offset in range(0, len(folds), n_splits)
            ]
            record.update(
                status="eligible", cv_raw_r2_mean=statistics.fmean(repeat_means),
                cv_raw_r2_std=float(np.std([fold["raw_r2"] for fold in record["folds"]])),
                cv_raw_r2_standard_error=statistics.stdev(repeat_means) / math.sqrt(n_repeats),
                repeat_mean_raw_r2=repeat_means,
                cv_rmse_mean=statistics.fmean(fold["rmse"] for fold in record["folds"]),
                cv_mae_mean=statistics.fmean(fold["mae"] for fold in record["folds"]),
                excess_complexity=max(fold["term_budget"]["excess_complexity"] for fold in record["folds"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            record["failure_reason"] = f"{type(exc).__name__}:{exc}"
        return record

    templates: dict[str, dict[str, Any]] = {}
    template_jobs = []
    for position, (_, row) in enumerate(candidate_frame.iterrows()):
        origin = row.get("candidate_origin")
        origin = origin if isinstance(origin, str) else "evolution"
        source = row.get("refit_source_equation")
        record: dict[str, Any] = {
            "candidate_id": f"SR_C{position + 1:04d}",
            "source_candidate_positions": [position],
            "source_candidate_origins": [origin],
            "source_cv_folds": list(row.get("source_cv_folds")) if isinstance(row.get("source_cv_folds"), list) else [],
            "status": "rejected", "failure_reason": None, "folds": [],
        }
        try:
            archive_expression = row.get("sympy_format", row.get("equation", ""))
            archive_expression = parse_source_expression(archive_expression, names)
            check(archive_expression)
            if not math.isfinite(float(row["loss"])):
                raise ValueError("nonfinite_archive_training_loss")
            if not isinstance(source, (str, sp.Basic)):
                if origin == "cv_winner":
                    raise ValueError("inherited_candidate_missing_original_source_template")
                source = row.get("equation", archive_expression)
            source_equation = str(source)
            source = parse_source_expression(source, names)
            _, source_budget = check(source)
            template, parameters, _ = parameterize_floating_constants(source, prefix="__review_c")
            signature = sp.srepr(template)
            if signature in templates:
                previous = templates[signature]
                previous["source_candidate_positions"].append(position)
                previous["source_candidate_origins"] = sorted(set(previous["source_candidate_origins"] + [origin]))
                previous["source_cv_folds"] = sorted(set(previous["source_cv_folds"] + record["source_cv_folds"]))
                continue
            record.update(
                source_equation=source_equation, parameterized_equation=str(template),
                parameter_names=list(map(str, parameters)),
                independent_parameter_count=len(parameters),
                structural_complexity=sum(1 for _ in sp.preorder_traversal(template)),
                source_term_budget=source_budget,
                total_term_count=source_budget["total_term_count"],
                linear_term_count=source_budget["linear_term_count"],
                nonlinear_term_count=source_budget["nonlinear_term_count"],
                interaction_term_count=source_budget["interaction_term_count"],
            )
            templates[signature] = record
            audit["candidates"].append(record)
            template_jobs.append((record, template, parameters))
        except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            record["failure_reason"] = f"{type(exc).__name__}:{exc}"
            if not any(item is record for item in audit["candidates"]):
                audit["candidates"].append(record)

    workers = max(1, min(int(n_jobs), len(template_jobs), os.cpu_count() or 1))
    from threadpoolctl import threadpool_limits
    if workers == 1:
        with threadpool_limits(limits=1):
            reviewed = [review_template(*job) for job in template_jobs]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            reviewed = Parallel(n_jobs=workers)(
                delayed(review_template)(*job) for job in template_jobs
            )
    # Ordered collection preserves archive order and the existing tie breakers.
    for (record, _, _), result in zip(template_jobs, reviewed):
        record.update(result)

    eligible = [record for record in audit["candidates"] if record["status"] == "eligible"]
    audit["source_archive_row_count"] = len(candidate_frame)
    audit["unique_template_count"] = len(templates)
    audit["eligible_template_count"] = len(eligible)
    if not eligible:
        audit["failure_reason"] = f"no_candidate_completed_all_{len(folds)}_fold_refits"
        return finish([])
    best = min(eligible, key=lambda record: (-record["cv_raw_r2_mean"], record["cv_raw_r2_std"], record["source_candidate_positions"][0]))
    threshold = best["cv_raw_r2_mean"] - best["cv_raw_r2_standard_error"]
    two_se_threshold = best["cv_raw_r2_mean"] - 2 * best["cv_raw_r2_standard_error"]
    within_one_se = [record for record in eligible if record["cv_raw_r2_mean"] >= threshold]

    def simplicity(record):
        return (
            record["excess_complexity"], record["total_term_count"],
            record["independent_parameter_count"], record["structural_complexity"],
        )

    def selection_key(record):
        return (*simplicity(record),
        -record["cv_raw_r2_mean"], record["cv_raw_r2_std"], record["source_candidate_positions"][0],
        )

    selected = min(within_one_se, key=selection_key)
    for record in eligible:
        record.update(
            delta_cv_raw_r2=record["cv_raw_r2_mean"] - best["cv_raw_r2_mean"],
            within_one_se=record["cv_raw_r2_mean"] >= threshold,
            within_two_se=record["cv_raw_r2_mean"] >= two_se_threshold,
            is_primary_selected=record is selected, is_highest_cv_mean=record is best,
            selected_in_one_se_layer=False, selected_in_two_se_layer=False,
        )

    def shortlist(pool):
        # Keep the two scientific anchors, then prefer the Pareto frontier.
        # Round-robin across simplicity levels avoids ten near-identical rows.
        chosen = [selected]
        if best is not selected:
            chosen.append(best)
        pending = [record for record in pool if record is not selected and record is not best]

        def dominated(record, others):
            return any(
                other is not record
                and all(a <= b for a, b in zip(simplicity(other), simplicity(record)))
                and other["cv_raw_r2_mean"] >= record["cv_raw_r2_mean"]
                and (simplicity(other) != simplicity(record)
                     or other["cv_raw_r2_mean"] > record["cv_raw_r2_mean"])
                for other in others
            )

        nondominated = {record["candidate_id"] for record in pool if not dominated(record, pool)}
        while pending and len(chosen) < 10:
            preferred = [record for record in pending if record["candidate_id"] in nondominated] or pending
            frontier = [record for record in preferred if not dominated(record, preferred)]
            levels: dict[tuple, list] = {}
            for record in sorted(frontier, key=selection_key):
                levels.setdefault(simplicity(record), []).append(record)
            while levels and len(chosen) < 10:
                active_levels = list(levels)
                remaining = 10 - len(chosen)
                if len(active_levels) > remaining:
                    active_levels = [active_levels[index] for index in np.linspace(
                        0, len(active_levels) - 1, remaining, dtype=int
                    )]
                for level in active_levels:
                    chosen.append(levels[level].pop(0))
                    if not levels[level]:
                        del levels[level]
                    if len(chosen) == 10:
                        break
            used = {record["candidate_id"] for record in chosen}
            pending = [record for record in pending if record["candidate_id"] not in used]
        return chosen

    layers = {
        "one_se": shortlist(within_one_se),
        "two_se": shortlist([record for record in eligible if record["within_two_se"]]),
    }
    for layer, records in layers.items():
        for record in records:
            record[f"selected_in_{layer}_layer"] = True
        audit["candidate_layers"][layer] = {
            "threshold": threshold if layer == "one_se" else two_se_threshold,
            "anchor_candidate_id": best["candidate_id"],
            "standard_error": best["cv_raw_r2_standard_error"],
            "standard_error_source": "sample_sd_of_repeat_mean_r2_divided_by_sqrt_repeat_count",
            "max_candidates": 10,
            "candidate_ids": [record["candidate_id"] for record in records],
            "selection_rule": "retain_primary_and_highest_cv_then_pareto_frontiers_with_simplicity_level_diversity",
            "official_model_selection": layer == "one_se",
        }
    union = {record["candidate_id"]: record for records in layers.values() for record in records}
    audit.update(
        highest_mean_source_candidate_position=best["source_candidate_positions"][0],
        highest_mean_cv_raw_r2=best["cv_raw_r2_mean"],
        highest_mean_standard_error=best["cv_raw_r2_standard_error"],
        one_se_threshold=threshold,
        two_se_threshold=two_se_threshold,
        highest_mean_candidate_id=best["candidate_id"],
        selected_candidate_id=selected["candidate_id"],
        one_se_eligible_source_positions=[record["source_candidate_positions"][0] for record in within_one_se],
        selected_source_candidate_position=selected["source_candidate_positions"][0],
    )
    final_trace: dict[str, Any] = {}
    audit["full_outer_training_refit"] = final_trace
    try:
        scaler = StandardScaler().fit(X)
        if not (
            np.allclose(scaler.mean_, outer_mean, rtol=1e-12, atol=1e-12)
            and np.allclose(scaler.scale_, outer_scale, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("supplied_outer_scaler_does_not_match_outer_training_rows")
        final_trace.update(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
        X_outer_scaled = scaler.transform(X)

        def refit_record(record):
            trace = {"status": "failed", "failure_reason": None}
            try:
                template = parse_source_expression(record["parameterized_equation"], names)
                parameters = [sp.Symbol(name) for name in record["parameter_names"]]
                function = sp.lambdify(symbols + parameters, template, "numpy")
                expression, budget, loss = fit_constants(template, parameters, function, X_outer_scaled, y, trace)
                prediction = evaluate_expression(expression, X_outer_scaled, names)
                denominator = float(np.sum((y - np.mean(y))**2))
                trace.update(
                    status="accepted", equation=str(expression), training_mse=loss,
                    training_rmse=float(np.sqrt(loss)),
                    training_mae=float(mean_absolute_error(y, prediction)),
                    training_raw_r2=1.0 - loss * len(y) / denominator if denominator > 0 else None,
                    term_budget=budget,
                )
            except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
                trace["failure_reason"] = f"{type(exc).__name__}:{exc}"
            return record["candidate_id"], trace

        # The official selected fit retains its failure semantics. Optional
        # display fits cannot replace it and never use external-test values.
        with threadpool_limits(limits=1):
            _, selected_trace = refit_record(selected)
        final_trace.update(selected_trace)
        selected["outer_training_refit"] = selected_trace
        if selected_trace["status"] != "accepted":
            raise ValueError(selected_trace["failure_reason"])
        optional = [record for record in union.values() if record is not selected]
        refit_workers = max(1, min(workers, len(optional)))
        if refit_workers == 1:
            with threadpool_limits(limits=1):
                refitted = [refit_record(record) for record in optional]
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                refitted = Parallel(n_jobs=refit_workers)(delayed(refit_record)(record) for record in optional)
        for candidate_id, trace in refitted:
            union[candidate_id]["outer_training_refit"] = trace
        for record in union.values():
            trace = record["outer_training_refit"]
            record["equation_scaled"] = trace.get("equation")
            record["equation_scaled_precision"] = "full_precision_sympy_float"
            record["equivalent_candidate_ids"] = []
        # Symbolic normalization is display-only: it never changes the fitted
        # parameter templates or turns shared coefficients into free ones.
        representatives: dict[str, dict[str, Any]] = {}
        canonical: dict[str, dict[str, Any]] = {}
        priority = [selected, best] + sorted(
            union.values(), key=lambda record: (-record['cv_raw_r2_mean'], *simplicity(record))
        )
        for record in priority:
            candidate_id = record["candidate_id"]
            if candidate_id in representatives or "equivalent_to_candidate_id" in record:
                continue
            trace = record["outer_training_refit"]
            key = None
            if trace["status"] == "accepted":
                expression = parse_source_expression(trace["equation"], names)
                expression = expression.xreplace({value: sp.Rational(value) for value in expression.atoms(sp.Float)})
                key = sp.srepr(sp.cancel(sp.expand(expression)))
            if key is not None and key in canonical:
                representative = canonical[key]
                representative["equivalent_candidate_ids"].append(candidate_id)
                record["equivalent_to_candidate_id"] = representative["candidate_id"]
            else:
                representatives[candidate_id] = record
                if key is not None:
                    canonical[key] = record
        for layer, records in layers.items():
            ids = []
            for record in records:
                candidate_id = record.get("equivalent_to_candidate_id", record["candidate_id"])
                if candidate_id not in ids:
                    ids.append(candidate_id)
                record[f"selected_in_{layer}_layer"] = candidate_id == record["candidate_id"]
                representatives[candidate_id][f"selected_in_{layer}_layer"] = True
            audit["candidate_layers"][layer]["candidate_ids"] = ids
        audit["shortlisted_candidates"] = list(representatives.values())
        expression = parse_source_expression(final_trace["equation"], names)
        budget = final_trace["term_budget"]
        loss = final_trace["training_mse"]
        origin = "+".join(selected["source_candidate_origins"])
        selected_candidate = {
            "candidate_id": selected["candidate_id"],
            "candidate_position": 0,
            "complexity": selected["structural_complexity"],
            "structural_complexity": selected["structural_complexity"],
            "excess_complexity": budget["excess_complexity"],
            "has_direct_quadratic": budget["has_direct_quadratic"],
            "term_budget": budget, "loss": loss, "candidate_origin": origin,
            "source_cv_folds": selected["source_cv_folds"],
            "source_candidate_positions": selected["source_candidate_positions"],
            "source_candidate_origins": selected["source_candidate_origins"],
            "selection_rule": rule, "excess_complexity_ceiling": int(excess_complexity_ceiling),
            "excess_complexity_ceiling_relaxed": False,
            "cv_raw_r2_mean": selected["cv_raw_r2_mean"],
            "cv_raw_r2_std": selected["cv_raw_r2_std"],
            "cv_raw_r2_standard_error": selected["cv_raw_r2_standard_error"],
            "total_term_count": selected["total_term_count"],
            "independent_parameter_count": selected["independent_parameter_count"],
        }
        final_trace.update(status="accepted", equation=str(expression), training_mse=loss, term_budget=budget)
        audit.update(status="accepted", selected_candidate=selected_candidate, selected_equation_scaled=str(expression))
        return finish([{
            "complexity": selected["structural_complexity"], "loss": loss, "score": 0.0,
            "equation": str(expression).replace("**", "^"), "sympy_format": expression,
            "lambda_format": partial(evaluate_expression, expression, feature_names=names),
            "candidate_origin": origin, "source_cv_folds": selected["source_cv_folds"],
            "refit_source_equation": selected["source_equation"],
            "source_candidate_positions": selected["source_candidate_positions"],
            "source_candidate_origins": selected["source_candidate_origins"],
        }])
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        final_trace.update(status="failed", failure_reason=f"{type(exc).__name__}:{exc}")
        audit["failure_reason"] = "selected_template_full_training_refit_failed"
        return finish([])


def stratified_duplex_indices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    n_strata: int = 4,
    metric: str = "mahalanobis",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, disjoint stratified-DUPLEX outer split."""
    X_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if X_values.ndim != 2 or len(X_values) != len(y_values):
        raise ValueError("X must be two-dimensional and aligned with y")
    if len(y_values) < 2:
        raise ValueError("at least two rows are required for an outer split")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be strictly between zero and one")
    if int(n_strata) < 2:
        raise ValueError("n_strata must be at least two")
    if metric not in {"mahalanobis", "euclidean"}:
        raise ValueError("metric must be 'mahalanobis' or 'euclidean'")

    try:
        y_bins = np.asarray(
            pd.qcut(y_values, q=int(n_strata), duplicates="drop", labels=False)
        )
    except ValueError:
        fallback_strata = min(3, max(2, len(y_values)))
        y_bins = np.asarray(
            pd.qcut(
                y_values,
                q=fallback_strata,
                duplicates="drop",
                labels=False,
            )
        )
    if pd.isna(y_bins).any():
        y_bins = np.zeros(len(y_values), dtype=int)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for stratum in np.unique(y_bins):
        stratum_indices = np.flatnonzero(y_bins == stratum)
        row_count = len(stratum_indices)
        if row_count < 2:
            train_indices.extend(int(index) for index in stratum_indices)
            continue

        test_count = min(
            row_count - 1,
            max(1, int(row_count * float(test_size))),
        )
        train_count = row_count - test_count
        X_stratum = X_values[stratum_indices]
        X_scaled = StandardScaler().fit_transform(X_stratum)

        if metric == "mahalanobis" and row_count > X_stratum.shape[1]:
            try:
                covariance = MinCovDet(
                    support_fraction=1,
                    random_state=int(random_state),
                ).fit(X_scaled).covariance_
                distances = cdist(
                    X_scaled,
                    X_scaled,
                    "mahalanobis",
                    VI=np.linalg.pinv(covariance),
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                distances = cdist(X_scaled, X_scaled, "euclidean")
        else:
            distances = cdist(X_scaled, X_scaled, "euclidean")
        if not np.isfinite(distances).all():
            distances = cdist(X_scaled, X_scaled, "euclidean")

        pair_distances = np.asarray(distances, dtype=float).copy()
        np.fill_diagonal(pair_distances, -np.inf)
        first_train, first_test = np.unravel_index(
            int(np.argmax(pair_distances)), pair_distances.shape
        )
        local_train = [int(first_train)]
        local_test = [int(first_test)]
        remaining = sorted(
            set(range(row_count)).difference(local_train).difference(local_test)
        )

        while remaining and (
            len(local_train) < train_count or len(local_test) < test_count
        ):
            if len(local_train) < train_count and remaining:
                min_distances = np.min(
                    distances[np.ix_(remaining, local_train)], axis=1
                )
                chosen = remaining[int(np.argmax(min_distances))]
                local_train.append(chosen)
                remaining.remove(chosen)
            if len(local_test) < test_count and remaining:
                min_distances = np.min(
                    distances[np.ix_(remaining, local_test)], axis=1
                )
                chosen = remaining[int(np.argmax(min_distances))]
                local_test.append(chosen)
                remaining.remove(chosen)

        local_train.extend(remaining)
        train_indices.extend(int(index) for index in stratum_indices[local_train])
        test_indices.extend(int(index) for index in stratum_indices[local_test])

    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    expected = np.arange(len(y_values), dtype=int)
    if np.intersect1d(train, test).size:
        raise RuntimeError("outer split produced overlapping train and test rows")
    if not np.array_equal(np.sort(np.concatenate([train, test])), expected):
        raise RuntimeError("outer split did not assign every row exactly once")
    return train, test


def build_split_assignments(
    names: Sequence[Any],
    source_row_indices: Sequence[int],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Build a stable Name/source-row assignment table for one outer split."""
    name_values = [str(value).strip() for value in names]
    source_rows = np.asarray(source_row_indices, dtype=int)
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    row_count = len(name_values)
    if len(source_rows) != row_count:
        raise ValueError("Name and source-row columns must have equal length")
    if any(not value for value in name_values) or len(set(name_values)) != row_count:
        raise ValueError("Name values must be nonblank and unique")
    if len(set(source_rows.tolist())) != row_count:
        raise ValueError("source_row_index values must be unique")
    if np.intersect1d(train, test).size:
        raise ValueError("train and test indices must be disjoint")
    assigned = np.sort(np.concatenate([train, test]))
    if not np.array_equal(assigned, np.arange(row_count, dtype=int)):
        raise ValueError("train and test indices must assign every row exactly once")

    split_by_position = {int(index): "Train" for index in train}
    split_by_position.update({int(index): "Test" for index in test})
    records = [
        {
            "source_row_index": int(source_rows[position]),
            "Name": name_values[position],
            "Split": split_by_position[position],
        }
        for position in range(row_count)
    ]
    return sorted(records, key=lambda row: int(row["source_row_index"]))


def build_prediction_records(
    names: Sequence[Any],
    source_row_indices: Sequence[int],
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    split: str,
    baseline_predicted: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create traceable prediction rows while preserving the supplied row order."""
    name_values = [str(value).strip() for value in names]
    source_rows = np.asarray(source_row_indices, dtype=int)
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    expected_length = len(name_values)
    if not all(
        len(values) == expected_length
        for values in (source_rows, actual_values, predicted_values)
    ):
        raise ValueError("prediction columns must have identical lengths")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted values must be finite")
    baseline_values = None
    if baseline_predicted is not None:
        baseline_values = np.asarray(baseline_predicted, dtype=float).reshape(-1)
        if len(baseline_values) != expected_length or not np.isfinite(
            baseline_values
        ).all():
            raise ValueError("baseline predictions must be finite and row-aligned")

    records: list[dict[str, Any]] = []
    for position in range(expected_length):
        record = {
            "source_row_index": int(source_rows[position]),
            "Name": name_values[position],
            "Split": str(split),
            "Actual": float(actual_values[position]),
            "Predicted": float(predicted_values[position]),
            "Error": float(actual_values[position] - predicted_values[position]),
        }
        if baseline_values is not None:
            record["Ridge_Predicted"] = float(baseline_values[position])
            record["Ridge_Error"] = float(
                actual_values[position] - baseline_values[position]
            )
        records.append(record)
    return records


def prepare_fold_scaled_data(
    X_outer_train_raw: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit a scaler on one fold's training rows and transform both partitions."""
    X_raw = np.asarray(X_outer_train_raw, dtype=float)
    train_idx = np.asarray(train_indices, dtype=int)
    validation_idx = np.asarray(validation_indices, dtype=int)
    if np.intersect1d(train_idx, validation_idx).size:
        raise ValueError("fold training and validation indices must be disjoint")
    if train_idx.size == 0 or validation_idx.size == 0:
        raise ValueError("fold training and validation partitions must be non-empty")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_validation = scaler.transform(X_raw[validation_idx])
    return scaler, X_train, X_validation


def evaluate_pareto_candidates(
    model: Any,
    pareto_rows: Sequence[Mapping[str, Any]],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: Sequence[str],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
    required_feature_count: int | None = None,
    excess_complexity_ceiling: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every Pareto row at its positional PySR candidate index.

    Candidates are never allowed to fall back to ``model.predict(X)``.  A
    Missing required symbols, an exception, a shape mismatch, or any non-finite
    prediction makes that candidate ineligible.  By default every input feature
    must remain after simplification; ``required_feature_count`` instead allows
    any subset with exactly that many known features.
    """
    X_values = np.asarray(X_eval, dtype=float)
    y_values = np.asarray(y_eval, dtype=float).reshape(-1)
    required = set(feature_names)
    feature_count = (
        len(required) if required_feature_count is None else required_feature_count
    )
    if not isinstance(feature_count, (int, np.integer)) or not (
        1 <= feature_count <= len(required)
    ):
        raise ValueError("required_feature_count must be between one and the input feature count")
    if excess_complexity_ceiling is not None and (
        not isinstance(excess_complexity_ceiling, (int, np.integer))
        or excess_complexity_ceiling < 0
    ):
        raise ValueError("excess_complexity_ceiling must be a nonnegative integer or None")
    audits: list[dict[str, Any]] = []
    bootstrap_indices: list[np.ndarray] = []
    if int(bootstrap_samples) > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        for _ in range(int(bootstrap_samples)):
            indices = rng.integers(0, len(y_values), size=len(y_values))
            sampled_y = y_values[indices]
            if np.sum((sampled_y - sampled_y.mean()) ** 2) > np.finfo(float).eps:
                bootstrap_indices.append(indices)

    for source in pareto_rows:
        row = dict(source)
        position = int(row["candidate_position"])
        equation = str(row.get("equation", ""))
        try:
            training_loss = float(row.get("loss"))
            if not math.isfinite(training_loss):
                training_loss = None
        except (TypeError, ValueError, OverflowError):
            training_loss = None
        try:
            parsed = sp.sympify(
                equation,
                locals={name: sp.Symbol(name) for name in feature_names},
            )
            unknown_symbols = sorted(
                str(symbol)
                for symbol in parsed.free_symbols
                if str(symbol) not in required
            )
        except Exception:
            parsed = None
            unknown_symbols = ["<parse_failure>"]
        variables = used_features(parsed, feature_names) if parsed is not None else []
        term_budget = analyze_term_budget(
            parsed if parsed is not None else equation, feature_names
        )
        audit = {
            "candidate_position": position,
            "candidate_frame_label": str(row.get("candidate_frame_label", "")),
            "equation": equation,
            "backend_equation": str(row.get("backend_equation", "")),
            "candidate_origin": row.get("candidate_origin", "evolution"),
            "source_cv_folds": row.get("source_cv_folds", []),
            "loss": training_loss,
            "pysr_score": row.get("pysr_score"),
            # ``complexity`` is PySR's raw structural node count.  It remains
            # in every audit, but validation selection uses the explicit
            # per-feature term budget below.
            "complexity": int(row.get("complexity", 10**9)),
            "structural_complexity": int(row.get("complexity", 10**9)),
            "used_variables": variables,
            "unknown_symbols": unknown_symbols,
            "all_features_used": set(variables) == required,
            "required_feature_count": int(feature_count),
            "feature_count_valid": len(variables) == feature_count,
            "excess_complexity_ceiling": (
                int(excess_complexity_ceiling) if excess_complexity_ceiling is not None else None
            ),
            "term_budget": term_budget,
            "excess_complexity": (
                None
                if term_budget["analysis_error"] is not None
                else int(term_budget["excess_complexity"])
            ),
            "has_direct_quadratic": (
                None
                if term_budget["analysis_error"] is not None
                else bool(term_budget["has_direct_quadratic"])
            ),
            "eligible": False,
            "failure_reason": None,
            "raw_r2": None,
            "raw_r2_standard_error": None,
            "valid_bootstrap_samples": len(bootstrap_indices),
            "rmse": None,
            "mae": None,
        }
        if unknown_symbols:
            audit["failure_reason"] = "unknown_or_unparseable_symbol"
            audits.append(audit)
            continue
        if not audit["feature_count_valid"]:
            audit["failure_reason"] = (
                "missing_required_features"
                if required_feature_count is None
                else "incorrect_feature_count"
            )
            audits.append(audit)
            continue
        if audit["loss"] is None:
            audit["failure_reason"] = "nonfinite_training_loss"
            audits.append(audit)
            continue

        if term_budget["analysis_error"] is not None:
            audit["failure_reason"] = "term_budget_analysis_failed"
            audits.append(audit)
            continue
        if (
            excess_complexity_ceiling is not None
            and audit["excess_complexity"] > excess_complexity_ceiling
        ):
            audit["failure_reason"] = "excess_complexity_ceiling_exceeded"
            audits.append(audit)
            continue

        try:
            raw_prediction = np.asarray(model.predict(X_values, index=position))
            if np.iscomplexobj(raw_prediction):
                if np.max(np.abs(np.imag(raw_prediction))) > 1e-12:
                    audit["failure_reason"] = "complex_prediction"
                    audits.append(audit)
                    continue
                raw_prediction = np.real(raw_prediction)
            prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
        except Exception as exc:  # PySR/domain failures must remain auditable.
            audit["failure_reason"] = f"prediction_error:{type(exc).__name__}:{exc}"
            audits.append(audit)
            continue
        if prediction.shape != y_values.shape:
            audit["failure_reason"] = (
                f"prediction_shape_mismatch:{prediction.shape}!={y_values.shape}"
            )
            audits.append(audit)
            continue
        if not np.isfinite(prediction).all():
            audit["failure_reason"] = "nonfinite_prediction"
            audits.append(audit)
            continue

        raw_r2 = float(r2_score(y_values, prediction))
        rmse = float(np.sqrt(mean_squared_error(y_values, prediction)))
        mae = float(mean_absolute_error(y_values, prediction))
        if not all(math.isfinite(value) for value in (raw_r2, rmse, mae)):
            audit["failure_reason"] = "nonfinite_metric"
            audits.append(audit)
            continue

        bootstrap_r2 = [
            float(r2_score(y_values[indices], prediction[indices]))
            for indices in bootstrap_indices
        ]
        bootstrap_r2 = [value for value in bootstrap_r2 if math.isfinite(value)]
        raw_r2_standard_error = (
            float(np.std(bootstrap_r2, ddof=1)) if len(bootstrap_r2) > 1 else 0.0
        )

        audit.update(
            {
                "eligible": True,
                "raw_r2": raw_r2,
                "raw_r2_standard_error": raw_r2_standard_error,
                "valid_bootstrap_samples": len(bootstrap_r2),
                "rmse": rmse,
                "mae": mae,
            }
        )
        audits.append(audit)
    return audits


def select_validation_candidate(
    candidate_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a paired-bootstrap one-standard-error candidate rule."""
    eligible = [dict(row) for row in candidate_audits if row.get("eligible")]
    if not eligible:
        failures = sorted(
            {
                str(row.get("failure_reason") or "unknown")
                for row in candidate_audits
            }
        )
        raise ValueError(
            "no finite all-feature Pareto candidate on validation data; "
            f"failure reasons: {failures}"
        )
    best_point = min(
        eligible,
        key=lambda row: (
            -float(row["raw_r2"]),
            int(row["excess_complexity"]),
            float(row["loss"]),
            int(row["structural_complexity"]),
            int(row["candidate_position"]),
        ),
    )
    threshold = float(best_point["raw_r2"]) - float(
        best_point.get("raw_r2_standard_error") or 0.0
    )
    within_one_se = [row for row in eligible if float(row["raw_r2"]) >= threshold]
    selected = min(
        within_one_se,
        key=lambda row: (
            int(row["excess_complexity"]),
            -float(row["raw_r2"]),
            float(row.get("raw_r2_standard_error") or 0.0),
            float(row["loss"]),
            int(row["structural_complexity"]),
            int(row["candidate_position"]),
        ),
    )
    selected["selection_rule"] = (
        "paired_bootstrap_one_standard_error_then_excess_complexity"
    )
    selected["best_point_candidate_position"] = int(
        best_point["candidate_position"]
    )
    selected["one_se_threshold"] = float(threshold)
    return selected


def select_training_candidate(
    candidate_audits: Sequence[Mapping[str, Any]],
    *,
    excess_complexity_ceiling: int | None = None,
) -> dict[str, Any]:
    """Select a refinement candidate under a CV-derived excess-term budget."""
    eligible = [dict(row) for row in candidate_audits if row.get("eligible")]
    if not eligible:
        failures = sorted(
            {
                str(row.get("failure_reason") or "unknown")
                for row in candidate_audits
            }
        )
        raise ValueError(
            "no finite all-feature Pareto candidate on outer-training data; "
            f"failure reasons: {failures}"
        )
    within_budget = (
        [
            row
            for row in eligible
            if int(row["excess_complexity"]) <= excess_complexity_ceiling
        ]
        if excess_complexity_ceiling is not None
        else eligible
    )
    if not within_budget:
        raise ValueError(
            "no eligible Pareto candidate satisfies the CV-derived excess "
            f"complexity ceiling of {excess_complexity_ceiling}"
        )
    selected = min(
        within_budget,
        key=lambda row: (
            float(row["loss"]),
            int(row["excess_complexity"]),
            int(row["structural_complexity"]),
            int(row["candidate_position"]),
        ),
    )
    selected["selection_rule"] = "cv_excess_budget_then_training_loss"
    selected["excess_complexity_ceiling"] = excess_complexity_ceiling
    selected["excess_complexity_ceiling_relaxed"] = False
    return selected


def select_config_one_standard_error(
    completed_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the lowest-excess stable configuration within one SE of best R2."""
    if not completed_results:
        raise ValueError("at least one completed configuration is required")
    results = [dict(result) for result in completed_results]
    highest_mean = min(
        results,
        key=lambda result: (
            -float(result["cv_raw_r2_mean"]),
            float(result["cv_raw_r2_std"]),
            str(result["config_id"]),
        ),
    )
    fold_results = list(highest_mean.get("fold_results", []))
    repeat_groups: dict[int, list[float]] = {}
    for fold in fold_results:
        repeat_groups.setdefault(int(fold.get("repeat", 1)), []).append(
            float(fold["raw_r2"])
        )
    repeat_means = [statistics.fmean(values) for values in repeat_groups.values()]
    if len(repeat_means) > 1:
        best_se = statistics.stdev(repeat_means) / math.sqrt(len(repeat_means))
        se_method = "repeat_mean_standard_error"
    else:
        fold_scores = [float(fold["raw_r2"]) for fold in fold_results]
        best_se = (
            statistics.stdev(fold_scores) / math.sqrt(len(fold_scores))
            if len(fold_scores) > 1
            else 0.0
        )
        se_method = "fold_standard_error"
    threshold = float(highest_mean["cv_raw_r2_mean"]) - best_se
    within_one_se = [
        result
        for result in results
        if float(result["cv_raw_r2_mean"]) >= threshold
    ]
    selected = min(
        within_one_se,
        key=lambda result: (
            float(result["cv_excess_complexity_mean"]),
            float(result["cv_raw_r2_std"]),
            -float(result["cv_raw_r2_mean"]),
            float(result["cv_complexity_mean"]),
            str(result["config_id"]),
        ),
    )
    details = {
        "rule": "one_standard_error_then_excess_complexity_and_stability",
        "highest_mean_config_id": str(highest_mean["config_id"]),
        "highest_mean_cv_r2": float(highest_mean["cv_raw_r2_mean"]),
        "highest_mean_standard_error": float(best_se),
        "standard_error_method": se_method,
        "eligibility_threshold": float(threshold),
        "eligible_config_ids": [
            str(result["config_id"]) for result in within_one_se
        ],
        "selected_config_id": str(selected["config_id"]),
    }
    return selected, details
