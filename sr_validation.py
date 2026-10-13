"""Validation helpers for the symbolic-regression workflow.

This module deliberately does not import PySR or Julia.  The small functions
can therefore be tested with fake models while the production pipeline still
uses PySR's positional candidate API.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sympy as sp
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sr_efficiency import used_features


def analyze_term_budget(
    equation: str,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Audit the explicit per-feature linear/nonlinear term allowance.

    Numeric coefficients and an intercept are free.  For every known feature,
    one linear term and one *direct* nonlinear term are also free.  With the
    default SR operator profile the direct nonlinear term is ``x**2``.  A
    single-argument SymPy function applied directly to one feature is treated
    the same way so the policy remains well-defined if a future, explicitly
    enabled operator profile adds a unary transform.

    Interactions, powers above two, repeated nonlinear transforms, and
    composite/nested transforms consume the auditable ``excess_complexity``
    budget.  The metric is intentionally separate from PySR's raw node count:
    raw node complexity remains useful for a hard resource cap, but it must not
    suppress a useful direct quadratic term during validation selection.
    """
    feature_names = [str(name) for name in feature_names]
    symbols = {name: sp.Symbol(name) for name in feature_names}
    result: dict[str, Any] = {
        "policy": "one_linear_and_one_direct_nonlinear_per_feature_free",
        "current_default_nonlinear_operator": "square",
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
    }
    try:
        expression = sp.expand(
            sp.sympify(
                str(equation),
                locals={
                    **symbols,
                    "square": lambda value: value**2,
                },
            )
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
        # Direct square: x**2.  Higher powers remain over budget even though
        # SymPy has simplified away whether they originated from nested square.
        if isinstance(core, sp.Pow) and core.base in symbols.values():
            if core.exp == 2:
                return str(core.base), "square"
            return None, None

        # Future unary profiles: exp(x), log(x), sqrt(x), ... are one direct
        # nonlinear term.  exp(x)**2 or exp(x**2) do not match and are
        # therefore treated as nested/composite below.
        if isinstance(core, sp.Function) and len(core.args) == 1:
            argument = core.args[0]
            if argument in symbols.values():
                return str(argument), str(core.func)
        return None, None

    for term in sp.Add.make_args(expression):
        _, core = term.as_coeff_Mul()
        term_symbols = sorted(
            (str(symbol) for symbol in core.free_symbols),
            key=feature_names.index,
        )
        if not term_symbols:
            continue

        if core in symbols.values():
            feature = str(core)
            if feature not in linear_seen:
                linear_seen.add(feature)
                continue

        nonlinear_feature, nonlinear_kind = _direct_nonlinear_feature(core)
        if nonlinear_feature is not None and nonlinear_feature not in nonlinear_seen:
            nonlinear_seen.add(nonlinear_feature)
            if nonlinear_kind == "square":
                quadratic_seen.add(nonlinear_feature)
            continue

        active_feature_count = len(set(term_symbols))
        operation_count = max(1, int(sp.count_ops(core, visual=False)))
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
            "excess_complexity": int(
                sum(row["penalty"] for row in penalized_terms)
            ),
        }
    )
    return result


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
) -> list[dict[str, Any]]:
    """Evaluate every Pareto row at its positional PySR candidate index.

    Candidates are never allowed to fall back to ``model.predict(X)``.  A
    missing required symbol, an exception, a shape mismatch, or any non-finite
    prediction makes that candidate ineligible while leaving the other Pareto
    candidates available for selection.
    """
    X_values = np.asarray(X_eval, dtype=float)
    y_values = np.asarray(y_eval, dtype=float).reshape(-1)
    required = set(feature_names)
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
        variables = used_features(equation, feature_names)
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
            unknown_symbols = ["<parse_failure>"]
        term_budget = analyze_term_budget(equation, feature_names)
        audit = {
            "candidate_position": position,
            "candidate_frame_label": str(row.get("candidate_frame_label", "")),
            "equation": equation,
            "backend_equation": str(row.get("backend_equation", "")),
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
        if not audit["all_features_used"]:
            audit["failure_reason"] = "missing_required_features"
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
    relaxed = not within_budget
    candidates = within_budget or eligible
    if relaxed:
        ordering = lambda row: (
            int(row["excess_complexity"]),
            float(row["loss"]),
            int(row["structural_complexity"]),
            int(row["candidate_position"]),
        )
    else:
        ordering = lambda row: (
            float(row["loss"]),
            int(row["excess_complexity"]),
            int(row["structural_complexity"]),
            int(row["candidate_position"]),
        )
    selected = min(candidates, key=ordering)
    selected["selection_rule"] = "cv_excess_budget_then_training_loss"
    selected["excess_complexity_ceiling"] = excess_complexity_ceiling
    selected["excess_complexity_ceiling_relaxed"] = bool(relaxed)
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
