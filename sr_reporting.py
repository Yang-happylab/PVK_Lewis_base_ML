"""Pure-Python reporting helpers for symbolic-regression equations.

This module deliberately does not import PySR or Julia.  It handles exact
standardized-to-raw substitution, numerical equation checks, and bounded
fixed-structure coefficient bootstrap analysis.
"""

from __future__ import annotations

from collections import Counter
from time import monotonic
from typing import Sequence

import numpy as np
import sympy as sp
from scipy.optimize import least_squares


def standardized_to_raw_expression(
    expression,
    feature_names: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
    *,
    preserved_symbol_names: Sequence[str] = (),
):
    """Substitute ``z=(x-mean)/scale`` without changing feature names."""
    expr = sp.sympify(expression)
    if not (len(feature_names) == len(means) == len(scales)):
        raise ValueError("feature_names, means, and scales must have equal length")

    feature_symbols = [sp.Symbol(name) for name in feature_names]
    preserved_symbols = {sp.Symbol(name) for name in preserved_symbol_names}
    unknown = expr.free_symbols - set(feature_symbols) - preserved_symbols
    if unknown:
        raise ValueError(f"Equation contains unknown symbols: {sorted(map(str, unknown))}")

    temporary_symbols = [sp.Dummy(f"{name}_raw") for name in feature_names]
    substitutions = {}
    for symbol, temporary, mean, scale in zip(
        feature_symbols, temporary_symbols, means, scales
    ):
        mean = float(mean)
        scale = float(scale)
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scaler parameters for {symbol}: mean={mean}, scale={scale}")
        substitutions[symbol] = (temporary - mean) / scale

    raw_expr = expr.xreplace(substitutions)
    raw_expr = raw_expr.xreplace(
        {
            temporary: original
            for temporary, original in zip(temporary_symbols, feature_symbols)
        }
    )
    return sp.simplify(raw_expr)


def parameterize_floating_constants(expression, prefix: str = "c"):
    """Replace each fitted floating-constant occurrence with a stable symbol.

    Integer atoms are intentionally retained because values such as the ``2``
    in ``square(x) -> x**2`` describe formula structure rather than a fitted
    PySR constant.  Equal-valued constants at different tree positions remain
    independent parameters; for example, a slope of 0.5 and an intercept of
    0.5 are not tied together during bootstrap refitting.
    """
    expr = sp.sympify(expression)
    parameters = []
    estimates = []

    def replace_occurrences(node):
        if isinstance(node, sp.Float):
            parameter = sp.Symbol(f"{prefix}{len(parameters)}")
            parameters.append(parameter)
            estimates.append(float(node))
            return parameter
        if not node.args:
            return node
        replaced_args = tuple(replace_occurrences(argument) for argument in node.args)
        try:
            return node.func(*replaced_args, evaluate=False)
        except TypeError:
            return node.func(*replaced_args)

    parameterized = replace_occurrences(expr)
    return parameterized, parameters, estimates


def evaluate_expression(
    expression,
    X,
    feature_names: Sequence[str],
    *,
    parameter_names: Sequence[str] = (),
    parameter_values: Sequence[float] = (),
):
    """Evaluate a SymPy equation and reject complex or non-finite predictions."""
    expr = sp.sympify(expression)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected X with {len(feature_names)} columns, received shape {X.shape}"
        )
    if len(parameter_names) != len(parameter_values):
        raise ValueError("parameter_names and parameter_values must have equal length")

    feature_symbols = [sp.Symbol(name) for name in feature_names]
    parameter_symbols = [sp.Symbol(name) for name in parameter_names]
    expected_symbols = set(feature_symbols + parameter_symbols)
    unknown = expr.free_symbols - expected_symbols
    if unknown:
        raise ValueError(f"Equation contains unknown symbols: {sorted(map(str, unknown))}")

    function = sp.lambdify(feature_symbols + parameter_symbols, expr, modules="numpy")
    arguments = [X[:, index] for index in range(X.shape[1])]
    arguments.extend(float(value) for value in parameter_values)
    with np.errstate(all="ignore"):
        values = np.asarray(function(*arguments))

    if values.ndim == 0:
        values = np.full(X.shape[0], values.item())
    values = values.reshape(-1)
    if values.size != X.shape[0]:
        raise ValueError(
            f"Equation returned {values.size} predictions for {X.shape[0]} rows"
        )
    if np.iscomplexobj(values):
        if np.any(np.abs(np.imag(values)) > 1e-12):
            raise ValueError("Equation produced complex predictions")
        values = np.real(values)
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Equation produced NaN or infinite predictions")
    return values


def bootstrap_fixed_structure_coefficients(
    expression,
    X,
    y,
    feature_names: Sequence[str],
    *,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 30042,
    max_nfev: int = 300,
    max_wall_seconds: float = 300.0,
):
    """Estimate conditional coefficient uncertainty for a fixed formula tree.

    Rows of the outer training set are sampled with replacement.  The selected
    symbolic structure and the outer-training scaler remain fixed; only PySR's
    floating constants are re-optimized.  This therefore excludes formula
    selection, scaler, Score-calibration, and external-test uncertainty.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y have incompatible shapes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if max_nfev < 1 or max_wall_seconds <= 0:
        raise ValueError("max_nfev and max_wall_seconds must be positive")

    parameterized, parameters, estimates = parameterize_floating_constants(expression)
    parameter_names = [str(parameter) for parameter in parameters]
    started = monotonic()
    base = {
        "schema_version": 1,
        "method": "nonparametric outer-training row bootstrap with fixed symbolic structure",
        "parameterized_equation_scaled": str(parameterized),
        "parameter_names": parameter_names,
        "original_estimates": [float(value) for value in estimates],
        "settings": {
            "requested_bootstrap_samples": int(n_bootstrap),
            "confidence_level": float(confidence_level),
            "random_state": int(random_state),
            "max_nfev_per_sample": int(max_nfev),
            "max_wall_seconds": float(max_wall_seconds),
            "scaler_refitted_per_bootstrap": False,
            "formula_structure_refitted_per_bootstrap": False,
            "external_test_used": False,
        },
        "limitations": [
            "Conditional on the selected symbolic structure.",
            "Conditional on the fixed outer-training StandardScaler.",
            "Does not include Score-calibration or measurement uncertainty.",
            "Does not include formula-selection or external-test uncertainty.",
        ],
    }
    if not parameters:
        base.update(
            {
                "status": "no_fitted_floating_constants",
                "attempted_samples": 0,
                "successful_samples": 0,
                "failure_counts": {},
                "coefficient_rows": [],
                "sample_rows": [],
                "elapsed_seconds": monotonic() - started,
            }
        )
        return base

    initial = np.asarray(estimates, dtype=float)
    rng = np.random.default_rng(random_state)
    successful_values = []
    sample_rows = []
    failures = Counter()

    def residuals(theta, X_sample, y_sample):
        try:
            prediction = evaluate_expression(
                parameterized,
                X_sample,
                feature_names,
                parameter_names=parameter_names,
                parameter_values=theta,
            )
            return prediction - y_sample
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return np.full(y_sample.shape, 1e12, dtype=float)

    attempted = 0
    for bootstrap_index in range(n_bootstrap):
        if monotonic() - started >= max_wall_seconds:
            failures["wall_time_limit"] += n_bootstrap - attempted
            break
        attempted += 1
        indices = rng.integers(0, y.size, size=y.size)
        X_sample = X[indices]
        y_sample = y[indices]
        record = {
            "bootstrap_index": int(bootstrap_index),
            "status": "failed",
            "failure_reason": None,
            "nfev": None,
            "bootstrap_rmse": None,
        }
        try:
            fit = least_squares(
                residuals,
                initial,
                args=(X_sample, y_sample),
                max_nfev=max_nfev,
                method="trf",
            )
            fitted = np.asarray(fit.x, dtype=float)
            prediction = evaluate_expression(
                parameterized,
                X_sample,
                feature_names,
                parameter_names=parameter_names,
                parameter_values=fitted,
            )
            if not fit.success or not np.all(np.isfinite(fitted)):
                reason = "optimizer_not_converged"
                failures[reason] += 1
                record["failure_reason"] = reason
            else:
                record.update(
                    {
                        "status": "ok",
                        "nfev": int(fit.nfev),
                        "bootstrap_rmse": float(
                            np.sqrt(np.mean((prediction - y_sample) ** 2))
                        ),
                    }
                )
                record.update(
                    {name: float(value) for name, value in zip(parameter_names, fitted)}
                )
                successful_values.append(fitted)
        except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            reason = type(exc).__name__
            failures[reason] += 1
            record["failure_reason"] = reason
        sample_rows.append(record)

    alpha = 1.0 - confidence_level
    coefficient_rows = []
    if successful_values:
        samples = np.asarray(successful_values, dtype=float)
        for index, (name, estimate) in enumerate(zip(parameter_names, initial)):
            values = samples[:, index]
            standard_error = float(np.std(values, ddof=1)) if len(values) > 1 else None
            coefficient_rows.append(
                {
                    "parameter": name,
                    "estimate": float(estimate),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_median": float(np.median(values)),
                    "bootstrap_standard_error": standard_error,
                    "ci_low": float(np.quantile(values, alpha / 2.0)),
                    "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
                    "confidence_level": float(confidence_level),
                    "successful_samples": int(len(values)),
                }
            )

    successful_count = len(successful_values)
    if successful_count == n_bootstrap:
        status = "ok"
    elif successful_count >= max(10, int(np.ceil(0.5 * n_bootstrap))):
        status = "partial"
    else:
        status = "insufficient_successful_samples"
    base.update(
        {
            "status": status,
            "attempted_samples": int(attempted),
            "successful_samples": int(successful_count),
            "failure_counts": dict(sorted(failures.items())),
            "coefficient_rows": coefficient_rows,
            "sample_rows": sample_rows,
            "elapsed_seconds": monotonic() - started,
        }
    )
    return base
