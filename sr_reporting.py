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


def parse_source_expression(expression, feature_names: Sequence[str] = ()):
    """Parse a source tree without distributing or merging fitted constants."""
    return sp.sympify(
        expression,
        locals={
            **{str(name): sp.Symbol(str(name)) for name in feature_names},
            "square": lambda value, **_: sp.Pow(value, 2, evaluate=False),
            "sqrt": lambda value, **_: sp.Pow(value, sp.Rational(1, 2), evaluate=False),
            "exp": lambda value, **_: sp.exp(value, evaluate=False),
        },
        evaluate=False,
    )


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
    expr = parse_source_expression(expression)
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


def neutral_parameter_start(template, parameters, X, feature_names, sign):
    """Make one neutral start feasible for affine true-root arguments on train X."""
    initial = np.full(len(parameters), float(sign))
    features = [sp.Symbol(str(name)) for name in feature_names]
    roots = [node.base for node in sp.preorder_traversal(template)
             if isinstance(node, sp.Pow) and node.exp == sp.Rational(1, 2)]
    if not roots:
        return initial
    margin = 1e-6
    functions = [sp.lambdify(features + list(parameters), argument, "numpy") for argument in roots]
    for _ in range(2 * len(roots) + 1):
        adjusted = False
        for argument, function in zip(roots, functions):
            values = _evaluate_compiled_expression(function, X, initial)
            minimum = float(np.min(values))
            if minimum >= margin:
                continue
            active = argument.free_symbols.intersection(features)
            try:
                affine = len(active) == 1 and sp.Poly(argument, *active).degree() <= 1
            except sp.PolynomialError:
                affine = False
            if not affine:
                raise ValueError("neutral_start_root_argument_has_no_affine_offset")
            for index, parameter in enumerate(parameters):
                derivative = sp.diff(argument, parameter)
                if derivative.free_symbols.intersection(features) or sp.diff(derivative, parameter) != 0:
                    continue
                slope = derivative.subs(dict(zip(parameters, initial)))
                if not slope.is_real or not slope.is_finite or float(slope) == 0:
                    continue
                initial[index] += (margin - minimum) / float(slope)
                adjusted = True
                break
            else:
                raise ValueError("neutral_start_root_argument_has_no_free_offset")
        if not adjusted:
            return initial
    if any(np.min(_evaluate_compiled_expression(function, X, initial)) < margin / 2
           for function in functions):
        raise ValueError("neutral_start_root_domain_adjustment_failed")
    return initial


def evaluate_expression(
    expression,
    X,
    feature_names: Sequence[str],
    *,
    parameter_names: Sequence[str] = (),
    parameter_values: Sequence[float] = (),
):
    """Evaluate a SymPy equation and reject complex or non-finite predictions."""
    expr = parse_source_expression(expression, feature_names)
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
    return _evaluate_compiled_expression(function, X, parameter_values)


def _evaluate_compiled_expression(function, X, parameter_values):
    """Apply an already-compiled formula with the same prediction checks."""
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
    compilation_error = None
    try:
        feature_symbols = [sp.Symbol(name) for name in feature_names]
        unknown = parameterized.free_symbols - set(feature_symbols + parameters)
        if unknown:
            raise ValueError(f"Equation contains unknown symbols: {sorted(map(str, unknown))}")
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected X with {len(feature_names)} columns, received shape {X.shape}"
            )
        function = sp.lambdify(feature_symbols + parameters, parameterized, modules="numpy")
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        # Keep invalid equations in the existing per-sample failure report.
        compilation_error = exc

    def predict_sample(theta, X_sample):
        if compilation_error is not None:
            raise compilation_error
        return _evaluate_compiled_expression(function, X_sample, theta)

    def residuals(theta, X_sample, y_sample):
        try:
            prediction = predict_sample(theta, X_sample)
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
            prediction = predict_sample(fitted, X_sample)
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


def write_candidate_comparison(audit, directory, feature_names, mean, scale):
    """Export the two conditional-CV display layers without changing selection."""
    import csv
    import json
    from pathlib import Path

    if not audit.get("candidate_layers"):
        return {}
    metric_keys = (
        "cv_raw_r2_mean", "cv_raw_r2_std", "cv_raw_r2_standard_error", "cv_rmse_mean",
        "cv_mae_mean", "repeat_mean_raw_r2", "excess_complexity", "structural_complexity",
        "total_term_count", "linear_term_count", "nonlinear_term_count", "interaction_term_count",
        "independent_parameter_count", "delta_cv_raw_r2", "within_one_se", "within_two_se",
        "source_equation", "source_candidate_positions", "source_candidate_origins", "source_cv_folds",
        "equivalent_candidate_ids", "source_term_budget",
    )

    def finite(value):
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value

    candidates = {}
    for record in audit.get("shortlisted_candidates", []):
        refit = record.get("outer_training_refit", {})
        equation = record.get("equation_scaled") if refit.get("status") == "accepted" else None
        folds = record.get("folds", [])
        item = {key: record.get(key) for key in metric_keys}
        item.update(
            candidate_id=record["candidate_id"], equation_scaled=equation,
            equation_raw=str(standardized_to_raw_expression(equation, feature_names, mean, scale)) if equation else None,
            outer_training_refit_status=refit.get("status", "unavailable"),
            outer_training_refit_failure=refit.get("failure_reason"),
            training_raw_r2=refit.get("training_raw_r2"), training_rmse=refit.get("training_rmse"),
            training_mae=refit.get("training_mae"),
            fold_raw_r2=[fold.get("raw_r2") for fold in folds],
            fold_rmse=[fold.get("rmse") for fold in folds], fold_mae=[fold.get("mae") for fold in folds],
            official_selected=record["candidate_id"] == audit.get("selected_candidate_id"),
            full_audit_reference=f"final_candidate_cv.json#candidate_id={record['candidate_id']}",
        )
        n_splits = int(audit.get("cv_n_splits", len(folds) or 1))
        for metric in ("rmse", "mae"):
            values = item[f"fold_{metric}"]
            item[f"repeat_mean_{metric}"] = [float(np.mean(values[i:i + n_splits]))
                                              for i in range(0, len(values), n_splits)]
        candidates[record["candidate_id"]] = finite(item)
    layers, rows = {}, []
    for name in ("one_se", "two_se"):
        layer = audit["candidate_layers"].get(name, {})
        ids = layer.get("candidate_ids", [])
        if len(ids) > 10:
            raise ValueError("Each candidate comparison layer must contain at most ten candidates")
        layers[name] = layer
        rows.extend({"layer": name, "layer_position": position, **candidates[candidate_id]}
                    for position, candidate_id in enumerate(ids, start=1))
    note = ("Conditional CV refits fixed discovered formula templates; it is not an independent "
            "validation of formula discovery. One-SE defines the official model; two-SE is an "
            "alternative display. No external-test data or metrics are used in this comparison.")
    summary = finite({"interpretation": note, "full_audit": "final_candidate_cv.json",
                      "candidate_layers": layers, "candidates": list(candidates.values())})
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    filenames = {suffix: f"candidate_comparison.{suffix}" for suffix in ("json", "csv", "txt")}
    (output / filenames["json"]).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / filenames["csv"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["layer", "candidate_id"])
        writer.writeheader()
        writer.writerows({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    text = [note]
    for name in layers:
        text.append(f"\n{name} (threshold={layers[name].get('threshold')})")
        for row in (value for value in rows if value["layer"] == name):
            text.extend((f"{row['candidate_id']}: CV R2={row['cv_raw_r2_mean']} SD={row['cv_raw_r2_std']} "
                         f"SE={row['cv_raw_r2_standard_error']}; RMSE={row['cv_rmse_mean']}; MAE={row['cv_mae_mean']}; "
                         f"excess={row['excess_complexity']}; parameters={row['independent_parameter_count']}; "
                         f"refit={row['outer_training_refit_status']}",
                         f"  Training R2={row['training_raw_r2']}; RMSE={row['training_rmse']}; "
                         f"MAE={row['training_mae']}; delta CV R2={row['delta_cv_raw_r2']}",
                         f"  Scaled: {row['equation_scaled']}", f"  Raw: {row['equation_raw']}"))
    (output / filenames["txt"]).write_text("\n".join(text) + "\n", encoding="utf-8")
    return filenames


def export_candidate_results(run_directory, *, dpi=600):
    """Report each fixed display candidate without refitting or changing selection.

    External-test diagnostics are produced only after the training-side shortlist
    and official equation have been saved. Existing model artifacts stay intact.
    """
    import json
    import re
    from pathlib import Path

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from project_paths import file_sha256

    run = Path(run_directory).resolve()
    source_names = ["candidate_comparison.json", "scaler_params.json", "data_encoded.csv",
                    "split_assignments.csv", "selected_candidate.json", "predictions.csv",
                    "resume_contract.json", "run_config.json", "final_candidate_cv.json"]
    source_hashes = {name: file_sha256(run / name) for name in source_names}
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    comparison, scaler, selected = (read(name) for name in source_names[:2] + [source_names[4]])
    contract = read("resume_contract.json")["contract"]
    for name, key in [("data_encoded.csv", "derived_data_sha256"),
                      ("split_assignments.csv", "split_assignments_sha256")]:
        if source_hashes[name] != contract[key]:
            raise ValueError(f"Saved candidate diagnostic input hash mismatch: {name}")
    data = pd.read_csv(run / "data_encoded.csv")
    split = pd.read_csv(run / "split_assignments.csv")
    if (not data.Name.is_unique or not split.Name.is_unique
            or set(data.Name) != set(split.Name)
            or not split.Split.isin(["Train", "Test"]).all()):
        raise ValueError("Candidate diagnostics require unique matching Name/split assignments")
    joined = data.merge(split, on=["Name", "source_row_index"], validate="one_to_one")
    if len(joined) != len(data):
        raise ValueError("Candidate diagnostic source_row_index does not match Name")
    train = joined.Split.eq("Train").to_numpy()
    if min(train.sum(), (~train).sum()) < 2:
        raise ValueError("Candidate diagnostics require at least two rows in each split")
    names = scaler["feature_names"]
    x = joined[names].to_numpy(dtype=float)
    y = joined[read("run_config.json")["target_column"]].to_numpy(dtype=float)
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    if (mean.shape != (len(names),) or scale.shape != mean.shape
            or not np.isfinite([mean, scale]).all() or (scale <= 0).any()
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Candidate diagnostic inputs/scaler must be finite and valid")
    z = (x - mean) / scale
    layers = comparison["candidate_layers"]
    ids = list(dict.fromkeys(cid for layer in layers.values() for cid in layer["candidate_ids"]))
    candidates = {row["candidate_id"]: row for row in comparison["candidates"]}
    if any(not re.fullmatch(r"SR_C\d+", cid) or cid not in candidates for cid in ids):
        raise ValueError("Candidate diagnostic IDs must match the saved shortlist")
    audit = read("final_candidate_cv.json")
    archived = {row["candidate_id"]: row for row in audit["shortlisted_candidates"]}
    if layers != audit["candidate_layers"]:
        raise ValueError("Candidate diagnostic layers differ from the full CV audit")
    for cid in ids:
        record = archived[cid]
        refit = record["outer_training_refit"]
        equation = record.get("equation_scaled") if refit["status"] == "accepted" else None
        raw = str(standardized_to_raw_expression(equation, names, mean, scale)) if equation else None
        if (candidates[cid]["equation_scaled"] != equation or candidates[cid]["equation_raw"] != raw
                or candidates[cid]["outer_training_refit_status"] != refit["status"]
                or any(candidates[cid][key] != record[key] for key in (
                    "cv_raw_r2_mean", "cv_raw_r2_standard_error", "cv_rmse_mean", "cv_mae_mean",
                    "excess_complexity", "total_term_count", "independent_parameter_count"))):
            raise ValueError(f"Candidate diagnostic record differs from the full CV audit: {cid}")
    official = [cid for cid in ids if candidates[cid]["official_selected"]]
    if (len(official) != 1 or sp.simplify(
            parse_source_expression(candidates[official[0]]["equation_scaled"], names)
            - parse_source_expression(selected["equation_scaled"], names)) != 0):
        raise ValueError("Candidate diagnostics do not match the saved official equation")
    saved = pd.read_csv(run / "predictions.csv").set_index("Name")
    if not saved.index.is_unique or set(saved.index) != set(joined.Name):
        raise ValueError("Saved official predictions do not match candidate diagnostic Names")
    destination = run / "candidate_results"
    if destination.exists():
        raise FileExistsError(f"Candidate results already exist: {destination}")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix=".candidate_results-", dir=run) as temporary:
        output = Path(temporary) / "candidate_results"
        output.mkdir()

        def write_json(path, value):
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        reports, predictions = [], {}
        for cid in ids:
            candidate = candidates[cid]
            folder = output / cid
            folder.mkdir()
            report = {**candidate,
                      "layers": [name for name, layer in layers.items() if cid in layer["candidate_ids"]],
                      "prediction_contract": "fixed saved full-precision equation; no fitting",
                      "test_role": "post-selection diagnostic; not used by automatic one-SE selection",
                      "residual_definition": "Actual - Predicted", "test_metrics": None,
                      "train_metrics": None, "status": "refit_unavailable"}
            if candidate["outer_training_refit_status"] == "accepted":
                try:
                    p = evaluate_expression(candidate["equation_raw"], x, names)
                    ps = evaluate_expression(candidate["equation_scaled"], z, names)
                    if not np.allclose(p, ps, rtol=1e-9, atol=1e-8):
                        raise ValueError("Saved raw/scaled candidate predictions differ")
                    if candidate["official_selected"]:
                        reference = saved.loc[joined.Name]
                        if (not np.allclose(p, reference.Predicted, rtol=1e-9, atol=1e-8)
                                or not np.allclose(y, reference.Actual, rtol=0, atol=1e-10)
                                or not np.array_equal(joined.Split, reference.Split)):
                            raise ValueError("Candidate does not reproduce saved official predictions")
                    for label, mask in [("train", train), ("test", ~train)]:
                        report[label + "_metrics"] = {
                            "n": int(mask.sum()), "r2": float(r2_score(y[mask], p[mask])),
                            "rmse": float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
                            "mae": float(mean_absolute_error(y[mask], p[mask])),
                        }
                    if not np.isclose(report["train_metrics"]["r2"], candidate["training_raw_r2"], atol=1e-9, rtol=0):
                        raise ValueError("Candidate does not reproduce its saved training R2")
                    frame = joined[["source_row_index", "Name", "Split"]].copy()
                    frame["Actual"], frame["Predicted"], frame["Error"] = y, p, y - p
                    frame.to_csv(folder / "predictions.csv", index=False, encoding="utf-8")
                    predictions[cid] = p
                    report["status"] = "ok"
                except ValueError as exc:
                    if candidate["official_selected"]:
                        raise
                    report.update(status="prediction_invalid", failure_reason=str(exc),
                                  train_metrics=None, test_metrics=None)
            else:
                report["failure_reason"] = candidate.get("outer_training_refit_failure")
            write_json(folder / "results.json", report)
            (folder / "results.txt").write_text(
                f"{cid}; layers={','.join(report['layers'])}; status={report['status']}\n"
                f"Scaled: {candidate['equation_scaled']}\nRaw: {candidate['equation_raw']}\n"
                f"CV R2={candidate['cv_raw_r2_mean']}; SE={candidate['cv_raw_r2_standard_error']}\n"
                f"Train: {report['train_metrics']}\nTest: {report['test_metrics']}\n"
                f"{report.get('failure_reason') or report['test_role']}\n", encoding="utf-8")
            reports.append(report)

        # Shared axes make candidate panels directly comparable; all points are shown.
        values = np.concatenate([y, *predictions.values()])
        limits = [float(values.min()), float(values.max())]
        pad = max((limits[1] - limits[0]) * 0.05, 0.1)
        limits = (limits[0] - pad, limits[1] + pad)
        residual_max = max(float(np.max(np.abs(y - p))) for p in predictions.values())
        residual_limit = max(residual_max * 1.05, 0.1)
        import matplotlib.pyplot as plt
        style = {"font.family": "Arial", "mathtext.fontset": "custom",
                 "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                 "font.size": 20, "axes.labelsize": 24, "axes.titlesize": 24,
                 "legend.fontsize": 20, "legend.frameon": False}

        def panel(ax, kind, p):
            for label, mask, color, alpha in [("Training Set", train, "red", 0.3),
                                               ("Test Set", ~train, "blue", 0.6)]:
                if kind == "distribution":
                    bins = np.linspace(-residual_limit, residual_limit, 25)
                    ax.hist((y - p)[mask], bins=bins, color=color, alpha=alpha, label=label)
                else:
                    ax.scatter(y[mask] if kind == "prediction" else p[mask],
                               p[mask] if kind == "prediction" else (y - p)[mask],
                               s=100, color=color, alpha=alpha, label=label)
            if kind == "prediction":
                ax.plot(limits, limits, "k--", linewidth=3, label="Ideal")
                ax.set(xlabel="Calculated Value", ylabel="Predicted Value", xlim=limits, ylim=limits)
                test_r2 = r2_score(y[~train], p[~train])
                ax.text(0.95, 0.05, f"Test $R^2$ = {test_r2:.4f}", transform=ax.transAxes,
                        fontsize=20, va="bottom", ha="right",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            elif kind == "residual":
                ax.axhline(0, color="black", linestyle="--", linewidth=3, label="Zero Residual")
                ax.set(xlabel="Predicted Value", ylabel="Residual (Actual - Predicted)",
                       xlim=limits, ylim=(-residual_limit, residual_limit))
            else:
                ax.axvline(0, color="black", linestyle="--", linewidth=2)
                ax.set(xlabel="Residual (Actual - Predicted)", ylabel="Count")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            if kind == "prediction":
                ax.legend(loc="upper left")
            else:
                ax.set_xlabel(ax.get_xlabel(), fontsize=20)
                ax.set_ylabel(ax.get_ylabel(), fontsize=20)
                ax.tick_params(labelsize=14)
                ax.legend(loc="best", fontsize=16)

        with plt.rc_context(style):
            for report in reports:
                cid = report["candidate_id"]
                if cid not in predictions:
                    continue
                p = predictions[cid]
                title = cid
                for kind, filename in [("prediction", "prediction_plot_combined.png"),
                                       ("residual", "residual_plot_combined.png"),
                                       ("distribution", "residual_distribution.png")]:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    try:
                        panel(ax, kind, p)
                        ax.set_title(title, pad=15)
                        fig.savefig(output / cid / filename, dpi=dpi, bbox_inches="tight")
                    finally:
                        plt.close(fig)
                fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
                try:
                    panel(axes[0], "prediction", p)
                    panel(axes[1], "residual", p)
                    fig.suptitle(title, fontsize=22)
                    fig.savefig(output / cid / "prediction_and_residual.png", dpi=dpi)
                finally:
                    plt.close(fig)

        rows = [{"candidate_id": r["candidate_id"], "layers": ",".join(r["layers"]),
                 "official_selected": r["official_selected"], "status": r["status"],
                 "cv_r2_mean": r["cv_raw_r2_mean"], "cv_r2_se": r["cv_raw_r2_standard_error"],
                 **{f"{part}_{key}": (r[part + "_metrics"] or {}).get(key)
                    for part in ["train", "test"] for key in ["n", "r2", "rmse", "mae"]},
                 **{key: r[key] for key in ["excess_complexity", "total_term_count", "independent_parameter_count"]}}
                for r in reports]
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, encoding="utf-8")
        if any(file_sha256(run / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Candidate diagnostic source files changed during export")
        write_json(output / "manifest.json", {
            "source_run": str(run), "source_sha256": source_hashes,
            "reporting_implementation_sha256": file_sha256(Path(__file__)),
            "automatic_selection_changed": False, "training_performed": False,
            "candidate_layers": layers, "candidate_ids": ids, "figure_font": "Arial",
            "figure_dpi": dpi, "prediction_axis_limits": limits,
            "residual_axis_limits": [-residual_limit, residual_limit],
            "artifacts_sha256": {str(path.relative_to(output)): file_sha256(path)
                                 for path in sorted(output.rglob("*")) if path.is_file()},
        })
        output.rename(destination)
    return destination


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from project_paths import PROJECT_ROOT, SYMBOLIC_REGRESSION_OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Export saved SR candidate diagnostics without training or Julia.")
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_directory
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory = directory.resolve()
    if not directory.is_relative_to(SYMBOLIC_REGRESSION_OUTPUT_DIR.resolve()):
        parser.error("--run-directory must be inside outputs/symbolic_regression")
    print(export_candidate_results(directory))
