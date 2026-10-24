"""Hash-bound Ridge stacking from frozen structure-model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preparation import build_split_manifest, file_sha256
from project_paths import PROJECT_ROOT
from structure_selection import cv_splits, evaluation_cv_contract


OOF_COLUMNS = ("Repeat", "Fold", "Name", "Score", "Prediction")
PREDICTION_COLUMNS = ("Name", "Split", "Score", "Prediction")
ALPHAS = np.logspace(-4, 4, 33)
PLOT_FONT = "Arial"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.loc[:, columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite values")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_model_relative(model_path: Path, value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to the model directory")
    resolved = (model_path.parent / path).resolve()
    try:
        resolved.relative_to(model_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the model directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _validate_one_artifact(alias: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "model_path", "feature_config_path", "model_sha256", "score_version",
        "campaign_round", "split_manifest", "cv_manifest", "training_data",
        "stacking_data",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"Model {alias} artifact is missing: {missing}")

    model_path = Path(artifact["model_path"])
    config_path = Path(artifact["feature_config_path"])
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Model {alias} model or feature configuration is missing")
    actual_model_hash = file_sha256(model_path)
    if actual_model_hash != artifact["model_sha256"]:
        raise ValueError(f"Model {alias} hash mismatch")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("model_sha256", "score_version", "split_manifest", "cv_manifest", "training_data"):
        if _canonical(config.get(key)) != _canonical(artifact[key]):
            raise ValueError(f"Model {alias} {key} does not match its feature configuration")
    if str(config.get("score_manifest", {}).get("campaign_round")) != str(
        artifact["campaign_round"]
    ):
        raise ValueError(f"Model {alias} campaign round does not match its feature configuration")

    stacking = artifact["stacking_data"]
    if not isinstance(stacking, Mapping) or stacking.get("schema_version") != 1:
        raise ValueError(f"Model {alias} has unsupported stacking_data")
    if _canonical(config.get("stacking_data")) != _canonical(stacking):
        raise ValueError(f"Model {alias} stacking_data does not match its feature configuration")
    for key in ("oof_file", "oof_sha256", "predictions_file", "predictions_sha256"):
        if not stacking.get(key):
            raise ValueError(f"Model {alias} stacking_data is missing {key}")

    oof_path = _resolve_model_relative(model_path, stacking["oof_file"], f"Model {alias} OOF file")
    prediction_path = _resolve_model_relative(
        model_path, stacking["predictions_file"], f"Model {alias} prediction file"
    )
    if file_sha256(oof_path) != stacking["oof_sha256"]:
        raise ValueError(f"Model {alias} OOF hash mismatch")
    if file_sha256(prediction_path) != stacking["predictions_sha256"]:
        raise ValueError(f"Model {alias} prediction hash mismatch")

    oof = pd.read_csv(oof_path)
    predictions = pd.read_csv(prediction_path)
    _require_columns(oof, OOF_COLUMNS, f"Model {alias} OOF data")
    _require_columns(predictions, PREDICTION_COLUMNS, f"Model {alias} prediction data")
    _numeric(oof, ("Repeat", "Fold", "Score", "Prediction"), f"Model {alias} OOF data")
    _numeric(predictions, ("Score", "Prediction"), f"Model {alias} prediction data")
    if oof["Name"].isna().any() or predictions["Name"].isna().any():
        raise ValueError(f"Model {alias} contains missing names")
    oof["Name"] = oof["Name"].astype(str)
    predictions["Name"] = predictions["Name"].astype(str)
    if predictions["Name"].duplicated().any():
        raise ValueError(f"Model {alias} prediction names are not unique")
    if set(predictions["Split"]) != {"Train", "Test"}:
        raise ValueError(f"Model {alias} prediction Split must contain Train and Test")

    split = artifact["split_manifest"]
    rebuilt = build_split_manifest(
        predictions["Name"],
        np.flatnonzero(predictions["Split"].eq("Train")),
        np.flatnonzero(predictions["Split"].eq("Test")),
        test_fraction=float(split["test_fraction"]),
        requested_strata=int(split["requested_strata"]),
        distance_metric=str(split["distance_metric"]),
    )
    if _canonical(rebuilt) != _canonical(split):
        raise ValueError(f"Model {alias} prediction identities do not match split_manifest")

    cv = artifact["cv_manifest"]
    repeats, folds = int(cv["n_repeats"]), int(cv["n_splits"])
    if not np.equal(oof["Repeat"], np.floor(oof["Repeat"])).all() or not np.equal(
        oof["Fold"], np.floor(oof["Fold"])
    ).all():
        raise ValueError(f"Model {alias} OOF Repeat/Fold values must be integers")
    oof[["Repeat", "Fold"]] = oof[["Repeat", "Fold"]].astype(int)
    train_names = predictions.loc[predictions["Split"].eq("Train"), "Name"].tolist()
    test_names = set(predictions.loc[predictions["Split"].eq("Test"), "Name"])
    if set(oof["Name"]).intersection(test_names):
        raise ValueError(f"Model {alias} OOF data contains DUPLEX test names")
    if oof.duplicated(["Repeat", "Name"]).any():
        raise ValueError(f"Model {alias} OOF data repeats a name within a repeat")
    if set(oof["Repeat"]) != set(range(1, repeats + 1)):
        raise ValueError(f"Model {alias} OOF repeats do not match cv_manifest")
    if not oof["Fold"].between(1, folds).all():
        raise ValueError(f"Model {alias} OOF folds do not match cv_manifest")
    expected_names = set(train_names)
    for repeat, group in oof.groupby("Repeat"):
        if set(group["Name"]) != expected_names or len(group) != len(train_names):
            raise ValueError(f"Model {alias} OOF repeat {repeat} does not cover every train name once")
        if set(group["Fold"]) != set(range(1, folds + 1)):
            raise ValueError(f"Model {alias} OOF repeat {repeat} does not contain every expected fold")

    train_scores = predictions.loc[predictions["Split"].eq("Train"), "Score"].to_numpy(dtype=float)
    expected_splits = cv_splits(
        train_scores,
        n_splits=folds,
        repeats=repeats,
        seed=int(cv["random_state"]),
    )
    for index, (_, validation_indices) in enumerate(expected_splits):
        repeat, fold = index // folds + 1, index % folds + 1
        expected_fold_names = {train_names[i] for i in validation_indices}
        actual_fold_names = set(
            oof.loc[oof["Repeat"].eq(repeat) & oof["Fold"].eq(fold), "Name"]
        )
        if actual_fold_names != expected_fold_names:
            raise ValueError(
                f"Model {alias} OOF Repeat/Fold assignments do not match cv_manifest"
            )

    full_scores = predictions.set_index("Name")["Score"]
    expected_oof_scores = oof["Name"].map(full_scores).to_numpy(dtype=float)
    if not np.allclose(oof["Score"], expected_oof_scores, rtol=0.0, atol=1e-12):
        raise ValueError(f"Model {alias} OOF and full-data Scores disagree")
    return {
        "model_path": model_path,
        "feature_config_path": config_path,
        "oof_path": oof_path,
        "prediction_path": prediction_path,
        "oof": oof,
        "predictions": predictions,
        "artifact": artifact,
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def _plot_results(frame: pd.DataFrame, path: Path, residuals: bool) -> None:
    with plt.rc_context({"font.family": PLOT_FONT}):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
        for split, color, alpha, label in (
            ("Train", "red", 0.3, "Training Set"),
            ("Test", "blue", 0.6, "Test Set"),
        ):
            part = frame.loc[frame["Split"].eq(split)]
            x = part["StackingPrediction"] if residuals else part["Score"]
            y = part["Score"] - part["StackingPrediction"] if residuals else part["StackingPrediction"]
            ax.scatter(x, y, label=label, color=color, alpha=alpha, s=100)
        if residuals:
            ax.axhline(0, color="black", linestyle="--", linewidth=3)
        else:
            values = frame[["Score", "StackingPrediction"]].to_numpy()
            bounds = [values.min(), values.max()]
            ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=3, label="Ideal")
            test = frame.loc[frame["Split"].eq("Test")]
            test_r2 = r2_score(test["Score"], test["StackingPrediction"])
            ax.text(0.95, 0.05, f"Test R² = {test_r2:.4f}", transform=ax.transAxes,
                    fontsize=20, va="bottom", ha="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_xlabel("Predicted Value" if residuals else "Calculated Value", fontsize=22)
        ax.set_ylabel("Residual" if residuals else "Predicted Value", fontsize=22)
        ax.tick_params(labelsize=20)
        ax.legend(fontsize=20, loc="best" if residuals else "upper left")
        ax.set_title(f"Stacking {'Residual' if residuals else 'Prediction'} Results", fontsize=24, pad=10)
        ax.grid(alpha=0.3)
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_predictions(frame: pd.DataFrame, path: Path) -> None:
    _plot_results(frame, path, residuals=False)


def _plot_residuals(frame: pd.DataFrame, path: Path) -> None:
    _plot_results(frame, path, residuals=True)


def train_stacking(models, artifacts, encoder, output_dir):
    """Fit and export a Ridge stack using only nested OOF training predictions."""
    # Retain encoder for existing callers; frozen base models own feature alignment.
    model_order = list(models)
    if len(model_order) < 2 or len(model_order) != len(set(model_order)):
        raise ValueError("Stacking requires at least two uniquely named base models")
    if set(model_order) != set(artifacts):
        raise ValueError("Loaded models and stacking artifacts do not match")

    loaded = {alias: _validate_one_artifact(alias, artifacts[alias]) for alias in model_order}
    first = loaded[model_order[0]]
    for field in ("score_version", "campaign_round", "split_manifest", "cv_manifest", "training_data"):
        contracts = {_canonical(evaluation_cv_contract(loaded[alias]["artifact"][field])
                                if field == 'cv_manifest' else loaded[alias]["artifact"][field])
                     for alias in model_order}
        if len(contracts) != 1:
            raise ValueError(f"Base models do not share one {field} contract")

    reference_full = first["predictions"].loc[:, ["Name", "Split", "Score"]].reset_index(drop=True)
    reference_assignments = first["oof"].loc[:, ["Repeat", "Fold", "Name", "Score"]].sort_values(
        ["Repeat", "Name"]
    ).reset_index(drop=True)
    full = reference_full.copy()
    oof = reference_assignments.loc[:, ["Name", "Score"]].drop_duplicates("Name").copy()
    oof = oof.set_index("Name").loc[
        reference_full.loc[reference_full["Split"].eq("Train"), "Name"]
    ].reset_index()

    for alias in model_order:
        prediction_data = loaded[alias]["predictions"]
        identity = prediction_data.loc[:, ["Name", "Split", "Score"]].reset_index(drop=True)
        if not identity.equals(reference_full):
            raise ValueError(f"Model {alias} full prediction identity, split, or Score differs")
        assignments = loaded[alias]["oof"].loc[:, ["Repeat", "Fold", "Name", "Score"]].sort_values(
            ["Repeat", "Name"]
        ).reset_index(drop=True)
        if not assignments.equals(reference_assignments):
            raise ValueError(f"Model {alias} OOF identity, assignment, or Score differs")
        full[f"{alias}_Prediction"] = prediction_data["Prediction"].to_numpy(dtype=float)
        averaged = loaded[alias]["oof"].groupby("Name", sort=False)["Prediction"].mean()
        oof[f"{alias}_Prediction"] = oof["Name"].map(averaged).to_numpy(dtype=float)

    feature_columns = [f"{alias}_Prediction" for alias in model_order]
    X_oof = oof[feature_columns].to_numpy(dtype=float)
    y_oof = oof["Score"].to_numpy(dtype=float)
    search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": ALPHAS},
        scoring="r2",
        cv=cv_splits(y_oof, n_splits=4, repeats=10, seed=42),
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X_oof, y_oof)
    fitted = search.best_estimator_
    oof["StackingPrediction"] = fitted.predict(X_oof)
    full["StackingPrediction"] = fitted.predict(full[feature_columns].to_numpy(dtype=float))

    destination = Path(output_dir) / "stacking"
    destination.mkdir(parents=True, exist_ok=True)
    stacking_model_path = destination / "Stacking_best_model.pkl"
    joblib.dump(fitted, stacking_model_path)
    full.to_csv(destination / "predictions.csv", index=False)
    oof.to_csv(destination / "oof_training_predictions.csv", index=False)

    search_frame = pd.DataFrame({
        "Alpha": search.cv_results_["param_ridge__alpha"].astype(float),
        "Mean_Meta_Selection_R2": search.cv_results_["mean_test_score"],
        "Std_Meta_Selection_R2": search.cv_results_["std_test_score"],
        "Rank_Meta_Selection_R2": search.cv_results_["rank_test_score"],
        "Scope": "meta_selection_only",
    }).sort_values("Alpha")
    search_frame.to_csv(destination / "ridge_search.csv", index=False)

    metric_rows = []
    scopes = {
        "meta_oof_fit": (oof["Score"].to_numpy(dtype=float), oof),
        "train_in_sample": (
            full.loc[full["Split"].eq("Train"), "Score"].to_numpy(dtype=float),
            full.loc[full["Split"].eq("Train")],
        ),
        "duplex_test": (
            full.loc[full["Split"].eq("Test"), "Score"].to_numpy(dtype=float),
            full.loc[full["Split"].eq("Test")],
        ),
    }
    for scope, (target, frame) in scopes.items():
        for model_name, column in [(alias, f"{alias}_Prediction") for alias in model_order] + [
            ("Stacking", "StackingPrediction")
        ]:
            metric_rows.append({"Model": model_name, "Scope": scope, **_metrics(target, frame[column])})
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(destination / "metrics.csv", index=False)

    scaler, ridge = fitted.named_steps["scaler"], fitted.named_steps["ridge"]
    raw_coefficients = ridge.coef_ / scaler.scale_
    raw_intercept = ridge.intercept_ - float(np.dot(raw_coefficients, scaler.mean_))
    source_records = []
    for alias in model_order:
        artifact, item = loaded[alias]["artifact"], loaded[alias]
        source_records.append({
            "model": alias,
            "model_file": _portable_path(item["model_path"]),
            "model_sha256": artifact["model_sha256"],
            "feature_config_file": _portable_path(item["feature_config_path"]),
            "feature_config_sha256": file_sha256(item["feature_config_path"]),
            "oof_file": _portable_path(item["oof_path"]),
            "oof_sha256": artifact["stacking_data"]["oof_sha256"],
            "predictions_file": _portable_path(item["prediction_path"]),
            "predictions_sha256": artifact["stacking_data"]["predictions_sha256"],
        })
    config = {
        "schema_version": 1,
        "stacking_model_file": stacking_model_path.name,
        "stacking_model_sha256": file_sha256(stacking_model_path),
        "model_order": model_order,
        "alpha": float(search.best_params_["ridge__alpha"]),
        "meta_cv": {
            "purpose": "meta_selection_only",
            "method": "RepeatedStratifiedKFold",
            "n_splits": 4,
            "n_repeats": 10,
            "requested_strata": 4,
            "random_state": 42,
            "scoring": "r2",
            "alpha_grid": ALPHAS.tolist(),
            "best_mean_r2": float(search.best_score_),
        },
        "raw_prediction_space_coefficients": dict(zip(model_order, raw_coefficients.tolist())),
        "raw_prediction_space_intercept": float(raw_intercept),
        "score_version": first["artifact"]["score_version"],
        "campaign_round": first["artifact"]["campaign_round"],
        "split_manifest": first["artifact"]["split_manifest"],
        "cv_manifest": first["artifact"]["cv_manifest"],
        "training_data": first["artifact"]["training_data"],
        "sources": source_records,
        "scope_notes": {
            "ridge_search": "Meta-model selection on averaged nested OOF predictions from DUPLEX training rows only.",
            "meta_oof_fit": "Training diagnostic from fitting and predicting the same averaged OOF feature rows; not full-stack OOF or an unbiased estimate.",
            "train_in_sample": "Diagnostic using frozen final base-model predictions on DUPLEX training rows.",
            "duplex_test": "Held-out from this stacking fit, but not claimed independent of prior project development history.",
        },
    }
    (destination / "model_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coefficient_lines = "\n".join(
        f"  {alias}: {config['raw_prediction_space_coefficients'][alias]:.12g}"
        for alias in model_order
    )
    metric_report = metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    report = (
        "Ridge structure stacking\n"
        f"Base model order: {', '.join(model_order)}\n"
        f"Selected alpha: {config['alpha']:.12g}\n"
        f"Best mean meta-selection CV R2 (4 folds x 10 repeats, seed 42): {search.best_score_:.6f}\n"
        "Raw prediction-space coefficients:\n"
        f"{coefficient_lines}\n"
        f"Raw prediction-space intercept: {raw_intercept:.12g}\n\n"
        "Metrics:\n"
        f"{metric_report}\n\n"
        "Ridge selection used only averaged nested OOF predictions from DUPLEX training rows.\n"
        "The base-model inputs in meta_oof_fit are genuine nested OOF predictions, while the Stacking prediction is fitted on those same rows.\n"
        "Therefore meta_oof_fit is a training diagnostic, not full-stack OOF or an unbiased estimate.\n"
        "The DUPLEX test rows were excluded from stacking fit and tuning; prior development history means they are not claimed as a new independent test.\n"
    )
    (destination / "final_report.txt").write_text(report, encoding="utf-8")
    _plot_predictions(full, destination / "Stacking_prediction_plot.png")
    _plot_residuals(full, destination / "Stacking_residual_plot.png")
    return fitted
