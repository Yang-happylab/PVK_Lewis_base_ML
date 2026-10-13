"""Within-campaign experimental Score validation for posterior molecules."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from device_normalization import configure_plot_style
from project_paths import PROJECT_ROOT


PLQY_PERCENT_COLUMNS = (
    "PLQY_percent_1",
    "PLQY_percent_2",
    "PLQY_percent_3",
)
XRD_RATIO_COLUMNS = (
    "XRD_ratio_1",
    "XRD_ratio_2",
    "XRD_ratio_3",
)
POSTERIOR_MEASUREMENT_COLUMNS = (*PLQY_PERCENT_COLUMNS, *XRD_RATIO_COLUMNS)
ANALYSIS_VERSION = "posterior-score-rank-v5"
DEFAULT_ALPHA = 0.05
EXACT_PERMUTATION_MAX_N = 9
MONTE_CARLO_PERMUTATIONS = 100_000
PERMUTATION_RANDOM_STATE = 42
PLOT_DPI = 300


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound_score_scale(
    provenance: dict[str, Any] | None,
    campaign_round: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load the immutable Score scale; posterior REF supplies normalization."""
    if provenance is None:
        return None, {"status": "unavailable-unbound-model"}
    consumer = provenance.get("consumer", {})
    manifest = provenance.get("score_manifest", {})
    root = PROJECT_ROOT.resolve()

    def checked_artifact(relative: str, expected_hash: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Score artifact escapes the project root") from exc
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"Score artifact is missing or hash-mismatched: {relative}")
        return path

    model_relative = str(consumer.get("model_file") or (
        Path(consumer["score_output_dir"]) / manifest["artifact_files"]["model_file"]
    ))
    model_path = checked_artifact(
        model_relative, manifest["artifact_sha256"]["model.json"]
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    checked_artifact(consumer["scores_file"], manifest["artifact_sha256"]["scores.csv"])
    if (
        model.get("score_version") != provenance.get("score_version")
        or model.get("campaign_round") != campaign_round
        or consumer.get("score_version") != provenance.get("score_version")
        or model.get("algorithm_version") != "score-calibration-v3"
        or model.get("score_definition")
        != "equal mean of independently library-standardized batch Scores; no final rescaling"
    ):
        raise ValueError("Bound Score model version or definition does not match")
    batches = model.get("batch_models")
    if not isinstance(batches, list) or not batches:
        raise ValueError("Bound Score model has no batch transformations")
    if batches[-1].get("campaign_round") != campaign_round:
        raise ValueError("Bound Score batches do not end at the selected campaign")
    for batch in batches:
        if batch.get("score_definition") != "z_PLQY - z_XRD":
            raise ValueError("Unsupported batch Score definition")
        for endpoint in ("PLQY", "XRD"):
            scale = batch["endpoint_scaling"][endpoint]
            mean = float(scale["mean"])
            sd = float(scale["standard_deviation"])
            if not np.isfinite([mean, sd]).all() or sd <= 0:
                raise ValueError("Invalid bound Score endpoint scaling")

    return model, {
        "status": "available",
        "score_version": model["score_version"],
        "score_model_file": model_relative,
        "score_model_sha256": file_sha256(model_path),
        "scores_sha256": manifest["artifact_sha256"]["scores.csv"],
        "batch_rounds": [batch["campaign_round"] for batch in batches],
    }


def coerce_measurement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate optional posterior measurements while preserving missing values."""
    missing = [
        column for column in POSTERIOR_MEASUREMENT_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Posterior CSV is missing measurement columns: " + ", ".join(missing)
        )

    result = frame.copy()
    for column in POSTERIOR_MEASUREMENT_COLUMNS:
        source = result[column]
        nonblank = source.notna()
        if source.dtype == object:
            nonblank &= source.astype(str).str.strip().ne("")
        numeric = pd.to_numeric(source.where(nonblank), errors="coerce")
        invalid = nonblank & numeric.isna()
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
            raise ValueError(
                f"Posterior measurement {column} contains nonnumeric values at "
                f"CSV rows {rows}"
            )
        observed = numeric.dropna().to_numpy(dtype=float)
        if observed.size and not np.isfinite(observed).all():
            raise ValueError(f"Posterior measurement {column} contains non-finite values")
        if observed.size and (observed < 0).any():
            raise ValueError(f"Posterior measurement {column} cannot be negative")
        if column.startswith("PLQY_") and observed.size and (observed > 100).any():
            raise ValueError(
                f"Posterior measurement {column} exceeds 100 percent"
            )
        result[column] = numeric.astype(float)
    return result


def _replicate_summary(values: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    count = int(observed.size)
    if count == 0:
        return {
            "N": 0,
            "Status": "missing",
            "Mean": np.nan,
            "SD": np.nan,
            "SEM": np.nan,
            "CI95_Lower": np.nan,
            "CI95_Upper": np.nan,
        }

    mean = float(np.mean(observed))
    if count < 2:
        sd = sem = ci_lower = ci_upper = np.nan
    else:
        sd = float(np.std(observed, ddof=1))
        sem = float(sd / math.sqrt(count))
        half_width = float(stats.t.ppf(0.975, count - 1) * sem)
        ci_lower = mean - half_width
        ci_upper = mean + half_width
    return {
        "N": count,
        "Status": "complete" if count == 3 else "partial",
        "Mean": mean,
        "SD": sd,
        "SEM": sem,
        "CI95_Lower": ci_lower,
        "CI95_Upper": ci_upper,
    }


def summarize_posterior_reference(
    reference: pd.DataFrame,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Summarize the one same-batch REF row, including pending measurements."""
    if len(reference) != 1 or not reference["Name"].eq("REF").all():
        raise ValueError("Each posterior round requires exactly one REF row")
    clean = coerce_measurement_columns(reference)
    summaries = {}
    for endpoint, columns in (("PLQY", PLQY_PERCENT_COLUMNS), ("XRD", XRD_RATIO_COLUMNS)):
        values = clean.loc[:, list(columns)].to_numpy(dtype=float).ravel()
        if (values[np.isfinite(values)] <= 0).any():
            raise ValueError(f"Posterior REF {endpoint} measurements must be positive")
        summaries[endpoint] = _replicate_summary(values)
    complete = all(value["N"] == 3 for value in summaries.values())
    means = {key: value["Mean"] for key, value in summaries.items()} if complete else None
    return means, {
        "status": "available" if complete else "unavailable-incomplete-posterior-reference",
        "reference_round": str(clean.iloc[0]["Round"]),
        "reference_name": "REF",
        "reference_means": means,
        "measurements": {
            column: None if pd.isna(clean.iloc[0][column]) else float(clean.iloc[0][column])
            for column in POSTERIOR_MEASUREMENT_COLUMNS
        },
        "replicate_summary": {
            endpoint: {key: None if pd.isna(value) else value for key, value in summary.items()}
            for endpoint, summary in summaries.items()
        },
        "normalization": "100 * molecule mean / same-batch posterior REF mean",
    }


def build_measurement_summary(
    frame: pd.DataFrame,
    predictions: Sequence[float],
    interval_lower: Sequence[float],
    interval_upper: Sequence[float],
    interval_status: str,
    score_model: dict[str, Any] | None,
    reference_means: dict[str, float] | None,
) -> pd.DataFrame:
    """Create one auditable experimental and prediction summary per input row."""
    clean = coerce_measurement_columns(frame)
    prediction_values = np.asarray(predictions, dtype=float).reshape(-1)
    lower_values = np.asarray(interval_lower, dtype=float).reshape(-1)
    upper_values = np.asarray(interval_upper, dtype=float).reshape(-1)
    if not (
        len(clean)
        == len(prediction_values)
        == len(lower_values)
        == len(upper_values)
    ):
        raise ValueError("Posterior measurements and predictions are not aligned")

    identity_columns = ["Round", "Name", "SMILES"]
    if "_Input_Order" in clean.columns:
        identity_columns.append("_Input_Order")
    summary = clean.loc[
        :, [*identity_columns, *POSTERIOR_MEASUREMENT_COLUMNS]
    ].copy()
    summary["Predicted_Score"] = prediction_values
    summary["Score_PI95_Lower"] = lower_values
    summary["Score_PI95_Upper"] = upper_values
    summary["Score_PI95_Status"] = interval_status

    plqy_records = []
    xrd_records = []
    for row in clean.itertuples(index=False):
        row_values = row._asdict()
        plqy_records.append(
            _replicate_summary([row_values[column] for column in PLQY_PERCENT_COLUMNS])
        )
        xrd_records.append(
            _replicate_summary([row_values[column] for column in XRD_RATIO_COLUMNS])
        )

    for prefix, records in (("PLQY", plqy_records), ("XRD", xrd_records)):
        record_frame = pd.DataFrame(records)
        for column in record_frame.columns:
            summary[f"{prefix}_{column}"] = record_frame[column].to_numpy()

    summary["Measurement_Complete"] = (
        summary["PLQY_Status"].eq("complete")
        & summary["XRD_Status"].eq("complete")
    )
    summary["Predicted_Rank_All"] = summary["Predicted_Score"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["Predicted_Rank"] = np.nan
    summary["PLQY_Relative_Ref_Percent"] = np.nan
    summary["XRD_Relative_Ref_Percent"] = np.nan
    summary["Observed_Score"] = np.nan
    summary["Observed_Score_Status"] = (
        "unavailable-unbound-model" if score_model is None else
        "unavailable-incomplete-posterior-reference" if reference_means is None else
        "incomplete_replicates"
    )
    summary["Observed_Rank"] = np.nan
    summary["Rank_Difference"] = np.nan
    if score_model is not None and reference_means is not None:
        complete = summary["Measurement_Complete"]
        plqy = 100.0 * summary.loc[complete, "PLQY_Mean"] / reference_means["PLQY"]
        xrd = 100.0 * summary.loc[complete, "XRD_Mean"] / reference_means["XRD"]
        summary.loc[complete, "PLQY_Relative_Ref_Percent"] = plqy
        summary.loc[complete, "XRD_Relative_Ref_Percent"] = xrd
        batch_scores = []
        for batch in score_model["batch_models"]:
            scaling = batch["endpoint_scaling"]
            p = scaling["PLQY"]
            x = scaling["XRD"]
            batch_scores.append(
                (plqy - p["mean"]) / p["standard_deviation"]
                - (xrd - x["mean"]) / x["standard_deviation"]
            )
        summary.loc[complete, "Observed_Score"] = np.mean(batch_scores, axis=0)
        summary.loc[complete, "Observed_Score_Status"] = "available"
        summary.loc[complete, "Predicted_Rank"] = summary.loc[
            complete, "Predicted_Score"
        ].rank(method="min", ascending=False)
        summary.loc[complete, "Observed_Rank"] = summary.loc[
            complete, "Observed_Score"
        ].rank(method="min", ascending=False)
        summary.loc[complete, "Rank_Difference"] = (
            summary.loc[complete, "Observed_Rank"]
            - summary.loc[complete, "Predicted_Rank"]
        )
    return summary


def _permutation_indices(sample_count: int) -> tuple[np.ndarray, str, int | None]:
    if sample_count <= EXACT_PERMUTATION_MAX_N:
        permutations = np.asarray(
            list(itertools.permutations(range(sample_count))), dtype=int
        )
        return permutations, "exhaustive exact permutation", None

    rng = np.random.default_rng(PERMUTATION_RANDOM_STATE)
    permutations = np.empty(
        (MONTE_CARLO_PERMUTATIONS, sample_count), dtype=int
    )
    base = np.arange(sample_count, dtype=int)
    for index in range(MONTE_CARLO_PERMUTATIONS):
        permutations[index] = rng.permutation(base)
    return (
        permutations,
        "Monte Carlo permutation with plus-one correction",
        PERMUTATION_RANDOM_STATE,
    )


def _rank_correlations_with_permutation(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Rank-correlation inputs must be aligned one-dimensional arrays")
    if len(x) < 3:
        raise ValueError("At least three molecules are required for rank correlation")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Rank-correlation inputs must be finite")

    x_ranks = stats.rankdata(x, method="average")
    y_ranks = stats.rankdata(y, method="average")
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    spearman_denominator = float(
        np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    )
    if spearman_denominator == 0:
        raise ValueError("Rank correlation is undefined for a constant variable")
    observed_spearman = float(
        np.sum(x_centered * y_centered) / spearman_denominator
    )

    pair_i, pair_j = np.triu_indices(len(x), k=1)
    x_signs = np.sign(x[pair_i] - x[pair_j])
    y_signs = np.sign(y[pair_i] - y[pair_j])
    kendall_denominator = float(
        np.sqrt(np.sum(x_signs != 0) * np.sum(y_signs != 0))
    )
    if kendall_denominator == 0:
        raise ValueError("Kendall tau-b is undefined for a constant variable")
    observed_kendall = float(
        np.sum(x_signs * y_signs) / kendall_denominator
    )

    permutations, method, random_seed = _permutation_indices(len(x))
    permuted_y_ranks = y_ranks[permutations]
    permuted_y_centered = permuted_y_ranks - np.mean(y_ranks)
    permuted_spearman = (
        permuted_y_centered @ x_centered / spearman_denominator
    )
    permuted_y = y[permutations]
    permuted_y_signs = np.sign(
        permuted_y[:, pair_i] - permuted_y[:, pair_j]
    )
    permuted_kendall = (
        permuted_y_signs @ x_signs / kendall_denominator
    )
    tolerance = 1e-12
    spearman_extreme = int(
        np.sum(np.abs(permuted_spearman) >= abs(observed_spearman) - tolerance)
    )
    kendall_extreme = int(
        np.sum(np.abs(permuted_kendall) >= abs(observed_kendall) - tolerance)
    )
    permutation_count = int(len(permutations))
    if random_seed is None:
        spearman_p = spearman_extreme / permutation_count
        kendall_p = kendall_extreme / permutation_count
    else:
        spearman_p = (spearman_extreme + 1) / (permutation_count + 1)
        kendall_p = (kendall_extreme + 1) / (permutation_count + 1)
    return {
        "Spearman_Rho": observed_spearman,
        "Spearman_P_Two_Sided": float(spearman_p),
        "Kendall_Tau_b": observed_kendall,
        "Kendall_P_Two_Sided": float(kendall_p),
        "P_Value_Method": method,
        "Permutation_Count": permutation_count,
        "Random_Seed": random_seed,
    }


def compute_rank_statistics(
    summary: pd.DataFrame,
    campaign_round: str,
) -> pd.DataFrame:
    """Compare predicted and experimental Score only within one campaign."""
    if not summary["Round"].eq(campaign_round).all():
        raise ValueError("Rank statistics cannot mix posterior rounds")
    complete = summary.loc[summary["Observed_Score"].notna()]
    row: dict[str, Any] = {
        "Round": campaign_round,
        "N_Molecules": int(len(complete)),
        "Status": "ok",
        "Spearman_Rho": np.nan,
        "Spearman_P_Two_Sided": np.nan,
        "Kendall_Tau_b": np.nan,
        "Kendall_P_Two_Sided": np.nan,
        "P_Value_Method": None,
        "Permutation_Count": np.nan,
        "Random_Seed": np.nan,
    }
    if complete["Name"].duplicated().any():
        row["Status"] = "duplicate molecule names"
    elif len(complete) < 3:
        row["Status"] = "insufficient_n_or_unbound_score"
    else:
        try:
            row.update(
                _rank_correlations_with_permutation(
                    complete["Predicted_Score"].to_numpy(dtype=float),
                    complete["Observed_Score"].to_numpy(dtype=float),
                )
            )
        except ValueError as exc:
            row["Status"] = str(exc)
    return pd.DataFrame([row])


def score_prediction_intervals(
    model_path: Path,
    model_contract: dict[str, Any],
    predictions: Sequence[float],
    *,
    expected_campaign_round: str,
    expected_score_version: str | None,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrate symmetric empirical Score intervals from consensus OOF residuals."""
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    unavailable = {
        "status": "unavailable-no-bound-oof-calibration",
        "method": None,
        "confidence_level": 1.0 - alpha,
    }
    calibration = model_contract.get("prediction_interval_calibration")
    if not isinstance(calibration, dict):
        return (
            np.full_like(predicted, np.nan),
            np.full_like(predicted, np.nan),
            unavailable,
        )

    required = {
        "schema_version",
        "method",
        "campaign_round",
        "score_version",
        "calibration_file",
        "calibration_sha256",
        "calibration_manifest_file",
        "calibration_manifest_sha256",
        "residual_column",
        "sample_count",
    }
    missing = sorted(required.difference(calibration))
    if missing:
        raise ValueError(
            "Prediction-interval calibration is missing fields: " + ", ".join(missing)
        )
    if int(calibration["schema_version"]) != 1:
        raise ValueError("Unsupported prediction-interval calibration schema")
    if calibration["method"] != "absolute-consensus-oof-residual-conformal":
        raise ValueError("Unsupported prediction-interval calibration method")
    if str(calibration["campaign_round"]) != expected_campaign_round:
        raise ValueError("Prediction-interval calibration campaign does not match model")
    if (
        expected_score_version is not None
        and str(calibration["score_version"]) != expected_score_version
    ):
        raise ValueError("Prediction-interval calibration Score version does not match")

    model_directory = model_path.parent.resolve()
    calibration_path = (model_directory / calibration["calibration_file"]).resolve()
    manifest_path = (
        model_directory / calibration["calibration_manifest_file"]
    ).resolve()
    for path, label in (
        (calibration_path, "Prediction-interval calibration"),
        (manifest_path, "Prediction-interval calibration manifest"),
    ):
        try:
            path.relative_to(model_directory)
        except ValueError as exc:
            raise ValueError(f"{label} escapes the model directory") from exc
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if file_sha256(calibration_path) != calibration["calibration_sha256"]:
        raise ValueError("Prediction-interval calibration SHA-256 mismatch")
    if file_sha256(manifest_path) != calibration["calibration_manifest_sha256"]:
        raise ValueError("Prediction-interval calibration manifest SHA-256 mismatch")

    calibration_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if calibration_manifest.get("analysis") != "cross-fitted-oof-shap":
        raise ValueError("Prediction-interval manifest is not cross-fitted OOF SHAP")
    if calibration_manifest.get("model_stage") != "selected_features":
        raise ValueError("Prediction-interval manifest model stage does not match")
    if str(calibration_manifest.get("campaign_round")) != expected_campaign_round:
        raise ValueError("Prediction-interval manifest campaign does not match")
    if (
        expected_score_version is not None
        and str(calibration_manifest.get("score_version")) != expected_score_version
    ):
        raise ValueError("Prediction-interval manifest Score version does not match")
    if calibration_manifest.get("outer_test_used_for_oof_shap") is not False:
        raise ValueError("Prediction-interval manifest includes the outer test set")

    calibration_frame = pd.read_csv(calibration_path, encoding="utf-8")
    residual_column = str(calibration["residual_column"])
    required_columns = {"Name", residual_column}
    if not required_columns.issubset(calibration_frame.columns):
        raise ValueError("Prediction-interval calibration columns are incomplete")
    if calibration_frame["Name"].astype(str).duplicated().any():
        raise ValueError("Prediction-interval calibration contains duplicate Names")
    residuals = pd.to_numeric(
        calibration_frame[residual_column], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(residuals).all():
        raise ValueError("Prediction-interval residuals must be finite")
    sample_count = len(residuals)
    if sample_count != int(calibration["sample_count"]):
        raise ValueError("Prediction-interval calibration sample count mismatch")
    if int(calibration_manifest.get("outer_training_samples", -1)) != sample_count:
        raise ValueError("Prediction-interval manifest sample count mismatch")
    if (
        calibration_manifest.get("artifacts", {}).get("consensus_values")
        != calibration_path.name
    ):
        raise ValueError("Prediction-interval manifest points to another calibration")
    if sample_count < 20:
        raise ValueError("At least 20 OOF samples are required for Score intervals")

    quantile_rank = min(
        sample_count,
        int(math.ceil((sample_count + 1) * (1.0 - alpha))),
    )
    radius = float(np.sort(np.abs(residuals))[quantile_rank - 1])
    metadata = {
        "status": "available",
        "method": "absolute-consensus-oof-residual-conformal",
        "coverage_note": (
            "Empirical interval calibrated on unique outer-training OOF residuals; "
            "nominal coverage is not guaranteed for out-of-distribution molecules."
        ),
        "confidence_level": 1.0 - alpha,
        "alpha": alpha,
        "calibration_file": calibration_path.name,
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_manifest_file": manifest_path.name,
        "calibration_manifest_sha256": calibration[
            "calibration_manifest_sha256"
        ],
        "sample_count": sample_count,
        "quantile_rank": quantile_rank,
        "absolute_residual_radius": radius,
    }
    return predicted - radius, predicted + radius, metadata


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "p = NA"
    if value < 0.0001:
        return "p < 0.0001"
    return f"p = {value:.4f}"


def _plot_rank_comparison(
    summary: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    output_path: Path,
    model_label: str | None = None,
) -> Path | None:
    """Draw measured rank connections or a prediction-only pending chart."""
    plotted = summary.loc[summary["Observed_Score"].notna()].copy()
    pending = plotted.empty
    if pending:
        plotted = summary.copy()
    configure_plot_style()
    rank_column = "Predicted_Rank_All" if pending else "Predicted_Rank"
    plotted = plotted.sort_values([rank_column, "Name"])
    colors = plt.get_cmap("viridis")(
        np.linspace(0.10, 0.90, len(plotted))
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    for color, row in zip(colors, plotted.itertuples(index=False)):
        predicted_rank = float(getattr(row, rank_column))
        if not pending:
            axis.plot([0, 1], [predicted_rank, float(row.Observed_Rank)],
                      color=color, linewidth=2.4, alpha=0.9)
        axis.scatter([0] if pending else [0, 1],
                     [predicted_rank] if pending else [predicted_rank, float(row.Observed_Rank)],
                     s=95, color=color, edgecolor="black", linewidth=0.65, zorder=3)
        axis.text(-0.08, predicted_rank, f"{row.Name}  {row.Predicted_Score:.3f}",
                  ha="right", va="center", fontsize=15)
        if not pending:
            axis.text(1.08, float(row.Observed_Rank), f"{row.Observed_Score:.3f}  {row.Name}",
                      ha="left", va="center", fontsize=15)

    statistic = statistics_frame.iloc[0]
    if pending:
        annotation = "Experimental Score pending\nRank statistics unavailable"
    elif statistic["Status"] == "ok":
        annotation = (
            f"n = {int(statistic['N_Molecules'])} molecules\n"
            rf"Spearman $\rho$ = {statistic['Spearman_Rho']:.3f}, "
            f"{_format_p_value(float(statistic['Spearman_P_Two_Sided']))}\n"
            rf"Kendall $\tau_b$ = {statistic['Kendall_Tau_b']:.3f}, "
            f"{_format_p_value(float(statistic['Kendall_P_Two_Sided']))}"
        )
    else:
        annotation = (
            f"n = {int(statistic['N_Molecules'])} molecules\n"
            f"Rank statistics unavailable: {statistic['Status']}"
        )
    axis.set_xlim(-0.55, 0.55 if pending else 1.55)
    axis.set_ylim(len(plotted) + 0.6, -1.3)
    axis.set_xticks([0] if pending else [0, 1],
                    ["Predicted Score rank"] if pending else
                    ["Predicted Score rank", "Experimental Score rank"], fontsize=18)
    axis.set_yticks(np.arange(1, len(plotted) + 1))
    axis.set_ylabel("Rank (1 = highest)", fontsize=20)
    title_suffix = "Predicted Score Ranking" if pending else "Score Rank Agreement"
    title = (f"{model_label.replace('_', ' ')}: {title_suffix}" if model_label
             else title_suffix)
    axis.set_title(title,
                   fontsize=22, pad=16)
    axis.text(0.02, 0.98, annotation, transform=axis.transAxes,
              ha="left", va="top", fontsize=16)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.65)
    axis.tick_params(axis="both", which="major", labelsize=16,
                     width=1.5, length=6, pad=7)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)
    figure.tight_layout(pad=1.2)
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)
    return output_path


def _plot_score_scatter(
    summary: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    output_path: Path,
    model_label: str | None = None,
) -> Path | None:
    """Plot predicted against same-round experimental Score."""
    plotted = summary.dropna(subset=["Observed_Score"]).sort_values("Predicted_Score")
    if plotted.empty:
        return None
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(10, 8))
    figure.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.89)
    x = plotted["Predicted_Score"].to_numpy(dtype=float)
    y = plotted["Observed_Score"].to_numpy(dtype=float)
    x_span, y_span = max(np.ptp(x), 1.0), max(np.ptp(y), 1.0)
    axis.set_xlim(x.min() - 0.2 * x_span, x.max() + 0.2 * x_span)
    axis.set_ylim(y.min() - 0.15 * y_span, y.max() + 0.35 * y_span)
    colors = plt.get_cmap("viridis")(np.linspace(0.10, 0.90, len(plotted)))
    axis.scatter(x, y, s=110, c=colors, edgecolor="black", linewidth=0.7, zorder=3)
    axis.set_xlabel("Predicted Score", fontsize=20)
    axis.set_ylabel("Experimental Score", fontsize=22)
    axis.set_title(
        f"{model_label.replace('_', ' ') if model_label else 'Model'}: Predicted vs. Experimental Score",
        fontsize=22, pad=16,
    )
    statistic = statistics_frame.iloc[0]
    annotation = f"n = {int(statistic['N_Molecules'])} molecules"
    if statistic["Status"] == "ok":
        annotation += (
            f"\nSpearman rho = {statistic['Spearman_Rho']:.3f}, "
            f"{_format_p_value(float(statistic['Spearman_P_Two_Sided']))}"
            f"\nKendall tau-b = {statistic['Kendall_Tau_b']:.3f}, "
            f"{_format_p_value(float(statistic['Kendall_P_Two_Sided']))}"
        )
    else:
        annotation += "\nRank statistics unavailable"
    statistics_text = axis.text(0.02, 0.98, annotation, transform=axis.transAxes,
                                ha="left", va="top", fontsize=16)
    axis.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.55)
    axis.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent().padded(-4)
    obstacles = [statistics_text.get_window_extent(renderer).padded(3)]
    for px, py in zip(x, y):
        cx, cy = axis.transData.transform((px, py))
        obstacles.append(matplotlib.transforms.Bbox.from_extents(cx-9, cy-9, cx+9, cy+9))
    offsets = [(dx, dy) for distance in (10, 22, 38, 55, 75)
               for dy in (distance, -distance) for dx in (8, -8, 0)]
    for row in plotted.itertuples(index=False):
        for dx, dy in offsets:
            label = axis.annotate(
                str(row.Name), (row.Predicted_Score, row.Observed_Score),
                xytext=(dx, dy), textcoords="offset points",
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                va="bottom" if dy > 0 else "top", fontsize=15,
                arrowprops=dict(arrowstyle="-", linestyle="--", color="#666666",
                                linewidth=0.8, shrinkA=3, shrinkB=6),
            )
            label.update_positions(renderer)
            box = plt.Text.get_window_extent(label, renderer=renderer).padded(2)
            if (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in obstacles)):
                obstacles.append(box)
                break
            label.remove()
        else:
            plt.close(figure)
            raise RuntimeError(f"Could not place a nonoverlapping label for {row.Name}")
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)
    return output_path


def write_rank_correlation_analysis(
    output_dir: Path,
    frame: pd.DataFrame,
    predictions: Sequence[float],
    campaign_round: str,
    interval_lower: Sequence[float],
    interval_upper: Sequence[float],
    interval_metadata: dict[str, Any],
    score_model: dict[str, Any] | None,
    reference_means: dict[str, float] | None,
    score_scale_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write reviewer-oriented posterior statistics, figures, and provenance."""
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = build_measurement_summary(
        frame,
        predictions,
        interval_lower,
        interval_upper,
        str(interval_metadata.get("status", "unknown")),
        score_model,
        reference_means,
    )
    statistics_frame = compute_rank_statistics(summary, campaign_round)

    summary_path = output_dir / "measurement_summary.csv"
    statistics_path = output_dir / "rank_correlation_statistics.csv"
    rank_plot_path = output_dir / "score_rank_comparison.png"
    scatter_plot_path = output_dir / "predicted_vs_experimental_score.png"
    manifest_path = output_dir / "analysis_manifest.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8", lineterminator="\n")
    statistics_frame.to_csv(
        statistics_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    written_rank_plot = _plot_rank_comparison(
        summary, statistics_frame, rank_plot_path, score_scale_metadata.get("prediction_model")
    )
    written_scatter_plot = _plot_score_scatter(
        summary, statistics_frame, scatter_plot_path,
        score_scale_metadata.get("prediction_model"),
    )

    artifacts = {
        "measurement_summary": summary_path,
        "rank_correlation_statistics": statistics_path,
    }
    if written_rank_plot is not None:
        artifacts["rank_plot"] = written_rank_plot
    if written_scatter_plot is not None:
        artifacts["score_scatter_plot"] = written_scatter_plot
    manifest = {
        "schema_version": 3,
        "analysis_version": ANALYSIS_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluated_sample_round": campaign_round,
        "measurement_columns": list(POSTERIOR_MEASUREMENT_COLUMNS),
        "plqy_unit": "percent; numeric values are percentage points and are not rescaled",
        "xrd_unit": "dimensionless ratio",
        "replicate_summary": {
            "expected_replicates": 3,
            "standard_deviation_ddof": 1,
            "confidence_interval": "two-sided 95% Student t interval for the mean",
        },
        "rank_statistics": {
            "statistical_unit": "unique molecule Name in the selected campaign",
            "primary": "Spearman rho",
            "robustness": "Kendall tau-b",
            "alternative": "two-sided",
            "exact_permutation_max_n": EXACT_PERMUTATION_MAX_N,
            "monte_carlo_permutations": MONTE_CARLO_PERMUTATIONS,
            "monte_carlo_random_state": PERMUTATION_RANDOM_STATE,
            "comparison": "predicted Score vs same-round experimental Score",
        },
        "experimental_score": score_scale_metadata,
        "score_prediction_interval": interval_metadata,
        "artifacts": {
            key: {
                "file": path.name,
                "sha256": file_sha256(path),
            }
            for key, path in artifacts.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "statistics": statistics_frame,
        "paths": {**artifacts, "manifest": manifest_path},
        "manifest": manifest,
    }
