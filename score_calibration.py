"""Versioned construction of the experiment-calibrated target ``Score``.

The module deliberately keeps score construction separate from model training.
It reads immutable repository inputs, fits the six univariate calibrations used
by the paper method, and writes a complete, versioned artifact below
``outputs/score_calibration``.  Importers should use :func:`ensure_score_artifact`
instead of reading a previously injected ``Score`` column from an input CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    is_campaign_at_or_before,
    normalize_campaign_round,
    resolve_campaign_round,
)
from project_paths import (
    PHYSICOCHEMICAL_DATA_FILE,
    PROJECT_ROOT,
    SCORE_CALIBRATION_DATA_FILE,
    SCORE_CALIBRATION_OUTPUT_DIR,
    require_input_file,
)


ALGORITHM_VERSION = "score-calibration-v3"
SCORE_COLUMN = "Score"
DESCRIPTORS = ("Binding_energy", "Transferred_charges", "Vbur")
ENDPOINTS = ("PLQY", "XRD")
REPLICATES = (1, 2, 3)

CALIBRATION_COLUMNS = (
    "Round",
    "Name",
    "Use_for_fit",
    "Binding_energy_eV",
    "Transferred_charges_e",
    "Vbur_percent",
    "PLQY_1",
    "PLQY_2",
    "PLQY_3",
    "XRD_ratio_1",
    "XRD_ratio_2",
    "XRD_ratio_3",
)

CALIBRATION_TO_MODEL_DESCRIPTOR = {
    "Binding_energy_eV": "Binding_energy",
    "Transferred_charges_e": "Transferred_charges",
    "Vbur_percent": "Vbur",
}

REFERENCE_SERIES_NAME = "Ref-PbI2"
REFERENCE_SERIES_COLOR = "#606060"
SERIES_COLORMAP = "viridis"
SERIES_COLORMAP_RANGE = (0.08, 0.92)

DESCRIPTOR_PLOT_CONFIG = {
    "Binding_energy": {
        "label": "Binding energy (eV)",
        "line_color": "#46ADD6",
        "xlim": (-2.0, 0.0),
        "xticks": np.arange(-2.0, 0.01, 0.4),
    },
    "Transferred_charges": {
        "label": "Number of transferred charges (e)",
        "line_color": "#638B59",
        "xlim": (0.04, 0.15),
        "xticks": np.arange(0.04, 0.151, 0.02),
    },
    "Vbur": {
        "label": r"$V_{\mathrm{bur}}$ (%)",
        "line_color": "#C55360",
        "xlim": (48.0, 70.0),
        "xticks": np.arange(50.0, 70.1, 5.0),
    },
}

ARTIFACT_FILENAMES = {
    "scores_file": "scores.csv",
    "fits_file": "fits.csv",
    "summary_file": "summary.json",
    "model_file": "model.json",
    "normalized_calibration_file": "normalized_calibration.csv",
    "batch_scores_file": "batch_scores.csv",
    "batch_agreement_file": "batch_agreement.csv",
    "plqy_binding_energy_plot": "plqy_binding_energy_correlation.png",
    "plqy_transferred_charges_plot": "plqy_transferred_charges_correlation.png",
    "plqy_vbur_plot": "plqy_vbur_correlation.png",
    "xrd_binding_energy_plot": "xrd_binding_energy_correlation.png",
    "xrd_transferred_charges_plot": "xrd_transferred_charges_correlation.png",
    "xrd_vbur_plot": "xrd_vbur_correlation.png",
}

PLOT_FILENAMES = {
    "PLQY": {
        "Binding_energy": ARTIFACT_FILENAMES["plqy_binding_energy_plot"],
        "Transferred_charges": ARTIFACT_FILENAMES[
            "plqy_transferred_charges_plot"
        ],
        "Vbur": ARTIFACT_FILENAMES["plqy_vbur_plot"],
    },
    "XRD": {
        "Binding_energy": ARTIFACT_FILENAMES["xrd_binding_energy_plot"],
        "Transferred_charges": ARTIFACT_FILENAMES[
            "xrd_transferred_charges_plot"
        ],
        "Vbur": ARTIFACT_FILENAMES["xrd_vbur_plot"],
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value cannot be written to a score artifact")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    name = str(value).strip().replace("\\", "")
    if re.fullmatch(r"Ref-PbI-?\(?2\)?", name, flags=re.IGNORECASE):
        return "Ref-PbI2"
    return name


def _is_reference_name(name: str) -> bool:
    return _normalise_name(name).casefold() == "ref-pbi2"


def _parse_use_for_fit(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", ""}
    result: list[bool] = []
    for raw in series:
        if pd.isna(raw):
            result.append(False)
            continue
        text = str(raw).strip().casefold()
        if text in true_values:
            result.append(True)
        elif text in false_values:
            result.append(False)
        else:
            raise ValueError(f"Unsupported Use_for_fit value: {raw!r}")
    return pd.Series(result, index=series.index, dtype=bool)


def _select_canonical_calibration(
    table: pd.DataFrame,
    campaign_round: str,
) -> pd.DataFrame:
    """Validate and select cumulative fit rows through one campaign."""
    missing = [column for column in CALIBRATION_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            "calibration.csv is missing required columns: " + ", ".join(missing)
        )
    measurement_columns = list(CALIBRATION_COLUMNS[-6:])
    if list(table.columns[-6:]) != measurement_columns:
        raise ValueError(
            "The final six calibration.csv columns must be: "
            + ", ".join(measurement_columns)
        )

    selected = table.loc[:, CALIBRATION_COLUMNS].copy()
    selected["Use_for_fit"] = _parse_use_for_fit(selected["Use_for_fit"])
    selected["Name"] = selected["Name"].map(_normalise_name)
    if selected["Round"].isna().any() or selected["Round"].astype(str).str.strip().eq("").any():
        raise ValueError("calibration.csv contains a missing or blank Round")
    selected["Round"] = selected["Round"].map(normalize_campaign_round)
    selected = selected.loc[
        selected["Round"].map(
            lambda value: is_campaign_at_or_before(value, campaign_round)
        )
    ].copy()
    selected = selected.loc[selected["Use_for_fit"]].copy()
    if selected.empty:
        raise ValueError(
            f"calibration.csv contains no cumulative Use_for_fit=true rows through "
            f"{campaign_round}"
        )
    if selected["Name"].eq("").any():
        raise ValueError("Calibration rows used for fitting must have a Name")
    duplicate_mask = selected.duplicated(["Round", "Name"], keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, ["Round", "Name"]].drop_duplicates()
        raise ValueError(
            "calibration.csv contains duplicate Round/Name rows: "
            + ", ".join(
                f"{row.Round}/{row.Name}" for row in duplicates.itertuples(index=False)
            )
        )

    numeric_columns = [
        "Binding_energy_eV",
        "Transferred_charges_e",
        "Vbur_percent",
        *(f"PLQY_{replicate}" for replicate in REPLICATES),
        *(f"XRD_ratio_{replicate}" for replicate in REPLICATES),
    ]
    for column in numeric_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Calibration column contains non-finite values: {column}")

    return selected.reset_index(drop=True)


def _expand_selected_calibration(selected: pd.DataFrame) -> pd.DataFrame:
    """Expand a validated compound table into the three replicate observations."""

    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        for replicate in REPLICATES:
            records.append(
                {
                    "Round": str(row["Round"]),
                    "Name": row["Name"],
                    "Use_for_fit": True,
                    "Replicate": replicate,
                    "Binding_energy": float(row["Binding_energy_eV"]),
                    "Transferred_charges": float(row["Transferred_charges_e"]),
                    "Vbur": float(row["Vbur_percent"]),
                    "PLQY": float(row[f"PLQY_{replicate}"]),
                    "XRD": float(row[f"XRD_ratio_{replicate}"]),
                }
            )
    return pd.DataFrame.from_records(records)


def _expand_canonical_calibration(
    table: pd.DataFrame,
    campaign_round: str | None = None,
) -> pd.DataFrame:
    """Validate and expand cumulative calibration rows for one campaign."""
    selected_round = (
        resolve_campaign_round()
        if campaign_round is None
        else normalize_campaign_round(campaign_round)
    )
    selected = _select_canonical_calibration(table, selected_round)
    return _expand_selected_calibration(selected)


def _calibration_subset_sha256(selected: pd.DataFrame) -> str:
    payload = selected.loc[:, CALIBRATION_COLUMNS].to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_calibration(
    path: Path,
    campaign_round: str,
) -> tuple[pd.DataFrame, str, str]:
    table = pd.read_csv(path)
    table = table.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    empty_unnamed = [column for column in table.columns
                     if str(column).startswith("Unnamed:") and table[column].isna().all()]
    table = table.drop(columns=empty_unnamed)
    selected = _select_canonical_calibration(table, campaign_round)
    return (
        _expand_selected_calibration(selected),
        "canonical-single-table-cumulative-campaign",
        _calibration_subset_sha256(selected),
    )


def _read_physicochemical_library(path: Path) -> pd.DataFrame:
    library = pd.read_csv(path)
    if SCORE_COLUMN in library.columns:
        raise ValueError(
            "data.csv must not store Score; it is generated from calibration.csv"
        )
    required = {"Name", *DESCRIPTORS}
    missing = sorted(required.difference(library.columns))
    if missing:
        raise ValueError(f"data.csv is missing required columns: {missing}")
    if library["Name"].duplicated().any():
        duplicates = library.loc[library["Name"].duplicated(), "Name"].tolist()
        raise ValueError(f"data.csv contains duplicate Name values: {duplicates}")
    if library["Name"].isna().any() or library["Name"].astype(str).str.strip().eq("").any():
        raise ValueError("data.csv contains a missing or blank Name")
    for descriptor in DESCRIPTORS:
        library[descriptor] = pd.to_numeric(library[descriptor], errors="raise")
        if not np.isfinite(library[descriptor].to_numpy(dtype=float)).all():
            raise ValueError(f"data.csv contains non-finite {descriptor} values")
    return library


def _apply_latest_library_descriptors(
    calibration: pd.DataFrame,
    library: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    corrected = calibration.copy()
    lookup = library.set_index("Name")
    corrections: list[dict[str, Any]] = []

    for (round_name, name), indices in corrected.groupby(
        ["Round", "Name"], sort=False
    ).groups.items():
        old_values = {
            descriptor: float(corrected.loc[indices, descriptor].iloc[0])
            for descriptor in DESCRIPTORS
        }
        if name in lookup.index:
            latest_values = {
                descriptor: float(lookup.at[name, descriptor])
                for descriptor in DESCRIPTORS
            }
            for descriptor, value in latest_values.items():
                corrected.loc[indices, descriptor] = value
            corrections.append(
                {
                    "Round": str(round_name),
                    "Name": name,
                    "descriptor_source": "data/data.csv",
                    "original": old_values,
                    "used": latest_values,
                    "changed": any(
                        not math.isclose(
                            old_values[key], latest_values[key], rel_tol=0, abs_tol=1e-12
                        )
                        for key in DESCRIPTORS
                    ),
                }
            )
        elif _is_reference_name(name):
            corrections.append(
                {
                    "Round": str(round_name),
                    "Name": name,
                    "descriptor_source": "data/calibration.csv",
                    "original": old_values,
                    "used": old_values,
                    "changed": False,
                }
            )
        else:
            raise ValueError(
                f"Calibration compound {name!r} is absent from data.csv. "
                "Only Ref-PbI2 may use descriptors stored solely in calibration.csv."
            )
    return corrected, corrections


def _normalize_round_reference(calibration: pd.DataFrame) -> pd.DataFrame:
    """Scale each endpoint to 100 times its same-round reference mean."""
    normalized = calibration.copy()
    for endpoint in ENDPOINTS:
        normalized[f"Raw_{endpoint}"] = calibration[endpoint]
    for round_name, rows in normalized.groupby("Round", sort=False):
        reference = rows.loc[rows["Name"].eq(REFERENCE_SERIES_NAME)]
        if len(reference) != len(REPLICATES) or set(reference["Replicate"]) != set(REPLICATES):
            raise ValueError(f"{round_name} requires three fitted Ref-PbI2 replicates")
        for endpoint in ENDPOINTS:
            values = reference[f"Raw_{endpoint}"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{round_name} Ref-PbI2 {endpoint} must be finite and positive")
            denominator = float(values.mean())
            normalized.loc[rows.index, f"Reference_mean_{endpoint}"] = denominator
            normalized.loc[rows.index, endpoint] = 100.0 * rows[f"Raw_{endpoint}"] / denominator
    return normalized


def _fit_univariate_models(calibration: pd.DataFrame) -> pd.DataFrame:
    fit_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        y = calibration[endpoint].to_numpy(dtype=float)
        if np.unique(y).size < 2:
            raise ValueError(f"Calibration endpoint has no variance: {endpoint}")
        for descriptor in DESCRIPTORS:
            x = calibration[descriptor].to_numpy(dtype=float)
            if np.unique(x).size < 2:
                raise ValueError(f"Calibration descriptor has no variance: {descriptor}")
            slope, intercept = np.polyfit(x, y, deg=1)
            prediction = slope * x + intercept
            residual_sum = float(np.sum(np.square(y - prediction)))
            total_sum = float(np.sum(np.square(y - y.mean())))
            r2 = 1.0 - residual_sum / total_sum
            fit_rows.append(
                {
                    "endpoint": endpoint,
                    "descriptor": descriptor,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": float(r2),
                    "n_observations": int(len(x)),
                }
            )

    fits = pd.DataFrame(fit_rows)
    fits["normalized_weight"] = 0.0
    for endpoint in ENDPOINTS:
        mask = fits["endpoint"].eq(endpoint)
        confidence = fits.loc[mask, "r2"].clip(lower=0.0)
        total = float(confidence.sum())
        if total <= 0:
            raise ValueError(f"All R2 confidence weights are zero for {endpoint}")
        fits.loc[mask, "normalized_weight"] = confidence / total
    return fits


def _predict_endpoint(
    frame: pd.DataFrame,
    fits: pd.DataFrame,
    endpoint: str,
) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=float)
    endpoint_fits = fits.loc[fits["endpoint"].eq(endpoint)]
    for row in endpoint_fits.itertuples(index=False):
        x = frame[row.descriptor].to_numpy(dtype=float)
        prediction += row.normalized_weight * (row.slope * x + row.intercept)
    return prediction


def _calculate_scores(
    library: pd.DataFrame,
    fits: pd.DataFrame,
    score_version: str,
    campaign_round: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = library.loc[:, ["Name", *DESCRIPTORS]].copy()
    endpoint_scaling: dict[str, dict[str, float]] = {}

    for endpoint in ENDPOINTS:
        prediction = _predict_endpoint(scores, fits, endpoint)
        mean = float(np.mean(prediction))
        standard_deviation = float(np.std(prediction, ddof=0))
        if standard_deviation <= 0:
            raise ValueError(f"Predicted {endpoint} has zero population standard deviation")
        scores[f"Predicted_{endpoint}"] = prediction
        scores[f"z_{endpoint}"] = (prediction - mean) / standard_deviation
        endpoint_scaling[endpoint] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "ddof": 0,
        }

    scores[SCORE_COLUMN] = scores["z_PLQY"] - scores["z_XRD"]
    scores["score_version"] = score_version

    model_fits = fits.to_dict(orient="records")
    combined_coefficients = {descriptor: 0.0 for descriptor in DESCRIPTORS}
    combined_intercept = 0.0
    signs = {"PLQY": 1.0, "XRD": -1.0}
    for endpoint in ENDPOINTS:
        scaling = endpoint_scaling[endpoint]
        sign = signs[endpoint]
        endpoint_fits = fits.loc[fits["endpoint"].eq(endpoint)]
        endpoint_intercept = 0.0
        for row in endpoint_fits.itertuples(index=False):
            combined_coefficients[row.descriptor] += (
                sign
                * row.normalized_weight
                * row.slope
                / scaling["standard_deviation"]
            )
            endpoint_intercept += row.normalized_weight * row.intercept
        combined_intercept += (
            sign
            * (endpoint_intercept - scaling["mean"])
            / scaling["standard_deviation"]
        )

    model = {
        "algorithm_version": ALGORITHM_VERSION,
        "score_version": score_version,
        "campaign_round": campaign_round,
        "descriptors": list(DESCRIPTORS),
        "endpoints": list(ENDPOINTS),
        "fits": model_fits,
        "endpoint_scaling": endpoint_scaling,
        "score_definition": "z_PLQY - z_XRD",
        "endpoint_units": "percent of same-round Ref-PbI2 mean (reference = 100)",
        "score_coefficients": combined_coefficients,
        "score_intercept": combined_intercept,
    }
    return scores, model


def _place_fit_annotation(axis, label, x, y, line_x, line_y):
    """Place a measured text box clear of markers and the fitted line."""
    from matplotlib.path import Path as PlotPath
    from matplotlib.transforms import Bbox

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    extent = label.get_window_extent(renderer)
    bounds = axis.get_window_extent(renderer)
    padding = 5 * figure.dpi / 72
    radius = (np.sqrt(44) / 2 + 2) * figure.dpi / 72
    points = axis.transData.transform(np.column_stack([x, y]))
    curve = PlotPath(axis.transData.transform(np.column_stack([line_x, line_y])))
    width, height = extent.width, extent.height
    if width + 2 * padding < bounds.width and height + 2 * padding < bounds.height:
        for cy in np.linspace(bounds.y1 - height / 2 - padding,
                              bounds.y0 + height / 2 + padding, 15):
            for cx in np.linspace(bounds.x0 + width / 2 + padding,
                                  bounds.x1 - width / 2 - padding, 15):
                box = Bbox.from_bounds(cx - width / 2 - padding,
                                       cy - height / 2 - padding,
                                       width + 2 * padding, height + 2 * padding)
                blocked = ((points[:, 0] >= box.x0 - radius)
                           & (points[:, 0] <= box.x1 + radius)
                           & (points[:, 1] >= box.y0 - radius)
                           & (points[:, 1] <= box.y1 + radius)).any()
                if not blocked and not curve.intersects_bbox(box, filled=False):
                    label.set_position(axis.transAxes.inverted().transform((cx, cy)))
                    return
    # Dense plots retain readable text above the axes, included by tight export.
    label.set_position((0.5, 1.04))
    label.set_verticalalignment("bottom")
    label.set_clip_on(False)


def _format_equation(slope: float, intercept: float) -> str:
    return f"y={slope:.5g}x{intercept:+.5g}"


def _series_palette(names: Iterable[str]) -> dict[str, str]:
    ordered = list(dict.fromkeys(names))
    palette: dict[str, str] = {}
    if REFERENCE_SERIES_NAME in ordered:
        palette[REFERENCE_SERIES_NAME] = REFERENCE_SERIES_COLOR

    series_names = [name for name in ordered if name != REFERENCE_SERIES_NAME]
    if series_names:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        color_scale = plt.get_cmap(SERIES_COLORMAP)
        if len(series_names) == 1:
            color_positions = np.array([0.5])
        else:
            color_positions = np.linspace(
                SERIES_COLORMAP_RANGE[0],
                SERIES_COLORMAP_RANGE[1],
                num=len(series_names),
            )
        palette.update(
            {
                name: to_hex(color_scale(position), keep_alpha=False)
                for name, position in zip(series_names, color_positions)
            }
        )
    return palette


def _plot_correlation_panel(
    calibration: pd.DataFrame,
    fits: pd.DataFrame,
    endpoint: str,
    descriptor: str,
    png_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    is_plqy = endpoint == "PLQY"
    figure, axis = plt.subplots(figsize=(6.2, 2.55))
    figure.subplots_adjust(
        left=0.20,
        right=0.77,
        top=0.96,
        bottom=0.28,
    )

    names = calibration["Name"].drop_duplicates().tolist()
    palette = _series_palette(names)
    display_labels = {"Ref-PbI2": r"Ref-PbI$_2$"}
    config = DESCRIPTOR_PLOT_CONFIG[descriptor]

    for name in names:
        subset = calibration.loc[calibration["Name"].eq(name)]
        color = palette[name]
        axis.scatter(
            subset[descriptor],
            subset[endpoint],
            marker="o",
            s=44,
            facecolor=color,
            edgecolor=color,
            linewidth=0.4,
            alpha=0.94,
            zorder=3,
        )
        axis.scatter(
            subset[descriptor],
            subset[endpoint],
            marker="o",
            s=7,
            facecolor="white",
            edgecolor="none",
            alpha=0.45,
            zorder=4,
        )

    fit = fits.loc[
        fits["endpoint"].eq(endpoint) & fits["descriptor"].eq(descriptor)
    ].iloc[0]
    x_values = calibration[descriptor].to_numpy(dtype=float)
    line_x = np.linspace(float(x_values.min()), float(x_values.max()), 200)
    line_y = fit["slope"] * line_x + fit["intercept"]
    axis.plot(
        line_x,
        line_y,
        color=config["line_color"],
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    equation = _format_equation(float(fit["slope"]), float(fit["intercept"]))
    fit_annotation = axis.text(
        0.52,
        0.84,
        equation + "\n" + f"R\N{SUPERSCRIPT TWO}={fit['r2']:.5g}",
        transform=axis.transAxes,
        color=config["line_color"],
        fontsize=9.5,
        ha="center",
        va="center",
    )

    default_low, default_high = config["xlim"]
    data_range = max(float(x_values.max() - x_values.min()), 1e-12)
    x_low = min(default_low, float(x_values.min() - 0.04 * data_range))
    x_high = max(default_high, float(x_values.max() + 0.04 * data_range))
    axis.set_xlim(x_low, x_high)
    if x_low == default_low and x_high == default_high:
        axis.set_xticks(config["xticks"])
    axis.set_xlabel(config["label"], labelpad=8)

    endpoint_values = calibration[endpoint].to_numpy(dtype=float)
    if is_plqy:
        y_high = max(0.065, float(endpoint_values.max() * 1.08))
        axis.set_ylim(0.0, y_high)
        if y_high <= 0.065:
            axis.set_yticks([0.00, 0.02, 0.04, 0.06])
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.set_ylabel("Relative PLQY (%)", labelpad=6)
    else:
        y_high = max(0.35, float(endpoint_values.max() * 1.08))
        axis.set_ylim(0.0, y_high)
        if y_high <= 0.35:
            axis.set_yticks([0.0, 0.1, 0.2, 0.3])
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_ylabel(
            "Relative XRD ratio (%)",
            labelpad=7,
        )

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="major", direction="out", length=4, width=0.9)
    axis.tick_params(which="minor", direction="out", length=2.2, width=0.8)
    axis.grid(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[name],
            markeredgecolor=palette[name],
            markersize=6.2,
            label=display_labels.get(name, name),
        )
        for name in names
    ]
    # Measure candidate layouts so longer names and future rounds wrap naturally.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_box = axis.get_window_extent(renderer)
    xlabel_box = axis.xaxis.label.get_window_extent(renderer)
    # Include the y-axis label/ticks in the horizontal footprint of the image.
    content_box = axis.get_tightbbox(renderer)
    legend_left = content_box.x0
    legend_right = max(axis_box.x1, content_box.x1)
    legend_width = legend_right - legend_left
    legend_anchor = figure.transFigure.inverted().transform(
        ((legend_left + legend_right) / 2,
         xlabel_box.y0 - 9 * figure.dpi / 72)
    )
    for columns in range(min(8, len(legend_handles)), 0, -1):
        # Matplotlib fills columns first; reorder to display source order by row.
        ordered_handles = [
            legend_handles[index]
            for column in range(columns)
            for index in range(column, len(legend_handles), columns)
        ]
        legend = axis.legend(
            handles=ordered_handles,
            ncol=columns,
            loc="upper center",
            bbox_to_anchor=legend_anchor,
            bbox_transform=figure.transFigure,
            frameon=False,
            borderaxespad=0,
            fontsize=7.5,
            markerscale=0.8,
            handlelength=0.9,
            handletextpad=0.3,
            columnspacing=0.4,
            labelspacing=0.35,
        )
        if legend.get_window_extent(renderer).width <= legend_width or columns == 1:
            legend.remove()
            left_fraction = figure.transFigure.inverted().transform((legend_left, 0))[0]
            legend = axis.legend(
                handles=ordered_handles, ncol=columns, loc="upper left",
                bbox_to_anchor=(left_fraction, legend_anchor[1],
                                legend_width / figure.bbox.width, 0),
                bbox_transform=figure.transFigure, mode="expand",
                frameon=False, borderaxespad=0, borderpad=0,
                fontsize=7.5, markerscale=0.8, handlelength=0.9,
                handletextpad=0.3, columnspacing=0.4, labelspacing=0.35,
            )
            break
        legend.remove()

    _place_fit_annotation(
        axis, fit_annotation, x_values, endpoint_values, line_x, line_y
    )
    figure.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def _artifact_is_complete(
    artifact_dir: Path,
    score_version: str,
    campaign_round: str,
) -> bool:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("score_version") != score_version:
        return False
    if manifest.get("campaign_round") != campaign_round:
        return False
    expected_hashes = manifest.get("artifact_sha256", {})
    for filename, expected in expected_hashes.items():
        path = artifact_dir / filename
        if not path.is_file() or _file_sha256(path) != expected:
            return False
    for filename in ARTIFACT_FILENAMES.values():
        path = artifact_dir / filename
        if not path.is_file() or expected_hashes.get(filename) != _file_sha256(path):
            return False
    return True


def _artifact_info(
    artifact_dir: Path,
    manifest: dict[str, Any],
    generated: bool,
) -> dict[str, Any]:
    info = {
        "score_version": manifest["score_version"],
        "score_output_dir": str(artifact_dir),
        "scores_file": str(artifact_dir / ARTIFACT_FILENAMES["scores_file"]),
        "manifest_path": str(artifact_dir / "manifest.json"),
        "campaign_round": manifest["campaign_round"],
        "data_sha256": manifest["data_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
        "calibration_subset_sha256": manifest["calibration_subset_sha256"],
        "algorithm_version": manifest["algorithm_version"],
        "implementation_sha256": manifest["implementation_sha256"],
        "fits_file": str(artifact_dir / ARTIFACT_FILENAMES["fits_file"]),
        "summary_file": str(artifact_dir / ARTIFACT_FILENAMES["summary_file"]),
        "model_file": str(artifact_dir / ARTIFACT_FILENAMES["model_file"]),
        "generated": generated,
    }
    return info


def ensure_score_artifact(
    force: bool = False,
    campaign_round: str | None = None,
) -> dict[str, Any]:
    """Create or reuse the cumulative Score artifact for one campaign."""
    selected_round = resolve_campaign_round(campaign_round)
    calibration_path = require_input_file(SCORE_CALIBRATION_DATA_FILE)
    data_path = require_input_file(PHYSICOCHEMICAL_DATA_FILE)
    calibration_sha256 = _file_sha256(calibration_path)
    calibration, calibration_schema, calibration_subset_sha256 = _read_calibration(
        calibration_path,
        selected_round,
    )
    data_sha256 = _file_sha256(data_path)
    implementation_sha256 = _file_sha256(Path(__file__))

    fingerprint_payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "campaign_round": selected_round,
        "implementation_file": "score_calibration.py",
        "implementation_sha256": implementation_sha256,
        "calibration_subset_sha256": calibration_subset_sha256,
        "data_sha256": data_sha256,
        "replicates": list(REPLICATES),
        "fit": "six univariate ordinary least-squares regressions independently per Round",
        "batch_aggregation": "equal mean after independent endpoint Z-scoring on the same library; no final rescaling",
        "weighting": "endpoint-specific nonnegative R2 normalized to one",
        "normalization": "208-row physicochemical library population Z-score, ddof=0",
        "experimental_normalization": "100 * raw endpoint / same-round Ref-PbI2 mean",
        "score_definition": "z_PLQY - z_XRD",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    score_version = f"{ALGORITHM_VERSION}-{fingerprint[:12]}"
    artifact_dir = (
        SCORE_CALIBRATION_OUTPUT_DIR
        / campaign_directory_name(selected_round)
        / score_version
    )

    if not force and _artifact_is_complete(
        artifact_dir,
        score_version,
        selected_round,
    ):
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        return _artifact_info(artifact_dir, manifest, generated=False)

    library = _read_physicochemical_library(data_path)
    calibration, descriptor_sources = _apply_latest_library_descriptors(
        calibration, library
    )
    calibration = _normalize_round_reference(calibration)
    batch_models, batch_tables, batch_fits = [], [], []
    for batch_round, batch in calibration.groupby("Round", sort=True):
        fit = _fit_univariate_models(batch)
        table, batch_model = _calculate_scores(library, fit, score_version, batch_round)
        table["Round"] = batch_round
        batch_tables.append(table)
        batch_models.append(batch_model)
        batch_fits.append(fit.assign(Round=batch_round))
    fits = pd.concat(batch_fits, ignore_index=True)
    scores = library.loc[:, ["Name", *DESCRIPTORS]].copy()
    for column in ["Predicted_PLQY", "Predicted_XRD", "z_PLQY", "z_XRD", SCORE_COLUMN]:
        scores[column] = np.mean([table.set_index("Name").loc[scores.Name, column].to_numpy()
                                  for table in batch_tables], axis=0)
    scores["score_version"] = score_version
    model = {
        "algorithm_version": ALGORITHM_VERSION, "score_version": score_version,
        "campaign_round": selected_round, "batch_models": batch_models,
        "batch_weights": {item["campaign_round"]: 1 / len(batch_models) for item in batch_models},
        "score_definition": "equal mean of independently library-standardized batch Scores; no final rescaling",
        "endpoint_units": "Predicted endpoints: mean relative percent; z endpoints: mean batch library Z-score",
        "score_coefficients": {d: float(np.mean([m["score_coefficients"][d] for m in batch_models])) for d in DESCRIPTORS},
        "score_intercept": float(np.mean([m["score_intercept"] for m in batch_models])),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    scores_path = artifact_dir / ARTIFACT_FILENAMES["scores_file"]
    fits_path = artifact_dir / ARTIFACT_FILENAMES["fits_file"]
    summary_path = artifact_dir / ARTIFACT_FILENAMES["summary_file"]
    model_path = artifact_dir / ARTIFACT_FILENAMES["model_file"]

    scores.to_csv(scores_path, index=False, encoding="utf-8", float_format="%.12g")
    fits.to_csv(fits_path, index=False, encoding="utf-8", float_format="%.12g")
    pd.concat(batch_tables, ignore_index=True).to_csv(
        artifact_dir / ARTIFACT_FILENAMES["batch_scores_file"], index=False, encoding="utf-8")
    from scipy.stats import spearmanr, kendalltau
    agreement = []
    for i, first in enumerate(batch_tables):
        for second in batch_tables[i + 1:]:
            aligned = second.set_index("Name").loc[first.Name, SCORE_COLUMN]
            agreement.append({"Round_A": first.Round.iloc[0], "Round_B": second.Round.iloc[0],
                              "n_library": len(first), "Spearman_rho": spearmanr(first.Score, aligned).statistic,
                              "Kendall_tau_b": kendalltau(first.Score, aligned).statistic})
    pd.DataFrame(agreement, columns=["Round_A", "Round_B", "n_library", "Spearman_rho", "Kendall_tau_b"]).to_csv(
        artifact_dir / ARTIFACT_FILENAMES["batch_agreement_file"], index=False, encoding="utf-8")
    calibration.to_csv(
        artifact_dir / ARTIFACT_FILENAMES["normalized_calibration_file"],
        index=False, encoding="utf-8", float_format="%.12g",
    )

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "score_version": score_version,
        "campaign_round": selected_round,
        "calibration_schema": calibration_schema,
        "calibration_compounds": calibration["Name"].drop_duplicates().tolist(),
        "calibration_rounds": calibration["Round"].drop_duplicates().tolist(),
        "fit_compound_round_rows": int(
            calibration.loc[:, ["Round", "Name"]].drop_duplicates().shape[0]
        ),
        "expanded_fit_observations": int(len(calibration)),
        "library_rows": int(len(library)),
        "descriptor_sources": descriptor_sources,
        "score_statistics": {
            "mean": float(scores[SCORE_COLUMN].mean()),
            "standard_deviation_ddof0": float(scores[SCORE_COLUMN].std(ddof=0)),
            "minimum": float(scores[SCORE_COLUMN].min()),
            "maximum": float(scores[SCORE_COLUMN].max()),
        },
    }
    _write_json(summary_path, summary)
    _write_json(model_path, model)

    for batch_round, batch in calibration.groupby("Round", sort=True):
        batch_dir = artifact_dir / "batches" / batch_round
        batch_dir.mkdir(parents=True, exist_ok=True)
        for endpoint, descriptor_files in PLOT_FILENAMES.items():
            for descriptor, filename in descriptor_files.items():
                _plot_correlation_panel(batch, fits.loc[fits.Round.eq(batch_round)],
                                        endpoint, descriptor, batch_dir / filename)
                if batch_round == batch_models[-1]["campaign_round"]:
                    import shutil
                    shutil.copyfile(batch_dir / filename, artifact_dir / filename)

    artifact_hashes = {
        filename: _file_sha256(artifact_dir / filename)
        for filename in ARTIFACT_FILENAMES.values()
    }
    artifact_hashes.update({p.relative_to(artifact_dir).as_posix(): _file_sha256(p)
                            for p in (artifact_dir / "batches").rglob("*.png")})
    manifest = {
        "artifact_schema_version": 3,
        "algorithm_version": ALGORITHM_VERSION,
        "score_version": score_version,
        "campaign_round": selected_round,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(data_path.relative_to(PROJECT_ROOT).as_posix()),
        "calibration_file": str(calibration_path.relative_to(PROJECT_ROOT).as_posix()),
        "data_sha256": data_sha256,
        "calibration_sha256": calibration_sha256,
        "calibration_subset_sha256": calibration_subset_sha256,
        "implementation_file": "score_calibration.py",
        "implementation_sha256": implementation_sha256,
        "calibration_schema": calibration_schema,
        "method": fingerprint_payload,
        "artifact_files": dict(ARTIFACT_FILENAMES),
        "artifact_sha256": artifact_hashes,
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    return _artifact_info(artifact_dir, manifest, generated=True)


def load_score_table(info: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the versioned 208-row score table, generating it when absent."""
    artifact_info = info or ensure_score_artifact()
    scores = pd.read_csv(artifact_info["scores_file"])
    required = {"Name", *DESCRIPTORS, "Predicted_PLQY", "Predicted_XRD", SCORE_COLUMN}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Versioned scores file is missing columns: {missing}")
    return scores


def apply_score_model(
    frame: pd.DataFrame,
    info: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply a stored calibration model to rows containing the three descriptors."""
    artifact_info = info or ensure_score_artifact()
    model = json.loads(Path(artifact_info["model_file"]).read_text(encoding="utf-8"))
    if "batch_models" in model:
        predictions = [_apply_single_score_model(frame, item) for item in model["batch_models"]]
        return sum(predictions) / len(predictions)
    return _apply_single_score_model(frame, model)


def _apply_single_score_model(frame: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Apply one batch using its saved, fixed library normalization constants."""
    missing = sorted(set(DESCRIPTORS).difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot apply score model; missing descriptors: {missing}")

    result = pd.DataFrame(index=frame.index)
    for endpoint in ENDPOINTS:
        prediction = np.zeros(len(frame), dtype=float)
        for fit in model["fits"]:
            if fit["endpoint"] != endpoint:
                continue
            values = pd.to_numeric(frame[fit["descriptor"]], errors="raise").to_numpy(
                dtype=float
            )
            prediction += fit["normalized_weight"] * (
                fit["slope"] * values + fit["intercept"]
            )
        scaling = model["endpoint_scaling"][endpoint]
        result[f"Predicted_{endpoint}"] = prediction
        result[f"z_{endpoint}"] = (
            prediction - scaling["mean"]
        ) / scaling["standard_deviation"]
    result[SCORE_COLUMN] = result["z_PLQY"] - result["z_XRD"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a cumulative, versioned Score calibration artifact."
    )
    add_campaign_arguments(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the selected artifact even when it is complete.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_round = resolve_campaign_round(args.campaign_round)
    info = ensure_score_artifact(
        force=args.force,
        campaign_round=campaign_round,
    )
    action = "Generated" if info["generated"] else "Reused"
    print(f"Campaign: {info['campaign_round']}")
    print(f"{action} score artifact: {info['score_version']}")
    print(f"Artifact directory: {info['score_output_dir']}")
    print(f"Scores: {info['scores_file']}")
    print(f"Manifest: {info['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
