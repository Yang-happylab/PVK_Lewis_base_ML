"""Compare calibrated Score ranks with observed best normalized device PCE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from device_normalization import configure_plot_style, normalize_group_label
from project_paths import (
    DEVICE_ANALYSIS_OUTPUT_DIR,
    DEVICE_SCORE_VALIDATION_OUTPUT_DIR,
    PROJECT_ROOT,
    file_sha256,
)
from score_calibration import ensure_score_artifact
from rank_statistics import (
    EXACT_PERMUTATION_MAX_N,
    MONTE_CARLO_PERMUTATIONS,
    PERMUTATION_RANDOM_STATE,
    rank_correlations_with_permutation,
)


ANALYSIS_VERSION = "device-score-validation-v1"
PLOT_DPI = 300
PLOT_SIZE = (10.0, 8.0)
CONTROL_TOLERANCE = 1e-6
GROUP_SUMMARY_COLUMNS = (
    "Round",
    "Name",
    "Group",
    "n_devices",
    "PCE_mean",
    "PCE_sd",
)
NORMALIZATION_SUMMARY_COLUMNS = (
    "Round",
    "Name",
    "Control_mean_PCE",
)
SCORE_COLUMNS = ("Name", "Score", "score_version")
ARTIFACT_FILENAMES = (
    "best_concentration_summary.csv",
    "rank_correlation_statistics.csv",
    "score_pce_rank_comparison.png",
    "score_vs_best_normalized_pce.png",
)





def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            "Run device_normalization.py for the selected campaign first."
        )
    return path


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    description: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{description} is missing columns: {', '.join(missing)}")


def read_group_summary(path: Path, campaign_round: str) -> pd.DataFrame:
    """Load and validate the existing normalized per-group device summary."""
    frame = pd.read_csv(_require_file(path, "Device group summary"), encoding="utf-8-sig")
    _require_columns(frame, GROUP_SUMMARY_COLUMNS, "Device group summary")
    if frame.empty:
        raise ValueError("Device group summary contains no rows")

    result = frame.loc[:, list(GROUP_SUMMARY_COLUMNS)].copy()
    for column in ("Round", "Name", "Group"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Device group summary contains a blank {column}")
    if not result["Round"].eq(campaign_round).all():
        observed = sorted(result["Round"].unique().tolist())
        raise ValueError(
            f"Device group summary round mismatch: expected {campaign_round}, "
            f"found {observed}"
        )
    if result.duplicated(["Round", "Name", "Group"]).any():
        raise ValueError("Device group summary has duplicate Round/Name/Group rows")

    canonical_groups = result["Group"].map(normalize_group_label)
    if not canonical_groups.equals(result["Group"]):
        raise ValueError(
            "Device group summary contains noncanonical Group labels; rerun "
            "device_normalization.py"
        )

    numeric_devices = pd.to_numeric(result["n_devices"], errors="coerce")
    if numeric_devices.isna().any() or not np.isfinite(numeric_devices).all():
        raise ValueError("Device group summary contains invalid n_devices values")
    if (numeric_devices < 2).any() or not np.equal(
        numeric_devices.to_numpy(dtype=float),
        np.floor(numeric_devices.to_numpy(dtype=float)),
    ).all():
        raise ValueError("Every device group must contain at least two devices")
    result["n_devices"] = numeric_devices.astype(int)

    for column in ("PCE_mean", "PCE_sd"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Device group summary contains non-finite {column}")
    if (result["PCE_sd"] < 0).any():
        raise ValueError("Device group summary contains a negative PCE_sd")

    for molecule, subset in result.groupby("Name", sort=False):
        control = subset.loc[subset["Group"].eq("Control")]
        tested = subset.loc[~subset["Group"].eq("Control")]
        if len(control) != 1:
            raise ValueError(f"{molecule} must have exactly one Control summary row")
        if tested.empty:
            raise ValueError(f"{molecule} has no tested non-Control concentration")
        if not math.isclose(
            float(control.iloc[0]["PCE_mean"]),
            100.0,
            rel_tol=0.0,
            abs_tol=CONTROL_TOLERANCE,
        ):
            raise ValueError(f"{molecule} Control PCE mean is not 100")
    return result


def read_normalization_summary(
    path: Path,
    campaign_round: str,
    expected_names: set[str],
) -> pd.DataFrame:
    """Load the Control raw PCE means recorded during normalization."""
    frame = pd.read_csv(
        _require_file(path, "Device normalization summary"),
        encoding="utf-8-sig",
    )
    _require_columns(
        frame,
        NORMALIZATION_SUMMARY_COLUMNS,
        "Device normalization summary",
    )
    result = frame.loc[:, list(NORMALIZATION_SUMMARY_COLUMNS)].copy()
    result["Round"] = result["Round"].astype(str).str.strip()
    result["Name"] = result["Name"].astype(str).str.strip()
    if not result["Round"].eq(campaign_round).all():
        raise ValueError("Device normalization summary round mismatch")
    if result["Name"].eq("").any() or result["Name"].duplicated().any():
        raise ValueError("Device normalization summary has blank or duplicate Name values")
    observed_names = set(result["Name"])
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            "Device summary Name mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    result["Control_mean_PCE"] = pd.to_numeric(
        result["Control_mean_PCE"], errors="coerce"
    )
    control_values = result["Control_mean_PCE"].to_numpy(dtype=float)
    if not np.isfinite(control_values).all() or (control_values <= 0).any():
        raise ValueError("Control_mean_PCE values must be finite and positive")
    return result


def read_scores(
    path: Path,
    expected_names: set[str],
    score_version: str,
) -> pd.DataFrame:
    """Load the exact versioned Score rows for the device molecules."""
    frame = pd.read_csv(_require_file(path, "Versioned Score table"), encoding="utf-8-sig")
    _require_columns(frame, SCORE_COLUMNS, "Versioned Score table")
    result = frame.loc[:, list(SCORE_COLUMNS)].copy()
    result["Name"] = result["Name"].astype(str).str.strip()
    if result["Name"].eq("").any() or result["Name"].duplicated().any():
        raise ValueError("Versioned Score table has blank or duplicate Name values")
    selected = result.loc[result["Name"].isin(expected_names)].copy()
    missing = sorted(expected_names - set(selected["Name"]))
    if missing:
        raise ValueError(f"Device molecules missing from Score table: {missing}")
    selected["Score"] = pd.to_numeric(selected["Score"], errors="coerce")
    if not np.isfinite(selected["Score"].to_numpy(dtype=float)).all():
        raise ValueError("Selected Score values must be finite")
    versions = set(selected["score_version"].astype(str))
    if versions != {score_version}:
        raise ValueError(
            f"Score version mismatch: expected {score_version}, found {sorted(versions)}"
        )
    return selected


def build_best_concentration_summary(
    group_summary: pd.DataFrame,
    normalization_summary: pd.DataFrame,
    scores: pd.DataFrame,
    campaign_round: str,
    score_version: str,
) -> pd.DataFrame:
    """Select each molecule's highest observed non-Control mean PCE."""
    tested = group_summary.loc[~group_summary["Group"].eq("Control")].copy()
    tested["concentration_percent"] = tested["Group"].str[:-1].astype(float)
    tested = tested.sort_values(
        ["Name", "PCE_mean", "concentration_percent"],
        ascending=[True, False, True],
        kind="stable",
    )
    best = tested.drop_duplicates("Name", keep="first").copy()
    best = best.merge(
        normalization_summary.loc[:, ["Name", "Control_mean_PCE"]],
        on="Name",
        how="left",
        validate="one_to_one",
    )
    best = best.merge(scores, on="Name", how="left", validate="one_to_one")

    best["PCE_sem"] = best["PCE_sd"] / np.sqrt(best["n_devices"])
    critical = stats.t.ppf(0.975, best["n_devices"] - 1)
    margin = critical * best["PCE_sem"]
    best["PCE_ci95_lower"] = best["PCE_mean"] - margin
    best["PCE_ci95_upper"] = best["PCE_mean"] + margin
    best["Raw_PCE_mean"] = (
        best["PCE_mean"] / 100.0 * best["Control_mean_PCE"]
    )
    best["PCE_vs_Control_points"] = best["PCE_mean"] - 100.0
    best["Above_Control_100"] = best["PCE_mean"] > 100.0
    best["Score_rank"] = best["Score"].rank(method="average", ascending=False)
    best["PCE_rank"] = best["PCE_mean"].rank(method="average", ascending=False)
    best["Rank_difference"] = best["PCE_rank"] - best["Score_rank"]
    best["Round"] = campaign_round
    best["score_version"] = score_version
    best = best.rename(
        columns={
            "Group": "Best_concentration",
            "concentration_percent": "Best_concentration_percent",
        }
    )
    columns = [
        "Round",
        "Name",
        "Score",
        "score_version",
        "Score_rank",
        "Best_concentration",
        "Best_concentration_percent",
        "n_devices",
        "PCE_mean",
        "PCE_sd",
        "PCE_sem",
        "PCE_ci95_lower",
        "PCE_ci95_upper",
        "Control_mean_PCE",
        "Raw_PCE_mean",
        "PCE_vs_Control_points",
        "Above_Control_100",
        "PCE_rank",
        "Rank_difference",
    ]
    return best.loc[:, columns].sort_values(
        ["Score_rank", "Name"], kind="stable"
    ).reset_index(drop=True)





def build_rank_statistics(
    summary: pd.DataFrame,
    campaign_round: str,
    score_version: str,
) -> pd.DataFrame:
    statistics = rank_correlations_with_permutation(
        summary["Score"], summary["PCE_mean"]
    )
    row = {
        "Round": campaign_round,
        "N_molecules": int(len(summary)),
        **statistics,
        "Score_version": score_version,
        "PCE_definition": (
            "highest arithmetic mean normalized PCE among non-Control concentrations"
        ),
    }
    return pd.DataFrame([row])


def _apply_axis_style(axis: plt.Axes) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        width=1.5,
        length=6,
        pad=7,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)


def _require_nonoverlapping_text(
    figure: plt.Figure,
    text_items: Sequence[plt.Text],
    description: str,
) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [plt.Text.get_window_extent(item, renderer=renderer) for item in text_items]
    for left_index, left_box in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            if left_box.overlaps(boxes[right_index]):
                raise RuntimeError(
                    f"Overlapping {description}: positions "
                    f"{left_index + 1} and {right_index + 1}"
                )


def _molecule_colors(summary: pd.DataFrame) -> dict[str, Any]:
    names = summary.sort_values(["Score_rank", "Name"])["Name"].tolist()
    positions = np.linspace(0.10, 0.90, max(len(names), 1))
    colormap = plt.get_cmap("viridis")
    return {name: colormap(position) for name, position in zip(names, positions)}


def _atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=PLOT_DPI)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_rank_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Draw a two-column rank connection chart."""
    configure_plot_style()
    colors = _molecule_colors(summary)
    figure, axis = plt.subplots(figsize=PLOT_SIZE)
    left_labels: list[plt.Text] = []
    right_labels: list[plt.Text] = []
    for row in summary.itertuples(index=False):
        y_values = [float(row.Score_rank), float(row.PCE_rank)]
        color = colors[str(row.Name)]
        axis.plot([0.0, 1.0], y_values, color=color, linewidth=2.4, alpha=0.9)
        axis.scatter(
            [0.0, 1.0],
            y_values,
            s=95,
            color=[color, color],
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )
        left_labels.append(
            axis.text(
                -0.08,
                y_values[0],
                f"{row.Name}  {row.Score:.3f}",
                ha="right",
                va="center",
                fontsize=15,
            )
        )
        right_labels.append(
            axis.text(
                1.08,
                y_values[1],
                f"{row.PCE_mean:.2f}  {row.Name}\n{row.Best_concentration}",
                ha="left",
                va="center",
                fontsize=15,
                linespacing=0.95,
            )
        )

    molecule_count = len(summary)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(molecule_count + 0.6, 0.4)
    axis.set_xticks([0.0, 1.0])
    axis.set_xticklabels(
        ["Score rank", "Best normalized PCE rank"], fontsize=18
    )
    axis.set_yticks(np.arange(1, molecule_count + 1))
    axis.set_ylabel("Rank (1 = highest)", fontsize=20)
    axis.set_title("Score and Device PCE Rank Agreement", fontsize=22, pad=16)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.65)
    _apply_axis_style(axis)
    figure.tight_layout(pad=1.2)
    _require_nonoverlapping_text(figure, left_labels, "left rank labels")
    _require_nonoverlapping_text(figure, right_labels, "right rank labels")
    _atomic_save_figure(figure, output_path)
    plt.close(figure)


def _annotate_scatter_points(
    figure: plt.Figure,
    axis: plt.Axes,
    summary: pd.DataFrame,
) -> list[plt.Text]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent().padded(-4)
    obstacles = []
    for row in summary.itertuples(index=False):
        x, y0 = axis.transData.transform((row.Score, row.PCE_mean - row.PCE_sd))
        _, y1 = axis.transData.transform((row.Score, row.PCE_mean + row.PCE_sd))
        obstacles.append(matplotlib.transforms.Bbox.from_extents(x-8, y0-8, x+8, y1+8))
    candidates = [(x, y) for distance in (10, 24, 40, 60, 85)
                  for y in (distance, -distance) for x in (8, -8, 0)]
    annotations: list[plt.Text] = []
    boxes: list[Any] = []
    for row in summary.sort_values(["PCE_mean", "Score"], ascending=False).itertuples(
        index=False
    ):
        chosen: plt.Text | None = None
        for x_offset, y_offset in candidates:
            annotation = axis.annotate(
                str(row.Name),
                (float(row.Score), float(row.PCE_mean)),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="center" if x_offset == 0 else ("left" if x_offset > 0 else "right"),
                va="bottom" if y_offset >= 0 else "top",
                fontsize=15,
                arrowprops=dict(arrowstyle="-", linestyle="--", color="#666666",
                                linewidth=0.8, shrinkA=3, shrinkB=6),
            )
            annotation.update_positions(renderer)
            box = plt.Text.get_window_extent(annotation, renderer=renderer).padded(2)
            if (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in boxes + obstacles)):
                chosen = annotation
                boxes.append(box)
                break
            annotation.remove()
        if chosen is None:
            raise RuntimeError(f"Could not place a nonoverlapping label for {row.Name}")
        annotations.append(chosen)
    return annotations


def _score_axis_segments(values: np.ndarray) -> list[tuple[float, float]]:
    """Omit interiors of Score gaps larger than one unit."""
    ordered = np.unique(values)
    groups = np.split(ordered, np.flatnonzero(np.diff(ordered) > 1.0) + 1)
    return [(float(group[0]) - 0.2, float(group[-1]) + 0.2) for group in groups]


def _format_p_value(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def plot_score_vs_pce(
    summary: pd.DataFrame,
    statistics_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Draw linear Score segments separated by explicit axis breaks."""
    configure_plot_style()
    colors = _molecule_colors(summary)
    segments = _score_axis_segments(summary["Score"].to_numpy(dtype=float))
    figure, axes = plt.subplots(
        1, len(segments), figsize=PLOT_SIZE, sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": [
            max(1.5, int(summary.Score.between(lo, hi).sum()))
            for lo, hi in segments
        ]},
    )
    axes = axes[0]
    lower = min(100.0, float((summary.PCE_mean - summary.PCE_sd).min()))
    upper = max(100.0, float((summary.PCE_mean + summary.PCE_sd).max()))
    span = max(upper - lower, 1.0)
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.90, wspace=0.16)
    labels = []
    for index, (axis, (lo, hi)) in enumerate(zip(axes, segments)):
        subset = summary.loc[summary.Score.between(lo, hi)]
        for row in subset.itertuples(index=False):
            axis.errorbar(
                row.Score, row.PCE_mean, yerr=row.PCE_sd, fmt="o",
                markersize=10, color=colors[str(row.Name)],
                markeredgecolor="black", markeredgewidth=0.7,
                elinewidth=1.8, capsize=4, zorder=3,
            )
        axis.set_xlim(lo, hi)
        axis.set_ylim(lower - 0.12 * span, upper + 0.18 * span)
        axis.axhline(100, color="#555555", linestyle="--", linewidth=1.2)
        axis.grid(True, color="#D0D0D0", linewidth=0.8, alpha=0.55)
        _apply_axis_style(axis)
        axis.tick_params(axis="y", labelsize=18)
        step = 0.2 if len(subset) >= 3 else 0.5
        ticks = np.arange(np.ceil((lo + 0.05) / step) * step, hi - 0.05, step)
        if len(subset) == 1:
            ticks = np.array([float(subset.Score.mean())])
        axis.set_xticks(ticks, labels=[f"{value:.1f}" for value in ticks])
        axis.tick_params(axis="x", labelsize=16)
        if index:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False, labelleft=False)
        if index < len(axes)-1:
            axis.spines["right"].set_visible(False)
        for edge in ([0] if index else []) + ([1] if index < len(axes)-1 else []):
            axis.plot([edge, edge], [0, 1], transform=axis.transAxes,
                      marker=[(-1, -0.6), (1, 0.6)], markersize=10,
                      linestyle="none", color="black", clip_on=False)
        labels.extend(_annotate_scatter_points(figure, axis, subset))
    axes[0].set_ylabel("Best normalized PCE (% of Control)", fontsize=22)
    figure.supxlabel("Score", fontsize=20, y=0.04)
    figure.suptitle("Score vs. Best-Concentration Device PCE", fontsize=24, y=0.98)
    stat = statistics_frame.iloc[0]
    figure.text(0.155, 0.88,
                f"n = {int(stat['N_molecules'])} molecules\n"
                f"Spearman rho = {stat['Spearman_rho']:.3f}, "
                f"p = {_format_p_value(stat['Spearman_p_two_sided'])}\n"
                f"Kendall tau-b = {stat['Kendall_tau_b']:.3f}, "
                f"p = {_format_p_value(stat['Kendall_p_two_sided'])}",
                fontsize=16, va="top", math_fontfamily="custom")
    _require_nonoverlapping_text(figure, labels, "molecule labels")
    _atomic_save_figure(figure, output_path)
    plt.close(figure)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            float_format="%.8f",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _complete_existing_artifact(path: Path, analysis_id: str) -> bool:
    manifest_path = path / "analysis_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("analysis_id") != analysis_id:
        return False
    hashes = manifest.get("artifact_sha256", {})
    return all(
        (path / filename).is_file()
        and hashes.get(filename) == file_sha256(path / filename)
        for filename in ARTIFACT_FILENAMES
    )


def run_analysis(campaign_round: str) -> tuple[Path, bool]:
    """Build or reuse one hash-bound device Score validation artifact."""
    device_root = DEVICE_ANALYSIS_OUTPUT_DIR / campaign_directory_name(campaign_round)
    group_summary_path = device_root / "group_summary.csv"
    normalization_summary_path = device_root / "normalization_summary.csv"
    group_summary = read_group_summary(group_summary_path, campaign_round)
    device_names = set(group_summary["Name"])
    normalization_summary = read_normalization_summary(
        normalization_summary_path,
        campaign_round,
        device_names,
    )

    score_info = ensure_score_artifact(campaign_round=campaign_round)
    scores_path = Path(score_info["scores_file"])
    score_version = str(score_info["score_version"])
    scores = read_scores(scores_path, device_names, score_version)
    summary = build_best_concentration_summary(
        group_summary,
        normalization_summary,
        scores,
        campaign_round,
        score_version,
    )
    statistics_frame = build_rank_statistics(
        summary,
        campaign_round,
        score_version,
    )

    implementation_path = Path(__file__).resolve()
    source_hashes = {
        "group_summary.csv": file_sha256(group_summary_path),
        "normalization_summary.csv": file_sha256(normalization_summary_path),
        "scores.csv": file_sha256(scores_path),
        implementation_path.name: file_sha256(implementation_path),
        "rank_statistics.py": file_sha256(PROJECT_ROOT / "rank_statistics.py"),
    }
    fingerprint_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "campaign_round": campaign_round,
        "score_version": score_version,
        "source_sha256": source_hashes,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    analysis_id = f"{ANALYSIS_VERSION}-{digest[:12]}"
    output_root = (
        DEVICE_SCORE_VALIDATION_OUTPUT_DIR
        / campaign_directory_name(campaign_round)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / analysis_id
    if output_dir.exists():
        if _complete_existing_artifact(output_dir, analysis_id):
            return output_dir, False
        raise FileExistsError(
            f"Incomplete device Score validation artifact already exists: {output_dir}"
        )

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{analysis_id}.", dir=output_root)
    )
    try:
        summary_path = temporary_dir / ARTIFACT_FILENAMES[0]
        statistics_path = temporary_dir / ARTIFACT_FILENAMES[1]
        rank_plot_path = temporary_dir / ARTIFACT_FILENAMES[2]
        scatter_plot_path = temporary_dir / ARTIFACT_FILENAMES[3]
        _atomic_write_csv(summary, summary_path)
        _atomic_write_csv(statistics_frame, statistics_path)
        plot_rank_comparison(summary, rank_plot_path)
        plot_score_vs_pce(summary, statistics_frame, scatter_plot_path)

        artifact_hashes = {
            filename: file_sha256(temporary_dir / filename)
            for filename in ARTIFACT_FILENAMES
        }
        statistic = statistics_frame.iloc[0]
        manifest = {
            "artifact_schema_version": 1,
            "analysis_id": analysis_id,
            "analysis_version": ANALYSIS_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_round": campaign_round,
            "score_version": score_version,
            "inputs": {
                "group_summary_file": _project_relative(group_summary_path),
                "normalization_summary_file": _project_relative(
                    normalization_summary_path
                ),
                "scores_file": _project_relative(scores_path),
                "score_manifest_file": _project_relative(
                    Path(score_info["manifest_path"])
                ),
                "source_sha256": source_hashes,
            },
            "method": {
                "device_statistic_unit": "device row within one concentration",
                "selection": (
                    "highest arithmetic mean normalized PCE among non-Control "
                    "concentrations"
                ),
                "tie_break": "lower numeric concentration",
                "normalized_pce_unit": "percent of within-file Control mean",
                "uncertainty": "sample SD, SEM, and two-sided 95% Student-t CI",
                "plot_error_bar": "sample SD",
                "score_axis": {
                    "type": "broken linear axis",
                    "gap_threshold": 1.0,
                    "segment_padding": 0.2,
                    "segment_width_rule": "max(1.5, molecule count) to fit labels",
                    "tick_step": "0.2 for >=3 molecules; 0.5 otherwise; singleton at Score",
                    "segments": _score_axis_segments(summary["Score"].to_numpy(dtype=float)),
                    "tick_labels": "original Score values",
                },
                "rank_method": "average rank for ties, descending",
                "p_value_method": statistic["P_value_method"],
                "permutation_count": int(statistic["Permutation_count"]),
                "random_seed": (
                    None
                    if pd.isna(statistic["Random_seed"])
                    else int(statistic["Random_seed"])
                ),
                "bootstrap": "not performed",
            },
            "molecule_count": int(len(summary)),
            "interpretation": (
                "This analysis evaluates monotonic rank agreement with the "
                "observed best tested concentration; it does not establish "
                "absolute PCE prediction accuracy or concentration-selection "
                "uncertainty."
            ),
            "implementation_file": implementation_path.name,
            "implementation_sha256": source_hashes[implementation_path.name],
            "artifact_sha256": artifact_hashes,
        }
        _atomic_write_json(manifest, temporary_dir / "analysis_manifest.json")
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return output_dir, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare versioned calibrated Score with each molecule's observed "
            "best normalized PCE from an existing device group_summary.csv."
        )
    )
    add_campaign_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_round = resolve_campaign_round(args.campaign_round)
    output_dir, generated = run_analysis(campaign_round)
    action = "Generated" if generated else "Reused"
    print(f"Campaign: {campaign_round}")
    print(f"{action} device Score validation artifact: {output_dir.name}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
