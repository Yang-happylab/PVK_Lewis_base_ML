"""Normalize raw device JV tables to Control means and draw box plots."""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    resolve_campaign_round,
)
from project_paths import DEVICE_ANALYSIS_OUTPUT_DIR, DEVICE_DATA_DIR


RAW_TO_NORMALIZED = {
    "Voc (V)": "Voc",
    "Efficiency (%)": "PCE",
    "Fill Factor (%)": "FF",
    "Jsc (mA/cm^2)": "Jsc",
}
NORMALIZED_COLUMNS = tuple(RAW_TO_NORMALIZED.values())
CONTROL_ALIASES = {"control", "con"}
PLOT_SPECS = (
    ("PCE", "PCE (% of Control)", "PCE_boxplot.png"),
    ("Voc", "Voc (% of Control)", "Voc_boxplot.png"),
    ("FF", "FF (% of Control)", "FF_boxplot.png"),
    ("Jsc", "Jsc (% of Control)", "Jsc_boxplot.png"),
)


@lru_cache(maxsize=1)
def configure_plot_style() -> Path:
    """Require Arial and apply it to every Matplotlib text element."""
    try:
        font_path = Path(font_manager.findfont("Arial", fallback_to_default=False))
    except ValueError as exc:
        raise RuntimeError(
            "Arial is required for device plots but is not installed"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 20,
            "axes.unicode_minus": False,
        }
    )
    return font_path


def normalize_group_label(value: object) -> str:
    """Return ``Control`` or a canonical three-decimal concentration label."""
    if pd.isna(value):
        raise ValueError("Group contains a blank value")
    label = str(value).strip()
    if label.casefold() in CONTROL_ALIASES:
        return "Control"
    if not label.endswith("%"):
        raise ValueError(f"Unsupported Group label {value!r}; expected Control or x%")
    try:
        concentration = float(label[:-1].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid concentration label: {value!r}") from exc
    if not math.isfinite(concentration) or concentration < 0:
        raise ValueError(f"Invalid concentration label: {value!r}")
    return f"{concentration:.3f}%"


def ordered_groups(labels: pd.Series) -> list[str]:
    """Place Control first and concentrations in ascending numeric order."""
    unique = set(labels.astype(str))
    concentrations = sorted(
        (label for label in unique if label != "Control"),
        key=lambda label: float(label[:-1]),
    )
    return (["Control"] if "Control" in unique else []) + concentrations


def read_raw_device_table(path: Path) -> pd.DataFrame:
    """Load one raw CSV and enforce the device-table schema."""
    table = pd.read_csv(path, encoding="utf-8-sig")
    table = table.dropna(axis=1, how="all")
    required = ["Group", *RAW_TO_NORMALIZED]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if table.empty:
        raise ValueError(f"{path.name} contains no device rows")

    clean = table.loc[:, required].copy()
    valid_group = clean["Group"].notna() & clean["Group"].astype(str).str.strip().ne("")
    for column in RAW_TO_NORMALIZED:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    valid = valid_group & np.isfinite(clean[list(RAW_TO_NORMALIZED)]).all(axis=1)
    if not valid.all():
        rows = (np.flatnonzero(~valid) + 2).tolist()
        print(f"SKIP {path.name}: incomplete/non-finite device records at CSV rows {rows}")
    clean = clean.loc[valid].copy()
    if clean.empty:
        raise ValueError(f"{path.name} contains no complete device rows")
    clean["Group"] = clean["Group"].map(normalize_group_label)
    return clean.reset_index(drop=True)


def normalize_device_table(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize every metric to the mean of all Control rows in one file."""
    control = raw["Group"].eq("Control")
    if not control.any():
        raise ValueError("Device table contains no Control rows")

    raw_columns = list(RAW_TO_NORMALIZED)
    control_means = raw.loc[control, raw_columns].mean()
    baseline_values = control_means.to_numpy(dtype=float)
    if (~np.isfinite(baseline_values)).any() or (baseline_values <= 0).any():
        raise ValueError("Control means must be finite and greater than zero")

    normalized = pd.DataFrame({"Group": raw["Group"].copy()})
    for raw_column, normalized_column in RAW_TO_NORMALIZED.items():
        normalized[normalized_column] = (
            raw[raw_column] / float(control_means[raw_column]) * 100.0
        )
    return normalized, control_means


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        table.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            float_format="%.8f",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _metric_colors(groups: list[str]) -> list[object]:
    concentrations = [group for group in groups if group != "Control"]
    cmap = plt.get_cmap("viridis")
    concentration_colors = {
        group: cmap(position)
        for group, position in zip(
            concentrations,
            np.linspace(0.12, 0.88, max(len(concentrations), 1)),
        )
    }
    return [
        "#737373" if group == "Control" else concentration_colors[group]
        for group in groups
    ]


def _require_nonoverlapping_text(
    fig: plt.Figure,
    text_items: list[plt.Text],
    description: str,
) -> None:
    """Reject a plot when any visible text bounding boxes overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer=renderer) for item in text_items]
    for left_index, left_box in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            if left_box.overlaps(boxes[right_index]):
                raise RuntimeError(
                    f"Overlapping {description}: positions "
                    f"{left_index + 1} and {right_index + 1}"
                )


def plot_metric(
    normalized: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw one standalone box plot with all device-level observations."""
    configure_plot_style()
    groups = ordered_groups(normalized["Group"])
    values_by_group = [
        normalized.loc[normalized["Group"].eq(group), metric].to_numpy(dtype=float)
        for group in groups
    ]
    colors = _metric_colors(groups)
    positions = np.arange(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    box = ax.boxplot(
        values_by_group,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.6},
        capprops={"linewidth": 1.6},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
        patch.set_linewidth(1.7)

    for position, values, color in zip(positions, values_by_group, colors):
        jitter = (
            np.linspace(-0.15, 0.15, len(values))
            if len(values) > 1
            else np.array([0.0])
        )
        ax.scatter(
            position + jitter,
            values,
            s=23,
            marker="D",
            facecolor=color,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            zorder=3,
        )

    all_values = normalized[metric].to_numpy(dtype=float)
    minimum = float(all_values.min())
    maximum = float(all_values.max())
    span = max(maximum - minimum, 1.0)
    ax.set_ylim(minimum - 0.12 * span, maximum + 0.70 * span)
    mean_labels: list[plt.Text] = []
    for position, values in zip(positions, values_by_group):
        mean_labels.append(ax.text(
            position,
            maximum + (0.08 + 0.30 * (int(position) % 2)) * span,
            f"mean\n{values.mean():.2f}",
            ha="center",
            va="bottom",
            fontsize=28,
            linespacing=0.95,
        ))

    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=40)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=32, rotation=90, ha="center")
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=32,
        width=1.5,
        length=6,
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)
    fig.tight_layout(pad=1.2)
    _require_nonoverlapping_text(fig, mean_labels, "mean labels")
    _require_nonoverlapping_text(
        fig,
        [label for label in ax.get_xticklabels() if label.get_visible()],
        "x-axis tick labels",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarize_groups(
    normalized: pd.DataFrame,
    campaign_round: str,
    molecule: str,
) -> pd.DataFrame:
    """Return one audit row per concentration, including sample SD."""
    rows: list[dict[str, object]] = []
    for group in ordered_groups(normalized["Group"]):
        subset = normalized.loc[normalized["Group"].eq(group)]
        row: dict[str, object] = {
            "Round": campaign_round,
            "Name": molecule,
            "Group": group,
            "n_devices": int(len(subset)),
        }
        for metric in NORMALIZED_COLUMNS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = (
                float(subset[metric].std(ddof=1)) if len(subset) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize every data/device raw CSV to its own Control mean and "
            "draw standalone device box plots. Existing *_norm.csv files are "
            "outputs and are never read as inputs."
        )
    )
    add_campaign_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    campaign_round = resolve_campaign_round(args.campaign_round)
    if not DEVICE_DATA_DIR.is_dir():
        raise FileNotFoundError(f"Device data directory not found: {DEVICE_DATA_DIR}")

    raw_files = sorted(
        path
        for path in DEVICE_DATA_DIR.glob("*.csv")
        if not path.stem.endswith("_norm")
    )
    if not raw_files:
        raise FileNotFoundError(f"No raw device CSV files found in {DEVICE_DATA_DIR}")

    output_root = (
        DEVICE_ANALYSIS_OUTPUT_DIR / campaign_directory_name(campaign_round)
    )
    group_summaries: list[pd.DataFrame] = []
    control_rows: list[dict[str, object]] = []

    for raw_path in raw_files:
        molecule = raw_path.stem
        try:
            raw = read_raw_device_table(raw_path)
            normalized, control_means = normalize_device_table(raw)
        except (ValueError, pd.errors.EmptyDataError) as exc:
            print(f"SKIP {raw_path.name}: {exc}")
            continue
        molecule_output = output_root / molecule
        normalized_path = molecule_output / f"{molecule}_norm.csv"
        _atomic_write_csv(normalized, normalized_path)

        for metric, ylabel, filename in PLOT_SPECS:
            plot_metric(normalized, metric, ylabel, molecule_output / filename)

        group_summaries.append(
            summarize_groups(normalized, campaign_round, molecule)
        )
        control_rows.append(
            {
                "Round": campaign_round,
                "Name": molecule,
                "raw_file": raw_path.name,
                "normalized_file": normalized_path.relative_to(output_root).as_posix(),
                "n_devices": int(len(raw)),
                "n_control": int(raw["Group"].eq("Control").sum()),
                **{
                    f"Control_mean_{normalized_column}": float(
                        control_means[raw_column]
                    )
                    for raw_column, normalized_column in RAW_TO_NORMALIZED.items()
                },
            }
        )
        print(
            f"{molecule}: normalized {len(raw)} devices with "
            f"{int(raw['Group'].eq('Control').sum())} Control devices"
        )

    if not group_summaries:
        print("No usable device files; no summaries were written.")
        return 1
    _atomic_write_csv(
        pd.concat(group_summaries, ignore_index=True),
        output_root / "group_summary.csv",
    )
    _atomic_write_csv(
        pd.DataFrame(control_rows),
        output_root / "normalization_summary.csv",
    )
    print(f"Normalized CSV files, plots, and summaries: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
