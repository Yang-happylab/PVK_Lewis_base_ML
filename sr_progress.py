"""Read-only progress viewer for symbolic-regression runs.

This module deliberately uses only the Python standard library.  It never
imports PySR or starts Julia, so it is safe to run beside an active search.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from project_paths import SYMBOLIC_REGRESSION_OUTPUT_DIR


_FOLD_RUN_ID = re.compile(r"^(cfg_[0-9a-f]+)_fold_(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _iter_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # The final line may be observed while the writer is appending.
                continue
            if isinstance(payload, dict):
                yield payload


def find_run_directory(
    output_root: Path = SYMBOLIC_REGRESSION_OUTPUT_DIR,
    *,
    campaign: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Find one repository-local run directory without accepting raw paths."""
    output_root = Path(output_root)
    campaign_dirs = (
        [output_root / f"campaign_{campaign}"]
        if campaign
        else sorted(output_root.glob("campaign_*"))
    )
    candidates: list[Path] = []
    for campaign_dir in campaign_dirs:
        if not campaign_dir.is_dir():
            continue
        if run_name:
            candidate = campaign_dir / run_name
            if candidate.is_dir():
                candidates.append(candidate)
        else:
            candidates.extend(path for path in campaign_dir.iterdir() if path.is_dir())
    candidates = [
        path
        for path in candidates
        if any(
            (path / filename).is_file()
            for filename in (
                "progress_status.json",
                "search_progress.jsonl",
                "sampled_configs.json",
            )
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No symbolic-regression run found below {output_root}"
        )

    def activity_time(path: Path) -> int:
        timestamps = [path.stat().st_mtime_ns]
        for filename in (
            "progress_status.json",
            "search_progress.jsonl",
            "symbolic_regression.log",
        ):
            artifact = path / filename
            if artifact.is_file():
                timestamps.append(artifact.stat().st_mtime_ns)
        return max(timestamps)

    return max(candidates, key=lambda path: (activity_time(path), str(path)))


def _cv_fold_total(run_config: dict[str, Any]) -> int | None:
    search = run_config.get("search", {})
    splits = search.get("cv_n_splits")
    repeats = search.get("cv_n_repeats")
    if splits is None:
        return None
    return int(splits) * int(repeats or 1)


def infer_status_from_jsonl(run_dir: Path) -> dict[str, Any]:
    """Build a concise snapshot for runs created before progress_status.json."""
    sampled_path = run_dir / "sampled_configs.json"
    sampled = _read_json(sampled_path) if sampled_path.is_file() else {}
    configurations = sampled.get("configurations", [])
    config_ids = [str(item.get("config_id")) for item in configurations]
    config_by_id = {
        str(item.get("config_id")): item
        for item in configurations
        if item.get("config_id") is not None
    }
    config_total = int(
        sampled.get("sampled_unique_configurations")
        or sampled.get("requested_unique_configurations")
        or len(configurations)
        or 0
    )
    run_config_path = run_dir / "run_config.json"
    run_config = _read_json(run_config_path) if run_config_path.is_file() else {}
    fold_total = _cv_fold_total(run_config)

    last_event = None
    for event in _iter_json_lines(run_dir / "search_progress.jsonl"):
        last_event = event
    if last_event is None:
        return {
            "schema_version": 1,
            "updated_at": datetime.fromtimestamp(
                run_dir.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds"),
            "phase": "configuration_search",
            "status": "waiting_for_first_chunk",
            "configuration": None,
            "fold": None,
            "decision": "waiting",
            "overall": {
                "terminal_configurations": None,
                "total_configurations": config_total,
            },
            "_source": "legacy_jsonl_fallback",
        }

    run_id = str(last_event.get("run_id", ""))
    match = _FOLD_RUN_ID.match(run_id)
    config_id = match.group(1) if match else None
    fold_index = int(match.group(2)) if match else None
    config_index = (
        config_ids.index(config_id) + 1
        if config_id is not None and config_id in config_ids
        else None
    )
    config = config_by_id.get(config_id, {})
    maximum_iterations = config.get("params", {}).get("niterations")
    requested_iterations = int(last_event.get("requested_iterations_total", 0))
    chunk_iterations = int(last_event.get("requested_chunk_iterations", 0))
    chunk_total = (
        int(math.ceil(int(maximum_iterations) / chunk_iterations))
        if maximum_iterations and chunk_iterations
        else None
    )
    progress_path = run_dir / "search_progress.jsonl"
    updated_at = datetime.fromtimestamp(
        progress_path.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")
    runtime = run_config.get("runtime", {})
    return {
        "schema_version": 1,
        "updated_at": updated_at,
        "phase": "cross_validation" if match else "refinement",
        "status": "fit_chunk_completed",
        "run_id": run_id,
        "configuration": (
            {
                "index": config_index,
                "total": config_total,
                "config_id": config_id,
            }
            if config_id is not None
            else None
        ),
        "fold": (
            {"index": fold_index, "total": fold_total}
            if fold_index is not None
            else None
        ),
        "chunk": {
            "index": int(last_event.get("chunk", 0)),
            "total": chunk_total,
        },
        "iterations": {
            "requested": requested_iterations,
            "maximum": int(maximum_iterations) if maximum_iterations else None,
        },
        "timing": {
            "elapsed_seconds": last_event.get("elapsed_seconds"),
            "fit_wall_seconds": last_event.get("fit_wall_seconds"),
        },
        "metrics": {
            "raw_validation_r2": last_event.get("raw_validation_r2"),
            "all_features_used": last_event.get("all_features_used"),
        },
        "cpu": {
            "effective_core_equivalents": last_event.get(
                "effective_core_equivalents"
            ),
            "logical_cpu_utilization_fraction": last_event.get(
                "logical_cpu_utilization_fraction"
            ),
            "visible_logical_processors": runtime.get("logical_processors"),
        },
        "runtime": {
            "configured_julia_threads": runtime.get("configured_julia_threads"),
            "actual_julia_threads": runtime.get("actual_julia_threads"),
            "blas_threads": runtime.get("blas_threads"),
        },
        "decision": last_event.get("decision"),
        "overall": {
            "terminal_configurations": None,
            "total_configurations": config_total,
        },
        "_source": "legacy_jsonl_fallback",
    }


def load_progress_snapshot(run_dir: Path) -> dict[str, Any]:
    status_path = Path(run_dir) / "progress_status.json"
    if status_path.is_file():
        payload = _read_json(status_path)
        payload["_source"] = "progress_status.json"
        return payload
    return infer_status_from_jsonl(Path(run_dir))


def _fraction(value: Any, maximum: Any) -> str:
    left = "?" if value is None else str(value)
    right = "?" if maximum is None else str(maximum)
    return f"{left}/{right}"


def format_progress_snapshot(run_dir: Path, status: dict[str, Any]) -> str:
    """Format a compact terminal view with no scientific-state mutation."""
    lines = [f"Run: {Path(run_dir)}", f"Phase: {status.get('phase', 'unknown')}"]
    configuration = status.get("configuration")
    if configuration:
        lines.append(
            "Configuration: "
            + _fraction(configuration.get("index"), configuration.get("total"))
            + f" [{configuration.get('config_id', 'unknown')}]"
        )
    overall = status.get("overall") or {}
    terminal = overall.get("terminal_configurations")
    if terminal is not None:
        lines.append(
            "Configurations terminal: "
            + _fraction(terminal, overall.get("total_configurations"))
            + " "
            + (
                f"(completed={overall.get('completed_configurations', 0)}, "
                f"pruned={overall.get('conservatively_pruned_configurations', 0)}, "
                f"failed={overall.get('failed_configurations', 0)})"
            )
        )
    elif configuration:
        lines.append(
            "Configurations terminal: unavailable for this legacy run; "
            "the current configuration index above is exact"
        )
    search_elapsed = overall.get("search_elapsed_seconds")
    if search_elapsed is not None:
        lines.append(f"Search elapsed: {float(search_elapsed):.1f} s")
    fold = status.get("fold")
    if fold:
        if fold.get("active"):
            active = ",".join(str(value) for value in fold["active"])
            lines.append(f"Active folds: {active}/{fold.get('total', '?')}")
        else:
            fold_value = fold.get("index", fold.get("completed"))
            fold_label = "Fold" if "index" in fold else "Folds completed"
            lines.append(f"{fold_label}: {_fraction(fold_value, fold.get('total'))}")
    parallel_workers = status.get("parallel_workers") or []
    for worker in sorted(parallel_workers, key=lambda row: int(row.get("fold", 0))):
        worker_status = worker.get("status") or {}
        worker_chunk = worker_status.get("chunk") or {}
        worker_iterations = worker_status.get("iterations") or {}
        worker_metrics = worker_status.get("metrics") or {}
        worker_r2 = worker_metrics.get("raw_validation_r2")
        r2_text = f"{float(worker_r2):.6f}" if worker_r2 is not None else "n/a"
        lines.append(
            f"  Fold {worker.get('fold', '?')}: chunk "
            f"{_fraction(worker_chunk.get('index'), worker_chunk.get('total'))}; "
            f"iterations {_fraction(worker_iterations.get('requested'), worker_iterations.get('maximum'))}; "
            f"R2={r2_text}; decision={worker_status.get('decision', 'running')}"
        )
    chunk = status.get("chunk")
    if chunk:
        lines.append(f"Chunk: {_fraction(chunk.get('index'), chunk.get('total'))}")
    iterations = status.get("iterations")
    if iterations:
        lines.append(
            "Iterations: "
            + _fraction(iterations.get("requested"), iterations.get("maximum"))
        )
    metrics = status.get("metrics") or {}
    raw_r2 = metrics.get("raw_validation_r2")
    lines.append(
        "Raw validation R2: "
        + (f"{float(raw_r2):.6f}" if raw_r2 is not None else "n/a")
    )
    cpu = status.get("cpu") or {}
    effective = cpu.get("effective_core_equivalents")
    utilization = cpu.get("logical_cpu_utilization_fraction")
    logical = cpu.get("visible_logical_processors")
    if effective is not None and utilization is not None:
        lines.append(
            f"CPU: {float(effective):.1f} effective cores; "
            f"{100.0 * float(utilization):.1f}% of {logical or '?'} logical processors"
        )
    runtime = status.get("runtime") or {}
    if runtime:
        lines.append(
            "Julia threads: configured="
            f"{runtime.get('configured_julia_threads', '?')}, actual="
            f"{runtime.get('actual_julia_threads', '?')}; BLAS="
            f"{runtime.get('blas_threads', '?')}"
        )
    timing = status.get("timing") or {}
    elapsed = timing.get("elapsed_seconds")
    if elapsed is not None:
        lines.append(f"Current fit elapsed: {float(elapsed):.1f} s")
    lines.append(f"Decision: {status.get('decision', 'unknown')}")
    lines.append(f"Updated: {status.get('updated_at', 'unknown')}")
    lines.append(f"Source: {status.get('_source', 'unknown')}")
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show symbolic-regression configuration/fold/chunk progress."
    )
    parser.add_argument(
        "--campaign",
        help="Campaign suffix such as R01; defaults to the latest active campaign.",
    )
    parser.add_argument(
        "--run",
        dest="run_name",
        help="Run-directory name below the selected campaign; never a raw path.",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=10.0,
        type=float,
        metavar="SECONDS",
        help="Refresh when progress changes (default interval: 10 seconds).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.watch is not None and args.watch < 1.0:
        raise SystemExit("--watch interval must be at least 1 second")
    try:
        run_dir = find_run_directory(
            campaign=args.campaign,
            run_name=args.run_name,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Progress unavailable: {exc}", file=sys.stderr)
        return 2

    previous = None
    while True:
        try:
            status = load_progress_snapshot(run_dir)
            rendered = format_progress_snapshot(run_dir, status)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rendered = f"Progress temporarily unavailable: {exc}"
        if rendered != previous:
            if previous is not None:
                print("\n" + "-" * 72 + "\n")
            print(rendered, flush=True)
            previous = rendered
        if args.watch is None:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
