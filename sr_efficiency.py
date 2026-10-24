"""Small, dependency-free helpers for efficient local PySR searches.

The functions in this module deliberately avoid importing NumPy or PySR so
that CPU/thread configuration can be applied before either numerical runtime
is initialized.  They also make the search controller testable without
starting Julia.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PC_LOGICAL_THREADS = 32


@dataclass(frozen=True)
class FoldExecutionLayout:
    """Resolved process/thread layout for independent CV fold workers."""

    worker_count: int
    threads_per_worker: int
    aggregate_threads: int
    visible_logical_processors: int
    source: str


def resolve_fold_execution_layout(
    *,
    logical_processors: int | None,
    julia_threads: int,
    requested_workers: str | int | None = None,
    requested_threads_per_worker: str | int | None = None,
    deterministic: bool = False,
) -> FoldExecutionLayout:
    """Resolve a bounded outer-fold process layout without oversubscription.

    The 13900KF default is four independent fold processes with 8 Julia
    threads each.  This bounded layout improved fold throughput in a balanced
    local comparison while keeping the aggregate at 32 Julia threads.
    Fold-level process parallelism is used while each individual symbolic
    search remains multithreaded.  Explicit settings are rejected when their
    aggregate Julia threads exceed the logical processors visible to the
    current OS/WSL.
    """
    visible = max(1, int(logical_processors or os.cpu_count() or 1))
    total_threads = int(julia_threads)
    if total_threads < 1:
        raise ValueError("julia_threads must be a positive integer")
    total_threads = min(total_threads, visible)

    if deterministic:
        return FoldExecutionLayout(
            worker_count=1,
            threads_per_worker=1,
            aggregate_threads=1,
            visible_logical_processors=visible,
            source="strict_deterministic_serial",
        )

    workers_explicit = requested_workers not in (None, "")
    threads_explicit = requested_threads_per_worker not in (None, "")
    if workers_explicit:
        worker_count = int(requested_workers)
    elif visible >= 32 and total_threads >= 32:
        worker_count = 4
    else:
        worker_count = 2 if visible >= 24 and total_threads >= 16 else 1
    if worker_count < 1:
        raise ValueError("SR_FOLD_WORKERS must be a positive integer")

    if threads_explicit:
        threads_per_worker = int(requested_threads_per_worker)
    else:
        threads_per_worker = max(1, total_threads // worker_count)
    if threads_per_worker < 1:
        raise ValueError("SR_FOLD_WORKER_THREADS must be a positive integer")

    aggregate_threads = worker_count * threads_per_worker
    if aggregate_threads > visible:
        raise ValueError(
            "fold worker layout oversubscribes visible logical processors: "
            f"{worker_count} workers x {threads_per_worker} threads = "
            f"{aggregate_threads}, visible={visible}"
        )
    source = "explicit" if workers_explicit or threads_explicit else "workstation_default"
    return FoldExecutionLayout(
        worker_count=worker_count,
        threads_per_worker=threads_per_worker,
        aggregate_threads=aggregate_threads,
        visible_logical_processors=visible,
        source=source,
    )


def resolve_julia_threads(
    requested: str | int | None = None,
    logical_processors: int | None = None,
    workstation_logical_thread_cap: int = DEFAULT_PC_LOGICAL_THREADS,
) -> int:
    """Resolve the Julia thread count for the supported personal workstation.

    The project workstation is an Intel Core i9-13900KF with 24 physical cores
    and 32 logical processors.  The default now exposes all 32 logical threads
    to Julia so PySR can use the processor's Hyper-Threading capacity.  An
    explicit ``SR_JULIA_THREADS`` value always wins; otherwise the resolved
    value is capped at 32 and never exceeds the processors visible to Windows
    or WSL.  Throughput is measured and reported rather than inferred from
    Task Manager utilization alone.
    """
    visible = int(logical_processors or os.cpu_count() or 1)
    if requested not in (None, ""):
        value = int(requested)
        if value < 1:
            raise ValueError("SR_JULIA_THREADS must be a positive integer")
        return min(value, visible)
    return max(1, min(int(workstation_logical_thread_cap), visible))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable short identifier for one hyperparameter combination."""
    return payload_sha256(dict(config))[:12]


def payload_sha256(payload: Any) -> str:
    """Return a full stable SHA-256 digest for a JSON-compatible contract."""
    serialized = json.dumps(
        _canonical_value(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one persisted input or run-state artifact without loading it whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def sample_unique_configs(
    param_grid: Mapping[str, Sequence[Any]],
    requested: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Sample reproducible configurations without replacement.

    The current SR grid is small enough to enumerate exactly.  Sampling the
    Cartesian-product indices avoids the repeated configurations produced by
    independent ``np.random`` draws.
    """
    if requested < 1:
        raise ValueError("requested must be positive")
    keys = list(param_grid)
    value_lists = [list(param_grid[key]) for key in keys]
    if any(not values for values in value_lists):
        raise ValueError("every parameter must contain at least one value")

    combinations_by_fingerprint: dict[str, dict[str, Any]] = {}
    for values in itertools.product(*value_lists):
        params = dict(zip(keys, values))
        combinations_by_fingerprint.setdefault(config_fingerprint(params), params)
    combinations = list(combinations_by_fingerprint.values())
    if requested > len(combinations):
        raise ValueError(
            f"requested {requested} configurations but only "
            f"{len(combinations)} unique combinations are available"
        )
    count = int(requested)
    indices = random.Random(int(random_state)).sample(range(len(combinations)), count)

    sampled: list[dict[str, Any]] = []
    for index in indices:
        params = combinations[index]
        fingerprint = config_fingerprint(params)
        seed_payload = f"{int(random_state)}:{fingerprint}".encode("ascii")
        config_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "big")
        sampled.append(
            {
                "config_id": f"cfg_{fingerprint}",
                "config_seed": config_seed,
                "params": params,
            }
        )
    return sampled


def used_features(equation: str, feature_names: Sequence[str]) -> list[str]:
    """Return exact symbols remaining after symbolic simplification.

    Importing SymPy is delayed until this function is called, so importing the
    module to configure Julia threads remains dependency-free.  A parse failure
    is treated as ineligible rather than falling back to substring matching.
    """
    try:
        import sympy as sp

        symbols = {name: sp.Symbol(name) for name in feature_names}
        parsed = equation if isinstance(equation, sp.Basic) else sp.sympify(str(equation), locals=symbols)
        expression = sp.simplify(parsed)
        free_symbols = expression.free_symbols
        return [name for name in feature_names if symbols[name] in free_symbols]
    except Exception:
        return []


def feasible_frontier_stats(
    equations: Iterable[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    all_features_verified: bool = False,
) -> dict[str, Any]:
    """Summarize candidates; eligible candidate audits may reuse their feature gate."""
    required = set(feature_names)
    feasible: list[tuple[float, int]] = []
    for candidate in equations:
        equation = str(candidate.get("equation", ""))
        if not all_features_verified and set(used_features(equation, feature_names)) != required:
            continue
        try:
            loss = float(candidate.get("loss", math.inf))
            complexity = int(candidate.get("complexity", 10**9))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(loss):
            feasible.append((loss, complexity))

    if not feasible:
        return {
            "has_all_features": False,
            "candidate_count": 0,
            "best_loss": None,
            "min_complexity": None,
            "points": [],
        }
    best_by_complexity: dict[int, float] = {}
    for loss, complexity in feasible:
        best_by_complexity[complexity] = min(
            loss,
            best_by_complexity.get(complexity, math.inf),
        )
    return {
        "has_all_features": True,
        "candidate_count": len(feasible),
        "best_loss": min(loss for loss, _ in feasible),
        "min_complexity": min(complexity for _, complexity in feasible),
        "points": [
            {"complexity": complexity, "loss": best_by_complexity[complexity]}
            for complexity in sorted(best_by_complexity)
        ],
    }


def frontier_improved(
    previous_points: Sequence[Mapping[str, Any]],
    current_points: Sequence[Mapping[str, Any]],
    relative_tolerance: float,
) -> bool:
    """Compare Pareto lower envelopes across every observed complexity budget."""
    if not current_points:
        return False
    if not previous_points:
        return True

    previous = [
        (int(point["complexity"]), float(point["loss"]))
        for point in previous_points
    ]
    current = [
        (int(point["complexity"]), float(point["loss"]))
        for point in current_points
    ]
    budgets = sorted({complexity for complexity, _ in previous + current})
    for budget in budgets:
        old_loss = min(
            (loss for complexity, loss in previous if complexity <= budget),
            default=math.inf,
        )
        new_loss = min(
            (loss for complexity, loss in current if complexity <= budget),
            default=math.inf,
        )
        if math.isfinite(new_loss) and (
            not math.isfinite(old_loss)
            or new_loss < old_loss * (1.0 - float(relative_tolerance))
        ):
            return True
    return False


def validation_frontier_stats(
    candidate_audits: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize finite eligible validation R-squared by complexity budget."""
    feasible: list[tuple[float, int]] = []
    for candidate in candidate_audits:
        if not candidate.get("eligible"):
            continue
        try:
            score = float(candidate.get("raw_r2"))
            complexity = int(candidate.get("excess_complexity"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(score):
            feasible.append((score, complexity))

    if not feasible:
        return {
            "has_all_features": False,
            "candidate_count": 0,
            "best_raw_r2": None,
            "min_complexity": None,
            "points": [],
        }

    best_by_complexity: dict[int, float] = {}
    for score, complexity in feasible:
        best_by_complexity[complexity] = max(
            score,
            best_by_complexity.get(complexity, -math.inf),
        )
    return {
        "has_all_features": True,
        "candidate_count": len(feasible),
        "best_raw_r2": max(score for score, _ in feasible),
        "min_complexity": min(complexity for _, complexity in feasible),
        "points": [
            {"complexity": complexity, "raw_r2": best_by_complexity[complexity]}
            for complexity in sorted(best_by_complexity)
        ],
    }


def validation_frontier_improved(
    previous_points: Sequence[Mapping[str, Any]],
    current_points: Sequence[Mapping[str, Any]],
    minimum_improvement: float,
) -> bool:
    """Compare validation-score upper envelopes at common complexity budgets.

    R-squared may be negative, so the improvement threshold is an absolute
    delta rather than a relative ratio.  Training loss is deliberately absent:
    a better training frontier must not reset a validation plateau.
    """
    if not current_points:
        return False
    if not previous_points:
        return True

    previous = [
        (int(point["complexity"]), float(point["raw_r2"]))
        for point in previous_points
    ]
    current = [
        (int(point["complexity"]), float(point["raw_r2"]))
        for point in current_points
    ]
    budgets = sorted({complexity for complexity, _ in previous + current})
    for budget in budgets:
        old_score = max(
            (score for complexity, score in previous if complexity <= budget),
            default=-math.inf,
        )
        new_score = max(
            (score for complexity, score in current if complexity <= budget),
            default=-math.inf,
        )
        if math.isfinite(new_score) and (
            not math.isfinite(old_score)
            or new_score > old_score + float(minimum_improvement)
        ):
            return True
    return False


def update_refinement_frontier(
    history, current, *, relative_tolerance, similar_loss_tolerance, reference=None
):
    """Retain historical loss/simplicity tradeoffs and detect meaningful progress."""
    reference = history if reference is None else reference
    points = [p for p in current if math.isfinite(p['loss']) and p['loss'] >= 0]
    improved = not reference and bool(points)
    for point in points:
        loss, simplicity = point['loss'], tuple(point['simplicity'])
        old_loss = min(
            (p['loss'] for p in reference if tuple(p['simplicity']) <= simplicity),
            default=math.inf,
        )
        lower_loss = math.isfinite(old_loss) and loss < old_loss * (1.0 - relative_tolerance)
        simpler = any(
            simplicity < tuple(p['simplicity'])
            and loss <= p['loss'] * (1.0 + similar_loss_tolerance)
            for p in reference
        ) and not any(
            tuple(p['simplicity']) <= simplicity
            and p['loss'] <= loss * (1.0 + similar_loss_tolerance)
            for p in reference
        )
        improved |= lower_loss or simpler
    best = {}
    for point in list(history) + points:
        key = tuple(point['simplicity'])
        if key not in best or point['loss'] < best[key]['loss']:
            best[key] = point
    frontier = []
    for key, point in sorted(best.items()):
        if not frontier or point['loss'] < frontier[-1]['loss']:
            frontier.append(point)
    return frontier, bool(improved)


@dataclass
class ChunkEarlyStopper:
    """Plateau detector driven only by eligible validation/Pareto evidence."""

    min_iterations: int
    patience: int
    min_score_improvement: float
    require_score: bool = True
    best_score: float = -math.inf
    plateau_checks: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    window_iterations: int | None = None
    last_progress_iteration: int | None = None

    def observe(
        self,
        *,
        cumulative_iterations: int,
        has_all_features: bool,
        score: float | None,
        frontier_improved: bool = False,
    ) -> tuple[bool, str]:
        """Record a checkpoint and decide whether the search has plateaued."""
        finite_score = score is not None and math.isfinite(float(score))
        candidate_eligible = bool(has_all_features) and (
            finite_score or not self.require_score
        )
        score_improved = bool(
            candidate_eligible
            and finite_score
            and float(score) > self.best_score + self.min_score_improvement
        )
        eligible_frontier_improved = bool(candidate_eligible and frontier_improved)
        if score_improved:
            self.best_score = float(score)

        eligible_for_stop = (
            candidate_eligible and cumulative_iterations >= self.min_iterations
        )
        progress = score_improved or eligible_frontier_improved
        if not candidate_eligible:
            # An incomplete or invalid formula must not mutate any plateau state.
            pass
        elif not eligible_for_stop or progress:
            self.plateau_checks = 0
        else:
            self.plateau_checks += 1

        if self.window_iterations is not None and candidate_eligible:
            if eligible_for_stop:
                if self.last_progress_iteration is None or progress:
                    self.last_progress_iteration = int(cumulative_iterations)
            else:
                self.last_progress_iteration = None
        should_stop = eligible_for_stop and (
            cumulative_iterations - self.last_progress_iteration >= self.window_iterations
            if self.window_iterations is not None and self.last_progress_iteration is not None
            else self.plateau_checks >= self.patience
        )
        reason = "plateau" if should_stop else "continue"
        self.history.append(
            {
                "cumulative_iterations": int(cumulative_iterations),
                "has_all_features": bool(has_all_features),
                "score": None if score is None else float(score),
                "candidate_eligible": bool(candidate_eligible),
                "score_improved": bool(score_improved),
                "frontier_improved": bool(eligible_frontier_improved),
                "plateau_checks": int(self.plateau_checks),
                "last_progress_iteration": self.last_progress_iteration,
                "decision": reason,
            }
        )
        return should_stop, reason


def conservative_fold_prune(
    challenger_scores: Sequence[float],
    incumbent_scores: Sequence[float] | None,
    *,
    minimum_folds: int,
    total_folds: int,
    safety_margin: float = 0.01,
) -> tuple[bool, dict[str, float | int | str]]:
    """Prune only when a partial configuration is clearly dominated.

    R-squared cannot exceed 1.  Missing folds are therefore assigned the
    theoretical maximum of 1, and pruning occurs only if that optimistic final
    mean still cannot reach the completed incumbent after a safety margin.
    This is stricter and safer than pruning from a small-sample confidence
    interval.
    """
    if len(challenger_scores) < int(minimum_folds):
        return False, {"reason": "minimum_folds_not_reached", "folds": len(challenger_scores)}
    if not incumbent_scores:
        return False, {"reason": "no_completed_incumbent", "folds": len(challenger_scores)}

    challenger = [float(value) for value in challenger_scores]
    incumbent = [float(value) for value in incumbent_scores]
    if len(challenger) >= int(total_folds):
        return False, {"reason": "challenger_already_complete", "folds": len(challenger)}
    challenger_mean = statistics.fmean(challenger)
    incumbent_mean = statistics.fmean(incumbent)
    remaining = int(total_folds) - len(challenger)
    optimistic_final_mean = (sum(challenger) + remaining * 1.0) / int(total_folds)
    prune = optimistic_final_mean + float(safety_margin) < incumbent_mean
    return prune, {
        "reason": "dominated" if prune else "confidence_intervals_overlap",
        "folds": len(challenger),
        "challenger_mean": challenger_mean,
        "optimistic_final_mean": optimistic_final_mean,
        "incumbent_mean": incumbent_mean,
        "safety_margin": float(safety_margin),
    }


def shutdown_process_pool_bounded(
    executor: Any,
    *,
    graceful_wait_seconds: float,
    terminate_wait_seconds: float,
    kill_wait_seconds: float,
    poll_seconds: float = 0.1,
) -> dict[str, Any]:
    """Shut down a process pool without an unbounded worker join.

    This helper is intended only after the caller has collected every required
    future result or has already decided to abort the run. Python 3.12 has no
    public ``terminate_workers`` method, so the worker ``Process`` objects are
    snapshotted before non-blocking shutdown. A normal exit gets a bounded
    grace period; only workers that remain alive are terminated, with ``kill``
    as a final short fallback.
    """
    if min(
        float(graceful_wait_seconds),
        float(terminate_wait_seconds),
        float(kill_wait_seconds),
    ) < 0:
        raise ValueError("process-pool shutdown timeouts must be non-negative")
    if float(poll_seconds) <= 0:
        raise ValueError("poll_seconds must be positive")

    started = time.monotonic()
    process_map = dict(getattr(executor, "_processes", {}) or {})
    processes = list(process_map.values())
    manager_thread = getattr(executor, "_executor_manager_thread", None)

    def process_pid(process: Any) -> int | None:
        value = getattr(process, "pid", None)
        return int(value) if value is not None else None

    def is_alive(process: Any) -> bool:
        try:
            return bool(process.is_alive())
        except (OSError, ValueError):
            return False

    def living() -> list[Any]:
        return [process for process in processes if is_alive(process)]

    def wait_for_exit(seconds: float) -> list[Any]:
        deadline = time.monotonic() + float(seconds)
        remaining = living()
        while remaining and time.monotonic() < deadline:
            time.sleep(
                min(
                    float(poll_seconds),
                    max(0.0, deadline - time.monotonic()),
                )
            )
            remaining = living()
        return remaining

    worker_pids = [pid for process in processes if (pid := process_pid(process))]
    executor.shutdown(wait=False, cancel_futures=True)
    remaining = wait_for_exit(float(graceful_wait_seconds))

    terminated_pids: list[int] = []
    for process in remaining:
        pid = process_pid(process)
        try:
            process.terminate()
            if pid is not None:
                terminated_pids.append(pid)
        except (OSError, ProcessLookupError, ValueError):
            pass
    remaining = wait_for_exit(float(terminate_wait_seconds))

    killed_pids: list[int] = []
    for process in remaining:
        pid = process_pid(process)
        try:
            process.kill()
            if pid is not None:
                killed_pids.append(pid)
        except (AttributeError, OSError, ProcessLookupError, ValueError):
            pass
    remaining = wait_for_exit(float(kill_wait_seconds))

    for process in processes:
        try:
            process.join(timeout=0)
        except (AssertionError, OSError, ValueError):
            pass

    if manager_thread is not None:
        try:
            manager_thread.join(timeout=float(kill_wait_seconds))
        except (RuntimeError, AttributeError):
            pass
    manager_alive = bool(
        manager_thread is not None
        and getattr(manager_thread, "is_alive", lambda: False)()
    )
    remaining_pids = [
        pid for process in remaining if (pid := process_pid(process))
    ]
    return {
        "worker_pids": worker_pids,
        "graceful_wait_seconds": float(graceful_wait_seconds),
        "terminate_wait_seconds": float(terminate_wait_seconds),
        "kill_wait_seconds": float(kill_wait_seconds),
        "graceful": not terminated_pids and not killed_pids and not remaining_pids,
        "forced_termination": bool(terminated_pids or killed_pids),
        "terminated_pids": terminated_pids,
        "killed_pids": killed_pids,
        "remaining_pids": remaining_pids,
        "manager_thread_alive": manager_alive,
        "elapsed_seconds": time.monotonic() - started,
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically persist lightweight search progress in the run directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def append_json_line(path: str | Path, payload: Any) -> None:
    """Append one finite JSON record without rewriting prior progress."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
        separators=(",", ":"),
    )
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized + "\n")
        handle.flush()
