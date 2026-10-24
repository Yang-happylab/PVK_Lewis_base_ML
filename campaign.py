"""Shared experiment-campaign discovery, validation, and CLI helpers."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from project_paths import SCORE_CALIBRATION_DATA_FILE, require_input_file


CAMPAIGN_PATTERN = re.compile(r"^R(?P<number>\d{2})$")


def normalize_campaign_round(value: object) -> str:
    """Return a canonical ``Rxx`` campaign label or raise a clear error."""
    campaign_round = str(value).strip().upper()
    if not CAMPAIGN_PATTERN.fullmatch(campaign_round):
        raise ValueError(
            f"Invalid campaign round {value!r}; expected R followed by two digits"
        )
    return campaign_round


def campaign_number(value: object) -> int:
    """Return the numeric order of a validated campaign label."""
    campaign_round = normalize_campaign_round(value)
    return int(campaign_round[1:])


def available_campaign_rounds(
    calibration_path: str | Path = SCORE_CALIBRATION_DATA_FILE,
) -> tuple[str, ...]:
    """Discover sorted campaign labels from the cumulative calibration table."""
    path = require_input_file(Path(calibration_path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Round" not in reader.fieldnames:
            raise ValueError(f"Calibration table is missing the Round column: {path}")
        rounds: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            values = [item for value in row.values()
                      for item in (value if isinstance(value, list) else [value])]
            if all(value is None or not str(value).strip() for value in values):
                continue
            raw_round = row.get("Round")
            if raw_round is None or not str(raw_round).strip():
                raise ValueError(
                    f"Calibration table contains a blank Round at row {row_number}"
                )
            try:
                rounds.add(normalize_campaign_round(raw_round))
            except ValueError as exc:
                raise ValueError(
                    f"Calibration table row {row_number} has an invalid Round"
                ) from exc
    if not rounds:
        raise ValueError(f"Calibration table contains no campaign rows: {path}")
    return tuple(sorted(rounds, key=campaign_number))


def resolve_campaign_round(
    requested: str | None = None,
    calibration_path: str | Path = SCORE_CALIBRATION_DATA_FILE,
) -> str:
    """Resolve an explicit campaign or default to the newest calibration round."""
    available = available_campaign_rounds(calibration_path)
    if requested is None:
        return available[-1]
    campaign_round = normalize_campaign_round(requested)
    if campaign_round not in available:
        raise ValueError(
            f"Campaign {campaign_round} is unavailable; calibration.csv contains: "
            + ", ".join(available)
        )
    return campaign_round


def add_campaign_arguments(
    parser: argparse.ArgumentParser,
    calibration_path: str | Path = SCORE_CALIBRATION_DATA_FILE,
) -> argparse.ArgumentParser:
    """Add mutually exclusive ``--Rxx`` flags discovered from calibration.csv."""
    available = available_campaign_rounds(calibration_path)
    group = parser.add_mutually_exclusive_group()
    for campaign_round in available:
        group.add_argument(
            f"--{campaign_round}",
            dest="campaign_round",
            action="store_const",
            const=campaign_round,
            help=f"Use cumulative calibration data through {campaign_round}.",
        )
    parser.set_defaults(campaign_round=None)
    parser.epilog = (
        (parser.epilog + " " if parser.epilog else "")
        + f"If no round flag is supplied, the newest calibration round is used "
        f"({available[-1]} currently)."
    )
    return parser


def campaign_directory_name(value: object) -> str:
    """Return the standard output directory component for a campaign."""
    return f"campaign_{normalize_campaign_round(value)}"


def is_campaign_at_or_before(value: object, selected: object) -> bool:
    """Return whether one campaign is part of a selected cumulative campaign."""
    return campaign_number(value) <= campaign_number(selected)
