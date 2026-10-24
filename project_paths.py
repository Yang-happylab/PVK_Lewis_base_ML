"""Portable, repository-local paths shared by all entry-point scripts."""

import hashlib
from pathlib import Path


# Resolve from this file instead of the process working directory. The project
# remains portable when its parent path contains spaces or CJK characters, and
# also when a script is launched from another working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PHYSICOCHEMICAL_DATA_FILE = DATA_DIR / "data.csv"
STRUCTURE_DATA_FILE = DATA_DIR / "data-s.csv"
SCORE_CALIBRATION_DATA_FILE = DATA_DIR / "calibration.csv"
POSTERIOR_VALIDATION_DATA_FILE = DATA_DIR / "posterior_validation.csv"
DEVICE_DATA_DIR = DATA_DIR / "device"

XGB_OUTPUT_DIR = OUTPUT_DIR / "xgb_interpretability"
SYMBOLIC_REGRESSION_OUTPUT_DIR = OUTPUT_DIR / "symbolic_regression"
STRUCTURE_MODELS_OUTPUT_DIR = OUTPUT_DIR / "structure_models"
GA_OUTPUT_DIR = OUTPUT_DIR / "genetic_algorithm"
SCORE_CALIBRATION_OUTPUT_DIR = OUTPUT_DIR / "score_calibration"
POSTERIOR_PREDICTION_OUTPUT_DIR = OUTPUT_DIR / "posterior_predictions"
DEVICE_ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "device_analysis"
DEVICE_SCORE_VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "device_score_validation"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a project input or artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input_file(path: Path) -> Path:
    """Return a repository input path, raising a clear error when it is absent."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"Copy the complete project folder and keep input CSV files in: {DATA_DIR}"
        )
    return path


def find_latest_model(model_root: Path, model_filename: str) -> Path:
    """Find the most recently written matching model below ``model_root``."""
    model_root = Path(model_root)
    candidates = [
        path for path in model_root.rglob(model_filename)
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Model file not found below {model_root}: {model_filename}"
        )

    # The path is a deterministic tie-breaker for files copied with identical
    # timestamps.
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
