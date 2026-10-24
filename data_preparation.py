"""Repository-local data loaders that inject a versioned generated Score."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import (
    PHYSICOCHEMICAL_DATA_FILE,
    PROJECT_ROOT,
    STRUCTURE_DATA_FILE,
    file_sha256,
    require_input_file,
)
from score_calibration import (
    DESCRIPTORS,
    SCORE_COLUMN,
    apply_score_model,
    ensure_score_artifact,
    load_score_table,
)

MODEL_FEATURE_CONFIG_SCHEMA_VERSION = 3
MODEL_SCORE_PROVENANCE_KEYS = (
    "score_version",
    "campaign_round",
    "score_output_dir",
    "scores_file",
    "manifest_path",
    "data_sha256",
    "calibration_sha256",
    "calibration_subset_sha256",
    "algorithm_version",
)
MODEL_SPLIT_MANIFEST_KEYS = (
    "split_schema_version",
    "method",
    "test_fraction",
    "requested_strata",
    "distance_metric",
    "row_count",
    "train_size",
    "test_size",
    "train_names_sha256",
    "test_names_sha256",
    "split_sha256",
)
MODEL_CV_MANIFEST_KEYS = (
    "method",
    "n_splits",
    "n_repeats",
    "requested_strata",
    "random_state",
)
MODEL_TRAINING_DATA_KEYS = (
    "dataset_kind",
    "source_file",
    "source_file_sha256",
    "row_count",
)
MODEL_EVALUATION_KEYS = (
    "cv_r2_mean",
    "cv_r2_std",
    "cv_mae_mean",
    "test_r2",
    "test_mse",
    "test_mae",
    "test_huber",
)


_file_sha256 = file_sha256


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_input_file(path))


def _validate_names(frame: pd.DataFrame, label: str) -> None:
    if "Name" not in frame.columns:
        raise ValueError(f"{label} must contain a Name column")
    if frame["Name"].isna().any() or frame["Name"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains a missing or blank Name")
    if frame["Name"].duplicated().any():
        duplicates = frame.loc[frame["Name"].duplicated(), "Name"].tolist()
        raise ValueError(f"{label} contains duplicate Name values: {duplicates}")


def build_split_manifest(
    names: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    test_fraction: float,
    requested_strata: int,
    distance_metric: str,
) -> dict[str, Any]:
    """Build a reproducible identity for a named stratified-DUPLEX split."""
    name_frame = pd.DataFrame({"Name": pd.Series(names).reset_index(drop=True)})
    _validate_names(name_frame, "split source")
    name_values = name_frame["Name"].astype(str).tolist()

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if train_indices.ndim != 1 or test_indices.ndim != 1:
        raise ValueError("Train and test indices must be one-dimensional")
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Training split contains duplicate indices")
    if len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Test split contains duplicate indices")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test splits overlap")

    combined = np.concatenate([train_indices, test_indices])
    expected = np.arange(len(name_values), dtype=int)
    if len(combined) != len(expected) or not np.array_equal(
        np.sort(combined), expected
    ):
        raise ValueError("Training and test indices do not cover every named row")

    train_names = [name_values[index] for index in train_indices]
    test_names = [name_values[index] for index in test_indices]
    split_identity = {
        "method": "stratified_duplex",
        "test_fraction": float(test_fraction),
        "requested_strata": int(requested_strata),
        "distance_metric": str(distance_metric),
        "train_names": train_names,
        "test_names": test_names,
    }
    return {
        "split_schema_version": 1,
        "method": split_identity["method"],
        "test_fraction": split_identity["test_fraction"],
        "requested_strata": split_identity["requested_strata"],
        "distance_metric": split_identity["distance_metric"],
        "row_count": len(name_values),
        "train_size": len(train_names),
        "test_size": len(test_names),
        "train_names_sha256": _canonical_json_sha256(train_names),
        "test_names_sha256": _canonical_json_sha256(test_names),
        "split_sha256": _canonical_json_sha256(split_identity),
    }


def _replace_score_by_name(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    _validate_names(frame, label)
    _validate_names(scores, "versioned scores")
    score_lookup = scores.set_index("Name")[SCORE_COLUMN]
    missing_names = sorted(set(frame["Name"]).difference(score_lookup.index))
    if missing_names:
        raise ValueError(
            f"{label} contains names absent from the versioned score artifact: "
            + ", ".join(missing_names)
        )

    result = frame.copy()
    generated_score = result["Name"].map(score_lookup)
    if generated_score.isna().any():
        raise ValueError(f"Failed to join every {label} row to a generated Score")
    result[SCORE_COLUMN] = generated_score.to_numpy(dtype=float)
    return result


def _validate_named_descriptors(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    label: str,
    tolerance: float = 1e-8,
) -> None:
    if not set(DESCRIPTORS).issubset(frame.columns):
        return
    reference = scores.set_index("Name")
    for descriptor in DESCRIPTORS:
        values = pd.to_numeric(frame[descriptor], errors="raise").to_numpy(dtype=float)
        expected = frame["Name"].map(reference[descriptor]).to_numpy(dtype=float)
        mismatch = ~np.isclose(values, expected, rtol=0.0, atol=tolerance)
        if mismatch.any():
            names = frame.loc[mismatch, "Name"].astype(str).tolist()
            raise ValueError(
                f"{label} has {descriptor} values inconsistent with data.csv for: "
                + ", ".join(names[:10])
            )


def _dataset_info(
    score_info: dict[str, Any],
    dataset_kind: str,
    source_path: Path,
    row_count: int,
    **details: Any,
) -> dict[str, Any]:
    info = dict(score_info)
    info.update(
        {
            "dataset_kind": dataset_kind,
            "source_file": str(source_path),
            "source_file_sha256": _file_sha256(source_path),
            "row_count": int(row_count),
            "score_column": SCORE_COLUMN,
            **details,
        }
    )
    return info


def load_scored_physicochemical_data(
    campaign_round: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``data.csv`` with the selected campaign Score attached by Name."""
    score_info = ensure_score_artifact(campaign_round=campaign_round)
    scores = load_score_table(score_info)
    source = _read_csv(PHYSICOCHEMICAL_DATA_FILE)
    _validate_named_descriptors(source, scores, "physicochemical data")
    result = _replace_score_by_name(source, scores, "physicochemical data")
    info = _dataset_info(
        score_info,
        "physicochemical",
        PHYSICOCHEMICAL_DATA_FILE,
        len(result),
        score_join="Name",
    )
    return result, info


def load_scored_structure_data(
    campaign_round: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return numerical structure features joined to generated Score by Name."""
    from structure_features import EXTRACTION_SETTINGS
    score_info = ensure_score_artifact(campaign_round=campaign_round)
    scores = load_score_table(score_info)
    source = pd.read_csv(require_input_file(STRUCTURE_DATA_FILE), keep_default_na=False)
    if SCORE_COLUMN in source.columns:
        raise ValueError(
            "data-s.csv must not store Score; it is attached from the versioned artifact"
        )
    result = _replace_score_by_name(source, scores, "structure data")
    _validate_named_descriptors(result, scores, "structure data")

    direct = apply_score_model(source, score_info)[SCORE_COLUMN].to_numpy(dtype=float)
    if not np.allclose(direct, result[SCORE_COLUMN], rtol=0.0, atol=2e-9):
        raise ValueError("Structure score join disagrees with the stored calibration model")

    info = _dataset_info(
        score_info,
        "structure",
        STRUCTURE_DATA_FILE,
        len(result),
        score_join="Name",
        descriptor_validation="Binding_energy, Transferred_charges, and Vbur",
        pandas_keep_default_na=False,
        structure_feature_extraction=EXTRACTION_SETTINGS,
    )
    return result, info


def expand_coordination_atom(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand accepted N/O/S/Cl tokens; ignore malformed or other element tokens."""
    result = frame.copy()
    atom_columns = {"n": "atom_N", "o": "atom_O", "s": "atom_S", "cl": "atom_Cl"}
    for column in atom_columns.values():
        result[column] = 0.0
    token_pattern = re.compile(r"^(\d*)([A-Za-z]+)$")
    for row_index, raw_value in result["Coordination_atom"].items():
        if pd.isna(raw_value):
            continue
        for token in str(raw_value).strip().split("-"):
            match = token_pattern.match(token)
            if not match:
                continue
            column = atom_columns.get(match.group(2).casefold())
            if column:
                result.at[row_index, column] += float(match.group(1)) if match.group(1) else 1.0
    return result.drop(columns=["Coordination_atom"])


def prepare_symbolic_regression_data(
    campaign_round: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive the symbolic-regression table from data.csv and current Score."""
    score_info = ensure_score_artifact(campaign_round=campaign_round)
    scores = load_score_table(score_info)
    source = _read_csv(PHYSICOCHEMICAL_DATA_FILE)
    source = _replace_score_by_name(source, scores, "physicochemical data")

    required = {"Coordination_atom", "MolWt", "N_lig", "RingNum", "gap", "μ", "MolVol"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(
            "Cannot derive symbolic-regression data; data.csv is missing columns: "
            + ", ".join(missing)
        )

    result = expand_coordination_atom(source).rename(
        columns={
            "MolWt": "Molecular_weight",
            "N_lig": "Denticity_of_ligand",
            "RingNum": "Ring_number",
            "gap": "Band_gap",
            "μ": "Dipol_moment",
            "MolVol": "Molecular_Volume",
        }
    )
    ordered_columns = [
        "Name",
        "Molecular_weight",
        "Denticity_of_ligand",
        "Ring_number",
        "HOMO",
        "LUMO",
        "Band_gap",
        "Dipol_moment",
        "fai_max",
        "fai_min",
        "Vbur",
        "Molecular_Volume",
        "TPSA",
        "Binding_energy",
        "Transferred_charges",
        SCORE_COLUMN,
        "atom_N",
        "atom_O",
        "atom_S",
        "atom_Cl",
    ]
    missing_derived = sorted(set(ordered_columns).difference(result.columns))
    if missing_derived:
        raise ValueError(f"Derived symbolic-regression table is incomplete: {missing_derived}")
    result = result.loc[:, ordered_columns]
    info = _dataset_info(
        score_info,
        "symbolic_regression",
        PHYSICOCHEMICAL_DATA_FILE,
        len(result),
        score_join="Name",
        derivation="Coordination_atom expansion and stable column renaming",
    )
    return result, info


def _portable_path(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _json_ready(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, str) and key and (
        key.endswith("_file")
        or key.endswith("_path")
        or key.endswith("_dir")
        or key == "source_file"
    ):
        return _portable_path(value)
    return value


def write_model_feature_config(
    config_path: str | Path,
    *,
    model_path: str | Path,
    model_type: str,
    target: str,
    full_feature_names: list[str],
    retained_feature_names: list[str],
    score_info: dict[str, Any],
    split_manifest: dict[str, Any],
    cv_manifest: dict[str, Any],
    evaluation: dict[str, Any],
    training_script: str,
) -> Path:
    """Write the complete, hash-bound sidecar consumed by the GA workflow."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")

    full_feature_names = [str(name) for name in full_feature_names]
    retained_feature_names = [str(name) for name in retained_feature_names]
    if not full_feature_names or len(full_feature_names) != len(set(full_feature_names)):
        raise ValueError("Full model feature names must be nonempty and unique")
    if not retained_feature_names or len(retained_feature_names) != len(
        set(retained_feature_names)
    ):
        raise ValueError("Retained model feature names must be nonempty and unique")
    unknown_features = sorted(set(retained_feature_names) - set(full_feature_names))
    if unknown_features:
        raise ValueError(f"Retained model features are unknown: {unknown_features}")

    missing_score_keys = sorted(
        key
        for key in MODEL_SCORE_PROVENANCE_KEYS
        if key not in score_info
        or score_info[key] is None
        or (isinstance(score_info[key], str) and not score_info[key].strip())
    )
    if missing_score_keys:
        raise ValueError(
            "Score provenance information is incomplete: "
            + ", ".join(missing_score_keys)
        )

    required_dataset_keys = set(MODEL_TRAINING_DATA_KEYS)
    missing_dataset_keys = sorted(required_dataset_keys.difference(score_info))
    if missing_dataset_keys:
        raise ValueError(
            "Training data provenance is incomplete: "
            + ", ".join(missing_dataset_keys)
        )

    if not isinstance(split_manifest, dict):
        raise ValueError("Split manifest must be a dictionary")
    missing_split_keys = sorted(
        set(MODEL_SPLIT_MANIFEST_KEYS).difference(split_manifest)
    )
    if missing_split_keys:
        raise ValueError(
            "Split manifest is incomplete: " + ", ".join(missing_split_keys)
        )

    if not isinstance(cv_manifest, dict):
        raise ValueError("CV manifest must be a dictionary")
    missing_cv_keys = sorted(set(MODEL_CV_MANIFEST_KEYS).difference(cv_manifest))
    if missing_cv_keys:
        raise ValueError(
            "CV manifest is incomplete: " + ", ".join(missing_cv_keys)
        )

    split_row_count = int(split_manifest["row_count"])
    split_train_size = int(split_manifest["train_size"])
    split_test_size = int(split_manifest["test_size"])
    training_row_count = int(score_info["row_count"])
    if split_row_count != split_train_size + split_test_size:
        raise ValueError("Split sizes do not add up to the split row count")
    if split_row_count != training_row_count:
        raise ValueError("Split row count does not match the training data row count")
    if int(split_manifest["split_schema_version"]) != 1:
        raise ValueError("Unsupported split manifest schema version")
    if not 0.0 < float(split_manifest["test_fraction"]) < 1.0:
        raise ValueError("Split test fraction must be between zero and one")
    if int(cv_manifest["n_splits"]) < 2 or int(cv_manifest["n_repeats"]) < 1:
        raise ValueError("CV manifest requires at least two folds and one repeat")

    required_evaluation_keys = set(MODEL_EVALUATION_KEYS)
    missing_evaluation_keys = sorted(required_evaluation_keys.difference(evaluation))
    if missing_evaluation_keys:
        raise ValueError(
            "Model evaluation is incomplete: "
            + ", ".join(missing_evaluation_keys)
        )
    numeric_evaluation = {
        key: float(evaluation[key])
        for key in sorted(required_evaluation_keys)
    }
    if not all(np.isfinite(value) for value in numeric_evaluation.values()):
        raise ValueError("Model evaluation contains a non-finite value")
    if numeric_evaluation["cv_r2_std"] < 0:
        raise ValueError("CV R2 standard deviation cannot be negative")

    score_manifest = {
        key: _json_ready(score_info[key], key)
        for key in MODEL_SCORE_PROVENANCE_KEYS
    }
    training_data = {
        key: _json_ready(score_info[key], key)
        for key in sorted(required_dataset_keys)
    }
    removed_feature_names = [
        name for name in full_feature_names if name not in retained_feature_names
    ]
    payload = {
        "feature_config_schema_version": MODEL_FEATURE_CONFIG_SCHEMA_VERSION,
        "model_type": str(model_type),
        "target": str(target),
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "training_script": str(training_script),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_feature_names": full_feature_names,
        "retained_feature_names": retained_feature_names,
        "removed_feature_names": removed_feature_names,
        "score_version": score_manifest["score_version"],
        "score_manifest": score_manifest,
        "training_data": training_data,
        "split_manifest": _json_ready(split_manifest),
        "cv_manifest": _json_ready(cv_manifest),
        "evaluation": numeric_evaluation,
        "structure_feature_extraction": score_info.get("structure_feature_extraction"),
    }
    oof_file = model_path.parent / 'feature_selection' / 'outer_oof_predictions.csv'
    predictions_file = model_path.parent / 'stacking_base_predictions.csv'
    payload['stacking_data'] = {
        'schema_version': 1,
        'oof_file': oof_file.relative_to(model_path.parent).as_posix(),
        'oof_sha256': file_sha256(oof_file),
        'predictions_file': predictions_file.name,
        'predictions_sha256': file_sha256(predictions_file),
    }

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def write_score_provenance(
    output_dir: str | Path,
    info_dict: dict[str, Any],
) -> Path:
    """Write the exact Score source and manifest beside a training run."""
    required = set(MODEL_SCORE_PROVENANCE_KEYS)
    missing = sorted(required.difference(info_dict))
    if missing:
        raise ValueError(f"Score provenance information is incomplete: {missing}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(info_dict["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "provenance_schema_version": 2,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_version": info_dict["score_version"],
        "campaign_round": info_dict["campaign_round"],
        "algorithm_version": info_dict["algorithm_version"],
        "data_sha256": info_dict["data_sha256"],
        "calibration_sha256": info_dict["calibration_sha256"],
        "calibration_subset_sha256": info_dict[
            "calibration_subset_sha256"
        ],
        "score_manifest_sha256": _file_sha256(manifest_path),
        "consumer": _json_ready(info_dict),
        "score_manifest": manifest,
    }
    provenance_path = destination / "score_provenance.json"
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance_path
