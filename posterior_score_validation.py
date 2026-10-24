"""Validate four within-round posterior Score predictions from three bound models."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sympy as sp
from rdkit import Chem
from sklearn.ensemble import VotingRegressor

from campaign import (
    add_campaign_arguments,
    campaign_directory_name,
    campaign_number,
    normalize_campaign_round,
    resolve_campaign_round,
)
from project_paths import (
    DATA_DIR,
    POSTERIOR_PREDICTION_OUTPUT_DIR,
    POSTERIOR_VALIDATION_DATA_FILE,
    PROJECT_ROOT,
    XGB_OUTPUT_DIR,
    GA_OUTPUT_DIR,
    SYMBOLIC_REGRESSION_OUTPUT_DIR,
    find_latest_model,
    require_input_file,
    file_sha256,
)
from posterior_validation_analysis import (
    XGB_SEED_AVERAGE_METADATA,
    POSTERIOR_MEASUREMENT_COLUMNS,
    coerce_measurement_columns,
    build_experimental_summary,
    load_bound_score_scale,
    score_prediction_intervals,
    summarize_posterior_reference,
    write_rank_correlation_analysis,
)
from structure_features import FEATURE_COLUMNS, EXTRACTION_SETTINGS, describe_structure
from sr_reporting import evaluate_expression, standardized_to_raw_expression


TARGET_NAME = "Score"
MODEL_LABELS = {"xgb": "XGB", "ga": "GA_Stacking", "sr": "SR"}
MEAN_MODEL_LABEL = "Mean_Score"
SR_FEATURE_MAPPING = {
    "Denticity_of_ligand": "N_lig", "atom_N": "atom_N", "fai_min": "fai_min",
    "Band_gap": "gap", "Dipol_moment": "\N{GREEK SMALL LETTER MU}", "HOMO": "HOMO",
    "Ring_number": "RingNum", "Molecular_weight": "MolWt",
    "Molecular_Volume": "MolVol", "LUMO": "LUMO", "fai_max": "fai_max",
    "TPSA": "TPSA", "atom_O": "atom_O", "atom_S": "atom_S", "atom_Cl": "atom_Cl",
}
MODEL_FILENAME = "best_model_selected_features.pkl"
MODEL_CONTRACT_FILENAME = "model_contract_selected_features.json"
METADATA_COLUMNS = ("Round", "Name", "SMILES")
POSTERIOR_DESCRIPTOR_COLUMNS = (
    "MolWt",
    "N_lig",
    "RingNum",
    "HOMO",
    "LUMO",
    "gap",
    "\N{GREEK SMALL LETTER MU}",
    "fai_max",
    "fai_min",
    "MolVol",
    "TPSA",
    "atom_N",
    "atom_O",
    "atom_S",
    "atom_Cl",
)
COUNT_COLUMNS = {"N_lig", "RingNum", "atom_N", "atom_O", "atom_S", "atom_Cl"}
UNBOUND_MODEL_FEATURES = (
    "N_lig",
    "HOMO",
    "gap",
    "\N{GREEK SMALL LETTER MU}",
    "fai_min",
    "atom_N",
)
POSTERIOR_RESULT_MEASUREMENT_COLUMNS = (
    *POSTERIOR_MEASUREMENT_COLUMNS,
    "Score_PI95_Lower",
    "Score_PI95_Upper",
    "Score_PI95_Status",
    "Measurement_Complete",
    "PLQY_Relative_Ref_Percent",
    "XRD_Relative_Ref_Percent",
    "Observed_Score",
    "Observed_Score_Status",
    "Predicted_Rank_All",
    "Predicted_Rank",
    "Observed_Rank",
    "Rank_Difference",
    "PLQY_N",
    "PLQY_Status",
    "PLQY_Mean",
    "PLQY_SD",
    "PLQY_SEM",
    "PLQY_CI95_Lower",
    "PLQY_CI95_Upper",
    "XRD_N",
    "XRD_Status",
    "XRD_Mean",
    "XRD_SD",
    "XRD_SEM",
    "XRD_CI95_Lower",
    "XRD_CI95_Upper",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")





def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash an effective posterior subset in a platform-independent form."""
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def portable_path(path: str | Path) -> str:
    """Return a project-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_input_path(value: str | None) -> Path:
    """Resolve the posterior CSV below the project data directory."""
    candidate = POSTERIOR_VALIDATION_DATA_FILE if value is None else Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Posterior input CSV must be located below {DATA_DIR}: {candidate}"
        ) from exc
    return require_input_file(candidate)


def resolve_model_path(value: str | None, campaign_round: str) -> Path:
    """Resolve an explicit model or discover the newest model in one campaign."""
    if value is None:
        model_root = XGB_OUTPUT_DIR / campaign_directory_name(campaign_round)
        candidates = sorted(
            (path for path in model_root.rglob(MODEL_FILENAME) if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True,
        )
        rejected = []
        for candidate in candidates:
            try:
                load_model_contract(candidate, campaign_round, allow_unbound_model=False)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                rejected.append(f"{candidate}: {exc}")
                continue
            return candidate.resolve()
        details = "\n".join(rejected) or f"No {MODEL_FILENAME} files found"
        raise FileNotFoundError(
            f"No complete bound selected-feature XGBoost model below {model_root}.\n{details}"
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"XGBoost model file not found: {candidate}")
    return candidate.resolve()


def load_model_contract(
    model_path: Path,
    campaign_round: str,
    *,
    allow_unbound_model: bool,
) -> tuple[list[str], dict[str, Any], Path | None]:
    """Load and validate the campaign and feature order bound to a model."""
    contract_path = model_path.with_name(MODEL_CONTRACT_FILENAME)
    if not contract_path.is_file():
        if not allow_unbound_model:
            raise FileNotFoundError(
                f"Model contract not found: {contract_path}. Retrain with "
                "XGB_Lewis_base.py, or explicitly pass --allow-unbound-model "
                "for a historical model with a verified feature order."
            )
        contract = {
            "schema_version": 0,
            "status": "explicit-unbound-historical-model",
            "model_file": model_path.name,
            "model_sha256": file_sha256(model_path),
            "model_stage": "selected_features",
            "campaign_round": campaign_round,
            "target": TARGET_NAME,
            "feature_names": list(UNBOUND_MODEL_FEATURES),
        }
        return list(UNBOUND_MODEL_FEATURES), contract, None

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required_fields = {
        "schema_version",
        "model_file",
        "model_sha256",
        "model_stage",
        "campaign_round",
        "target",
        "feature_names",
    }
    missing = sorted(required_fields.difference(contract))
    if missing:
        raise ValueError(
            f"Model contract is missing required fields: {', '.join(missing)}"
        )
    if contract["schema_version"] not in (2, 3):
        raise ValueError("Unsupported posterior model-contract schema version")
    if contract["model_file"] != model_path.name:
        raise ValueError("Model contract points to a different model filename")
    if contract["model_sha256"] != file_sha256(model_path):
        raise ValueError("Model SHA-256 does not match the model contract")
    if contract["model_stage"] != "selected_features":
        raise ValueError("Posterior prediction requires the selected-feature model")
    if contract["target"] != TARGET_NAME:
        raise ValueError("Posterior prediction requires a Score model")
    contract_round = normalize_campaign_round(contract["campaign_round"])
    if contract_round != campaign_round:
        raise ValueError(
            f"Model contract is bound to {contract_round}, not {campaign_round}"
        )

    feature_names = contract["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Model contract feature_names must be a nonempty list")
    feature_names = [str(feature) for feature in feature_names]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Model contract contains duplicate feature names")
    unknown = sorted(set(feature_names).difference(POSTERIOR_DESCRIPTOR_COLUMNS))
    if unknown:
        raise ValueError(f"Model contract contains unsupported features: {unknown}")

    calibration = contract.get("prediction_interval_calibration")
    if not isinstance(calibration, dict) or not calibration.get("score_version"):
        raise ValueError(
            "Incomplete XGBoost model contract: bound Score and consensus OOF "
            "calibration are required"
        )
    provenance, _ = load_score_provenance(model_path)
    if provenance is None or provenance.get("campaign_round") != campaign_round:
        raise ValueError("XGBoost model requires matching campaign Score provenance")
    _, score_scale = load_bound_score_scale(provenance, campaign_round)
    score_version = score_scale["score_version"]
    if (calibration["score_version"] != score_version
            or contract.get("score_version", score_version) != score_version):
        raise ValueError("XGBoost model Score binding differs from its Score provenance")
    score_prediction_intervals(
        model_path, contract, [], expected_campaign_round=campaign_round,
        expected_score_version=score_version,
    )
    return feature_names, contract, contract_path


def validate_seed_average_model(model, feature_names):
    """Verify the frozen equal-weight predictor against the five-seed contract."""
    from xgboost import XGBRegressor

    seeds = XGB_SEED_AVERAGE_METADATA["model_seeds"]
    if (not isinstance(model, VotingRegressor)
            or len(model.estimators) != len(seeds)
            or len(getattr(model, "estimators_", [])) != len(seeds)):
        raise ValueError("Five-seed XGBoost requires five fitted VotingRegressor members")
    if model.weights is not None:
        weights = np.asarray(model.weights, dtype=float)
        if (weights.shape != (len(seeds),) or not np.isfinite(weights).all()
                or weights[0] <= 0 or not np.equal(weights, weights[0]).all()):
            raise ValueError("Five-seed XGBoost requires equal positive member weights")
    signatures = []
    for member, seed in zip(model.estimators_, seeds):
        if not isinstance(member, XGBRegressor):
            raise ValueError("Five-seed predictor contains a non-XGBoost member")
        params = member.get_params()
        if params.pop("random_state", None) != seed:
            raise ValueError("Five-seed XGBoost member seed order differs from its contract")
        signatures.append(joblib.hash(params))
        if getattr(member, "n_features_in_", None) != len(feature_names):
            raise ValueError("Five-seed XGBoost member feature count differs from its contract")
        names = getattr(member, "feature_names_in_", None)
        if names is None or list(names) != feature_names:
            raise ValueError("Five-seed XGBoost member feature order differs from its contract")
    if len(set(signatures)) != 1:
        raise ValueError("Five-seed XGBoost member parameters differ beyond random_state")


def load_posterior_data(
    csv_path: Path,
    feature_names: Sequence[str],
    campaign_round: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], int, pd.DataFrame]:
    """Load posterior rounds through the selected model campaign."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next((row for row in csv.reader(handle) if any(value.strip() for value in row)), [])
    duplicates = sorted({column for column in header if column and header.count(column) > 1})
    if duplicates:
        raise ValueError(f"Posterior CSV contains duplicate columns: {duplicates}")
    frame = pd.read_csv(csv_path, encoding="utf-8")
    frame = frame.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
    frame = frame.drop(columns=[
        column for column in frame.columns
        if str(column).startswith("Unnamed:") and frame[column].isna().all()
    ])

    required_columns = [
        *METADATA_COLUMNS,
        *POSTERIOR_DESCRIPTOR_COLUMNS,
        *POSTERIOR_MEASUREMENT_COLUMNS,
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Posterior CSV is missing contract columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Posterior CSV contains no samples")

    for column in ("Round", "Name"):
        if frame[column].isna().any():
            raise ValueError(f"Posterior CSV contains a missing {column}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Posterior CSV contains a blank {column}")
    frame["Round"] = frame["Round"].map(normalize_campaign_round)

    duplicate_mask = frame.duplicated(["Round", "Name"], keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, ["Round", "Name"]].drop_duplicates()
        details = ", ".join(
            f"{row.Round}/{row.Name}" for row in duplicates.itertuples(index=False)
        )
        raise ValueError(f"Posterior CSV contains duplicate Round/Name rows: {details}")

    input_rounds = sorted(frame["Round"].unique(), key=campaign_number)
    input_row_count = len(frame)
    frame["_Input_Order"] = np.arange(1, input_row_count + 1, dtype=int)
    eligible = frame.loc[
        frame["Round"].map(campaign_number).le(campaign_number(campaign_round))
    ].copy()
    if eligible.empty:
        raise ValueError(f"Posterior CSV contains no samples for {campaign_round}")
    eligible = coerce_measurement_columns(eligible.reset_index(drop=True))
    evaluated_rounds = sorted(eligible["Round"].unique(), key=campaign_number)
    reference = eligible.loc[eligible["Name"].eq("REF")].copy()
    if not reference.groupby("Round").size().reindex(evaluated_rounds, fill_value=0).eq(1).all():
        raise ValueError("Each evaluated round requires exactly one posterior REF row")
    eligible = eligible.loc[~eligible["Name"].eq("REF")].reset_index(drop=True)
    if not eligible.groupby("Round").size().reindex(evaluated_rounds, fill_value=0).gt(0).all():
        raise ValueError("Each evaluated posterior round requires molecules")
    if eligible["SMILES"].isna().any():
        raise ValueError("Posterior CSV contains a missing SMILES")
    eligible["SMILES"] = eligible["SMILES"].astype(str).str.strip()
    for index, row in enumerate(eligible.itertuples(index=False)):
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(f"Invalid posterior SMILES for {row.Name}: {row.SMILES}")
        Chem.RemoveStereochemistry(mol)
        eligible.at[index, "SMILES"] = Chem.MolToSmiles(mol, canonical=True)
    if eligible.duplicated(["Round", "SMILES"]).any():
        raise ValueError("Posterior CSV repeats a molecular structure within a round")

    features = eligible.loc[:, list(feature_names)].copy()
    for column in feature_names:
        features[column] = pd.to_numeric(features[column], errors="raise")
    incomplete = [column for column in feature_names if features[column].isna().any()]
    if incomplete:
        raise ValueError(
            f"The {campaign_round} model requires completed posterior values for: "
            + ", ".join(incomplete)
        )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Posterior model features contain non-finite values")
    for column in COUNT_COLUMNS.intersection(feature_names):
        values = features[column].to_numpy(dtype=float)
        if (values < 0).any() or not np.allclose(values, np.round(values)):
            raise ValueError(f"{column} must contain nonnegative integer counts")

    return eligible, features, input_rounds, evaluated_rounds, input_row_count, reference


def load_score_provenance(model_path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Find Score provenance stored above a project-native model."""
    project_root = PROJECT_ROOT.resolve()
    try:
        model_path.resolve().relative_to(project_root)
    except ValueError:
        return None, None

    for directory in model_path.parents:
        provenance_path = directory / "score_provenance.json"
        if provenance_path.is_file():
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            return payload, provenance_path
        if directory == project_root:
            break
    return None, None


def safe_component(value: str, label: str) -> str:
    """Validate a value before using it as one directory component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not component:
        raise ValueError(f"{label} cannot be converted to a safe directory name")
    return component


def _parse_sr_formula(equation, names):
    """Accept only the arithmetic and real unary operators emitted by SR."""
    if not isinstance(equation, str) or len(equation) > 20000:
        raise ValueError("Unsupported symbolic-regression formula syntax")
    try:
        tree = ast.parse(equation, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Unsupported symbolic-regression formula syntax") from exc
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
               ast.Constant, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.Pow, ast.UAdd, ast.USub)
    nodes = list(ast.walk(tree))
    if len(nodes) > 2048:
        raise ValueError("Unsupported symbolic-regression formula syntax")
    functions = {"sqrt", "exp"}
    function_nodes = {id(node.func) for node in nodes if isinstance(node, ast.Call)}

    def numeric_exponent(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = numeric_exponent(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            denominator = numeric_exponent(node.right)
            if denominator != 0:
                return numeric_exponent(node.left) / denominator
        raise ValueError("Unsupported symbolic-regression formula operators")

    for node in nodes:
        if (not isinstance(node, allowed)
                or (isinstance(node, ast.Name) and (
                    node.id not in names
                    and not (id(node) in function_nodes and node.id in functions)))
                or (isinstance(node, ast.Constant) and (
                    type(node.value) not in (int, float) or not np.isfinite(float(node.value))))
                or (isinstance(node, ast.Call) and (
                    not isinstance(node.func, ast.Name) or node.func.id not in functions
                    or len(node.args) != 1 or node.keywords))):
            raise ValueError("Unsupported symbolic-regression formula syntax")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = numeric_exponent(node.right)
            if (exponent < 0 or exponent.p > 256 or exponent.q > 256
                    or exponent.q & (exponent.q - 1)):
                raise ValueError("Unsupported symbolic-regression formula operators")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if numeric_exponent(node.right) == 0:
                raise ValueError("Unsupported symbolic-regression formula operators")
    try:
        expression = sp.sympify(equation, locals={
            **{name: sp.Symbol(name) for name in names}, "sqrt": sp.sqrt, "exp": sp.exp,
        })
        for node in sp.preorder_traversal(expression):
            if not isinstance(node, (sp.Symbol, sp.Number, sp.Add, sp.Mul, sp.Pow, sp.exp)):
                raise ValueError("Unsupported symbolic-regression formula operators")
            if isinstance(node, sp.Number) and (node.is_real is not True or node.is_finite is not True):
                raise ValueError("Unsupported symbolic-regression formula operators")
            if isinstance(node, sp.Pow):
                exponent = node.exp
                if not exponent.is_number or not exponent.is_real:
                    raise ValueError("Unsupported symbolic-regression formula operators")
                rational = sp.Rational(str(exponent))
                # Square/root composition produces bounded nonnegative dyadic powers.
                if (rational < 0 or rational.p > 256 or rational.q > 256
                        or rational.q & (rational.q - 1)):
                    raise ValueError("Unsupported symbolic-regression formula operators")
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Unsupported symbolic-regression formula operators") from exc
    return sp.simplify(expression)


def _sr_unary_equations_match(raw, expected, symbols):
    """Extend the polynomial coefficient audit recursively through sqrt/exp."""
    unary_terms = []
    dummy_symbols = []

    def polynomial_matches(left, right):
        difference = sp.Poly(left - right, *symbols, *dummy_symbols)
        return np.allclose(np.array(difference.coeffs(), dtype=float), 0, rtol=0, atol=1e-8)

    def lower(node):
        is_root = isinstance(node, sp.Pow) and not node.exp.is_integer
        if node.func == sp.exp or is_root:
            operation = ("root", node.exp) if is_root else ("exp",)
            argument = lower(node.base if is_root else node.args[0])
            for old_operation, old_argument, dummy in unary_terms:
                if operation == old_operation and polynomial_matches(argument, old_argument):
                    return dummy
            dummy = sp.Dummy(f"sr_unary_{len(unary_terms)}")
            unary_terms.append((operation, argument, dummy))
            dummy_symbols.append(dummy)
            return dummy
        return node.func(*(lower(argument) for argument in node.args)) if node.args else node

    lowered_raw = lower(raw)
    lowered_expected = lower(expected)
    return polynomial_matches(lowered_raw, lowered_expected)


def load_prediction_model(method, campaign_round, model_path=None, allow_unbound_model=False,
                          *, sr_candidate_id=None):
    """Load frozen predictors and their ordered inputs; never refit any model."""
    artifacts = {}

    def read_json(path):
        artifacts[portable_path(path)] = file_sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def checked_file(relative, expected_hash):
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT.resolve()) or file_sha256(path) != expected_hash:
            raise ValueError(f"Invalid bound model artifact: {relative}")
        artifacts[portable_path(path)] = expected_hash
        return path

    campaign_dir = campaign_directory_name(campaign_round)
    if method == "xgb":
        path = model_path or resolve_model_path(None, campaign_round)
        names, contract, contract_path = load_model_contract(
            path, campaign_round, allow_unbound_model=allow_unbound_model
        )
        model = joblib.load(path)
        if contract["schema_version"] == 3:
            validate_seed_average_model(model, names)
        elif isinstance(model, VotingRegressor):
            raise ValueError("Seed-average XGBoost requires a schema 3 model contract")
        if getattr(model, "n_features_in_", len(names)) != len(names):
            raise ValueError("XGBoost feature count differs from its contract")
        descriptor_names = names

        def predict(frame):
            inputs = frame[names].astype(float)
            return model.predict(inputs), inputs

    elif method == "ga":
        path = find_latest_model(GA_OUTPUT_DIR / campaign_dir, "Stacking_best_model.pkl")
        contract_path = path.with_name("model_config.json")
        contract = read_json(contract_path)
        if contract["schema_version"] != 1 or contract["stacking_model_file"] != path.name:
            raise ValueError("Unsupported stacking contract schema")
        checked_file(portable_path(path), contract["stacking_model_sha256"])
        order = contract["model_order"]
        sources = {item["model"]: item for item in contract["sources"]}
        if len(order) < 2 or len(order) != len(set(order)) or set(order) != set(sources):
            raise ValueError("Stacking source models do not match its input order")
        bases = []
        for alias in order:
            source = sources[alias]
            base_path = checked_file(source["model_file"], source["model_sha256"])
            config = read_json(checked_file(source["feature_config_file"], source["feature_config_sha256"]))
            if (config["feature_config_schema_version"] != 3
                    or config["model_sha256"] != source["model_sha256"]
                    or config["target"] != TARGET_NAME
                    or config["score_version"] != contract["score_version"]
                    or config["score_manifest"]["campaign_round"] != campaign_round
                    or config["full_feature_names"] != FEATURE_COLUMNS
                    or config["structure_feature_extraction"] != EXTRACTION_SETTINGS
                    or config["cv_manifest"] != contract["cv_manifest"]
                    or config["split_manifest"] != contract["split_manifest"]
                    or config["training_data"] != contract["training_data"]):
                raise ValueError(f"Incompatible stacking base contract: {alias}")
            kept = config["retained_feature_names"]
            if not kept or len(kept) != len(set(kept)) or not set(kept).issubset(FEATURE_COLUMNS):
                raise ValueError(f"Invalid stacking features: {alias}")
            base = joblib.load(base_path)
            if getattr(base, "n_features_in_", len(kept)) != len(kept):
                raise ValueError(f"Stacking base feature count differs: {alias}")
            bases.append((base, kept))
        model = joblib.load(path)
        if getattr(model, "n_features_in_", len(order)) != len(order):
            raise ValueError("Stacking input count differs from its model order")
        names, descriptor_names = FEATURE_COLUMNS, []

        def predict(frame):
            inputs = pd.DataFrame(
                np.stack([describe_structure(s, halogen_proxy=True) for s in frame.SMILES]),
                columns=FEATURE_COLUMNS,
            )
            base_predictions = [np.asarray(base.predict(inputs[kept].to_numpy()), dtype=float)
                                for base, kept in bases]
            if not np.isfinite(np.column_stack(base_predictions)).all():
                raise ValueError("Stacking base returned non-finite predictions")
            return model.predict(np.column_stack(base_predictions)), inputs

    elif method == "sr":
        path = (Path(model_path) if model_path is not None else find_latest_model(
            SYMBOLIC_REGRESSION_OUTPUT_DIR / campaign_dir, "selected_candidate.json"))
        contract_path = path
        contract = read_json(path)
        scaler = read_json(path.with_name("scaler_params.json"))
        run_config = read_json(path.with_name("run_config.json"))
        results = read_json(path.with_name("results.json"))
        if contract["schema_version"] != 2:
            raise ValueError("Unsupported symbolic-regression candidate schema")
        names = scaler["feature_names"]
        configurable_features = "required_feature_count" in contract
        required_count = contract.get("required_feature_count", 6)
        if (not isinstance(names, list) or not names
                or len(set(names)) != len(names)
                or not set(names).issubset(SR_FEATURE_MAPPING)):
            raise ValueError("Symbolic regression requires unique supported candidate descriptors")
        if (type(required_count) is not int or not 1 <= required_count <= len(names)
                or (not configurable_features and len(names) != 6)
                or (configurable_features and any(
                    record.get("required_feature_count") != required_count
                    or type(record.get("required_feature_count")) is not int
                    for record in (run_config, results)
                ))):
            raise ValueError("Symbolic-regression required feature counts must match the saved run")
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        if mean.shape != (len(names),) or scale.shape != (len(names),) or not np.isfinite([mean, scale]).all() or (scale <= 0).any():
            raise ValueError("Invalid symbolic-regression scaler")
        if (run_config["campaign_round"] != campaign_round
                or results["campaign_round"] != campaign_round
                or run_config["score_version"] != contract["score_version"]
                or results["score_version"] != contract["score_version"]
                or run_config["target_column"] != TARGET_NAME
                or run_config["feature_columns"] != names or results["feature_names"] != names
                or results["final_candidate_position"] != contract["candidate_position"]
                or any(results[key] != contract[key] for key in ("equation_raw", "equation_scaled"))):
            raise ValueError("Symbolic-regression candidate does not match its saved run")
        audit = results["equation_audit"]
        audited_features = {item["feature"]: item for item in audit["features"]}
        if audit["target"]["standardized_again_by_symbolic_regression"] or not audit["numerical_consistency"]["accepted"] or any(
            audited_features[name]["standardization_mean"] != m
            or audited_features[name]["standardization_scale"] != s
            for name, m, s in zip(names, mean, scale)
        ):
            raise ValueError("Symbolic-regression scaler does not match its equation audit")
        selected_lists = [record.get("selected_feature_names")
                          for record in (run_config, results, contract)]
        if configurable_features or any(value is not None for value in selected_lists):
            selected = selected_lists[0]
            if (not isinstance(selected, list) or len(selected) != required_count
                    or len(set(selected)) != required_count
                    or selected != [name for name in names if name in selected]
                    or any(value != selected for value in selected_lists)):
                raise ValueError("Symbolic-regression selected descriptor lists must match the required ordered inputs")
        else:
            selected = list(names)
        if sr_candidate_id is not None:
            comparison = read_json(path.with_name("candidate_comparison.json"))
            review = read_json(path.with_name("final_candidate_cv.json"))
            eligible = {cid for layer in review["candidate_layers"].values()
                        for cid in layer["candidate_ids"]}
            if sr_candidate_id not in eligible:
                raise ValueError("Requested SR candidate is outside the saved CV display layers")
            record = next(row for row in review["shortlisted_candidates"]
                          if row["candidate_id"] == sr_candidate_id)
            display = next(row for row in comparison["candidates"]
                           if row["candidate_id"] == sr_candidate_id)
            if (record["outer_training_refit"]["status"] != "accepted"
                    or display["outer_training_refit_status"] != "accepted"
                    or record["equation_scaled"] != display["equation_scaled"]
                    or comparison["candidate_layers"] != review["candidate_layers"]):
                raise ValueError("Requested SR candidate does not match its full CV audit")
            scaled_symbols = _parse_sr_formula(display["equation_scaled"], names).free_symbols
            selected = [name for name in names if sp.Symbol(name) in scaled_symbols]
            if len(selected) != required_count:
                raise ValueError("Requested SR candidate has the wrong descriptor count")
            contract = {**contract,
                "automatic_candidate_position": contract["candidate_position"],
                "automatic_selection_rule": contract.get("selection_rule"),
                "automatic_final_candidate_cv": contract.get("final_candidate_cv"),
                "candidate_position": None, "candidate_id": sr_candidate_id,
                "candidate_origin": "explicit_saved_candidate",
                "source_cv_folds": record["source_cv_folds"],
                "term_budget": record["source_term_budget"],
                "structural_complexity": record["structural_complexity"],
                "excess_complexity": record["excess_complexity"],
                "has_direct_quadratic": record["source_term_budget"].get("has_direct_quadratic"),
                "equation_raw": display["equation_raw"],
                "equation_scaled": display["equation_scaled"],
                "selected_feature_names": selected,
                "selection_rule": "explicit_user_selection_from_saved_cv_layers",
                "prediction_contract": "saved full-precision raw/scaled candidate equations",
                "final_candidate_cv": {key: record[key] for key in (
                    "candidate_id", "cv_raw_r2_mean", "cv_raw_r2_standard_error",
                    "within_one_se", "within_two_se", "excess_complexity")},
            }
        raw = _parse_sr_formula(contract["equation_raw"], names)
        scaled = _parse_sr_formula(contract["equation_scaled"], names)
        symbols = [sp.Symbol(name) for name in selected]
        if raw.free_symbols != set(symbols) or scaled.free_symbols != set(symbols):
            raise ValueError("Symbolic-regression formulas must contain exactly the selected descriptors")
        selected_positions = [names.index(name) for name in selected]
        mean, scale = mean[selected_positions], scale[selected_positions]
        names = list(selected)
        descriptor_names = [SR_FEATURE_MAPPING[name] for name in names]
        if raw.is_polynomial(*symbols) and scaled.is_polynomial(*symbols):
            difference = sp.Poly(raw - scaled.subs({
                symbol: (symbol - m) / s for symbol, m, s in zip(symbols, mean, scale)
            }), *symbols)
            equations_match = np.allclose(
                np.array(difference.coeffs(), dtype=float), 0, rtol=0, atol=1e-8,
            )
        else:
            expected_raw = standardized_to_raw_expression(scaled, names, mean, scale)
            equations_match = _sr_unary_equations_match(raw, expected_raw, symbols)
        if not equations_match:
            raise ValueError("Raw/scaled symbolic formulas do not match the saved scaler")

        def predict(frame):
            inputs = frame[descriptor_names].astype(float)
            inputs.columns = names
            values = evaluate_expression(raw, inputs.to_numpy(), names)
            scaled_values = evaluate_expression(
                scaled, (inputs.to_numpy() - mean) / scale, names,
            )
            if not np.allclose(values, scaled_values, rtol=1e-9, atol=1e-9):
                raise ValueError("Raw/scaled symbolic posterior predictions differ")
            return values, inputs

    else:
        raise ValueError(f"Unknown posterior model: {method}")

    artifacts[portable_path(path)] = file_sha256(path)
    if contract_path is not None:
        artifacts[portable_path(contract_path)] = file_sha256(contract_path)
    provenance, provenance_path = load_score_provenance(path)
    if contract_path is None:
        provenance, provenance_path = None, None
    elif (provenance is None or provenance["campaign_round"] != campaign_round
          or (method != "xgb" and (contract["campaign_round"] != campaign_round
              or contract["score_version"] != provenance["score_version"]))):
        raise ValueError(f"Invalid {method} campaign or Score provenance")
    if provenance_path is not None:
        artifacts[portable_path(provenance_path)] = file_sha256(provenance_path)
    return {"predict": predict, "path": path, "contract": contract,
            "contract_path": contract_path, "provenance": provenance,
            "provenance_path": provenance_path, "feature_names": names,
            "descriptor_names": descriptor_names, "artifacts": artifacts}


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create isolated file and console logging for one prediction run."""
    logger = logging.getLogger("posterior_score_validation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "prediction.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close run-specific handlers so Windows releases the log file."""
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def predict_posterior(
    input_path: Path,
    model_path: Path | None,
    *,
    campaign_round: str,
    run_id: str,
    allow_unbound_model: bool,
    model_names: Sequence[str] = tuple(MODEL_LABELS),
    sr_run_directory: Path | None = None,
    sr_candidate_id: str | None = None,
) -> Path:
    """Predict through one model campaign and validate each experiment round separately."""
    if (not model_names or len(model_names) != len(set(model_names))
            or not set(model_names).issubset(MODEL_LABELS)):
        raise ValueError("Select each posterior model at most once: xgb, ga, sr")
    if allow_unbound_model and tuple(model_names) != ("xgb",):
        raise ValueError("Historical unbound models require --models xgb")
    if allow_unbound_model and model_path is None:
        raise ValueError("Historical unbound models require an explicit --model path")
    if (sr_run_directory is not None or sr_candidate_id is not None) and "sr" not in model_names:
        raise ValueError("An explicit SR run/candidate requires the sr model")
    predictors = {name: load_prediction_model(
        name, campaign_round,
        model_path if name == "xgb" else (
            Path(sr_run_directory) / "selected_candidate.json"
            if name == "sr" and sr_run_directory is not None else None),
        allow_unbound_model,
        **({"sr_candidate_id": sr_candidate_id} if name == "sr" and sr_candidate_id is not None else {})
    ) for name in model_names}
    feature_names = list(dict.fromkeys(
        column for item in predictors.values() for column in item["descriptor_names"]
    ))
    (
        frame,
        features,
        input_rounds,
        evaluated_rounds,
        input_row_count,
        reference,
    ) = load_posterior_data(
        input_path,
        feature_names,
        campaign_round,
    )
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain 1-64 ASCII letters, digits, dots, underscores, or hyphens"
        )

    primary = predictors[model_names[0]]
    model_path = primary["path"]
    model_contract, model_contract_path = primary["contract"], primary["contract_path"]
    score_provenance, score_provenance_path = primary["provenance"], primary["provenance_path"]
    score_model, score_scale_metadata = load_bound_score_scale(score_provenance, campaign_round)
    score_version = score_provenance["score_version"] if score_provenance else None
    for name, item in predictors.items():
        metadata = (score_scale_metadata if item is primary else
                    load_bound_score_scale(item["provenance"], campaign_round)[1])
        if any(metadata.get(key) != score_scale_metadata.get(key) for key in (
            "score_version", "score_model_sha256", "scores_sha256"
        )):
            raise ValueError("Posterior models are bound to different Score scales")
        values, inputs = item["predict"](frame)
        values = np.asarray(values)
        if values.shape != (len(frame),) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError(f"{name} returned invalid posterior predictions")
        item["predictions"], item["inputs"] = values.astype(float), inputs
        if name == "xgb":
            lower, upper, interval = score_prediction_intervals(
                item["path"], item["contract"], values,
                expected_campaign_round=campaign_round, expected_score_version=score_version,
            )
        else:
            lower = upper = np.full(len(frame), np.nan)
            interval = {"status": "unavailable-no-bound-consensus-oof-residuals"}
        item["interval"] = (lower, upper, interval)
    predictions = primary["predictions"]
    references = {}
    for round_name in evaluated_rounds:
        round_reference = reference.loc[reference["Round"].eq(round_name)]
        means, metadata = summarize_posterior_reference(round_reference)
        metadata.update({
            "reference_file": portable_path(input_path),
            "reference_subset_sha256": dataframe_sha256(
                round_reference.drop(columns="_Input_Order")
            ),
        })
        references[round_name] = (means, metadata)
    score_scale_metadata["posterior_references"] = {
        round_name: metadata for round_name, (_, metadata) in references.items()
    }
    interval_lower, interval_upper, interval_metadata = primary["interval"]
    if set(model_names) == set(MODEL_LABELS):
        mean_predictions = np.mean(
            [predictors[name]["predictions"] for name in MODEL_LABELS], axis=0
        )
        predictors["mean"] = {
            "predictions": mean_predictions,
            "inputs": pd.DataFrame(index=frame.index),
            "interval": (
                np.full(len(frame), np.nan), np.full(len(frame), np.nan),
                {"status": "unavailable-no-bound-ensemble-oof-residuals"},
            ),
            "feature_names": [],
            "contract": {
                "method": "unweighted-arithmetic-mean-of-predicted-scores",
                "component_models": list(MODEL_LABELS),
                "component_model_sha256": {
                    name: file_sha256(predictors[name]["path"]) for name in MODEL_LABELS
                },
                "score_version": score_version,
            },
            "artifacts": {},
        }

    output_dir = (
        POSTERIOR_PREDICTION_OUTPUT_DIR
        / f"model_{campaign_round}"
        / safe_component(run_id, "run_id")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(output_dir)

    experimental_summaries = {
        round_name: build_experimental_summary(
            frame.loc[frame["Round"].eq(round_name)].reset_index(drop=True),
            score_model, references[round_name][0],
        ) for round_name in evaluated_rounds
    }
    analyses, summaries, result_tables, statistic_tables = {}, {}, [], []
    comparison = frame[["Round", "Name", "SMILES"]].copy()
    comparison["Validation_Role"] = np.where(
        frame["Round"].eq(campaign_round), "same_round_model", "retrospective_model"
    )
    identity = pd.MultiIndex.from_frame(frame[["Round", "Name"]])
    for name, item in predictors.items():
        label = MODEL_LABELS.get(name, MEAN_MODEL_LABEL)
        lower, upper, interval = item["interval"]
        analyses[name] = {}
        for round_name in evaluated_rounds:
            positions = np.flatnonzero(frame["Round"].eq(round_name).to_numpy())
            reference_means, reference_metadata = references[round_name]
            role = ("same_round_model" if round_name == campaign_round
                    else "retrospective_model")
            analyses[name][round_name] = write_rank_correlation_analysis(
                output_dir / name / f"round_{round_name}" / "rank_correlation",
                frame.iloc[positions].reset_index(drop=True), item["predictions"][positions],
                round_name, lower[positions], upper[positions], interval,
                score_model, reference_means,
                {**score_scale_metadata, "posterior_reference": reference_metadata,
                 "prediction_model": f"{label} {round_name}"
                 + (" retrospective" if role == "retrospective_model" else ""),
                 "validation_role": role},
                experimental_summary=experimental_summaries[round_name],
            )
            statistics = analyses[name][round_name]["statistics"].copy()
            statistics.insert(0, "Model", label)
            statistics.insert(2, "Validation_Role", role)
            statistic_tables.append(statistics)
        summary = pd.concat(
            [analyses[name][round_name]["summary"] for round_name in evaluated_rounds],
            ignore_index=True,
        ).set_index(["Round", "Name"]).loc[identity].reset_index()
        summaries[name] = summary
        results = frame[["Round", "Name", "SMILES", "_Input_Order"]].rename(
            columns={"Round": "Sample_Round", "_Input_Order": "Input_Order"}
        ).copy()
        results.insert(0, "Model_Round", campaign_round)
        results.insert(0, "Model", label)
        results["Validation_Role"] = comparison["Validation_Role"].to_numpy()
        results["Predicted_Score"] = item["predictions"]
        for column in POSTERIOR_RESULT_MEASUREMENT_COLUMNS:
            results[column] = summary[column].to_numpy()
        for column in item["inputs"]:
            results[column] = item["inputs"][column].to_numpy()
        result_tables.append(results.sort_values(
            ["Sample_Round", "Predicted_Rank_All", "Name"], kind="stable"
        ))
        comparison[f"{label}_Score"] = item["predictions"]
        comparison[f"{label}_Rank"] = summary.Predicted_Rank_All.to_numpy()
        logger.info("%s prediction range: %.4f to %.4f", name,
                    item["predictions"].min(), item["predictions"].max())
    rank_analysis = analyses[model_names[0]][evaluated_rounds[-1]]
    measurement_summary = summaries[model_names[0]]
    comparison["Observed_Score"] = measurement_summary.Observed_Score.to_numpy()
    comparison["Observed_Rank"] = measurement_summary.Observed_Rank.to_numpy()
    comparison.to_csv(output_dir / "score_comparison.csv", index=False, encoding="utf-8")
    pd.concat(statistic_tables, ignore_index=True).to_csv(
        output_dir / "rank_correlation_statistics.csv", index=False, encoding="utf-8"
    )
    results = pd.concat(result_tables, ignore_index=True)

    predictions_path = output_dir / "predictions.csv"
    results.to_csv(predictions_path, index=False, encoding="utf-8", lineterminator="\n")

    effective_columns = [
        *METADATA_COLUMNS,
        *POSTERIOR_DESCRIPTOR_COLUMNS,
        *POSTERIOR_MEASUREMENT_COLUMNS,
    ]
    effective_subset = frame.loc[:, effective_columns]
    round_counts = {
        round_name: int((frame["Round"] == round_name).sum())
        for round_name in evaluated_rounds
    }
    manifest = {
        "schema_version": 7,
        "primary_model": model_names[0],
        "models": {name: {
            "feature_names": item["feature_names"], "contract": item["contract"],
            "artifacts": item["artifacts"], "score_prediction_interval": item["interval"][2],
            "analysis_artifacts": {round_name: {key: {
                "file": path.relative_to(output_dir).as_posix(), "sha256": file_sha256(path)
            } for key, path in analysis["paths"].items()}
                for round_name, analysis in analyses[name].items()},
        } for name, item in predictors.items()},
        "comparison_artifacts": {filename: file_sha256(output_dir / filename) for filename in (
            "score_comparison.csv", "rank_correlation_statistics.csv"
        )},
        "structure_feature_extraction": EXTRACTION_SETTINGS if "ga" in predictors else None,
        "structure_features_sha256": file_sha256(PROJECT_ROOT / "structure_features.py") if "ga" in predictors else None,
        "created_at": datetime.now().astimezone().isoformat(),
        "model_round": campaign_round,
        "input_sample_rounds": input_rounds,
        "evaluated_sample_rounds": evaluated_rounds,
        "ignored_other_rounds": [
            value for value in input_rounds if value not in evaluated_rounds
        ],
        "input_row_count": int(input_row_count),
        "evaluated_row_count": int(len(frame)),
        "row_count_by_round": round_counts,
        "target": TARGET_NAME,
        "feature_names": primary["feature_names"],
        "input_file": portable_path(input_path),
        "input_sha256": file_sha256(input_path),
        "effective_input_sha256": dataframe_sha256(effective_subset),
        "model_file": portable_path(model_path),
        "model_sha256": file_sha256(model_path),
        "model_contract": model_contract,
        "model_contract_file": (
            portable_path(model_contract_path) if model_contract_path else None
        ),
        "model_contract_sha256": (
            file_sha256(model_contract_path) if model_contract_path else None
        ),
        "score_version": (
            score_provenance.get("score_version") if score_provenance else None
        ),
        "score_provenance_file": (
            portable_path(score_provenance_path) if score_provenance_path else None
        ),
        "score_provenance_sha256": (
            file_sha256(score_provenance_path) if score_provenance_path else None
        ),
        "prediction_file": predictions_path.name,
        "prediction_sha256": file_sha256(predictions_path),
        "prediction_summary": {
            "minimum": float(predictions.min()),
            "maximum": float(predictions.max()),
            "mean": float(predictions.mean()),
            "standard_deviation_ddof0": float(predictions.std(ddof=0)),
        },
        "score_prediction_interval": interval_metadata,
        "experimental_score": score_scale_metadata,
        "posterior_validation_analysis": {
            "analysis_version": rank_analysis["manifest"]["analysis_version"],
            "measurement_complete_rows": int(
                measurement_summary["Measurement_Complete"].sum()
            ),
            "rounds": {round_name: {
                "validation_role": ("same_round_model" if round_name == campaign_round
                                    else "retrospective_model"),
                "measurement_complete_rows": int(
                    analysis["summary"]["Measurement_Complete"].sum()
                ),
            } for round_name, analysis in analyses[model_names[0]].items()},
        },
        "script_file": Path(__file__).name,
        "script_sha256": file_sha256(Path(__file__)),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info("Posterior validation completed for %d samples", len(frame))
    logger.info("Model campaign: %s", campaign_round)
    logger.info("Evaluated sample rounds: %s", ", ".join(evaluated_rounds))
    logger.info("Model: %s", model_path)
    logger.info("Feature order: %s", primary["feature_names"])
    logger.info("Prediction range: %.4f to %.4f", predictions.min(), predictions.max())
    logger.info(
        "Complete experimental measurement rows: %d",
        int(measurement_summary["Measurement_Complete"].sum()),
    )
    logger.info(
        "Score prediction interval: %s",
        interval_metadata.get("status", "unknown"),
    )
    logger.info("Rank-correlation outputs: %s", output_dir)
    logger.info("Results: %s", predictions_path)
    logger.info("Manifest: %s", manifest_path)
    close_logger(logger)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface with dynamic campaign flags."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate posterior Score within one campaign using XGBoost, GA "
            "stacking, symbolic regression and their mean predicted Score."
        )
    )
    add_campaign_arguments(parser)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_LABELS),
                        default=list(MODEL_LABELS),
                        help="Models to evaluate; default: xgb ga sr and their mean Score.")
    parser.add_argument("--sr-run", help="Optional saved SR run directory below outputs/symbolic_regression/.")
    parser.add_argument("--sr-candidate", help="Explicit candidate ID from the saved one-SE/two-SE layers.")
    parser.add_argument(
        "--input",
        help="Project-local posterior CSV below data/; defaults to posterior_validation.csv.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional XGBoost model path. By default, the newest selected-feature "
            "model inside the selected campaign is used."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique output run identifier; defaults to a local timestamp.",
    )
    parser.add_argument(
        "--allow-unbound-model",
        action="store_true",
        help=(
            "Explicitly allow a historical model without a native model contract; "
            "the verified six-feature order is then recorded as unbound."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the posterior prediction command."""
    args = build_parser().parse_args(argv)
    campaign_round = resolve_campaign_round(args.campaign_round)
    input_path = resolve_input_path(args.input)
    if args.model and "xgb" not in args.models:
        raise ValueError("--model specifies XGBoost and requires --models xgb")
    model_path = resolve_model_path(args.model, campaign_round) if "xgb" in args.models else None
    sr_run = None
    if args.sr_run:
        sr_run = Path(args.sr_run)
        sr_run = (sr_run if sr_run.is_absolute() else PROJECT_ROOT / sr_run).resolve()
        if not sr_run.is_relative_to((SYMBOLIC_REGRESSION_OUTPUT_DIR / campaign_directory_name(campaign_round)).resolve()):
            raise ValueError("--sr-run must be inside the selected SR campaign")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = predict_posterior(
        input_path,
        model_path,
        campaign_round=campaign_round,
        run_id=run_id,
        allow_unbound_model=args.allow_unbound_model,
        model_names=args.models,
        sr_run_directory=sr_run,
        sr_candidate_id=args.sr_candidate,
    )
    print(f"Posterior validation outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
