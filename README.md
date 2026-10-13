# PVK_Lewis_base_ML

[English](README.md) | [Chinese](README.cn.md)

Machine-learning workflows for calibrating, modeling, interpreting, and screening Lewis-base additives for perovskites.

This repository accompanies a manuscript currently under review. Citation and DOI information will be updated after publication.

## Repository layout

```text
.
├── data/
│   ├── calibration.csv
│   ├── data-s.csv
│   ├── data.csv
│   ├── device/                    # Raw device JV tables
│   └── posterior_validation.csv
├── outputs/                       # Generated at runtime
├── device_normalization.py        # Device normalization and box plots
├── device_score_validation.py     # Score and best-device-PCE rank validation
├── score_calibration.py           # Experimental Score calibration
├── data_preparation.py            # Shared runtime data preparation
├── campaign.py                    # Rxx campaign selection
├── project_paths.py               # Repository-local paths
├── GA_EN_KRR_BR_Lewis_base.py     # EN/KRR/BR training
├── GA_RF_XGB_Lewis_base.py        # RF/XGB training
├── GA_Lewis_base.py               # Genetic-algorithm screening
├── XGB_Lewis_base.py              # XGBoost interpretation and SHAP
├── posterior_score_validation.py   # Four-way posterior Score validation
├── posterior_validation_analysis.py
├── SR_Lewis_base.py               # Symbolic regression
├── sr_progress.py                 # Read-only SR progress viewer
├── sr_efficiency.py               # Symbolic-regression search helpers
├── sr_validation.py               # Splits, metrics, and Pareto selection
├── sr_reporting.py                # Raw equations and coefficient uncertainty
├── output_files_summary.md
└── requirements.txt
```

All paths are resolved from the repository directory. The project can be copied to another location and run without editing source paths.

Run logs and feature-name text files use UTF-8. Select UTF-8 when opening them in a text editor to display scientific symbols and Chinese paths correctly.

## Installation

```bash
git clone https://github.com/Yang-happylab/PVK_Lewis_base_ML.git
cd PVK_Lewis_base_ML
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS.

`SR_Lewis_base.py` additionally requires Julia and a compatible PySR 1.x installation.

## Input data

| File | Purpose |
|---|---|
| `data/calibration.csv` | Cumulative experimental data used to calibrate Score |
| `data/data-s.csv` | `Name`, canonical two-dimensional `SMILES`, 32 numerical structure features, and three DFT descriptors retained for Score consistency checks |
| `data/data.csv` | Named physicochemical and DFT descriptors for XGBoost and symbolic regression, including stable `Name` values |
| `data/device/<Name>.csv` | Raw R01 device-level JV parameters; files without `_norm` are authoritative |
| `data/posterior_validation.csv` | Posterior `Name`/`SMILES`, descriptors, and PLQY/XRD measurements by experimental round |

`Score` is generated at runtime and joined by `Name`. It is not stored in `data.csv` or `data-s.csv`. Derived tables, including normalized device data, are written under `outputs/`.

In `data/data.csv`, `HOMO`, `LUMO`, and `gap` are expressed in eV. `Coordination_atom` supplies elemental counts for `atom_N`, `atom_O`, `atom_S`, and `atom_Cl`; upper- and lower-case element labels are combined. `RingNum` counts ring-based sites that can coordinate simultaneously, rather than all rings in the molecular structure.

## Campaign selection

Campaign-aware scripts accept flags such as `--R01` and `--R02`. Without a round flag, the newest valid round in `data/calibration.csv` is selected.

Calibration accumulates rounds through the selected campaign; posterior validation predicts all available molecules through that campaign and evaluates each experimental round separately.

## Workflows

### Score calibration

Creates the versioned Score table, calibration parameters, provenance, and descriptor-correlation figures.
Each included round is fitted independently. Its endpoint predictions are Z-scored on the same full molecular library, then batch Scores are averaged equally without further rescaling. `batch_scores.csv` preserves individual batch Scores; `batch_agreement.csv` reports descriptive library rank agreement. Six plots per batch are stored under `batches/Rxx/`; root-level plots show the latest included batch only. Downstream models must use the selected Score version.

Each `Round` is one batch. PLQY and XRD are scaled to `100 * raw value / same-round Ref-PbI2 mean` before the existing fits and Score calculation. Each included round requires three positive Ref-PbI2 measurements per endpoint with `Use_for_fit=true`. Inputs retain raw values; `normalized_calibration.csv` records raw values, reference means, and relative values. Predicted endpoints and plot percentages are relative to the reference (100), not absolute PLQY or XRD values.

```bash
python score_calibration.py
python score_calibration.py --R01
```

Output: `outputs/score_calibration/campaign_Rxx/<score_version>/`

### Structure models and genetic algorithm

Trains the five structure models used as GA fitness functions from 32 numerical descriptors, then screens candidate Lewis-base structures.

```bash
python GA_EN_KRR_BR_Lewis_base.py --R01
python GA_RF_XGB_Lewis_base.py --R01
python GA_Lewis_base.py --R01
```

Run both training scripts before the GA and use the same campaign round for all three commands.
`structure_features.py` defines the ordered 32 features.

After GA completes, `top10_structures.png` shows the Top 10 molecular structures in two columns, with rank and Score matching `optimal_samples.csv`.

`structure_selection.py` uses RFE for ElasticNet, BayesianRidge, RF and XGB, and backward selection for KRR. Both outer evaluation and inner selection use 4-fold × 3-repeat CV. Each fit tunes all features, compares up to two near-best configurations with distinct predictive parameters, selects a feature subset using the one-standard-error rule, then retunes that subset. Each tuning stage samples up to 588 configurations for RF, 1288 for XGB, 988 for EN/BR and 788 for KRR. Inner scores are selection diagnostics; outer CV evaluates the complete procedure. The external test set is evaluated after selection. RF/XGB trees use one thread per parallel search job.

GA eligibility uses outer nested-CV metrics. Eligible models supply per-molecule nested OOF predictions to a Ridge Stacking model; GA fitness is its predicted Score. The DUPLEX test set is excluded from Stacking training. Base models remain fixed during the GA run. Model extraction settings must match the structure feature extractor.

Each base model saves selection diagnostics and OOF predictions under `feature_selection/`. The GA run saves the Stacking model, metrics, predictions, coefficients and plots under `stacking/`. Ridge CV scores are tuning diagnostics; `duplex_test` metrics evaluate the complete fitted stack on the held-out split.

The RF/XGB parameter search ranges are configured in `GA_RF_XGB_Lewis_base.py`, covering tree capacity, sampling and regularization.

Canonical SMILES without stereochemistry identifies molecules; features do not define identity. The GA saves all SMILES-unique candidates encountered across generations in `candidate_archive.csv` and selects `optimal_samples.csv` from this historical archive, including the final population. Both tables include `SMILES`, `Center`, and `Arm1`–`Arm4`; equivalent assembly routes are merged, while distinct structures with identical descriptors are retained.

The 31 center templates support 11 arm choices. `H` leaves the attachment position unsubstituted. NH3, H2O, H2S, and HCl are valid designs but have unsupported donor environments; each run lists them in `excluded_structures.csv`.

RDKit is pinned in `requirements.txt`; its version and extraction settings are recorded with model artifacts. Four descriptors are constant in the current library and are handled by the training variance filter.

Outputs:

- `outputs/structure_models/campaign_Rxx/`
- `outputs/genetic_algorithm/campaign_Rxx/`

### XGBoost interpretation

Runs one joint XGBoost search over 58,888 unique candidates, distributed as evenly as possible across feature counts 4, 6, 8, 10, and all available features (currently 13). With seed 42, 60% of candidates come from the focused branch and 40% from the exploration branch; tree depths concentrate on 2–8 while retaining some depth-10 and depth-12 candidates. Each 4-fold × 3-repeat training-CV fold fits one fixed baseline XGBoost model (200 trees, depth 4, learning rate 0.05, seed 42) on that fold's training rows. TreeExplainer mean absolute SHAP values on the same training rows define one shared feature ranking; each requested count takes its top features and restores their original column order before fitting candidates.

The top two candidates at each feature count are reranked by an additional 4-fold × 5-repeat review with seed 2026. The one-standard-error rule uses repeat-mean uncertainty: among results within the best mean minus its SE, it selects the smallest feature count and then the highest mean. The best reviewed all-feature candidate supplies the companion `full_features` model; the final `selected_features` winner may also use all features. The external test set is evaluated only after selection. Search diagnostics are saved under `Score/joint_search/`. Outputs include the full-feature and selected-feature models, plots, contracts, and fixed-final-model OOF SHAP artifacts. Final-model importance values and feature-Score scatter ordering use training-set mean absolute SHAP, while consensus OOF SHAP is the primary interpretation. All search, selection, fitting, and interpretation logic is implemented in `XGB_Lewis_base.py`.

```bash
python XGB_Lewis_base.py
python XGB_Lewis_base.py --R01
```

Output: `outputs/xgb_interpretability/campaign_Rxx/`

### Posterior model validation

`posterior_score_validation.py` applies XGBoost, GA Stacking, symbolic regression and their equally weighted mean predicted Score to all posterior molecules through the selected model campaign. The models share a bound Score version. Each experimental round uses its own `Name=REF` row and completed triplicates to calculate experimental Score on that fixed scale. Rankings and plots are reported separately by round; earlier-round results from a later model are retrospective.

```bash
python posterior_score_validation.py --R02
python posterior_score_validation.py --R01 --models ga sr
```

The command and `predict_posterior` function default to `xgb ga sr`, producing four prediction groups including their mean Score; `--models` selects a subset, without the three-model mean. Models belong to the selected campaign; model Rxx evaluates all available rounds through Rxx without pooling rank statistics. Without a round flag, the latest calibration round is selected.

Output: `outputs/posterior_predictions/model_Rxx/`

### Symbolic regression

Searches for an interpretable Score equation in which all six descriptors remain present. The publication grammar treats every descriptor symmetrically and uses the domain-safe operators `+`, `-`, `*`, and `square`. One linear term and one direct square per descriptor carry no excess-complexity penalty; interactions, higher powers, repeated nonlinear terms, and nested transforms are penalized during validation-based selection.

The default run attempts 40 unique, fixed-seed configurations under repeated cross-validation. Each fold fits its scaler only on fold-training rows. Pareto candidates are addressed by their explicit row position and selected from validation predictions with a recorded one-standard-error rule; the independent outer test set is used only once after refinement. Reports distinguish raw R-squared/RMSE/MAE from selection and complexity fields, and include a same-split Ridge baseline.

Both the standardized-feature equation and the directly usable raw-unit equation are first-class outputs.

```bash
python SR_Lewis_base.py
python SR_Lewis_base.py --R01
```

An interrupted run can be resumed explicitly from its existing run directory:

```bash
python SR_Lewis_base.py --resume-run outputs/symbolic_regression/campaign_R01/<run_directory>
```

Resume is deliberately configuration-level. The program recomputes the governed inputs from `data/`, verifies a strict contract covering the data, Score version, split, sampled configurations, scientific implementation files and settings, runtime layout, and package versions, and then skips only validated terminal configurations. An interrupted configuration is rerun, and an interrupted refinement starts as a clean, separately named attempt. The CV selection is saved before fold workers are stopped. Worker shutdown waits up to 60 seconds by default and force-cleans only idle processes that still fail to exit; change that grace period with `SR_FOLD_SHUTDOWN_GRACE_SECONDS` if needed.

In another terminal, inspect the newest run once or follow it continuously
without importing PySR or starting another Julia runtime:

```bash
python sr_progress.py
python sr_progress.py --watch 10
```

The viewer reports the exact current configuration out of 40, active fold worker(s), chunk,
requested iteration budget, validation R-squared, measured effective CPU-core
use, configured/actual Julia threads, and stop decision. New runs update a
compact atomic `progress_status.json` after every chunk. The viewer can also
read existing runs that only contain the detailed `search_progress.jsonl`.

Optional runtime settings:

```powershell
$env:SR_JULIA_THREADS = "32"
$env:SR_FOLD_WORKERS = "4"
$env:SR_FOLD_WORKER_THREADS = "8"
$env:SR_BLAS_THREADS = "1"
$env:SR_FOLD_SHUTDOWN_GRACE_SECONDS = "60"
$env:SR_REFINEMENT_TIMEOUT_SECONDS = "7200"
$env:SR_COEFFICIENT_BOOTSTRAPS = "200"
python SR_Lewis_base.py --R01
```

On an i9-13900KF-class workstation, the default CV scheduler runs four isolated fold processes with 8 Julia threads each (`4x8`), while refinement may use the full resolved `SR_JULIA_THREADS` budget. BLAS helper pools remain single-threaded and the resolver rejects layouts whose aggregate Julia threads exceed the logical processors visible to Windows or WSL. Set `SR_FOLD_WORKERS=2` and `SR_FOLD_WORKER_THREADS=16` to use two fold workers, or use `1` and `32` to disable fold-level process parallelism. Logs report wall time, effective cores, configured/actual threads, and each active fold because Task Manager occupancy alone is not a throughput measurement. `SR_DETERMINISTIC=1` forces the slower one-worker, one-thread strict deterministic mode. The workflow never changes the operating-system power plan.

Paired-bootstrap candidate uncertainty is calculated only at a fold's terminal checkpoint; intermediate checkpoints retain finite-value, six-feature, validation, and Pareto audits without repeatedly paying that bootstrap cost. A CV plateau is driven by the historical best validation-R-squared upper envelope across the explicit excess-complexity budgets. Improving training loss alone cannot reset that plateau. Refinement has no fold validation data and therefore retains its bounded outer-training loss-frontier rule. The configured ceilings are 600 iterations/500 seconds per fold and 3,000 iterations/7,200 seconds for refinement.

Output: `outputs/symbolic_regression/campaign_Rxx/`

### Device normalization and plots

Normalizes each raw molecule file by dividing all device rows by the corresponding metric mean across all `Control` rows in that file. Files containing `_norm` in their names are never read as inputs. The current device data contain R01 only.

Available raw CSVs are processed independently. Rows missing Group or any finite JV metric are skipped with CSV row numbers; files with missing columns, invalid labels, or no valid Control baseline are skipped with a reason. Summaries contain only successfully processed files. The round argument labels outputs; it does not filter raw files by experimental round.

```bash
python device_normalization.py --R01
```

Outputs:

- `outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv`
- `outputs/device_analysis/campaign_Rxx/<Name>/{PCE,Voc,FF,Jsc}_boxplot.png`
- `outputs/device_analysis/campaign_Rxx/group_summary.csv`
- `outputs/device_analysis/campaign_Rxx/normalization_summary.csv`

### Device Score validation

Sample size and rank statistics appear inside the upper-left corner of the scatter plot.

Compares the versioned calibrated Score with each molecule's observed best non-Control normalized PCE. Run device normalization first: this workflow reads the existing `group_summary.csv` directly and uses `normalization_summary.csv` only to recover the corresponding raw PCE mean. It does not repeat normalization.

```bash
python device_score_validation.py --R01
```

Outputs:

- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/best_concentration_summary.csv`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/rank_correlation_statistics.csv`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/score_pce_rank_comparison.png`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/score_vs_best_normalized_pce.png`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/analysis_manifest.json`

## Shared modules

`data_preparation.py`, `campaign.py`, `project_paths.py`, `sr_efficiency.py`, `sr_validation.py`, and `sr_reporting.py` are imported automatically by the entry-point scripts and are not intended to be run directly. `sr_progress.py` is a standalone, read-only diagnostic entry point.

## Outputs and version control

All generated models, logs, tables, figures, and analysis summaries are written under `outputs/`.

See [output_files_summary.md](output_files_summary.md) for the complete artifact list.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the associated paper:

> Author names. *Title of the paper*. Journal, year (under review).
> Code: [https://github.com/Yang-happylab/PVK_Lewis_base_ML](https://github.com/Yang-happylab/PVK_Lewis_base_ML)
