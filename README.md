# PVK_Lewis_base_ML

[English](README.md) | [Chinese](README.cn.md)

Machine-learning workflows for calibrating, modeling, interpreting, and screening Lewis-base additives for perovskites.

This repository accompanies a manuscript under review. Citation and DOI information will be updated after publication.

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
├── structure_training.py          # Shared structure-training splits and plots
├── rank_statistics.py             # Shared rank permutation statistics
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

Use Python 3.12.3. Project dependency versions are pinned in `requirements.txt`.

```bash
git clone https://github.com/Yang-happylab/PVK_Lewis_base_ML.git
cd PVK_Lewis_base_ML
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

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

In `data/data.csv`, `HOMO`, `LUMO`, and `gap` are expressed in eV. `N_lig` is the maximum number of N/O/S/Cl/Br donor atoms that can coordinate simultaneously to one Pb in a single coordination mode. `Coordination_atom` describes that same mode; `atom_N`, `atom_O`, `atom_S`, and `atom_Cl` count its donors by element, combining upper- and lower-case labels. `atom_Cl` counts Cl only. There is no Br count column, so the four elemental counts can sum to less than `N_lig`. `RingNum` counts distinct heterocycles contributing donors in that same mode, rather than all molecular rings. These coordination descriptors are separate from the GA structure features' candidate-donor counts.

## Campaign selection

Campaign-aware scripts accept flags such as `--R01` and `--R02`. Without a round flag, the newest valid round in `data/calibration.csv` is selected.

Calibration accumulates rounds through the selected campaign; posterior validation predicts all available molecules through that campaign and evaluates each experimental round separately.

## Workflows

### Score calibration

Creates the versioned Score table, calibration parameters, provenance, and descriptor-correlation figures.
Each included round is fitted independently. Its endpoint predictions are Z-scored on the same full molecular library, then batch Scores are averaged equally without further rescaling. `batch_scores.csv` preserves individual batch Scores; `batch_agreement.csv` reports descriptive library rank agreement. Six plots per batch are stored under `batches/Rxx/`; root-level plots show the latest included batch only. Downstream models must use the selected Score version.

Each `Round` is one batch. PLQY and XRD are scaled to `100 * raw value / same-round Ref-PbI2 mean` before fitting and Score calculation. Each included round requires three positive Ref-PbI2 measurements per endpoint with `Use_for_fit=true`. Inputs retain raw values; `normalized_calibration.csv` records raw values, reference means, and relative values. Predicted endpoints and plot percentages are relative to the reference (100), not absolute PLQY or XRD values.

```bash
python score_calibration.py
python score_calibration.py --R01
```

Output: `outputs/score_calibration/campaign_Rxx/<score_version>/`

### Structure models and genetic algorithm

Trains five structure models from 32 numerical descriptors and combines eligible models for genetic screening of Lewis-base structures.

```bash
python GA_EN_KRR_BR_Lewis_base.py --R01
python GA_RF_XGB_Lewis_base.py --R01
python GA_Lewis_base.py --R01
```

Run both training scripts before the GA and use the same campaign round for all three commands.
`structure_features.py` defines the ordered 32 features.

After GA completes, `top10_structures.png` shows the Top 10 molecular structures in two columns, with rank and Score matching `optimal_samples.csv`.

`structure_selection.py` uses a nested tune–select–retune procedure: tune all features, compare near-best configurations with distinct predictive parameters, select a feature subset by the one-standard-error rule, then retune that subset. ElasticNet, BayesianRidge, RF and XGB use RFE; KRR uses backward selection. Outer evaluation and inner selection both use 4-fold × 3-repeat CV. Preprocessing, feature selection and tuning stay within the corresponding training rows. Inner scores guide selection, outer CV evaluates the complete procedure, and the external DUPLEX test set is evaluated after selection.

GA eligibility uses outer nested-CV metrics. Eligible models supply per-molecule nested OOF predictions to a Ridge Stacking model; GA fitness is its predicted Score. The DUPLEX test set is excluded from Stacking training. Base models remain fixed during the GA run. Model extraction settings must match the structure feature extractor.

Each base model saves selection diagnostics and OOF predictions under `feature_selection/`. The GA run saves the Stacking model and its evaluation under `stacking/`. Ridge CV scores guide tuning; held-out test metrics evaluate the fitted stack.

Canonical SMILES without stereochemistry identifies molecules. The GA saves all unique structures encountered across generations in `candidate_archive.csv` and selects `optimal_samples.csv` from this archive, including the final population. Assembly routes are retained; equivalent routes are merged, while distinct structures with identical descriptors remain separate.

The 31 center templates support 11 arm choices. `H` leaves the attachment position unsubstituted. NH3, H2O, H2S, and HCl are valid designs but have unsupported donor environments; each run lists them in `excluded_structures.csv`.

RDKit is pinned in `requirements.txt`; its version and extraction settings are recorded with model artifacts. Four descriptors are constant in the molecular library and are handled by the training variance filter.

Outputs:

- `outputs/structure_models/campaign_Rxx/`
- `outputs/genetic_algorithm/campaign_Rxx/`

### XGBoost interpretation

`XGB_Lewis_base.py` first creates the stratified DUPLEX outer split, then determines the high-correlation feature filter using only outer-training rows and applies the same retained columns to the test set. The full-library correlation heatmap is descriptive. Each XGBoost fit uses five models with seeds 8, 18, 188, 1888 and 8888, identical parameters and features, and equally averaged predictions. Feature count and model parameters are selected jointly using 4-fold × 3-repeat CV, with the same folds for all five members and R² calculated from their mean prediction. Each fold fits a fixed baseline ensemble on its training rows; mean absolute ensemble SHAP on those same rows defines the feature ranking shared by its candidates.

Leading search candidates and saved candidates that pass compatibility checks undergo a common 4-fold × 5-repeat review. Every candidate is evaluated anew. Within one standard error of the best mean, selection favors fewer features, shallower trees and fewer trees, followed by higher CV performance. The same simplicity rule selects the companion `full_features` model; `selected_features` may also retain all available features. The outer test set remains locked until selection is complete.

Consensus OOF SHAP provides the primary interpretation: signed member SHAP values are averaged within each fold ensemble, then across the three held-out repeats for each outer-training molecule. The same ensemble OOF predictions supply the consensus residuals for posterior Score interval calibration. Seed-specific OOF values and ranking stability are saved as diagnostics. SHAP from the final fitted ensemble and its feature–Score scatter ordering use training-set mean absolute ensemble SHAP and provide supplementary interpretation. Search diagnostics are saved under `Score/joint_search/`; model and interpretation outputs are grouped by model stage.

```bash
python XGB_Lewis_base.py
python XGB_Lewis_base.py --R01
```

Output: `outputs/xgb_interpretability/campaign_Rxx/`

### Symbolic regression

`SR_Lewis_base.py` searches for an interpretable Score equation containing exactly the requested number of distinct descriptors after simplification. Its default inputs are `atom_N`, `gap`, `fai_min`, `μ`, `HOMO`, and `TPSA`, read directly from `data/data.csv`, with all six required in the formula. A descriptor may appear only in a nonlinear or interaction term; a linear term is optional. `--features` configures the candidate inputs using supported raw names or internal aliases, and `--selected-count` sets the required count from one through the candidate-pool size (default six). Aliases must resolve to unique features. Missing, nonnumeric or nonfinite required values or Score reject the input with the affected `Name` and column.

SR uses a deterministic stratified DUPLEX outer split calculated from the same ordered descriptor inputs used for formula fitting: by default, the six descriptors listed above. Sixteen search configurations are compared by 4-fold × 2-repeat CV on outer-training rows, with each scaler fitted only on its fold's training rows. Only complete CV results enter configuration-level one-standard-error selection. The operators `+`, `-`, `*`, `square`, `sqrt`, and `exp` treat all descriptors symmetrically. Each descriptor has one free linear term and one free nonlinear use: a direct square, real square root, exponential, or pairwise interaction. An interaction uses both descriptors' allowances; additional uses, higher powers and composite transforms consume the excess-complexity budget. Affine arguments are allowed for roots and exponentials. Invalid real-root domains and nonfinite predictions are rejected without absolute-value or clipping substitutions.

Eligible equations are retained across search chunks. One refinement inherits the winning configuration's CV equations, converts their scaling to the outer-training basis and refits their original floating constants on outer-training rows. Inherited and refinement equations share one archive. A common excess-complexity ceiling, configurable with `--max-excess-complexity` (default five), applies throughout CV search, inheritance, refinement, candidate checks and final review. CV plateau stopping follows the best eligible validation frontier accumulated across search chunks. Refinement checks its accumulated eligible outer-training loss and simplicity frontier every 25 iterations. After 200 warm-up iterations, it stops following 100 iterations without a meaningful loss reduction at equal or lower complexity, or a simpler formula at comparable loss. The two-hour limit also applies. These checks control stopping only; final formula selection uses the joint candidate review.

The joint review deduplicates inherited and refinement structures and applies fixed-structure 4-fold × 5-repeat CV on outer-training rows, using separate splits with seed 2026. Each fold fits its own scaler and refits the original floating constants from two neutral starts while preserving the formula's structure; affine root offsets may be initialized for feasibility using that fold's training inputs alone. Every refitted candidate must satisfy the exact variable count, finite-value checks and shared complexity limit. Within one standard error of the highest CV mean, official selection favors lower excess complexity, fewer combined terms, fewer independent parameters, then lower structural complexity. The standard error is calculated from repeat means. This review is a conditional selection diagnostic for candidate structures discovered by search and refinement, reported separately from configuration CV.

The one-SE and two-SE display layers each contain at most ten candidates, using the same highest-mean/standard-error anchor. They retain performance and simplicity representatives, share candidate IDs and merge algebraically equivalent fitted equations. Only shortlisted candidates receive outer-training constant refits. `candidate_comparison.json`, `.csv`, and `.txt` report full-precision standardized/raw equations, CV and training metrics, repeat/fold diagnostics, complexity and refit status. Candidate eligibility and automatic one-SE selection use training-side CV only; two-SE candidates are alternatives.

After the shortlist and automatic selection are fixed, `candidate_results/<candidate_id>/` exports each unique displayed candidate's equations, CV/train/test metrics, Name-linked predictions, prediction-versus-actual and residual plots, residual histogram, and a combined prediction/residual figure. Shared candidates are exported once. `summary.csv` supports comparison of predictive performance, simplicity and chemical interpretability; these diagnostics do not automatically change the selected model. Failed refits or invalid predictions retain an explicit status. Figures use Arial and common axis limits across candidates. A completed run can export these results without training or Julia:

```bash
python sr_reporting.py --run-directory outputs/symbolic_regression/campaign_R02/<run_directory>
```

The exporter preserves existing model files and refuses to overwrite an existing `candidate_results` directory. Standalone plots use the XGBoost 10-by-8-inch canvas with ordinary axis aspect; square paper derivatives require an explicit request.

Linear Ridge and additive quadratic Ridge provide diagnostic baselines using the final SR variables and the same outer split. The quadratic baseline includes each variable's linear and squared terms, with no interactions. Each baseline tunes its own regularization strength by training-side CV with fold-training scalers; these CV scores are conditional tuning diagnostics. Outer-test evaluation follows fixed equations and baselines; test metrics do not enter automatic selection. Outputs include standardized and raw-unit equations, model metrics, candidate diagnostics, predictions and provenance. See [output_files_summary.md](output_files_summary.md) for artifact details.

```bash
python SR_Lewis_base.py
python SR_Lewis_base.py --R01
python SR_Lewis_base.py --R02
```

Resume an interrupted run from its explicit directory:

```bash
python SR_Lewis_base.py --resume-run outputs/symbolic_regression/campaign_R01/<run_directory>
```

Resume requires matching inputs, Score version, candidate features, required count, split, implementation and run settings. Validated completed configurations are reused; interrupted work is rerun. Worker failures identify the affected fold and report the error text.

Inspect the newest run or follow progress in another terminal:

```bash
python sr_progress.py
python sr_progress.py --watch 10
```

The viewer reads saved progress without importing PySR or starting Julia.

Output: `outputs/symbolic_regression/campaign_Rxx/`

### Posterior model validation

`posterior_score_validation.py` applies XGBoost, GA Stacking, symbolic regression and their equally weighted mean predicted Score to all posterior molecules through the selected model campaign. The models share a bound Score version. Each experimental round uses its own `Name=REF` row and completed triplicates to calculate experimental Score on that fixed scale. Rankings and plots are reported separately by round; earlier-round results from a later model are retrospective.

Automatic XGBoost discovery considers only completed models in the selected campaign with valid Score and consensus OOF residual bindings.

```bash
python posterior_score_validation.py --R02
python posterior_score_validation.py --R01 --models ga sr
python posterior_score_validation.py --R02 --sr-run outputs/symbolic_regression/campaign_R02/<run_directory> --sr-candidate <candidate_id>
```

The command and `predict_posterior` function default to `xgb ga sr`, producing four prediction groups including their mean Score; `--models` selects a subset, without the three-model mean. `--sr-run` binds a saved SR run, and `--sr-candidate` explicitly uses an accepted fitted equation from that run's one-SE/two-SE layers. Its full CV audit, scaler and equations are verified; the manifest records this explicit choice and its sources while preserving the saved automatic model. Each round's experimental summary is computed once and shared by its prediction groups. Models belong to the selected campaign; model Rxx evaluates all available rounds through Rxx without pooling rank statistics. Without a round flag, the latest calibration round is selected.

Output: `outputs/posterior_predictions/model_Rxx/`

### Device normalization and plots

Normalizes each raw molecule file by dividing all device rows by the corresponding metric mean across all `Control` rows in that file. Files containing `_norm` in their names are never read as inputs. The device data contain R01 only.

Available raw CSVs are processed independently. Rows missing Group or any finite JV metric are skipped with CSV row numbers; files with missing columns, invalid labels, or no valid Control baseline are skipped with a reason. Summaries contain only successfully processed files. The round argument labels outputs; it does not filter raw files by experimental round.

`2Py2N.csv` contains final-device validation data and is automatically excluded during file discovery, before normalization or plotting. The source CSV is retained and the reason is printed; it contributes no rows to either summary. If no usable files remain, the command returns status 1 without writing outputs.

```bash
python device_normalization.py --R01
```

Outputs:

- `outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv`
- `outputs/device_analysis/campaign_Rxx/<Name>/{PCE,Voc,FF,Jsc}_boxplot.png`
- `outputs/device_analysis/campaign_Rxx/group_summary.csv`
- `outputs/device_analysis/campaign_Rxx/normalization_summary.csv`

### Device Score validation

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

`data_preparation.py`, `campaign.py`, `project_paths.py`, `structure_training.py`, `rank_statistics.py`, `sr_efficiency.py`, `sr_validation.py`, and `sr_reporting.py` are imported automatically by the entry-point scripts and are not intended to be run directly. `data_preparation.py` provides shared `Coordination_atom` parsing. `structure_training.py` supplies structure-model validation, DUPLEX splitting and common plots; training and reports use the CV configuration in `structure_selection.py`. Device and posterior reports use `rank_statistics.py` with their documented CSV column names. `sr_progress.py` is a standalone, read-only diagnostic entry point.

## Outputs and version control

All generated models, logs, tables, figures, and analysis summaries are written under `outputs/`.

See [output_files_summary.md](output_files_summary.md) for the complete artifact list.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the associated paper:

> Author names. *Title of the paper*. Journal, year (under review).
> Code: [https://github.com/Yang-happylab/PVK_Lewis_base_ML](https://github.com/Yang-happylab/PVK_Lewis_base_ML)
