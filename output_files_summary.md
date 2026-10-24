# Output Files Summary

## Structure feature selection

Both structure-training entry points save the following files under `Score/<model>/feature_selection/` in their run directories:

| File | Contents |
|---|---|
| `inner_search.csv` | Tuning candidates, inner R² mean/SD, repeat-mean SE and Mean_fit_seconds, identified by Stage and Phase |
| `outer_XX_selection_path.csv`, `final_selection_path.csv` | Path, parameters, feature count, per-repeat R², mean/SD, repeat-mean SE and joint one-SE decision; diagnostic only |
| `outer_fold_metrics.csv` | Nested outer-CV R², RMSE, MAE and selected features |
| `outer_oof_predictions.csv` | Held-out training predictions by unique Name, repeat and fold |
| `feature_stability.csv` | Outer-fold selection frequency and final feature selection |
| `subset_overlap.csv` | Pairwise Jaccard overlap of outer-fold feature subsets |
| `selection.json` | Selection method, CV settings, sampling rule, candidate budget and count, selected path, final features and parameters |

`training.log` records tuning stages, KRR deletion progress and time per outer fit. Model sidecars record feature-extraction settings. Each tuning stage samples up to 588 RF, 1288 XGB, 988 EN/BR or 788 KRR candidates.

Model sidecars preserve ordered feature names and the outer nested-CV evaluation used by GA. KRR permutation importance describes the training set only. Selection frequencies and fold SD are descriptive, not confidence intervals.

Chinese version: [output_files_summary.cn.md](output_files_summary.cn.md).

All paths are relative to the repository root. `<run_timestamp>` uses `YYYYMMDD_HHMMSS`; SR run directories also include a local or SLURM job identifier.

## device_normalization.py

Summaries include successfully processed files and complete device records only. Skipped rows/files and reasons are printed to the console; existing outputs for skipped files are not refreshed.

The exact molecule name `2Py2N` (CSV filename stem with surrounding whitespace stripped) is excluded during file discovery as final-device validation outside this analysis. Its source CSV is retained, no per-molecule outputs or summary rows are generated, and the reason is printed. If no usable files remain, no outputs are written and the command returns status 1.

Normalized companion table:

```text
outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv
```

Each file contains `Group`, `Voc`, `PCE`, `FF` and `Jsc`. Each value is the raw device measurement divided by the corresponding mean across all `Control` rows in the same raw molecule file, then multiplied by 100. Normalized files are not read as inputs.

Output root:

```text
outputs/device_analysis/campaign_Rxx/
```

| File | Description |
|---|---|
| `<Name>/PCE_boxplot.png` | Normalized PCE box plot with device observations and group means |
| `<Name>/Voc_boxplot.png` | Normalized open-circuit-voltage box plot |
| `<Name>/FF_boxplot.png` | Normalized fill-factor box plot |
| `<Name>/Jsc_boxplot.png` | Normalized short-circuit-current-density box plot |
| `group_summary.csv` | Device count and mean/sample SD for the four normalized metrics, by molecule and concentration |
| `normalization_summary.csv` | Raw and normalized filenames, device and Control counts, and the four raw Control means used for normalization |

Each plot is a standalone 300-dpi PNG on a fixed 10:8 canvas. All text uses Arial and all four axes spines are visible. `Control` is neutral gray, concentrations use a continuous color scale, and the horizontal reference at 100 marks the within-file Control mean.

## device_score_validation.py

The scatter plot shows sample size and rank statistics in its upper-left corner.

Output root:

```text
outputs/device_score_validation/campaign_Rxx/<analysis_id>/
```

`<analysis_id>` has the form `device-score-validation-v1-<12-hex-hash>` and is bound to the analysis implementation, device `group_summary.csv` and `normalization_summary.csv`, and the exact versioned `scores.csv`. The workflow does not renormalize raw data or read per-molecule `_norm.csv` files.

| File | Description |
|---|---|
| `best_concentration_summary.csv` | One row per molecule with Score/rank, the non-Control concentration with the highest normalized PCE mean, device count, mean, sample SD, SEM, two-sided 95% Student-t CI, recovered raw PCE mean, Control difference, PCE rank and rank difference |
| `rank_correlation_statistics.csv` | Spearman rho and Kendall tau-b, two-sided permutation p-values, and exact Score/PCE definitions |
| `score_pce_rank_comparison.png` | Two-column rank connection chart for descending Score and best normalized PCE, labeled with Score, PCE mean and selected concentration |
| `score_vs_best_normalized_pce.png` | Score versus best normalized PCE with sample-SD error bars, molecule labels, a Control=100 reference and rank statistics |
| `analysis_manifest.json` | Input paths and SHA-256 hashes, implementation/helper hashes, Score version, selection/tie rules, statistical method, interpretation limits and output hashes |

Control is excluded from concentration selection. Exact mean ties go to the lower numeric concentration; a best tested concentration below 100 remains below Control. For at most nine molecules, both rank-correlation p-values use all `n!` permutations; larger samples use 100,000 fixed-seed Monte Carlo permutations with plus-one correction.

## score_calibration.py

Each round is fitted independently, then library-standardized batch Scores are averaged equally. `fits.csv` includes `Round`; `batch_scores.csv` stores each batch's library predictions and Score; `batch_agreement.csv` reports descriptive pairwise Spearman/Kendall agreement, not independent experimental validation. `model.json` binds all batch models and their equal weights. Each batch's six plots are under `batches/Rxx/`; root PNGs show the latest included batch. The manifest hashes all batch PNGs.

Output root:

```text
outputs/score_calibration/campaign_Rxx/<score_version>/
```

`<score_version>` is derived from the effective cumulative calibration subset through `Rxx`, the physicochemical library, algorithm version and implementation hash. Appending a later campaign does not change an earlier campaign's subset hash.

| File | Description |
|---|---|
| `scores.csv` | `Name`, three DFT descriptors, predicted and standardized endpoints, generated `Score` and Score version |
| `fits.csv` | Six univariate linear fits, R-squared, observation counts and endpoint-specific normalized weights |
| `model.json` | Score equation, fit parameters, endpoint scaling and combined linear coefficients |
| `summary.json` | Included calibration rows, descriptor sources, library size and Score statistics |
| `normalized_calibration.csv` | Expanded measurements, same-round Ref-PbI2 means and relative endpoints (reference mean = 100); fits and predictions use this scale |
| `manifest.json` | Campaign, full source-file hash, effective cumulative-subset hash, implementation hash and artifact hashes |
| `plqy_binding_energy_correlation.png` | PLQY versus binding energy |
| `plqy_transferred_charges_correlation.png` | PLQY versus transferred charge |
| `plqy_vbur_correlation.png` | PLQY versus buried volume |
| `xrd_binding_energy_correlation.png` | XRD ratio versus binding energy |
| `xrd_transferred_charges_correlation.png` | XRD ratio versus transferred charge |
| `xrd_vbur_correlation.png` | XRD ratio versus buried volume |

## GA_EN_KRR_BR_Lewis_base.py

Output root:

```text
outputs/structure_models/campaign_Rxx/en_krr_br/<run_timestamp>/
```

Important files:

| Relative path inside the run | Description |
|---|---|
| `training.log` | UTF-8 training log |
| `final_report.txt` | Configuration and available model metrics |
| `data-structure.csv` | Ordered 32 numerical structure features and Score |
| `score_provenance.json` | Campaign, Score version, source/subset hashes and calibration manifest used by the run |
| `Score/feature_names.txt` | UTF-8 mapping of encoded feature names |
| `Score/Score_distribution.png` | Train/test target distribution |
| `Score/<model>/<model>_best_model.pkl` | Trained ElasticNet, KRR or BayesianRidge pipeline |
| `Score/<model>/<model>_feature_config.json` | Hash-bound model sidecar with feature names, Score and structure-data sources, DUPLEX/CV contracts and structured CV/test metrics |
| `Score/<model>/stacking_base_predictions.csv` | Named Train/Test predictions from the final base model, bound with nested OOF predictions to its sidecar |
| `Score/<model>/<model>_feature_importance_<method>.csv` | Feature-importance values |
| `Score/<model>/<model>_prediction_plot.png` | Calculated versus predicted values |
| `Score/<model>/<model>_residual_plot.png` | Residual plot |
| `Score/<model>/<model>_feature_importance.png` | Feature-importance plot |

## GA_RF_XGB_Lewis_base.py

Output root:

```text
outputs/structure_models/campaign_Rxx/rf_xgb/<run_timestamp>/
```

The directory layout is the same as above, with `<model>` equal to `RF` or `XGB`.

## GA_Lewis_base.py

Output root:

```text
outputs/genetic_algorithm/campaign_Rxx/GA_<run_timestamp>/
```

| File | Description |
|---|---|
| `score_provenance.json` | Score-version provenance inherited from loaded models, when available |
| `optimization_history.csv` | `Generation`, `Best`, `Average` and `Min` fitness values |
| `model_selection.csv` | Pass/fail decision for every candidate model, including campaign, CV mean/SD, CV mean minus SD, Test R2, artifact paths and provenance |
| `stacking/Stacking_best_model.pkl` | Scaled Ridge pipeline using ordered base-model predictions |
| `stacking/model_config.json` | Input model order, provenance, selected regularization, coefficients, intercept and evaluation scope |
| `stacking/metrics.csv` | Base-model and Stacking R2, RMSE, MAE and Spearman by evaluation scope |
| `stacking/predictions.csv` | Named Train/Test targets and base-model/Stacking predictions |
| `stacking/oof_training_predictions.csv` | Per-molecule nested OOF inputs and fitted Stacking predictions; training diagnostics |
| `stacking/ridge_search.csv` | Repeated-CV Ridge parameter-selection diagnostics, not full-stack validation |
| `stacking/final_report.txt` | Stacking parameters and evaluation summary |
| `stacking/Stacking_prediction_plot.png`, `stacking/Stacking_residual_plot.png` | Train/Test prediction and residual plots |
| `candidate_archive.csv` | All canonical-SMILES-unique candidates across generations, including the final population; columns: Rank, SMILES, Center, Arm1–Arm4 and PredictedScore; sorted by Score then SMILES |
| `optimal_samples.csv` | Top candidates selected from the cross-generation candidate archive, using the same columns and identity rules |
| `top10_structures.png` | Top 10 two-dimensional structures in CSV order with rank and Score; two columns, up to five rows |
| `excluded_structures.csv` | Valid designs excluded from GA prediction due to unsupported donor environments: SMILES, Formula and Reason |
| `GA_plot.png` | Best and average fitness by generation |
| `GA_best_score_10gen.png` | Best fitness in the first ten generations |
| `run_<run_timestamp>.log` | Execution log, including the model files loaded |

## XGB_Lewis_base.py

Output root:

```text
outputs/xgb_interpretability/campaign_Rxx/<run_timestamp>/
```

| Relative path inside the run | Description |
|---|---|
| `training.log` | UTF-8 training log |
| `Score/joint_search/cv_results.csv` | All 58,888 joint-search candidates across feature counts 4, 6, 8, 10 and all available features, with branch, parameters, feature count, per-split ensemble-prediction R², aggregate scores, ranks and timing |
| `Score/joint_search/search_progress.csv` | Candidate-order diagnostic for the joint search; not a wall-clock progress record |
| `Score/joint_search/cv_split_assignments.csv` | Per-fold `Name` membership for the seed-42 4-fold × 3-repeat training-CV search |
| `Score/joint_search/search_summary.json` | Joint parameter spaces, 60/40 focused/exploration allocation, feature-count allocation, candidate/CV seeds, five model seeds and equal averaging, candidate counts, CV settings and search provenance |
| `Score/joint_search/refinement_results.csv` | Seed-2026 4-fold × 5-repeat re-evaluation of leading search candidates and compatible saved finalists, with source run/candidate IDs, repeat means, SE and the one-SE decision |
| `Score/joint_search/refinement_split_assignments.csv` | Per-fold `Name` membership for the additional candidate review |
| `Score/joint_search/feature_paths.csv` | Mean-absolute ensemble SHAP feature ranks from fixed five-seed baseline XGBoost fits on each training fold and the full outer-training set; requested subsets take top-ranked features in original column order |
| `Score/joint_search/selection.json` | Diagnostic scope, one-SE thresholds, refinement seed/repeats, selected candidate IDs, ordered features, parameters and Selection CV R² for `full_features` and `selected_features`; records accepted/skipped saved-candidate sources with reasons and hashes, finalist count and selection rule |
| `final_report.txt` | Full/selected-feature metrics, selected features, labeled 4-fold × 5-repeat Selection CV R² and consensus OOF SHAP rank; written atomically after validation |
| `data_encoded.csv` | Runtime preprocessing result for this workflow |
| `score_provenance.json` | Generated Score version and calibration provenance |
| `EDA/full_correlation_matrix.png` | Predictor heatmap before correlation filtering; excludes BE, NTC, Vbur and Score |
| `Score/Score_distribution.png` | Train/test target distribution |
| `Score/full_features/best_model_full_features.pkl` | Full-feature five-member XGBoost VotingRegressor with equally averaged predictions |
| `Score/selected_features/best_model_selected_features.pkl` | Selected-feature five-member XGBoost VotingRegressor for posterior prediction |
| `Score/<model_stage>/model_contract_<model_stage>.json` | Schema-3 model SHA-256, campaign, target, stage, ordered prediction features, ensemble type, seeds and averaging rule; the selected-feature contract also binds ensemble OOF interval-calibration artifacts and hashes |
| `Score/<model_stage>/feature_importance_<model_stage>.csv` | Training-set mean absolute SHAP importance after averaging signed SHAP across the final ensemble members |
| `Score/<model_stage>/prediction_plot.png` | Calculated versus predicted values |
| `Score/<model_stage>/residual_plot.png` | Residual plot |
| `Score/<model_stage>/feature_importance.png` | Training-set mean absolute SHAP importance plot for the fitted final model |
| `Score/selected_features/top_feature_score_scatters/01_<feature>_vs_Score.png` … | One separate full-dataset scatter per selected feature, ordered by final-model training-set mean absolute SHAP |
| `Score/<model_stage>/shap_summary_<model_stage>.png` | SHAP summary plot, when analysis succeeds |
| `Score/<model_stage>/shap_importance_<model_stage>.png` | Mean absolute SHAP importance, when analysis succeeds |
| `Score/<model_stage>/shap_statistics_<model_stage>.csv` | SHAP statistics, when analysis succeeds |
| `Score/<model_stage>/shap_sample_frequency_<model_stage>.csv` | SHAP sample frequencies, when analysis succeeds |
| `Score/selected_features/oof_shap/oof_shap_summary_selected_features.png` | Publication-oriented beeswarm with one consensus OOF SHAP vector per unique outer-training sample |
| `Score/selected_features/oof_shap/oof_shap_importance_selected_features.png` | Mean absolute consensus OOF SHAP bar chart |
| `Score/selected_features/oof_shap/oof_shap_values_selected_features.csv` | Held-out ensemble SHAP values and predictions: member means within each fold, three rows per outer-training `Name` |
| `Score/selected_features/oof_shap/oof_shap_seed_values_selected_features.csv` | Held-out member SHAP, base values and predictions by seed and fold: fifteen rows per outer-training `Name` |
| `Score/selected_features/oof_shap/oof_shap_seed_importance_selected_features.csv` | Per-seed importance and ranks after within-`Name` repeat averaging, plus feature-value versus SHAP Spearman direction |
| `Score/selected_features/oof_shap/oof_shap_consensus_selected_features.csv` | Per-`Name` ensemble mean/SD across three OOF repeats; these unique-molecule consensus residuals calibrate empirical posterior Score intervals |
| `Score/selected_features/oof_shap/oof_shap_feature_stability_selected_features.csv` | Consensus importance, fold variability/ranks, seed median/minimum/maximum ranks and seed Top-3 frequency |
| `Score/selected_features/oof_shap/oof_shap_fold_importance_selected_features.csv` | Per-fold mean absolute SHAP values and ranks for the fixed final feature set |
| `Score/selected_features/oof_shap/oof_shap_fold_metrics_selected_features.csv` | Diagnostic per-fold validation metrics and SHAP additivity error; not a substitute for outer-test performance |
| `Score/selected_features/oof_shap/oof_shap_manifest_selected_features.json` | Campaign, Score version, fixed-feature/parameter contract, model seeds/averaging rule, CV settings, member and repeat aggregation, diagnostics, performance-role warning and artifact names |

Automatic posterior discovery selects the newest selected-feature XGBoost model with complete, valid Score and consensus OOF bindings. Model, Score and OOF artifact hashes must match; incomplete explicit models are rejected.

## posterior_score_validation.py

Output root:

```text
outputs/posterior_predictions/model_Rxx/<run_id>/
```

`Rxx` is the selected model campaign. Evaluation includes each experimental round through `Rxx`; `<run_id>` defaults to `YYYYMMDD_HHMMSS` and may be set explicitly.

| File | Description |
|---|---|
| `prediction.log` | Selected round, feature order, prediction range and artifact paths |
| `predictions.csv` | Predictions identified by `Model`, `Sample_Round` and `Name`, with experimental Score, within-round ranks and `Validation_Role` |
| `score_comparison.csv` | One row per experimental round and molecule with predicted Scores/ranks and the corresponding experimental Score/rank |
| `rank_correlation_statistics.csv` | One row per model and experimental round with Spearman rho, Kendall tau-b and two-sided permutation p-values; no pooled statistic |
| `<model>/round_Rxx/rank_correlation/measurement_summary.csv` | Round-specific replicate summaries, reference-normalized endpoints, experimental Score, ranks and interval fields |
| `<model>/round_Rxx/rank_correlation/rank_correlation_statistics.csv` | Round-specific rank statistics |
| `<model>/round_Rxx/rank_correlation/score_rank_comparison.png` | Arial rank-connection plot with correlation statistics at upper left |
| `<model>/round_Rxx/rank_correlation/predicted_vs_experimental_score.png` | Arial scatter plot of predicted versus experimental Score, with molecule labels and rank statistics |
| `<model>/round_Rxx/rank_correlation/analysis_manifest.json` | Round-specific Score/reference provenance, implementation/helper hashes, inference rules, interval status and artifact hashes |
| `run_manifest.json` | Input/model hashes, model round, evaluated rounds, per-round posterior REF hashes, validation roles and artifact hashes |

The statistical unit is one unique molecule within each experimental round; REF is excluded. Experimental Score uses complete molecule and same-batch posterior REF triplicates. REF means come from each round's `Name=REF` row in `posterior_validation.csv`; Score v3 constants are model-bound and are not fitted on posterior molecules. All-empty records and unnamed empty columns are ignored. For `n <= 9`, p-values use exhaustive two-sided permutations; larger samples use 100,000 fixed-seed Monte Carlo permutations with plus-one correction. PLQY is stored as percentage points; XRD ratio is dimensionless.

For structure-model inputs, `structure_features.describe_structure(smiles, halogen_proxy=True)` calculates the ordered 32 features. Terminal carbon-bound Br/I use the `n_Cl_donor` and `T_*Cl` slots; the original `SMILES` and DFT `atom_Cl` retain the actual elements. Default structure-training extraction accepts C/N/O/S/Cl and uses no halogen proxy.

`<model>` is `xgb`, `ga`, `sr` or `mean`. The command and function default evaluate the three models and their arithmetic mean predicted Score. Each round uses its own posterior REF and has separate figures/statistics; earlier-round predictions from a later model are marked retrospective. `--models` selects a subset without the three-model mean. Model campaigns and the scientific Score model/table hashes must agree. XGBoost uses its selected-feature model contract; GA uses the saved stacking order and hash-bound base-model features; SR evaluates the saved raw/scaled equations and checks the scaler against its equation audit without starting Julia. The input CSV is not rewritten by prediction.

Only XGBoost has an empirical 95% Score interval from hash-validated consensus OOF residuals for unique outer-training molecules, not the independent outer DUPLEX test set. GA/SR/mean interval fields are blank with an explicit unavailable status.

An explicit `--sr-run` and `--sr-candidate` selection binds an accepted saved display candidate, its comparison/full CV audit and raw/scaled equations in `run_manifest.json`. The saved automatic SR model remains intact. The three-model mean uses the explicitly chosen SR predictions together with the bound XGBoost and GA models.

## SR_Lewis_base.py

Output root:

```text
outputs/symbolic_regression/campaign_Rxx/<run_timestamp>_job<job_id>/
```

Inputs come directly from `data/data.csv`. Default formula inputs are `atom_N`, `gap`, `fai_min`, `μ`, `HOMO`, `TPSA`; `--features` configures the pool and `--selected-count` sets the exact descriptor count after simplification (default six). The scaler retains all ordered candidate inputs; `selected_feature_names` separately records the final formula's descriptors. The outer split uses the same ordered inputs as formula fitting, calculated from finite source values. `--max-excess-complexity` sets the shared ceiling (default five).

| File | Description |
|---|---|
| `data_encoded.csv` | Runtime table from data/data.csv and campaign Score, with stable source_row_index; candidate descriptors and Score must be finite numeric values |
| `score_provenance.json` | Score version and calibration provenance |
| `run_config.json` | Campaign, Score version, ordered candidates, split settings, required count, final descriptors, search/review CV policy, shared complexity ceiling, objective, selection/refinement/bootstrap settings, runtime, units and package versions |
| `sampled_configs.json` | The 16 unique fixed-seed search configurations with stable IDs and seeds |
| `resume_contract.json` | Hash-bound input, Score, candidate/count, split, configuration and implementation contracts for explicit resume |
| `configuration_results/<config_id>.json` | Atomic completed or failed configuration result; validated contiguous prefixes support resume |
| `cv_selection_checkpoint.json` | Configuration-selection decision and bounded worker-shutdown report |
| `split_assignments.csv` | Stable source_row_index, Name and outer Train/Test membership |
| `search_progress.jsonl` | Append-only chunk audit: candidate metrics, complexity, rejections, stopping decisions and timing, merged in deterministic fold order; refinement also records the accumulated loss/simplicity frontier, completed-chunk iteration count and last meaningful-progress iteration |
| `progress_status.json` | Atomic current configuration, active folds, worker progress, metrics and timing |
| `fold_workers/<config_id>/fold_<NN>/progress_status.json` | Live status for one isolated CV worker |
| `fold_workers/<config_id>/fold_<NN>/search_progress.jsonl` | Per-worker append-only chunk audit |
| `fold_workers/<config_id>/fold_<NN>/pysr_runs/<config_id>_fold_<NN>/` | PySR warm-start state for one CV fold |
| `fold_workers/<config_id>/fold_<NN>/<resume_id>/` | Clean attempt with a short unique identifier for an explicitly rerun interrupted configuration |
| `pysr_runs/refinement/` | Warm-start state for final refinement |
| `pysr_runs/refinement_<resume_id>/` | Clean explicit-resume refinement attempt |
| `symbolic_regression.log` | Inputs, objective hash, search/refinement progress, equations, scaling definitions, consistency audit, coefficient intervals and completion/errors |
| `results.txt` | Equations, units, complexity audit, conditional coefficient uncertainty, final metrics, labeled configuration-search CV and final-formula conditional CV |
| `results.json` | Equations/audits, candidate/final descriptors, required count, final metrics and search/refinement records; top-level cv_* describes configuration-search CV, nested final_candidate_cv contains conditional final-formula selection diagnostics, and ridge_quadratic_baseline contains the additive quadratic diagnostic |
| `selected_candidate.json` | Final PySR candidate_position, indexed-prediction contract, required/final descriptors, scaled/raw equations, complexity audit and final/configuration selection rules |
| `ridge_baseline.json` | Ridge using exactly the final SR selected_feature_names, identical outer/CV rows, fold-training scalers, conditional alpha-tuning metrics, equations and final metrics |
| `ridge_quadratic_baseline.json` | Additive quadratic Ridge diagnostic using the same final SR variables and outer split; raw linear and individual squared terms with no interactions, fold-training scaling of each basis column, separate alpha tuning, equations and final metrics |
| `cv_winner_inheritance.json` | Inherited winning-configuration CV formulas, source folds, scaler conversion, outer-training constant refits, accepted/rejected candidates and the shared complexity budget |
| `final_candidate_cv.json` | Joint fixed-structure 4-fold × 5-repeat review on seed-2026 splits: fold-only scaling/refits, exact-count/complexity checks, conditional metrics, official one-SE selection and two-SE alternatives sharing the best-mean/SE anchor; at most ten candidates per layer and shortlisted outer-training constant refits |
| `candidate_comparison.json`, `candidate_comparison.csv`, `candidate_comparison.txt` | Two candidate display layers with common IDs, full-precision scaled/raw formulas, CV R-squared/SD/repeat-mean SE/RMSE/MAE, fold/repeat diagnostics, training metrics, complexity/parameter counts and explicit refit failures; no external-test comparison |
| `candidate_results/summary.csv`, `candidate_results/manifest.json` | Post-selection metrics for each unique displayed candidate, layer membership, source/output hashes, shared plot ranges and export settings; test metrics do not alter automatic selection |
| `candidate_results/<candidate_id>/results.json`, `results.txt`, `predictions.csv` | Full-precision equations, CV diagnostics, train/test R-squared/RMSE/MAE, complexity and Name/source-row-linked actual values, predictions and Actual-minus-Predicted residuals; failed candidates retain status/reason instead of fabricated metrics |
| `candidate_results/<candidate_id>/prediction_plot_combined.png`, `residual_plot_combined.png`, `residual_distribution.png`, `prediction_and_residual.png` | Arial figures for each valid displayed equation: prediction versus actual with Ideal line/grid, residual scatter, residual histogram and two-panel comparison; shared axis limits across candidates |
| `equation_audit.json` | Scaled/raw and parameterized equations, units, scaler definitions, term-budget audit, numerical-equivalence checks and coefficient-bootstrap summary |
| `coefficient_uncertainty.csv` | Original constants, bootstrap means/medians, conditional standard errors and percentile confidence intervals |
| `coefficient_bootstrap_samples.csv` | Successful and failed fixed-structure outer-training bootstrap fits |
| `pysr_model.pkl` | Optional serialized PySR model; use selected_candidate.json for its prediction contract |
| `scaler.pkl` | Optional serialized fitted StandardScaler |
| `scaler_params.json` | Required atomic export of all ordered candidate features, source units, target definition and outer-training scaling parameters |
| `pysr_model_with_scaler.pkl` | Optional model/scaler bundle with selected position, features, target, equations and audit |
| `predictions.csv` | source_row_index, Name, split, SR/linear-Ridge predictions and residuals, plus additive quadratic Ridge predictions in Ridge_quadratic_Predicted |
| `prediction_plot_combined.png` | Final-model predictions; title distinguishes train/test R² from configuration-search CV R² |
| `residual_plot_combined.png` | Final-model residuals; title distinguishes train/test RMSE from configuration-search CV RMSE |
| `Score_distribution.png` | Train/test target distribution |

`scaler_params.json` is required; a write failure prevents successful completion. Pickle exports are optional and attempted independently. Check the log and required artifacts rather than directory presence alone. SR plots use Arial.

Per-candidate diagnostics are generated after saving the selected model. `python sr_reporting.py --run-directory outputs/symbolic_regression/campaign_Rxx/<run_directory>` exports them for a completed saved run without Julia, refitting, model-file changes or replacement of an existing diagnostic directory.

Run `python sr_progress.py` for a status snapshot or `python sr_progress.py --watch 10` for read-only monitoring. Explicit resume verifies the saved input, Score, ordered candidates, required count and split before replacing governed files.
