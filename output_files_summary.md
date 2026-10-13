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
| `analysis_manifest.json` | Input paths and SHA-256 hashes, Score version, selection/tie rules, statistical method, interpretation limits and output hashes |

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
| `optimal_samples.csv` | Top candidates selected from the historical archive, using the same columns and identity rules |
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
| `Score/joint_search/cv_results.csv` | All 58,888 joint-search candidates across feature counts 4, 6, 8, 10 and all available features, with branch, parameters, feature count, per-split and aggregate scores, ranks and timing |
| `Score/joint_search/search_progress.csv` | Candidate-order diagnostic for the joint search; not a wall-clock progress record |
| `Score/joint_search/cv_split_assignments.csv` | Per-fold `Name` membership for the seed-42 4-fold × 3-repeat training-CV search |
| `Score/joint_search/search_summary.json` | Joint parameter spaces, 60/40 focused/exploration allocation, feature-count allocation, seeds, candidate counts, CV settings and search provenance |
| `Score/joint_search/refinement_results.csv` | Seed-2026 4-fold × 5-repeat review of the top two candidates per feature count, with repeat means, SE and the final one-SE decision |
| `Score/joint_search/refinement_split_assignments.csv` | Per-fold `Name` membership for the additional candidate review |
| `Score/joint_search/feature_paths.csv` | Mean-absolute-SHAP feature ranks from the fixed baseline XGBoost on each training fold and the full outer-training set; requested subsets take top-ranked features in original column order |
| `Score/joint_search/selection.json` | Diagnostic scope, one-SE threshold, refinement seed/repeats, and candidate ID, ordered features, parameters and Selection CV R² for `full_features` and `selected_features` |
| `final_report.txt` | Full/selected-feature metrics, selected features, labeled 4-fold × 5-repeat Selection CV R² and consensus OOF SHAP rank; written atomically after validation |
| `data_encoded.csv` | Runtime preprocessing result for this workflow |
| `score_provenance.json` | Generated Score version and calibration provenance |
| `EDA/full_correlation_matrix.png` | Predictor heatmap before correlation filtering; excludes BE, NTC, Vbur and Score |
| `Score/Score_distribution.png` | Train/test target distribution |
| `Score/full_features/best_model_full_features.pkl` | Full-feature XGBoost model |
| `Score/selected_features/best_model_selected_features.pkl` | Selected-feature XGBoost model for posterior prediction |
| `Score/<model_stage>/model_contract_<model_stage>.json` | Model SHA-256, campaign, target, stage and exact ordered prediction features; the selected-feature contract also binds the OOF interval-calibration artifact and hash |
| `Score/<model_stage>/feature_importance_<model_stage>.csv` | Training-set mean absolute SHAP importance for the fitted final model |
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
| `Score/selected_features/oof_shap/oof_shap_values_selected_features.csv` | Held-out SHAP values and predictions: three rows per outer-training `Name` |
| `Score/selected_features/oof_shap/oof_shap_consensus_selected_features.csv` | Per-`Name` mean/SD across three OOF repeats; unique-sample consensus residuals calibrate empirical posterior Score intervals using training data |
| `Score/selected_features/oof_shap/oof_shap_feature_stability_selected_features.csv` | Consensus importance, fold variability, median/range ranks and top-rank frequency |
| `Score/selected_features/oof_shap/oof_shap_fold_importance_selected_features.csv` | Per-fold mean absolute SHAP values and ranks for the fixed final feature set |
| `Score/selected_features/oof_shap/oof_shap_fold_metrics_selected_features.csv` | Diagnostic per-fold validation metrics and SHAP additivity error; not a substitute for outer-test performance |
| `Score/selected_features/oof_shap/oof_shap_manifest_selected_features.json` | Campaign, Score version, fixed-feature/parameter contract, CV settings, aggregation, diagnostics, performance-role warning and artifact names |

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
| `<model>/round_Rxx/rank_correlation/analysis_manifest.json` | Round-specific Score/reference provenance, inference rules, interval status and artifact hashes |
| `run_manifest.json` | Input/model hashes, model round, evaluated rounds, per-round posterior REF hashes, validation roles and artifact hashes |

The statistical unit is one unique molecule within each experimental round; REF is excluded. Experimental Score uses complete molecule and same-batch posterior REF triplicates. REF means come from each round's `Name=REF` row in `posterior_validation.csv`; Score v3 constants are model-bound and are not fitted on posterior molecules. All-empty records and unnamed empty columns are ignored. For `n <= 9`, p-values use exhaustive two-sided permutations; larger samples use 100,000 fixed-seed Monte Carlo permutations with plus-one correction. PLQY is stored as percentage points; XRD ratio is dimensionless.

For structure-model inputs, `structure_features.describe_structure(smiles, halogen_proxy=True)` calculates the ordered 32 features. Terminal carbon-bound Br/I use the `n_Cl_donor` and `T_*Cl` slots; the original `SMILES` and DFT `atom_Cl` retain the actual elements. Default structure-training extraction accepts C/N/O/S/Cl and uses no halogen proxy.

`<model>` is `xgb`, `ga`, `sr` or `mean`. The command and function default evaluate the three models and their arithmetic mean predicted Score. Each round uses its own posterior REF and has separate figures/statistics; earlier-round predictions from a later model are marked retrospective. `--models` selects a subset without the three-model mean. Model campaigns and the scientific Score model/table hashes must agree. XGBoost uses its selected-feature model contract; GA uses the saved stacking order and hash-bound base-model features; SR evaluates the saved raw/scaled equations and checks the scaler against its equation audit without starting Julia. The input CSV is not rewritten by prediction.

Only XGBoost has an empirical 95% Score interval from hash-validated consensus OOF residuals for unique outer-training molecules, not the independent outer DUPLEX test set. GA/SR/mean interval fields are blank with an explicit unavailable status.

## SR_Lewis_base.py

Output root:

```text
outputs/symbolic_regression/campaign_Rxx/<run_timestamp>_job<job_id>/
```

| File | Description |
|---|---|
| `data_encoded.csv` | Runtime symbolic-regression table derived from `data/data.csv` and campaign Score, with stable zero-based `source_row_index` |
| `score_provenance.json` | Generated Score version and calibration provenance |
| `run_config.json` | Campaign/profile, parent and fold-worker process/thread layout, configured/actual Julia and BLAS threads, bounded worker shutdown, explicit-resume policy, power-plan policy, fold-only scaling, zero-parsimony term budget, explicit Pareto selection, CV/refinement/bootstrap limits, units, package versions and Score version |
| `sampled_configs.json` | 40 fixed-seed configurations sampled without replacement, with stable IDs and seeds |
| `resume_contract.json` | Hash-bound data, Score, split, sampled configurations, scientific implementation/settings, runtime layout and package versions for explicit resume |
| `configuration_results/<config_id>.json` | Atomic result for a completed, pruned or failed configuration; only a validated contiguous prefix is reusable |
| `cv_selection_checkpoint.json` | Validation-based configuration decision saved before worker shutdown, plus the bounded-shutdown report |
| `split_assignments.csv` | Stable `source_row_index`, `Name` and outer Train/Test split for each modeled row |
| `search_progress.jsonl` | Parent chunk audit merged in deterministic fold order after each worker finishes: seeds, iterations, wall/process CPU time, raw validation fields, Pareto positions, rejection reasons, complexity, terminal-only bootstrap, validation/training frontiers and stop state |
| `progress_status.json` | Atomic current-state snapshot during worker execution, configuration changes and shutdown: configuration out of 40, active folds, worker chunk/iteration/R²/decision, measured CPU use, Julia/BLAS threads and terminal counts |
| `fold_workers/<config_id>/fold_<NN>/progress_status.json` | Atomic live status from an isolated fold worker, read by the parent progress aggregator |
| `fold_workers/<config_id>/fold_<NN>/search_progress.jsonl` | Per-worker audit retained alongside the deterministically merged parent audit |
| `fold_workers/<config_id>/fold_<NN>/pysr_runs/<config_id>_fold_<NN>/` | PySR warm-start state for an isolated CV fold |
| `fold_workers/<config_id>/fold_<NN>/resume_<timestamp>/` | Clean worker attempt for an explicitly rerun interrupted non-terminal configuration |
| `pysr_runs/refinement/` | Bounded warm-start state for final refinement |
| `pysr_runs/refinement_resume_<timestamp>/` | Clean explicit-resume refinement attempt; interrupted Julia state is not reused |
| `symbolic_regression.log` | Full log with a concise line after each chunk (config/fold/chunk/iterations/R²/CPU/decision), runtime mode, scaled/raw equations, source units, scaler definitions, consistency audit and coefficient intervals |
| `results.txt` | Equations, term-budget audit, unit/scaling interpretation, conditional coefficient uncertainty, raw metrics, selection rule, terminal counts and search report |
| `results.json` | Machine-readable equations/audits, raw metrics, configuration/fold records, completed/pruned/failed states, selection rules, refinement, runtime metadata and artifact map |
| `selected_candidate.json` | Stable PySR `candidate_position`, indexed-prediction contract, scaled/raw equations, direct-quadratic/excess-complexity audit and configuration-selection rule |
| `ridge_baseline.json` | Leakage-safe Ridge comparator using the same outer split and CV fold indices, without test access |
| `equation_audit.json` | Scaled/raw and parameterized equations, feature/target units, scaler definitions, term-budget audit, numerical-equivalence checks, bootstrap settings/limits and coefficient summary |
| `coefficient_uncertainty.csv` | Original constants, bootstrap means/medians, conditional standard errors and percentile confidence intervals |
| `coefficient_bootstrap_samples.csv` | Successful and failed fixed-structure outer-training bootstrap fits with recorded seeds |
| `pysr_model.pkl` | PySR model object, when serialization succeeds; use `selected_candidate.json` for the prediction contract |
| `scaler.pkl` | Fitted `StandardScaler`, when serialization succeeds |
| `scaler_params.json` | Feature order, source units, target definition and outer-training scaling parameters |
| `pysr_model_with_scaler.pkl` | Model, scaler, selected position, prediction contract, features, target, equations and audit bundle, when serialization succeeds |
| `predictions.csv` | `source_row_index`, `Name`, split, symbolic-regression/Ridge predictions and residuals |
| `prediction_plot_combined.png` | Calculated versus predicted values |
| `residual_plot_combined.png` | Residual plot |
| `Score_distribution.png` | Train/test target distribution |

Some interpretability or serialization artifacts are best-effort and may be skipped with a warning; check the run log rather than relying on directory presence alone.

The publication grid uses fixed-seed sampling without replacement, and direct execution attempts all 40 configurations. There is no global R-squared target, global patience/window termination or Top-8 promotion. Each fold scaler is fitted on its training rows; the independent outer DUPLEX test set stays locked until the final indexed candidate is fixed. A partial configuration is pruned only when even an R-squared of 1 for every unobserved fold cannot match a completed incumbent. Raw R-squared/RMSE/MAE remain separate from selection scores.

Default operators are the domain-safe, symmetric `+`, `-`, `*` and `square`; division, arbitrary powers, `exp`, `log` and `sqrt` are not searched. PySR parsimony is zero. Each feature may have one linear and one direct square term without excess complexity; interactions, higher powers, repeated nonlinear terms and nested/composite transforms are penalized during one-standard-error selection. Raw node complexity is an audit field and hard tree-size limit, not the scientific selection penalty.

Each CV fit uses 100-iteration warm-start chunks, with limits of 600 iterations and 500 seconds. Intermediate chunks check finite predictions, six-feature eligibility, validation metrics, and training-loss/validation-R-squared frontiers. Paired bootstrap runs only at the terminal chunk. Refinement uses 250-iteration chunks, a 3,000-iteration limit and a default 7,200-second wall-time limit. On a 32-logical-processor workstation, CV defaults to four isolated processes with 8 Julia threads each and one BLAS helper thread per process; configurations remain serial and pruning is checked after each completed worker batch. Refinement may use the full resolved Julia-thread budget. The resolver prevents thread oversubscription and records process/thread settings and timings; the workflow does not change the operating-system power plan. `SR_DETERMINISTIC=1` selects strict one-process/one-thread mode. Run `python sr_progress.py` for a one-time view or `python sr_progress.py --watch 10` for read-only live monitoring; existing runs can also be read from `search_progress.jsonl`.
