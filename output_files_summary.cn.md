# 输出文件摘要

## 结构变量筛选

两个结构训练入口均在各自运行目录的 `Score/<model>/feature_selection/` 保存以下文件：

| 文件 | 内容 |
|---|---|
| `inner_search.csv` | 调参候选、内层 R² 均值/标准差、重复均值标准误差及 Mean_fit_seconds，以 Stage、Phase 区分 |
| `outer_XX_selection_path.csv`、`final_selection_path.csv` | 路径编号、参数、变量数、各次重复 R²、均值/标准差、重复均值标准误差及联合 one-SE 决定，仅作诊断 |
| `outer_fold_metrics.csv` | 外层嵌套 CV 的 R²、RMSE、MAE 及保留变量 |
| `outer_oof_predictions.csv` | 按唯一 Name、重复和折记录的训练集折外预测 |
| `feature_stability.csv` | 外层折入选频率与最终变量选择 |
| `subset_overlap.csv` | 各外层折变量集合的两两 Jaccard 重合度 |
| `selection.json` | 筛选方法、CV 设置、抽样规则、预算与实际候选数、选中路径、最终变量和参数 |

`training.log` 记录调参阶段、KRR 删除进度及各外层拟合耗时。结构侧车记录特征提取设置。每阶段参数搜索 RF 最多 588 组，XGB 最多 1288 组，EN/BR 最多 988 组，KRR 最多 788 组。

模型侧车保留有序变量名，并记录 GA 所需的外层嵌套 CV 评价。KRR 置换重要性仅描述训练集，不使用外部测试集。入选频率与折间标准差是描述性结果，不是置信区间。

英文版本：[output_files_summary.md](output_files_summary.md)。

以下路径均相对于仓库根目录。`<run_timestamp>` 使用 `YYYYMMDD_HHMMSS` 格式；SR 运行目录还会追加本地或 SLURM 作业标识。

## device_normalization.py

汇总仅包含成功处理的文件及完整器件记录。跳过的行、文件及原因打印到控制台；被跳过文件的既有产物不会刷新。

归一化伴随表：

```text
outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv
```

每个文件包含 `Group`、`Voc`、`PCE`、`FF` 和 `Jsc`。每个数值均为同一原始分子文件中对应指标的原始器件测量值除以全部 `Control` 行的对应均值，再乘以 100。已有的归一化文件不会被读取为输入。

分析输出根目录：

```text
outputs/device_analysis/campaign_Rxx/
```

| 文件 | 说明 |
|---|---|
| `<Name>/PCE_boxplot.png` | 包含每个器件观测值和组均值的归一化 PCE 箱线图 |
| `<Name>/Voc_boxplot.png` | 归一化开路电压箱线图 |
| `<Name>/FF_boxplot.png` | 归一化填充因子箱线图 |
| `<Name>/Jsc_boxplot.png` | 归一化短路电流密度箱线图 |
| `group_summary.csv` | 每个分子、每个浓度的器件数，以及四项归一化指标的均值和样本 SD |
| `normalization_summary.csv` | 原始文件和归一化文件名、器件数、Control 数，以及用于归一化的四个原始 Control 均值 |

四张图均为独立的 300 dpi PNG，画布固定为 10:8。所有文字使用 Arial，四周坐标轴框线均可见；`Control` 使用中性灰色，浓度使用连续色阶；100 的水平参考线表示文件内 Control 均值。

## device_score_validation.py

散点图在图内左上角显示样本数和秩统计。

输出根目录：

```text
outputs/device_score_validation/campaign_Rxx/<analysis_id>/
```

`<analysis_id>` 的格式为 `device-score-validation-v1-<12位十六进制哈希>`，并绑定分析实现、已有器件 `group_summary.csv` 和 `normalization_summary.csv`，以及精确版本的 `scores.csv`。该流程不会重新归一化原始数据，也不会读取分子级 `_norm.csv` 文件。

| 文件 | 说明 |
|---|---|
| `best_concentration_summary.csv` | 每个分子一行，包含 Score/排名、非 Control 浓度中归一化 PCE 均值最高的浓度、器件数、均值、样本 SD、SEM、双侧 95% Student-t 置信区间、恢复的原始 PCE 均值、与 Control 的差值、PCE 排名和排名差 |
| `rank_correlation_statistics.csv` | Spearman rho 和 Kendall tau-b、双侧置换检验 p 值，以及精确的 Score/PCE 定义 |
| `score_pce_rank_comparison.png` | 比较降序 Score 排名和最佳归一化 PCE 排名的双列连接图，并标注 Score、PCE 均值和所选浓度 |
| `score_vs_best_normalized_pce.png` | Score 与最佳归一化 PCE 的散点图，包含样本 SD 误差棒、分子标签、Control=100 参考线和排名统计量 |
| `analysis_manifest.json` | 输入路径及 SHA-256 哈希、Score 版本、确定性的选择/并列规则、统计方法、解释限制和输出哈希 |

浓度选择时排除 Control。均值完全相同时选择数值更低的浓度；最佳测试浓度低于 100 时仍保留其低于 Control 的结果，不替换为 Control。分子数不超过九个时，两个秩相关 p 值使用全部 `n!` 个排列；更大样本使用固定随机种子的 100,000 次 Monte Carlo 排列并采用加一校正。

## score_calibration.py

每一轮先分别拟合，再对经过分子库标准化的批次 Score 做等权平均。`fits.csv` 包含 `Round`；`batch_scores.csv` 保存每个批次的分子库预测值和 Score；`batch_agreement.csv` 保存描述性的批次间 Spearman/Kendall 一致性，不代表独立的实验验证。`model.json` 绑定所有批次模型及其等权重。每个批次的六张图位于 `batches/Rxx/`；根目录 PNG 只展示最后一个纳入的批次。manifest 会哈希所有批次 PNG。


输出根目录：

```text
outputs/score_calibration/campaign_Rxx/<score_version>/
```

`<score_version>` 由截至 `Rxx` 的有效累计校正子集、理化性质库、算法版本和实现哈希共同确定。后续追加的 campaign 不会改变更早 campaign 的子集哈希。

| 文件 | 说明 |
|---|---|
| `scores.csv` | `Name`、三个 DFT 描述符、预测端点、标准化端点、生成的 `Score` 和 Score 版本 |
| `fits.csv` | 六条单变量线性拟合、R-squared、观测数和端点特异的归一化权重 |
| `model.json` | Score 方程、拟合参数、端点缩放和合并线性系数 |
| `summary.json` | 纳入的校正行、描述符来源、分子库大小和 Score 统计量 |
| `normalized_calibration.csv` | 展开的测量值、同轮 Ref-PbI2 均值和相对端点（参照均值=100）；拟合和预测端点使用此相对尺度 |
| `manifest.json` | campaign、完整源文件哈希、有效累计子集哈希、实现哈希和产物哈希 |
| `plqy_binding_energy_correlation.png` | PLQY 与结合能的关系图 |
| `plqy_transferred_charges_correlation.png` | PLQY 与电荷转移量的关系图 |
| `plqy_vbur_correlation.png` | PLQY 与埋藏体积的关系图 |
| `xrd_binding_energy_correlation.png` | XRD 比值与结合能的关系图 |
| `xrd_transferred_charges_correlation.png` | XRD 比值与电荷转移量的关系图 |
| `xrd_vbur_correlation.png` | XRD 比值与埋藏体积的关系图 |

## GA_EN_KRR_BR_Lewis_base.py

输出根目录：

```text
outputs/structure_models/campaign_Rxx/en_krr_br/<run_timestamp>/
```

重要文件：

| 运行目录内的相对路径 | 说明 |
|---|---|
| `training.log` | UTF-8 训练日志 |
| `final_report.txt` | 配置和可用模型指标 |
| `data-structure.csv` | 有序的 32 个数值结构特征及 Score |
| `score_provenance.json` | 本次运行使用的 campaign、Score 版本、源数据/子集哈希和校正 manifest |
| `Score/feature_names.txt` | UTF-8 编码的完整特征名映射 |
| `Score/Score_distribution.png` | 训练集/测试集目标分布 |
| `Score/<model>/<model>_best_model.pkl` | 训练得到的 ElasticNet、KRR 或 BayesianRidge pipeline |
| `Score/<model>/<model>_feature_config.json` | 与哈希绑定的模型侧车，包含特征名、Score 和结构数据来源、DUPLEX/CV 契约以及结构化 CV/测试指标 |
| `Score/<model>/stacking_base_predictions.csv` | 最终基模按 Name 保存的 Train/Test 预测，与嵌套 OOF 预测一起绑定到模型侧车 |
| `Score/<model>/<model>_feature_importance_<method>.csv` | 特征重要性数值 |
| `Score/<model>/<model>_prediction_plot.png` | 计算值与预测值图 |
| `Score/<model>/<model>_residual_plot.png` | 残差图 |
| `Score/<model>/<model>_feature_importance.png` | 特征重要性图 |

## GA_RF_XGB_Lewis_base.py

输出根目录：

```text
outputs/structure_models/campaign_Rxx/rf_xgb/<run_timestamp>/
```

目录结构与上方相同，其中 `<model>` 为 `RF` 或 `XGB`。

## GA_Lewis_base.py

输出根目录：

```text
outputs/genetic_algorithm/campaign_Rxx/GA_<run_timestamp>/
```

| 文件 | 说明 |
|---|---|
| `score_provenance.json` | 从已加载模型继承的 Score 版本来源信息（如可用） |
| `optimization_history.csv` | `Generation`、`Best`、`Average` 和 `Min` 适应度值 |
| `model_selection.csv` | 每个候选模型的通过/不通过决定，包括 campaign、CV 均值、CV 标准差、均值减标准差、Test R2、产物路径和来源信息 |
| `stacking/Stacking_best_model.pkl` | 以有序基模预测为输入的标准化与 Ridge pipeline |
| `stacking/model_config.json` | 输入模型顺序、来源、选中正则化参数、系数、截距与评价范围 |
| `stacking/metrics.csv` | 按评价范围汇总的基模与 Stacking R2、RMSE、MAE、Spearman 指标 |
| `stacking/predictions.csv` | 按 Name 保存的 Train/Test 目标与基模、Stacking 预测 |
| `stacking/oof_training_predictions.csv` | 每分子的嵌套 OOF 输入及 Stacking 拟合预测，仅作训练诊断 |
| `stacking/ridge_search.csv` | Ridge 重复 CV 参数选择诊断，不代表完整集成验证 |
| `stacking/final_report.txt` | Stacking 参数与评价汇总 |
| `stacking/Stacking_prediction_plot.png`、`stacking/Stacking_residual_plot.png` | Train/Test 预测与残差图 |
| `candidate_archive.csv` | 跨代遇到的全部规范 SMILES 唯一候选，包括最终种群；列为 Rank、SMILES、Center、Arm1–Arm4、PredictedScore；按 Score 降序、SMILES 作为并列排序键 |
| `optimal_samples.csv` | 从历史档案中选取的 Top 候选，使用相同的列和身份规则 |
| `top10_structures.png` | 按 CSV 顺序绘制 Top 10 二维分子结构，标注排名和 Score；两列，最多五行 |
| `excluded_structures.csv` | 因供体环境不受支持而排除出 GA 预测的合法结构：SMILES、Formula、Reason |
| `GA_plot.png` | 各代最佳和平均适应度 |
| `GA_best_score_10gen.png` | 前十代中的最佳适应度 |
| `run_<run_timestamp>.log` | 执行日志，包括实际加载的模型文件 |

## XGB_Lewis_base.py

输出根目录：

```text
outputs/xgb_interpretability/campaign_Rxx/<run_timestamp>/
```

| 运行目录内的相对路径 | 说明 |
|---|---|
| `training.log` | UTF-8 训练日志 |
| `Score/joint_search/cv_results.csv` | 联合搜索的全部 58,888 组候选，覆盖 4、6、8、10 和全部可用特征，记录分支、参数、特征数、各折分数、汇总分数、排名及计时 |
| `Score/joint_search/search_progress.csv` | 按候选评价顺序记录联合搜索状态的诊断表；不表示墙钟进度 |
| `Score/joint_search/cv_split_assignments.csv` | 随机种子 42 的 4 折 × 3 次重复训练 CV 中每一折按 `Name` 记录的成员关系 |
| `Score/joint_search/search_summary.json` | 联合参数空间、focused/exploration 的 60/40 分配、特征数分配、随机种子、实际候选数、CV 设置和搜索来源信息 |
| `Score/joint_search/refinement_results.csv` | 每种特征数前两名候选的随机种子 2026、4 折 × 5 次重复复核结果，包含重复均值、标准误差和最终 one-SE 决定 |
| `Score/joint_search/refinement_split_assignments.csv` | 额外候选复核中每一折按 `Name` 记录的成员关系 |
| `Score/joint_search/feature_paths.csv` | 固定基线 XGBoost 在各训练折上得到的平均绝对 SHAP 特征排名，以及在完整外层训练集上得到的最终排名；各目标子集取排名前列特征并恢复原始列顺序 |
| `Score/joint_search/selection.json` | 诊断范围、one-SE 阈值、复核随机种子/重复数，以及 `full_features` 和 `selected_features` 选择的候选 ID、有序特征、参数与 Selection CV R² |
| `final_report.txt` | 完整特征和选中特征指标、选中特征、明确标注的 4 折 × 5 次重复 Selection CV R² 以及共识 OOF SHAP 排名；所有报告内容有效后原子写入 |
| `data_encoded.csv` | 本流程的运行时预处理结果 |
| `score_provenance.json` | 生成的 Score 版本和校正来源信息 |
| `EDA/full_correlation_matrix.png` | 相关性筛选前的自变量热图；排除 BE、NTC、Vbur 和 Score |
| `Score/Score_distribution.png` | 训练集/测试集目标分布 |
| `Score/full_features/best_model_full_features.pkl` | 全特征 XGBoost 模型 |
| `Score/selected_features/best_model_selected_features.pkl` | 用于后验预测的选中特征 XGBoost 模型 |
| `Score/<model_stage>/model_contract_<model_stage>.json` | 模型 SHA-256、campaign、目标、模型阶段和下游预测使用的精确有序特征名；选中特征契约还绑定 OOF 区间校准产物及其哈希 |
| `Score/<model_stage>/feature_importance_<model_stage>.csv` | 最终拟合模型在训练集上的平均绝对 SHAP 重要性数值 |
| `Score/<model_stage>/prediction_plot.png` | 计算值与预测值图 |
| `Score/<model_stage>/residual_plot.png` | 残差图 |
| `Score/<model_stage>/feature_importance.png` | 最终拟合模型在训练集上的平均绝对 SHAP 重要性图 |
| `Score/selected_features/top_feature_score_scatters/01_<feature>_vs_Score.png` … | 每个选中特征各生成一张独立的全数据集散点图，按最终模型的训练集平均绝对 SHAP 排序 |
| `Score/<model_stage>/shap_summary_<model_stage>.png` | SHAP 摘要图，分析成功时生成 |
| `Score/<model_stage>/shap_importance_<model_stage>.png` | 平均绝对 SHAP 重要性，分析成功时生成 |
| `Score/<model_stage>/shap_statistics_<model_stage>.csv` | SHAP 统计量，分析成功时生成 |
| `Score/<model_stage>/shap_sample_frequency_<model_stage>.csv` | SHAP 样本频次，分析成功时生成 |
| `Score/selected_features/oof_shap/oof_shap_summary_selected_features.png` | 基于每个外层训练样本一个共识 OOF SHAP 向量的论文导向 beeswarm 图 |
| `Score/selected_features/oof_shap/oof_shap_importance_selected_features.png` | 共识 OOF SHAP 平均绝对值条形图 |
| `Score/selected_features/oof_shap/oof_shap_values_selected_features.csv` | 所有折留出 SHAP 值和预测值：每个外层训练 `Name` 三行 |
| `Score/selected_features/oof_shap/oof_shap_consensus_selected_features.csv` | 每个 `Name` 在三次 OOF 重复中的均值和标准差；其唯一分子共识残差是后验 Score 经验区间的训练集内来源 |
| `Score/selected_features/oof_shap/oof_shap_feature_stability_selected_features.csv` | 共识重要性、折间变异、中位/范围排名和高排名频次 |
| `Score/selected_features/oof_shap/oof_shap_fold_importance_selected_features.csv` | 固定最终特征集的逐折平均绝对 SHAP 值和排名 |
| `Score/selected_features/oof_shap/oof_shap_fold_metrics_selected_features.csv` | 逐折验证指标和 SHAP 加和误差的诊断记录；不替代外层测试集性能 |
| `Score/selected_features/oof_shap/oof_shap_manifest_selected_features.json` | campaign、Score 版本、固定特征/固定参数契约、CV 设置、聚合定义、诊断指标、性能角色警告和产物名 |


## posterior_score_validation.py

输出根目录：

```text
outputs/posterior_predictions/model_Rxx/<run_id>/
```

`Rxx` 指定模型轮次；评估覆盖截至该轮的各实验轮次。`<run_id>` 默认为 `YYYYMMDD_HHMMSS`，也可以显式设置。

| 文件 | 说明 |
|---|---|
| `prediction.log` | 所选轮次、特征顺序、预测范围和产物路径 |
| `predictions.csv` | 按 `Model`、`Sample_Round` 和 `Name` 标识的预测、实测 Score、轮内排名及 `Validation_Role` |
| `score_comparison.csv` | 每实验轮次每分子一行，各模型预测 Score/排名及对应实测 Score/排名 |
| `rank_correlation_statistics.csv` | 每模型每实验轮次一行的 Spearman rho、Kendall tau-b 和双侧置换 p 值，不合并轮次 |
| `<model>/round_Rxx/rank_correlation/measurement_summary.csv` | 本轮重复测量摘要、相对参照端点、实测 Score、排名及区间字段 |
| `<model>/round_Rxx/rank_correlation/rank_correlation_statistics.csv` | 本轮秩统计 |
| `<model>/round_Rxx/rank_correlation/score_rank_comparison.png` | Arial 排名连接图，统计位于左上角 |
| `<model>/round_Rxx/rank_correlation/predicted_vs_experimental_score.png` | 预测 Score 与实测 Score 的 Arial 散点图，标注分子及秩统计 |
| `<model>/round_Rxx/rank_correlation/analysis_manifest.json` | 本轮 Score/参照来源、推断规则、区间状态及产物哈希 |
| `run_manifest.json` | 输入/模型哈希、模型轮次、被评估轮次、各轮后验 REF 哈希、验证角色及产物哈希 |

统计单位是各实验轮次内的唯一分子，不包含 REF。实测 Score 使用分子与同期后验 REF 两项指标的完整三重复。REF 均值分别取自 `posterior_validation.csv` 各轮次的 `Name=REF` 行；Score v3 常数绑定模型，不在后验分子上重新拟合。全空记录及全空无名列会被忽略。当 `n <= 9` 时，p 值使用穷举双侧置换检验；更大样本使用固定种子的 100,000 次 Monte Carlo 置换并采用加一校正。PLQY 以百分点保存，XRD 比值为无量纲量。

结构模型输入使用 `structure_features.describe_structure(smiles, halogen_proxy=True)` 计算有序 32 维特征。与碳成单键的末端 Br/I 使用 `n_Cl_donor` 和 `T_*Cl` 槽位；原始 `SMILES` 与 DFT `atom_Cl` 按实际元素记录。结构训练默认提取接受 C/N/O/S/Cl，不启用卤素代理。

`<model>` 为 `xgb`、`ga`、`sr` 或 `mean`。命令行和函数默认评估三个模型及其预测 Score 的算术平均。各实验轮次使用各自后验 REF，分别输出图和统计；后期模型预测早期分子标记为回顾性。`--models` 可选择子集，此时不生成三模型平均。模型轮次以及科学 Score 模型/表格哈希必须一致。XGBoost 使用选中特征模型契约；GA 使用保存的堆叠顺序及哈希绑定的基模特征；SR 使用保存的原始/标准化公式，并按公式审计核对 scaler，不启动 Julia。预测不重写输入 CSV。

只有 XGBoost 使用哈希校验的外层训练集唯一分子共识 OOF 残差生成经验 95% Score 区间，不使用独立的外层 DUPLEX 测试集。GA/SR/平均 Score 的区间字段为空并明确标为不可用。

## SR_Lewis_base.py

输出根目录：

```text
outputs/symbolic_regression/campaign_Rxx/<run_timestamp>_job<job_id>/
```

| 文件 | 说明 |
|---|---|
| `data_encoded.csv` | 从 `data/data.csv` 和生成的 campaign Score 运行时派生的符号回归表，包含稳定的从零开始的 `source_row_index` |
| `score_provenance.json` | 生成的 Score 版本和校正来源信息 |
| `run_config.json` | campaign、发表配置、父进程和 fold-worker 进程/线程布局、配置及实际 Julia 和 BLAS 线程、受限 worker 关闭设置/结果、显式恢复策略、不变的电源方案策略、仅折内缩放、零 parsimony 项预算、显式 Pareto 选择、CV/精修/bootstrap 硬限制、单位、包版本和 Score 版本 |
| `sampled_configs.json` | 固定随机种子、无放回抽样的 40 个配置，包含稳定 ID 和种子 |
| `resume_contract.json` | 对数据、Score、划分、抽样配置、科学实现文件及设置、运行布局和包版本进行严格哈希绑定的显式恢复身份契约 |
| `configuration_results/<config_id>.json` | 一个已完成、保守剪枝或失败配置的原子完整结果；只有经过验证的连续前缀可复用 |
| `cv_selection_checkpoint.json` | fold-worker 关闭前保存的显式验证配置决策，以及最终的受限关闭报告 |
| `split_assignments.csv` | 每个建模行稳定的 `source_row_index`、`Name` 和外层 `Train`/`Test` 划分 |
| `search_progress.jsonl` | 每个 worker 完成后按确定的折顺序合并的父进程追加式分块审计：种子、请求迭代数、墙钟/进程 CPU 计时、原始验证字段、Pareto 位置审计、拒绝原因、结构/超额复杂度、仅终止阶段的 one-standard-error bootstrap、验证和训练前沿字段以及停止状态 |
| `progress_status.json` | worker 运行期间、配置转换时及受限关闭期间更新的小型原子当前状态快照：40 个配置中的精确位置、活动折、每个 worker 的分块/迭代/R-squared/决定、实测 CPU 使用、Julia/BLAS 线程、终止计数和决定 |
| `fold_workers/<config_id>/fold_<NN>/progress_status.json` | 一个隔离 fold worker 所拥有、由父进程进度聚合器读取的原子实时状态 |
| `fold_workers/<config_id>/fold_<NN>/search_progress.jsonl` | 除确定性合并的父进程审计外保留的每个 worker 追加式审计 |
| `fold_workers/<config_id>/fold_<NN>/pysr_runs/<config_id>_fold_<NN>/` | 一个隔离 CV fold 的 PySR warm-start 状态 |
| `fold_workers/<config_id>/fold_<NN>/resume_<timestamp>/` | 中断的非终止配置被显式重跑时使用的干净 worker 尝试目录 |
| `pysr_runs/refinement/` | 最终精修的受限 warm-start 状态 |
| `pysr_runs/refinement_resume_<timestamp>/` | 显式恢复时创建的干净精修尝试目录，不复用中断的 Julia 状态 |
| `symbolic_regression.log` | 完整执行日志，包括每个分块后的简洁记录（config/fold/chunk/iterations/R²/CPU/decision）、运行模式、缩放/原始方程、源单位、缩放器定义、一致性审计和系数区间 |
| `results.txt` | 可读方程、项预算审计、单位/缩放解释、条件性系数不确定性、原始指标、选择规则、40 配置终止计数和搜索报告 |
| `results.json` | 机器可读的方程/审计、原始指标、配置和折记录、完成/剪枝/失败状态、选择规则、精修执行、运行时元数据和产物映射 |
| `selected_candidate.json` | 稳定的 PySR `candidate_position`、按索引预测契约、缩放/原始方程、直接二次项和超额复杂度审计以及配置选择规则 |
| `ridge_baseline.json` | 在相同外层划分上、使用相同 CV 折索引且不接触测试集拟合的无泄漏 Ridge 对照 |
| `equation_audit.json` | 稳定的缩放/原始及参数化方程、特征/目标单位、缩放器定义、项预算审计、数值等价性检查、bootstrap 设置/限制和系数摘要 |
| `coefficient_uncertainty.csv` | 原始常数、bootstrap 均值/中位数、条件标准误和百分位置信区间 |
| `coefficient_bootstrap_samples.csv` | 记录种子的固定结构外层训练 bootstrap 拟合成功和失败情况 |
| `pysr_model.pkl` | PySR 模型对象，序列化成功时生成；应使用 `selected_candidate.json`，不要依赖隐式模型选择 |
| `scaler.pkl` | 拟合的 `StandardScaler`，序列化成功时生成 |
| `scaler_params.json` | 特征顺序、源单位、目标定义和外层训练缩放参数 |
| `pysr_model_with_scaler.pkl` | 模型、缩放器、选中位置、显式预测契约、特征、目标、方程和审计包，序列化成功时生成 |
| `predictions.csv` | `source_row_index`、`Name`、划分、符号回归和 Ridge 预测值以及残差 |
| `prediction_plot_combined.png` | 计算值与预测值图 |
| `residual_plot_combined.png` | 残差图 |
| `Score_distribution.png` | 训练集/测试集目标分布 |

部分可解释性或序列化产物采用 best-effort 方式，可能在警告后跳过；应查看运行日志，不要只依据目录是否存在来判断。

发表配置网格采用固定种子无放回抽样；直接运行会尝试全部 40 个配置，不存在全局 R-squared 目标、全局 patience/window 终止或 Top-8 晋级。每个折的缩放器只在该折训练行上拟合，独立的外层 DUPLEX 测试集在固定一个最终带索引的候选前始终锁定。只有当理论最佳情况（所有未观测折均取 R-squared=1）仍无法达到已完成的 incumbent 时，才允许剪枝部分配置。原始 R-squared/RMSE/MAE 不会被改名为选择分数。

默认算子为定义域安全且对称的 `+`、`-`、`*` 和 `square`；不搜索除法、任意幂、`exp`、`log` 或 `sqrt`。PySR parsimony 为零。每个特征允许一个线性项和一个直接平方项，且不增加超额复杂度；交互、高于二次的幂、重复非线性项以及嵌套/复合变换会在 one-standard-error 选择中受到惩罚。原始节点复杂度仍作为审计字段和树规模硬资源上限，不作为科学选择惩罚。

每个 CV fit 在 100 次迭代的 warm-start 分块中保留 600 次迭代和 500 秒上限。中间分块评估有限预测、六特征资格、原始验证指标以及训练损失和验证 R-squared 前沿；配对 bootstrap 仅在终止分块计算一次。精修保留 250 次迭代分块、3,000 次迭代上限和默认 7,200 秒墙钟上限。在 32 个逻辑处理器的工作站上，CV 默认使用四个 spawned 进程、每个进程 8 个 Julia 线程，并为每个进程使用一个 BLAS 辅助线程；配置顺序仍为串行，并在每个完成的 worker 批次后评估保守剪枝。精修可以使用解析后的完整 Julia 线程预算。解析器防止总线程超配，记录全部进程/线程设置和计时，流程不会修改操作系统电源方案。`SR_DETERMINISTIC=1` 选择严格的单进程/单线程确定性模式。运行 `python sr_progress.py` 可查看一次性状态，运行 `python sr_progress.py --watch 10` 可只读实时查看；已有运行会回退到 `search_progress.jsonl`。
