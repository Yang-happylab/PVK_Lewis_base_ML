# PVK_Lewis_base_ML

[English](README.md) | 中文

本项目提供一套用于钙钛矿路易斯碱添加剂校正、建模、解释与筛选的机器学习流程。

本仓库对应一篇正在审稿的文章，论文发表后将补充引用信息与 DOI。

## 仓库结构

```text
.
├── data/
│   ├── calibration.csv
│   ├── data-s.csv
│   ├── data.csv
│   ├── device/                    # 器件 JV 原始数据
│   └── posterior_validation.csv
├── outputs/                       # 运行时生成
├── device_normalization.py        # 器件归一化与箱线散点图
├── device_score_validation.py     # Score 与最佳器件 PCE 排名验证
├── score_calibration.py           # 实验 Score 校正
├── data_preparation.py            # 公共运行时数据准备
├── campaign.py                    # Rxx 轮次选择
├── project_paths.py               # 仓库相对路径
├── structure_training.py          # 公共结构训练划分与绘图
├── rank_statistics.py             # 公共秩置换统计
├── GA_EN_KRR_BR_Lewis_base.py     # EN/KRR/BR 训练
├── GA_RF_XGB_Lewis_base.py        # RF/XGB 训练
├── GA_Lewis_base.py               # 遗传算法筛选
├── XGB_Lewis_base.py              # XGBoost 解释与 SHAP
├── posterior_score_validation.py   # 四组后验 Score 验证
├── posterior_validation_analysis.py
├── SR_Lewis_base.py               # 符号回归
├── sr_progress.py                 # 只读符号回归进度查看器
├── sr_efficiency.py               # 符号回归搜索辅助模块
├── sr_validation.py               # 划分、指标与 Pareto 选择
├── sr_reporting.py                # 原始量公式与系数不确定性
├── output_files_summary.md
└── requirements.txt
```

所有路径均从仓库目录自动解析。将整个项目复制到其他位置后，无需修改源码路径即可运行。

运行日志和特征名文本文件使用 UTF-8。使用文本编辑器查看时请选择 UTF-8 编码，以正确显示科学符号和中文路径。

## 安装

使用 Python 3.12.3，项目依赖版本固定在 `requirements.txt` 中。

```bash
git clone https://github.com/Yang-happylab/PVK_Lewis_base_ML.git
cd PVK_Lewis_base_ML
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

运行 `SR_Lewis_base.py` 还需要 Julia 和兼容的 PySR 1.x 环境。

## 输入数据

| 文件 | 用途 |
|---|---|
| `data/calibration.csv` | 用于校正 Score 的累积实验数据 |
| `data/data-s.csv` | `Name`、二维规范 `SMILES`、32 个数值结构特征，以及用于 Score 一致性检查的三个 DFT 描述符 |
| `data/data.csv` | 用于 XGBoost 和符号回归的理化与 DFT 描述符，包含稳定的 `Name` |
| `data/device/<Name>.csv` | R01 器件级 JV 原始参数；无 `_norm` 后缀的文件是唯一权威输入 |
| `data/posterior_validation.csv` | 按轮次保存的后验 `Name`/`SMILES`、描述符及 PLQY/XRD 实验数据 |

`Score` 在运行时生成，并严格按 `Name` 合并，不保存在 `data.csv` 或 `data-s.csv` 中。包括器件标准化数据在内的派生表均写入 `outputs/`。

`data/data.csv` 中的 `HOMO`、`LUMO` 和 `gap` 单位为 eV。`N_lig` 是单一配位模式下可同时与同一 Pb 配位的 N/O/S/Cl/Br 供体原子数的最大值。`Coordination_atom` 描述同一配位模式，`atom_N`、`atom_O`、`atom_S` 和 `atom_Cl` 按元素统计其中供体，同一元素的大小写标签合并；`atom_Cl` 仅统计 Cl。没有 Br 计数列，因此四项元素计数之和可小于 `N_lig`。`RingNum` 统计同一配位模式中提供供体的不同杂环数，不是分子中的全部环数。这些配位描述符与 GA 结构特征中的候选供体计数分别定义。

## 实验轮次

支持轮次的脚本接受 `--R01`、`--R02` 等参数。不提供轮次参数时，程序自动选择 `data/calibration.csv` 中最新的有效轮次。

校正累计使用截至所选轮次的数据；后验预测覆盖截至所选模型轮次的全部已有分子，实验验证逐轮进行。

## 运行流程

### Score 校正

生成版本化 Score 表、校正参数、来源记录及描述符相关图。
每个纳入轮次独立拟合，端点预测在同一完整分子库上分别做 Z-score，再将各批 Score 等权平均，不进行二次缩放。`batch_scores.csv` 保留各批 Score；`batch_agreement.csv` 报告分子库排序一致性，仅作为描述性结果。每批六张图位于 `batches/Rxx/`，根目录图片仅展示最新纳入批次。下游模型须使用所选 Score 版本。

每个 `Round` 对应一个批次。PLQY 和 XRD 先按 `100 × 原始值 / 同轮 Ref-PbI2 均值` 归一化，再执行拟合与 Score 计算。每个纳入轮次的 Ref-PbI2 均须设置 `Use_for_fit=true`，且每项指标具有三个正值测量。输入保留原始值；`normalized_calibration.csv` 记录原始值、参照均值和相对值。预测端点及图中百分比均相对于参照（100），不代表绝对 PLQY 或 XRD 值。

```bash
python score_calibration.py
python score_calibration.py --R01
```

输出：`outputs/score_calibration/campaign_Rxx/<score_version>/`

### 结构模型与遗传算法

使用 32 个数值描述符训练五个结构模型，并组合符合准入条件的模型，对候选路易斯碱结构进行遗传筛选。

```bash
python GA_EN_KRR_BR_Lewis_base.py --R01
python GA_RF_XGB_Lewis_base.py --R01
python GA_Lewis_base.py --R01
```

运行 GA 前需先完成两份训练脚本，并确保三个命令使用相同轮次。
`structure_features.py` 定义有序的 32 个特征。

GA 完成后，`top10_structures.png` 以两列展示 Top 10 分子结构，标注与 `optimal_samples.csv` 一致的排名和 Score。

`structure_selection.py` 采用嵌套的“调参—筛选—重新调参”流程：先对全部特征调参，比较接近最优且预测参数不同的配置，再按一个标准误差规则选择变量子集，最后对子集重新调参。ElasticNet、BayesianRidge、RF、XGB 使用 RFE，KRR 使用向后选择。外层评价与内层选择均采用 4 折 × 3 次重复 CV，预处理、特征筛选与调参只使用对应的训练行。内层指标指导选择，外层 CV 评价完整流程；外部 DUPLEX 测试集在选择结束后评价。

GA 准入使用外层嵌套 CV 指标。入选模型提供按分子汇总的嵌套 OOF 预测，用于训练 Ridge Stacking；GA 适应度为其预测 Score。DUPLEX 测试集不参与 Stacking 训练，GA 运行期间基模固定。模型的提取设置须与结构特征提取器一致。

各基模的筛选诊断和 OOF 预测保存在 `feature_selection/`，GA 运行在 `stacking/` 下保存 Stacking 模型及其评价。Ridge CV 指标指导调参，留出测试集指标评价拟合后的集成模型。

分子身份采用不含立体信息的规范 SMILES。GA 将历代出现的全部唯一结构保存为 `candidate_archive.csv`，并从包含最终种群的归档中选出 `optimal_samples.csv`。归档保留组装路径，合并等价路径；不同结构即使描述符相同仍分别保留。

31 个中心模板支持 11 种支臂选项，其中 H 表示该连接位点不接取代基。NH3、H2O、H2S、HCl 属于合法设计，但供体环境不受特征定义支持，列入每次运行的 `excluded_structures.csv`。

`requirements.txt` 固定 RDKit 版本，模型产物记录版本和提取参数。分子库中四个描述符为常量，由训练方差过滤处理。

输出：

- `outputs/structure_models/campaign_Rxx/`
- `outputs/genetic_algorithm/campaign_Rxx/`

### XGBoost 解释

`XGB_Lewis_base.py` 先执行分层 DUPLEX 外层划分，仅用外层训练行确定高相关特征过滤规则，并将同一组保留列应用于测试集；全库相关性热图用于描述性分析。每次 XGBoost 拟合使用种子为 8、18、188、1888、8888 的五个模型，参数与特征相同，预测等权平均。特征数量与模型参数通过 4 折 × 3 次重复 CV 联合选择，五个成员共用相同折划分，以平均预测计算 R²。每折仅用该折训练行拟合固定基线集成，再由相同训练行上的集成 SHAP 平均绝对值生成该折所有候选共用的特征排名。

本次搜索中的领先候选与已保存且通过兼容性核验的候选，共同执行 4 折 × 5 次重复复核，所有候选均重新评价。一个标准误差范围内依次优先选择更少特征、更浅的树和更少的树，再比较 CV 表现。配套的 `full_features` 模型也按同一简约规则选择，`selected_features` 可以保留全部可用特征。外层测试集在选择完成前保持锁定。

共识 OOF SHAP 作为主要解释：每折先对成员的有符号 SHAP 值求平均，再对每个外层训练分子的三次留出重复求平均。同一套集成 OOF 预测提供后验 Score 区间校准所用的共识残差，并保存种子成员的 OOF 值及排名稳定性诊断。最终拟合集成的 SHAP 及“特征–Score”散点图顺序采用训练集平均绝对集成 SHAP，作为补充解释。搜索诊断保存在 `Score/joint_search/`，模型和解释产物按模型阶段组织。

```bash
python XGB_Lewis_base.py
python XGB_Lewis_base.py --R01
```

输出：`outputs/xgb_interpretability/campaign_Rxx/`

### 符号回归

`SR_Lewis_base.py` 搜索可解释的 Score 公式，要求简化后恰好包含指定数量的不同描述符。默认直接从 `data/data.csv` 读取 `atom_N`、`gap`、`fai_min`、`μ`、`HOMO`、`TPSA` 六项输入，公式须包含全部六项。变量可只通过非线性或交互项出现，不要求包含一次项。`--features` 以受支持的原始名称或内部别名配置候选输入，`--selected-count` 指定准确变量数，可取 1 至候选池大小，默认六项。别名解析后的特征必须唯一。任一必需值或 Score 缺失、非数值或非有限时，程序按受影响的 `Name` 和列名拒绝输入。

SR 使用确定性的分层 DUPLEX 外层划分，直接使用与公式拟合相同的有序输入描述符计算，默认即上述六项描述符。通过外层训练行的 4 折 × 2 次重复 CV 比较 16 个配置，每折标准化器仅在该折训练行上拟合。只有完成全部 CV 的配置才能参与配置层面的 one-standard-error 选择。`+`、`-`、`*`、`square`、`sqrt`、`exp` 对所有描述符完全对称。每个描述符有一个免费线性项和一次免费非线性使用额度：直接平方、实数根号、指数或两变量交互；交互同时占用两个描述符的额度。额外使用、更高次幂及复合变换消耗超额复杂度预算。根号与指数允许仿射自变量。实数根号定义域无效或预测非有限时拒绝候选，不使用绝对值或裁剪替代。

合格公式跨搜索分块保留。单次精修继承获胜配置的 CV 公式，将其尺度转换为外层训练尺度，并在外层训练行上重新拟合原公式浮点常数。继承公式与精修公式共同归档。统一超额复杂度上限由 `--max-excess-complexity` 配置，默认为五，贯穿 CV 搜索、继承、精修、候选门禁和最终复核。CV 平台早停依据搜索分块累计的合格候选最佳验证前沿。精修每25次迭代检查累计合格候选的训练 loss 与简约性前沿；预热200次后，连续100次没有同等或更低复杂度下的明显误差改善，也没有相近误差下更简洁的公式时停止，同时保留两小时上限。这些检查仅控制停止，最终公式由联合候选复核选择。

联合复核对继承与精修结构去重，在外层训练行上执行固定结构的 4 折 × 5 次重复 CV，采用种子 2026 生成的单独折划分。每折独立拟合标准化器，从两个中性初值重新拟合原公式浮点常数，并保留公式结构；仿射根号中的偏置可仅依据该折训练输入调整初值以满足定义域。每个重拟合候选均须满足准确变量数、有限值与统一复杂度上限。正式模型在最高 CV 均值减一个标准误的范围内，依次优先选择更低的超额复杂度、更少的合并后项数、更少的独立参数及更低的结构复杂度。标准误使用各次重复的均值计算。该复核评价搜索与精修发现的候选结构，属于条件性选择诊断，与配置 CV 分开报告。

one-SE 与 two-SE 两层各展示不超过十个候选，共用最高均值及其标准误作为基准。两层保留不同性能与简洁程度的代表，共用候选编号，并合并数学等价的拟合公式。仅入选清单的候选在全部外层训练行上重新拟合常数。`candidate_comparison.json`、`.csv`、`.txt` 报告高精度标准化及原始单位公式、CV 与训练指标、重复及逐折诊断、复杂度和重拟合状态。候选准入与自动 one-SE 选择仅使用训练侧 CV；two-SE 为备选。

候选清单及自动选择固定后，`candidate_results/<candidate_id>/` 为两层中每个不同候选导出公式、CV/训练/测试指标、按 Name 关联的预测值、预测对真值图、残差图、残差直方图及预测与残差组合图。跨层共享的候选只导出一次。`summary.csv` 便于综合比较预测表现、简洁性与化学可解释性；这些诊断不会自动更换选中模型。重拟合失败或预测无效的候选保留明确状态。图片采用 Arial，候选间使用相同坐标范围。已完成的运行可不训练、不启动 Julia，单独导出：

```bash
python sr_reporting.py --run-directory outputs/symbolic_regression/campaign_R02/<run_directory>
```

导出保留已有模型文件，若 `candidate_results` 目录已存在则拒绝覆盖。独立图片沿用 XGBoost 的 10×8 英寸画布及常规坐标比例；仅在明确要求时单独生成正方形论文图片。

线性 Ridge 与加性二次 Ridge 均为诊断基线，采用最终 SR 变量及相同外层划分。二次基线包含各变量的线性项和平方项，不含交互项。两种基线分别通过训练侧 CV 和折训练标准化器选择正则化强度，其 CV 指标属于条件性调参诊断。外层测试评价在公式与基线固定后进行，测试指标不参与自动选择。产物包括标准化公式、原始单位公式、模型指标、候选诊断、预测与来源记录；文件明细见 [output_files_summary.md](output_files_summary.md)。

```bash
python SR_Lewis_base.py
python SR_Lewis_base.py --R01
python SR_Lewis_base.py --R02
```

若运行被中断，可显式指定原运行目录继续：

```bash
python SR_Lewis_base.py --resume-run outputs/symbolic_regression/campaign_R01/<run_directory>
```

恢复要求输入、Score 版本、候选特征、要求的变量数、划分、实现与运行设置匹配。核验通过的已完成配置可复用，中断的工作重新执行。工作进程失败时报告对应折及错误文本。

在另一终端查看最近一次运行，或持续跟踪进度：

```bash
python sr_progress.py
python sr_progress.py --watch 10
```

查看器只读取保存的进度，不导入 PySR 或启动 Julia。

输出：`outputs/symbolic_regression/campaign_Rxx/`

### 模型后验验证

`posterior_score_validation.py` 使用 XGBoost、GA Stacking、符号回归及三者等权平均的预测 Score，预测截至所选模型轮次的全部后验分子。模型绑定共同的 Score 版本。各实验轮次分别使用同期 `Name=REF` 和完整三重复，在固定标尺上计算实测 Score；排名统计和图按轮次分别输出。后期模型对较早轮次的结果标记为回顾性。

自动发现 XGBoost 时，仅接受所选轮次中已经完成、且具有有效 Score 与共识 OOF 残差绑定的模型。

```bash
python posterior_score_validation.py --R02
python posterior_score_validation.py --R01 --models ga sr
python posterior_score_validation.py --R02 --sr-run outputs/symbolic_regression/campaign_R02/<run_directory> --sr-candidate <candidate_id>
```

命令行和 `predict_posterior` 函数默认运行 `xgb ga sr` 并输出含平均 Score 的四组结果；`--models` 可选择子集，此时不生成三模型平均。`--sr-run` 绑定已保存的 SR 运行，`--sr-candidate` 显式采用该运行 one-SE/two-SE 清单内已完成拟合的公式；程序核验其完整 CV 审计、标准化器及公式，清单记录本次显式选择与来源，保留已保存的自动模型。每轮实验摘要只计算一次，由该轮各预测组共用。模型属于所选轮次；模型 Rxx 评估截至 Rxx 的全部已有轮次，但不合并秩统计。不提供轮次参数时选择最新校正轮次。

输出：`outputs/posterior_predictions/model_Rxx/`

### 器件归一化与绘图

将每个分子原始文件中的全部器件行除以同一文件全部 `Control` 行的对应指标均值。文件名带 `_norm` 的 CSV 绝不作为输入。器件数据仅包含 R01。

逐个处理已有原始 CSV。缺少 Group 或任一有限 JV 指标的记录会跳过并打印 CSV 行号；缺列、标签无效或无有效 Control 基准的文件会跳过并说明原因。汇总只包含成功处理的文件。轮次参数指定输出标签，不按实验轮次筛选原始文件。

`2Py2N.csv` 属于最终器件验证数据，在文件发现阶段自动排除，不进入归一化或绘图。原始 CSV 保留并打印排除原因，不向两份汇总写入记录；若没有可用文件，命令返回状态码 1，不写入产物。

```bash
python device_normalization.py --R01
```

输出：

- `outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv`
- `outputs/device_analysis/campaign_Rxx/<Name>/{PCE,Voc,FF,Jsc}_boxplot.png`
- `outputs/device_analysis/campaign_Rxx/group_summary.csv`
- `outputs/device_analysis/campaign_Rxx/normalization_summary.csv`

### 器件 Score 验证

比较版本化校正 Score 与每个分子观测到的最佳非 Control 归一化 PCE。应先运行器件归一化流程；本流程直接读取现有 `group_summary.csv`，仅使用 `normalization_summary.csv` 恢复对应的原始 PCE 均值，不重复归一化。

```bash
python device_score_validation.py --R01
```

输出：

- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/best_concentration_summary.csv`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/rank_correlation_statistics.csv`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/score_pce_rank_comparison.png`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/score_vs_best_normalized_pce.png`
- `outputs/device_score_validation/campaign_Rxx/<analysis_version>/analysis_manifest.json`

## 公共模块

`data_preparation.py`、`campaign.py`、`project_paths.py`、`structure_training.py`、`rank_statistics.py`、`sr_efficiency.py`、`sr_validation.py` 和 `sr_reporting.py` 由入口脚本自动导入，不需要单独运行。`data_preparation.py` 提供公共 `Coordination_atom` 解析。`structure_training.py` 提供结构模型数据核查、DUPLEX 划分及公共绘图；训练与报告均使用 `structure_selection.py` 中的 CV 配置。器件与后验报告使用 `rank_statistics.py`，并采用各自文档规定的 CSV 列名。`sr_progress.py` 是独立、只读的诊断入口。

## 输出与版本控制

所有运行生成的模型、日志、表格、图片和分析汇总均保存在 `outputs/` 下。

完整产物列表见 [output_files_summary.md](output_files_summary.md)。

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

## 引用

如使用本项目，请引用对应论文：

> 作者姓名，*论文标题*，期刊，年份（审稿中）。
> 代码：[https://github.com/Yang-happylab/PVK_Lewis_base_ML](https://github.com/Yang-happylab/PVK_Lewis_base_ML)
