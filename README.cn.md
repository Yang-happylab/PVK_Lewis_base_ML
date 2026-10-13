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

```bash
git clone https://github.com/Yang-happylab/PVK_Lewis_base_ML.git
cd PVK_Lewis_base_ML
python -m venv .venv
python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境；Linux/macOS 使用 `source .venv/bin/activate`。

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

`data/data.csv` 中的 `HOMO`、`LUMO` 和 `gap` 单位为 eV。`Coordination_atom` 提供 `atom_N`、`atom_O`、`atom_S` 和 `atom_Cl` 的元素计数，同一元素的大小写标签合并统计。`RingNum` 统计可同时配位的环位点数，而非分子结构中的全部环数。

## 实验轮次

支持轮次的脚本接受 `--R01`、`--R02` 等参数。不提供轮次参数时，程序自动选择 `data/calibration.csv` 中最新的有效轮次。

校正累计使用截至所选轮次的数据；后验预测覆盖截至所选模型轮次的全部已有分子，实验验证逐轮进行。

## 运行流程

### Score 校正

生成版本化 Score 表、校正参数、来源记录及描述符相关图。
每个纳入轮次独立拟合，端点预测在同一完整分子库上分别做 Z-score，再将各批 Score 等权平均，不进行二次缩放。`batch_scores.csv` 保留各批 Score；`batch_agreement.csv` 报告分子库排序一致性，仅作为描述性结果。每批六张图位于 `batches/Rxx/`，根目录图片仅展示最新纳入批次。下游模型须使用所选 Score 版本。

每个 `Round` 对应一个批次。PLQY 和 XRD 先按 `100 × 原始值 / 同轮 Ref-PbI2 均值` 归一化，再执行原有拟合与 Score 计算。每个纳入轮次的 Ref-PbI2 均须设置 `Use_for_fit=true`，且每项指标具有三个正值测量。输入保留原始值；`normalized_calibration.csv` 记录原始值、参照均值和相对值。预测端点及图中百分比均相对于参照（100），不代表绝对 PLQY 或 XRD 值。

```bash
python score_calibration.py
python score_calibration.py --R01
```

输出：`outputs/score_calibration/campaign_Rxx/<score_version>/`

### 结构模型与遗传算法

使用 32 个数值描述符训练五个作为 GA 适应度函数的结构模型，然后筛选候选路易斯碱结构。

```bash
python GA_EN_KRR_BR_Lewis_base.py --R01
python GA_RF_XGB_Lewis_base.py --R01
python GA_Lewis_base.py --R01
```

运行 GA 前需先完成两份训练脚本，并确保三个命令使用相同轮次。
`structure_features.py` 定义有序的 32 个特征。

GA 完成后，`top10_structures.png` 以两列展示 Top 10 分子结构，标注与 `optimal_samples.csv` 一致的排名和 Score。

`structure_selection.py` 对 ElasticNet、BayesianRidge、RF、XGB 使用 RFE，对 KRR 使用向后选择。外层评价与内层选择均采用 4 折 × 3 次重复 CV。每次拟合依次执行全变量调参、比较最多两组接近最优且预测参数不同的筛选配置、按一个标准误差规则选择变量子集、对子集重新调参。每阶段参数搜索 RF 最多 588 组，XGB 最多 1288 组，EN/BR 最多 988 组，KRR 最多 788 组。内层指标仅作选择诊断，外层 CV 评价完整流程；外部测试集在筛选完成后评价。RF/XGB 每个并行搜索任务内使用单线程。

GA 准入使用外层嵌套 CV 指标。入选模型提供按分子汇总的嵌套 OOF 预测，用于训练 Ridge Stacking；GA 适应度为其预测 Score。DUPLEX 测试集不参与 Stacking 训练，GA 运行期间基模固定。模型的提取设置须与结构特征提取器一致。

各基模的筛选诊断和 OOF 预测保存在 `feature_selection/`。GA 运行在 `stacking/` 下保存 Stacking 模型、指标、预测、系数与图。Ridge CV 指标用于调参诊断；`duplex_test` 指标评价完整集成在留出集上的表现。

RF/XGB 参数搜索范围在 `GA_RF_XGB_Lewis_base.py` 中配置，涵盖树容量、抽样与正则化。

分子身份采用规范 SMILES，不由特征向量定义。GA 将历代出现的全部 SMILES 去重候选保存为 `candidate_archive.csv`，并从包含最终种群的历史档案中选出 `optimal_samples.csv`。两张表均包含 `SMILES`、`Center` 和 `Arm1`–`Arm4`；等价组装路径合并，不同结构即使描述符相同仍予保留。

31 个中心模板支持 11 种支臂选项，其中 H 表示该连接位点不接取代基。NH3、H2O、H2S、HCl 属于合法设计，但供体环境不受特征定义支持，列入每次运行的 `excluded_structures.csv`。

`requirements.txt` 固定 RDKit 版本，模型产物记录版本和提取参数。当前库中四个描述符为常量，由训练方差过滤处理；

输出：

- `outputs/structure_models/campaign_Rxx/`
- `outputs/genetic_algorithm/campaign_Rxx/`

### XGBoost 解释

XGBoost 采用一次联合搜索，共评价 58,888 组唯一候选，并尽量等量分配到 4、6、8、10 和全部可用特征（当前为 13）五种特征数。随机种子 42 下，60% 候选来自 focused 分支，40% 来自 exploration 分支；树深主要覆盖 2–8，同时保留少量深度 10 和 12。4 折 × 3 次重复训练 CV 的每个折仅用该折训练数据拟合一个固定基线 XGBoost 模型（200 棵树、深度 4、学习率 0.05、随机种子 42）。TreeExplainer 在相同训练行上的平均绝对 SHAP 值给出一条共享特征排名；每种目标特征数取排名前 K 项，并按原始列顺序组装后再拟合候选。

每种特征数的前两名候选再用随机种子 2026 执行额外的 4 折 × 5 次重复复核。一个标准误差规则使用各重复均值的标准误差：在最佳均值减去其标准误差的范围内先选择最少特征数，再选择最高均值。全特征候选复核结果中的最佳者用于配套的 `full_features` 模型；最终 `selected_features` 胜者也可以使用全部特征。外部测试集只在选择结束后评价。搜索诊断统一保存在 `Score/joint_search/`。产物包括全特征和选中特征模型、图、契约及固定最终模型 OOF SHAP 文件。最终模型重要性数值和“特征-Score”散点图顺序使用训练集平均绝对 SHAP，共识 OOF SHAP 是主要解释结果。搜索、选择、拟合与解释的科学逻辑均位于 `XGB_Lewis_base.py`。

```bash
python XGB_Lewis_base.py
python XGB_Lewis_base.py --R01
```

输出：`outputs/xgb_interpretability/campaign_Rxx/`

### 模型后验验证

`posterior_score_validation.py` 使用 XGBoost、GA Stacking、符号回归及三者等权平均的预测 Score，预测截至所选模型轮次的全部后验分子。模型绑定共同的 Score 版本。各实验轮次分别使用同期 `Name=REF` 和完整三重复，在固定标尺上计算实测 Score；排名统计和图按轮次分别输出。后期模型对较早轮次的结果标记为回顾性。

```bash
python posterior_score_validation.py --R02
python posterior_score_validation.py --R01 --models ga sr
```

命令行和 `predict_posterior` 函数默认运行 `xgb ga sr` 并输出含平均 Score 的四组结果；`--models` 可选择子集，此时不生成三模型平均。模型属于所选轮次；模型 Rxx 评估截至 Rxx 的全部已有轮次，但不合并秩统计。不提供轮次参数时选择最新校正轮次。

输出：`outputs/posterior_predictions/model_Rxx/`

### 符号回归

搜索保留全部六个描述符的可解释 Score 方程。发表配置对六个描述符一视同仁，默认只使用定义域安全的 `+`、`-`、`*` 和 `square`。每个描述符允许一个线性项和一个直接平方项而不增加超额复杂度；交互项、更高次幂、同一变量的重复非线性项以及嵌套变换会在基于验证集的选择阶段受到压制。

默认运行会尝试 40 组固定种子、无放回且不重复的配置，并执行重复交叉验证。每一折的标准化器只在该折训练行上拟合。Pareto 候选按明确的行位置寻址，并依据验证预测和可追溯的 one-standard-error 规则选择；独立外层测试集只在精修结束后使用一次。报告严格区分原始 R²/RMSE/MAE、选择分数和复杂度字段，同时提供使用相同外层划分的 Ridge 基线。

标准化特征公式和可直接代入原始单位的反标准化公式都是正式产物。

```bash
python SR_Lewis_base.py
python SR_Lewis_base.py --R01
```

若运行被中断，可显式指定原运行目录继续：

```bash
python SR_Lewis_base.py --resume-run outputs/symbolic_regression/campaign_R01/<run_directory>
```

程序会从 `data/` 重新生成受控输入，严格核对数据、Score 版本、拆分、采样配置、科学实现文件与参数、运行布局和软件包版本，仅跳过已验证的终态配置。正在运行时中断的配置会从头重跑；中断的 refinement 会以单独命名的新尝试重新开始。CV 选择会在停止 fold workers 之前先保存。worker 默认最多等待 60 秒正常退出，只有确认没有待处理任务后，才会强制清理仍未退出的空闲进程；必要时可用 `SR_FOLD_SHUTDOWN_GRACE_SECONDS` 调整等待时间。

可在另一个终端中查看最近一次运行，或持续跟踪；该命令不会导入 PySR，
也不会启动第二个 Julia 运行时：

```bash
python sr_progress.py
python sr_progress.py --watch 10
```

查看器会明确显示当前是第几组/40、正在运行的 fold worker、chunk、累计迭代预算、验证 R²、
实测等效 CPU 核心数、配置/实际 Julia 线程数以及终止决定。新运行会在每个
chunk 后原子更新简洁的 `progress_status.json`；查看器也兼容仅含详细
`search_progress.jsonl` 的既有运行。

可选运行参数：

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

在 i9-13900KF 同级工作站上，CV 默认使用四个相互隔离的 fold 进程，每个进程使用 8 个 Julia 线程（`4x8`）；最终精修可使用 `SR_JULIA_THREADS` 解析出的完整线程预算。每个 BLAS 辅助线程池仍固定为 1 个线程，解析器会拒绝总 Julia 线程数超过 Windows 或 WSL 可见逻辑处理器数的布局。设置 `SR_FOLD_WORKERS=2` 和 `SR_FOLD_WORKER_THREADS=16` 可使用两个 fold 进程；设置为 `1` 和 `32` 则关闭 fold 级进程并行。日志会报告墙钟时间、等效核心数、配置/实际线程数和所有活动 folds；任务管理器占用率本身不是吞吐率证据。`SR_DETERMINISTIC=1` 会切换到较慢的单进程单线程严格确定性模式。流程不会修改操作系统电源方案。

候选的配对 bootstrap 只在每折终止检查点计算；中间检查点仍执行有限值、六变量、验证指标和 Pareto 审计，但不重复支付 bootstrap 成本。CV 平台状态依据各超额复杂度预算下历史最佳的验证 R² 上包络判断，单独改善训练 loss 不会重置验证平台计数；没有折内验证数据的最终精修仍使用有界的外层训练 loss 前沿。每折上限为 600 次迭代/500 秒，精修上限为 3000 次迭代/7200 秒。

输出：`outputs/symbolic_regression/campaign_Rxx/`

### 器件归一化与绘图

将每个分子原始文件中的全部器件行除以同一文件全部 `Control` 行的对应指标均值。文件名带 `_norm` 的 CSV 绝不作为输入。当前器件数据仅包含 R01。

逐个处理已有原始 CSV。缺少 Group 或任一有限 JV 指标的记录会跳过并打印 CSV 行号；缺列、标签无效或无有效 Control 基准的文件会跳过并说明原因。汇总只包含成功处理的文件。轮次参数指定输出标签，不按实验轮次筛选原始文件。

```bash
python device_normalization.py --R01
```

输出：

- `outputs/device_analysis/campaign_Rxx/<Name>/<Name>_norm.csv`
- `outputs/device_analysis/campaign_Rxx/<Name>/{PCE,Voc,FF,Jsc}_boxplot.png`
- `outputs/device_analysis/campaign_Rxx/group_summary.csv`
- `outputs/device_analysis/campaign_Rxx/normalization_summary.csv`

### 器件 Score 验证

散点图的样本数和秩统计标注位于图内左上角。

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

`data_preparation.py`、`campaign.py`、`project_paths.py`、`sr_efficiency.py`、`sr_validation.py` 和 `sr_reporting.py` 由入口脚本自动导入，不需要单独运行。`sr_progress.py` 是独立、只读的诊断入口。

## 输出与版本控制

所有运行生成的模型、日志、表格、图片和分析汇总均保存在 `outputs/` 下。

完整产物列表见 [output_files_summary.md](output_files_summary.md)。

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

## 引用

如使用本项目，请引用对应论文：

> 作者姓名，*论文标题*，期刊，年份（审稿中）。
> 代码：[https://github.com/Yang-happylab/PVK_Lewis_base_ML](https://github.com/Yang-happylab/PVK_Lewis_base_ML)
