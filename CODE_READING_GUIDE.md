# DL_fusion 项目代码阅读指南

> 适用对象：第一次接触本项目的开发者/研究者。
> 本文按"先懂业务、再懂数据、最后懂模型"的顺序组织，读完可以对每一部分代码的含义和调用关系有整体把握。

## 1. 项目在解决什么问题

桥梁在服役中可能出现**材料刚度削弱**与**支座沉降**等损伤。项目基于可观测的多模态响应，对桥梁安全状态做**分层辅助评估**：

| 任务层次 | 内容 | 监督形式 |
|---|---|---|
| 材料损伤 | 6 种候选材料是否削弱 | 多标签二分类 |
| 支座沉降 | 4 个支座是否沉降 + 幅值 | 二分类 + 回归 |
| 区域风险 | 7 个结构区域的 4 级风险 | 4 级分类 |
| 全桥安全 | 整体安全状态（取区域最大值） | 4 级分类 |

输入是两种异构响应：**多温度静态位移**（`[6温度步, 6测点, 3方向]`）和**参考温度下振动加速度时程**（`[4000步, 6测点, 3方向]`），通过双分支网络提取特征后融合。

```
result_*.json (V2 数据)
   │  StructuralDataset
   ▼
batch {disp, ace, targets, condition, temperature, quality_metrics, metadata}
   │  build_model (static_only / dynamic_only / fusion)
   ▼
StaticBranch ──┐  ClasswiseLateFusionHead (α 逐类融合)
DynamicBranch ─┘  ReliabilityGatedFusionHead (质量门控, 备用)
   │  multitask_loss (多任务 + 层级一致性)
   ▼
checkpoint (.pt) + logs (指标/图/样本预测) + runs (实验隔离)
```

## 2. 建议阅读顺序

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `README.md` / `RESEARCH_PLAN.md` | 项目是什么、研究计划是什么 |
| 2 | `DL_model/DL_config.py` | **全部超参数**，先建立全局概念 |
| 3 | `DL_model/safety_rules.py` | 安全标签（真值）是怎么定义的 |
| 4 | `DL_model/dataset.py` | 数据 JSON → 训练张量的转换 |
| 5 | `DL_model/model.py` | 网络结构 |
| 6 | `DL_model/trainer.py` | 损失、指标、训练/评估工具 |
| 7 | `DL_model/main.py` | **训练入口**，把以上所有串起来 |
| 8 | `plot/` 包 | 各种图怎么画出来 |
| 9 | `scripts/audit_dataset.py`、`scripts/make_splits.py` | 数据审计与划分 |
| 10 | `DL_model/predict.py`、`quick_predict.py` | 训练完怎么推理 |

快捷定位表：想看懂"某个超参数"→ `DL_config.py`；"标签怎么来的"→ `safety_rules.py`；"一个 batch 长什么样"→ `dataset.py` + `trainer.py` 的 `collate_fn`；"网络有几层"→ `model.py`；"训练过程发生了什么"→ `main.py` 的 `train_one_model` + `trainer.py` 的 `train_epoch`/`validate`。

## 3. 目录结构

```
DL_fusion/
├── DL_model/
│   ├── DL_config.py        # 全局配置（数据/模型/训练/增强超参数）
│   ├── safety_rules.py     # 工程安全规则 → 四层安全标签
│   ├── dataset.py          # 数据集加载器（V2 JSON → 张量）
│   ├── model.py            # 网络结构（分支/融合头/任务头）
│   ├── trainer.py          # 训练/评估工具（损失、指标、增强、EMA）
│   ├── main.py             # 训练/测试/推理主入口
│   ├── predict.py          # 批量预测脚本
│   ├── quick_predict.py    # 单文件快速推理
│   ├── checkpoints/        # 模型检查点（{model_type}/{run}_best.pt）
│   └── logs/               # 训练日志、指标 JSON、图片
├── plot/                   # 绘图包（prediction/training/evaluation/comparison/data_visualization）
├── scripts/                # 数据审计与划分脚本（实验准备）
├── runs/                   # （新增）实验隔离输出目录 runs/<data_version>/<run_id>/
├── README.md               # 项目说明
├── RESEARCH_PLAN.md        # 研究计划
└── DATASET_AUDIT.md        # 数据审计报告
```

## 4. 各文件详解

### 4.1 `DL_config.py` — 全局配置

整个项目的"总开关"，所有超参数集中在这里。阅读时按块理解：

- **路径与版本**（顶部）：`data_root`（当前指向 `E:\working\DL_data\temperature_multistep_json_result`）、`save_root`/`log_root`（checkpoints 与 logs）、`experiment_root`/`data_version`/`run_id`/`split_manifest`（实验隔离与追溯，配合 `main.py` 的 `--run-id` 使用）。
- **任务规模**：`n_materials`（6）、`n_supports`（4）、`n_regions`（7）、`n_safety_levels`（4）、节点/通道数。
- **训练**：`batch_size`、`lr`、`n_epochs`、`early_stop`、`selection_metric`（选 checkpoint 的评分标准）、阈值搜索设置。
- **划分**：`train_ratio`/`val_ratio`/`split_strategy`/`group_by`（按结构状态分组划分，防泄漏）。
- **损失权重**：`pos_weight`（正样本加权）、`aux_weight`（幅值回归）、`support_*`、`region_weight`、`global_state_weight`、`rule_consistency_weight`（层级一致性）。
- **融合策略**：`fusion_*` 一族的配置，注意当前设置 `fusion_train_alpha_only=False`、`fusion_safety_source="fused"`。
- **增强**：`*_aug_*` 一族（动态/静态分别配置）。
- **底部常量**：`CANDIDATE_IDS`、`SUPPORT_NODES`、`STATIC_NODES`、`DYNAMIC_NODES`。

要点：改实验配置几乎都在这一个文件里改；`Config.method_dir()` 决定每个模型类型输出到哪个子目录。

### 4.2 `safety_rules.py` — 工程安全规则（标签的定义）

把"物理量"转成"安全标签"的规则，**既用于数据集生成，也用于训练**。

- **常量**：候选材料 ID `[8,9,19,20,24,41]`、支座节点 `[221,513,395,687]`、4 级风险 `["safe","warning","risk","danger"]`。
- **阈值**（`SafetyThresholds`）：材料缩放因子 ≤ 0.90/0.80/0.70 → 预警/风险/危险；支座沉降 ≥ 2/5/7 mm → 预警/风险/危险。注意材料缩放因子"**越小越严重**"（1.0 = 无损，0.8 = 刚度损失 20%）。
- **区域定义**（`REGION_DEFINITIONS`）：7 个区域，每个区域由材料 ID 和支座节点归属；区域风险 = 区域内材料/支座风险最大值；全桥风险 = 所有区域最大值。
- **核心函数** `build_safety_labels()`：输入材料缩放因子和支座沉降 → 输出材料二分类标签、材料/支座/区域/全桥风险等级。这就是 `dataset.py` 里 `_labels_for_sample` 的默认逻辑。

要点：标签不是模型预测出来的，而是由"物理真值 → 阈值规则"生成的；模型学的是响应 → 标签的映射。

### 4.3 `dataset.py` — 数据加载

V2 数据集加载器，职责是"**一个 JSON → 一个样本的所有张量**"。

- **读取**：`glob` 找 `result_*.json`；支持两种响应存储：JSON 内嵌数组，或 `array_store` 指向的 HDF5（惰性加载，`lazy_responses` 模式，读取时才校验 SHA-256）。
- **兼容两种 schema**：V2 嵌套结构（`responses.static.disp`）和旧版扁平结构，`_nested_response()` 负责查找。
- **节点映射**：`node_maps` 把节点 ID 映射到数组列索引；`_as_response_array()` 按训练所需的节点集合（静态/动态测点）切出 `[steps, nodes, channels]` 张量。
- **标签构建**：`_labels_for_sample()` 优先读 JSON 里的 `safety_labels`，缺失时用 `safety_rules.build_safety_labels` 现算。
- **温度条件**：`_temperature_condition()` 计算温度相对参考温度的温差，归一化为 `condition`（`delta/20`）。
- **新增的追溯信息**：`_sample_metadata()` 提取每个样本的 `sample_id`、`structural_state_id`、激励/温度种子等 15 个字段；`quality_features` 把 `quality_metrics` 转为数值张量（8 个标量特征，缺失的键补 0 并警告）；`get_sample_metadata(include_hash=True)` 可算文件 SHA-256。
- **分组**：`get_group_labels()` 提供按损伤模式/缩放档位/文件块/安全状态/结构状态五种分组，用于防泄漏划分。
- **归一化**：`fit_normalizer()` 只从训练集算均值/标准差，`normalizer_state_dict()` 存进 checkpoint。
- **`__getitem__` 返回的 batch 元素**：`disp`、`strain`（零占位）、`ace`、`target`（材料标签）、`raw_sf`（缩放因子真值，仅监督用）、`support_*`、`region_target`、`global_target`、`condition`、`temperature_C`、`temperature_steps_C`、`quality_metrics`、`metadata`。

要点：先看 `__getitem__` 的返回值，就知道模型"能看到什么、监督什么"。

### 4.4 `model.py` — 网络结构

- **通用组件**：
  - `PositionEncoding`：测点坐标 → 中心化/缩放 + 傅里叶特征 → MLP 位置编码（避免原始毫米坐标直接进入）。
  - `PositionFiLM`：位置嵌入调制特征（`feat * (1 + γ) + β`）。
  - `AttentionPool`：对节点 token 做注意力池化。
  - `SafeBN`：batch_size=1 时跳过 BatchNorm（防推理时崩）。
- **两个分支**：
  - `StaticBranch`：每个测点的多温度位移序列展平 → 节点 MLP → 位置 FiLM → 注意力池化 → 节点级特征（64 维）。旧 strain 输入已移除，仅保留兼容参数。
  - `DynamicBranch`：六测点三向加速度 → 3 层 1D 卷积（每层后 MaxPool）→ 自适应最大池化 → MLP。
- **任务头**：`MultiTaskPredictionHead` 输出材料分类/回归、支座分类/回归、区域风险（reshape 为 `[B, n_regions, 4]`）、全桥风险。
- **两个融合头**：
  - `ClasswiseLateFusionHead`（**当前主模型在用**）：`logits = α * static + (1-α) * dynamic`，每个材料一个可学习权重 α（sigmoid 约束在 0~1）。α 接近 1 信静态、接近 0 信动态，**可解释、可汇报**。
  - `ReliabilityGatedFusionHead`（备用方案）：以动态为锚点，门控 `gate` 决定"静态修正"注入多少；门控输入含两分支特征、差异、乘积和分支置信度；**只能解释为信息注入比例，不是物理可靠性概率**。
- **三个模型**：`StaticOnlyModel`、`DynamicOnlyModel`、`DualBranchFusion`，由 `build_model(config)` 按 `config.model_type` 选择。
- **融合模型的三阶段训练**（`stage` 属性）：`pretrain`（双分支各训 + 对比损失）→ `fusion_head`（只训融合头/α）→ `finetune`（端到端微调）。`fusion_safety_source` 决定安全任务（支座/区域/全桥）用哪个特征（static/dynamic/fused）。

### 4.5 `trainer.py` — 训练与评估工具

不直接训练，提供训练所需的"零件"：

- **损失**：`multitask_loss()` 把材料分类/回归、支座分类/回归、区域、全桥、层级一致性（`relu(区域最高期望等级 - 全桥期望等级)`）按 `config` 权重加和。
- **指标**：`compute_metrics` / `compute_full_metrics` 输出宏 F1、AUC、精确率/召回率、Exact Match，以及支座/区域指标；`find_best_thresholds` 在验证集上按召回率约束搜阈值。
- **增强**：`augment_dynamic_response`（噪声/幅值/裁剪/平移/测点失活）、`augment_static_response`（位移计误差模拟）、`apply_fusion_modality_dropout`（模态失活）。
- **EMA**：`EMAModel` 指数滑动平均权重，训练更稳。
- **训练循环**：`train_epoch` 一个 epoch 的前向/反向/指标统计。
- **评估循环**：`validate` 验证/测试；`return_predictions=True` 时额外返回逐样本预测记录（`build_prediction_rows`），由 `save_sample_predictions` 存成 JSON —— 支撑错误分析和论文作图。
- **数据装载**：`collate_fn` + `make_loader`（注意 `pin_memory=False` 是刻意设置，服务器上别改回去）。

### 4.6 `main.py` — 训练/测试/推理入口

把 config → dataset → model → trainer 串成完整流程，**阅读的终点**：

- **CLI 参数**：`--mode`（train/test/infer）、`--model`（fusion/static_only/dynamic_only/all）、`--data-dir`、`--epochs/--batch/--lr/--seed`、`--resume`；**新增**：`--run-id`、`--output-dir`、`--split-manifest`、`--data-version`。
- **实验隔离**：`configure_run_dirs()` 把输出写到 `runs/<data_version>/<run_id>/`（不指定时保持旧目录兼容）。
- **划分**：`split_indices()` / `split_indices_by_group()` 按 `group_by` 做组级划分（同一结构状态不跨集合）；`save_split_manifest()` / `load_split_manifest()` 保存/加载 ID 级划分清单，加载时校验每个样本的 SHA-256（数据被替换会报错）。
- **训练**：`train_one_model()` 是核心，流程为：
  1. 建数据集、划分、建 loader；
  2. 按模型类型设置阶段：融合模型 warm start（`warm_start_fusion_branches` 加载单分支权重）→ `pretrain` → `fusion_head`（可选 `set_fusion_alpha_only_trainable` 只训 α）→ `finetune`；
  3. 每个 epoch：`train_epoch` + `validate`，按 `selection_score`（默认 `0.7*F1 + 0.3*AUC`）保存最优 checkpoint（含 history、归一化参数、划分、阈值、配置）；
  4. 测试：`validate(return_predictions=True)` → 画评估图 → 存样本预测 JSON → 汇总到 `method_comparison.json`。
- **跨域**：`build_external_dataset()` 加载参考模型数据集，`plot_domain_comparison` 画同源 vs 跨域对比。

### 4.7 `predict.py` / `quick_predict.py` — 推理

- `predict.py`：批量推理。`PredictContext` 是从单个 JSON 提取节点坐标/响应布局的轻量类（不用扫整个数据集），配合 checkpoint 里的归一化统计量；`load_model` 加载权重，`predict_one` 单样本推理（返回完整预测），`print_result` 打印，并用 `plot.save_prediction_figures` 画预测图。
- `quick_predict.py`：改 `file_path` 直接运行的快速脚本。

### 4.8 `plot/` 包 — 绘图

按功能分模块，所有图统一中文字体、300 dpi：

| 模块 | 图 |
|---|---|
| `prediction.py` | 单样本预测结果图（材料/支座/区域/概率） |
| `training.py` | 训练历史曲线（`plot_history`，6 子图） |
| `evaluation.py` | 评估四联图 + 门控诊断 JSON |
| `comparison.py` | 多模型对比、论文图、跨域对比 |
| `data_visualization.py` | 原始响应可视化（振动时序 / 温度-位移），可直接命令行运行 |
| `logs/replot_history.py` | 从已有 checkpoint 重出训练曲线（不用重训） |
| `logs/regenerate_plots.py` | 从 `method_comparison.json` 重出对比图 |

### 4.9 `scripts/` — 公共数据工具与分阶段实验

- `audit_dataset.py`：扫描数据目录，输出样本清单（含 SHA-256）、唯一结构状态/激励/条件组数量、温度步分布、质量指标键、数组完整性键、标签分布 → `dataset_audit.json/.txt`。**数据冻结的第一步**。
- `make_splits.py`：按 `--group-by`（默认 `structural_state_id`）做组级划分，贪婪算法把大组优先分配到目标比例，输出 ID 级 `split_manifest.json`（含每组样本哈希）。用 `--seed` 固定划分。
- `stage0/check_smoke_batch.py`：阶段 0.1+0.4——batch 契约检查（必需字段/精确 shape/dtype/有限值/标签范围）+ forward/backward 检查 + 防泄漏审计（模型只接收 `MODEL_INPUT_KEYS` 白名单）。
- `stage0/make_smoke_dir.py` + `stage0/overfit_tiny.py`：阶段 0.2——取 8–16 个样本建 tiny-set（全部用于训练），关闭增强/正则/辅助任务只优化材料 BCE，自动验收过拟合（loss 下降 ≥50%、宏 F1 ≥0.90、Exact Match ≥0.75）。
- `stage0/shuffle_labels.py` + `stage0/run_negative_control.py`：阶段 0.3——按固定划分以结构状态组为单位**仅置换 train 的监督束**（同组激励共享 donor；safety_labels + 缩放真值 + 支座真值整体置换），val/test 保留真实标签并核验 SHA-256，训练后与 prevalence 先验比较（AUPRC gap ≤0.10）。
- `stage0/run_stage0.py`：阶段 0 统一执行入口，一键跑完 0.1–0.4 并生成 `stage0_report.json`（任一失败则整体不通过）。
- `stage0/stage0_common.py`：阶段 0 共享工具——输入白名单、监督边界、无固定点置换、HDF5 依赖物化。
- `stage0/stage0_train_utils.py`：阶段 0 诊断训练工具（只优化材料 BCE、prevalence 基线、梯度有限性检查）。

## 5. 数据流全景（从文件到结果）

```
result_*.json（V2 schema，一个文件一个结构状态）
   │
   ├─ scripts/audit_dataset.py ──► dataset_audit.json（数据冻结审计）
   ├─ scripts/make_splits.py ──► split_manifest.json（ID 级划分，含哈希）
   │
   ▼
StructuralDataset（读取 + 归一化 + 标签 + 质量特征 + 元数据）
   │  collate_fn（trainer.py）
   ▼
batch 字典 ──► build_model（model.py）
   │              │
   │              ├─ StaticBranch（多温度位移）
   │              ├─ DynamicBranch（加速度时程）
   │              └─ 融合头（α 逐类融合 / 门控）
   │
   ├─ multitask_loss（trainer.py）──► 反向传播训练
   ├─ validate ──► 指标 + 样本预测 JSON
   ▼
checkpoint .pt（含 history/归一化/划分/阈值/配置）
   │  plot/logs/replot_history.py、regenerate_plots.py
   ▼
logs/ 下的图与 JSON ──► 论文图表
```

## 6. 关键概念速查

| 概念 | 含义 | 在哪定义 |
|---|---|---|
| 材料缩放因子 | 材料刚度保留比例，越小越严重，1.0=无损 | `safety_rules.py` |
| 4 级风险 | safe/warning/risk/danger = 0/1/2/3 | `safety_rules.py` |
| 分层任务 | 材料多标签 + 支座分类/回归 + 区域/全桥分级 | `MultiTaskPredictionHead` |
| group split | 同一结构状态整体进同一集合，防泄漏 | `main.py` |
| split manifest | 样本级划分清单（ID + 哈希），可复现、可校验 | `main.py` / `scripts/make_splits.py` |
| α（alpha） | 逐材料融合权重，`α×static+(1-α)×dynamic` | `ClasswiseLateFusionHead` |
| gate | 门控，决定修正注入比例（不是物理概率） | `ReliabilityGatedFusionHead` |
| warm start | 融合模型用训练好的单分支权重初始化 | `main.py` |
| 层级一致性 | 全桥等级 ≥ 区域最大等级 | `multitask_loss` |
| selection_score | 选最优 checkpoint 的评分（默认 0.7×F1+0.3×AUC） | `main.py` |
| V2 schema | `4.0-temperature-state-json`，多温度位移 + 参考温度振动 | `dataset.py` |
| 质量特征 | `quality_metrics` 的 8 个标量（RMS/峰值/激励误差等） | `dataset.py` |

## 7. 常见调试定位

| 现象 | 先查哪 |
|---|---|
| 训练 loss 不降 / NaN | `trainer.py` 的 `multitask_loss`、`train_epoch`；`DL_config.py` 的 lr/增强 |
| 划分和上次不一样 | 用 `--split-manifest` 固定划分；数据变了会报哈希错误 |
| checkpoint 被覆盖 | 用 `--run-id` 隔离输出目录 |
| 中文图变方块 | `plot/_config.py` 字体列表；`_utils.py` 的 `_configure_chinese_font` |
| 某个指标算不出来 | `compute_full_metrics` 是否有对应任务输入（pretrain 阶段无支座/区域指标） |
| 推理报错 | `predict.py` 的 `PredictContext`；checkpoint 是否用同一套 config |
| 样本预测文件 | `logs/{model_type}/{run}_test_predictions.json` |

## 8. 一句话总结每个文件的角色

- `DL_config.py` = 实验的总开关
- `safety_rules.py` = 真值标签的"宪法"
- `dataset.py` = JSON 世界的翻译官
- `model.py` = 神经网络本身
- `trainer.py` = 训练/评估工具箱
- `main.py` = 总指挥（流程编排）
- `predict.py` = 训练成果的使用者
- `plot/` = 论文图表工厂
- `scripts/` = 实验前的地基工人
