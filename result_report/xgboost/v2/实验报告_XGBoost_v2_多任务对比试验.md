# XGBoost v2 多任务对比试验报告

## 1. 实验用途

本次实验是统一 FEM 垂向 InSAR 观测输入契约下的 XGBoost 多任务基线。它在 `v1` 的两个可比任务基础上，增加了 `material_active`、`settlement_type` 和 `stiffness_reduction` 三个结构状态属性任务，用于判断固定宽度统计特征对不同标签语义的识别能力。

报告中的 `fine_state_family` 与 `v1` 的 `state_family` 是同一细粒度状态分类任务，只是新版训练代码使用了更明确的公开名称。`v2` 与 `v1` 使用相同的数据规模、窗口定义、归一化统计和 XGBoost 主参数，因此可以进行直接的基线对比。

## 2. 数据版本

- 数据类型：静态 FEM 垂向观测源域
- `training_report.json` 记录的数据根目录：`E:\\personal_files\\Dongyaxun\\DL_data\\qianhai7_yearly_temperature_dataset_merged`
- 结构状态数：11,940
- 窗口总数：250,740
- 训练窗口：175,077
- 验证窗口：37,590（由验证集混淆矩阵汇总）
- 测试窗口：38,073（由测试集混淆矩阵汇总）
- 窗口长度：6
- 窗口步长：1
- 虚拟点数范围：50-100
- 观测噪声标准差：0.2 mm
- 缺失概率：0.05
- 划分随机种子：20260924
- 训练批大小：32
- 输入特征数：21
- 外部 InSAR Excel：未参与训练

当前仓库中的 `configs/qianhai7_v12_baseline.yaml` 默认数据根目录是 `E:/working/DL_data/qianhai_test`，与本报告记录的实际 `data_root` 不一致。这说明本次运行很可能使用了命令行数据根目录覆盖；正式归档时应同时保存实际运行命令或确认覆盖参数，否则仅凭配置文件无法完全复现实验。

## 3. 配置文件与模型结构

本次实验对应配置文件：

`configs/qianhai7_v12_baseline.yaml`

模型配置：

```yaml
model:
  name: xgboost
  params:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
```

模型使用 `fixed_features` 将每个长度为 6 的观测窗口压缩为 21 维表格特征，包含垂向观测的均值、标准差、首末值、极值、有效比例，温度/时间统计量以及平均 `x/y` 坐标。训练阶段拟合五个相互独立的 XGBoost 分类器：

1. `fine_state_family`：22 个实际出现的细粒度状态类别；
2. `global_severity`：4 个严重度类别；
3. `material_active`：二分类；
4. `settlement_type`：类别值 `0/1/3` 的三分类；
5. `stiffness_reduction`：二分类。

## 4. 训练耗时

- 特征提取：1,747.38 s
- `fine_state_family` 模型拟合：28.48 s
- `global_severity` 模型拟合：4.82 s
- 五个 XGBoost 模型拟合总计：1,790.59 s
- 进程总耗时：2,930.00 s，约 48.83 min

主要耗时仍来自 HDF5 数据读取、虚拟观测处理和 21 维特征提取，而不是树模型拟合。报告只单独记录了细粒度状态和严重度两个模型的拟合时间，另外三个属性头的时间包含在 `fit_total_sec` 中但没有单独拆分；由总时间估算，三者及相关开销约为 9.90 s。

相较 `v1`，特征提取增加 30.46 s，模型拟合增加 43.16 s，进程总耗时增加 370.45 s，主要原因是 `v2` 额外拟合了三个属性分类器。

## 5. 权重和归一化文件

- 细粒度状态权重：`models/xgboost_state.json`
- 严重度权重：`models/xgboost_severity.json`
- 归一化统计：`normalization_stats.json`
- 类别映射：`training_report.json` 的 `class_mappings` 字段

`v1` 与 `v2` 的 `normalization_stats.json` 内容完全一致，SHA-256 均为 `6BDC0A32332A0C52F15EAE0F2758B998683B455BD744A60D09E6EBBBDA180CC0`。

当前 `v2/models/` 目录只归档了 `state` 和 `severity` 两个模型文件。`material_active`、`settlement_type`、`stiffness_reduction` 的指标已写入报告，但对应的三个模型权重路径没有写入报告，也没有出现在该目录中。因此，若要独立部署或复核新增任务，建议重新保存这三个模型并在报告中记录路径。

## 6. 验证集指标

| 任务 | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Cross Entropy |
|---|---:|---:|---:|---:|---:|---:|
| `fine_state_family` | 0.5369 | 0.2408 | 0.2029 | 0.1950 | NaN | 1.1935 |
| `global_severity` | 0.4872 | 0.4353 | 0.4705 | 0.4292 | 0.7558 | 1.0429 |
| `material_active` | 0.6957 | 0.6738 | 0.6200 | 0.6213 | 0.7307 | 0.5755 |
| `settlement_type` | 0.9788 | 0.9365 | 0.9097 | 0.9220 | 0.9969 | 0.0501 |
| `stiffness_reduction` | 0.7668 | 0.7743 | 0.7471 | 0.7520 | 0.7986 | 0.5416 |

验证集五任务 `combined_loss` 为 3.4037。这个值不能与 `v1` 的 2.2358 直接比较，因为 `v1` 只汇总了两个任务，而 `v2` 汇总了五个任务。

## 7. 测试集指标

| 任务 | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Cross Entropy |
|---|---:|---:|---:|---:|---:|---:|
| `fine_state_family` | 0.5481 | 0.2298 | 0.1953 | 0.1849 | 0.9216 | 1.2228 |
| `global_severity` | 0.5000 | 0.4612 | 0.4902 | 0.4459 | 0.7651 | 1.0353 |
| `material_active` | 0.7023 | 0.6651 | 0.6210 | 0.6255 | 0.7215 | 0.5705 |
| `settlement_type` | 0.9744 | 0.9155 | 0.8886 | 0.9008 | 0.9960 | 0.0573 |
| `stiffness_reduction` | 0.7699 | 0.7778 | 0.7487 | 0.7541 | 0.8061 | 0.5274 |

测试集五任务 `combined_loss` 为 3.4133。混淆矩阵和原始指标均保存在 `training_report.json` 的 `metrics.test` 字段中。

## 8. 与 v1 的可比结果

以下只比较同一任务：`v1.state_family` 对应 `v2.fine_state_family`，以及 `global_severity`。

| 数据集 | 任务 | Accuracy 变化 | Macro F1 变化 | ROC-AUC 变化 | Cross Entropy 变化 |
|---|---|---:|---:|---:|---:|
| 验证集 | `fine_state_family` | +0.0008 | -0.0016 | 不适用 | +0.0002 |
| 测试集 | `fine_state_family` | -0.0006 | +0.0021 | -0.0013 | -0.0001 |
| 验证集 | `global_severity` | -0.0008 | -0.0022 | +0.0002 | +0.0005 |
| 测试集 | `global_severity` | +0.0011 | +0.0009 | -0.0001 | +0.0011 |

结论是：在两个旧任务上，`v2` 与 `v1` 的性能基本持平，没有形成实质性提升；测试集严重度 Macro F1 仅提升约 0.0009，属于很小的波动。新增任务中，`settlement_type` 最容易识别，`stiffness_reduction` 次之，`material_active` 仍受到类别不均衡和特征表达能力限制。

## 9. 类别不均衡与错误模式

测试集按真实类别统计的召回率如下：

- `global_severity`：类别 `0/1/2/3` 分别为 0.784、0.159、0.326、0.692。类别 1 和 2 的识别明显弱于类别 0 和 3，是 Macro F1 偏低的主要原因。
- `material_active`：类别 `0/1` 分别为 0.367、0.875，模型明显偏向样本更多的类别 1。
- `stiffness_reduction`：类别 `0/1` 分别为 0.888、0.609，正类漏检仍较多。
- `settlement_type`：类别 `0/1/3` 分别为 1.000、0.701、0.965，整体很好，但少数类别 1 的召回率仍显著低于另外两类。
- `fine_state_family`：虽然测试 Accuracy 为 0.5481、ROC-AUC 为 0.9216，但 Macro F1 只有 0.1849，说明概率排序能力与离散决策后的少数类识别之间存在明显差距，不能只用 Accuracy 或 ROC-AUC 判断该任务效果。

细粒度状态验证集的 ROC-AUC 记录为 NaN，原因是当前 ROC-AUC 计算条件不满足多类别有效计算要求；这不应解释为模型“没有排序能力”，而应结合 Macro F1、类别召回和混淆矩阵评价。

## 10. 类别映射

### `fine_state_family`（报告中的 `class_mappings.state_family`）

```text
[0, 1, 3, 4, 5, 7, 8, 10, 11, 22, 23, 24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39]
```

### `global_severity`

```text
[0, 1, 2, 3]
```

新增任务的实际类别值记录在各自指标对象的 `class_values` 字段中：`material_active=[0,1]`，`settlement_type=[0,1,3]`，`stiffness_reduction=[0,1]`。预测解释时必须保留全局类别值，不能把局部 XGBoost 类别索引直接当作全局标签。

## 11. 结果判断与后续建议

1. `v2` 已形成可用的五任务 XGBoost 基线。`settlement_type` 测试 Macro F1 为 0.9008，`stiffness_reduction` 为 0.7541，`material_active` 为 0.6255，说明固定统计特征对属性级任务的有效性差异明显。
2. `fine_state_family` 和 `global_severity` 与 `v1` 基本持平，当前增加任务没有明显改善主任务，不能据此宣称模型能力提升。
3. 后续神经网络对比应固定 `v2` 的五任务标签语义、同一数据划分、归一化统计和测试集；主结果至少同时报告 Accuracy、Macro Precision、Macro Recall、Macro F1、ROC-AUC、Cross Entropy 和混淆矩阵。
4. 针对 `global_severity`、`material_active` 和 `stiffness_reduction`，应增加类别权重、分层采样或阈值校准实验，并报告少数类召回率，避免总体 Accuracy 掩盖漏检。
5. 应修正 XGBoost 训练归档：保存三个新增属性模型的权重、记录每个任务的拟合耗时，并把实际 `--data-root` 或完整运行命令写入报告。

## 12. 原始结果文件

- 训练报告：`training_report.json`
- 归一化统计：`normalization_stats.json`
- 模型目录：`models/`
- 混淆矩阵、类别映射及所有原始指标：`training_report.json` 的 `metrics`、`class_mappings` 字段
