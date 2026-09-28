# XGBoost 首次对比试验记录

## 1. 实验用途

本次实验作为统一 FEM 垂向 InSAR 观测输入契约下的第一个传统机器学习基线。后续 LSTM、ResNet1D、Transformer 和时空图模型应复用相同的数据划分、归一化统计、窗口定义和标签语义。

## 2. 数据版本

- 数据类型：静态 FEM 垂向观测源域
- 数据根目录：`E:\\personal_files\\Dongyaxun\\DL_data\\qianhai7_yearly_temperature_dataset_merged`
- 结构状态数：11,940
- 窗口总数：250,740
- 训练窗口：175,077
- 验证窗口：37,590
- 测试窗口：38,073
- 窗口长度：6
- 窗口步长：1
- 虚拟点数范围：50-100
- 观测噪声标准差：0.2 mm
- 缺失概率：0.05
- 划分随机种子：20260924
- 训练批大小：32
- 输入特征数：21

数据集级 `_run_manifest.json` 未提供；本次运行使用 HDF5 内嵌 manifest。正式论文归档时建议补充数据集版本号、Git commit 或生成日期。

## 3. 配置文件

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

输入由窗口垂向观测统计量、温度/时间统计量和平均 `x/y` 坐标组成。外部 InSAR Excel 未参与训练。

## 4. 训练耗时

- 特征提取：1,716.93 s
- `state_family` 模型拟合：26.35 s
- `global_severity` 模型拟合：4.15 s
- XGBoost 拟合总计：1,747.42 s
- 进程总耗时：2,559.55 s，约 42.66 min

主要耗时来自 HDF5 数据读取、虚拟观测处理和 21 维固定特征提取，而不是决策树拟合。

## 5. 权重和归一化文件

- 状态分类权重：`models/xgboost_state.json`
- 严重度分类权重：`models/xgboost_severity.json`
- 归一化统计：`normalization_stats.json`
- 原始机器上的报告中记录的 `runs\\models\\...` 是相对路径；本目录中的 `models/` 文件才是本次结果归档对应的实际权重。

## 6. 验证集指标

### 6.1 `state_family`

- Accuracy：0.5361
- Macro Precision：0.2494
- Macro Recall：0.2035
- Macro F1：0.1966
- ROC-AUC：NaN（多类别真实标签分布不满足当前 ROC-AUC 计算条件）
- Cross Entropy：1.1933

### 6.2 `global_severity`

- Accuracy：0.4879
- Macro Precision：0.4367
- Macro Recall：0.4717
- Macro F1：0.4314
- ROC-AUC：0.7556
- Cross Entropy：1.0425

验证集混淆矩阵和原始指标 JSON 保存在：`training_report.json` 的 `metrics.validation` 字段中。

## 7. 测试集指标

### 7.1 `state_family`

- Accuracy：0.5487
- Macro Precision：0.2295
- Macro Recall：0.1926
- Macro F1：0.1828
- ROC-AUC：0.9229
- Cross Entropy：1.2229

### 7.2 `global_severity`

- Accuracy：0.4988
- Macro Precision：0.4601
- Macro Recall：0.4888
- Macro F1：0.4450
- ROC-AUC：0.7652
- Cross Entropy：1.0342

测试集混淆矩阵和原始指标 JSON 保存在：`training_report.json` 的 `metrics.test` 字段中。

## 8. 类别映射

### `state_family`

```text
[0, 1, 3, 4, 5, 7, 8, 10, 11, 22, 23, 24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39]
```

### `global_severity`

```text
[0, 1, 2, 3]
```

这些是 XGBoost 当前训练子集使用的全局类别 ID。模型内部会将它们映射为连续局部类别，预测解释时必须保留上述映射。

## 9. 结果判断

本次训练可以作为第一次完整数据集 XGBoost 对比基线。`global_severity` 的测试 Macro F1 为 0.4450、ROC-AUC 为 0.7652，具备后续模型比较价值。

`state_family` 的 Accuracy 为 0.5487，但 Macro F1 仅为 0.1828，说明类别不均衡和小类别识别不足明显。正式比较时不能只报告 Accuracy 或 ROC-AUC，应同时报告 Macro Precision、Macro Recall、Macro F1 和混淆矩阵。

后续模型必须固定：数据根目录版本、结构状态划分、窗口长度、虚拟点实现、归一化文件、标签字段和测试集，才能进行公平对比。
