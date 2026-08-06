# DL_model — 多任务结构安全评估网络

基于深度学习的桥梁多工况安全评估系统。采用**双分支架构**（静态分支 + 动态分支），通过 **ClasswiseLateFusionHead** 为每种材料学习独立的 static/dynamic 融合权重，同时输出**材料损伤、支座沉降、区域风险、全局安全**四个层次的安全评估。

## 项目结构

```
DL_model/
├── __init__.py              # 包初始化
├── DL_config.py             # 全局配置（超参数、阶段控制、增强参数）
├── model.py                 # SafeBN / PositionEncoding / PositionFiLM / AttentionPool /
                             #   PredictionHead / MultiTaskPredictionHead /
                             #   StaticBranch / DynamicBranch /
                             #   ClasswiseLateFusionHead / ReliabilityGatedFusionHead /
                             #   StaticOnlyModel / DynamicOnlyModel / DualBranchFusion /
                             #   build_model
├── dataset.py               # 数据集加载、归一化、分组标签、安全标签构建
├── trainer.py               # EMAModel / 数据增强 / 模态暂失 /
                             #   multitask_loss / train_epoch / validate / 评估指标
├── main.py                  # 主入口 + 分组划分 + 训练调度 + 可视化
├── predict.py               # 批量预测脚本
├── quick_predict.py         # 快速单文件预测脚本
├── safety_rules.py          # 工程安全规则（阈值、区域定义、标签构建）
├── checkpoints/             # 模型检查点
└── logs/                    # 训练日志与可视化图表
```

## 核心思路

### 问题定义

- 共 43 种材料，其中 **6 种候选材料**（ID: 8, 9, 19, 20, 24, 41）可能发生损伤
- 损伤定义为材料缩放因子 < 1.0（二值化标签）
- 附加任务：支座沉降检测、区域风险分级、全局安全评估
- **公平对比**：所有模型均从零训练，不使用 warm start

### 多任务输出

模型除材料损伤分类外，同时输出：

| 任务 | 输出头 | 监督信号 |
|------|--------|----------|
| 材料损伤分类 | `material_classifier` | BCE（损伤/健康） |
| 材料刚度回归 | `material_regressor` | MSE（缩放因子） |
| 支座沉降分类 | `support_classifier` | BCE（异常/正常） |
| 支座沉降回归 | `support_regressor` | MSE（沉降值） |
| 区域风险分级 | `region_classifier` | CrossEntropy（4 级） |
| 全局安全状态 | `global_classifier` | CrossEntropy（4 级） |

### 训练顺序

`static_only → dynamic_only → fusion`，三个模型分别独立训练，最终在测试集上公平对比。

### 双分支架构

```
静态分支 (StaticBranch)                    动态分支 (DynamicBranch)
  │                                            │
位移(4,3,3) + 应变(4,3,3)                   加速度时程(1000,5,3)
  │                                            │
Fourier PositionEncoding + FiLM              Fourier PositionEncoding + FiLM
  │                                            │
MLP + AttentionPool                          Conv1d + MLP + Dropout(0.35)
  │                                            │
64 维特征 ───────────────┐            ┌────── 64 维特征
                         │            │
                         └── ClasswiseLateFusionHead ──┘
                                    │
                    fused = alpha_i * static + (1 - alpha_i) * dynamic
                                    │
                    每个材料类别有独立的可学习 alpha_i
                         ┌──────────┼──────────┐
                         │          │          │
                  material_head  safety_head  fused_emb
                  (分类+回归)    (支座+区域+全局)
```

#### 融合策略

采用 **ClasswiseLateFusionHead**：每个材料类别 i 有一个可学习权重 α_i：

```
logits_i = α_i × static_logits_i + (1 - α_i) × dynamic_logits_i
```

- α_i 越大 → 该材料更信任静态分支
- α_i 越小 → 该材料更信任动态分支
- α 初始化：大部分材料 0.90（偏向 static），材料 41 为 0.20（偏向 dynamic）
- α 有 L2 正则约束（weight=0.02）防止偏离初始值过远

#### 安全评估来源

`safety_source = "static"`：区域风险和全局安全评估基于静态分支特征（因 static 对整体刚度的敏感性更高）。

#### 静态分支 (StaticBranch)

- 3 个静力测点
- 每节点 24 维（4 步 × 3 通道 × 2 种响应）→ Fourier PositionEncoding(16 维) + PositionFiLM
- 3 层 MLP + SafeBN + GELU + Dropout(0.15) → 每节点 64 维
- AttentionPool → 64 维 + LayerNorm

#### 动态分支 (DynamicBranch)

- 5 个动力测点
- Fourier PositionEncoding + sensor-wise PositionFiLM
- 3 层 Conv1d (kernel=9/7/5) + SafeBN + GELU + MaxPool1d → 128 维
- AdaptiveMaxPool1d + MLP head + Dropout(0.35) + LayerNorm → 64 维

#### 融合头 (ClasswiseLateFusionHead)

1. 两个分支各自经过 `PredictionHead` 得到分类 logits 和回归值
2. `alpha = sigmoid(alpha_logit)` — 每材料独立
3. `logits = alpha * static_logits + (1 - alpha) * dynamic_logits`
4. `emb_proj` 融合两个分支的嵌入特征

### 训练配置

**单分支训练（static_only / dynamic_only）：**
- 单阶段训练，300 epochs
- static_only min_epochs=100，dynamic_only min_epochs=150
- 使用 EMA + Label Smoothing

**融合训练：支持两种模式**

#### Legacy 三阶段训练（`fusion_use_legacy_schedule=True`）

```
Stage 1 ─ 联合预训练 (220 epochs, lr=5e-4)
├── 两个分支 + 各自分类头同时训练，fusion head 不参与梯度
├── Loss = BCE(s) + BCE(d) + aux_weight[MSE(s)+MSE(d)] + contrastive_weight × Contrastive
├── Contrastive Loss: 同一样本 static/dynamic 特征拉近，不同样本推开
└── 目的：两个分支从零学到互补特征

Stage 2 ─ 融合头训练 (160 epochs, lr=1e-4)
├── 冻结 static_branch + dynamic_branch 全部参数
├── 仅训练 fusion head + safety head
└── 目的：分支特征稳定后学习融合策略

Stage 3 ─ 端到端微调 (300 epochs, lr=1e-5)
├── 解冻全部参数 + EMA(0.999)
├── 极小学习率全局微调
└── 目的：消除阶段接口偏差
```

#### 默认模式（`fusion_use_legacy_schedule=False`）

端到端训练，不区分阶段，直接训练完整模型。

### 核心训练策略

#### 1. 多任务损失

```
L = L_mat_cls
  + aux_weight × L_mat_reg
  + support_cls_weight × L_support_cls
  + support_reg_weight × L_support_reg
  + region_weight × L_region
  + global_state_weight × L_global
  + rule_consistency_weight × L_consistency
```

规则一致性损失：约束 `global_level ≥ max(region_levels)`，确保全局评估不低于最差区域。

#### 2. 数据增强（二级体系）

**动态增强**（轻量——真实振动数据噪声低）：

| 参数 | generic | dynamic_only | fusion |
|------|---------|-------------|--------|
| prob | 0.30 | 0.25 | 0.30 |
| noise σ | 0.015 | 0.012 | 0.012 |
| amp scale | ±5% | ±4% | ±4% |
| time crop | 96% | 97% | 96% |
| time shift | 2% | 2% | 2% |
| sensor dropout | 3% | 3% | 3% |
| channel dropout | 1.5% | 1.5% | 1.5% |

**静态增强**（较重——模拟位移计/应变片测量误差）：

| 参数 | generic | static_only | fusion |
|------|---------|-------------|--------|
| prob | 0.55 | 0.50 | 0.45 |
| noise σ | 0.03 | 0.03 | 0.025 |
| scale | ±8% | ±8% | ±6% |
| sensor drop | 6% | 6% | 5% |

增强生效链路：`train_epoch` → 检查 model_type → `_aug_value` 用模型专属值覆写。

#### 3. 模态暂失

`fusion_modality_dropout=0.0`：默认关闭。设为 >0 时随机清零一个分支，同时跳过该分支的辅助监督损失。

#### 4. 其他技术

| 技术 | 参数 | 说明 |
|------|------|------|
| EMA | 0.999 | 全程启用 |
| Label Smoothing | 0.05 | {0,1} → {0.025, 0.975} |
| Threshold Tuning | True | 验证时搜索每材料最优阈值 |
| Recall Constraints | {3:0.70, 5:0.70} | 材料索引 3/5 召回率≥70% |
| Selection Metric | `f1_auc` | 0.7×F1 + 0.3×AUC |
| Gradient Clipping | 5.0 | 防止梯度爆炸 |

### 损失函数总览

| 场景 | 损失公式 |
|------|----------|
| 单分支 | `BCE + aux_weight × MSE` |
| Legacy S1 | `BCE(s) + BCE(d) + aux_weight[MSE(s)+MSE(d)] + contrastive_weight × InfoNCE` |
| 融合（默认） | `BCE(f) + aux_weight × MSE + support + region + global + consistency` (fusion_branch_weight=0 时) |
| fusion_branch_weight>0 | 上述 + branch_weight × [BCE(s)+BCE(d)] |
| Legacy S2/S3 | 同 branch_weight>0 |

## 数据格式

`data_dir/result_*.json`：

| 字段 | 形状 | 说明 |
|------|------|------|
| `disp` | (steps, nodes, 3) | 静力位移 |
| `strain` | (steps, nodes, 3) | 静力应变 |
| `ace` | (time_steps, nodes, 3) | 加速度时程 |
| `material_scaling_factors` | (n_materials,) | <1 为损伤 |
| `material_ids` | (n_materials,) | 有序 ID 列表 |
| `node_coords` | dict | 节点号 → [x, y, z] |
| `node_maps` | dict | `disp`/`strain`/`acc` 响应张量列索引到节点号的映射 |
| `support_settlement` | dict | 支座沉降数据 |
| `safety_labels` | dict | 材料、支座、区域和全局安全监督标签 |
| `region_definitions` | dict | 区域和材料/支座归属 |

测点由 `node_maps` 自动读取，无法读取时回退到配置默认值：
- STATIC: [43, 65, 87, 153, 175, 197]
- DYNAMIC: [23, 67, 133, 155, 177, 645]

## 工程安全规则

### 阈值

| 级别 | 材料缩放因子 | 支座沉降 |
|------|:----------:|:------:|
| safe | > 0.90 | < 2.0 mm |
| warning | ≤ 0.90 | ≥ 2.0 mm |
| risk | ≤ 0.80 | ≥ 5.0 mm |
| danger | ≤ 0.70 | ≥ 7.0 mm |

### 区域划分（7 区域）

| 区域 | 材料 | 支座 |
|------|------|------|
| left_support1_zone | - | 221 |
| left_support2_zone | - | 513 |
| right_support2_zone | - | 395 |
| right_zone | 9, 20 | - |
| mid_zone | 24, 41 | - |
| left_zone | 8, 19 | - |
| right_support1_zone | - | 687 |

## 使用方法

```bash
# 训练三种模型 + 对比（默认）
python _lab_platform/DL_model/main.py --model all

# 仅训练融合模型
python _lab_platform/DL_model/main.py --model fusion

# 仅训练单分支
python _lab_platform/DL_model/main.py --model static_only

# 测试
python _lab_platform/DL_model/main.py --mode test

# 推理（自动使用 EMA 权重）
python _lab_platform/DL_model/main.py --mode infer --input path/to/result_xxx.json

# 批量预测
python _lab_platform/DL_model/predict.py path/to/data.json

# 快速单文件预测（编辑 quick_predict.py 中的 file_path）
python _lab_platform/DL_model/quick_predict.py
```

## 配置项

### 路径与模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_root` | `E:/working/DL_data/data` | 数据根目录 |
| `corrected_data_dir` | `.../corrected_model_dataset` | 修正模型数据集 |
| `reference_data_dir` | `.../reference_model_dataset` | 参考模型数据集 |
| `data_dir` | `corrected_data_dir` | 当前使用的数据目录 |
| `pos_dim` | 16 | 位置编码维度 |
| `static_dim` / `dynamic_dim` | 128 | 分支隐藏层 |
| `fusion_dim` | 128 | 融合层 |
| `n_materials` | 6 | 输出类别数（自动从 CANDIDATE_MATERIAL_IDS 读取） |
| `n_supports` | 4 | 支座数量 |
| `n_regions` | 7 | 区域数量 |
| `n_safety_levels` | 4 | 安全等级数 |

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批大小 |
| `lr` | 5e-4 | 单分支学习率 |
| `weight_decay` | 2e-4 | 权重衰减 |
| `n_epochs` | 300 | 单分支训练轮数 |
| `early_stop` | 40 | 早停 patience |
| `min_epochs` | 100 | static_only 最小轮次 |
| `dynamic_only_min_epochs` | 150 | dynamic_only 最小轮次 |

### 数据划分

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_ratio` / `val_ratio` | 0.70 / 0.15 | 数据划分比例（测试集=0.15） |
| `split_strategy` | `group` | 按组划分 |
| `group_by` | `file_block` | 按文件块分组划分，避免数据泄漏 |
| `seed` | 42 | 随机种子 |
| `pos_weight` | 5.0 | BCE 正类权重上限 |

### 多任务权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `aux_weight` | 0.2 | 材料回归辅助损失权重 |
| `support_cls_weight` | 0.8 | 支座分类损失权重 |
| `support_reg_weight` | 0.4 | 支座回归损失权重 |
| `region_weight` | 0.5 | 区域风险损失权重 |
| `global_state_weight` | 0.5 | 全局安全损失权重 |
| `rule_consistency_weight` | 0.05 | 规则一致性损失权重 |

### 融合训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fusion_anchor` | `static` | 融合锚点（static 或 dynamic） |
| `fusion_safety_source` | `static` | 安全评估特征来源 |
| `fusion_alpha_init` | `[0.90, 0.90, 0.90, 0.90, 0.20, 0.90]` | 每材料 α 初始值（材料41=0.20偏向dynamic） |
| `fusion_alpha_l2_weight` | 0.02 | α 偏离初始值的 L2 惩罚 |
| `fusion_branch_weight` | 0.0 | 分支辅助监督权重（关闭） |
| `fusion_gate_weight` | 0.0 | gate 正则权重（关闭） |
| `fusion_gate_bias` | -2.0 | gate 初始偏置 |
| `fusion_per_class_gate_bias` | `[-2.5, -2.5, -2.5, -2.5, -0.5, -2.5]` | 每材料 gate 偏置 |
| `fusion_strong_dynamic_classes` | `[0,1,2,3,5]` | 动态分支强类 |
| `fusion_per_class_gate_weight` | 0.0 | 每类 gate 正则（关闭） |
| `fusion_margin_weight` | 0.0 | dynamic margin 权重（关闭） |
| `fusion_static_margin_weight` | 0.0 | static margin 权重（关闭） |
| `fusion_modality_dropout` | 0.0 | 模态暂失概率（关闭） |
| `fusion_freeze_dynamic_epochs` | 0 | 冻结动态分支 epoch 数（关闭） |
| `fusion_use_legacy_schedule` | False | 是否使用三阶段训练 |
| `fusion_warm_start` | True | 从预训练单分支 warm start |
| `fusion_train_alpha_only` | True | warm start 后仅训练 α（冻结分支和 head） |

### Legacy 三阶段训练参数（`fusion_use_legacy_schedule=True` 时启用）

| 参数 | S1 | S2 | S3 |
|------|-----|-----|-----|
| epochs | 220 | 160 | 300 |
| lr | 5e-4 | 1e-4 | 1e-5 |
| 冻结 | 无 | static/dynamic 分支 | 无 |
| EMA | ✗ | ✗ | ✓ |
| 对比 loss 权重 | 0.20 | — | — |
| 温度 | 0.07 | — | — |

### 增强参数

| 参数 | 通用 | static_only | dynamic_only | fusion |
|------|------|-------------|-------------|--------|
| **动态** prob | 0.30 | — | 0.25 | 0.30 |
| 动态 noise σ | 0.015 | — | 0.012 | 0.012 |
| 动态 amp | ±5% | — | ±4% | ±4% |
| 动态 time crop | 96% | — | 97% | 96% |
| 动态 time shift | 2% | — | 2% | 2% |
| 动态 sensor drop | 3% | — | 3% | 3% |
| 动态 channel drop | 1.5% | — | 1.5% | 1.5% |
| **静态** prob | 0.55 | 0.50 | — | 0.45 |
| 静态 noise σ | 0.03 | 0.03 | — | 0.025 |
| 静态 scale | ±8% | ±8% | — | ±6% |
| 静态 sensor drop | 6% | 6% | — | 5% |

### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `label_smoothing` | 0.05 | 标签平滑 |
| `ema_decay` | 0.999 | EMA 衰减率 |
| `threshold_tuning` | True | 启用阈值搜索 |
| `threshold_recall_constraints` | `{3: 0.70, 5: 0.70}` | 召回率约束 |
| `selection_metric` | `f1_auc` | 模型选择指标 |
| `device` | cuda / cpu | 自动检测 |

## 输出

- **checkpoints/`{model_type}/{run_name}_best.pt`**：模型权重 + EMA + 优化器 + 历史 + 归一化 + 划分索引 + 阈值 + 配置
- **logs/`{model_type}/{run_name}_history.png`**：6 子图训练曲线（Loss, F1, Accuracy, Precision/Recall, AUC, LR）
- **logs/`{model_type}/{run_name}_evaluation.png`**：混淆矩阵 + 各类别指标 + ROC + 概率分布
- **logs/`{model_type}/{run_name}_fusion_diagnostics.json`**：融合诊断（α 值、gate 均值、分支分歧度）
- **logs/method_comparison.json / .png**：三种模型测试指标对比
- **logs/paper_*.png**：论文级对比图（核心指标、各类别F1、融合权重、辅助任务）

## 预测输出格式

```json
{
  "material_ids": [8, 9, 19, 20, 24, 41],
  "material_probs": [0.02, ...],
  "material_preds": [0, ...],
  "material_scaling_pred": [0.98, ...],
  "support_nodes": [...],
  "support_probs": [...],
  "support_preds": [...],
  "support_disp_mm": [...],
  "region_names": ["left_support1_zone", ...],
  "region_levels": [0, ...],
  "global_level": 0
}
```

## 依赖

- PyTorch ≥ 2.0
- NumPy / scikit-learn / tqdm
- matplotlib（可选，用于绘图）
