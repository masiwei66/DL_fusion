# DL_fusion — 桥梁多模态安全状态评估

基于深度学习的**桥梁多工况安全状态评估**系统。模型同时读取**多温度静态位移**（静力响应）与**参考温度下振动加速度时程**（动力响应），通过**双分支网络**进行特征提取与融合，输出**材料损伤、支座沉降、区域风险、全桥安全**四个层次的安全评估结果。

项目当前使用 **V2 温度状态数据协议**（schema `4.0-temperature-state-json`），默认数据目录为 `E:/working/DL_data/data_new`。

---

## 目录

- [研究目标](#研究目标)
- [项目结构](#项目结构)
- [核心方法](#核心方法)
- [数据格式](#数据格式)
- [工程安全规则](#工程安全规则)
- [安装与依赖](#安装与依赖)
- [开发与训练环境](#开发与训练环境)
- [使用方式](#使用方式)
- [绘图功能](#绘图功能)
- [输出说明](#输出说明)
- [当前状态与已知限制](#当前状态与已知限制)
- [后续工作安排](#后续工作安排)
- [相关文档](#相关文档)

---

## 研究目标

桥梁结构在服役中可能出现**材料刚度削弱**与**支座沉降**等损伤。本项目的目标不是对损伤做精确反演，而是基于可观测的多模态响应，对桥梁安全状态做**分层辅助评估**：

| 任务层次 | 内容 | 监督形式 |
|---|---|---|
| 材料损伤 | 6 种候选材料是否存在削弱 | 多标签二分类 |
| 支座沉降 | 4 个支座是否发生沉降 + 幅值 | 二分类 + 回归 |
| 区域风险 | 7 个结构区域的 4 级风险 | 4 级分类 |
| 全桥安全 | 整体安全状态（取区域最大值） | 4 级分类 |

研究口径与实验路线详见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

---

## 项目结构

```
DL_fusion/
├── DL_model/                    # 核心训练/推理代码
│   ├── DL_config.py             # 全局配置（数据路径、模型、训练、增强超参数）
│   ├── safety_rules.py          # 工程安全规则（阈值、区域定义、标签构建）
│   ├── model.py                 # 网络结构（StaticBranch / DynamicBranch / 融合头）
│   ├── dataset.py               # V2 数据集加载、归一化、分组、标签构建
│   ├── trainer.py               # EMA、增强、多任务损失、train/validate、指标
│   ├── main.py                  # 训练/测试/推理主入口 + 分组划分 + 结果对比绘图
│   ├── predict.py               # 批量预测脚本（输出 JSON + 预测图）
│   ├── quick_predict.py         # 快速单文件推理
│   ├── checkpoints/             # 模型检查点
│   └── logs/                    # 训练日志、指标 JSON、对比图
├── plot/                        # 绘图包（按功能分模块）
│   ├── prediction.py            # 单样本预测结果图
│   ├── training.py              # 训练历史曲线
│   ├── evaluation.py            # 评估四联图 + 门控诊断导出
│   ├── comparison.py            # 多模型性能对比图
│   ├── data_visualization.py    # 原始响应可视化（振动时序 / 温度-位移）
│   ├── _config.py               # 全局绘图配置（配色、中文字体、名称映射）
│   ├── _utils.py                # 通用绘图辅助函数
│   └── logs/                    # 生成的对比图与回归脚本
├── DATASET_AUDIT.md             # 数据集结构与质量审计报告
├── RESEARCH_PLAN.md             # 研究路线与实验计划
└── README.md
```

---

## 核心方法

### 网络架构

整体为**双分支 + 融合**结构：

```
静态分支 StaticBranch                    动态分支 DynamicBranch
  │ 多温度位移 disp [6,6,3]                 │ 加速度时程 ace [4000,6,3]
  │ 测点坐标 → 位置编码                      │ 测点坐标 → 位置编码
  │ MLP + AttentionPool                     │ 1D CNN + 时序池化 + MLP
  └──────────────┐               ┌──────────┘
                 └── 融合头 ──────┘
                      │
       class-wise alpha 融合 (每个材料独立权重)
                      │
        ┌─────────────┼──────────────┐
    材料损伤     支座沉降          区域/全桥风险
```

- **静态分支**：对每个测点的多温度位移序列做位置感知的 MLP 编码，再用注意力池化得到节点级特征。V2 协议不再把旧版应变作为主输入。
- **动态分支**：对加速度时程做多层 1D 卷积 + 时序最大池化，位置信息通过 FiLM 注入。
- **温度条件注入**：归一化温度通过 FiLM 式特征调制注入两个分支，不增加额外 checkpoint 参数。
- **位置编码**：`PositionEncoding` 将 3D 测点坐标先中心化/尺度归一化，再叠加傅里叶特征，避免把原始坐标尺度直接带入响应通道。

### 融合策略

- **ClasswiseLateFusionHead**（当前双分支融合模型使用）：每个材料类别学习独立的可解释权重 αᵢ：

  ```
  logits_i = αᵢ × static_logits_i + (1 − αᵢ) × dynamic_logits_i
  ```

  αᵢ 接近 1 表示该材料更信任静态分支，接近 0 表示更信任动态分支，便于分析和报告。

- **ReliabilityGatedFusionHead**（`StaticAnchoredFusionHead` 别名）：基于分支置信度与跨模态交互的**逐材料门控**融合头，支持静态/动态锚点，作为替代融合方案保留在代码中。

- 安全评估（支座/区域/全桥）特征来源由 `fusion_safety_source` 控制（默认 `static`）。

### 训练方式

按 `static_only → dynamic_only → fusion` 的顺序独立训练，最终在测试集上公平对比。

- **多任务损失**：材料分类/回归 + 支座分类/回归 + 区域风险 + 全桥风险 + 规则一致性损失。规则一致性约束 `全局等级 ≥ 区域最大等级`。
- **数据增强**：动态增强（轻量，噪声/幅值/裁剪/平移/测点暂失）与静态增强（较重，模拟位移计测量误差）分级配置，见 `DL_config.py`。
- **其他技术**：EMA 权重、Label Smoothing、验证集阈值搜索（含召回率约束）、梯度裁剪。
- **融合 warm start**：从训练好的单分支 warm start，然后只训练融合头/α 再端到端微调（可由配置开关控制）。

---

## 数据格式

当前使用 **V2 温度状态 JSON 协议**，一个 JSON 文件对应一个结构状态，位于数据目录下的 `result_*.json`。

| 字段 | 形状 | 说明 |
|---|---|---|
| `responses.static.disp` | `[6, 6, 3]` float32 | 6 温度步 × 6 静态测点 × (UX,UY,UZ) 位移，mm |
| `environment.temperature.temperature_steps_C` | `[6]` | 与静态位移一一对应的温度步 |
| `responses.dynamic.ace` | `[4000, 6, 3]` float32 | 参考温度下 6 测点三向加速度时程，mm/s² |
| `responses.dynamic.time_s` | `[4000]` float64 | 时间轴（20 s，200 Hz） |
| `responses.dynamic.force_N` | `[4000, 1]` | 动力激励力时程 |
| `safety_labels` | object | 材料/支座/区域/全桥的监督标签 |
| `node_coords` / `node_maps` | object | 节点坐标与响应张量列索引映射 |
| `material_scaling_factors` | `[n]` | 材料刚度缩放真值（**仅作监督/审计，不作为输入**） |
| `support_settlement` | object | 支座沉降节点与幅值 |
| `region_definitions` | object | 区域与材料/支座归属 |
| `quality_metrics` / `array_integrity` | object | 质量指标与数组 SHA-256 完整性 |

**关键节点集合：**

| 集合 | 节点 ID |
|---|---|
| 候选材料 | `[8, 9, 19, 20, 24, 41]` |
| 静态测点 | `[43, 65, 87, 153, 175, 197]` |
| 动态测点 | `[23, 67, 133, 155, 177, 645]` |
| 支座 | `[221, 513, 395, 687]` |

**数据口径**（与 `dataset_roles` 声明一致）：条件输入为温度步，响应输入为静态位移与动态加速度，监督目标为材料/支座/区域/全桥安全标签。`material_scaling_factors`、`support_settlement` 等结构状态真值**只能用于生成标签、辅助回归或审计，不能直接送入分类模型**，否则会造成目标泄漏。

更详细的字段说明与质量审计结论见 [DATASET_AUDIT.md](DATASET_AUDIT.md)。

---

## 工程安全规则

安全规则由 `safety_rules.py` 定义，将材料刚度损失与支座沉降映射为四级风险。

**风险阈值：**

| 级别 | 材料缩放因子 | 支座沉降 |
|---|---|---|
| safe（安全） | > 0.90 | < 2.0 mm |
| warning（预警） | ≤ 0.90 | ≥ 2.0 mm |
| risk（风险） | ≤ 0.80 | ≥ 5.0 mm |
| danger（危险） | ≤ 0.70 | ≥ 7.0 mm |

**7 个结构区域**：4 个支座区域（左1/左2/右1/右2支座区）+ 3 个结构区域（右胯 / 跨中 / 左胯），由 `REGION_DEFINITIONS` 定义。

全桥风险等级取所有区域风险的最大值。

---

## 安装与依赖

### 环境要求

- Python 3.12+
- PyTorch 2.x（含 CUDA 支持版本请参考官方安装命令）

### 依赖安装（含清华源加速）

```bash
# 建议使用虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy matplotlib scikit-learn tqdm h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
```

导出依赖清单（可选）：

```bash
pip freeze > requirements.txt
```

---

## 开发与训练环境

本项目采用**本地开发 + 远程训练**的分离模式，请务必区分：

| 环节 | 机器 | 说明 |
|---|---|---|
| 代码修改与调试 | 本地（当前电脑） | 本地**无 GPU**，只做代码编辑、链路调试、数据检查、绘图和训练前准备 |
| 正式训练 | 远程服务器（4090 GPU） | 本地确认代码可运行后，将代码**复制到服务器**执行训练 |
| 数据存储 | 本地 `E:\working\DL_data\temperature_multistep_json_result` | V2 数据集（schema `4.0-temperature-state-json`），一个 JSON 对应一个结构状态，仍在生成中 |
| 训练产物查看 | 本地 | checkpoint / logs 从服务器拷回本地后出图 |

注意事项：

- **本地不跑正式训练**。训练前先在本地完成链路检查（1 epoch 冒烟测试，见"使用方式"），确认无误再上服务器。
- 正式实验数据目录以 `E:\working\DL_data\temperature_multistep_json_result` 为准；训练前需确认 `DL_config.py` 的 `data_root` 指向该目录。
- 服务器训练用的代码要与本地保持一致：改代码 → 本地验证 → 再复制到服务器，避免两边版本漂移。
- 旧数据（2816 个样本，位于 `E:/personal_files/Dongyaxun/DL_data/data_new`）的结果**不能与新数据混用**。

---

## 使用方式

### 训练 / 测试 / 推理（`DL_model/main.py`）

```powershell
# 先做 1 个 epoch 的链路检查
python DL_model/main.py --model static_only --epochs 1 --batch 2 --data-dir E:/working/DL_data/data_new

# 训练三种模型并做对比（默认：static_only → dynamic_only → fusion）
python DL_model/main.py --model all

# 仅训练融合模型
python DL_model/main.py --model fusion

# 指定数据目录（修正模型数据）与跨域参考数据
python DL_model/main.py --model all --data-dir E:/working/DL_data/data_new --reference-data-dir E:/working/DL_data/data_new/reference_model_dataset

# 测试
python DL_model/main.py --mode test --model fusion

# 单文件推理
python DL_model/main.py --mode infer --model fusion --input path/to/result_xxx.json
```

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `train` | `train` / `test` / `infer` |
| `--model` | `all` | `fusion` / `static_only` / `dynamic_only` / `all` |
| `--data-dir` | 配置值 | 修正模型数据集目录（`--corrected-data-dir` 的别名） |
| `--reference-data-dir` | 配置值 | 参考模型数据集（仅跨域测试用） |
| `--epochs` / `--batch` / `--lr` / `--seed` | 配置值 | 覆盖默认训练超参数 |
| `--input` | 无 | 推理模式的输入 JSON |
| `--resume` | 无 | checkpoint 路径（恢复/测试/推理） |

### 批量预测（`DL_model/predict.py`）

接受文件或目录参数，对每个样本输出预测结果与预测图：

```powershell
# 预测单个文件
python DL_model/predict.py path/to/result_xxx.json

# 预测整个目录下的所有 result_*.json
python DL_model/predict.py E:/working/DL_data/data_new
```

### 快速单文件推理（`DL_model/quick_predict.py`）

编辑脚本顶部的 `file_path` 后直接运行即可。

### 原始响应可视化（`plot/data_visualization.py` CLI）

绘制 V2 样本的振动时序与温度-位移曲线：

```powershell
python plot/data_visualization.py --input path/to/result_xxx.json --output-dir plot/outputs
# 可选参数：--vib-node --vib-components --static-node --disp-components --time-range --formats --dpi
```

### 重新生成对比图（`plot/logs/regenerate_plots.py`）

修改绘图包后，从已有 `method_comparison.json` 快速刷新对比图，无需重新训练：

```powershell
cd plot/logs
python regenerate_plots.py
```

---

## 绘图功能

绘图代码集中在 `plot/` 包，按功能分模块组织，全部使用中文注释：

| 模块 | 功能 | 主要函数 |
|---|---|---|
| `prediction.py` | 单样本预测图（材料/支座/区域/汇总） | `save_prediction_figures`、`plot_prediction_material`、`plot_prediction_support`、`plot_prediction_region`、`plot_prediction_damage_probability`、`plot_prediction_summary` |
| `training.py` | 训练历史曲线（损失/F1/准确率/精确率召回率/AUC/学习率） | `plot_history` |
| `evaluation.py` | 评估四联图 + 门控诊断 JSON | `plot_evaluation`、`save_gate_diagnostics` |
| `comparison.py` | 多模型对比图（方法对比/论文图/跨域对比） | `plot_method_comparison`、`plot_paper_figures`、`plot_domain_comparison`、`extract_reference_summaries` |
| `data_visualization.py` | 原始响应可视化 | `plot_vibration_timeseries`、`plot_temperature_displacement`、`plot_sample_responses`、`load_v2_sample` |

Python 中使用：

```python
from plot import plot_history, save_prediction_figures, plot_evaluation, plot_paper_figures
```

绘图自动配置中文字体（优先微软雅黑/黑体），并统一了论文级配色与 300 dpi 高分辨率输出。

---

## 输出说明

- **checkpoints/**`{model_type}/{run_name}_best.pt`：模型权重 + EMA + 优化器 + 训练历史 + 归一化参数 + 划分索引 + 阈值 + 配置。
- **logs/**`{model_type}/{run_name}_history.png`：6 子图训练曲线。
- **logs/**`{model_type}/{run_name}_evaluation.png`：混淆矩阵 + 各材料指标 + ROC + 概率分布。
- **logs/**`{model_type}/{run_name}_fusion_diagnostics.json`：融合诊断（α 权重、门控、分支分歧度）。
- **logs/method_comparison.json / .png**：多模型测试指标对比。
- **logs/paper_*.png**：论文级对比图（核心指标、各类别 F1、融合权重、辅助任务）。
- **logs/domain_comparison.png**：修正模型同源测试 vs 参考模型跨域测试对比。
- **logs/prediction_figures/**：批量预测时为每个样本生成的预测图。

---

## 当前状态与已知限制

> 详细说明见 [DATASET_AUDIT.md](DATASET_AUDIT.md) 与 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

当前状态（2026-08-09）：

- **正式训练数据集正在生成中**：早期审计时 `data_new` 仅有 11 个结构状态，只适合代码链路冒烟测试；新数据集生成完成后将按主线文档进入正式实验。
- 历史 `method_comparison.json` 和若干 checkpoint 来自 2816 个样本的旧实验（`E:/personal_files/Dongyaxun/DL_data/data_new`），**不能与新数据集结果混用**。

当前代码的已知限制（正式实验前需解决）：

1. **静态分支对温度利用不充分**：`temperature_steps_C` 虽被读取，但每个温度步的显式条件信息未充分编码。
2. **质量指标未进入训练输出**：`quality_metrics` 和缺测 mask 尚未成为 `StructuralDataset` 的主要输出，质量感知融合缺少输入基础。
3. **主模型仍用后融合**：仓库已有 `ReliabilityGatedFusionHead`，但主模型 `DualBranchFusion` 当前仍主要使用 `ClasswiseLateFusionHead`。
4. **评估指标不足**：缺少 AUPRC、RMSE、weighted Kappa、全桥状态指标和概率校准指标。
5. **可追溯性不足**：划分与 checkpoint 元数据主要记录数组索引，缺少样本 ID、数据版本和哈希。
6. **激励多样性不足**：旧数据所有样本复用同一条激励力时程，无法评价对未见激励的泛化。

---

## 后续工作安排

正式实验按阶段推进（详见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 与《深度学习实验后续工作主线》）：

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| 前置 | 数据冻结与审计（数据版本、样本清单、数据字典、标签分布、质量审计） | 全部样本可读，同一结构状态不跨集合 |
| P0 | 实验基础设施（run 元数据、ID 级 split manifest、输出目录隔离、评估脚本） | 每次运行可追溯 |
| 阶段 0 | 烟雾测试与负控（单 batch 检查、小样本过拟合、标签打乱负控） | 排除泄漏与链路错误 |
| 阶段 1 | 严格基线（先验/传统/单模态/late fusion 基线矩阵，5 种子） | 主结果表可复现 |
| 阶段 2 | 输入表示修正（逐温度显式编码、质量与 mask 进入 dataset） | 温度信息被显式利用 |
| 阶段 3 | 推荐主模型 `quality_aware_regional_fusion` | 严格划分下增益稳定 |
| 阶段 4 | 消融实验（A1–A8） | 每个模块必要性有答案 |
| 阶段 5 | 鲁棒性实验（噪声/缺测/温度步/标定误差/模态缺失） | 性能-扰动曲线 |
| 阶段 6 | 泛化与跨域（未见状态 / 未见种子 / 跨域） | 三层泛化结论成立 |
| 收尾 | 论文结果组织（Table 1–3、Figure 1–5） | 所有结论可由脚本复现 |

关键原则：5 个随机种子（`13, 29, 42, 71, 101`）；所有方法共用同一 split manifest；阈值只在验证集选择；测试集只评估一次；负结果和适用边界必须报告。

---

## 相关文档

- [RESEARCH_PLAN.md](RESEARCH_PLAN.md) — 研究计划：数据冻结、实验基础设施、阶段 0–6、指标体系、执行顺序与代码修改清单。
- 《深度学习实验后续工作主线》— 详细执行骨架（位于 `E:\working\my_project\具体工作\深度学习相关\深度学习实验后续工作主线.md`，Word 版：`深度学习实验后续工作安排.docx`）。
- [DATASET_AUDIT.md](DATASET_AUDIT.md) — `data_new` 数据集结构、字段说明与质量审计结论。
- [DL_model/README.md](DL_model/README.md) — 模型细节：架构、损失、训练策略与全部配置项。
- [DL_model/dataset_generator.md](DL_model/dataset_generator.md) — 数据集生成说明。
