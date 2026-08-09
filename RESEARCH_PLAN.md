# 桥梁多模态安全状态评估研究计划

> 更新日期：2026-08-09。本计划已与《深度学习实验后续工作主线》（`E:\working\my_project\具体工作\深度学习相关\深度学习实验后续工作主线.md`，下称"主线文档"）对齐。主线文档是详细执行骨架，本文件是仓库内的研究计划与阶段安排。

## 1. 研究定位与口径

- 研究对象：桥梁安全状态**辅助评估**，不是通用图像识别或精确损伤反演。
- 公共样本单元：`data_domain - structural_state_id - stage/window - component/region`。
- V2 输入：`responses.static.disp`（多温度位移）、`environment.temperature`、`responses.dynamic.ace`（参考温度下加速度时程）、测点坐标、质量指标和缺测掩码。
- V2 监督：材料候选多标签、支座状态、区域风险等级、全桥风险等级；幅值回归作为辅助任务。
- `material_scaling_factors`、有限元生成参数和其他真值字段只能用于标签、分组或审计，**不能作为模型输入**。
- 主切分按 `structural_state_id` / `state_family_id` / 实验批次进行；同一状态的温度步、滑动窗口和重复采集不得跨集合。
- 关键结论至少使用 5 个随机种子（`13, 29, 42, 71, 101`），报告均值±标准差或 95% 置信区间。

## 2. 总体研究问题

**主问题**：在严格未见结构状态划分下，双模态融合是否比最佳单模态和普通后融合具有稳定增益？

支撑问题：

1. 静态多温度位移、动态振动响应分别贡献什么信息？
2. 显式温度编码、质量指标、缺测 mask、区域级融合和层级一致性约束是否真正必要？
3. 在传感器缺测、动态噪声、温度步减少、幅值标定误差和跨域数据下，融合模型是否仍然可靠？

论文定位表述（谨慎版）：

> 本研究面向桥梁多源响应辅助状态评估，构建可追溯的数据划分和多任务评估流程，比较静态、动态与融合模型在未见结构状态上的泛化性能，并进一步分析质量退化和模态缺失条件下的鲁棒性。

**边界**：在没有真实桥梁强标签前，不宣称"精确损伤定位"或"直接替代工程检测"，只报告辅助识别、风险分级、复核触发和异常区域提示。

## 3. 数据冻结与审计

正式实验开始前先做一次数据冻结，防止后续所有指标混在一起。

必须冻结的内容：

- `data_version`：建议格式 `v2_YYYYMMDD_Nxxxx`（`Nxxxx` 为样本数）。
- 样本清单：每个 JSON 的 `sample_id`、文件名、`structural_state_id`、`state_family_id`、`replicate_id`、`excitation_seed`、`temperature_seed`、`scenario`、标签摘要和 SHA-256。
- 数据字典：静态位移、动态加速度、温度、质量指标、标签和元数据字段的 shape、单位、用途和禁用范围。
- 标签分布表：材料、支座、区域、全桥风险等级在训练/验证/测试的分布。
- 质量审计表：NaN/Inf、shape、dtype、时间轴、采样率、激励 RMS、动态通道 RMS、静态温度步与位移序列对应关系。

通过条件：

- 所有样本可由 `StructuralDataset` 读取；静态 `[n_temp, n_static_nodes, 3]`，动态 `[n_time, n_dynamic_nodes, 3]`。
- 标签长度和类别空间固定；同一 `structural_state_id`（或更严格的 `state_family_id`）不跨集合。
- 归一化统计量只来自训练集；阈值和校准参数只来自验证集；测试集不能参与调参。

## 4. 实验基础设施改造

先改基础设施，再改模型，否则模型越复杂，结果越难解释。

| 优先级 | 模块 | 建议修改 | 目的 |
|---|---|---|---|
| P0 | 配置系统 | 增加 `data_version`、`run_id`、`seed`、`split_manifest`、`output_dir` | 每次运行可追溯 |
| P0 | 数据划分 | 保存 ID 级 split manifest，不只保存数组索引 | 防止新旧数据和不同划分混用 |
| P0 | 输出目录 | 用 `runs/<data_version>/<split>/<model>/<seed>/<timestamp>` | 避免不同实验互相覆盖 |
| P0 | 数据集读取 | 输出 `sample_id`、`structural_state_id`、温度步、质量向量、缺测 mask | 支撑严格评估与质量感知模型 |
| P0 | 评估脚本 | 保存 sample-level predictions | 支撑错误分析、配对检验和论文作图 |
| P1 | 模型注册 | 用统一 registry 管理 `static_only`、`dynamic_only`、`concat`、`late_fusion`、`quality_gated` | 保证实验入口一致 |
| P1 | 指标模块 | 增加 AUPRC、RMSE、weighted Kappa、ECE、Brier、NLL | 覆盖分类、回归、等级和校准 |
| P1 | 聚合脚本 | 汇总 5 个随机种子的均值、标准差、置信区间 | 支撑正式结果表 |

注：上表 P0/P1 为**优先级**（P0 先做，P1 随后），与后文"阶段 0-6"的执行阶段编号不是一回事。

每次运行需保留的最小元数据：Git commit/代码快照哈希、`data_version` 和数据清单哈希、split manifest 路径和哈希、模型超参数与种子、训练集归一化状态、最优 epoch 与早停依据、验证集阈值、校准方法、评估脚本版本。

## 5. 阶段 0：烟雾测试与负控

新数据生成完成后，不要直接跑完整模型，先用最小成本排除数据泄漏、标签错位和训练链路错误。

| 实验 | 做法 | 通过条件 |
|---|---|---|
| 0.1 单 batch 读数检查 | 一个 batch 内所有张量 shape/dtype/mask/标签/样本 ID 检查 | 静态、动态、温度、质量、标签都能进入 forward，无 NaN/Inf |
| 0.2 小样本过拟合 | 取 8–16 个样本，训练 20–50 个 mini-epochs | 训练 loss 明显下降，训练集指标接近过拟合 |
| 0.3 标签打乱负控 | 训练集标签随机打乱，输入不变 | 验证/测试性能接近随机或类别先验水平 |
| 0.4 真值字段审计 | 确认 `material_scaling_factors`、`support_settlement` 等未进入模型输入 | dataset 输出只含观测量、坐标、温度、质量、mask 和监督标签 |

快速起步命令（1 epoch 链路检查，数据生成完成后先跑这条）：

```powershell
python DL_model/main.py --model static_only --epochs 1 --batch 2 --data-dir E:/working/DL_data/data_new
```

## 6. 阶段 1：严格基线

基线矩阵（基线弱则融合模型赢了也没说服力）：

| 类别 | 方法 | 用途 |
|---|---|---|
| 先验基线 | Dummy / prevalence baseline | 数据不平衡下的最低参照 |
| 传统静态 | 静态工程特征 + tree model | 多温度位移是否已有显著信息 |
| 传统动态 | 动态频域/RMS/峰值特征 + tree model | 振动特征的非深度学习上限 |
| 深度静态 | 当前 `static_only` | 单模态深度基线 |
| 深度动态 | 当前 `dynamic_only` | 单模态深度基线 |
| 简单融合 | feature concat | 早期拼接是否足够 |
| 分数融合 | static/dynamic 固定 0.5 平均 | 学习型融合是否必要 |
| 当前融合 | class-wise late fusion | 现有模型主基线 |
| 推荐新模型 | quality-aware regional fusion | 后续主模型 |

训练原则：最终结果用 5 个种子（`13, 29, 42, 71, 101`）；所有方法用完全相同的 split manifest 与训练/验证/测试集合；阈值只在验证集选择；测试集只评估一次。

## 7. 阶段 2：输入表示修正

解决当前"读了温度但没有充分用温度"的问题：

- 静态分支：把 `temperature_steps_C` 作为每个温度步的显式条件输入（温度 token / FiLM / 温度位置编码），保留测点坐标和温度步索引，不要把 6 个温度步当无序通道。
- 动态分支：保留当前 1D CNN 轻量起点；输入保留测点坐标、采样率、激励摘要和通道质量；数据足够前不同时堆叠 Transformer/GNN。
- 质量与 mask：`quality_metrics` 进入 dataset 输出；先构造少量稳定质量特征（动态通道 RMS、峰值、平稳性、激励 RMS 相对误差、静态温度步有效 mask）；显式加入模态/传感器/温度步缺失 mask。

通过条件：静态-only 移除温度显式输入后出现可解释变化；质量感知模型在缺测/噪声下优于不用质量/mask 的版本；每个错误样本可追踪到结构状态、温度种子、激励种子和质量指标。

## 8. 阶段 3：推荐主模型 `quality_aware_regional_fusion`

结构骨架：静态/动态编码器分别输出全局特征和区域 token → 通过测点/材料/支座到区域的映射建立区域级候选表示 → 融合头接收静态特征、动态特征、质量向量和缺测 mask 输出任务相关 gate → 对材料、支座、区域、全桥等级分别输出 → 保留单模态旁路（模态缺失仍可推理）。

损失函数：材料多标签 BCE/focal BCE；支座分类 BCE；支座沉降回归 MAE/SmoothL1（辅助）；区域等级 ordinal/cross entropy；全桥等级 cross entropy；层级一致性（全桥 ≥ 区域最大等级）；校准（temperature scaling）作为训练后处理，不急于写进训练损失。

注意事项：gate 只解释为"信息注入比例"，不是物理可靠性概率；新模型只在 random split 提升、strict state split 不提升时，优先解释为泄漏或过拟合；静态分支对材料损伤不敏感也要保留并报告。

## 9. 阶段 4：消融实验

与主模型相同 split、相同种子、相同训练预算，只做回答机制问题的消融：

| 编号 | 消融项 | 回答的问题 |
|---|---|---|
| A1 | 移除逐温度编码，只留平均温度 | 多温度条件是否被利用 |
| A2 | 移除测点坐标/位置编码 | 空间信息是否必要 |
| A3 | 移除质量指标与缺测 mask | 质量感知是否带来鲁棒性 |
| A4 | 区域交互替换为简单 concat | 区域级交互是否必要 |
| A5 | learned gate 替换为固定 0.5 平均 | 学习型融合是否必要 |
| A6 | 移除层级一致性约束 | 全桥与区域标签是否更容易冲突 |
| A7 | 移除辅助回归任务 | 支座沉降辅助监督是否有益 |
| A8 | 移除 warm start | 训练策略是否影响结论 |

主要观察：主指标是否下降、危险等级召回是否下降、层级违规率是否上升、缺测/噪声下是否更脆弱、gate 分布是否退化为单一模态。

## 10. 阶段 5：鲁棒性实验

只测试时扰动，不重新训练（数据增强对照除外）。

| 扰动类型 | 梯度 |
|---|---|
| 动态噪声 SNR | clean, 30 dB, 20 dB, 10 dB, 5 dB |
| 动态传感器 dropout | 0, 1, 2, 3 of 6 |
| 静态温度步保留 | 6, 4, 3, 2 |
| 幅值标定误差 | ±5%, ±10%, ±20% |
| 模态缺失 | static-only / dynamic-only / full inference |

执行要求：扰动种子固定并记录；只作用于测试输入不改标签；每种扰动保存 sample-level predictions；输出性能-扰动强度曲线；同时报告平均性能和危险等级召回。若融合模型 clean 最好但轻微缺测后迅速崩溃，不应宣称鲁棒；质量感知模型在低质量输入下主动降权且性能下降更缓，可作为主模型优势证据。

## 11. 阶段 6：泛化与跨域

三层递进，不要一开始就把所有跨域问题混在一起：

1. **未见结构状态**：按 `structural_state_id` / `state_family_id` 划分，论文主结果以这一层为准。
2. **未见激励或温度种子**：同一状态可在训练中出现，但测试用未见 `excitation_seed` / `temperature_seed`，回答模型是否记住固定激励/温度组合。
3. **FE → 缩尺/实桥域**：先 zero-shot 只评估不微调，再少量目标域微调。若接入实桥数据，可考虑的线索源包括 COSMO-InSAR、温度、运营振动和巡检/养护记录；实桥无强标签时只报告一致性、异常提示和复核线索，不输出精确损伤位置、损伤程度或多年退化趋势。

跨域指标：目标域整体指标、源域→目标域性能下降、校准变化、高风险召回变化、错误案例按区域/模态质量/数据域归因。

## 12. 指标体系

- 材料多标签：主指标 macro-AUPRC；辅助 macro-F1、micro-F1、Exact Match、per-class recall、ROC-AUC。
- 支座：分类 macro-AUPRC、macro-F1、per-support recall；沉降回归 MAE、RMSE。
- 区域/全桥等级：macro-F1、weighted Kappa、等级 MAE、confusion matrix。
- 可信度与安全：ECE、Brier、NLL、危险等级 recall、层级规则违规率（全桥预测等级低于区域最大预测等级的比例）。

结果报告：5 种子均值±标准差；图中误差条或 95% CI；主模型与最佳基线做配对比较（配对单位为 seed 或 sample-level）。

## 13. 推荐执行顺序

1. 冻结新数据版本，生成数据字典、样本清单、标签分布和质量审计。
2. 实现 ID 级 split manifest，按 `structural_state_id` / `state_family_id` 严格划分。
3. 改造输出目录和 run metadata，保证不同模型、种子、数据版本不互相覆盖。
4. 扩展 `StructuralDataset`：输出样本 ID、温度步、质量向量和缺测 mask。
5. 扩展评估指标，保存 sample-level predictions。
6. 跑烟雾测试、过拟合小样本、标签打乱负控。
7. 跑传统基线、单模态深度基线和当前 late fusion。
8. 修正静态分支的逐温度条件编码。
9. 接入 `ReliabilityGatedFusionHead`，形成质量感知后融合版本。
10. 实现区域级 token 融合，形成主模型 `quality_aware_regional_fusion`。
11. 跑 5 seed 主实验，生成主结果表。
12. 跑消融实验。
13. 跑鲁棒性实验，生成性能-扰动曲线。
14. 若有缩尺/实桥数据，再做跨域评估和错误案例分析。
15. 汇总论文图表、主结论、负结果和适用边界。

## 14. 代码修改清单

| 文件 | 修改方向 |
|---|---|
| `DL_model/DL_config.py` | 增加实验 ID、数据版本、split、输出目录、种子、指标和模型类型配置 |
| `DL_model/dataset.py` | 返回 `sample_id`、`structural_state_id`、`temperature_steps_C`、`quality_metrics`、mask 和泄漏字段检查 |
| `DL_model/model.py` | 显式温度编码、质量感知融合头、区域 token 输出和模型 registry |
| `DL_model/trainer.py` | 新增指标、sample-level prediction 保存、校准、run metadata |
| `DL_model/main.py` | 统一 CLI 参数、run directory、split manifest 加载和多 seed 调度 |
| 新增 `scripts/audit_dataset.py` | 数据冻结与审计 |
| 新增 `scripts/make_splits.py` | ID 级严格划分 |
| 新增 `scripts/evaluate_corruptions.py` | 鲁棒性扰动测试 |
| 新增 `scripts/aggregate_runs.py` | 多 seed 结果聚合 |

**注意**：保留当前未提交修改，不要回退已有设置，例如 `fusion_train_alpha_only = False`、`fusion_safety_source = "fused"`、DataLoader `pin_memory=False`。

## 15. 论文结果组织

- Table 1：数据集统计（样本数、状态数、标签分布、场景分布、数据域）。
- Table 2：主结果（传统基线、单模态、concat、late fusion、主模型）。
- Table 3：消融实验。
- Figure 1：数据与模型流程图。
- Figure 2：各模型 per-class recall 或 AUPRC。
- Figure 3：鲁棒性曲线（扰动强度 × macro-AUPRC 或危险等级 recall）。
- Figure 4：校准图/可靠性图。
- Figure 5：典型错误案例和区域风险热图。

结果段落逻辑：先报告严格未见结构状态主结果 → 解释单模态贡献 → 用消融证明温度编码、质量/mask、区域融合不是装饰 → 用鲁棒性证明质量感知融合更稳定 → 报告失败案例和边界条件。

**结果台账（硬性要求）**：每张结果表和每幅图必须记录 `data_version`、模型/代码版本、数据域、split 策略、随机种子、超参数、阈值、校准方法、评估脚本和运行日期，建议统一保存到 `DL_model/logs/<model_type>/`，并用一个 JSON 汇总表作为论文结果唯一来源。

## 16. 完成标准

当且仅当以下内容齐备，可认为实验主线完成：

- 有冻结的数据版本和可复验的数据审计报告。
- 有 ID 级 split manifest，严格保证状态不泄漏。
- 所有主模型和基线共用同一组 split 与 seeds。
- 主结果、消融、鲁棒性和校准指标都能由脚本复现。
- 每个 checkpoint 可追溯到代码版本、数据版本、划分、种子和配置。
- sample-level predictions 可用于错误分析。
- 论文中每个性能结论都有对应实验表或图支撑。
- 明确报告负结果和适用边界。

核心思想：先把实验地基钉牢（数据版本、划分、指标、输出都可追溯），再让模型逐步长复杂。无论结果是提升、持平还是暴露短板，都能转化为可信的研究结论。
