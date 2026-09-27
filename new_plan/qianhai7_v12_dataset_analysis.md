# 前海7号 v12 数据集分析与生成流程

本文档说明前海7号参数化有限元数据集 v12 的数据内容、标签语义、HDF5 存储结构、数据生成流程和当前产物审计结果。

## 1. 数据集定位

v12 是带有反事实工程标签的 FEM 源域数据集，主要用于生成已知结构状态下的完整桥面三维位移场，并在训练阶段动态生成 InSAR-like 源域观测。

当前代码和数据位置：

- 生成器：`E:/working/PyAnsys/_lab_platform/dataset_generator_qianhai`
- 配置：`E:/working/PyAnsys/_lab_platform/dataset_generator_qianhai/config.py`
- 数据目录：`E:/working/DL_data/qianhai7_yearly_temperature_dataset_v12`
- schema：`qianhai7_fem_deck_field_v12`
- 当前样本文件：`qh7_state_000000.h5` 至 `qh7_state_000019.h5`，共 20 条结构状态轨迹

一个 HDF5 文件代表一个结构状态、一条 26 期 FEM 轨迹和一条年度温度轨迹，不是一个单独温度点。

v12 不是前海7号真实桥梁的数字孪生，也不表示真实桥梁已经发生确定损伤。状态参数是生成器主动设置的反事实参数，标签来源为 `synthetic_parameterized_fem_counterfactual`。

材料缩放因子、支座沉降、支座刚度、局部桥面状态和完整 FEM 节点场属于生成标签、训练期教师或审计信息，不能直接作为部署端学生模型输入，否则会造成标签泄漏。

## 2. 样本中的数据

### 2.1 网格、节点和坐标

| HDF5 路径 | 当前形状 | 含义 |
|---|---:|---|
| `/mesh/node_ids` | `[17323]` | 全部 FEM 节点编号 |
| `/mesh/node_coords` | `[17323, 3]` | 节点坐标，单位 mm，列为 X/Y/Z |
| `/mesh/element_connectivity` | `[17023, 4]` | 全部单元连接关系 |
| `/mesh/element_node_count` | `[17023]` | 每个单元实际节点数 |
| `/mesh/deck_node_ids` | `[17195]` | 桥面节点编号 |
| `/mesh/deck_node_mask` | `[17323]` | 全节点是否属于桥面 |
| `/mesh/deck_element_ids` | `[16920]` | 桥面单元编号 |
| `/mesh/deck_element_connectivity` | `[16920, 4]` | 桥面单元连接关系 |
| `/mesh/support_group_names` | `[5]` | 支座/基础节点组名称 |
| `/mesh/support_node_ids` | `[53]` | 53 个支座/基础节点编号 |
| `/mesh/support_node_group_index` | `[53]` | 支座节点对应的组索引 |
| `/mesh/support_node_mask` | `[17323]` | 全节点是否属于支座节点 |

节点坐标已存储在 `/mesh/node_coords`，单位属性为 `mm`。节点编号与坐标通过同一位置索引对应，节点编号不能直接当作数组下标使用。

支座/基础节点组已经拆分为：

```text
abutment_0_foundation
abutment_3_foundation
pier_1_foundation_a
pier_1_foundation_b
pier_2_foundation
```

`pier_1_foundation_a` 和 `pier_1_foundation_b` 是两个独立桥墩基础组。施加沉降、保存参数和细粒度标签时保持独立，计划级区域统计时才可以汇总为 `pier_1`。

### 2.2 FEM 位移和模态

```text
/responses/fem/disp_at_acquisition   [26, 17323, 3]
/responses/fem/modal_frequencies     [20]
```

`disp_at_acquisition` 的三个分量为 `UX、UY、UZ`，单位 mm；第一个维度是 26 个 FEM 获取时相，第二个维度是全部节点。垂向分量为 `UZ`，垂向正号为向上。

`modal_frequencies` 是参考温度下的模态频率，属于训练期教师或刚度特征的可选数据，不是与 26 个温度时相逐一对应的动态观测。

### 2.3 环境和时间轴

```text
/environment/temperature_by_acquisition_C   [26]
/environment/elapsed_days                  [26]
/environment/day_of_year                   [26]
/environment/delta_days                    [26]
```

默认锚点日期为 `2025-01-01`，共 26 个时相，相邻时相约 14 天，覆盖约一年。日期、年内日序、经过天数和相邻时间间隔写入 manifest 和 HDF5。

温度生成规则：

1. 如果提供 26 个温度值，直接使用该曲线；
2. 否则由月度平均高温、平均低温推导月度中心温度和月度扰动标准差；
3. 再叠加相邻时相相关的随机扰动并裁剪到温度边界。

当前 v12 运行清单记录的设置包括：月度中心温度为平均高温和平均低温的中点，月度扰动标准差为 `(平均高温 - 平均低温) / 4`，扰动相关系数为 `0.75`，温度范围为 `8.0°C` 至 `38.0°C`，参考温度为 `25.0°C`。

温度是 FEM 条件和模型输入，不是结构状态标签。

### 2.4 状态参数和物理真值

主要字段：

```text
/state/scenario_family
/state/material_ids
/state/material_scale_by_material
/state/support_group_names
/state/support_settlement_by_acquisition_mm
/state/support_stiffness_group_ids
/state/support_dof_names
/state/support_trans_scale_by_group
/state/support_rot_scale_by_group
/state/support_baseline_stiffness
/state/support_effective_stiffness
/state/local_deck_region_names
/state/local_deck_settlement_by_acquisition_mm
/state/local_deck_node_ids
```

这些字段分别记录材料刚度缩放、支座组逐期沉降、支座平移/转动刚度、基准/有效刚度、局部桥面区域和局部桥面逐期沉降。

局部桥面区域为：

```text
left_abutment_deck
pier_1_deck
midspan_deck
pier_2_deck
right_abutment_deck
```

局部桥面节点由桥面节点坐标和 `LOCAL_DECK_REGIONS` 的 X/Y 范围解析，不是固定外部节点编号。当前 20 条轨迹中局部桥面沉降数组均为零，说明字段和区域定义已经存在，但本批产物尚未实际覆盖该机制。

## 3. 标签体系

### 3.1 三个必须区分的概念

`scenario_family` 表示整条 26 期轨迹在采样阶段选择的粗粒度扰动组合，当前可取：

```text
healthy
material_only
support_only
combined
```

`state_family_by_acquisition` 表示每个 FEM 获取时相实际已经激活的细粒度工程状态族，是 v12 的主要状态族标签。支座沉降发生前不会提前标为沉降状态；材料和刚度若从第 1 期激活，则从第 1 期参与标签。

`severity` 表示生成器参数幅值对应的等级，不是对真实桥梁的工程诊断结论。

### 3.2 机制标签

```text
/labels/mechanism_names
/labels/mechanism_mask
```

机制轴为：

```text
0  material_degradation
1  support_settlement
2  support_stiffness_change
3  local_deck_settlement
4  local_deck_stiffness_change
```

`local_deck_stiffness_change` 当前明确标记为 `disabled_not_implemented`，不能视为已经生成。`local_deck_settlement` 已实现。

### 3.3 材料标签

```text
/labels/material_active_by_acquisition        [26]
/labels/material_severity_levels_by_material  [6]
```

材料 ID 为 `2、3、4、5、6、7`。材料严重度按缩放因子划分：

| 等级代码 | 名称 | 刚度缩放因子 |
|---:|---|---:|
| 0 | safe | `> 0.90` |
| 1 | warning | `0.81 - 0.90` |
| 2 | risk | `0.71 - 0.80` |
| 3 | danger | `0.20 - 0.70` |

材料状态默认从第 1 期开始，在全年保持不变。

### 3.4 支座沉降标签

```text
/labels/support_active_by_acquisition          [26, 5]
/labels/support_severity_level_by_acquisition  [26, 5]
```

5 列与 `/mesh/support_group_names` 的组顺序一致。沉降默认在 0-based 的第 `2` 至 `17` 期之间发生，包含上下界；发生后按 `linear_then_stable` 单调增长，增长步数为 `2` 至 `6`，达到最终幅值后保持稳定。

`support_severity_level_by_acquisition` 当前实际按规范限值比例生成，而不是按固定毫米阈值生成。清单中的比例阈值为：

- `ratio > 0.05`：warning；
- `ratio > 0.30`：risk；
- `ratio > 0.70`：danger。

因此，0.5/2/5 mm 只能作为数据集内部的旧版合成毫米分档说明，不能继续作为当前 v12 `support_severity_level_by_acquisition` 的字段定义。当前应使用 `support_settlement_ratio_to_limit_by_acquisition` 与 `support_normative_condition_grade_by_acquisition` 表达规范锚定语义，并在训练契约中明确使用哪一种严重度。

| 等级代码 | 名称 | 当前实际判断依据 |
|---:|---|---:|
| 0 | safe | `ratio <= 0.05` |
| 1 | warning | `0.05 < ratio <= 0.30` |
| 2 | risk | `0.30 < ratio <= 0.70` |
| 3 | danger | `ratio > 0.70` |

`support_early_anomaly_by_acquisition` 仍按绝对沉降量大于 `2.0 mm` 判断，不能与上述严重度等级混用。

### 3.5 支座刚度标签

```text
/state/support_trans_scale_by_group             [8]
/state/support_rot_scale_by_group               [8]
/state/support_baseline_stiffness               [8, 6]
/state/support_effective_stiffness              [8, 6]
/labels/support_stiffness_active_by_group       [8]
/labels/support_stiffness_active_by_acquisition [26, 8]
/labels/support_stiffness_severity_by_group     [8]
```

8 组索引是有限元支承实常数分组轴，不等于 5 个基础节点组。平移刚度使用 `UX、UY、UZ`，转动刚度使用 `ROTX、ROTY、ROTZ`。

支座刚度变化从第 1 期参与 `state_family_by_acquisition`。如果刚度缩放因子小于 1，则会包含：

```text
support_or_foundation_stiffness_reduction
```

刚度严重度使用与材料刚度相同的缩放区间。全局严重度取材料、支座沉降、局部桥面沉降和支座刚度严重度的最大值。

### 3.6 局部桥面标签

```text
/labels/local_deck_active_by_acquisition         [26, 5]
/labels/local_deck_severity_level_by_acquisition [26, 5]
/state/local_deck_settlement_by_acquisition_mm  [26, 5]
```

局部桥面沉降发生后按 `linear_then_stable` 发展，严重度阈值与支座沉降相同。局部桥面刚度变化仍是禁用机制。

### 3.7 细粒度状态族标签

```text
/labels/state_family_by_acquisition         [26]
/labels/fine_state_family_by_acquisition   [26]
/labels/coarse_state_family_by_acquisition [26]
/labels/state_family_code_by_acquisition   [26]
/labels/state_family_vocabulary             [40]
```

`state_family_by_acquisition` 和 `fine_state_family_by_acquisition` 应完全一致。`coarse_state_family_by_acquisition` 保留旧的 `healthy/material_only/support_only/combined` 语义，不能与细粒度标签混用。

细粒度词表覆盖健康、材料退化、独立基础沉降、多墩等量沉降、多墩差异沉降、桥台基础竖向变化、局部桥面沉降、支座/基础刚度降低，以及这些状态的材料和刚度组合族。代表性名称包括：

```text
healthy
material_degradation
pier_1_foundation_a_settlement
pier_1_foundation_b_settlement
pier_2_settlement
multi_pier_settlement
differential_pier_settlement
north_or_end_support_vertical_shift
deck_local_settlement
support_or_foundation_stiffness_reduction
mixed_support_settlement
material_degradation_and_deck_local_settlement
material_degradation_and_differential_pier_settlement
deck_local_settlement_and_support_or_foundation_stiffness_reduction
material_degradation_and_support_or_foundation_stiffness_reduction
```

完整词表以每个 HDF5 文件中的 `/labels/state_family_vocabulary` 为准，编码以其数组索引为准。

核心映射规则：

| 实际激活状态 | 细粒度族 |
|---|---|
| 没有激活机制 | `healthy` |
| 只有材料退化 | `material_degradation` |
| 只有 `pier_1_foundation_a` | `pier_1_foundation_a_settlement` |
| 只有 `pier_1_foundation_b` | `pier_1_foundation_b_settlement` |
| 只有 `pier_2_foundation` | `pier_2_settlement` |
| 两个或多个桥墩组等量运动 | `multi_pier_settlement` |
| 多个桥墩组幅值不一致 | `differential_pier_settlement` |
| 任一桥台基础组 | `north_or_end_support_vertical_shift` |
| 多个支座/桥墩类别混合 | `mixed_support_settlement` |
| 局部桥面沉降 | `deck_local_settlement` |
| 只有支座刚度变化 | `support_or_foundation_stiffness_reduction` |
| 材料退化与其他族同时激活 | 在对应族前加 `material_degradation_and_` |
| 沉降族与支座刚度同时激活 | 在对应沉降族后追加刚度族 |

### 3.8 窗口标签

manifest 中的 `windowing.windows` 从 26 期轨迹滚动生成长度为 `4、6、8` 的窗口，默认主窗口长度为 6。窗口目标为窗口末期状态：

```text
target_state_family = state_family_by_acquisition[target_index]
```

窗口同时保存目标日期、状态编码、材料严重度、支座沉降、支座严重度和全局严重度。同一个 `structural_state_id` 产生的全部窗口必须进入同一数据划分。

## 4. HDF5 文件结构

每个生产样本是一个自包含 HDF5 文件：

```text
qh7_state_000000.h5
├── mesh/                 # 节点、坐标、单元、桥面集合、支座组
├── responses/fem/        # 26期完整位移场、20阶模态频率
├── environment/          # 温度、日期特征、delta_days
├── state/                # 场景、材料、支座和局部桥面参数
├── labels/               # 机制、严重度和细粒度状态族
└── manifest_json         # 内嵌 UTF-8 manifest
```

`mesh` 保存节点、坐标、单元连接、桥面节点/单元集合以及支座节点组。单元连接采用固定宽度数组，实际节点数由 `element_node_count` 提供；桥面壳单元保留 MAPDL 原始周界顺序。

`responses/fem` 保存 `/responses/fem/disp_at_acquisition [26, N_node, 3]` 和 `/responses/fem/modal_frequencies [20]`。前者是完整 FEM 位移场，后者是参考温度下的教师级模态特征。

`environment` 保存温度和时间轴；`state` 保存生成器设置的反事实参数；`labels` 保存由参数推导的监督目标。

`manifest_json` 是内嵌 UTF-8 JSON，记录 schema、生成器版本、数据库哈希、随机种子、状态参数、标签版本、数组路径、shape、dtype、单位和数组级 SHA-256。

## 5. 数据生成器的完整流程

生成器入口为：

```text
E:/working/PyAnsys/_lab_platform/dataset_generator_qianhai/main.py
```

主要实现文件为：

| 文件 | 作用 |
|---|---|
| `config.py` | 输出目录、样本数、节点组、温度、风险范围和采样轴 |
| `state_sampling.py` | 状态采样、沉降轨迹、细粒度标签和窗口标签 |
| `mapdl_adapter.py` | 网格读取、状态施加、模态和静力求解 |
| `dataset_writer.py` | manifest 构建和 HDF5 写入 |
| `run_control.py` | 运行签名、续跑索引和目录护栏 |
| `main.py` | 温度轨迹、样本循环和求解/导出入口 |

### 5.1 运行护栏和随机种子

`config.py` 中的主要配置包括 `RUN_SOLVE`、`OUTPUT_DIR`、`SAMPLES`、`START_INDEX`、`DB_PATH`、材料随机化范围、支座节点组、温度、沉降、局部桥面区域和支座刚度参数。

启动时，`run_control.py` 根据 schema、数据库哈希、随机种子、温度设置、节点分组、状态采样配置和模态设置计算运行签名，并写入 `_run_manifest.json`。如果输出目录属于另一套配置，生成器会拒绝续写，防止不同版本数据混用。

第 `i` 个结构状态使用随机种子：

```text
RANDOM_SEED_BASE + i
```

续跑从下一个未完成 HDF5 编号开始，不改变已有样本的随机序列，也不会把 `.plan.json` 误认为完成的 FEM 样本。

### 5.2 生成年度温度轨迹

生成器先创建 26 个 FEM 获取日期，然后使用显式 26 点温度曲线，或从月度基线插值得到中心温度并叠加按月标准差和相邻时相相关的随机扰动。温度最终裁剪到配置边界，并写入 manifest 和 `/environment/temperature_by_acquisition_C`，用于后续 26 次热-静力求解。

### 5.3 采样结构状态

当前整条轨迹从以下四个场景中抽取：

| 场景 | 概率 |
|---|---:|
| `healthy` | 0.15 |
| `material_only` | 0.35 |
| `support_only` | 0.20 |
| `combined` | 0.30 |

支座场景内部的子设计权重为：

| 子设计 | 权重 |
|---|---:|
| `support_settlement_only` | 0.35 |
| `support_stiffness_only` | 0.25 |
| `support_settlement_and_stiffness` | 0.20 |
| `local_deck_settlement_only` | 0.20 |

随后采样材料 ID、材料缩放等级、基础节点组、沉降幅值/发生期/增长步数、支承刚度组及平移/转动刚度缩放。局部桥面状态从 5 个坐标区域中选择，并由桥面节点坐标解析实际节点。需要注意，当前运行清单中的支座子设计为 `support_settlement_only`、`support_stiffness_only` 和 `support_settlement_and_stiffness`，并未包含旧文档中写出的 `local_deck_settlement_only`；当前 20 条轨迹也未出现局部桥面沉降。

### 5.4 生成逐期标签

对每个获取时相计算材料 active 和严重度、支座沉降 active 和严重度、局部桥面 active 和严重度、支座刚度 active 和严重度、全局最大严重度、细粒度状态族和整数编码，随后为 4、6、8 期滚动窗口附加窗口末期标签。

标签先由反事实状态参数确定，再执行 FEM 求解；FEM 求解负责产生与这些状态一致的物理响应。`state_family_by_acquisition` 在异常机制发生前可以为 `healthy`，因此 `coarse_state_family_by_acquisition` 也可能在同一条 `support_only` 或 `combined` 轨迹的早期时相为 `healthy`；整条轨迹的 `scenario_family` 表示采样设计，不等于每个时相已经激活的状态。

### 5.5 读取网格和拓扑审计

MAPDL 适配器会恢复数据库，读取全部节点、坐标、单元和连接关系；优先使用 `DECKE` component 识别桥面，必要时按已审计的材料、截面和 `SHELL181` 条件回退。随后检查 53 个支座节点全部存在、局部桥面区域能够解析到实际桥面节点，并保留桥面壳单元原始周界顺序。

### 5.6 施加状态并计算 FEM

材料状态会缩放每个材料实际定义的弹性模量，而不是只修改 `EX`。支座刚度通过 v4 参数规范对应的实常数施加平移和转动刚度缩放。随后在参考温度和当前刚度状态下提取 20 阶模态频率；模态计算不施加支座沉降。

每个热-静力时相依次清理并施加当前支座节点 `UZ` 沉降、清理并施加当前局部桥面节点 `UZ` 沉降、清理温度荷载、设置参考温度并施加当前温度、执行静力求解，最后从 RST 读取全部节点 `UX、UY、UZ` 并排列到统一节点顺序。

最终得到 `[26, N_node, 3]` 的完整 FEM 位移场。

### 5.7 写入 HDF5 和生成训练观测

生成器将网格、响应、环境、状态参数和标签写入 HDF5，并把完整 manifest 写入 `/manifest_json`。数组描述包含 shape、dtype、单位和 SHA-256。

v12 主文件保存完整 FEM 桥面场，不保存固定真实 PS 观测。训练 loader 后续从桥面场动态采样可变数量虚拟传感点，使用拓扑和形函数插值 `UZ`，执行参考处理并加入噪声、缺测、点偏置和质量字段，形成 `source_view`。

真实 InSAR 属于独立的 `target_view`，使用真实日期、坐标、质量和缺测信息。完整 FEM 节点场、状态参数和标签不能直接作为部署端学生模型输入。

## 6. 当前 v12 产物审计

截至 2026-09-22，数据目录中实际包含：

```text
_run_manifest.json
qh7_state_000000.h5
...
qh7_state_000019.h5
```

共 20 条结构状态轨迹；每条轨迹包含：

- 26 个 FEM 获取时相；
- 4、6、8 期三种滚动窗口，共 63 个窗口；
- `[26, 17323, 3]` 的完整三维位移场；
- 20 阶参考温度模态频率；
- 温度、时间间隔、支座/材料/刚度参数和逐期标签。

20 条轨迹的粗粒度采样场景分布为：

| `scenario_family` | 轨迹数 |
|---|---:|
| `healthy` | 3 |
| `material_only` | 7 |
| `support_only` | 5 |
| `combined` | 5 |

实际状态族覆盖包括：

- `healthy`；
- `material_degradation`；
- `support_or_foundation_stiffness_reduction`；
- `north_or_end_support_vertical_shift` 与材料/刚度组合；
- `differential_pier_settlement`；
- `mixed_support_settlement`；
- `pier_1_foundation_b_settlement` 与材料/刚度组合；
- `material_degradation_and_differential_pier_settlement`；
- 上述状态的若干组合族。

当前尚未出现的主要内容是：

- `pier_1_foundation_a_settlement` 独立状态；
- `pier_2_settlement` 独立状态；
- `multi_pier_settlement` 等量沉降；
- `deck_local_settlement`，20 条轨迹的局部桥面沉降数组最大值均为 0；
- 动态加速度、激励力和动态时程。

支座沉降最终最大值范围为 `0–161.454 mm`。非零轨迹中既有约 `7–26 mm` 的低/中幅状态，也有约 `55–161 mm` 的高幅、接近或超过规范参考限值的状态。沉降均按“发生前为零、单调增长、达到峰值后稳定”的轨迹保存，发生期和增长步数均落在运行清单范围内。

### 6.1 当前产物完整性与标签一致性

对 20 个 HDF5 文件执行了批量检查，结果如下：

- 所有文件均可读取，schema 一致；
- `/responses/fem/disp_at_acquisition` 均为 `[26, 17323, 3]`；
- `/mesh/node_coords` 均为 `[17323, 3]`，单位为 mm；
- `/responses/fem/modal_frequencies` 均为 `[20]`；
- 所有响应、温度、沉降和模态数组均未发现 NaN 或 Inf；
- `support_active_by_acquisition` 与支座沉降是否非零完全一致；
- 局部桥面 active 标签与当前全零沉降数组一致；
- `state_family_by_acquisition` 与 `fine_state_family_by_acquisition` 一致；
- 状态族整数编码可以正确索引目录清单中的 40 项词表；
- 支座严重度与规范限值比例阈值一致；
- 沉降轨迹均单调，窗口目标与窗口末期状态、沉降和严重度一致；
- 每个 HDF5 的数组描述中，位移场和模态频率的 SHA-256 均可复算。

发现一项需要保留在训练契约中的语义差异：3 条 `support_only` 轨迹的前期 `coarse_state_family_by_acquisition` 为 `healthy`，沉降发生后才变为 `support_only`。这与逐期激活逻辑一致，不应把整条轨迹的 `scenario_family` 复制到所有时相作为标签。

## 7. 批量生成和正式训练前要求

### 7.1 当前内容是否足以支撑后续模型训练

这里不讨论样本数量，只判断数据字段和物理内容是否形成训练闭环。

**可以直接支撑的源域任务：**

1. 基于完整 FEM 位移场和温度/时间轴的逐期状态族分类；
2. 支座组沉降量、沉降是否激活、规范限值比例和趋势预测；
3. 材料严重度、支座刚度变化和全局严重度的多任务训练；
4. 从完整桥面场派生区域位移、虚拟 PS 观测和区域异常评分；
5. 使用滚动窗口进行“窗口末期状态”预测。

**目前不能单独支撑的任务：**

1. 直接训练依赖动态加速度或激励力的融合模型，因为 v12 HDF5 没有动态响应组；
2. 直接训练局部桥面沉降识别，因为当前 20 条轨迹该机制实际没有正样本；
3. 仅依靠这些文件训练部署端真实 InSAR 模型，因为文件保存的是完整 FEM 场，不是经过真实 PS 坐标、参考定义、缺测、噪声和偏置处理后的 observation space；
4. 将 `modal_frequencies` 当作逐时相输入，因为它是一条结构状态级、参考温度下的一次性教师特征；
5. 将材料缩放、支座沉降、有效刚度、完整 FEM 场或其他 state 字段直接作为部署输入，否则会产生目标泄漏。

因此，当前内容的判断是：

> **v12 已足以支撑“静态 FEM 源域 + 温度/时间 + 结构状态标签”的模型训练和数据链路验证；尚不足以单独支撑完整的部署端 InSAR 模型、动态融合模型或覆盖全部设计状态族的模型训练。**

后续训练前至少需要在 loader 中完成：

- 从 `/responses/fem/disp_at_acquisition` 映射到真实或拟真的 PS 坐标；
- 提取部署端可用的 `UZ` 和有效掩码；
- 加入真实日期/不规则时间间隔、参考处理、缺测、噪声、点偏置和质量字段；
- 明确训练目标使用 `dataset_severity`、`support_settlement_ratio_to_limit` 还是 `normative_condition_grade`；
- 按 `structural_state_id` 固定划分，禁止把同一轨迹的滚动窗口拆到不同集合。

### 7.2 必须覆盖的基础状态族

正式数据集不能只依靠随机抽样后再观察结果，至少应定向保证以下基础族有足够结构状态样本：

```text
healthy
material_degradation
pier_1_foundation_a_settlement
pier_1_foundation_b_settlement
pier_2_settlement
multi_pier_settlement
differential_pier_settlement
north_or_end_support_vertical_shift
deck_local_settlement
support_or_foundation_stiffness_reduction
```

随后再增加材料退化、支座刚度和其他状态的工程合理组合。每个族不能只有一个结构状态，否则无法进行状态级训练、验证、测试和泛化评估。

### 7.3 批量生成后的审计项目

批量生成完成后，应重新统计：

1. HDF5 文件数量和可读性；
2. 每个基础状态族的结构状态数；
3. 每个组合状态族的结构状态数；
4. 26 期 acquisition-level 标签数量；
5. `safe`、`warning`、`risk`、`danger` 的分布；
6. 各支座组 active 次数和沉降幅值分布；
7. 5 个局部桥面区域的覆盖；
8. 8 组支座刚度轴的 active 和 scale 分布；
9. 材料 ID 和材料缩放分布；
10. 温度、日期和 `delta_days` 完整性；
11. 节点坐标、桥面拓扑和位移场 shape 一致性；
12. manifest、数据库指纹和数组级 SHA-256 一致性。

### 7.4 数据划分原则

训练、验证和测试必须按 `structural_state_id` 或 `split_group` 划分。一个 HDF5 文件的全部 26 期、全部滚动窗口和任何观测实现都必须进入同一个集合，不能按时相或窗口随机拆分。

## 8. 最终结论

当前 v12 已经不是单样本冒烟数据，而是包含 20 条结构状态轨迹的可用源域数据快照。它已经具备：

- 完整桥梁网格、节点坐标和桥面拓扑；
- 26 期年度温度和时间轴；
- 26 期全节点三维 FEM 位移；
- 材料、支座沉降、支座刚度和规范参考量；
- 逐期细粒度状态族、严重度、趋势和窗口末期目标；
- 20 阶结构状态级模态特征；
- 内嵌 manifest、数据库指纹和数组哈希。

从“数据集所包含的内容”看，**可以进入静态 FEM 源域模型开发和 observation-space loader 开发**。但它仍不是完整的最终训练集：局部桥面沉降尚未实际覆盖，动态响应未保存，真实 InSAR 观测域还需要单独构造，且支座严重度存在“规范限值比例”与“旧固定毫米阈值”两套语义，训练前必须冻结字段契约。

正式训练前的最低内容闭环是：

1. 明确并冻结部署输入、训练期 privileged information 和监督标签；
2. 将完整 FEM 场转换为 observation space，再加入缺测、噪声、参考和质量建模；
3. 对局部桥面沉降和独立桥墩状态补齐正样本；
4. 如需动态融合，增加动态响应及其激励/边界条件审计；
5. 按结构状态级别生成 split manifest，并对窗口目标和标签口径做最终冻结。

