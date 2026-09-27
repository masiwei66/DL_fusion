# 前海7号参数化 FEM 反事实数据与垂向 InSAR 数据格式

> 研究总纲、最终模型输入输出和各类数据在训练中的作用见：[研究总纲与 AI-数据契约](<研究总纲与 AI-数据契约.md>)。

> 本文件的 FEM 不是高精度真实桥梁反演模型，而是一个与前海7号主要几何、桥面/桥墩位置和基本传力关系一致的参数化反事实数据生成器。目标是归档标签已知的完整桥面位移场，并在训练阶段按任意传感布局动态生成稀疏源域观测；数据生成不绑定真实 PS 点位。

旧固定 PS 布局示例：[qianhai7_sample_v1_example.json](qianhai7_sample_v1_example.json)。该文件只用于旧字段迁移参考，不是新的完整桥面场生产契约。

## 2. 数据角色与训练边界

同一份数据可以被保存，但必须按用途区分三种视图，避免训练时把不可部署信息混入学生输入：

### `source_view`：由完整 FEM 桥面场动态生成的带标签点集源域

规范 HDF5 只保存完整 FEM 桥面位移场、桥面拓扑、环境条件和状态标签。训练 loader 每次采样可变数量的虚拟传感点，通过样本相关 `H_space(S)` 插值 UZ，再施加参考基准、随机失活、噪声和质量退化，形成包含 `vertical_disp`、坐标、`valid_mask`、质量、温度/日期条件的 `source_view`。同一 FEM 样本可以产生多个不同布局和观测实现。

### `target_view`：真实垂向 InSAR 目标域

包含真实 InSAR 的垂向位移、点位、质量、缺测、温度和日期条件。目标域可以在无标签条件下参与边缘域适配；如果有事件级或区域级弱证据，只能按证据粒度参与验证或受控条件适配。目标域窗口必须保存用途：

```text
adaptation  # 可用于无标签域适配或高置信度条件适配
validation  # 用于选超参数、阈值和早停
final_test  # 方案冻结后只用于一次最终评估
```

同一目标域窗口不能同时承担上述三种用途。`final_test` 不得参与标准化统计量、伪标签阈值、域适配权重或模型选择。

### `teacher_view`：训练期特权 FEM 动力视图

可包含与 `structural_state_id` 关联的 FEM 静态位移和动态加速度/速度时程，仅供特权教师学习状态、区域或关系表示。它不能与真实 InSAR 直接做 JDA/MMD/CORAL，也不能作为学生部署端输入。若动力边界、激励和阻尼不可信，该视图可以为空，主线必须仍由 `source_view -> target_view -> InSAR 学生模型` 完成。

域适配的唯一对齐对象是 `source_view` 和 `target_view` 经过同一点集时序接口得到的布局无关表示。两域不要求点数、坐标或点 ID 相同；不得直接用 FEM 全场节点位移或动态加速度与目标 InSAR 对齐。

## 3. 文件组织

建议每个“结构状态 + 温度轨迹”使用一个自包含 HDF5：

```text
qh7_state_000001.h5         # 网格、完整桥面场、标签、环境条件和内嵌 manifest
```

不要把 FEM 节点坐标和完整位移数组嵌入 JSON。manifest 内嵌在 HDF5；如需导出 JSON，只导出不含大数组的审计副本。

## 4. JSON 必须保存的内容

1. `schema_version`、`data_version`、`sample_id`、`bridge_id`、`structural_state_id`。
2. 坐标系、单位、X/Y/Z 轴定义和垂向形变正负约定。
3. 仿真温度轨迹、日期和时间特征；真实 InSAR 日期由目标域数据单独保存。
4. FEM 网格文件、节点坐标、单元连接和版本哈希。
5. 支座位移、支承刚度、桥面局部沉降和局部刚度等状态参数。它们只能作为源域标签或审计信息，不能输入部署端学生模型。
6. FEM 全场静态位移、可选动态振动的 HDF5 路径、shape、dtype 和单位。
7. 桥面节点/单元集合、垂向分量定义，以及动态虚拟传感采样器的版本和默认参数。
8. 训练观测实现若被缓存，则保存其虚拟点坐标、`space_mapping`、插值权重、mask、噪声种子和观测算子版本。
9. 区域异常标签、严重度、标签来源和真实事件标签状态。
10. 原始 InSAR、传感器、FEM 模型和生成器来源。
11. 每个数组的 shape、dtype 和 SHA-256。
12. `source_view`、`target_view`、`teacher_view` 的角色声明、字段白名单和可变点数约束。
13. 目标域窗口的 `usage`（`adaptation`、`validation` 或 `final_test`）、划分依据、标准化统计量来源和是否参与伪标签生成。
14. `structural_state_id` 与静态轨迹、温度工况、激励工况和动力响应的关联关系；教师视图不得只靠文件名推断对应状态。

### v12 细粒度状态族标签

当前生成器 schema `qianhai7_fem_deck_field_v12` 已将每期的
`/labels/state_family_by_acquisition` 固定为细粒度工程状态族，并以
`/labels/state_family_code_by_acquisition` 和
`/labels/state_family_vocabulary` 提供稳定编码。`scenario_family` 仍表示整条
轨迹的粗粒度采样组合；旧语义保存在
`/labels/coarse_state_family_by_acquisition`，不能与细粒度标签混用。

支座组与细粒度族的映射为：

- 仅 `pier_1_foundation_a`：`pier_1_foundation_a_settlement`；
- 仅 `pier_1_foundation_b`：`pier_1_foundation_b_settlement`；
- 1#墩两个独立基础组同时等量运动：`multi_pier_settlement`；
- 多个墩组幅值不一致：`differential_pier_settlement`；
- 仅 `pier_2_foundation`：`pier_2_settlement`；
- 任一桥台基础组：`north_or_end_support_vertical_shift`；
- 局部桥面区域：`deck_local_settlement`；
- 支承刚度变化：`support_or_foundation_stiffness_reduction`；
- 材料、沉降和刚度同时启用时，在基础族名称前加
  `material_degradation_and_`，并在沉降族后追加刚度族。

其中 `pier_1_foundation_a` 和 `pier_1_foundation_b` 在网格、节点组、沉降施加和
标签中保持独立；只有在计划级区域统计时才可将二者汇总为 1# 墩区域。窗口目标
读取窗口末期的细粒度 `target_state_family` 和对应编码，训练/验证/测试应按
`structural_state_id` 或 `state_family_id` 进行状态级切分。

旧示例 JSON 使用固定点位，仅供字段迁移参考。新的仿真源域生产数据使用 26 个连续年度温度子步和完整桥面场；训练 loader 从完整轨迹滚动切出 4、6、8 期窗口，并在每次读取时动态生成可变数量的虚拟传感点。真实目标域保留全部现有 PS 点和32个不规则 InSAR 日期，通过实际 mask 和质量字段进入同一模型接口，不要求与26个 FEM 子步或虚拟点逐一对应。

## 5. HDF5 建议数据集

```text
/mesh/node_coords                    [N_node, 3]
/mesh/element_nodes                  [N_element, N_conn_max]  # 按原始周界顺序，空位填充
/mesh/element_node_count             [N_element]
/mesh/deck_node_ids                  [N_deck_node]
/mesh/deck_node_mask                 [N_node]
/mesh/deck_element_ids               [N_deck_element]
/responses/fem/disp_at_acquisition  [N_substep=26, N_node, 3]
/responses/fem/deck_uz_at_acquisition [N_substep=26, N_deck_node]  # 可选冗余加速缓存
/responses/fem/disp_static           [N_node, 3]
/responses/fem/dynamic_acc           [T_dynamic, N_sensor, 3]  # 可选
/links/structural_state_id            [N_trajectory]
```

`target_window_usage` 属于独立的真实目标域协议，不写入 FEM 源域 HDF5。

规范样本不必保存真实 PS 或固定 `space_mapping`。训练缓存如需落盘，`space_mapping` 和 `space_weights` 应与当前虚拟布局 ID 绑定，并保存壳单元实际节点数及形函数权重；不能假设所有单元固定为4节点，也不能通过排序节点号破坏周界顺序。

## 6. 有限元计算步骤

### 动力响应的定位

动力响应只由 FEM 生成，不要求与真实 InSAR 日期一一对应，也不应伪装成目标域输入。它属于训练期教师的特权信息：教师使用同一 `structural_state_id` 关联的静态/动力响应，状态参数用于监督、分组和审计；学生部署时只使用 InSAR。动力响应的时间轴是激励试验时间轴，InSAR 的时间轴是低频卫星获取日期，两者不做逐时相配对。

教师默认从 FEM 响应中学习状态。只有在单独设计并标注的特权输入消融中，才允许把状态参数送入教师。任何情况下都不能把它们送入最终学生模型。

但是，动力计算必须从数据协议第一版就纳入设计。每个 FEM 样本至少要有统一的 `structural_state_id`，并可关联当前这条年度温度轨迹、可选的 `excitation_id` 和 `dynamic_response_id`。当前实现不启用同一状态多温度案例；若未来开展跨温度消融，再增加 `temperature_case_id` 的一对多关联，不能把独立振动序列事后拼到静态样本上。

### A. 模型准备

1. 读取 `qianhai7_bridge_model_v3_shell.mac`，统一 N-mm-s 单位。
2. 核对三跨 50+80+50 m、桥宽、墩台位置、支座位置和 FE 坐标原点。
3. 输出完整桥面壳网格的节点坐标和单元连接。
4. 导出稳定的桥面节点/单元 component；将 JGC、Q、W、S 和全部现有 InSAR 点坐标转换到同一 FE 坐标系，真实点只作为目标域资产，不写入每个源域样本。

### B. 桥面集合与通用点位映射

1. 从 MAPDL component 或经审计的单元集合确定桥面壳单元，不能把全模型节点等同于桥面节点。
2. 保留壳单元原始周界节点顺序，并验证投影面积、连通性和坐标单位。
3. 实现接受任意传感点坐标的 `H_space(S)`，计算包含单元、形函数权重、映射类型和距离。
4. 用已知解析位移场和未见随机点布局做插值误差检查。
5. 真实 PS 全部保留；deck 外点通过 mask、最近映射距离和用途字段管理，不能无条件当作桥面响应。

### C. 基准健康状态

1. 施加自重、恒载和实际边界条件。
2. 在多个温度场下完成热-静力分析。
3. 输出每个温度工况的全场三维位移。
4. 在 JGC/Q/W/S 点输出可比较的位移分量。
5. 保存求解收敛信息、边界条件和温度场版本。

### D. 参数化异常状态

每次只改变一类主要因素，并记录参数：

1. 支座水平位移；
2. 支座竖向沉降；
3. 桥墩不均匀沉降；
4. 局部桥面或梁段刚度降低；
5. 支承刚度变化；
6. 组合异常状态。

每种状态使用多个异常幅值，并让整套数据覆盖不同季节温度子步。当前一个状态只对应一条 26 子步年度温度轨迹；同一状态的多个独立温度案例留作后续温度不确定性消融。健康状态、弱异常、明显异常和组合异常都必须覆盖，不能只生成危险状态。

### E. 归档完整桥面场并动态生成训练观测

1. 数据生成器保存完整 FEM 三维位移、桥面节点/单元集合和可选桥面 UZ 加速缓存，不读取真实 PS 文件。
2. 训练 loader 每次采样 `K` 个虚拟传感点；`K` 可变并覆盖最终真实 PS 稀疏度。
3. 用 `H_space(S)` 从桥面场插值 UZ，进行参考日期/参考区域差分。
4. 在 loader 中加入点随机失活、空间成片失活、时相缺失、异方差噪声和基准偏差，并显式输出坐标、`valid_mask` 和质量。
5. 仿真源域按26个年度温度子步保存日期、月份、年内日序和 `delta_days`；真实 InSAR 的32个不规则时相在目标域按自身日期窗口化，不做逐期一一匹配。
6. 推理时关闭虚拟点采样，改用全部现有真实 PS 坐标、垂向观测和实际 mask，调用同一个点集时序模型。

### F. 动力计算：仅作为条件性扩展

没有与 InSAR 同日期的实测振动时，不应把动力分支作为主线。只有在边界条件、激励和阻尼具有可信依据时，才将其作为不可部署的源域特权信息扩展。建议步骤为：

1. 模态分析，检查频率和振型；
2. 参数化支承/刚度状态下的模态变化；
3. 选定激励和阻尼，进行瞬态响应；
4. 输出少量动力节点的加速度/速度时程；
5. 将动态响应与同一 `structural_state_id` 关联，作为训练期教师特权信息保存，不作为 InSAR 学生部署输入。

不应把“没有实测同步振动”作为不计算动力响应的理由；真正的限制是：如果前海7号动力边界、激励和阻尼没有可靠依据，动力分支只能作为受控仿真教师消融，不能宣称它代表真实桥梁动力状态。主论文应同时报告“无动力教师”和“有动力教师”两种结果。

## 7. 生成器代码应如何重构

不要继续在旧 `dataset_generator_v2.py` 中增加前海7号特例。建议拆成：

```text
qianhai7_config.py
qianhai7_ansys_adapter.py
qianhai7_state_sampler.py
observation_operator.py
virtual_sensor_sampler.py
qianhai7_dataset_writer.py
audit_qianhai7_dataset.py
```

生成循环应为：

```text
sample state parameters
  -> restore qianhai7 DB
  -> apply temperature and state boundary conditions
  -> solve thermal-static FEM
  -> export full-field displacement
  -> export audited deck node/element sets
  -> write self-contained HDF5 + manifest + hashes

training loader:
  read full deck field
  -> sample variable virtual sensor layout
  -> interpolate UZ + add documented dropout/noise
  -> return coordinates + values + mask + quality
```

## 8. 与旧 JSON 的兼容策略

旧 `4.0-temperature-state-json` 数据保留为旧桥梁基线，不要与当前 qianhai7 源域混合训练。加载器应根据 `schema_version` 分流：

- `4.0-temperature-state-json`：读取 `responses.static.disp`、`responses.dynamic.ace`；
- `qianhai7_fem_counterfactual_vertical_insar_v5`：可读取26子步 FEM 全场；旧 `/responses/insar_like/*` 仅作为派生缓存忽略，不再作为新主线输入。
- 新全场归档协议：读取完整桥面场和桥面集合，由 loader 动态构造点集训练视图；正式 schema 名称在生成器实现时冻结。

标签字段也要分开。旧的材料 1–43 和旧支座节点不能迁移成前海7号的结构状态标签。

## 9. 新全场归档执行协议

生产样本单位为“一个结构状态、一条26子步年度状态轨迹”，不再把某次 PS 布局或噪声实现固化为 FEM 样本身份；当前每条轨迹只有一个温度案例。模型样本由 `window_id + observation_realization_id` 表示，窗口长度支持4、6、8个子步，虚拟点坐标、位移、温度、日期、`delta_days`、mask 和 quality 必须同步生成与切片。

温度轨迹按全年平均气温曲线生成 26 个连续子步，并记录参考温度、随机扰动参数、来源和有效掩码。当前一个 `structural_state_id` 对应一条温度轨迹；“同一状态多个温度案例”属于后续温度不确定性消融，不是当前数据事实。未知温度使用 `null` 与 mask 表示。

HDF5 统一使用：

```text
/responses/fem/disp_at_acquisition              [26,N_node,3]
/mesh/deck_node_ids                             [N_deck_node]
/mesh/deck_node_mask                            [N_node]
/mesh/deck_element_ids                          [N_deck_element]
/responses/fem/dynamic_acc                      [T_dynamic,N_sensor,3]  # 可选归档
/responses/fem/modal_frequencies                [N_mode]               # 可选归档
/responses/fem/mode_shapes                      [N_mode,N_node]        # 可选归档
```

全场数组既是审计事实，也是动态虚拟观测的唯一源数据。训练时先从26期完整轨迹切出窗口，再对桥面场采样可变 `K` 点，形成 `[W,K]` 位移、`[K,3]` 坐标和 `[W,K]` mask；部署时 `K=P_real`，使用现有 PS。动态数组必须标记 `teacher_only=true` 和 `paired_with_insar_dates=false`。

