# dataset_generator.py 数据集生成配置指南

更换桥模型后，需要修改以下内容以确保数据集与模型对应。

---

## 1. 路径配置

`dataset_generator.py` 第 18-31 行：

| 变量 | 说明 |
|------|------|
| `config_file_path` | 模型配置文件 (config.yaml) |
| `input_directory_path` | 结构输入文件目录 |
| `db_path` | ANSYS .db 文件所在目录 |
| `result_dir` | 数据集输出目录 |
| `stc.ansys.model_resume(...)` | 恢复的 .db 文件名（第31行） |

---

## 2. 材料分区

### 2.1 随机化材料列表

`dataset_generator.py` 第 94-102 行：

```python
materia_changed = [8, 9, 19, 20, 24, 41]  # 需要随机退化的材料ID
materia_list1 = [m for m in range(1, 44) if m not in materia_changed]  # 保持原值的材料
```

- `materia_changed`：可能被随机缩放的材料编号
- `materia_list1`：始终使用原始弹性模量的材料编号
- `range(1, 44)` 中的 44 取决于模型材料总数

### 2.2 随机化区域

`dataset_generator.py` 第 104-113 行：

```python
mat_region = [materia_list1, materia_list2, materia_selected]
randomize_region = [2]  # 本次迭代随机化第几个分区 (0-based)
```

- `materia_selected`：从 `materia_changed` 中随机抽取 1-6 个，**本次迭代实际退化**的材料
- `materia_list2`：`materia_changed` 中本次未被选中的材料
- `randomize_region = [2]` 表示只随机化 `mat_region[2]`（即 `materia_selected`）

### 2.3 候选材料ID（safety_rules.py）

`DL_model/safety_rules.py` 第 17 行：

```python
CANDIDATE_MATERIAL_IDS = [8, 9, 19, 20, 24, 41]
```

**必须与 `materia_changed` 保持一致**，否则安全标签与材料缩放因子不匹配。

---

## 3. 支座沉降

`safety_rules.py` 第 18-21 行：

```python
SUPPORT_SETTLEMENT_NODES = [221, 513, 395, 687]  # 支座节点编号
SUPPORT_SETTLEMENT_DOF = "UZ"                     # 沉降方向
SUPPORT_SETTLEMENT_MIN_MM = 0.0                   # 随机沉降下限 (mm)
SUPPORT_SETTLEMENT_MAX_MM = 8.0                   # 随机沉降上限 (mm)
```

随机策略（`make_support_settlement_case()`）：每次迭代从 4 个节点中随机选 1~4 个，各自赋独立随机值。

如需固定沉降值或关闭沉降，参见文档末尾"支座沉降手动控制"。

---

## 4. 荷载工况

`dataset_generator.py` 第 192-199 行：

```python
load_list = [
    {**support_settlement['load_step']},       # 子步1: 支座沉降
    {23: ("FZ", -100)},                        # 子步2: 节点23竖向力
    {67: ("FZ", -100)},                        # 子步3: 节点67竖向力
    {133: ("FZ", -100)},                       # 子步4: 节点133竖向力
    {177: ("FZ", -100)},                       # 子步5: 节点177竖向力
]
```

- 荷载节点 ID（23, 67, 133, 177）必须替换为实际模型的加载点
- 力值（-100）和方向（FZ）根据实际荷载设计确定
- 共 5 步（4 荷载步 + 支座沉降步），`import_result(steps=5, ...)` 需要与之对应

---

## 5. 结果提取节点

`dataset_generator.py` 第 205-223 行：

| 行号 | 变量 | 当前值 | 说明 |
|------|------|--------|------|
| 205 | `node_id` | `[153, 175, 197, 43, 65, 87]` | 静力位移/应变提取节点 |
| 212 | `sampling_time` | `5` | 瞬态采样时长 (s) |
| 213 | `sampling_frequency` | `200` | 瞬态采样频率 (Hz) |
| 214 | `node_ids` | `[120]` | 瞬态求解的激励节点 |
| 223 | `node_id` | `[23, 645, 67, 133, 155, 177]` | 瞬态加速度提取节点 |

> **注意**：DL_config.py 末尾定义了回退默认值 `STATIC_NODES = [43, 65, 87, 153, 175, 197]` 和 `DYNAMIC_NODES = [23, 67, 133, 155, 177, 645]`，但这些仅是回退值。实际运行时从 `result_*.json` 的 `node_maps` 字段自动读取测点，因此 dataset_generator.py 中的节点列表和 DL_model 的配置应使用相同的节点集合。

---

## 6. 安全阈值

`safety_rules.py` 第 38-44 行：

```python
material_warning = 0.90   # 刚度退化至 90% → warning
material_risk    = 0.80   # 刚度退化至 80% → risk
material_danger  = 0.70   # 刚度退化至 70% → danger
support_warning_mm = 2.0  # 沉降 ≥ 2mm → warning
support_risk_mm    = 5.0  # 沉降 ≥ 5mm → risk
support_danger_mm  = 7.0  # 沉降 ≥ 7mm → danger
```

### 空间分区映射（`safety_rules.py` 第 52-88 行）

```python
REGION_DEFINITIONS = {
    "left_support1_zone":  {"material_ids": [],        "support_nodes": [221]},
    "left_support2_zone":  {"material_ids": [],        "support_nodes": [513]},
    "right_support2_zone": {"material_ids": [],        "support_nodes": [395]},
    "right_zone":          {"material_ids": [9, 20],   "support_nodes": []},
    "mid_zone":            {"material_ids": [24, 41],  "support_nodes": []},
    "left_zone":           {"material_ids": [8, 19],   "support_nodes": []},
    "right_support1_zone": {"material_ids": [],        "support_nodes": [687]},
}
```

区域划分同时用于 dataset_generator 生成安全标签和 DL_model 训练时输出区域风险。

---

## 7. 材料随机化统计参数

`random_mat.py` 中的参数：

```python
a_miu, a_sig = -0.35, 0.2   # 整体缩放因子的对数均值与标准差
b_miu, b_sig = 0, 0.12      # 个体缩放因子的对数均值与标准差
total_shift = round(np.random.uniform(0.2, 0.9), 2)  # 实际缩放系数范围
```

---

## 8. 安全标签构建流程

`dataset_generator.py` 第 125-127 行调用 `build_safety_labels()`：

```python
safety_labels = build_safety_labels(
    material_ids=material_ids,
    material_scaling_factors=material_scaling_factors,
    support_settlement_mm=support_settlement['values_mm'],
    region_definitions=REGION_DEFINITIONS,
)
```

生成的标签包含：
- `material_labels`：6 个候选材料的二值标签（损伤/健康）
- `material_risk_levels`：6 个候选材料的四级风险等级
- `support_labels`：4 个支座的二值标签
- `support_risk_levels`：4 个支座的四级风险等级
- `region_risk_levels`：7 个区域的四级风险等级
- `global_level`：全局安全等级

---

## 9. 支座沉降手动控制

### 固定值

在 `dataset_generator.py` 中替换 `make_support_settlement_case()` 调用：

```python
support_settlement = {
    "node_ids": [221, 513, 395, 687],
    "direction": "UZ",
    "values_mm": [2.0, 4.0, 0.0, 0.0],
    "active": [1, 1, 0, 0],
    "active_node_ids": [221, 513],
    "load_step": {221: ("UZ", -2.0), 513: ("UZ", -4.0),
                  395: ("UZ", -0.0), 687: ("UZ", -0.0)},
}
```

### 完全关闭

```python
support_settlement = make_support_settlement_case(min_mm=0.0, max_mm=0.0)
```

### 调整随机范围

```python
support_settlement = make_support_settlement_case(min_mm=1.0, max_mm=5.0)
```
