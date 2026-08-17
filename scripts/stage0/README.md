# 阶段 0 实验说明

阶段 0 用于排除数据读取、训练链路、标签错位和真值泄漏问题，不用于比较正式模型性能。

## 前置条件

先对完整数据生成固定的结构状态级划分。脚本会在保持 ``structural_state_id``
互斥的前提下平衡材料标签分布；重复的源 ``sample_id`` 会按激励或文件名生成唯一
记录 ID，并保存在 ``source_sample_id`` 中供追溯。正常实验和负控实验必须使用相同
的 train/val/test 样本：

```powershell
python scripts/make_splits.py E:/working/DL_data/data_new `
  --output E:/working/DL_data/data_new/split_stage0.json `
  --group-by structural_state_id --seed 42
```

确认 manifest 的 train、val、test 均非空，并且结构状态不跨集合。

## 推荐统一执行

```powershell
python scripts/stage0/run_stage0.py E:/working/DL_data/data_new `
  --split-manifest E:/working/DL_data/data_new/split_stage0.json `
  --work-dir E:/working/DL_data/stage0_run_20260813 `
  --model static_only --seeds 13,29,42
```

最终查看 `stage0_report.json`。任何子实验失败时总报告均为 `passed: false`，且进程返回非零退出码。

## 各实验的边界

- **0.1 单 batch**：检查必需字段、精确 shape/dtype、有限值、标签范围、样本 ID，并执行 forward/backward。
- **0.2 tiny overfit**：8-16 个样本全部用于训练，只优化材料分类；关闭增强、EMA、平滑、正则和辅助任务。
- **0.3 标签负控**：以 split manifest 的分组字段（默认 ``structural_state_id``）为置换单位，仅置换 train 的监督；同一状态的全部激励共享同一 donor 状态监督，val/test 保留真实标签。默认完整置换多任务监督束，诊断训练只使用材料标签。
- **0.4 防泄漏**：模型只能接收观测输入白名单；`raw_sf`、支座沉降和各级 target 不传入 forward。

## 重要备注

1. `make_smoke_dir.py` 和 `shuffle_labels.py` 拒绝写入非空目录，避免旧 `result_*.json` 污染实验。
2. 紧凑 JSON 若通过相对 `array_store.file` 引用 HDF5，脚本会优先创建硬链接，失败时复制。
3. `shuffle_labels.py --supervision material_labels` 只能配合关闭全部辅助损失的训练；普通多任务训练应使用默认 `all`。
4. 负控建议至少运行 3 个置换种子。总报告会给出 validation/test gap 的均值和范围；单次验收阈值仍需在实验前固定。
5. ``stage0_permutation_audit.json`` 会记录组级 donor 错排、逐材料标签匹配率/相关系数以及 val/test 文件 SHA-256 核验结果。
6. 阈值必须在查看实验结果前确定。不得通过事后降低过拟合阈值或放宽负控容差来制造“通过”。
7. 阶段 0 的 checkpoint 和指标不得与正式主实验结果混用。
