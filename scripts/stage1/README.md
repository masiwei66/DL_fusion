# 阶段 1：严格基线执行包

阶段 1 的目标是回答两个问题：

1. 在冻结的结构状态级 split 上，静态、动态和融合模型是否能泛化；
2. 融合模型是否稳定优于先验、传统特征和单模态深度基线。

阶段 1 不修改 `DL_model` 的训练逻辑。`run_stage1.py` 只负责校验数据、固定种子、隔离输出目录、调用现有训练入口并汇总结果。

## 执行前检查

必须准备：

- 与阶段 0 完全相同的数据目录；
- 已冻结的 `split_stage1.json`（推荐由 `scripts/make_splits.py` 生成）；
- 数据文件 SHA-256 不得变化；
- 正式训练在 GPU 机器执行，本地只做 dry-run 或短 epoch 链路检查。

生成新 split（仅在数据尚未冻结时执行一次）：

```powershell
.\.venv\Scripts\python.exe scripts\make_splits.py `
  E:\working\DL_data\temperature_multistep_json_result `
  --output E:\working\DL_data\temperature_multistep_json_result\split_stage1.json `
  --group-by structural_state_id --seed 42
```

## 先做 dry-run

该命令只校验 manifest 并生成执行计划，不启动训练：

```powershell
.\.venv\Scripts\python.exe scripts\stage1\run_stage1.py `
  E:\working\DL_data\temperature_multistep_json_result `
  --split-manifest E:\working\DL_data\temperature_multistep_json_result\split_stage1.json `
  --output-dir E:\working\DL_data\report\stage1_v2 `
  --dry-run
```

## 推荐执行顺序

### 1. 一轮深度链路检查

先只检查三种深度模型能否完成一轮训练：

```powershell
.\.venv\Scripts\python.exe scripts\stage1\run_stage1.py `
  <data_dir> --split-manifest <split_stage1.json> `
  --output-dir <stage1_smoke_dir> --skip-prior --skip-traditional `
  --deep-epochs 1 --batch 2
```

### 2. 正式阶段 1

默认执行：

- prevalence baseline；
- static/dynamic response feature + random forest；
- `static_only`、`dynamic_only`、`fusion`；
- 种子 `13,29,42,71,101`；
- 每个种子独立目录。

```powershell
.\.venv\Scripts\python.exe scripts\stage1\run_stage1.py `
  <data_dir> --split-manifest <split_stage1.json> `
  --output-dir <stage1_v2> --data-version v2_YYYYMMDD_Nxxxx
```

阶段 1 的完整训练预算由 `DL_model/DL_config.py` 控制；如需小规模试跑，可显式传 `--deep-epochs`、`--batch` 和 `--lr`。

如需补充 feature-concat 基线，在正式运行命令中加入 `--include-concat`。它会复用同一 seed 下的单模态 checkpoint 初始化两个编码器，输出写入 `deep/seed_*/logs/concat_fusion/`，不会覆盖原有三类模型。

训练完成后，必须显式导出验证集预测，供后处理基线确定阈值：

```powershell
foreach ($seed in 13,29,42,71,101) {
  foreach ($model in "static_only","dynamic_only","fusion") {
    .\.venv\Scripts\python.exe DL_model\main.py `
      --mode test --model $model --seed $seed --eval-split val `
      --data-dir <data_dir> --split-manifest <split_manifest> `
      --data-version v2_stage1_N3500 --output-dir <stage1_v2>\deep\seed_$seed `
      --resume <stage1_v2>\deep\seed_$seed\checkpoints\$model\${model}_best.pt
  }
}
```

随后运行固定分数融合和任务路由基线：

```powershell
.\.venv\Scripts\python.exe scripts\stage1\posthoc_baselines.py `
  <val_export_dir> --test-stage1-dir <stage1_v2> `
  --output <writable_followup_dir>\posthoc_baselines.json
```

结构状态级 paired bootstrap（例如动态模型 vs 当前融合模型）：

```powershell
.\.venv\Scripts\python.exe scripts\stage1\state_bootstrap.py `
  <stage1_v2>\deep\seed_42\logs\dynamic_only\dynamic_only_test_predictions.json `
  <stage1_v2>\deep\seed_42\logs\fusion\fusion_test_predictions.json `
  --iterations 2000 --output <stage1_v2>\bootstrap_seed42_dynamic_vs_fusion.json
```

### 3. 重新汇总

若训练已完成但只需要重算汇总：

```powershell
.\.venv\Scripts\python.exe scripts\stage1\aggregate_results.py <stage1_v2>
```

## 输出结构

```text
stage1_v2/
  stage1_plan.json
  stage1_job_report.json
  stage1_results.json
  baseline/prior_baseline.json
  traditional/static/seed_*/report.json
  traditional/dynamic/seed_*/report.json
  deep/seed_*/logs/<model>/<model>_test_predictions.json
  deep/seed_*/checkpoints/<model>/<model>_best.pt
```

`stage1_results.json` 的主比较指标是测试集材料 `macro_auprc`，同时汇总 macro-F1、micro-F1 和 Exact Match。深度模型还会汇总支座分类、支座沉降 MAE、区域风险和全桥风险指标。测试集阈值使用模型训练流程保存的验证集阈值；不得根据测试结果重新调阈值。

## 阶段 1 判读规则

- 先看 5 个 seed 的均值和离散程度，不看单次最好结果；
- 融合模型只有在严格结构状态 split 下稳定优于最佳单模态，才进入阶段 2；
- 若融合不稳定，先按结构状态、质量指标和模态错误样本做分析，不直接增加模型复杂度；
- 阶段 1 的 checkpoint、日志和指标不能与阶段 0 结果混用。
