# Qianhai7 Training Framework

This package is the training entry point for the v3 Qianhai7 archive and its
four-task, four-level acquisition labels.
It follows the `new_plan` contract:

```text
FEM full-field HDF5
    -> virtual observation operator
    -> shared variable-point window batch
    -> interchangeable student model
    -> source supervision / observation reconstruction / domain adaptation
```

The FEM archive is the source of truth. Models never read state parameters or the
full FEM field directly. The loader derives an InSAR-like view from `UZ`, coordinates,
dates, temperature, quality and masks. A future real-InSAR adapter must return the
same `WindowBatch` interface.

## Layout

```text
qianhai7_training/
  config.py
  contracts.py
  data/       # archive index, HDF5 adapter, observation view, windows, dataset
  models/     # stable model interface and registry
  training/   # losses and model-agnostic engine
  cli/        # audit and training entry points
configs/
  qianhai7_v12_baseline.yaml
```

The important seam is `WindowBatch`: the neural comparison models consume the same
node-level feature contract `[B, W, P, 9]`. LSTM/GRU/Transformer/ResNet apply a
masked, no-graph node pooling operation before their temporal encoder; GNN-LSTM
instead propagates information over the coordinate kNN graph before pooling.
The regional-statistics adapter remains available through `input_mode: regional`
for ablation studies.

```powershell
python -m qianhai7_training.cli.audit_dataset --data-root E:\working\DL_data\qianhai7_yearly_temperature_dataset_v3_final
python -m qianhai7_training.cli.train --data-root E:\working\DL_data\qianhai7_yearly_temperature_dataset_v3_final
```

The four targets are `deck_material`, `pier_material`, `pier_settlement`, and
`bearing_stiffness`. Each sample uses the label at the window's terminal
acquisition, encoded as `0=safe`, `1=warning`, `2=risk`, `3=danger`. The shared
PyTorch model contract returns four logits for each task.

The implemented PyTorch comparisons are `mlp`, `resnet1d`, `resnet50_1d`,
`lstm`, `gru`, `transformer`, and `gnn_lstm`. The GNN baseline is a
dependency-free dynamic coordinate kNN-GCN followed by an LSTM; it uses the
same `WindowBatch` input contract and four regional classification heads.
`xgboost` is supported through a tabular adapter. It summarizes each observation
window into fixed-width statistics and trains one four-class estimator per task.
Select it by setting `model.name: xgboost` and put XGBoost parameters under
`model.params` (for example `n_estimators`, `max_depth`, and `learning_rate`).

The current archive has 26 acquisition steps. The default primary window is 6;
window lengths 4 and 8 remain experiment variants. Splits are by
`structural_state_id`, never by derived windows.

When `observation_realizations` is greater than one, training uses
`realization_sampling: balanced_random` by default: every epoch contains a
balanced mixture of point/noise realizations while keeping one pass over the
training windows. Validation and test loaders evaluate all
`validation_realizations` (defaulting to the training realization count), so
checkpoint selection measures expected performance across observation layouts
instead of a single realization. The old one-realization-per-epoch behavior is
available only as `realization_sampling: rotate` for controlled comparisons.

Select a model in the experiment YAML's `model.name`. All implemented PyTorch
models share the same `WindowBatch` input and four-head output contract.
