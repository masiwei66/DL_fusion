# Qianhai7 Training Framework

This package is the new training entry point for `qianhai7_yearly_temperature_dataset_v12`.
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

The important seam is `WindowBatch`: TCN, GRU/LSTM, Transformer, GNN and the final
migration model must consume this same contract.

```powershell
python -m qianhai7_training.cli.audit_dataset --data-root E:\working\DL_data\qianhai7_yearly_temperature_dataset_v12
python -m qianhai7_training.cli.train --data-root E:\working\DL_data\qianhai7_yearly_temperature_dataset_v12
```

Select the model in `configs/qianhai7_v12_baseline.yaml` by changing only
`model.name`: `mlp`, `xgboost`, `resnet1d`, `lstm`, `gru`, `transformer`, or
`gnn_lstm`. The latter six are registered interface slots; until their model
and adapter implementations are added, selecting one fails explicitly instead
of silently falling back to MLP.

The current archive has 26 acquisition steps. The default primary window is 6;
window lengths 4 and 8 remain experiment variants. Splits are by
`structural_state_id`, never by derived windows.

The first implemented comparison models are `xgboost`, `resnet1d`, and `lstm`.
Select one in `model.name`. XGBoost uses a fixed-width masked-statistics adapter;
the two PyTorch models use the same point-pooled temporal adapter and the same
`WindowBatch` input contract.
