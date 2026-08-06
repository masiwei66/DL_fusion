"""Quick single-file inference helper.

Edit `file_path` and run this script after training a model.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DL_config import Config
from dataset import StructuralDataset
from predict import load_model, predict_one, print_result


file_path = "E:/working/DL_data/modal_updating_result/result_000000.json"


def main():
    full_path = os.path.abspath(file_path)
    if not os.path.exists(full_path):
        print(f"Input file not found: {full_path}")
        return

    cfg = Config()
    ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    dataset = StructuralDataset(cfg.data_dir, normalize=True, fit_normalizer=False)
    dataset.apply_to_config(cfg)
    model, ckpt = load_model(ckpt_path, cfg)
    dataset.apply_to_config(cfg)
    dataset.load_normalizer_state_dict(ckpt.get("normalizer"))
    device = torch.device(cfg.device)
    model = model.to(device)

    with open(full_path, encoding="utf-8") as f:
        raw = json.load(f)
    result = predict_one(
        model,
        raw,
        dataset,
        device,
        thresholds=ckpt.get("eval_thresholds"),
        return_full=True,
    )
    print_result(full_path, result)


if __name__ == "__main__":
    main()
