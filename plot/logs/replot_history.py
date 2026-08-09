"""从已保存的 checkpoint 重新生成训练历史曲线（无需重新训练）。

用法（在服务器项目根目录）：
    python plot/logs/replot_history.py

前提：服务器已安装 matplotlib，且 DL_model/checkpoints 下有对应 checkpoint。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "DL_model"))

import torch
from DL_config import Config
from plot import plot_history

MODEL_TYPES = ("static_only", "dynamic_only", "fusion")


def main():
    cfg = Config()
    for model_type in MODEL_TYPES:
        save_dir = Config.method_dir(cfg.save_root, model_type)
        ckpt_path = os.path.join(save_dir, f"{model_type}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"跳过 {model_type}: 找不到 checkpoint {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        history = ckpt.get("history")
        if not history:
            print(f"跳过 {model_type}: checkpoint 中没有 history")
            continue
        log_dir = Config.method_dir(cfg.log_root, model_type)
        os.makedirs(log_dir, exist_ok=True)
        out = os.path.join(log_dir, f"{model_type}_history.png")
        plot_history(history, out)
        print(f"已生成训练曲线: {out}")


if __name__ == "__main__":
    main()
