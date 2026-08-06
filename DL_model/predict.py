"""Batch inference for bridge multi-condition safety assessment."""

import glob
import json
import os
import sys

# 将项目根目录加入模块搜索路径，确保绘图包 plot 可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from DL_config import (
    CANDIDATE_IDS,
    DYNAMIC_NODES,
    REGION_NAMES,
    STATIC_NODES,
    SUPPORT_NODES,
    Config,
)
from dataset import (
    _as_response_array,
    _coords_for_nodes,
    _has_response,
    _nodes_from_sample,
    _temperature_condition,
)
from model import build_model
from plot import save_prediction_figures
from safety_rules import RISK_LEVELS, SUPPORT_SETTLEMENT_MAX_MM

# —— 默认输入路径：设为 None 则必须通过命令行传参；设为路径字符串则可直接运行 ——
DEFAULT_INPUT = r"E:\working\DL_data\data_new\result_000000.json"


class PredictContext:
    """Lightweight replacement for StructuralDataset during inference.

    Reads node positions and response layout from a single JSON sample
    (no training-set scan), and takes normalizer stats from the checkpoint.
    """

    def __init__(self, sample, normalizer_state):
        node_coords = sample["node_coords"]

        self.static_node_ids = _nodes_from_sample(sample, "disp", STATIC_NODES, "disp")
        self.dynamic_node_ids = _nodes_from_sample(sample, "ace", DYNAMIC_NODES, "acc", "ace")

        self.static_pos = torch.tensor(
            _coords_for_nodes(node_coords, self.static_node_ids), dtype=torch.float32
        )
        self.dynamic_pos = torch.tensor(
            _coords_for_nodes(node_coords, self.dynamic_node_ids), dtype=torch.float32
        )

        support_meta = sample.get("support_settlement", {})
        self.support_nodes = [int(n) for n in support_meta.get("node_ids", SUPPORT_NODES)]
        safety_meta = sample.get("safety_labels", {})
        self.region_names = list(
            safety_meta.get(
                "region_names",
                sample.get("region_definitions", {}).keys() or REGION_NAMES,
            )
        )

        if normalizer_state:
            self.disp_mean = normalizer_state["disp_mean"]
            self.disp_std = normalizer_state["disp_std"]
            self.strain_mean = normalizer_state["strain_mean"]
            self.strain_std = normalizer_state["strain_std"]
            self.ace_mean = normalizer_state["ace_mean"]
            self.ace_std = normalizer_state["ace_std"]
        else:
            self.disp_mean = self.disp_std = torch.zeros(1)
            self.strain_mean = self.strain_std = torch.zeros(1)
            self.ace_mean = self.ace_std = torch.zeros(1)

    def n_static_nodes(self):
        return len(self.static_node_ids)

    def n_dynamic_nodes(self):
        return len(self.dynamic_node_ids)

    def make_input_tensors(self, raw_data):
        disp_arr = _as_response_array(raw_data, "disp", self.static_node_ids, "disp")
        if _has_response(raw_data, "strain"):
            strain_arr = _as_response_array(
                raw_data, "strain", self.static_node_ids, "strain"
            )
        else:
            strain_arr = np.zeros_like(disp_arr, dtype=np.float32)
        ace_arr = _as_response_array(raw_data, "ace", self.dynamic_node_ids, "acc", "ace")

        disp = torch.tensor(disp_arr, dtype=torch.float32)
        strain = torch.tensor(strain_arr, dtype=torch.float32)
        ace = torch.tensor(ace_arr, dtype=torch.float32)

        disp = self._normalize(disp.unsqueeze(0), self.disp_mean, self.disp_std).squeeze(0)
        strain = self._normalize(strain.unsqueeze(0), self.strain_mean, self.strain_std).squeeze(0)
        ace = self._normalize(ace.unsqueeze(0), self.ace_mean, self.ace_std).squeeze(0)
        return disp, strain, ace

    @staticmethod
    def _normalize(tensor, mean, std):
        if mean.numel() == 1:
            return tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        if mean.dim() == 1 and mean.numel() == tensor[0].numel():
            orig_shape = tensor.shape
            flat = tensor.reshape(orig_shape[0], -1)
            return ((flat - mean) / std).reshape(orig_shape)
        return (tensor - mean) / std


def _load_sample(path):
    with open(path, encoding="utf-8") as file:
        sample = json.load(file)
    sample["_sample_path"] = path
    manifest_path = os.path.join(os.path.dirname(path), "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        for key in (
            "node_coords",
            "material_nodes",
            "node_maps",
            "region_definitions",
            "response_metadata",
        ):
            if key not in sample and key in manifest:
                sample[key] = manifest[key]
    return sample


def apply_checkpoint_config(config, ckpt_config):
    for key, value in (ckpt_config or {}).items():
        if hasattr(config, key):
            setattr(config, key, value)


def load_model(checkpoint_path, config):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    apply_checkpoint_config(config, ckpt.get("config"))
    model_type = ckpt.get("config", {}).get("model_type", config.model_type)
    config.set_model_type(model_type)
    model = build_model(config)
    if "model_ema" in ckpt:
        model.load_state_dict(ckpt["model"])
        model.load_state_dict(ckpt["model_ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded model: {model_type} "
        f"(epoch {ckpt.get('epoch', '?')}, val F1={ckpt.get('best_f1', 0):.4f})"
    )
    return model, ckpt


def _make_batch(raw_data, ctx, device):
    disp, strain, ace = ctx.make_input_tensors(raw_data)
    _temperature_c, delta_temperature_c = _temperature_condition(raw_data)
    return {
        "disp": disp.unsqueeze(0).to(device),
        "strain": strain.unsqueeze(0).to(device),
        "ace": ace.unsqueeze(0).to(device),
        "static_pos": ctx.static_pos.to(device),
        "dynamic_pos": ctx.dynamic_pos.to(device),
        "condition": torch.tensor(
            [[delta_temperature_c / 20.0]], dtype=torch.float32, device=device
        ),
    }


def predict_one(model, raw_data, ctx, device, thresholds=None, return_full=False):
    """Predict one JSON sample.

    *ctx* may be a PredictContext (fast path) or a StructuralDataset (backward
    compatible, but incurs a full training-set scan).
    """

    batch = _make_batch(raw_data, ctx, device)
    with torch.no_grad():
        output = model(batch)
        material_probs = torch.sigmoid(output["material_logits"]).squeeze(0).cpu().numpy()
        thresholds_arr = np.asarray(thresholds) if thresholds is not None else np.full_like(material_probs, 0.5)
        material_preds = (material_probs > thresholds_arr).astype(int)
        support_probs = torch.sigmoid(output["support_logits"]).squeeze(0).cpu().numpy()
        support_preds = (support_probs > 0.5).astype(int)
        scale = float(SUPPORT_SETTLEMENT_MAX_MM)
        support_disp = torch.sigmoid(output["support_reg"]).squeeze(0).cpu().numpy() * scale
        region_levels = output["region_logits"].argmax(dim=-1).squeeze(0).cpu().numpy()
        material_reg = output["material_reg"].squeeze(0).cpu().numpy()

    if not return_full:
        return material_probs, material_preds

    result = {
        "material_ids": list(CANDIDATE_IDS),
        "material_probs": material_probs,
        "material_preds": material_preds,
        "material_scaling_pred": material_reg,
        "support_nodes": list(ctx.support_nodes),
        "support_probs": support_probs,
        "support_preds": support_preds,
        "support_disp_mm": support_disp,
        "region_names": list(ctx.region_names),
        "region_levels": region_levels,
        "global_level": int(output["global_logits"].argmax(dim=-1).item()),
    }
    result["ground_truth"] = _extract_ground_truth(raw_data)
    return result


def _extract_ground_truth(raw_data):
    """Safely extract ground-truth labels from a result_*.json sample.

    Returns a dict with all ground-truth fields, or empty dict if the sample
    has no safety labels (e.g. raw field data without annotations).
    """
    safety = raw_data.get("safety_labels")
    material_ids = raw_data.get("material_ids", [])
    material_sf = raw_data.get("material_scaling_factors", [])

    if safety is None and not material_ids:
        return {}

    # Build material-level ground truth for the 6 candidate materials.
    cand = CANDIDATE_IDS

    # Build id-to-index mapping if available (needed for scaling factor lookup).
    idx_map = {}
    if material_ids:
        idx_map = {int(m): i for i, m in enumerate(material_ids)}

    if safety is not None and "material_labels" in safety:
        mat_labels = safety["material_labels"]
        mat_risk = safety.get("material_risk_levels", [0] * len(cand))
    elif material_ids and material_sf is not None:
        # Fallback: compute from scaling factors when safety_labels is missing.
        mat_labels = []
        mat_risk = []
        for mid in cand:
            sf = float(material_sf[idx_map[mid]]) if mid in idx_map else 1.0
            mat_labels.append(int(sf < 1.0))
            if sf <= 0.70:
                mat_risk.append(3)
            elif sf <= 0.80:
                mat_risk.append(2)
            elif sf <= 0.90:
                mat_risk.append(1)
            else:
                mat_risk.append(0)
    else:
        return {}

    # Support ground truth.
    support = raw_data.get("support_settlement", {}) if raw_data.get("support_settlement") else {}
    support_values = support.get("values_mm", [])
    if safety is not None:
        sup_labels = safety.get("support_labels", [0] * len(support_values or [0] * 4))
        sup_risk = safety.get("support_risk_levels", [0] * len(sup_labels))
    else:
        sup_labels = [int(abs(float(v)) >= 2.0) for v in support_values]
        sup_risk = [0] * len(sup_labels)

    # Region and global labels.
    region_names_from_data = list((safety or {}).get(
        "region_names",
        raw_data.get("region_definitions", {}).keys(),
    ))
    region_risk_gt = (safety or {}).get("region_risk_levels", [0] * len(region_names_from_data or [0] * 7))
    global_level_gt = (safety or {}).get("global_level", None)

    # Material scaling factors for candidate materials.
    mat_sf_gt = None
    if material_ids and material_sf is not None:
        mat_sf_gt = [float(material_sf[idx_map[m]]) if m in idx_map else 1.0 for m in cand]

    return {
        "material_labels": mat_labels,
        "material_risk_levels": mat_risk,
        "material_sf": mat_sf_gt,
        "support_labels": sup_labels,
        "support_risk_levels": sup_risk,
        "support_values_mm": support_values if support_values else None,
        "region_names": region_names_from_data if region_names_from_data else None,
        "region_risk_levels": region_risk_gt,
        "global_level": global_level_gt,
    }


def _risk_name(level):
    return RISK_LEVELS[int(np.clip(level, 0, len(RISK_LEVELS) - 1))]


def print_result(filename, result):
    gt = result.get("ground_truth") or {}
    has_gt = bool(gt)

    print(f"\n{'=' * 64}")
    print(f"File: {os.path.basename(filename)}")
    print(f"{'=' * 64}")

    # ── Material condition ──
    header = f"{'ID':>8} {'pred':>10} {'prob':>10} {'sf_pred':>10}"
    if has_gt:
        header += f" {'true':>10} {'true_sf':>10}"
    print(f"\nMaterial condition")
    print(header)

    for i, mid in enumerate(result.get("material_ids", CANDIDATE_IDS)):
        state = "damaged" if result["material_preds"][i] else "healthy"
        line = (
            f"{mid:>8} {state:>10} {result['material_probs'][i]:>9.1%} "
            f"{result['material_scaling_pred'][i]:>10.3f}"
        )
        if has_gt:
            gt_labels = gt.get("material_labels", [])
            gt_sf = gt.get("material_sf")
            true_state = "damaged" if (i < len(gt_labels) and gt_labels[i]) else "healthy"
            true_sf = f"{gt_sf[i]:>10.3f}" if gt_sf and i < len(gt_sf) else "         -"
            line += f" {true_state:>10} {true_sf}"
        print(line)

    # ── Support settlement ──
    header = f"{'node':>8} {'pred':>10} {'prob':>10} {'pred_mm':>10}"
    if has_gt:
        header += f" {'true':>10} {'true_mm':>10}"
    print(f"\nSupport settlement condition")
    print(header)

    for i, node in enumerate(result.get("support_nodes", SUPPORT_NODES)):
        state = "risk" if result["support_preds"][i] else "normal"
        line = (
            f"{node:>8} {state:>10} {result['support_probs'][i]:>9.1%} "
            f"{result['support_disp_mm'][i]:>10.3f}"
        )
        if has_gt:
            gt_labels = gt.get("support_labels", [])
            gt_mm = gt.get("support_values_mm")
            true_state = "risk" if (i < len(gt_labels) and gt_labels[i]) else "normal"
            true_mm = f"{gt_mm[i]:>10.3f}" if gt_mm and i < len(gt_mm) else "         -"
            line += f" {true_state:>10} {true_mm}"
        print(line)

    # ── Region risk ──
    region_names = result.get("region_names", REGION_NAMES)
    header = f"{'Region':<24} {'pred':>10}"
    if has_gt:
        header += f" {'true':>10}"
    print(f"\nRegion risk")
    print(header)
    for i, name in enumerate(region_names):
        line = f"  {name:<22} {_risk_name(result['region_levels'][i]):>10}"
        if has_gt:
            gt_region = gt.get("region_risk_levels", [])
            gt_name = "?"
            if gt_region and i < len(gt_region):
                gt_name = _risk_name(gt_region[i])
            line += f" {gt_name:>10}"
        print(line)

    # ── Global level ──
    if has_gt and gt.get("global_level") is not None:
        print(f"\nGlobal safety:  pred={_risk_name(result.get('global_level', 0))}"
              f"  true={_risk_name(gt['global_level'])}")
    elif "global_level" in result:
        print(f"\nGlobal safety:  {_risk_name(result['global_level'])}")


def main():
    args = sys.argv[1:]
    if not args and DEFAULT_INPUT:
        args = [DEFAULT_INPUT]
    if not args:
        print(__doc__)
        sys.exit(1)

    files = []
    for arg in args:
        if os.path.isdir(arg):
            files.extend(sorted(glob.glob(os.path.join(arg, "result_*.json"))))
        elif os.path.isfile(arg):
            files.append(arg)
        else:
            print(f"Warning: not found, skipped: {arg}")
    if not files:
        print("No valid input files.")
        sys.exit(1)

    cfg = Config()
    checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model, ckpt = load_model(checkpoint_path, cfg)
    device = torch.device(cfg.device)
    model = model.to(device)
    thresholds = ckpt.get("eval_thresholds")

    # Build prediction context from the first input file + checkpoint normalizer.
    first_sample = _load_sample(files[0])
    ctx = PredictContext(first_sample, ckpt.get("normalizer"))
    figure_dir = os.path.join(cfg.log_root, "prediction_figures")

    for path in files:
        raw = _load_sample(path)
        result = predict_one(model, raw, ctx, device, thresholds=thresholds, return_full=True)
        print_result(path, result)
        figure_paths = save_prediction_figures(result, path, figure_dir)
        if figure_paths:
            for figure_path in figure_paths:
                print(f"Saved prediction figure: {figure_path}")
        else:
            print("Prediction figures were not generated. Please check matplotlib availability.")

    print(f"\nDone. Predicted {len(files)} file(s).")


if __name__ == "__main__":
    main()
