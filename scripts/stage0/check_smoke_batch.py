"""阶段 0.1 + 0.4：batch 数据契约、forward/backward 与防泄漏检查。

备注：dataset 的 batch 可以同时携带输入和监督，但模型 forward 必须只接收
``stage0_common.MODEL_INPUT_KEYS`` 白名单中的观测量。仅检查黑名单键名不足以证明
没有泄漏，因此本脚本会剥离全部监督字段后再执行模型检查。
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "DL_model"))

import torch
import torch.nn.functional as F

from DL_config import Config
from dataset import StructuralDataset
from model import build_model
from trainer import make_loader
from stage0_common import (
    MODEL_INPUT_KEYS,
    TARGET_BATCH_KEYS,
    report_envelope,
    sanitized_model_input,
    save_json,
)


REQUIRED_KEYS = {
    "disp", "strain", "ace", "target", "raw_sf", "support_disp",
    "support_target", "region_target", "global_target", "condition",
    "temperature_C", "temperature_steps_C", "quality_metrics",
    "static_pos", "dynamic_pos", "metadata",
}


def _shape(value):
    return list(value.shape)


def check_batch(batch, dataset):
    """验证必需字段、shape、dtype、有限值、标签范围和样本 ID。"""
    problems = []
    details = {"keys": sorted(batch), "tensors": {}}
    missing = sorted(REQUIRED_KEYS - set(batch))
    if missing:
        problems.append(f"缺少必需 batch 字段: {missing}")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value).all().item())
            details["tensors"][key] = {
                "shape": _shape(value), "dtype": str(value.dtype), "finite": finite,
            }
            if not finite:
                problems.append(f"{key} 含 NaN/Inf")

    if missing:
        return problems, details

    batch_size = batch["disp"].shape[0]
    expected = {
        "disp": [batch_size, dataset.static_steps, dataset.n_static_nodes, dataset.static_channels],
        "strain": [batch_size, dataset.static_steps, dataset.n_static_nodes, dataset.static_channels],
        "ace": [batch_size, dataset.dynamic_steps, dataset.n_dynamic_nodes, dataset.dynamic_channels],
        "target": [batch_size, len(dataset.targets[0])],
        "raw_sf": [batch_size, len(dataset.raw_sf[0])],
        "support_disp": [batch_size, len(dataset.support_disp[0])],
        "support_target": [batch_size, len(dataset.support_targets[0])],
        "region_target": [batch_size, len(dataset.region_targets[0])],
        "global_target": [batch_size],
        "condition": [batch_size, 1],
        "temperature_C": [batch_size],
        "temperature_steps_C": [batch_size, dataset.static_steps],
        "quality_metrics": [batch_size, len(dataset.quality_feature_names)],
        "static_pos": [dataset.n_static_nodes, 3],
        "dynamic_pos": [dataset.n_dynamic_nodes, 3],
    }
    # mask 是阶段 2 才会成为 dataset 的稳定输出；若当前数据已提供，则阶段 0
    # 必须检查其 batch 维和 0/1 取值，不能静默接受错位的 mask。
    optional_masks = {key: batch[key] for key in ("static_mask", "dynamic_mask", "temperature_mask") if key in batch}
    for key, shape in expected.items():
        if _shape(batch[key]) != shape:
            problems.append(f"{key} shape={_shape(batch[key])}，期望 {shape}")

    float_keys = {
        "disp", "strain", "ace", "target", "raw_sf", "support_disp",
        "support_target", "condition", "temperature_C", "temperature_steps_C",
        "quality_metrics", "static_pos", "dynamic_pos",
    }
    long_keys = {"region_target", "global_target"}
    for key in float_keys:
        if batch[key].dtype != torch.float32:
            problems.append(f"{key} dtype={batch[key].dtype}，期望 torch.float32")
    for key in long_keys:
        if batch[key].dtype != torch.int64:
            problems.append(f"{key} dtype={batch[key].dtype}，期望 torch.int64")

    for key in ("target", "support_target"):
        if not bool(((batch[key] == 0) | (batch[key] == 1)).all().item()):
            problems.append(f"{key} 必须是 0/1 标签")
    for key in ("region_target", "global_target"):
        values = batch[key]
        if values.numel() and (values.min().item() < 0 or values.max().item() >= 4):
            problems.append(f"{key} 必须位于 [0, 3]")
    for key, values in optional_masks.items():
        if values.shape[0] != batch_size:
            problems.append(f"{key} batch 维 {values.shape[0]} != {batch_size}")
        if not bool(((values == 0) | (values == 1)).all().item()):
            problems.append(f"{key} 必须是 0/1 mask")

    metadata = batch["metadata"]
    if len(metadata) != batch_size:
        problems.append(f"metadata 数量 {len(metadata)} != batch size {batch_size}")
    sample_ids = [str(item.get("sample_id", "")) for item in metadata]
    if any(not value for value in sample_ids):
        problems.append("metadata 中存在空 sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        problems.append("单 batch 中 sample_id 不唯一")
    details["sample_ids"] = sample_ids
    details["expected_shapes"] = expected
    return problems, details


def check_input_boundary(batch):
    """确认净化后的模型输入不包含任何监督字段。"""
    model_input = sanitized_model_input(batch)
    problems = []
    leaked = sorted(set(model_input) & TARGET_BATCH_KEYS)
    unknown = sorted(set(model_input) - MODEL_INPUT_KEYS)
    if leaked:
        problems.append(f"模型输入仍含监督字段: {leaked}")
    if unknown:
        problems.append(f"模型输入含未登记字段: {unknown}")
    if not {"disp", "strain", "ace", "static_pos", "dynamic_pos"} <= set(model_input):
        problems.append("模型输入缺少基础观测量或坐标")
    return problems, {"model_input_keys": sorted(model_input), "removed_keys": sorted(set(batch) - set(model_input))}


def check_models(batch, check_backward=False):
    """仅用白名单输入检查三个模型；可选验证梯度有限且参数发生更新。"""
    results = {}
    problems = []
    base_input = sanitized_model_input(batch)
    target = batch["target"]
    for model_type in ("static_only", "dynamic_only", "fusion"):
        cfg = Config()
        cfg.set_model_type(model_type)
        cfg.n_materials = target.shape[1]
        cfg.n_supports = batch["support_target"].shape[1]
        cfg.n_regions = batch["region_target"].shape[1]
        cfg.static_steps = batch["disp"].shape[1]
        cfg.static_channels = batch["disp"].shape[3]
        cfg.n_static_nodes = batch["disp"].shape[2]
        cfg.dynamic_channels = batch["ace"].shape[3]
        cfg.n_dynamic_nodes = batch["ace"].shape[2]
        model = build_model(cfg)
        model.train(check_backward)
        inputs = {key: value for key, value in base_input.items() if isinstance(value, torch.Tensor)}
        before = None
        optimizer = None
        if check_backward:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
        output = model(inputs)
        bad_outputs = [
            key for key, value in output.items()
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item()
        ]
        entry = {"output_shapes": {key: _shape(value) for key, value in output.items() if isinstance(value, torch.Tensor)}}
        if bad_outputs:
            problems.append(f"{model_type} 输出含 NaN/Inf: {bad_outputs}")
        if check_backward:
            loss = F.binary_cross_entropy_with_logits(output["material_logits"], target)
            optimizer.zero_grad()
            loss.backward()
            bad_grads = [
                name for name, value in model.named_parameters()
                if value.grad is not None and not torch.isfinite(value.grad).all().item()
            ]
            grad_count = sum(value.grad is not None for value in model.parameters())
            optimizer.step()
            changed = sum(
                not torch.equal(before[name], value.detach())
                for name, value in model.named_parameters()
            )
            entry.update({"loss": float(loss.item()), "gradient_parameter_count": grad_count, "updated_parameter_count": changed})
            if bad_grads:
                problems.append(f"{model_type} 梯度含 NaN/Inf: {bad_grads}")
            if grad_count == 0 or changed == 0:
                problems.append(f"{model_type} backward 后没有参数更新")
        results[model_type] = entry
    return problems, results


def main():
    parser = argparse.ArgumentParser(description="阶段0.1/0.4：严格 batch 契约与防泄漏检查")
    parser.add_argument("data_dir", help="result_*.json 所在目录")
    parser.add_argument("--samples", type=int, default=8, help="读取前 N 个样本组成检查 batch")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--check-model", action="store_true", help="用净化后的输入检查三个模型 forward")
    parser.add_argument("--check-backward", action="store_true", help="同时检查梯度和参数更新（隐含 --check-model）")
    parser.add_argument("--report", default=None, help="可选 JSON 报告路径")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.corrected_data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0
    dataset = StructuralDataset(args.data_dir, normalize=True, fit_normalizer=False)
    dataset.apply_to_config(cfg)
    n = min(args.samples, len(dataset))
    if n < 1:
        raise ValueError("数据集为空")
    loader = make_loader(dataset, list(range(n)), dataset.static_pos, dataset.dynamic_pos, False, cfg)
    batch = next(iter(loader))

    problems, contract = check_batch(batch, dataset)
    boundary_problems, boundary = check_input_boundary(batch)
    problems.extend(boundary_problems)
    models = {}
    if args.check_model or args.check_backward:
        model_problems, models = check_models(batch, check_backward=args.check_backward)
        problems.extend(model_problems)

    report = report_envelope(
        "stage0.1+0.4",
        not problems,
        {"data_dir": os.path.abspath(args.data_dir), "contract": contract, "input_boundary": boundary, "models": models, "problems": problems},
        notes=[
            "forward 只接收 MODEL_INPUT_KEYS 白名单；batch 中监督字段不会传入模型。",
            "--samples 限制检查 batch，但 StructuralDataset 当前仍会读取完整元数据。",
        ],
    )
    if args.report:
        save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
