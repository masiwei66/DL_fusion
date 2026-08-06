"""Plot raw V2 vibration and temperature-displacement responses.

The module can be imported from Python or executed as a command-line tool.
Only response values, node maps, coordinates, temperature and sample metadata
are read. Labels and finite-element generation truth are not used as curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._config import GRID_COLOR
from ._utils import _configure_chinese_font


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73"]
DYNAMIC_COMPONENTS = ("AX", "AY", "AZ")
STATIC_COMPONENTS = ("UX", "UY", "UZ")


def load_v2_sample(path):
    """Load and minimally validate one V2 result JSON sample."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        sample = json.load(file)
    responses = sample.get("responses", {})
    if "static" not in responses or "dynamic" not in responses:
        raise ValueError(f"{path} must contain responses.static and responses.dynamic")
    return sample


def _component_indices(selected, available, option_name):
    if not selected:
        return list(range(len(available)))
    normalized = [str(value).upper() for value in selected]
    invalid = [value for value in normalized if value not in available]
    if invalid:
        raise ValueError(
            f"invalid {option_name}: {invalid}; choose from {list(available)}"
        )
    return [available.index(value) for value in normalized]


def _resolve_node_index(sample, response_kind, node_id):
    section = sample["responses"][response_kind]
    node_ids = [int(value) for value in section.get("node_ids", [])]
    if node_id is None:
        if not node_ids:
            raise ValueError(f"responses.{response_kind}.node_ids is empty")
        node_id = node_ids[0]
    node_id = int(node_id)

    map_key = "disp" if response_kind == "static" else "acc"
    node_map = sample.get("node_maps", {}).get(map_key, {})
    if str(node_id) in node_map:
        return node_id, int(node_map[str(node_id)])
    if node_id in node_ids:
        return node_id, node_ids.index(node_id)
    raise ValueError(
        f"node {node_id} is not present in responses.{response_kind}; "
        f"available nodes: {node_ids}"
    )


def _sample_caption(sample):
    index = sample.get("sample_index", {})
    sample_id = index.get("sample_id", "unknown_sample")
    scenario = sample.get("structural_state", {}).get("scenario", "unknown")
    return str(sample_id), str(scenario)


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    _configure_chinese_font()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
    })
    return plt


def _finish_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, output_stem, formats=("png",), dpi=300):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for file_format in formats:
        file_format = str(file_format).lower().lstrip(".")
        if file_format not in {"png", "pdf", "svg"}:
            raise ValueError("formats must be selected from png, pdf, and svg")
        path = output_stem.with_suffix(f".{file_format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(path)
    return saved


def plot_vibration_timeseries(
    sample_or_path,
    node_id=None,
    components=None,
    output_stem=None,
    formats=("png",),
    dpi=300,
    time_range=None,
):
    """Plot acceleration time series for a selected dynamic node."""
    sample = (
        load_v2_sample(sample_or_path)
        if isinstance(sample_or_path, (str, Path))
        else sample_or_path
    )
    node_id, node_index = _resolve_node_index(sample, "dynamic", node_id)
    dynamic = sample["responses"]["dynamic"]
    acceleration = np.asarray(dynamic["ace"], dtype=np.float64)
    time_s = np.asarray(dynamic["time_s"], dtype=np.float64)
    if acceleration.ndim != 3 or acceleration.shape[0] != time_s.size:
        raise ValueError("dynamic ace/time_s shapes are inconsistent")

    names = tuple(
        str(value).upper()
        for value in dynamic.get("components", DYNAMIC_COMPONENTS)
    )
    indices = _component_indices(components, names, "vibration component")
    if time_range is not None:
        start, end = map(float, time_range)
        if end <= start:
            raise ValueError("time range end must be greater than start")
        mask = (time_s >= start) & (time_s <= end)
        if not np.any(mask):
            raise ValueError(f"time range {time_range} does not overlap the sample")
        time_s = time_s[mask]
        acceleration = acceleration[mask]

    plt = _prepare_matplotlib()
    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(7.2, 2.25 * len(indices)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    unit = dynamic.get("unit", "mm/s²")
    for position, (ax, component_index) in enumerate(zip(axes, indices)):
        component = names[component_index]
        signal = acceleration[:, node_index, component_index]
        ax.plot(time_s, signal, color=OKABE_ITO[component_index % 3])
        ax.axhline(0.0, color="#666666", linewidth=0.6, alpha=0.7)
        ax.set_ylabel(f"{component} ({unit})")
        ax.text(
            0.99,
            0.93,
            f"RMS = {np.sqrt(np.mean(signal ** 2)):.3f} {unit}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "#C7C7C7",
                "pad": 2.5,
            },
        )
        _finish_axes(ax)
        if position < len(indices) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("时间 (s)")
    sample_id, scenario = _sample_caption(sample)
    fig.suptitle(
        f"节点 {node_id} 振动加速度时序 | {sample_id} | {scenario}",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_stem is None:
        output_stem = Path.cwd() / f"{sample_id}_vibration_node_{node_id}"
    saved = _save_figure(fig, output_stem, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


def plot_temperature_displacement(
    sample_or_path,
    node_id=None,
    components=None,
    output_stem=None,
    formats=("png",),
    dpi=300,
):
    """Plot temperature-displacement curves for a selected static node."""
    sample = (
        load_v2_sample(sample_or_path)
        if isinstance(sample_or_path, (str, Path))
        else sample_or_path
    )
    node_id, node_index = _resolve_node_index(sample, "static", node_id)
    static = sample["responses"]["static"]
    displacement = np.asarray(static["disp"], dtype=np.float64)
    temperature_values = static.get("temperature_steps_C")
    if temperature_values is None:
        temperature_values = (
            sample.get("environment", {})
            .get("temperature", {})
            .get("temperature_steps_C")
        )
    temperature = np.asarray(temperature_values, dtype=np.float64)
    if displacement.ndim != 3 or displacement.shape[0] != temperature.size:
        raise ValueError("static disp/temperature shapes are inconsistent")

    names = tuple(
        str(value).upper()
        for value in static.get("components", STATIC_COMPONENTS)
    )
    indices = _component_indices(components, names, "displacement component")
    order = np.argsort(temperature)
    temperature = temperature[order]
    displacement = displacement[order]

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    markers = ("o", "s", "^")
    unit = static.get("unit", "mm")
    for component_index in indices:
        component = names[component_index]
        values = displacement[:, node_index, component_index]
        ax.plot(
            temperature,
            values,
            color=OKABE_ITO[component_index % 3],
            marker=markers[component_index % 3],
            markersize=5.0,
            label=component,
        )
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel(f"位移 ({unit})")
    ax.legend(frameon=False, ncol=len(indices), loc="best")
    _finish_axes(ax)
    sample_id, scenario = _sample_caption(sample)
    ax.set_title(
        f"节点 {node_id} 温度—位移响应 | {sample_id} | {scenario}",
        fontweight="bold",
        loc="left",
    )
    fig.tight_layout()

    if output_stem is None:
        output_stem = (
            Path.cwd() / f"{sample_id}_temperature_displacement_node_{node_id}"
        )
    saved = _save_figure(fig, output_stem, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


def plot_sample_responses(
    input_path,
    output_dir,
    vibration_node=None,
    vibration_components=None,
    static_node=None,
    displacement_components=None,
    formats=("png", "pdf"),
    dpi=300,
    time_range=None,
):
    """Generate both requested response figures and return saved paths."""
    sample = load_v2_sample(input_path)
    sample_id, _scenario = _sample_caption(sample)
    output_dir = Path(output_dir)
    dynamic_id, _ = _resolve_node_index(sample, "dynamic", vibration_node)
    static_id, _ = _resolve_node_index(sample, "static", static_node)
    saved = []
    saved.extend(
        plot_vibration_timeseries(
            sample,
            node_id=dynamic_id,
            components=vibration_components,
            output_stem=output_dir / f"{sample_id}_vibration_node_{dynamic_id}",
            formats=formats,
            dpi=dpi,
            time_range=time_range,
        )
    )
    saved.extend(
        plot_temperature_displacement(
            sample,
            node_id=static_id,
            components=displacement_components,
            output_stem=(
                output_dir
                / f"{sample_id}_temperature_displacement_node_{static_id}"
            ),
            formats=formats,
            dpi=dpi,
        )
    )
    return saved


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot V2 vibration time series and temperature-displacement responses."
        )
    )
    parser.add_argument(
        "--input", required=True, help="Path to one result_*.json file."
    )
    parser.add_argument("--output-dir", default="plot/outputs")
    parser.add_argument("--vib-node", type=int, default=None, help="Dynamic node ID.")
    parser.add_argument(
        "--vib-components",
        nargs="+",
        choices=DYNAMIC_COMPONENTS,
        default=list(DYNAMIC_COMPONENTS),
    )
    parser.add_argument(
        "--static-node", type=int, default=None, help="Static node ID."
    )
    parser.add_argument(
        "--disp-components",
        nargs="+",
        choices=STATIC_COMPONENTS,
        default=list(STATIC_COMPONENTS),
    )
    parser.add_argument(
        "--time-range", nargs=2, type=float, metavar=("START", "END")
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png", "pdf"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main():
    args = build_parser().parse_args()
    paths = plot_sample_responses(
        input_path=args.input,
        output_dir=args.output_dir,
        vibration_node=args.vib_node,
        vibration_components=args.vib_components,
        static_node=args.static_node,
        displacement_components=args.disp_components,
        formats=args.formats,
        dpi=args.dpi,
        time_range=args.time_range,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
