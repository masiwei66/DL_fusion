"""模型性能对比图。

在多个模型（静态单分支 / 动态单分支 / 融合模型）训练完成后，对比它们的
核心测试指标：
- 方法核心指标对比柱状图
- 论文向多指标对比图（核心指标、各材料 F1、融合权重、辅助任务）
- 修正模型同源测试与参考模型跨域独立测试对比
"""

import os

import numpy as np

from ._config import CANDIDATE_IDS, GRID_COLOR
from ._utils import (
    _configure_chinese_font,
    _metric_name_cn,
    _model_name_cn,
    _style_result_axes,
    _white_legend,
)


def plot_method_comparison(summaries, save_path):
    """绘制多个模型的核心测试指标（F1/AUC/精确率/召回率）对比柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        labels = [_model_name_cn(s["model_type"]) for s in summaries]
        metrics = ["f1", "auc", "precision", "recall"]
        x = np.arange(len(labels))
        width = 0.18

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            values = [s.get(metric, 0.0) for s in summaries]
            ax.bar(x + (i - 1.5) * width, values, width, label=_metric_name_cn(metric))

        ax.set_xlabel("模型")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("不同模型核心指标对比")
        _style_result_axes(ax)
        _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=8)

        plt.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved method comparison plot: {save_path}")
    except ImportError:
        pass


def plot_paper_figures(summaries, log_root, prefix="paper"):
    """批量绘制论文向的紧凑对比图，输出到 log_root 目录。

    包含：核心性能指标对比、各候选材料 F1 对比、融合模型分材料权重、
    辅助安全任务性能对比，共最多 4 张图。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        labels = [_model_name_cn(s["model_type"]) for s in summaries]
        x = np.arange(len(labels))

        # 图 1：核心性能指标对比
        core_metrics = [
            ("f1", "材料宏F1"),
            ("auc", "AUC"),
            ("accuracy", "标签准确率"),
            ("exact_match", "完全匹配率"),
        ]
        width = 0.18
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for i, (metric, label) in enumerate(core_metrics):
            values = [s.get(metric, 0.0) for s in summaries]
            ax.bar(x + (i - 1.5) * width, values, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("核心性能指标对比")
        _style_result_axes(ax)
        _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, fontsize=8)
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        path = os.path.join(log_root, f"{prefix}_core_metrics.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved paper figure: {path}")

        # 图 2：各候选材料 F1 对比
        material_labels = [str(mid) for mid in CANDIDATE_IDS]
        x = np.arange(len(material_labels))
        width = min(0.25, 0.8 / max(1, len(summaries)))
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for i, summary in enumerate(summaries):
            values = [summary.get(f"f1_{mid}", 0.0) for mid in CANDIDATE_IDS]
            offset = (i - (len(summaries) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=_model_name_cn(summary["model_type"]))
        ax.set_xticks(x)
        ax.set_xticklabels(material_labels)
        ax.set_xlabel("材料编号")
        ax.set_title("各候选材料F1对比")
        _style_result_axes(ax, ylabel="F1")
        _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=8)
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        path = os.path.join(log_root, f"{prefix}_per_class_f1.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved paper figure: {path}")

        # 图 3：融合模型分材料权重（静态/动态分支占比）
        fusion = next((s for s in summaries if s.get("model_type") == "fusion"), None)
        if fusion is not None and any(f"alpha_static_{mid}" in fusion for mid in CANDIDATE_IDS):
            static_w = [fusion.get(f"alpha_static_{mid}", 0.0) for mid in CANDIDATE_IDS]
            dynamic_w = [fusion.get(f"alpha_dynamic_{mid}", 1.0 - sw) for mid, sw in zip(CANDIDATE_IDS, static_w)]
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            ax.bar(material_labels, static_w, label="静态分支权重", color="#4C78A8")
            ax.bar(material_labels, dynamic_w, bottom=static_w, label="动态分支权重", color="#F58518")
            ax.set_xlabel("材料编号")
            ax.set_ylabel("融合权重")
            ax.set_ylim(0, 1.0)
            ax.set_title("融合模型分材料权重")
            ax.grid(True, alpha=0.35, axis="y", color=GRID_COLOR, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
            plt.tight_layout(rect=(0, 0, 1, 0.92))
            path = os.path.join(log_root, f"{prefix}_fusion_alpha.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved paper figure: {path}")

        # 图 4：辅助安全任务性能对比（支座 F1 / 区域宏 F1）
        aux_metrics = [
            ("support_f1", "支座F1"),
            ("region_f1_macro", "区域宏F1"),
        ]
        if any(any(metric in s for metric, _ in aux_metrics) for s in summaries):
            method_x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9, 5.4))
            width = 0.28
            for i, (metric, label) in enumerate(aux_metrics):
                values = [s.get(metric, 0.0) for s in summaries]
                offset = (i - (len(aux_metrics) - 1) / 2) * width
                ax.bar(method_x + offset, values, width, label=label)
            ax.set_xticks(method_x)
            ax.set_xticklabels(labels)
            ax.set_title("辅助安全任务性能对比")
            _style_result_axes(ax)
            _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
            plt.tight_layout(rect=(0, 0, 1, 0.92))
            path = os.path.join(log_root, f"{prefix}_auxiliary_tasks.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved paper figure: {path}")

    except ImportError:
        pass


def plot_domain_comparison(summaries, save_path):
    """绘制修正模型同源测试与参考模型独立测试的性能对比图。"""
    if not any("reference_f1" in s or "reference_auc" in s for s in summaries):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        metrics = [
            ("f1", "材料宏F1"),
            ("auc", "AUC"),
            ("exact_match", "完全匹配率"),
            ("support_f1", "支座F1"),
            ("region_f1_macro", "区域宏F1"),
        ]
        labels = [_model_name_cn(s["model_type"]) for s in summaries]
        x = np.arange(len(labels))
        width = 0.34

        fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.8), sharey=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, (metric, label) in zip(axes, metrics):
            same_values = [s.get(metric, 0.0) for s in summaries]
            ref_values = [s.get(f"reference_{metric}", np.nan) for s in summaries]
            ax.bar(x - width / 2, same_values, width, label="修正模型同源测试", color="#4C78A8")
            ax.bar(x + width / 2, ref_values, width, label="参考模型独立测试", color="#F58518")
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            _style_result_axes(ax)
            # 标注修正模型相对参考模型的性能差值（百分比）
            for idx, value in enumerate(ref_values):
                if np.isfinite(value):
                    delta = value - same_values[idx]
                    ax.text(
                        idx,
                        min(1.02, max(value, same_values[idx]) + 0.035),
                        f"{delta:+.2%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#8A4A00" if delta < 0 else "#1F6B3A",
                    )
        axes[0].set_ylabel("指标值")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            fontsize=9,
            frameon=True,
        )
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#BFBFBF")
        frame.set_alpha(1.0)
        fig.suptitle("修正模型同源测试与参考模型独立测试性能对比", y=1.12, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved domain comparison plot: {save_path}")
    except ImportError:
        pass


def extract_reference_summaries(summaries):
    """抽取 summary 中以 reference_ 开头的指标，返回用普通指标名的副本。

    用于跨域对比时，将"参考模型独立测试"的指标整理为可直接绘图的格式。
    """
    reference = []
    for summary in summaries:
        item = {"model_type": summary.get("model_type")}
        for key, value in summary.items():
            if key.startswith("reference_"):
                item[key[len("reference_"):]] = value
        if len(item) > 1:
            reference.append(item)
    return reference
