"""单样本预测结果绘图。

针对单个样本的预测输出，生成论文可用的分任务图：
- 材料刚度系数与损伤概率图
- 支座沉降与风险概率图
- 区域风险等级热力条
- 材料损伤概率分布图
- 多面板汇总图
"""

import os

import numpy as np

from ._config import PRED_COLOR, PROB_COLOR, RISK_COLORS, TRUE_COLOR
from ._utils import (
    _configure_chinese_font,
    _get_ground_truth,
    _material_arrays,
    _prediction_base,
    _prepare_prediction_axes,
    _reorder_regions,
    _region_arrays,
    _region_name_cn,
    _risk_name_cn,
    _safe_save,
    _support_arrays,
    _white_legend,
)


def plot_prediction_material(result, save_path):
    """绘制单样本材料预测图：预测/真实刚度系数柱状图 + 损伤概率曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ids, probs, preds, pred_sf, true_labels, true_sf = _material_arrays(result)
    n = len(ids)
    x = np.arange(n)
    has_true_sf = true_sf is not None and len(true_sf) == n

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    width = 0.34 if has_true_sf else 0.52
    ax.bar(x - width / 2 if has_true_sf else x, pred_sf, width, color=PRED_COLOR,
           label="预测刚度系数")
    if has_true_sf:
        ax.bar(x + width / 2, true_sf, width, color=TRUE_COLOR, alpha=0.82,
               label="真实刚度系数")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, label="无损水平")
    ax.set_ylim(0, max(1.15, float(np.nanmax(pred_sf)) + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_xlabel("材料编号")
    _prepare_prediction_axes(ax, None, "刚度系数")

    # 双 y 轴叠加损伤概率曲线
    ax_prob = ax.twinx()
    ax_prob.plot(x, probs, color=PROB_COLOR, marker="o", linewidth=1.8,
                 label="损伤概率")
    ax_prob.axhline(0.5, color=PROB_COLOR, linestyle=":", linewidth=1.0)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("损伤概率")
    ax_prob.spines["top"].set_visible(False)

    # 在每个材料上方标注损伤/正常
    for i in range(n):
        label = "损伤" if int(preds[i]) else "正常"
        color = "#8B1E1E" if int(preds[i]) else "#204A73"
        ax_prob.text(i, min(1.02, probs[i] + 0.07), label, ha="center",
                     va="bottom", fontsize=8.5, color=color, fontweight="bold")

    # 合并左右两轴的图例并置于顶部
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax_prob.get_legend_handles_labels()
    _white_legend(
        ax,
        handles + handles2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def plot_prediction_support(result, save_path):
    """绘制单样本支座预测图：预测/真实沉降柱状图 + 风险概率曲线，并标注预警阈值。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    nodes, probs, preds, pred_mm, true_labels, true_mm = _support_arrays(result)
    n = len(nodes)
    x = np.arange(n)
    has_true_mm = true_mm is not None and len(true_mm) == n

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.34 if has_true_mm else 0.52
    ax.bar(x - width / 2 if has_true_mm else x, pred_mm, width, color=PRED_COLOR,
           label="预测沉降")
    if has_true_mm:
        ax.bar(x + width / 2, true_mm, width, color=TRUE_COLOR, alpha=0.82,
               label="真实沉降")
    # 绘制预警/风险/危险三级阈值线
    for value, label, color in [(2.0, "预警", "#A6761D"), (5.0, "风险", "#B65C00"), (7.0, "危险", "#9E2F2F")]:
        ax.axhline(value, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(n - 0.48, value + 0.05, label, color=color, fontsize=8,
                ha="right", va="bottom")
    upper = max(8.0, float(np.nanmax(pred_mm)) + 1.0)
    if has_true_mm:
        upper = max(upper, float(np.nanmax(true_mm)) + 1.0)
    ax.set_ylim(0, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.set_xlabel("支座节点")
    _prepare_prediction_axes(ax, None, "沉降值 (mm)")

    # 双 y 轴叠加风险概率曲线
    ax_prob = ax.twinx()
    ax_prob.plot(x, probs, color=PROB_COLOR, marker="o", linewidth=1.8,
                 label="风险概率")
    ax_prob.axhline(0.5, color=PROB_COLOR, linestyle=":", linewidth=1.0)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("风险概率")
    ax_prob.spines["top"].set_visible(False)

    # 在每个支座上方标注风险/正常
    for i in range(n):
        label = "风险" if int(preds[i]) else "正常"
        color = "#8B1E1E" if int(preds[i]) else "#204A73"
        ax_prob.text(i, min(1.02, probs[i] + 0.07), label, ha="center",
                     va="bottom", fontsize=8.5, color=color, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax_prob.get_legend_handles_labels()
    _white_legend(
        ax,
        handles + handles2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def plot_prediction_region(result, save_path):
    """绘制单样本区域风险等级热力条，展示各区域预测/真实风险等级。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        return None

    names, pred_levels, true_levels = _region_arrays(result)
    names, pred_levels, true_levels = _reorder_regions(names, pred_levels, true_levels)
    names_cn = [_region_name_cn(name) for name in names]
    has_true = true_levels is not None and len(true_levels) == len(pred_levels)
    rows = [pred_levels]
    row_labels = ["预测值"]
    if has_true:
        rows.append(true_levels)
        row_labels.append("真实值")
    data = np.vstack(rows)

    # 按风险等级颜色映射绘制热力条
    fig_h = 2.7 if has_true else 2.2
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    cmap = ListedColormap(RISK_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names_cn, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 在每个格子内标注风险等级
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            level = int(data[r, c])
            text_color = "white" if level in (0, 3) else "#222222"
            ax.text(c, r, _risk_name_cn(level), ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    # 预测与真实不一致的区域用黑色框标出
    if has_true:
        for c, ok in enumerate(pred_levels == true_levels):
            if not bool(ok):
                ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, 2, fill=False,
                                           edgecolor="#111111", linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels([_risk_name_cn(i) for i in range(4)])
    fig.tight_layout()
    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def plot_prediction_damage_probability(result, save_path):
    """绘制单样本材料损伤概率分布图（独立版）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    ids, mat_probs, mat_preds, pred_sf, true_mat, true_sf = _material_arrays(result)
    x = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = np.where(mat_preds.astype(bool), TRUE_COLOR, PRED_COLOR)
    ax.bar(x, mat_probs, color=colors, alpha=0.86, label="预测损伤概率")
    ax.axhline(0.5, color="#222222", linestyle=":", linewidth=1.0, label="判别阈值")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("材料编号")
    _prepare_prediction_axes(ax, None, "损伤概率")
    # 在每个柱子上方标注概率值
    for i, prob in enumerate(mat_probs):
        ax.text(i, min(1.02, prob + 0.035), f"{prob:.0%}", ha="center",
                va="bottom", fontsize=8)
    _white_legend(
        ax,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def plot_prediction_summary(result, save_path):
    """绘制单样本预测结果的紧凑多面板汇总图（材料概率 / 支座沉降 / 区域风险）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        return None

    ids, mat_probs, mat_preds, pred_sf, true_mat, true_sf = _material_arrays(result)
    nodes, sup_probs, sup_preds, pred_mm, true_sup, true_mm = _support_arrays(result)
    names, pred_levels, true_levels = _region_arrays(result)
    names, pred_levels, true_levels = _reorder_regions(names, pred_levels, true_levels)
    gt = _get_ground_truth(result)

    # 2 行布局：上排左=材料损伤概率，上排右=支座沉降，下排=区域风险热力条
    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.45, wspace=0.28)

    # 左上：材料损伤概率柱状图
    ax_mat = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ids))
    colors = np.where(mat_preds.astype(bool), "#E45756", "#4C78A8")
    ax_mat.bar(x, mat_probs, color=colors, alpha=0.86)
    ax_mat.axhline(0.5, color="#222222", linestyle=":", linewidth=1.0)
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels(ids)
    ax_mat.set_ylim(0, 1.05)
    ax_mat.set_xlabel("材料编号")
    _prepare_prediction_axes(ax_mat, None, "损伤概率")
    for i, prob in enumerate(mat_probs):
        ax_mat.text(i, prob + 0.035, f"{prob:.0%}", ha="center", va="bottom", fontsize=8)

    # 右上：支座沉降柱状图（含预警阈值线）
    ax_sup = fig.add_subplot(gs[0, 1])
    sx = np.arange(len(nodes))
    has_true_mm = true_mm is not None and len(true_mm) == len(nodes)
    width = 0.34 if has_true_mm else 0.52
    ax_sup.bar(sx - width / 2 if has_true_mm else sx, pred_mm, width,
               color=PRED_COLOR, label="预测值")
    if has_true_mm:
        ax_sup.bar(sx + width / 2, true_mm, width, color=TRUE_COLOR,
                   alpha=0.82, label="真实值")
    for value in [2.0, 5.0, 7.0]:
        ax_sup.axhline(value, color="#777777", linestyle="--", linewidth=0.8, alpha=0.65)
    ax_sup.set_xticks(sx)
    ax_sup.set_xticklabels(nodes)
    ax_sup.set_xlabel("支座节点")
    _prepare_prediction_axes(ax_sup, None, "沉降值 (mm)")
    _white_legend(ax_sup, fontsize=8, loc="upper left")

    # 下排：区域风险等级热力条
    ax_reg = fig.add_subplot(gs[1, :])
    names_cn = [_region_name_cn(name) for name in names]
    has_true_region = true_levels is not None and len(true_levels) == len(pred_levels)
    region_rows = [pred_levels]
    row_labels = ["预测值"]
    if has_true_region:
        region_rows.append(true_levels)
        row_labels.append("真实值")
    region_data = np.vstack(region_rows)
    cmap = ListedColormap(RISK_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax_reg.imshow(region_data, cmap=cmap, norm=norm, aspect="auto")
    ax_reg.set_xticks(np.arange(len(names)))
    ax_reg.set_xticklabels(names_cn, rotation=25, ha="right")
    ax_reg.set_yticks(np.arange(len(row_labels)))
    ax_reg.set_yticklabels(row_labels)
    ax_reg.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax_reg.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_reg.grid(which="minor", color="white", linewidth=1.5)
    ax_reg.tick_params(which="minor", bottom=False, left=False)
    for r in range(region_data.shape[0]):
        for c in range(region_data.shape[1]):
            level = int(region_data[r, c])
            text_color = "white" if level in (0, 3) else "#222222"
            ax_reg.text(c, r, _risk_name_cn(level), ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_reg, ticks=[0, 1, 2, 3], fraction=0.02, pad=0.02)
    cbar.ax.set_yticklabels([_risk_name_cn(i) for i in range(4)])

    # 图标题与整体状态说明
    if gt and "global_level" in result and gt.get("global_level") is not None:
        global_text = (
            f"整体状态：预测为{_risk_name_cn(result['global_level'])}；"
            f"真实为{_risk_name_cn(gt['global_level'])}"
        )
    elif "global_level" in result:
        global_text = f"整体状态：预测为{_risk_name_cn(result['global_level'])}"
    else:
        global_text = ""
    fig.suptitle("单样本安全状态预测结果", fontsize=13, fontweight="bold")
    if global_text:
        fig.text(0.5, 0.935, global_text, ha="center", va="center", fontsize=10)

    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def save_prediction_figures(result, input_path, output_dir):
    """为单个预测结果批量生成多张 PNG 图，返回已保存的文件路径列表。

    包含 1 张紧凑汇总图和 4 张分任务图，可直接用于论文。
    """
    os.makedirs(output_dir, exist_ok=True)
    base = _prediction_base(input_path)
    figure_specs = [
        ("prediction_summary", plot_prediction_summary),
        ("damage_probability", plot_prediction_damage_probability),
        ("material_condition", plot_prediction_material),
        ("support_settlement", plot_prediction_support),
        ("region_risk", plot_prediction_region),
    ]
    saved = []
    for suffix, func in figure_specs:
        path = os.path.join(output_dir, f"{base}_{suffix}.png")
        out = func(result, path)
        if out:
            saved.append(out)
    return saved
