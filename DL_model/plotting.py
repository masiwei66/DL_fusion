"""Training and evaluation plotting utilities.

All functions silently fall back to a text export (history) or no-op
(evaluation / comparison) when matplotlib is not available.
"""

import json
import os

import numpy as np

try:
    from .safety_rules import CANDIDATE_MATERIAL_IDS, RISK_LEVELS
except ImportError:
    from safety_rules import CANDIDATE_MATERIAL_IDS, RISK_LEVELS

CANDIDATE_IDS = CANDIDATE_MATERIAL_IDS

MODEL_NAMES_CN = {
    "static_only": "静态单分支",
    "dynamic_only": "动态单分支",
    "fusion": "融合模型",
}

METRIC_NAMES_CN = {
    "f1": "材料宏F1",
    "auc": "AUC",
    "accuracy": "标签准确率",
    "exact_match": "完全匹配率",
    "precision": "精确率",
    "recall": "召回率",
    "support_f1": "支座F1",
    "support_f1_macro": "支座F1",
    "region_f1_macro": "区域宏F1",
}


RISK_COLORS = ["#4C78A8", "#F2CF5B", "#F58518", "#D94E4E"]
PRED_COLOR = "#4C78A8"
TRUE_COLOR = "#E45756"
PROB_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"

RISK_LEVEL_NAMES_CN = ["安全", "预警", "风险", "危险"]
REGION_NAMES_CN = {
    "left_support1_zone": "左1号支座区",
    "left_support2_zone": "左2号支座区",
    "right_support2_zone": "右2号支座区",
    "right_zone": "右胯区域",
    "mid_zone": "跨中区域",
    "left_zone": "左胯区域",
    "right_support1_zone": "右1号支座区",
}
REGION_DISPLAY_ORDER = [
    "left_support1_zone",
    "left_support2_zone",
    "left_zone",
    "mid_zone",
    "right_zone",
    "right_support1_zone",
    "right_support2_zone",
]
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DengXian",
    "FangSong",
    "KaiTi",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
CHINESE_FONT_FILES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]


def _risk_name(level):
    idx = int(np.clip(level, 0, len(RISK_LEVELS) - 1))
    return RISK_LEVELS[idx]


def _risk_name_cn(level):
    idx = int(np.clip(level, 0, len(RISK_LEVEL_NAMES_CN) - 1))
    return RISK_LEVEL_NAMES_CN[idx]


def _region_name_cn(name):
    return REGION_NAMES_CN.get(str(name), str(name))


def _configure_chinese_font():
    import matplotlib
    from matplotlib import font_manager

    for font_path in CHINESE_FONT_FILES:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass

    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in CHINESE_FONT_CANDIDATES if name in available]
    if fonts:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = fonts + matplotlib.rcParams.get("font.sans-serif", [])
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"


def _white_legend(ax, *args, **kwargs):
    legend = ax.legend(*args, **kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#BFBFBF")
        frame.set_alpha(1.0)
    return legend


def _model_name_cn(name):
    return MODEL_NAMES_CN.get(str(name), str(name))


def _metric_name_cn(name):
    return METRIC_NAMES_CN.get(str(name), str(name))


def _style_result_axes(ax, ylabel="指标值"):
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _to_numpy(values, dtype=float):
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _get_ground_truth(result):
    gt = result.get("ground_truth") or {}
    return gt if gt else None


def _prepare_prediction_axes(ax, title=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _safe_save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _prediction_base(input_path):
    stem = os.path.splitext(os.path.basename(str(input_path)))[0]
    return stem or "prediction"


def _material_arrays(result):
    gt = _get_ground_truth(result)
    ids = [str(v) for v in result.get("material_ids", CANDIDATE_IDS)]
    probs = _to_numpy(result.get("material_probs", []), float)
    preds = _to_numpy(result.get("material_preds", []), int)
    pred_sf = _to_numpy(result.get("material_scaling_pred", []), float)
    true_labels = true_sf = None
    if gt:
        true_labels = _to_numpy(gt.get("material_labels"), int)
        true_sf = _to_numpy(gt.get("material_sf"), float)
    return ids, probs, preds, pred_sf, true_labels, true_sf


def _support_arrays(result):
    gt = _get_ground_truth(result)
    nodes = [str(v) for v in result.get("support_nodes", [])]
    probs = _to_numpy(result.get("support_probs", []), float)
    preds = _to_numpy(result.get("support_preds", []), int)
    pred_mm = _to_numpy(result.get("support_disp_mm", []), float)
    true_labels = true_mm = None
    if gt:
        true_labels = _to_numpy(gt.get("support_labels"), int)
        true_mm = _to_numpy(gt.get("support_values_mm"), float)
    return nodes, probs, preds, pred_mm, true_labels, true_mm


def _region_arrays(result):
    gt = _get_ground_truth(result)
    names = [str(v) for v in result.get("region_names", [])]
    pred_levels = _to_numpy(result.get("region_levels", []), int)
    true_levels = None
    if gt:
        true_levels = _to_numpy(gt.get("region_risk_levels"), int)
    return names, pred_levels, true_levels


def _reorder_regions(names, pred_levels, true_levels=None):
    if pred_levels is None or len(names) != len(pred_levels):
        return names, pred_levels, true_levels

    index_by_name = {name: idx for idx, name in enumerate(names)}
    ordered_indices = [
        index_by_name[name]
        for name in REGION_DISPLAY_ORDER
        if name in index_by_name
    ]
    ordered_indices.extend(
        idx
        for idx, name in enumerate(names)
        if name not in REGION_DISPLAY_ORDER
    )

    names = [names[idx] for idx in ordered_indices]
    pred_levels = pred_levels[ordered_indices]
    if true_levels is not None and len(true_levels) == len(ordered_indices):
        true_levels = true_levels[ordered_indices]
    return names, pred_levels, true_levels


def plot_prediction_material(result, save_path):
    """Save a paper-style material prediction figure for one sample."""
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

    ax_prob = ax.twinx()
    ax_prob.plot(x, probs, color=PROB_COLOR, marker="o", linewidth=1.8,
                 label="损伤概率")
    ax_prob.axhline(0.5, color=PROB_COLOR, linestyle=":", linewidth=1.0)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("损伤概率")
    ax_prob.spines["top"].set_visible(False)

    for i in range(n):
        label = "损伤" if int(preds[i]) else "正常"
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
        ncol=4,
        fontsize=8,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = _safe_save(fig, save_path)
    plt.close(fig)
    return path


def plot_prediction_support(result, save_path):
    """Save a support-settlement prediction figure for one sample."""
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

    ax_prob = ax.twinx()
    ax_prob.plot(x, probs, color=PROB_COLOR, marker="o", linewidth=1.8,
                 label="风险概率")
    ax_prob.axhline(0.5, color=PROB_COLOR, linestyle=":", linewidth=1.0)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel("风险概率")
    ax_prob.spines["top"].set_visible(False)

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
    """Save a region-risk prediction heat strip for one sample."""
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

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            level = int(data[r, c])
            text_color = "white" if level in (0, 3) else "#222222"
            ax.text(c, r, _risk_name_cn(level), ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

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
    """Save a stand-alone material damage-probability figure for one sample."""
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
    """Save a compact multi-panel prediction summary for one sample."""
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

    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.45, wspace=0.28)

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
    """Create several PNG figures for one prediction result.

    Returns a list of saved paths. The figures are designed for direct use in
    a manuscript: one compact summary and four task-specific panels.
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



def plot_history(history, save_path):
    """Plot training curves with Chinese labels for manuscript use."""
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in history], label="训练集", color="#2196F3")
        ax.plot(epochs, [h["val_loss"] for h in history], label="验证集", color="#FF5722")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失")
        ax.set_title("损失曲线")
        _white_legend(ax, loc="upper right")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs, [h["train_f1"] for h in history], label="训练集", color="#2196F3")
        ax.plot(epochs, [h["val_f1"] for h in history], label="验证集", color="#FF5722")
        if "val_f1_at_05" in history[0]:
            ax.plot(
                epochs,
                [h["val_f1_at_05"] for h in history],
                label="验证集(阈值0.5)",
                color="#FF9800",
                linestyle="--",
            )
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("F1")
        ax.set_title("F1曲线")
        _white_legend(ax, loc="lower right")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(epochs, [h["train_acc"] for h in history], label="训练标签准确率", color="#2196F3")
        ax.plot(epochs, [h["val_acc"] for h in history], label="验证标签准确率", color="#FF5722")
        if "val_exact_match" in history[0]:
            ax.plot(
                epochs,
                [h["val_exact_match"] for h in history],
                label="验证完全匹配率",
                color="#9C27B0",
                linestyle="--",
            )
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("准确率")
        ax.set_title("准确率")
        _white_legend(ax, loc="lower right")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, [h["train_precision"] for h in history], label="训练精确率", color="#2196F3")
        ax.plot(epochs, [h["val_precision"] for h in history], label="验证精确率", color="#FF5722")
        ax.plot(epochs, [h["train_recall"] for h in history], label="训练召回率", linestyle="--", color="#4CAF50")
        ax.plot(epochs, [h["val_recall"] for h in history], label="验证召回率", linestyle="--", color="#FF9800")
        if "val_recall_at_05" in history[0]:
            ax.plot(
                epochs,
                [h["val_recall_at_05"] for h in history],
                label="验证召回率(阈值0.5)",
                linestyle=":",
                color="#795548",
            )
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("指标值")
        ax.set_title("精确率与召回率")
        _white_legend(ax, fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(epochs, [h["train_auc"] for h in history], label="训练集", color="#2196F3")
        ax.plot(epochs, [h["val_auc"] for h in history], label="验证集", color="#FF5722")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("AUC")
        ax.set_title("AUC")
        _white_legend(ax, loc="lower right")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(epochs, [h["lr"] for h in history], color="#9C27B0")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("学习率")
        ax.set_title("学习率曲线")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved history plot: {save_path}")
    except ImportError:
        txt_path = save_path.replace(".png", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\ttrain_f1\tval_f1\n")
            for h in history:
                f.write(
                    f"{h['epoch']}\t{h['train_loss']:.4f}\t{h['val_loss']:.4f}\t"
                    f"{h['train_f1']:.4f}\t{h['val_f1']:.4f}\n"
                )
        print(f"Saved history text: {txt_path}")


def plot_evaluation(y_true, y_pred, y_prob, save_path):
    """Plot evaluation panels with Chinese labels and opaque legends."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            auc,
            f1_score,
            multilabel_confusion_matrix,
            precision_score,
            recall_score,
            roc_curve,
        )

        n_classes = len(CANDIDATE_IDS)
        labels = [str(c) for c in CANDIDATE_IDS]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax = axes[0, 0]
        ml_cm = multilabel_confusion_matrix(y_true, y_pred)
        cm_view = np.array([[cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]] for cm in ml_cm])
        im = ax.imshow(cm_view, cmap="Blues", aspect="auto")
        ax.set_xlabel("统计项")
        ax.set_ylabel("材料编号")
        ax.set_title("多标签混淆矩阵")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["TN", "FP", "FN", "TP"])
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(labels)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("样本数")
        max_value = max(1, cm_view.max())
        for i in range(n_classes):
            for j in range(4):
                ax.text(
                    j, i, str(cm_view[i, j]), ha="center", va="center",
                    color="white" if cm_view[i, j] > max_value / 2 else "black",
                )

        ax = axes[0, 1]
        f1_per = [f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(n_classes)]
        prec_per = [
            precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            for i in range(n_classes)
        ]
        recall_per = [
            recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            for i in range(n_classes)
        ]
        x = np.arange(n_classes)
        w = 0.25
        ax.bar(x - w, f1_per, w, label="F1", color="#2196F3")
        ax.bar(x, prec_per, w, label="精确率", color="#4CAF50")
        ax.bar(x + w, recall_per, w, label="召回率", color="#FF9800")
        ax.set_xlabel("材料编号")
        ax.set_ylabel("指标值")
        ax.set_title("各材料分类指标")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        _style_result_axes(ax)
        _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=8)

        ax = axes[1, 0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="随机猜测")
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
        for i in range(n_classes):
            if len(np.unique(y_true[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc_i = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, color=colors[i % len(colors)],
                label=f"材料{labels[i]} (AUC={auc_i:.3f})",
            )
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
        ax.set_title("ROC曲线")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _white_legend(ax, fontsize=8, loc="lower right")

        ax = axes[1, 1]
        neg_probs = y_prob[y_true == 0]
        pos_probs = y_prob[y_true == 1]
        if len(neg_probs) > 0:
            ax.hist(neg_probs, bins=40, alpha=0.75, color="#2196F3", label=f"正常样本 (n={len(neg_probs)})")
        if len(pos_probs) > 0:
            ax.hist(pos_probs, bins=40, alpha=0.75, color="#FF5722", label=f"损伤样本 (n={len(pos_probs)})")
        ax.set_xlabel("预测损伤概率")
        ax.set_ylabel("样本数")
        ax.set_title("预测概率分布")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _white_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved evaluation plot: {save_path}")
    except ImportError:
        pass


def save_gate_diagnostics(metrics, save_path):
    """Persist fusion diagnostics (alpha, gate, disagreement) as a JSON report."""
    fusion_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("alpha_")
        or key.startswith("gate_")
        or key.startswith("branch_disagreement_")
    }
    if not fusion_metrics:
        return
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fusion_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved fusion diagnostics: {save_path}")



def plot_method_comparison(summaries, save_path):
    """Plot core test metrics for all trained methods with Chinese labels."""
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
    """Save compact Chinese paper-oriented comparison figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        labels = [_model_name_cn(s["model_type"]) for s in summaries]
        x = np.arange(len(labels))

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
    """Plot same-domain vs reference-model cross-domain performance."""
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
    """Return summaries using reference_* metrics as normal metric names."""
    reference = []
    for summary in summaries:
        item = {"model_type": summary.get("model_type")}
        for key, value in summary.items():
            if key.startswith("reference_"):
                item[key[len("reference_"):]] = value
        if len(item) > 1:
            reference.append(item)
    return reference


