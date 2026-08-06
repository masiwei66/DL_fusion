"""绘图通用辅助函数。

包含：中文字体配置、图例/坐标轴样式、图片保存、预测结果数据抽取与整理等，
供各绘图模块复用。函数以下划线开头，表示内部工具，不对外暴露。
"""

import os

import numpy as np

from ._config import (
    CANDIDATE_IDS,
    CHINESE_FONT_CANDIDATES,
    CHINESE_FONT_FILES,
    GRID_COLOR,
    METRIC_NAMES_CN,
    MODEL_NAMES_CN,
    REGION_DISPLAY_ORDER,
    REGION_NAMES_CN,
    RISK_LEVELS,
    RISK_LEVEL_NAMES_CN,
)


def _risk_name(level):
    """将风险等级数值映射为英文名称（截断到合法范围）。"""
    idx = int(np.clip(level, 0, len(RISK_LEVELS) - 1))
    return RISK_LEVELS[idx]


def _risk_name_cn(level):
    """将风险等级数值映射为中文名称。"""
    idx = int(np.clip(level, 0, len(RISK_LEVEL_NAMES_CN) - 1))
    return RISK_LEVEL_NAMES_CN[idx]


def _region_name_cn(name):
    """将区域英文标识映射为中文名称，无映射时原样返回。"""
    return REGION_NAMES_CN.get(str(name), str(name))


def _configure_chinese_font():
    """配置 matplotlib 使用中文字体，避免图中中文显示为方框。

    会优先注册常见 Windows 中文字体文件，再从候选字体中选择可用的字体。
    """
    import matplotlib
    from matplotlib import font_manager

    # 手动注册系统中存在的常见中文字体
    for font_path in CHINESE_FONT_FILES:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass

    # 从已注册字体中选择第一个可用的中文字体
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in CHINESE_FONT_CANDIDATES if name in available]
    if fonts:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = fonts + matplotlib.rcParams.get("font.sans-serif", [])
    # 解决负号显示为方块的问题，并将文本保存为可编辑矢量
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"


def _white_legend(ax, *args, **kwargs):
    """创建不透明白底、灰色边框的图例，避免遮挡图中内容。"""
    legend = ax.legend(*args, **kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#BFBFBF")
        frame.set_alpha(1.0)
    return legend


def _model_name_cn(name):
    """将模型类型英文标识映射为中文名称。"""
    return MODEL_NAMES_CN.get(str(name), str(name))


def _metric_name_cn(name):
    """将评估指标英文标识映射为中文名称。"""
    return METRIC_NAMES_CN.get(str(name), str(name))


def _style_result_axes(ax, ylabel="指标值"):
    """统一样式化"指标类"图表的坐标轴（y 轴范围 0~1.05，隐藏上右框线）。"""
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _to_numpy(values, dtype=float):
    """将输入转换为 numpy 数组；None 则返回 None。"""
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _get_ground_truth(result):
    """从预测结果字典中取出真实值子字典，不存在则返回 None。"""
    gt = result.get("ground_truth") or {}
    return gt if gt else None


def _prepare_prediction_axes(ax, title=None, ylabel=None):
    """统一样式化"预测结果图"的坐标轴（左侧标题、上右框线隐藏、y 网格线）。"""
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _safe_save(fig, path, dpi=300):
    """保存图片为高分辨率并紧凑裁剪，返回保存路径。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _prediction_base(input_path):
    """从输入文件路径提取文件名主干，用作输出图片的文件名前缀。"""
    stem = os.path.splitext(os.path.basename(str(input_path)))[0]
    return stem or "prediction"


def _material_arrays(result):
    """从预测结果中抽取材料的 ID、概率、预测值、真实值等数组。"""
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
    """从预测结果中抽取支座的节点、概率、预测值、真实值等数组。"""
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
    """从预测结果中抽取区域的名称、预测等级、真实等级等数组。"""
    gt = _get_ground_truth(result)
    names = [str(v) for v in result.get("region_names", [])]
    pred_levels = _to_numpy(result.get("region_levels", []), int)
    true_levels = None
    if gt:
        true_levels = _to_numpy(gt.get("region_risk_levels"), int)
    return names, pred_levels, true_levels


def _reorder_regions(names, pred_levels, true_levels=None):
    """按固定显示顺序重排区域，保证图中区域顺序与结构布置一致。"""
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
