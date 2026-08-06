"""DL_model 包 — 双分支融合网络用于结构损伤识别（6 种候选材料）。"""

from .DL_config import Config, CANDIDATE_IDS, STATIC_NODES, DYNAMIC_NODES
from .dataset import StructuralDataset
from .model import (
    PositionEncoding, StaticBranch, DynamicBranch,
    ReliabilityGatedFusionHead, StaticAnchoredFusionHead,
    StaticOnlyModel, DynamicOnlyModel, DualBranchFusion, build_model,
)

# 绘图函数采用延迟导入，避免本包与绘图包 plot 相互导入产生循环依赖
_PLOT_EXPORTS = {
    "plot_history",
    "plot_evaluation",
    "plot_method_comparison",
    "plot_paper_figures",
    "save_gate_diagnostics",
}


def __getattr__(name):
    """延迟暴露绘图函数：`DL_model.plot_history` 仍可像以前一样直接使用。"""
    if name in _PLOT_EXPORTS:
        import plot
        return getattr(plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
