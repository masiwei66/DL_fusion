"""DL_model 包 — 双分支融合网络用于结构损伤识别（6 种候选材料）。"""

from .DL_config import Config, CANDIDATE_IDS, STATIC_NODES, DYNAMIC_NODES
from .dataset import StructuralDataset
from .model import (
    PositionEncoding, StaticBranch, DynamicBranch,
    ReliabilityGatedFusionHead, StaticAnchoredFusionHead,
    StaticOnlyModel, DynamicOnlyModel, DualBranchFusion, build_model,
)
from .plotting import (
    plot_history,
    plot_evaluation,
    plot_method_comparison,
    plot_paper_figures,
    save_gate_diagnostics,
)
