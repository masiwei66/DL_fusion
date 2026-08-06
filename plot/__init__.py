"""绘图包入口。

对外统一导出全部绘图函数，调用方只需 `from plot import ...`。
按功能划分的子模块：
- prediction  单样本预测结果图
- training    训练过程曲线
- evaluation  评估结果图与门控诊断
- comparison  多模型性能对比图
"""

from .prediction import (
    save_prediction_figures,
    plot_prediction_material,
    plot_prediction_support,
    plot_prediction_region,
    plot_prediction_damage_probability,
    plot_prediction_summary,
)
from .training import plot_history
from .evaluation import plot_evaluation, save_gate_diagnostics
from .comparison import (
    plot_method_comparison,
    plot_paper_figures,
    plot_domain_comparison,
    extract_reference_summaries,
)


def load_v2_sample(*args, **kwargs):
    from .data_visualization import load_v2_sample as implementation

    return implementation(*args, **kwargs)


def plot_sample_responses(*args, **kwargs):
    from .data_visualization import plot_sample_responses as implementation

    return implementation(*args, **kwargs)


def plot_temperature_displacement(*args, **kwargs):
    from .data_visualization import (
        plot_temperature_displacement as implementation,
    )

    return implementation(*args, **kwargs)


def plot_vibration_timeseries(*args, **kwargs):
    from .data_visualization import plot_vibration_timeseries as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "save_prediction_figures",
    "plot_prediction_material",
    "plot_prediction_support",
    "plot_prediction_region",
    "plot_prediction_damage_probability",
    "plot_prediction_summary",
    "plot_history",
    "plot_evaluation",
    "save_gate_diagnostics",
    "plot_method_comparison",
    "plot_paper_figures",
    "plot_domain_comparison",
    "extract_reference_summaries",
    "load_v2_sample",
    "plot_sample_responses",
    "plot_temperature_displacement",
    "plot_vibration_timeseries",
]
