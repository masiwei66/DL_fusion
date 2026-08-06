"""评估结果图与诊断报告。

对测试集整体评估结果绘图：
- 多标签混淆矩阵、各材料分类指标、ROC 曲线、预测概率分布
- 融合模型门控/分支诊断信息导出（JSON）
"""

import json

import numpy as np

from ._config import CANDIDATE_IDS
from ._utils import _configure_chinese_font, _style_result_axes, _white_legend


def plot_evaluation(y_true, y_pred, y_prob, save_path):
    """绘制整体评估四联图：混淆矩阵 / 各材料指标 / ROC 曲线 / 概率分布。

    参数:
        y_true: 真实标签矩阵
        y_pred: 预测标签矩阵
        y_prob: 预测概率矩阵
        save_path: 输出图片保存路径
    """
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

        # 左上：多标签混淆矩阵（每个材料一行的 TN/FP/FN/TP 统计）
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

        # 右上：各材料 F1/精确率/召回率对比柱状图
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

        # 左下：ROC 曲线（逐材料绘制）
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

        # 右下：正常/损伤样本的预测概率分布直方图
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
    """将融合模型的诊断指标（alpha 权重、门控、分支分歧度）导出为 JSON 报告。"""
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
