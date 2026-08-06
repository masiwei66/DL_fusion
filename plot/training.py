"""训练过程曲线绘图。

将训练过程中每个 epoch 记录的损失、F1、准确率、精确率/召回率、AUC、学习率
等指标绘制为多子图曲线，用于训练结束后快速检查收敛情况，并可直接用于论文。
"""

from ._utils import _configure_chinese_font, _white_legend


def plot_history(history, save_path):
    """绘制训练历史曲线（2 行 3 列共 6 个子图）。

    若 matplotlib 不可用，则退化为输出等价的文本表格文件。
    """
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_chinese_font()
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 子图 1：损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in history], label="训练集", color="#2196F3")
        ax.plot(epochs, [h["val_loss"] for h in history], label="验证集", color="#FF5722")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失")
        ax.set_title("损失曲线")
        _white_legend(ax, loc="upper right")
        ax.grid(True, alpha=0.3)

        # 子图 2：F1 曲线（含阈值 0.5 的验证 F1）
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

        # 子图 3：准确率曲线（含完全匹配率）
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

        # 子图 4：精确率与召回率曲线（含阈值 0.5 的验证召回率）
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

        # 子图 5：AUC 曲线
        ax = axes[1, 1]
        ax.plot(epochs, [h["train_auc"] for h in history], label="训练集", color="#2196F3")
        ax.plot(epochs, [h["val_auc"] for h in history], label="验证集", color="#FF5722")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("AUC")
        ax.set_title("AUC")
        _white_legend(ax, loc="lower right")
        ax.grid(True, alpha=0.3)

        # 子图 6：学习率曲线
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
        # matplotlib 不可用时的文本导出回退
        txt_path = save_path.replace(".png", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\ttrain_f1\tval_f1\n")
            for h in history:
                f.write(
                    f"{h['epoch']}\t{h['train_loss']:.4f}\t{h['val_loss']:.4f}\t"
                    f"{h['train_f1']:.4f}\t{h['val_f1']:.4f}\n"
                )
        print(f"Saved history text: {txt_path}")
