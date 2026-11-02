from __future__ import annotations

from pathlib import Path

from docx import Document


SOURCE = Path(r"E:\working\my_project\具体工作\实验相关\当前对比试验流程实验报告.docx")


def set_text(paragraph, text: str) -> None:
    paragraph.clear()
    paragraph.add_run(text)


def update_table(table) -> None:
    rows = [
        ("模型", "统一节点输入", "模型内部适配", "关键结构"),
        ("LSTM", "[B, 6, P, 9]", "无图掩码池化为 [B, 6, 9] 后输入 LSTM", "2层，hidden=128，dropout=0.1"),
        ("GRU", "[B, 6, P, 9]", "无图掩码池化为 [B, 6, 9] 后输入 GRU", "2层，hidden=128，dropout=0.1"),
        ("Transformer", "[B, 6, P, 9]", "无图掩码池化为 [B, 6, 9] 后输入 Transformer", "2层，d_model=128，4头，FFN=256"),
        ("ResNet-50 1D", "[B, 6, P, 9]", "无图掩码池化为 [B, 6, 9]，转置为 [B, 9, 6]", "layers=[3,4,6,3]，base_width=32"),
        ("GNN-LSTM", "[B, 6, P, 9]", "KNN-GCN得到 [B, 6, 64] 后输入 LSTM", "K=8，GCN=2层，LSTM hidden=128"),
        ("XGBoost", "固定长度表格特征", "独立的固定宽度统计特征适配器", "300棵树，depth=6，lr=0.05"),
    ]
    for row, values in zip(table.rows, rows):
        for cell, value in zip(row.cells, values):
            cell.text = value


def main() -> None:
    doc = Document(SOURCE)
    p = doc.paragraphs

    set_text(p[92], "4.4.6 共享节点级输入与模型适配")
    set_text(
        p[93],
        "为保证不同神经网络模型使用相同的观测信息，所有模型首先从统一的 WindowBatch 构造节点级输入。设批次大小为 B，时间窗口长度为 W=6，观测点数量为 P，每个观测点包含 9 个特征，则共享输入张量为 G ∈ R^(B×6×P×9)。P 为当前批次中经过补齐后的观测点容量，不同样本的实际观测点数量可以不同；补齐点通过节点掩码排除，不参与后续特征聚合和图传播。",
    )
    set_text(p[94], "有效节点掩码定义为：")
    set_text(p[95], "Mᵦ,ₚ = mᵖʳᵉˢᵉⁿᵗᵦ,ₚ · mᵐᵃᵖᵦ,ₚ")
    set_text(p[96], "其中，mᵖʳᵉˢᵉⁿᵗ 表示观测点是否为真实点而非补齐点，mᵐᵃᵖ 表示该点是否成功映射。")
    set_text(p[97], "实际观测掩码定义为：")
    set_text(p[98], "Aᵦ,ₜ,ₚ = mᵒᵇˢᵦ,ₜ,ₚ · Mᵦ,ₚ")
    set_text(
        p[99],
        "其中，mᵒᵇˢ 表示该观测点在当前时刻是否有效。共享节点特征按以下顺序排列：位移、观测有效掩码、观测质量、x 坐标、y 坐标、映射距离、映射有效标志、温度和采集时间间隔。",
    )
    set_text(p[100], "第 b 个样本、第 t 个时刻和第 p 个观测点的 9 维输入为：")
    set_text(
        p[101],
        "gᵦ,ₜ,ₚ = [uᵦ,ₜ,ₚAᵦ,ₜ,ₚ，Aᵦ,ₜ,ₚ，qᵦ,ₜ,ₚAᵦ,ₜ,ₚ，xᵦ,ₚMᵦ,ₚ，yᵦ,ₚMᵦ,ₚ，dᵦ,ₚMᵦ,ₚ，mᵐᵃᵖᵦ,ₚMᵦ,ₚ，Tᵦ,ₜmᵀᵦ,ₜ，Δtᵦ,ₜmᵀᵦ,ₜ]。",
    )
    set_text(p[102], "因此，共享节点级输入张量为：G = [gᵦ,ₜ,ₚ] ∈ R^(B×6×P×9)。")
    set_text(p[103], "对于 LSTM、GRU、Transformer 和 ResNet，模型不显式构造空间邻接关系，而是进行无图掩码池化：")
    set_text(p[104], "ḡᵦ,ₜ = Σₚ Mᵦ,ₚ gᵦ,ₜ,ₚ / max(Σₚ Mᵦ,ₚ，1)。")
    set_text(p[105], "池化后得到 Ḡ ∈ R^(B×6×9)，再分别输入 LSTM、GRU、Transformer 或 ResNet-50 1D。")
    set_text(p[106], "其中 ResNet-50 1D 将 Ḡ 转置为卷积输入 [B,9,6]；因此各模型共享同一节点级输入契约，但在内部使用不同的结构适配方式。")
    set_text(p[107], "区域统计特征接口 temporal_features 保留用于后续消融实验，不作为当前主对比实验的默认输入。")

    set_text(p[108], "4.4.7 GNN-LSTM节点级空间适配")
    set_text(p[109], "GNN-LSTM 使用与其他神经模型完全相同的节点级输入 G ∈ R^(B×6×P×9)，但不直接对节点特征求平均，而是根据观测点坐标构造每个样本的动态 KNN 图。")
    set_text(p[110], "图卷积的初始节点表示为 H⁽⁰⁾ᵦ,ₜ = Gᵦ,ₜ。第 l 层图卷积表示为：")
    set_text(p[111], "H⁽ˡ⁺¹⁾ᵦ,ₜ = σ(Ãᵦ H⁽ˡ⁾ᵦ,ₜ W⁽ˡ⁾)。")
    set_text(p[112], "其中 Ãᵦ 为根据观测点坐标构造并归一化的 KNN 邻接矩阵，K=8。")
    set_text(p[113], "经过两层图卷积后，节点隐藏表示为 H⁽ᴸ⁾ ∈ R^(B×P×64)。")
    set_text(p[114], "随后进行有效节点池化：")
    set_text(p[115], "zᵦ,ₜ = Σₚ Mᵦ,ₚ H⁽ᴸ⁾ᵦ,ₜ,ₚ / max(Σₚ Mᵦ,ₚ，1)。")
    set_text(p[116], "6 个时刻组成 Z ∈ R^(B×6×64)，再输入 LSTM 进行时间建模。")
    set_text(p[117], "因此，GNN-LSTM 与其他模型的观测来源和原始特征完全一致，差异仅在于其使用 KNN-GCN 显式传播节点间空间信息。")
    set_text(p[118], "")
    set_text(p[119], "")
    set_text(p[120], "")

    set_text(p[138], "5.3 共享节点级输入与无图时序模型")
    set_text(
        p[139],
        "LSTM、GRU、Transformer 和 ResNet-50 1D 均从共享节点级输入 [B,6,P,9] 出发。模型内部对有效节点进行无图掩码池化，得到 [B,6,9] 的区域级时序表示。LSTM、GRU 和 Transformer 直接处理该序列，ResNet-50 1D 将其转置为 [B,9,6] 后进行一维卷积。该过程使用的是共享节点级特征的平均适配，不是原先单独构造的均值、标准差、最小值和最大值统计量。",
    )
    set_text(p[140], "原有区域统计特征 temporal_features 仍保留为可选消融接口，用于比较预先统计压缩与共享节点级输入之间的差异，不纳入当前主对比实验。")
    set_text(p[141], "5.4 GNN-LSTM节点级空间适配")
    set_text(
        p[142],
        "GNN-LSTM 与其他神经模型接收相同的 [B,6,P,9] 节点级输入。模型根据节点坐标动态构造 KNN 图，经两层 GCN 将每个时刻的节点表示映射为 64 维，再进行有效节点池化形成 [B,6,64]，最后由两层 LSTM 建模时间变化。该模型与普通时序模型的区别是显式利用空间邻接关系，而不是使用不同来源或不同质量等级的观测特征。",
    )

    set_text(
        p[166],
        "本实验的公平性控制包括：所有神经模型使用相同的有限元观测生成过程、虚拟观测点采样范围、噪声和缺测设置、归一化统计量、时间窗口长度、结构状态划分、观测实现采样策略、批大小、训练轮数和优化器设置。所有神经模型均从相同的节点级输入 [B,6,P,9] 出发；普通时序模型采用无图节点池化，GNN-LSTM 采用基于坐标的 KNN-GCN。因此，本实验属于统一观测输入协议下的模型结构对比，而不是所有模型使用完全相同中间表示的比较。",
    )

    update_table(doc.tables[5])
    doc.save(SOURCE)


if __name__ == "__main__":
    main()
