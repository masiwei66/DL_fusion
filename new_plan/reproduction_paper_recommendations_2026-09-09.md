# 面向 FEM→真实 InSAR 研究的可复现论文定位

检索日期：2026-09-09  
研究路径：参数化 FEM 反事实源域 → FE 点位/垂向分量/参考基准/重访观测算子 → 稀疏真实垂向 InSAR 学生模型；源域振动（若后续启用）作为不可部署特权信息；按结构状态和激励进行严格泛化评估。

## 结论先行

最适合作为第一篇完整复现对象的是 **Ren, Liu & Ou (2025), Deformation Pattern Classification of Sea-Crossing Bridge InSAR Time Series Based on a Transfer Learning Framework**。它与当前课题共享“合成有标签序列→少量真实 InSAR 微调→概率输出”的任务形态，且有开放 PDF；但它使用解析形变模板，不是参数化 FEM 反事实状态和垂向观测算子，因此适合作为“任务/迁移流程基线”，不能作为你的方法创新来源。

第二优先级是 **Gosliga et al. (2020), On the application of domain adaptation in structural health monitoring**，用于复现 SHM 中 source-labelled/target-unlabelled 的域适配协议和四个案例；它能直接支撑 MMD/CORAL/DANN 对照，但需从正文获取数据或按论文案例重建数据。

第三优先级是 **Benfenati et al. (2025), Foundation Models for Structural Health Monitoring**，用于复现时序掩码预训练、异常检测和知识蒸馏工程流程；它不是 InSAR 或跨模态特权迁移，论文已明确 KD 也可能在小模型上增加噪声，因此应定位为“时序预训练/蒸馏实现基线”。

同时，必须加入两类非深度学习物理基线：**Zhou et al. (2025) TBSI + FEM updating** 和 **Lorenz et al. (2024) InSAR + geodetic + structural simulation**。否则无法判断提升来自 AI 还是来自温度/FEM 解释本身。

## P0：建议实际复现的论文

### 1. Ren et al. 2025：合成 InSAR 序列到真实桥梁的迁移

**文献**：Lichen Ren, Chengyin Liu, Jinping Ou, “Deformation Pattern Classification of Sea-Crossing Bridge InSAR Time Series Based on a Transfer Learning Framework”, *Remote Sensing*, 17(21), 3567, 2025。DOI: [10.3390/rs17213567](https://doi.org/10.3390/rs17213567)；开放 PDF：[MDPI PDF](https://www.mdpi.com/2072-4292/17/21/3567/pdf?version=1761660338)。OpenAlex 记录：W4415623543。

**论文任务**：PS-InSAR 获取跨海桥形变序列，GNSS 误差小于 10%；定义 stable、linear、step、piecewise-linear、power-law、temperature-related 六类；用大量带标签仿真序列训练 TCN、Transformer、TFT、ROCKET，再用少量真实数据微调，并进行加权集成。摘要报告 ROCKET 在仿真数据上的准确率 96.27%，真实结果以多模型置信度和全局一致性输出概率。

**你应复现的最小版本**：

1. 按论文六类模板生成与真实 InSAR 相同长度/采样间隔的序列；固定训练/验证/真实微调划分。
2. 复现 TCN、Transformer、TFT、ROCKET 四个模型，统一随机种子和训练预算。
3. 复现单模型置信度、全局一致性与加权集成，报告 accuracy、macro-F1、校准和置信区间。
4. 将“解析模板”替换为你的参数化 FEM→FE 点位→垂向分量→重访观测算子，保持分类头、微调协议和指标不变。

**对当前课题的价值**：提供最接近的 transfer-learning 任务协议和真实 InSAR 概率输出基线。你的增量应体现在 参数化 FEM 反事实状态生成、垂向观测空间对齐、可辨识性约束和（可选的）振动特权迁移；不能再把“仿真数据 + Transformer + 少量真实微调”作为新颖性。

**可复现性判断**：开放 PDF，摘要和 DOI 可核验；训练代码、真实数据和精确划分需阅读全文及补充材料核对，当前不能假设作者已公开代码。

### 2. Gosliga et al. 2020：SHM 域适配标准基线

**文献**：J. Gosliga et al., “On the application of domain adaptation in structural health monitoring”, *Mechanical Systems and Signal Processing*, 2020。DOI: [10.1016/j.ymssp.2019.106550](https://doi.org/10.1016/j.ymssp.2019.106550)；开放 PDF：[ScienceDirect PDF](https://www.sciencedirect.com/science/article/pii/S088832701930771X/pdf)。OpenAlex 记录：197 次引用（截至检索日）。

**论文任务**：讨论训练结构与测试结构分布不同的问题，以 labelled source、unlabelled target 为设定，展示三种 domain adaptation 技术和四个 SHM 案例，目标是让源域分类器泛化到不同结构/工况。

**你应复现的最小版本**：

1. 先按 source-only 建立严格未见结构状态/激励的分类基线。
2. 分别实现 feature-level MMD、CORAL 和 adversarial/DANN；所有方法共用编码器、批大小、epoch、seed 和目标域无标签输入。
3. 对比 random split 与 state-family split，报告 macro-AUPRC、危险等级 recall、ECE/Brier 和跨激励性能下降。
4. 在你的物理映射前后各跑一次，形成“域适配 + 物理观测算子”二维消融。

**对当前课题的价值**：这是 MMD/CORAL/DANN 的场景化出处，适合支撑你的标准域适配对照和“直接特征对齐可能不尊重观测机制”的论证。

**可复现性判断**：方法设定和案例由摘要核验；原始案例数据、特征工程和超参数需全文核对，若数据不可得，应使用你的 FEM-InSAR-like 数据做同协议重建，并明确标注“协议复现”而非“数据复现”。

### 3. Benfenati et al. 2025：SHM 时序预训练与蒸馏工程基线

**文献**：L. Benfenati et al., “Foundation Models for Structural Health Monitoring”, *IEEE Transactions on Sustainable Computing*, 2025。DOI: [10.1109/TSUSC.2025.3592097](https://doi.org/10.1109/TSUSC.2025.3592097)；全文：[arXiv:2404.02944v2](https://arxiv.org/html/2404.02944v2)；代码/权重：[eml-eda/tle-supervised](https://github.com/eml-eda/tle-supervised)。

**论文任务与实现**：三座意大利桥/高架桥的 100 Hz 垂向加速度；5 s 或 60 s 窗口转为 100×100 谱图；ViT 掩码自编码器以 0.8 mask ratio 预训练 200 epochs；下游做异常检测和交通回归；KD 用任务 MAE 与教师/学生预测 RMSE 各 0.5。

**你应复现的最小版本**：

1. 先在 FEM 动态振动上复现谱图、MAE 预训练和异常检测流程。
2. 再做“FEM 位移+振动教师 → InSAR 学生”的 generalized distillation；学生输入中禁止振动。
3. 复现无 KD、普通 KD、关系/状态蒸馏三组，并测试小模型容量；报告教师性能、学生性能、蒸馏增益和负迁移。

**边界**：该论文 KD 是同模态模型压缩，不是跨传感器特权信息迁移；其约 99.9% 异常检测结果来自特定加固前后振动任务，不能引用为 InSAR 损伤识别预期精度。

## P1：必须纳入的物理与传统对比

### 4. Zhou et al. 2025：TBSI + InSAR + FEM 更新

Yun Zhou et al., “Temperature-based structural identification of long-span bridges using InSAR observation and meteorological shared data”, *Advances in Structural Engineering*, 28(15), 2780–2802, 2025。DOI: [10.1177/13694332251340715](https://doi.org/10.1177/13694332251340715)；预印本：[SSRN 10.2139/ssrn.5101330](https://doi.org/10.2139/ssrn.5101330)。

摘要明确描述：气象数据计算结构温度，InSAR 位移作为输出，建立热-结构耦合 FEM，用 Differential Evolution 识别异常热膨胀系数并定位/量化假定损伤，案例量化误差在 5% 以内。

**复现目标**：实现温度回归、FEM thermal-structural updating、DE 参数识别；在相同 InSAR-like 测试集上报告区域定位、参数误差、危险等级召回。它是你的强物理反演基线，不是 AI 基线。

### 5. Lorenz et al. 2024：InSAR 与大地测量/结构模拟交叉验证

R. Lorenz et al., “Thermal deformation monitoring of a highway bridge: combined analysis of geodetic and satellite-based InSAR measurements with structural simulations”, *Journal of Civil Structural Health Monitoring*, 2024。DOI: [10.1007/s13349-024-00779-9](https://doi.org/10.1007/s13349-024-00779-9)；开放 PDF：[Springer PDF](https://link.springer.com/content/pdf/10.1007/s13349-024-00779-9.pdf)。

**复现目标**：统一 InSAR、传统大地测量和结构数值模拟的时间尺度/点位/温度对照，报告不同采样率和测点密度对解释稳定性的影响。它可直接指导你设计点位减少、重访时间改变和 LOS 噪声实验。

### 6. Zhou et al. 2024：PS-InSAR 桥梁温度变形工程基线

周云等，《基于 InSAR 技术的大跨桥梁温度变形监测研究》，《湖南大学学报（自然科学版）》51(3), 2024，DOI: [10.16339/j.cnki.hdxbzkb.2024026](https://doi.org/10.16339/j.cnki.hdxbzkb.2024026)。

论文使用 2017–2018 年 59 景 Sentinel-1，PS-InSAR 得到 LOS 位移，结合 SAR 几何求支座纵向位移，并用线性温度模型和 FEM 温度作用模拟进行对照；摘要报告 LOS 误差约在 [-10, 10] mm。

**复现目标**：先复现 LOS→结构方向转换、温度线性回归、FEM 位移趋势对照；再把你的学生模型输出与该工程基线进行同点位误差比较。

## P2：方法学对照论文（建议引用并实现简化版）

| 论文 | 用途 | 你需要实现的最小对照 |
|---|---|---|
| Lopez-Paz et al., “Unifying distillation and privileged information”, ICLR 2016, [arXiv:1511.03643](https://arxiv.org/abs/1511.03643) | FEM 振动作为 privileged information 的理论近邻 | generalized distillation；与观测可辨识性加权蒸馏比较 |
| Hinton et al., “Distilling the Knowledge in a Neural Network”, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | 普通 soft-target KD 参照 | logits KD + hard-label loss，温度和权重做验证集选择 |
| Sun & Saenko, “Deep CORAL”, [arXiv:1607.01719](https://arxiv.org/abs/1607.01719) | 二阶统计域对齐 | encoder feature covariance alignment |
| Tobin et al., “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World”, IROS 2017, [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | 仿真到真实的随机化思想 | 温度、LOS 几何、点位误差、噪声和缺测随机化 |
| Nie et al., “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)”, ICLR 2023, [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) | InSAR 短时序编码器 | 缩小 patch 的 PatchTST 与 MLP/GRU 对照 |
| Zeng et al., “Are Transformers Effective for Time Series Forecasting?”, AAAI 2023, [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | 防止复杂模型无必要 | DLinear/一层线性/温度回归基线 |
| Soni et al., “Prediction of InSAR deformation time-series using improved LSTM deep learning model”, *Scientific Reports* 2024, [10.1038/s41598-024-83084-1](https://doi.org/10.1038/s41598-024-83084-1) | InSAR 时序预测邻近 | LSTM 预测与异常残差评分 |
| Ma et al., “Prediction of InSAR time-series deformation using deep convolutional neural networks”, 2019, [10.1080/2150704X.2019.1692390](https://doi.org/10.1080/2150704X.2019.1692390) | InSAR CNN 时序邻近 | 1D/2D CNN 预测基线 |

## 补充候选：用于真实目标域和特权信息边界

这些论文适合第二轮复现或写作支撑，不建议在第一轮同时全部实现：

| 论文 | 复现定位 |
|---|---|
| Milillo et al., “Pre-Collapse Space Geodetic Observations of Critical Infrastructure: The Morandi Bridge, Genoa, Italy”, *Remote Sensing* 2019, [10.3390/rs11121403](https://doi.org/10.3390/rs11121403) | 真实 InSAR 事故前异常与时间/空间连续性评价；不能把异常趋势直接当分类标签 |
| Ng et al., “Time-Series Analysis on PS-InSAR Derived Displacements of the Hong Kong–Zhuhai–Macao Bridge”, *Remote Sensing* 2021, [10.3390/rs13040546](https://doi.org/10.3390/rs13040546) | 86 景 Sentinel-1A、温度效应、EMD/STL 和重访时序预处理 |
| Cavalagli et al., “Interpretation of Bridge Health Monitoring Data from Satellite InSAR Technology”, *Remote Sensing* 2023, [10.3390/rs15215242](https://doi.org/10.3390/rs15215242) | 环境载荷—LOS 位移的物理解释与无强标签目标域评价 |
| Pimentel et al., “On statistic alignment for domain adaptation in structural health monitoring”, 2022, [10.1177/14759217221110441](https://doi.org/10.1177/14759217221110441) | 类别不平衡、partial/open-set DA；适合补强 CORAL/MMD 对照 |
| “Domain adaptation for structural health monitoring via physics-informed and self-attention-enhanced generative adversarial learning”, 2024, [10.1016/j.ymssp.2024.111236](https://doi.org/10.1016/j.ymssp.2024.111236) | physics-informed DA 对照；注意其物理约束不等于 InSAR LOS 观测算子 |
| “A domain adaptation approach to damage classification with an application to bridge monitoring”, 2024, [10.1016/j.ymssp.2024.111135](https://doi.org/10.1016/j.ymssp.2024.111135) | 最贴近桥梁 FEM/监测域分类的应用近邻，需核对源/目标传感器和标签协议 |
| Capellini et al., “Foundations and applicability of transfer learning for structural health monitoring of bridges”, 2023, [10.1016/j.ymssp.2023.110766](https://doi.org/10.1016/j.ymssp.2023.110766) | 迁移学习适用边界与负结果依据 |
| Vapnik & Vashist, “A new learning paradigm: Learning using privileged information”, 2009, [10.1016/j.neunet.2009.06.042](https://doi.org/10.1016/j.neunet.2009.06.042) | FEM 振动→InSAR 学生的 LUPI/SVM+ 理论基线 |
| Vapnik et al., “Learning using privileged information: SVM+ and weighted SVM”, 2014, [10.1016/j.neunet.2014.02.002](https://doi.org/10.1016/j.neunet.2014.02.002) | 特权模态蒸馏的非深度 baseline；可用 [arXiv:1306.3161](https://arxiv.org/abs/1306.3161) 重建 |

## 推荐的复现实验包

### 最小可发表对比矩阵

1. 物理基线：温度线性回归、Zhou-TBSI/DE、FEM 参数更新。
2. 源域基线：FEM source-only、FEM 位移-only、FEM 位移+振动融合。
3. 目标域基线：InSAR-only 的线性/GRU/PatchTST/TCN。
4. 域适配：MMD、CORAL、DANN；每个方法都做“无物理映射”和“有物理映射”。
5. 蒸馏：普通 KD、Lopez-Paz generalized distillation、振动特权 KD、关系/状态一致性蒸馏。
6. 完整方法：物理观测算子 + 振动特权教师 + 状态条件对齐 + 多激励一致性 + 质量/mask。

### 复现顺序

先复现 Ren 的合成→真实分类流程；然后复现 source-only/MMD/CORAL/DANN；再复现普通 KD；最后替换为参数化 FEM→垂向观测算子并加入事件级弱监督。振动特权迁移只有在动力 FEM 具备可信边界条件时再加入。每一步沿用同一 state-family split、5 个 seed（13/29/42/71/101）和相同测试扰动。

### 必须记录的证据

- “数据复现”还是“协议复现”；
- 论文原始数据/代码 URL、commit 或下载日期；
- 训练/验证/测试划分是否可获得；
- 真实目标域是否有强标签；
- 论文指标与本项目指标的可比性；
- 论文方法在你的观测算子下是否出现负迁移。

## 新颖性边界

不能单独宣称首次的组件：Transformer、GNN、MMD、CORAL、DANN、LOS 投影、温度建模、FEM、知识蒸馏、合成→真实微调、概率输出。更稳妥的核心表述是：**以参数化 FEM 反事实状态为源域，在与真实垂向 InSAR 一致的点位/基准/时间观测空间中进行状态迁移，并用事件级弱监督和不确定性控制减少模型误差导致的负迁移**。这一表述只有在上述基线、消融、未见激励和真实 InSAR 案例完成后才可写成已验证贡献。

## 检索来源与证据等级

- OpenAlex API（polite pool，2026-09-09）用于题名、DOI、年份、引用量和开放获取入口核对；例如 Ren：W4415623543，Zhou TBSI：DOI 10.1177/13694332251340715，Gosliga：DOI 10.1016/j.ymssp.2019.106550。
- Ren、Gosliga、Zhou TBSI、Lorenz 的摘要/出版商元数据用于任务描述；Benfenati 使用 arXiv v2 全文及其 GitHub 仓库核对实现细节。
- 对未获得全文或代码的论文，本文只给出“需全文核对”的复现建议，不把数据规模、超参数或代码可用性写成确定事实。


