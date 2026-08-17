"""桥梁多工况安全评估的全局配置。"""

import os
import sys

import torch

try:
    from .safety_rules import (
        CANDIDATE_MATERIAL_IDS,
        REGION_DEFINITIONS,
        SUPPORT_SETTLEMENT_MAX_MM,
        SUPPORT_SETTLEMENT_NODES,
    )
except ImportError:  # 支持从本文件夹直接运行脚本。
    from safety_rules import (
        CANDIDATE_MATERIAL_IDS,
        REGION_DEFINITIONS,
        SUPPORT_SETTLEMENT_MAX_MM,
        SUPPORT_SETTLEMENT_NODES,
    )


_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """训练、模型、数据及工程规则相关的超参数。"""

    # V2 温度状态数据集是标准训练输入。
    data_root = "E:/working/DL_data/temperature_multistep_json_result"
    corrected_data_dir = data_root
    reference_data_dir = os.path.join(data_root, "reference_model_dataset")
    data_dir = corrected_data_dir
    save_root = os.path.join(_DIR, "checkpoints")
    log_root = os.path.join(_DIR, "logs")
    experiment_root = os.path.join(_DIR, "runs")
    data_version = "data_new"
    run_id = None
    split_manifest = None
    model_type = "fusion"
    run_name = "fusion"

    pos_dim = 16
    static_dim = 128
    dynamic_dim = 128
    fusion_dim = 128
    n_materials = len(CANDIDATE_MATERIAL_IDS)
    n_supports = len(SUPPORT_SETTLEMENT_NODES)
    n_regions = len(REGION_DEFINITIONS)
    n_safety_levels = 4
    support_disp_scale_mm = SUPPORT_SETTLEMENT_MAX_MM
    static_steps = 6
    static_channels = 3
    n_static_nodes = 6
    dynamic_channels = 3
    n_dynamic_nodes = 6
    temperature_condition_scale = 0.1
    static_node_ids = None
    dynamic_node_ids = None
    support_nodes = SUPPORT_SETTLEMENT_NODES
    region_names = list(REGION_DEFINITIONS.keys())

    batch_size = 32
    lr = 5e-4
    weight_decay = 2e-4
    n_epochs = 300
    early_stop = 40
    min_epochs = 100
    dynamic_only_min_epochs = 150
    selection_metric = "f1_auc"
    threshold_tuning = True
    threshold_recall_constraints = {3: 0.70, 5: 0.70}

    train_ratio = 0.70
    val_ratio = 0.15
    split_strategy = "group"
    group_by = "structural_state"

    pos_weight = 5.0
    aux_weight = 0.2
    support_cls_weight = 0.8
    support_reg_weight = 0.4
    region_weight = 0.5
    global_state_weight = 0.5
    rule_consistency_weight = 0.05

    # 融合策略的参数
    #   - 开启后，总损失额外加上静态/动态两个分支各自的分类损失，防止"融合头学好了但分支退化"
    #   - 0.0 = 关闭
    fusion_branch_weight = 0.0
    fusion_gate_weight = 0.0
    #   - 训练融合模型前，先把训练好的 static_only 和 dynamic_only 的分支权重加载进来，而不是从零开始
    #   - 好处：训练快、更稳；"warm start"就是"热启动"，站在前人肩膀
    fusion_warm_start = True
    #  — 是否只训练融合权重 α
    fusion_train_alpha_only = False
    # — 门控融合头的锚点分支
    #   - 只在 ReliabilityGatedFusionHead 里用：以哪个分支为基础、另一个分支以"修正"方式加入（static=静态为锚，dynamic=动态为锚）
    #   - 当前主模型不用它（ClasswiseLateFusionHead 没有锚点概念），这是给阶段 3 备用融合头准备的
    fusion_anchor = "static"
    #— 安全任务的"信息来源"
    fusion_safety_source = "fused"
    fusion_alpha_init = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    fusion_alpha_l2_weight = 0.02
    # — 门控偏置的初始值
    #   - 门控 = sigmoid(..., 偏置)。初始偏置 -2.0 → sigmoid(-2) ≈ 0.12，即门控初始几乎关闭（默认信任锚点分支），让模型自己决定何时开门
    fusion_gate_bias = -2.0
    #  — 逐材料门控偏置
    #   - 大部分材料 -2.5（更保守、更不开门），索引 4（材料 24）是 -0.5（初始就允许门控开一些）——和 fusion_alpha_init 里材料 24 信动态的逻辑一致
    fusion_per_class_gate_bias = [-2.5, -2.5, -2.5, -2.5, -0.5, -2.5]
    #  — 逐类门控抑制
    #   - 组合使用：对"动态分支已经很强的材料类别"（索引 0,1,2,3,5），额外惩罚它们打开门控——因为那些类别动态已经很好，不需要静态修正来捣乱
    #   - 权重 0.0 = 当前关闭（这两个是为门控融合头准备的）
    fusion_strong_dynamic_classes = [0, 1, 2, 3, 5]
    fusion_per_class_gate_weight = 0.0
    #  — 融合 vs 动态的"保底约束"
    #   - 公式：relu(融合损失 - 动态分支损失 + margin)——要求融合结果不比动态分支差（允许差一点点，margin=0.015 是容差）
    #   - 防止融合模型把动态的强项搞砸
    #   - 0.0 = 关闭
    fusion_margin_weight = 0.0
    fusion_margin = 0.015
    #  — 同上，对比对象换成静态分支
    #   - 要求融合不比静态分支差（容差 0.002 更小）
    #   - 0.0 = 关闭
    fusion_static_margin_weight = 0.0
    fusion_static_margin = 0.002
    # — 模态随机失活
    # - 训练时随机丢弃一个模态（如这次只给静态、不给动态），逼模型学会"单模态也能撑住"
    # - 0.0 = 关闭。这是为鲁棒性准备的（阶段 5 缺测实验会用到）
    fusion_modality_dropout = 0.0
    # — 前 N 轮冻结动态分支
    fusion_freeze_dynamic_epochs = 0


    # 把硬标签"软化"。0.05 意味着标签从 0 / 1 变成 0.025 / 0.975——正确答案让出 5% 的概率给错误答案
    label_smoothing = 0.05
    #  — 指数滑动平均（EMA）
    ema_decay = 0.999

    #   状态 A 的静态特征 ←——拉近——→ 状态 A 的动态特征
    #   状态 B 的静态特征 ←——推远——→ 状态 A 的动态特征
    fusion_contrastive_weight = 0.20
    #  作用：控制对比损失对"难分样本"的敏感度。温度越低，对接近的负样本惩罚越尖锐、区分越狠；温度越高，对所有样本一视同仁。
    fusion_contrastive_temperature = 0.07

    #   三阶段训练规划（融合模型的核心策略）
    #   融合模型不是一口气训练完的，而是分三个阶段，每阶段不同的轮数和学习率，像"先学骨架，再学拼装，最后精雕细琢"：
    fusion_stage1_epochs = 220
    fusion_stage2_epochs = 160
    fusion_stage3_epochs = 300
    fusion_stage1_lr = 5e-4
    fusion_stage2_lr = 1e-4
    fusion_stage3_lr = 1e-5
    # 作用：兼容开关。做消融实验（比如 A8"训练策略是否影响结论"）时，对比新旧调度对结果的影响，用它一键切换。
    fusion_use_legacy_schedule = False

    seed = 42
    num_workers = 0 if sys.platform == "win32" else 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 动态分支（振动数据）的数据增强参数
    dynamic_augment = True
    #  — 增强触发概率
    aug_prob = 0.30
    # — 加噪声强度
    aug_noise_std = 0.015
    # — 幅值缩放幅度
    aug_amp_scale = 0.05
    #  — 时间裁剪比例
    aug_time_crop_ratio = 0.96
    # — 时间平移比例
    aug_time_shift_ratio = 0.02
    # — 测点失活
    aug_sensor_dropout = 0.03
    # — 通道失活
    aug_channel_dropout = 0.015

    dynamic_only_aug_prob = 0.25
    dynamic_only_aug_noise_std = 0.012
    dynamic_only_aug_amp_scale = 0.04
    dynamic_only_aug_time_crop_ratio = 0.97
    dynamic_only_aug_time_shift_ratio = 0.02
    dynamic_only_aug_sensor_dropout = 0.03
    dynamic_only_aug_channel_dropout = 0.015

    fusion_aug_prob = 0.30
    fusion_aug_noise_std = 0.012
    fusion_aug_amp_scale = 0.04
    fusion_aug_time_crop_ratio = 0.96
    fusion_aug_time_shift_ratio = 0.02
    fusion_aug_sensor_dropout = 0.03
    fusion_aug_channel_dropout = 0.015

    static_augment = True
    static_aug_prob = 0.55
    static_aug_noise_std = 0.03
    static_aug_scale = 0.08
    static_aug_sensor_dropout = 0.06

    static_only_aug_prob = 0.50
    static_only_aug_noise_std = 0.03
    static_only_aug_scale = 0.08
    static_only_aug_sensor_dropout = 0.06

    fusion_static_aug_prob = 0.45
    fusion_static_aug_noise_std = 0.025
    fusion_static_aug_scale = 0.06
    fusion_static_aug_sensor_dropout = 0.05

    def __init__(self):
        self.save_dir = self.method_dir(self.save_root, self.model_type)
        self.log_dir = self.method_dir(self.log_root, self.model_type)
        self.run_name = self.model_type
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def method_dir(root, model_type):
        names = {
            "fusion": "fusion",
            "static_only": "static_only",
            "dynamic_only": "dynamic_only",
        }
        if model_type not in names:
            raise ValueError(f"Unknown model type: {model_type}")
        return os.path.join(root, names[model_type])

    def set_model_type(self, model_type):
        self.model_type = model_type
        self.run_name = model_type
        self.save_dir = self.method_dir(self.save_root, model_type)
        self.log_dir = self.method_dir(self.log_root, model_type)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


CANDIDATE_IDS = CANDIDATE_MATERIAL_IDS
SUPPORT_NODES = SUPPORT_SETTLEMENT_NODES
REGION_NAMES = list(REGION_DEFINITIONS.keys())
STATIC_NODES = [43, 65, 87, 153, 175, 197]
DYNAMIC_NODES = [23, 67, 133, 155, 177, 645]
