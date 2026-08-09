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
    data_root = "E:/working/DL_data/data_new"
    corrected_data_dir = data_root
    reference_data_dir = os.path.join(data_root, "reference_model_dataset")
    data_dir = corrected_data_dir
    save_root = os.path.join(_DIR, "checkpoints")
    log_root = os.path.join(_DIR, "logs")
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

    fusion_branch_weight = 0.0
    fusion_gate_weight = 0.0
    fusion_warm_start = True
    fusion_train_alpha_only = False
    fusion_anchor = "static"
    fusion_safety_source = "fused"
    fusion_alpha_init = [0.90, 0.90, 0.90, 0.90, 0.20, 0.90]
    fusion_alpha_l2_weight = 0.02
    fusion_gate_bias = -2.0
    fusion_per_class_gate_bias = [-2.5, -2.5, -2.5, -2.5, -0.5, -2.5]
    fusion_strong_dynamic_classes = [0, 1, 2, 3, 5]
    fusion_per_class_gate_weight = 0.0
    fusion_margin_weight = 0.0
    fusion_margin = 0.015
    fusion_static_margin_weight = 0.0
    fusion_static_margin = 0.002
    fusion_modality_dropout = 0.0
    fusion_freeze_dynamic_epochs = 0

    label_smoothing = 0.05
    ema_decay = 0.999
    fusion_contrastive_weight = 0.20
    fusion_contrastive_temperature = 0.07
    fusion_stage1_epochs = 220
    fusion_stage2_epochs = 160
    fusion_stage3_epochs = 300
    fusion_stage1_lr = 5e-4
    fusion_stage2_lr = 1e-4
    fusion_stage3_lr = 1e-5
    fusion_use_legacy_schedule = False

    seed = 42
    num_workers = 0 if sys.platform == "win32" else 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dynamic_augment = True
    aug_prob = 0.30
    aug_noise_std = 0.015
    aug_amp_scale = 0.05
    aug_time_crop_ratio = 0.96
    aug_time_shift_ratio = 0.02
    aug_sensor_dropout = 0.03
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
