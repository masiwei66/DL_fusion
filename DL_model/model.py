"""桥梁结构损伤识别的模型定义。

融合模型采用置信度感知、按材料类别设置的门控，并以动态预测为锚点。
仅动态（dynamic-only）模式是更强的基线（尤其针对材料 8/9/19/20/41），
因此融合以动态 logits 为锚点，让静态响应以门控修正的形式进入，
主要使静态优于动态的材料 24 受益。
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def _apply_temperature_condition(features, batch, scale):
    """在不增加检查点参数的前提下注入归一化后的温度信息。"""
    condition = batch.get("condition")
    if condition is None or float(scale) == 0.0:
        return features
    condition = condition.to(device=features.device, dtype=features.dtype)
    if condition.ndim == 1:
        condition = condition[:, None]
    basis = torch.linspace(
        -1.0,
        1.0,
        features.size(-1),
        device=features.device,
        dtype=features.dtype,
    )
    return features + float(scale) * condition[:, :1] * basis.unsqueeze(0)


class SafeBN(nn.BatchNorm1d):
    """BatchNorm1d 的包装器：当 batch_size=1 时跳过归一化。"""

    def forward(self, x):
        if x.size(0) <= 1:
            return x
        return super().forward(x)


class PositionEncoding(nn.Module):
    """用于 3D 传感器坐标的可学习傅里叶编码。

    桥梁模型中的坐标单位为毫米，数值可能较大。编码器首先将每个传感器
    布局转换为居中且经过尺度归一化的相对坐标，然后在小规模 MLP 之前
    加入傅里叶特征。这样既保留了几何信息的表达能力，又不会将原始坐标
    尺度注入到响应通道中。解决思路：坐标 → 归一化 → 傅里叶展开 → MLP
    直接坐标让网络"看不清"细微差异，sin/cos 把差异放大
    """

    def __init__(self, input_dim=3, pos_dim=16, num_freqs=4):
        # ：这里的频率是"正弦波震荡的快慢"——4 个频率 = 4 个粗细不同的刻度，让神经网络能从多个尺度分辨测点位置
        super().__init__()
        self.num_freqs = num_freqs
        fourier_dim = input_dim * (1 + 2 * num_freqs)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(fourier_dim, pos_dim * 2)),
            ('act1', nn.GELU()),
            ('fc2', nn.Linear(pos_dim * 2, pos_dim)),
            ('act2', nn.GELU()),
            ('ln', nn.LayerNorm(pos_dim)),
        ]))

    @staticmethod
    def _normalize_coords(coords):
        coords = coords.float()
        center = coords.mean(dim=0, keepdim=True)
        span = coords.max(dim=0, keepdim=True).values - coords.min(dim=0, keepdim=True).values
        scale = span.max().clamp(min=1.0)
        return (coords - center) / scale

    def forward(self, coords):
        coords = self._normalize_coords(coords)
        freqs = torch.arange(
            1, self.num_freqs + 1, device=coords.device, dtype=coords.dtype
        ).view(1, 1, -1)
        angles = coords.unsqueeze(-1) * freqs * torch.pi
        fourier = torch.cat([
            coords,
            torch.sin(angles).flatten(1),
            torch.cos(angles).flatten(1),
        ], dim=-1)
        # 每个测点的位置指纹——"原始坐标 + 每个频率下的 sin/cos 响应
        return self.mlp(fourier)


class PositionFiLM(nn.Module):
    """基于位置嵌入的特征级线性调制（FiLM） Feature-wise Linear Modulation。"""

    def __init__(self, pos_dim, feat_dim, scale=0.1):
        super().__init__()
        self.scale = scale
        self.to_gamma_beta = nn.Linear(pos_dim, feat_dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, feat, pos_emb):
        gamma, beta = self.to_gamma_beta(pos_emb).chunk(2, dim=-1)
        return feat * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * beta


class AttentionPool(nn.Module):
    """对传感器/节点 token 进行的小规模注意力池化。"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.score = nn.Sequential(OrderedDict([
            ('ln', nn.LayerNorm(dim)),
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('act1', nn.Tanh()),
            ('fc2', nn.Linear(hidden_dim, 1)),
        ]))

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens), dim=1)
        return (tokens * weights).sum(dim=1)


class StaticBranch(nn.Module):
    """多温度位移分支，带位置 FiLM。

    V2 版本有意移除了旧的应变（strain）输入。``strain`` 仍保留为可选
    参数，以便旧调用方仍可调用该分支，而网络契约仅由与温度对齐的位移
    定义。
    """

    def __init__(self, pos_dim=16, hidden_dim=128, out_dim=64,
                 n_steps=4, n_channels=3):
        super().__init__()
        self.pos_enc = PositionEncoding(pos_dim=pos_dim)

        node_input_dim = int(n_steps) * int(n_channels)
        self.node_mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(node_input_dim, hidden_dim)),
            ('bn1', SafeBN(hidden_dim)),
            ('act1', nn.GELU()),
            ('drop1', nn.Dropout(0.15)),
            ('fc2', nn.Linear(hidden_dim, hidden_dim // 2)),
            ('bn2', SafeBN(hidden_dim // 2)),
            ('act2', nn.GELU()),
            ('fc3', nn.Linear(hidden_dim // 2, out_dim)),
            ('ln3', nn.LayerNorm(out_dim)),
        ]))
        self.pos_film = PositionFiLM(pos_dim, out_dim, scale=0.15)
        self.pool = AttentionPool(out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, disp, strain=None, node_coords=None):
        if node_coords is None:
            node_coords = strain
        batch_size = disp.size(0)
        n_nodes = disp.size(2)

        disp_feat = disp.permute(0, 2, 1, 3).reshape(batch_size, n_nodes, -1)
        node_feat = disp_feat

        tokens = self.node_mlp(node_feat.reshape(-1, node_feat.size(-1)))
        tokens = tokens.view(batch_size, n_nodes, -1)

        pos = self.pos_enc(node_coords.to(disp.device))
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.pos_film(tokens, pos)

        return self.out_norm(self.pool(tokens))


class DynamicBranch(nn.Module):
    """加速度时间序列分支，带逐传感器的位置 FiLM。"""

    def __init__(self, pos_dim=16, hidden_dim=128, out_dim=64,
                 n_nodes=5, n_channels=3):
        super().__init__()
        self.pos_enc = PositionEncoding(pos_dim=pos_dim)
        self.sensor_film = PositionFiLM(pos_dim, int(n_channels), scale=0.1)
        in_channels = int(n_nodes) * int(n_channels)

        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels, 32, kernel_size=9, padding=4)),
            ('bn1', SafeBN(32)),
            ('act1', nn.GELU()),
            ('pool1', nn.MaxPool1d(2)),
            ('conv2', nn.Conv1d(32, 64, kernel_size=7, padding=3)),
            ('bn2', SafeBN(64)),
            ('act2', nn.GELU()),
            ('pool2', nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(64, 128, kernel_size=5, padding=2)),
            ('bn3', SafeBN(128)),
            ('act3', nn.GELU()),
        ]))

        self.temporal_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128, hidden_dim)),
            ('bn1', SafeBN(hidden_dim)),
            ('act1', nn.GELU()),
            ('drop1', nn.Dropout(0.35)),
            ('fc2', nn.Linear(hidden_dim, out_dim)),
            ('ln2', nn.LayerNorm(out_dim)),
        ]))

    def forward(self, ace, node_coords):
        batch_size = ace.size(0)
        device = ace.device

        pos = self.pos_enc(node_coords.to(device))
        pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, ace.size(1), -1, -1)
        ace = self.sensor_film(ace, pos)

        x = ace.permute(0, 2, 3, 1).reshape(batch_size, -1, ace.size(1))
        x = self.conv(x)
        x = self.temporal_pool(x).squeeze(-1)
        return self.head(x)


class PredictionHead(nn.Module):
    """分类头，外加一个辅助的比例因子回归头。"""

    def __init__(self, input_dim, n_materials):
        super().__init__()
        self.classifier = nn.Linear(input_dim, n_materials)
        self.regressor = nn.Linear(input_dim, n_materials)

    def forward(self, feat):
        return self.classifier(feat), self.regressor(feat)


class MultiTaskPredictionHead(nn.Module):
    """面向材料、支座沉降、区域风险与整体状态的多个任务头。"""

    def __init__(self, input_dim, config):
        super().__init__()
        self.n_regions = config.n_regions
        self.n_safety_levels = config.n_safety_levels
        self.material_classifier = nn.Linear(input_dim, config.n_materials)
        self.material_regressor = nn.Linear(input_dim, config.n_materials)
        self.support_classifier = nn.Linear(input_dim, config.n_supports)
        self.support_regressor = nn.Linear(input_dim, config.n_supports)
        self.region_classifier = nn.Linear(
            input_dim, config.n_regions * config.n_safety_levels
        )
        self.global_classifier = nn.Linear(input_dim, config.n_safety_levels)

    def forward(self, feat):
        region_logits = self.region_classifier(feat)
        region_logits = region_logits.view(
            feat.size(0), self.n_regions, self.n_safety_levels
        )
        return {
            'material_logits': self.material_classifier(feat),
            'material_reg': self.material_regressor(feat),
            'support_logits': self.support_classifier(feat),
            'support_reg': self.support_regressor(feat),
            'region_logits': region_logits,
            'global_logits': self.global_classifier(feat),
        }


class ClasswiseLateFusionHead(nn.Module):
    """面向静态与动态预测的轻量级逐类别后融合。

    每个材料类别使用一个可学习的静态权重：

        fused = alpha * static + (1 - alpha) * dynamic

    学习得到的 alpha 值易于汇报与解读：alpha 接近 1 表示该类更依赖
    静态响应，alpha 接近 0 表示该类更依赖动态响应。
    """

    def __init__(self, branch_dim, n_materials, alpha_init=None):
        super().__init__()
        self.static_head = PredictionHead(branch_dim, n_materials)
        self.dynamic_head = PredictionHead(branch_dim, n_materials)
        self.emb_proj = nn.Sequential(OrderedDict([
            ('ln1', nn.LayerNorm(branch_dim * 2)),
            ('fc1', nn.Linear(branch_dim * 2, branch_dim)),
            ('act1', nn.GELU()),
        ]))

        if alpha_init is None:
            alpha_init = torch.full((n_materials,), 0.7)
        else:
            alpha_init = torch.as_tensor(alpha_init, dtype=torch.float32)
            if alpha_init.numel() != n_materials:
                raise ValueError(
                    f'fusion_alpha_init length {alpha_init.numel()} '
                    f'!= n_materials {n_materials}'
                )
        alpha_init = alpha_init.clamp(0.01, 0.99)
        self.alpha_logit = nn.Parameter(torch.logit(alpha_init))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, static_feat, dynamic_feat):
        static_logits, static_reg = self.static_head(static_feat)
        dynamic_logits, dynamic_reg = self.dynamic_head(dynamic_feat)
        alpha = self.alpha.view(1, -1)

        logits = alpha * static_logits + (1.0 - alpha) * dynamic_logits
        reg_out = alpha * static_reg + (1.0 - alpha) * dynamic_reg
        alpha_mean = alpha.mean()
        fused_emb = self.emb_proj(torch.cat([
            static_feat,
            alpha_mean * static_feat + (1.0 - alpha_mean) * dynamic_feat,
        ], dim=1))

        aux = {
            'static_logits': static_logits,
            'dynamic_logits': dynamic_logits,
            'static_reg': static_reg,
            'dynamic_reg': dynamic_reg,
            'fusion_alpha': self.alpha.detach(),
        }
        return logits, reg_out, fused_emb, aux


class ReliabilityGatedFusionHead(nn.Module):
    """带逐类别门控的可靠性门控多模态融合。

    该头先预测两个分支的输出，再利用交互特征与分支置信度学习一个
    逐材料的修正门控。全程使用动态锚点——对于大多数材料，
    dynamic_only 是更强的基线。通过逐类别门控偏置与逐类别门控正则化，
    可以在动态已近乎完美的类别（如材料 20、41）上抑制门控，同时允许
    门控在静态提供互补信息的类别（如材料 24）上打开。
    """

    def __init__(self, branch_dim, fusion_dim, n_materials, dropout=0.2,
                 anchor='dynamic', gate_bias=-2.0, per_class_gate_bias=None):
        super().__init__()
        if anchor not in ('static', 'dynamic'):
            raise ValueError(f'unknown fusion anchor: {anchor}')
        self.anchor = anchor
        interaction_dim = branch_dim * 4

        self.static_norm = nn.LayerNorm(branch_dim)
        self.dynamic_norm = nn.LayerNorm(branch_dim)
        self.static_head = PredictionHead(branch_dim, n_materials)
        self.dynamic_head = PredictionHead(branch_dim, n_materials)

        gate_input_dim = interaction_dim + n_materials * 3
        self.gate = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(gate_input_dim, fusion_dim)),
            ('ln1', nn.LayerNorm(fusion_dim)),
            ('act1', nn.GELU()),
            ('drop1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(fusion_dim, n_materials)),
        ]))
        self.cls_residual = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(gate_input_dim, fusion_dim)),
            ('ln1', nn.LayerNorm(fusion_dim)),
            ('act1', nn.GELU()),
            ('drop1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(fusion_dim, n_materials)),
        ]))
        self.reg_residual = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(gate_input_dim, fusion_dim)),
            ('ln1', nn.LayerNorm(fusion_dim)),
            ('act1', nn.GELU()),
            ('drop1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(fusion_dim, n_materials)),
        ]))
        self.emb_proj = nn.Sequential(OrderedDict([
            ('ln1', nn.LayerNorm(branch_dim * 2)),
            ('fc1', nn.Linear(branch_dim * 2, branch_dim)),
            ('act1', nn.GELU()),
        ]))
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        with torch.no_grad():
            if per_class_gate_bias is not None:
                bias_tensor = torch.as_tensor(per_class_gate_bias, dtype=torch.float32)
                if bias_tensor.shape[0] != n_materials:
                    raise ValueError(
                        f'per_class_gate_bias length {bias_tensor.shape[0]} '
                        f'!= n_materials {n_materials}'
                    )
                self.gate.fc2.bias.copy_(bias_tensor)
            else:
                self.gate.fc2.bias.fill_(float(gate_bias))

    @staticmethod
    def _confidence(logits):
        probs = torch.sigmoid(logits)
        return torch.abs(probs - 0.5) * 2.0

    def forward(self, static_feat, dynamic_feat):
        static_feat = self.static_norm(static_feat)
        dynamic_feat = self.dynamic_norm(dynamic_feat)

        static_logits, static_reg = self.static_head(static_feat)
        dynamic_logits, dynamic_reg = self.dynamic_head(dynamic_feat)

        interaction = torch.cat([
            static_feat,
            dynamic_feat,
            torch.abs(static_feat - dynamic_feat),
            static_feat * dynamic_feat,
        ], dim=1)
        confidence = torch.stack([
            self._confidence(static_logits),
            self._confidence(dynamic_logits),
            torch.abs(torch.sigmoid(static_logits) - torch.sigmoid(dynamic_logits)),
        ], dim=-1).flatten(1)
        gate_input = torch.cat([interaction, confidence], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))

        cls_residual = self.res_scale * self.cls_residual(gate_input)
        reg_residual = self.res_scale * self.reg_residual(gate_input)
        if self.anchor == 'dynamic':
            cls_correction = (static_logits - dynamic_logits) + cls_residual
            reg_correction = (static_reg - dynamic_reg) + reg_residual
            logits = dynamic_logits + gate * cls_correction
            reg_out = dynamic_reg + gate * reg_correction
            base_feat = dynamic_feat
            correction_feat = static_feat - dynamic_feat
        else:
            cls_correction = (dynamic_logits - static_logits) + cls_residual
            reg_correction = (dynamic_reg - static_reg) + reg_residual
            logits = static_logits + gate * cls_correction
            reg_out = static_reg + gate * reg_correction
            base_feat = static_feat
            correction_feat = dynamic_feat - static_feat

        emb_gate = gate.mean(dim=1, keepdim=True)
        fused_emb = self.emb_proj(torch.cat([
            base_feat,
            base_feat + emb_gate * correction_feat,
        ], dim=1))

        aux = {
            'static_logits': static_logits,
            'dynamic_logits': dynamic_logits,
            'static_reg': static_reg,
            'dynamic_reg': dynamic_reg,
            'gate': gate,
            'anchor': self.anchor,
            'branch_disagreement': confidence.view(confidence.size(0), -1, 3)[:, :, 2],
        }
        self.last_gate = gate.detach()
        return logits, reg_out, fused_emb, aux


# 为旧版 import 与检查点元数据保留的向后兼容别名。
StaticAnchoredFusionHead = ReliabilityGatedFusionHead


class StaticOnlyModel(nn.Module):
    """仅静态（static-only）基线模型。"""

    def __init__(self, config):
        super().__init__()
        out_dim = config.fusion_dim // 2
        self.static_branch = StaticBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.static_dim,
            out_dim=out_dim,
            n_steps=getattr(config, 'static_steps', 4),
            n_channels=getattr(config, 'static_channels', 3),
        )
        self.temperature_condition_scale = getattr(
            config, "temperature_condition_scale", 0.1
        )
        self.head = MultiTaskPredictionHead(out_dim, config)

    def forward(self, batch, return_emb=False):
        feat = self.static_branch(batch['disp'], batch['strain'], batch['static_pos'])
        feat = _apply_temperature_condition(
            feat, batch, self.temperature_condition_scale
        )
        output = self.head(feat)
        if return_emb:
            return output, feat
        return output


class DynamicOnlyModel(nn.Module):
    """仅动态（dynamic-only）基线模型。"""

    def __init__(self, config):
        super().__init__()
        out_dim = config.fusion_dim // 2
        self.dynamic_branch = DynamicBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.dynamic_dim,
            out_dim=out_dim,
            n_nodes=getattr(config, 'n_dynamic_nodes', 5),
            n_channels=getattr(config, 'dynamic_channels', 3),
        )
        self.temperature_condition_scale = getattr(
            config, "temperature_condition_scale", 0.1
        )
        self.head = MultiTaskPredictionHead(out_dim, config)

    def forward(self, batch, return_emb=False):
        feat = self.dynamic_branch(batch['ace'], batch['dynamic_pos'])
        feat = _apply_temperature_condition(
            feat, batch, self.temperature_condition_scale
        )
        output = self.head(feat)
        if return_emb:
            return output, feat
        return output


class DualBranchFusion(nn.Module):
    """带轻量级逐类别后融合的双分支模型。

    支持通过 ``stage`` 控制的三种训练阶段：

    - ``'pretrain'``：使用对比损失的传统联合分支训练；
      forward 返回 (static_logits, static_reg, dynamic_logits, dynamic_reg,
      static_feat, dynamic_feat)。
    - ``'fusion_head'`` / ``'finetune'``：端到端后融合。
    """

    def __init__(self, config):
        super().__init__()
        branch_dim = config.fusion_dim // 2
        self.static_branch = StaticBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.static_dim,
            out_dim=branch_dim,
            n_steps=getattr(config, 'static_steps', 4),
            n_channels=getattr(config, 'static_channels', 3),
        )
        self.dynamic_branch = DynamicBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.dynamic_dim,
            out_dim=branch_dim,
            n_nodes=getattr(config, 'n_dynamic_nodes', 5),
            n_channels=getattr(config, 'dynamic_channels', 3),
        )
        self.temperature_condition_scale = getattr(
            config, "temperature_condition_scale", 0.1
        )
        self.fusion = ClasswiseLateFusionHead(
            branch_dim=branch_dim,
            n_materials=config.n_materials,
            alpha_init=getattr(config, 'fusion_alpha_init', None),
        )
        self.safety_source = getattr(config, 'fusion_safety_source', 'static')
        if self.safety_source not in ('static', 'dynamic', 'fused'):
            raise ValueError(f'unknown fusion_safety_source: {self.safety_source}')
        self.safety_head = MultiTaskPredictionHead(branch_dim, config)
        self._stage = 'finetune'

    @property
    def stage(self):
        return self._stage

    def set_stage(self, stage):
        if stage not in ('pretrain', 'fusion_head', 'finetune'):
            raise ValueError(f'unknown training stage: {stage}')
        self._stage = stage

    def _safety_feature(self, static_feat, dynamic_feat, fused_feat):
        if self.safety_source == 'static':
            return static_feat
        if self.safety_source == 'dynamic':
            return dynamic_feat
        return fused_feat

    def forward(self, batch, return_emb=False):
        static_feat = self.static_branch(batch['disp'], batch['strain'], batch['static_pos'])
        dynamic_feat = self.dynamic_branch(batch['ace'], batch['dynamic_pos'])
        static_feat = _apply_temperature_condition(
            static_feat, batch, self.temperature_condition_scale
        )
        dynamic_feat = _apply_temperature_condition(
            dynamic_feat, batch, self.temperature_condition_scale
        )

        if self._stage == 'pretrain':
            static_logits, static_reg = self.fusion.static_head(static_feat)
            dynamic_logits, dynamic_reg = self.fusion.dynamic_head(dynamic_feat)
            # 仍运行完整的融合头，以便填充 last_aux
            _logits, _reg_out, _fused, aux = self.fusion(static_feat, dynamic_feat)
            self.last_aux = aux
            return static_logits, static_reg, dynamic_logits, dynamic_reg, static_feat, dynamic_feat

        logits, reg_out, fused, aux = self.fusion(static_feat, dynamic_feat)
        safety_feat = self._safety_feature(static_feat, dynamic_feat, fused)
        output = self.safety_head(safety_feat)
        output['material_logits'] = logits
        output['material_reg'] = reg_out
        output['material_emb'] = fused
        output['safety_emb'] = safety_feat
        aux['safety_source'] = self.safety_source
        self.last_aux = aux

        if return_emb:
            return output, fused
        return output


class ConcatFusionModel(nn.Module):
    """静态/动态编码器特征直接拼接的简单早期融合基线。"""

    def __init__(self, config):
        super().__init__()
        branch_dim = config.fusion_dim // 2
        self.static_branch = StaticBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.static_dim,
            out_dim=branch_dim,
            n_steps=getattr(config, "static_steps", 4),
            n_channels=getattr(config, "static_channels", 3),
        )
        self.dynamic_branch = DynamicBranch(
            pos_dim=config.pos_dim,
            hidden_dim=config.dynamic_dim,
            out_dim=branch_dim,
            n_nodes=getattr(config, "n_dynamic_nodes", 5),
            n_channels=getattr(config, "dynamic_channels", 3),
        )
        self.temperature_condition_scale = getattr(config, "temperature_condition_scale", 0.1)
        self.head = MultiTaskPredictionHead(branch_dim * 2, config)

    def forward(self, batch, return_emb=False):
        static_feat = self.static_branch(batch["disp"], batch["strain"], batch["static_pos"])
        dynamic_feat = self.dynamic_branch(batch["ace"], batch["dynamic_pos"])
        static_feat = _apply_temperature_condition(static_feat, batch, self.temperature_condition_scale)
        dynamic_feat = _apply_temperature_condition(dynamic_feat, batch, self.temperature_condition_scale)
        fused = torch.cat([static_feat, dynamic_feat], dim=1)
        output = self.head(fused)
        output["material_emb"] = fused
        output["safety_emb"] = fused
        if return_emb:
            return output, fused
        return output


def build_model(config):
    """根据 config.model_type 构建模型。"""

    model_type = getattr(config, 'model_type', 'fusion')
    if model_type == 'fusion':
        return DualBranchFusion(config)
    if model_type == 'concat_fusion':
        return ConcatFusionModel(config)
    if model_type == 'static_only':
        return StaticOnlyModel(config)
    if model_type == 'dynamic_only':
        return DynamicOnlyModel(config)
    raise ValueError(f'unknown model_type: {model_type}')
