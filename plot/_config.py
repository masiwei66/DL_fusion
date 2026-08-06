"""绘图包全局配置。

集中管理绘图相关的常量：颜色方案、中文字体候选、模型/指标/区域名称的中文映射等。
其他绘图模块统一从本模块导入常量，保证全项目出图风格一致。
"""

# —— 业务常量导入 ——
# 优先以包方式导入（便于 IDE 静态解析，且避免 sys.path 技巧）；
# 直接运行脚本且项目根目录不在搜索路径时，回退为扁平导入。
try:
    from DL_model.safety_rules import CANDIDATE_MATERIAL_IDS, RISK_LEVELS
except ImportError:
    from safety_rules import CANDIDATE_MATERIAL_IDS, RISK_LEVELS

# 候选材料 ID 的简写别名（与原 plotting.py 保持一致）
CANDIDATE_IDS = CANDIDATE_MATERIAL_IDS

# —— 模型名称中文映射 ——
MODEL_NAMES_CN = {
    "static_only": "静态单分支",
    "dynamic_only": "动态单分支",
    "fusion": "融合模型",
}

# —— 评估指标名称中文映射 ——
METRIC_NAMES_CN = {
    "f1": "材料宏F1",
    "auc": "AUC",
    "accuracy": "标签准确率",
    "exact_match": "完全匹配率",
    "precision": "精确率",
    "recall": "召回率",
    "support_f1": "支座F1",
    "support_f1_macro": "支座F1",
    "region_f1_macro": "区域宏F1",
}

# —— 风险等级配色（由低到高：安全 / 预警 / 风险 / 危险）——
RISK_COLORS = ["#4C78A8", "#F2CF5B", "#F58518", "#D94E4E"]
PRED_COLOR = "#4C78A8"   # 预测值颜色
TRUE_COLOR = "#E45756"   # 真实值颜色
PROB_COLOR = "#333333"   # 概率曲线颜色
GRID_COLOR = "#D9D9D9"   # 网格线颜色

# —— 风险等级中文名称 ——
RISK_LEVEL_NAMES_CN = ["安全", "预警", "风险", "危险"]

# —— 区域名称中文映射 ——
REGION_NAMES_CN = {
    "left_support1_zone": "左1号支座区",
    "left_support2_zone": "左2号支座区",
    "right_support2_zone": "右2号支座区",
    "right_zone": "右胯区域",
    "mid_zone": "跨中区域",
    "left_zone": "左胯区域",
    "right_support1_zone": "右1号支座区",
}

# 区域在图中显示的固定顺序
REGION_DISPLAY_ORDER = [
    "left_support1_zone",
    "left_support2_zone",
    "left_zone",
    "mid_zone",
    "right_zone",
    "right_support1_zone",
    "right_support2_zone",
]

# —— 中文字体候选名（按优先级排列）——
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DengXian",
    "FangSong",
    "KaiTi",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]

# 中文字体常见安装路径（Windows），用于手动注册字体
CHINESE_FONT_FILES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]
