"""统一策略政策层。

市场仓位、标的数量、单票上限和组合权重只能在这里定义。扫描器、
仓位计算器、selection 校验器和回测引擎都必须调用本模块，禁止再
各自维护一套仓位常量。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionCap:
    low: int
    high: int

    @property
    def text(self) -> str:
        return f"{self.low}-{self.high}%"


# 正式组合允许 0-3 只；3 只是上限，不再是必须凑满的目标。
MAX_SELECTION_COUNT = 3
TARGET_SELECTION_COUNT = MAX_SELECTION_COUNT  # 兼容旧调用，语义为“最多”
RANK_WEIGHT_RATIOS = (0.40, 0.35, 0.25)
MAX_SAME_RISK_CLUSTER = 2
CORE_ENTRY_SCORE = 65.0
FALLBACK_ENTRY_SCORE = 65.0  # 不再启用低门槛补位，保留名称仅兼容旧导入
MAX_NEW_POSITION_ONE_DAY_DOMINANCE = 0.60

# 主线、买点、风险三层必须分别通过。主线门槛随市场状态收紧，
# 避免在全市场都弱时因为“相对第一”而被迫交易。
MAINLINE_SCORE_FLOORS: dict[str, float] = {
    "主升": 58.0,
    "震荡": 62.0,
    "退潮": 66.0,
    "退潮末期": 70.0,
    "冰点": 75.0,
    "未知": 70.0,
}
MIN_TIMING_SCORE = 50.0
MAX_ENTRY_RISK_PENALTY = 18.0
MAX_SELECTION_BY_STATE: dict[str, int] = {
    "主升": 3,
    "震荡": 2,
    "退潮": 2,
    "退潮末期": 1,
    "冰点": 1,
    "未知": 1,
}

# 中期下跌中的短线反抽最容易制造“高分追入、下一周止损”。非防守
# 新仓若60日趋势仍为负，必须同时满足更高评分和20日反转强度。
NEGATIVE_R60_REVERSAL_SCORE = 60.0
NEGATIVE_R60_REVERSAL_R20 = 15.0
DEFENSIVE_CATEGORIES = {"金融", "公用事业", "红利", "债券", "货币"}
HIGH_BETA_CATEGORIES = {"科技", "军工", "新能源"}
HIGH_BETA_NAME_KEYWORDS = (
    "科技",
    "芯片",
    "半导体",
    "人工智能",
    "AI",
    "算力",
    "通信",
    "云计算",
    "机器人",
    "工业母机",
    "航空航天",
    "科创",
    "创业板",
    "中证1000",
    "中证2000",
)

# 组合允许主动空仓。这里的 high 是各市场状态下、候选充足时的风险预算
# 上限；low 固定为 0，不再把“资金利用率”当成必须凑标的的理由。
MARKET_POSITION_CAPS: dict[str, PositionCap] = {
    "主升": PositionCap(80, 100),
    "震荡": PositionCap(70, 100),
    "退潮": PositionCap(60, 90),
    "退潮末期": PositionCap(60, 80),
    "冰点": PositionCap(0, 40),
    "未知": PositionCap(60, 80),
}

SINGLE_POSITION_CAPS: dict[str, int] = {
    "stock": 100,
    "etf": 100,
    "lof": 100,
    "qdii": 25,
    "cash": 100,
}

QDII_PREMIUM_RISK = {
    "normal": (2.0, 1.0, "溢价正常"),
    "watch": (5.0, 0.80, "溢价明显，降权并等待回落"),
    "high": (float("inf"), 0.55, "高溢价，原则上不新开仓"),
    "unknown": (None, 0.85, "溢价未知，跨境/QDII 先降权"),
}


def get_position_cap(market_state: str) -> PositionCap:
    return MARKET_POSITION_CAPS.get(market_state, MARKET_POSITION_CAPS["未知"])


def category_root(category: str) -> str:
    """Return the policy category before an optional ``/sub-theme`` suffix."""
    return str(category or "其他").split("/")[0]


def get_target_exposure(market_state: str) -> int:
    """返回当前市场状态的标准目标仓位。"""
    return get_position_cap(market_state).high


def allocate_ranked_weights(
    market_state: str,
    count: int = TARGET_SELECTION_COUNT,
) -> list[float]:
    """将风险预算按实际通过门禁的标的数量归一化分配。

    一只获得全部市场风险预算；两只按 40:35 分配；三只按 40:35:25
    分配。这里只放大已经通过全部质量门禁的标的，不会加入弱候选补位。
    """
    if count <= 0:
        return []
    count = min(int(count), TARGET_SELECTION_COUNT)
    target = float(get_target_exposure(market_state))
    ratios = list(RANK_WEIGHT_RATIOS[:count])
    if not ratios:
        return []

    ratio_sum = sum(ratios)
    weights = [round(target * ratio / ratio_sum, 1) for ratio in ratios]
    weights[-1] = round(target - sum(weights[:-1]), 1)
    return weights


def allocate_instrument_weights(
    market_state: str,
    instruments: list[dict],
) -> list[float]:
    """按排名和产品上限分配账户仓位，未使用的风险预算保留为现金。"""
    base = allocate_ranked_weights(market_state, len(instruments))
    return [
        min(
            weight,
            float(single_position_cap(
                item.get("product_type", "ETF"),
                bool(item.get("is_qdii", False)),
            )),
            15.0
            if item.get("is_qdii", False) and item.get("premium_pct") is None
            else float("inf"),
        )
        for weight, item in zip(base, instruments)
    ]


def max_selection_count(market_state: str) -> int:
    return MAX_SELECTION_BY_STATE.get(market_state, MAX_SELECTION_BY_STATE["未知"])


def max_same_risk_cluster(market_state: str) -> int:
    """防守市场不允许两个席位同时押在同一风险簇。"""
    return MAX_SAME_RISK_CLUSTER if market_state in {"主升", "震荡"} else 1


def mainline_score_floor(market_state: str) -> float:
    return MAINLINE_SCORE_FLOORS.get(market_state, MAINLINE_SCORE_FLOORS["未知"])


def carry_mainline_floor(market_state: str) -> float:
    """旧仓只获得5分缓冲，不因续持身份无限绕过主线门槛。"""
    return max(45.0, mainline_score_floor(market_state) - 5.0)


def score_layers_pass(score: dict, market_state: str) -> tuple[bool, str]:
    """检查主线、买点和风险三层，不允许总分掩盖任一层失败。"""
    total = float(score.get("total") or 0)
    raw_total = float(score.get("raw_total") or total)
    mainline = float(score.get("mainline") or 0)
    timing = float(score.get("timing") or 0)
    risk_penalty = float(score.get("risk_penalty") or 0)
    one_day_dominance = float(score.get("one_day_dominance") or 0)
    floor = mainline_score_floor(market_state)
    if raw_total < CORE_ENTRY_SCORE:
        return False, f"折溢价前总分{raw_total:.1f}<{CORE_ENTRY_SCORE:.0f}"
    if mainline < floor:
        return False, f"主线分{mainline:.1f}<{floor:.0f}"
    if timing < MIN_TIMING_SCORE:
        return False, f"买点分{timing:.1f}<{MIN_TIMING_SCORE:.0f}"
    if risk_penalty > MAX_ENTRY_RISK_PENALTY:
        return False, f"风险惩罚{risk_penalty:.1f}>{MAX_ENTRY_RISK_PENALTY:.0f}"
    if one_day_dominance > MAX_NEW_POSITION_ONE_DAY_DOMINANCE:
        return (
            False,
            "5日涨幅单日贡献"
            f"{one_day_dominance:.0%}>{MAX_NEW_POSITION_ONE_DAY_DOMINANCE:.0%}",
        )
    if not bool(score.get("mainline_pass", True)):
        return False, str(score.get("gate_reason") or "主线持续性门禁未通过")
    if not bool(score.get("timing_pass", True)):
        return False, str(score.get("gate_reason") or "买点门禁未通过")
    return True, ""


def get_target_cash(market_state: str) -> int:
    return 100 - get_target_exposure(market_state)


def is_ice_point(market_state: str) -> bool:
    return market_state == "冰点"


def is_defensive_state(market_state: str) -> bool:
    """退潮及更弱状态都允许主动提高现金。"""
    return market_state in {"退潮", "退潮末期", "冰点", "未知"}


def requires_aggressive_deployment(market_state: str) -> bool:
    """非冰点组合必须达到对应状态的最低风险预算。"""
    return not is_ice_point(market_state)


def instrument_key(product_type: str, is_qdii: bool = False) -> str:
    if is_qdii:
        return "qdii"
    t = (product_type or "").lower()
    if t == "lof":
        return "lof"
    if t == "stock":
        return "stock"
    if t == "cash":
        return "cash"
    return "etf"


def single_position_cap(product_type: str, is_qdii: bool = False) -> int:
    return SINGLE_POSITION_CAPS[instrument_key(product_type, is_qdii)]


def premium_discount_factor(is_qdii: bool, premium_pct: float | None) -> tuple[float, str]:
    """返回跨境/QDII 折溢价降权系数和说明。"""
    if not is_qdii:
        return 1.0, ""
    if premium_pct is None:
        _, factor, note = QDII_PREMIUM_RISK["unknown"]
        return factor, note
    abs_premium = abs(float(premium_pct))
    if abs_premium < QDII_PREMIUM_RISK["normal"][0]:
        return QDII_PREMIUM_RISK["normal"][1], QDII_PREMIUM_RISK["normal"][2]
    if abs_premium < QDII_PREMIUM_RISK["watch"][0]:
        return QDII_PREMIUM_RISK["watch"][1], QDII_PREMIUM_RISK["watch"][2]
    return QDII_PREMIUM_RISK["high"][1], QDII_PREMIUM_RISK["high"][2]


def risk_cluster(
    category: str,
    industry: str,
    instrument_name: str = "",
) -> str:
    """Map different labels with similar tail risk into one portfolio cluster."""
    category = str(category or "其他")
    major_category = category_root(category)
    industry = str(industry or "其他")
    name = str(instrument_name or "")

    if major_category in HIGH_BETA_CATEGORIES or any(
        keyword.lower() in name.lower()
        for keyword in HIGH_BETA_NAME_KEYWORDS
    ):
        return "高弹性成长"
    if major_category in {"资源", "商品"} or industry == "周期":
        return "资源商品"
    if major_category in DEFENSIVE_CATEGORIES or industry in {
        "金融",
        "公用事业",
        "红利",
        "债券",
    }:
        return "防御价值"
    if major_category == "医药" or industry == "医药":
        return "医药"
    if major_category == "消费" or industry == "消费":
        return "消费"
    if major_category in {"跨境", "LOF"}:
        return "跨境"
    if major_category == "宽基" or industry == "宽基":
        return "宽基"
    return f"{category}/{industry}"


def new_position_trend_gate(
    *,
    category: str,
    market_state: str,
    score: float,
    ret_20d: float,
    ret_60d: float,
) -> tuple[bool, str]:
    """Reject weak-medium-trend rebounds unless reversal evidence is unusually strong."""
    category = str(category or "其他")
    major_category = category_root(category)
    if major_category in DEFENSIVE_CATEGORIES:
        return True, ""
    if ret_60d >= 0:
        return True, ""
    if score >= NEGATIVE_R60_REVERSAL_SCORE and ret_20d >= NEGATIVE_R60_REVERSAL_R20:
        return True, "60日趋势为负，但达到强反转门槛"
    return (
        False,
        f"60日趋势{ret_60d:+.1f}%仍为负，评分{score:.1f}/20日{ret_20d:+.1f}%"
        f"未同时达到{NEGATIVE_R60_REVERSAL_SCORE:.0f}分和"
        f"{NEGATIVE_R60_REVERSAL_R20:.0f}%反转门槛",
    )

# ============================================================
# 行业差异化阈值（供 etf_analyzer 使用）
# 数据来源：stock-analyzer-skill experts/sector_specialist.md
# ============================================================
INDUSTRY_THRESHOLDS: dict[str, dict] = {
    "医药":     {"roe_min": 12, "growth_min": 20, "gross_min": 50, "debt_max": 50,
                 "pe_ref": "行业50%分位", "pe_cheap": 25, "pe_fair": 40, "pe_warn": 60,
                 "risk_note": "集采、临床失败"},
    "科技":     {"roe_min": 10, "growth_min": 30, "gross_min": 30, "debt_max": 50,
                 "pe_ref": "<60倍", "pe_cheap": 40, "pe_fair": 60, "pe_warn": 100,
                 "risk_note": "国产替代节奏、制裁"},
    "消费":     {"roe_min": 15, "growth_min": 10, "gross_min": 40, "debt_max": 50,
                 "pe_ref": "<30倍", "pe_cheap": 25, "pe_fair": 35, "pe_warn": 50,
                 "risk_note": "消费降级、渠道变迁"},
    "金融":     {"roe_min": 10, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "PB<0.7(大行)", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "pb_cheap": 0.8, "pb_fair": 1.2, "pb_warn": 2.0,
                 "risk_note": "利率、不良率"},
    "周期":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": 60,
                 "pe_ref": "行业50%分位", "pe_cheap": 20, "pe_fair": 35, "pe_warn": 55,
                 "risk_note": "商品价格、库存周期"},
    "新能源":   {"roe_min": 8, "growth_min": 25, "gross_min": 20, "debt_max": 60,
                 "pe_ref": "<35x", "pe_cheap": 20, "pe_fair": 35, "pe_warn": 55,
                 "risk_note": "产能过剩、补贴退坡"},
    "军工":     {"roe_min": 6, "growth_min": 15, "gross_min": 20, "debt_max": 60,
                 "pe_ref": "<35x", "pe_cheap": 20, "pe_fair": 35, "pe_warn": 55,
                 "risk_note": "订单节奏、军改"},
    "公用事业": {"roe_min": 6, "growth_min": 5, "gross_min": 20, "debt_max": 70,
                 "pe_ref": "<20x", "pe_cheap": 15, "pe_fair": 25, "pe_warn": 40,
                 "risk_note": "电价政策"},
    "红利":     {"roe_min": 8, "growth_min": 5, "gross_min": 20, "debt_max": 50,
                 "pe_ref": "<15x", "pe_cheap": 10, "pe_fair": 18, "pe_warn": 30,
                 "risk_note": "利率、风格切换"},
    "跨境":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "N/A", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "汇率、地缘"},
    "债券":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "N/A", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "利率、信用"},
    "宽基":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "—", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "—"},
    "商品":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "N/A", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "商品价格、展期损耗"},
    "其他":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "N/A", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "主题持续性不足"},
    "货币":     {"roe_min": None, "growth_min": None, "gross_min": None, "debt_max": None,
                 "pe_ref": "N/A", "pe_cheap": None, "pe_fair": None, "pe_warn": None,
                 "risk_note": "现金管理工具，不作进攻方向"},
}
