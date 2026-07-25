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


TARGET_SELECTION_COUNT = 3
RANK_WEIGHT_RATIOS = (0.40, 0.35, 0.25)
MAX_SAME_RISK_CLUSTER = 2
CORE_ENTRY_SCORE = 45.0
FALLBACK_ENTRY_SCORE = 30.0

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

# 非冰点现金不得超过 40%。目标仓位取区间上限，区间下限供
# selection_guard 判断是否出现异常低资金利用率。
MARKET_POSITION_CAPS: dict[str, PositionCap] = {
    "主升": PositionCap(90, 100),
    "震荡": PositionCap(80, 90),
    "退潮": PositionCap(60, 70),
    "退潮末期": PositionCap(60, 60),
    "冰点": PositionCap(20, 40),
    "未知": PositionCap(60, 60),
}

SINGLE_POSITION_CAPS: dict[str, int] = {
    "stock": 40,
    "etf": 40,
    "lof": 35,
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


def get_target_exposure(market_state: str) -> int:
    """返回当前市场状态的标准目标仓位。"""
    return get_position_cap(market_state).high


def allocate_ranked_weights(
    market_state: str,
    count: int = TARGET_SELECTION_COUNT,
) -> list[float]:
    """按排名分配目标仓位，默认三只为 40%/35%/25% 的强弱结构。

    返回值是账户总资产百分比，而不是组合内部权重。三只标的时总和
    精确等于市场目标仓位；标的不足时不机械把弱机会的仓位放大。
    """
    if count <= 0:
        return []
    target = float(get_target_exposure(market_state))
    ratios = list(RANK_WEIGHT_RATIOS[:count])
    if not ratios:
        return []

    weights = [round(target * ratio, 1) for ratio in ratios]
    if count == TARGET_SELECTION_COUNT:
        weights[-1] = round(target - sum(weights[:-1]), 1)
    return weights


def get_target_cash(market_state: str) -> int:
    return 100 - get_target_exposure(market_state)


def is_ice_point(market_state: str) -> bool:
    return market_state == "冰点"


def is_defensive_state(market_state: str) -> bool:
    """只有冰点允许现金 > 40%。退潮/退潮末期现金仍须 < 40%（规则 8/9）。"""
    return market_state in {"冰点"}


def requires_aggressive_deployment(market_state: str) -> bool:
    """冰点以外所有市场状态均要求至少 60% 仓位。"""
    return market_state != "冰点"


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
    industry = str(industry or "其他")
    name = str(instrument_name or "")

    if category in HIGH_BETA_CATEGORIES or any(
        keyword.lower() in name.lower()
        for keyword in HIGH_BETA_NAME_KEYWORDS
    ):
        return "高弹性成长"
    if category in {"资源", "商品"} or industry == "周期":
        return "资源商品"
    if category in DEFENSIVE_CATEGORIES or industry in {
        "金融",
        "公用事业",
        "红利",
        "债券",
    }:
        return "防御价值"
    if category == "医药" or industry == "医药":
        return "医药"
    if category == "消费" or industry == "消费":
        return "消费"
    if category in {"跨境", "LOF"}:
        return "跨境"
    if category == "宽基" or industry == "宽基":
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
    if category in DEFENSIVE_CATEGORIES:
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
