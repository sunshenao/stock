"""
统一风控与仓位规则。

所有脚本和操作手册都应以这里为准，避免 ETF 扫描、凯利仓位和
selection 手工记录出现不同仓位口径。
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


MARKET_POSITION_CAPS: dict[str, PositionCap] = {
    "主升": PositionCap(80, 100),
    "震荡": PositionCap(60, 80),
    # 用户要求：除冰点外，现金必须 <=40%，所以退潮/未知也至少部署 60%。
    "退潮": PositionCap(60, 60),
    # 退潮末期：多主线从加速期跌回观察/衰弱、或基准跌破 MA20*0.98，
    # 允许现金 50-70%，只做主力真流入的独立叙事。
    "退潮末期": PositionCap(30, 50),
    "冰点": PositionCap(20, 40),
    "未知": PositionCap(60, 60),
}

MARKET_POSITION_FLOORS: dict[str, int] = {
    "主升": 80,
    "震荡": 60,
    "退潮": 60,
    "退潮末期": 30,
    "未知": 60,
    "冰点": 20,
}

MARKET_DISCOUNT: dict[str, float] = {
    "主升": 1.00,
    "震荡": 0.85,
    "退潮": 0.65,
    "退潮末期": 0.50,
    "冰点": 0.40,
    "未知": 0.50,
}

STAGE_BONUS: dict[str, float] = {
    "扩散期": 0.05,
    "加速期": 0.02,
    "确认期": 0.00,
    "萌芽期": -0.03,
}

SINGLE_POSITION_CAPS: dict[str, int] = {
    "stock": 35,
    "etf": 35,
    "lof": 35,
    "qdii": 35,
    "cash": 100,
}

STOCK_GRADE_CAPS: dict[str, int] = {
    "A": 35,
    "B": 30,
    "C": 20,
    "D": 0,
}

EXPERT_CONSENSUS_CAPS = [
    (40, 20),  # consensus < 40
    (45, 25),  # consensus < 45
]

CATALYST_CAPS: dict[str, int] = {
    "none": 0,
    "occurred_only": 15,
    "confirmed_pending": 35,
}

QDII_PREMIUM_RISK = {
    "normal": (2.0, 1.0, "溢价正常"),
    "watch": (5.0, 0.80, "溢价明显，降权并等待回落"),
    "high": (float("inf"), 0.55, "高溢价，原则上不新开仓"),
    "unknown": (None, 0.85, "溢价未知，跨境/QDII 先降权"),
}


def get_position_cap(market_state: str) -> PositionCap:
    return MARKET_POSITION_CAPS.get(market_state, MARKET_POSITION_CAPS["未知"])


def get_position_floor(market_state: str) -> int:
    return MARKET_POSITION_FLOORS.get(market_state, MARKET_POSITION_FLOORS["未知"])


def is_ice_point(market_state: str) -> bool:
    return market_state == "冰点"


def is_defensive_state(market_state: str) -> bool:
    """冰点或退潮末期，允许现金 > 40%。"""
    return market_state in {"冰点", "退潮末期"}


def requires_aggressive_deployment(market_state: str) -> bool:
    """主升/震荡/退潮/未知必须尽量把现金压到 40% 以内；冰点和退潮末期允许高现金。"""
    return not is_defensive_state(market_state)


def get_market_discount(market_state: str) -> float:
    return MARKET_DISCOUNT.get(market_state, MARKET_DISCOUNT["未知"])


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


def cap_by_grade(grade: str | None) -> int:
    if not grade:
        return 100
    return STOCK_GRADE_CAPS.get(str(grade).upper(), 100)


def cap_by_expert_consensus(consensus: float | None) -> int:
    if consensus is None:
        return 100
    for threshold, cap in EXPERT_CONSENSUS_CAPS:
        if consensus < threshold:
            return cap
    return 100


def cap_by_catalyst(catalyst: str | None) -> int:
    if not catalyst:
        return 100
    return CATALYST_CAPS.get(catalyst, 100)


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

# ============================================================
# 行业差异化阈值（统一维护，etf_analyzer / stock_checkup 均引用此表）
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
