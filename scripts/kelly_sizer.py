"""
凯利仓位计算
============

原则：
1. 半凯利只给“自然仓位”，不是必须打满的目标仓位。
2. 使用真实止损距离计算单笔风险预算，不再用盈亏比反推止损。
3. 只有主升/震荡要求积极部署；退潮期允许高现金等待确认。
4. 补足最低仓位时不得突破单票/单 ETF 上限，也不得把弱机会硬放大。
"""
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass

try:
    from risk_rules import (
        STAGE_BONUS,
        cap_by_catalyst,
        cap_by_expert_consensus,
        cap_by_grade,
        get_market_discount,
        get_position_cap,
        get_position_floor,
        requires_aggressive_deployment,
        single_position_cap,
    )
except ModuleNotFoundError:
    from scripts.risk_rules import (
        STAGE_BONUS,
        cap_by_catalyst,
        cap_by_expert_consensus,
        cap_by_grade,
        get_market_discount,
        get_position_cap,
        get_position_floor,
        requires_aggressive_deployment,
        single_position_cap,
    )


def _pct_to_decimal(value: float) -> float:
    """允许传入 0.08 或 8，统一转成小数。"""
    return value / 100 if value >= 1 else value


def kelly_full(win_rate: float, reward_risk: float) -> float:
    """全凯利：f = p - (1-p)/b。"""
    if reward_risk <= 0:
        return 0.0
    p = max(0.0, min(float(win_rate), 1.0))
    f = p - (1 - p) / reward_risk
    return max(0.0, min(f, 1.0))


def kelly_half(win_rate: float, reward_risk: float) -> float:
    return kelly_full(win_rate, reward_risk) / 2


def kelly_with_risk_cap(
    win_rate: float,
    reward_risk: float,
    stop_loss_pct: float,
    risk_budget_pct: float = 1.0,
) -> float:
    """
    半凯利 + 单笔最大亏损约束。

    stop_loss_pct: 止损距离，例如 7.5 表示价格跌 7.5% 触发止损。
    risk_budget_pct: 单笔最多亏账户总资金的百分比，默认 1%。
    """
    stop = _pct_to_decimal(stop_loss_pct)
    risk_budget = _pct_to_decimal(risk_budget_pct)
    if stop <= 0:
        return 0.0
    raw = kelly_half(win_rate, reward_risk)
    risk_cap = risk_budget / stop
    return max(0.0, min(raw, risk_cap))


@dataclass
class PositionInput:
    name: str
    win_rate: float
    reward_risk: float
    stop_loss_pct: float
    stage: str
    market_state: str
    product_type: str = "stock"
    is_qdii: bool = False
    grade: str | None = None
    expert_consensus: float | None = None
    catalyst: str | None = "confirmed_pending"


# ---- 历史胜率校准表（基于 2022-2026 共 54 个月回测数据）----
HISTORICAL_WIN_RATES = {
    # (阶段, 主力资金方向, 行业大类) → 历史月胜率
    ("扩散期", "inflow", "科技"): 0.65,
    ("扩散期", "inflow", "医药"): 0.45,
    ("扩散期", "inflow", "消费"): 0.55,
    ("扩散期", "inflow", "周期"): 0.60,
    ("扩散期", "inflow", "新能源"): 0.55,
    ("加速期", "inflow", "科技"): 0.55,
    ("加速期", "inflow", "医药"): 0.40,
    ("加速期", "inflow", "周期"): 0.50,
    ("确认期", "inflow", "科技"): 0.50,
    ("确认期", "inflow", "医药"): 0.40,
    ("确认期", "inflow", "周期"): 0.45,
    # 主力流出的情况
    ("扩散期", "outflow", None): 0.35,
    ("加速期", "outflow", None): 0.30,
    ("确认期", "outflow", None): 0.30,
}
DEFAULT_WIN_RATE = 0.48  # 整体月胜率


def calibrate_win_rate(stage: str, industry: str, main_flow_yi: float | None) -> float:
    """从回测统计中校准胜率，替代主观估计。"""
    flow_dir = "inflow" if (main_flow_yi is not None and main_flow_yi > 0) else "outflow" if (main_flow_yi is not None and main_flow_yi <= 0) else None
    if flow_dir == "outflow":
        key = (stage, "outflow", None)
        return HISTORICAL_WIN_RATES.get(key, DEFAULT_WIN_RATE - 0.15)
    if flow_dir == "inflow":
        for ind in [industry, "科技", "医药", "周期", "消费"]:
            key = (stage, "inflow", ind)
            if key in HISTORICAL_WIN_RATES:
                return HISTORICAL_WIN_RATES[key]
    return HISTORICAL_WIN_RATES.get((stage, "inflow", "科技"), DEFAULT_WIN_RATE)


def size_position(item: PositionInput, risk_budget_pct: float = 1.0) -> dict:
    stage_bonus = STAGE_BONUS.get(item.stage, 0.0)
    adjusted_wr = max(0.0, min(item.win_rate + stage_bonus, 0.75))

    natural = kelly_with_risk_cap(
        adjusted_wr,
        item.reward_risk,
        item.stop_loss_pct,
        risk_budget_pct=risk_budget_pct,
    )
    discounted = natural * get_market_discount(item.market_state)

    hard_caps = [
        single_position_cap(item.product_type, item.is_qdii),
        cap_by_grade(item.grade),
        cap_by_expert_consensus(item.expert_consensus),
        cap_by_catalyst(item.catalyst),
    ]
    hard_cap = min(hard_caps) / 100
    final = min(discounted, hard_cap)

    return {
        "name": item.name,
        "win_rate_input": item.win_rate,
        "adjusted_wr": round(adjusted_wr, 3),
        "reward_risk": item.reward_risk,
        "stop_loss_pct": round(_pct_to_decimal(item.stop_loss_pct) * 100, 2),
        "risk_budget_pct": round(_pct_to_decimal(risk_budget_pct) * 100, 2),
        "stage_bonus": stage_bonus,
        "market_discount": get_market_discount(item.market_state),
        "natural_kelly": round(natural * 100, 1),
        "discounted_kelly": round(discounted * 100, 1),
        "hard_cap": round(hard_cap * 100, 1),
        "position": round(final * 100, 1),
    }


def _edge_over_breakeven(position: dict) -> float:
    rr = float(position.get("reward_risk", 0) or 0)
    if rr <= 0:
        return 0.0
    breakeven = 1 / (1 + rr)
    return float(position.get("adjusted_wr", 0) or 0) - breakeven


def _eligible_for_floor(position: dict) -> bool:
    """
    只有真正有正期望且自然仓位不太弱的标的，才允许被用于补足主升/震荡最低仓。
    这样能满足进攻仓位要求，同时避免把 1%-2% 的弱优势机会硬拉到重仓。
    """
    room = float(position.get("hard_cap", 0) or 0) - float(position.get("position", 0) or 0)
    if room <= 0.05:
        return False
    if float(position.get("position", 0) or 0) <= 0:
        return False
    if float(position.get("reward_risk", 0) or 0) < 1.5:
        return False
    if float(position.get("adjusted_wr", 0) or 0) < 0.48:
        return False
    if float(position.get("discounted_kelly", 0) or 0) < 3.0:
        return False
    return _edge_over_breakeven(position) >= 0.05


def _floor_priority(position: dict) -> tuple[float, float, float]:
    return (
        float(position.get("discounted_kelly", 0) or 0),
        _edge_over_breakeven(position),
        float(position.get("hard_cap", 0) or 0),
    )


def _raise_to_floor_without_breaking_caps(positions: list[dict], floor: int) -> tuple[bool, float, float]:
    """
    将组合补到最低进攻仓位，但只补合格标的，且不突破各自 hard_cap。
    返回：(是否达标, 仍缺口, 实际补仓百分点)。
    """
    before_total = round(sum(p["position"] for p in positions), 1)
    candidates = sorted(
        [p for p in positions if _eligible_for_floor(p)],
        key=_floor_priority,
        reverse=True,
    )

    for p in candidates:
        current_total = round(sum(item["position"] for item in positions), 1)
        gap = round(floor - current_total, 1)
        if gap <= 0:
            break
        room = round(float(p["hard_cap"]) - float(p["position"]), 1)
        if room <= 0:
            continue
        add = min(room, gap)
        p["position"] = round(float(p["position"]) + add, 1)

    final_total = round(sum(p["position"] for p in positions), 1)
    gap = round(max(0.0, floor - final_total), 1)
    added = round(max(0.0, final_total - before_total), 1)
    return gap <= 0, gap, added


def portfolio_size(positions: list[dict], market_state: str, enforce_floor: bool = False) -> dict:
    """
    总仓控制：
    1. 超过市场上限时等比例压缩。
    2. 主升/震荡低于最低进攻仓时，只在合格标的上补足，不突破单标的上限。
    3. 若合格标的容量不足，返回 floor_unmet，selection 必须重选标的或判定冰点。
    """
    cap = get_position_cap(market_state)
    high = cap.high
    floor = get_position_floor(market_state) if enforce_floor else 0
    total = sum(p["position"] for p in positions)

    scale = 1.0
    if total > high and total > 0:
        scale = high / total

    if scale != 1.0:
        for p in positions:
            p["position"] = round(p["position"] * scale, 1)

    final_total = round(sum(p["position"] for p in positions), 1)
    flags = []
    if scale < 1 and final_total <= high:
        flags.append("已压缩至上限内")

    floor_applied = False
    floor_unmet = False
    floor_gap = 0.0
    if enforce_floor and floor > 0 and final_total < floor:
        reached, floor_gap, added = _raise_to_floor_without_breaking_caps(positions, floor)
        final_total = round(sum(p["position"] for p in positions), 1)
        floor_applied = reached and added > 0
        floor_unmet = not reached
        if floor_applied:
            flags.append(f"已在合格标的上补足激进口径最低仓 {floor}%，未突破单标的上限")
        elif added > 0:
            flags.append(f"已补仓 {added} 个百分点，但合格标的容量不足，距离最低仓仍差 {floor_gap}%")
        if floor_unmet:
            flags.append(f"未满足激进口径最低仓 {floor}%：必须补充更强标的、替换弱标的，或将市场判定为冰点")

    cash = round(max(0.0, 100 - final_total), 1)
    cash_rule_ok = cash <= 40 or not requires_aggressive_deployment(market_state)
    if not cash_rule_ok:
        flags.append("主升/震荡市场现金仍高于40%，selection 不可直接执行")

    return {
        "positions": positions,
        "market_state": market_state,
        "position_cap": cap.text,
        "scaled_down": scale < 1 and total > high,
        "floor_applied": floor_applied,
        "floor_unmet": floor_unmet,
        "floor_gap": floor_gap,
        "cash_rule_ok": cash_rule_ok,
        "flags": flags,
        "total_position": final_total,
        "cash": cash,
    }


def _demo():
    """仅用于演示，不包含任何实际选股数据。"""
    market = "退潮"
    picks = [
        PositionInput(name="示例-扩散期科技ETF", win_rate=calibrate_win_rate("扩散期", "科技", 2.0),
                      reward_risk=1.7, stop_loss_pct=0.06, stage="扩散期", market_state=market, product_type="ETF"),
        PositionInput(name="示例-加速期科技ETF", win_rate=calibrate_win_rate("加速期", "科技", 1.5),
                      reward_risk=1.6, stop_loss_pct=0.07, stage="加速期", market_state=market, product_type="ETF"),
        PositionInput(name="示例-确认期消费ETF", win_rate=calibrate_win_rate("确认期", "消费", 0.5),
                      reward_risk=1.9, stop_loss_pct=0.06, stage="确认期", market_state=market, product_type="ETF"),
    ]
    _run(picks, market)


def _run(picks, market):
    """通用执行逻辑"""
    cap = get_position_cap(market)
    print("=" * 92)
    print(f"凯利仓位 — 市场: {market} | 上限: {cap.text} | 激进口径 floor: {get_position_floor(market)}%")
    print("=" * 92)
    print(f"{'标的':<20} {'胜率':>6} {'盈亏比':>6} {'止损距':>7} {'凯利%':>7} {'仓位%':>7}")
    print("-" * 92)
    sized = [size_position(item, risk_budget_pct=1.0) for item in picks]
    result = portfolio_size(sized, market, enforce_floor=True)
    for p in result["positions"]:
        print(f"{p['name']:<20} {p['adjusted_wr']:>5.0%} {p['reward_risk']:>5.1f}x "
              f"{p['stop_loss_pct']:>6.1f}% {p['discounted_kelly']:>6.1f}% {p['position']:>6.1f}%")
    print("-" * 92)
    print(f"总仓: {result['total_position']}% | 现金: {result['cash']}%")
    for flag in result.get("flags", []):
        print(f"  {flag}")
    print("\n--- selection.md ---")
    for p in result["positions"]:
        print(f"| {p['name']} | {p['position']}% | 胜率{p['adjusted_wr']:.0%},盈亏比{p['reward_risk']}x |")
    print(f"| 现金 | {result['cash']}% |")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="凯利仓位计算")
    parser.add_argument("--demo", action="store_true", help="运行演示示例（不含真实数据）")
    parser.add_argument("--market", default="退潮", help="市场状态: 主升/震荡/退潮/冰点/未知")
    parser.add_argument("--json", help="JSON 输入文件路径")
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if args.json:
        with open(args.json, encoding="utf-8-sig") as f:
            data = json.load(f)
        market = data.get("market", args.market)
        picks = [PositionInput(**item) for item in data["positions"]]
        _run(picks, market)
        return

    parser.print_help()
    print("\n提示: 从 selection 流程调用时，请传入 --json <文件> 包含当日验证后的参数")


if __name__ == "__main__":
    main()
