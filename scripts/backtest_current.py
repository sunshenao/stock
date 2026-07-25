"""日度扫描快照诊断回放。

该工具用于检查周度框架在日内风险事件中的行为，不是正式选股入口：
- 规则 2: 涨跌比三档仓位（替代 MA5 择时）
- 规则 8: 单日>7%不追
- 规则 9: 5日>25%/20日>35%不新开
- 规则 11: 大簇崩盘检测+该大类禁止新开
- 规则 12: 假突破检测
- 15% 移动止损
- QDII 商品豁免（油/金/铜/银）
- 冰点期只看确认期+扩散期

用法:
  python scripts/backtest_current.py --days full
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from risk_rules import (
    TARGET_SELECTION_COUNT,
    allocate_ranked_weights,
    get_target_exposure,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK = os.path.join(ROOT, "codex", "stock")
BENCHMARK_CODE = "510300"

# ============================================================
# 策略参数
# ============================================================
COST_BPS = 8.0
LIQ_MIN = 1.0
BOND_SECTORS = ("债券", "货币")
BAD_STAGES = ("观察期", "萌芽期", "衰弱期", "弱势期", "休眠期", "衰竭期")
REBALANCE_DAYS = 3
TRAILING_STOP = 15.0

# 规则 6
DD_SOFT_LIMIT = 8.0
DD_HARD_LIMIT = 12.0
DD_RELEASE = 5.0
BREAKER_MAX_DAYS = 10
BREAKER_COOLDOWN_DAYS = 5

# 规则 8 / 9
CHASE_1D_LIMIT = 7.0
RET5_EXTREME = 25.0
RET20_EXTREME = 35.0
RET20_HIGH_RISK = 15.0

# 规则 11
CLUSTER_CRASH_PCT = -5.0
CLUSTER_MIN_COUNT = 3

# 规则 12
FALSE_BREAK_WIDTH = 1.5
FALSE_BREAK_CSI300 = 3.0

# QDII 商品豁免
COMMODITY_QDII_CODES = {"513350", "159197", "513650"}  # 标普油气, 石油招商, etc


@dataclass
class ETFData:
    code: str = ""
    name: str = ""
    sector: str = ""
    stage: str = ""
    score: float = 0.0
    amount_yi: float = 0.0
    pct: float = 0.0
    close: float = 0.0
    ret5: float = 0.0
    ret20: float = 0.0
    ret60: float = 0.0
    amount_ratio: float = 1.0
    industry: str = ""
    direction: str = ""
    is_qdii: bool = False

    @property
    def is_diffusion(self) -> bool:
        return "扩散" in self.stage

    @property
    def is_confirmed(self) -> bool:
        return "确认" in self.stage

    @property
    def is_accel_top(self) -> bool:
        return "加速见顶" in self.stage

    @property
    def has_warning(self) -> bool:
        return "⚠" in self.stage


def load_all_scans() -> dict[str, dict]:
    scans = {}
    for d in sorted(glob.glob(os.path.join(STOCK, "2026-*"))):
        date = os.path.basename(d)
        scan_path = os.path.join(d, "scan.json")
        if not os.path.exists(scan_path):
            continue
        with open(scan_path, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for r in data["rows"]:
            if not r.get("code"):
                continue
            rows.append(ETFData(
                code=r["code"], name=r["name"],
                sector=r.get("sector", ""),
                stage=r.get("stage", ""),
                score=r.get("score", 0) or 0,
                amount_yi=r.get("amount_yi", 0) or 0,
                pct=r.get("pct", 0) or 0,
                close=r.get("close", 0) or 0,
                ret5=r.get("ret_5d", 0) or 0,
                ret20=r.get("ret_20d", 0) or 0,
                ret60=r.get("ret_60d", 0) or 0,
                amount_ratio=r.get("amount_ratio", 1) or 1,
                industry=r.get("industry", "") or "",
                direction=r.get("direction", "") or "",
                is_qdii=r.get("is_qdii", False),
            ))
        scans[date] = {
            "bench": data.get("bench_pct", 0) or 0,
            "rows": rows,
            "by_code": {r.code: r for r in rows},
        }
    return scans


def calc_ret(prices, code, from_date, to_date):
    p = prices.get(code, {})
    if from_date in p and to_date in p and p[from_date]:
        return (p[to_date] - p[from_date]) / p[from_date] * 100
    return None


def calc_width_ratio(rows: list[ETFData]) -> float:
    up = sum(1 for r in rows if r.pct > 0)
    dn = sum(1 for r in rows if r.pct < 0)
    if dn == 0:
        return float(up) if up > 0 else 1.0
    return up / dn


def is_eligible(
    r: ETFData,
    ice_mode: bool = False,
    crashing_clusters: set | None = None,
    for_new_position: bool = True,
) -> bool:
    """完整合格池检查，包含 QDII 分层和冰点限制。"""
    if not r.code:
        return False
    if r.sector in BOND_SECTORS:
        return False
    if r.stage in BAD_STAGES:
        return False
    if for_new_position and r.is_accel_top:
        return False
    if for_new_position and r.has_warning:
        return False
    if r.amount_yi < LIQ_MIN:
        return False
    # QDII 分层：商品 QDII 豁免，权益 QDII 排除
    if r.is_qdii and r.code not in COMMODITY_QDII_CODES:
        return False
    # 冰点期：只看确认期+扩散期（QDII 商品豁免阶段限制）
    if ice_mode and not r.is_qdii:
        if not (r.is_diffusion or r.is_confirmed):
            return False
    # 规则 11: 崩盘大类禁止新开
    if crashing_clusters and get_major_sector(r.sector) in crashing_clusters:
        return False
    if for_new_position:
        if r.pct > CHASE_1D_LIMIT:
            return False
        if r.ret5 > RET5_EXTREME or r.ret20 > RET20_EXTREME:
            return False
    return True


def get_major_sector(sector: str) -> str:
    """从 sector 字段反推大类（兼容 scan.json 的旧分类）。"""
    tech = ("科技", "半导体", "芯片", "通信", "软件", "计算机", "云计算",
            "消费电子", "传媒", "游戏", "信息技术", "物联网", "机器人")
    medical = ("医药", "医疗", "创新药", "生物医药", "中药", "医疗器械")
    consume = ("消费", "食品饮料", "家电", "养殖", "农业", "旅游", "汽车", "物流")
    resource = ("资源", "有色", "煤炭", "钢铁", "稀土", "化工", "建材", "油气", "黄金股")
    finance = ("金融", "银行", "证券", "保险", "房地产")
    new_energy = ("新能源", "光伏", "电池", "碳中和")
    military = ("军工", "航空航天", "船舶", "机械", "智能制造")
    utility = ("公用事业", "电力", "基建", "电网")
    commodity = ("商品", "黄金", "白银", "豆粕", "能化")

    for kw in tech:
        if kw in sector: return "科技"
    for kw in medical:
        if kw in sector: return "医药"
    for kw in consume:
        if kw in sector: return "消费"
    for kw in resource:
        if kw in sector: return "资源"
    for kw in finance:
        if kw in sector: return "金融"
    for kw in new_energy:
        if kw in sector: return "新能源"
    for kw in military:
        if kw in sector: return "军工"
    for kw in utility:
        if kw in sector: return "公用事业"
    for kw in commodity:
        if kw in sector: return "商品"
    return "其他"


def _return_over_days(prices, code: str, dates: list[str], index: int, lookback: int) -> float:
    """只用当前日及以前的快照收盘价计算动量。"""
    current = prices.get(code, {}).get(dates[index])
    if not current:
        return 0.0
    start = max(0, index - lookback)
    for j in range(start, index):
        previous = prices.get(code, {}).get(dates[j])
        if previous:
            return (current / previous - 1) * 100
    return 0.0


def _market_state_from_proxy(benchmark_nav: list[float], width: float, bench_pct: float) -> str:
    """扫描快照缺少全A宽度时，用沪深300真实净值趋势为主、ETF宽度为辅。"""
    nav = benchmark_nav[-1]
    ma5 = sum(benchmark_nav[-5:]) / min(5, len(benchmark_nav))
    ma20 = sum(benchmark_nav[-20:]) / min(20, len(benchmark_nav))
    gap20 = nav / ma20 - 1 if ma20 else 0.0

    if bench_pct <= -3.0 or (len(benchmark_nav) >= 20 and gap20 <= -0.05 and width < 0.5):
        return "冰点"
    if len(benchmark_nav) >= 20 and gap20 <= -0.02:
        return "退潮末期"
    if nav < ma20 or (nav < ma5 and width < 0.5):
        return "退潮"
    if nav >= ma5 and nav >= ma20 * 1.02 and width >= 1.0:
        return "主升"
    return "震荡"


def _pick_ranked(
    rows: list[ETFData],
    target_count: int,
    preferred_codes: set[str] | None = None,
    continuation_bonus: float = 6.0,
) -> list[ETFData]:
    """最多两个同大类；合格老持仓获得有限延续加分，降低无效换手。"""
    selected: list[ETFData] = []
    major_count: dict[str, int] = {}
    preferred_codes = preferred_codes or set()
    ranked = sorted(
        rows,
        key=lambda item: -(item.score + (continuation_bonus if item.code in preferred_codes else 0.0)),
    )
    for row in ranked:
        major = get_major_sector(row.industry or row.sector)
        if major_count.get(major, 0) >= 2:
            continue
        selected.append(row)
        major_count[major] = major_count.get(major, 0) + 1
        if len(selected) >= target_count:
            break
    return selected


def _scaled_rank_weights(market_state: str, target_exposure: float, count: int) -> list[float]:
    standard = get_target_exposure(market_state)
    weights = allocate_ranked_weights(market_state, count)
    if standard <= 0 or not weights:
        return []
    scale = target_exposure / standard
    scaled = [round(weight * scale, 1) for weight in weights]
    if count == TARGET_SELECTION_COUNT:
        scaled[-1] = round(target_exposure - sum(scaled[:-1]), 1)
    return scaled


def run_backtest(scans, prices, dates, opens=None, verbose=True):
    """逐日扫描快照回放；盘后信号在下一交易日开盘执行。"""
    opens = opens or prices
    holdings: dict[str, dict] = {}
    executed_holdings: dict[str, dict] = {}
    nav = 1.0
    peak_nav = 1.0
    benchmark_nav: list[float] = []
    breaker = False
    breaker_days = 0
    breaker_cooldown = 0
    rebalance_counter = REBALANCE_DAYS
    confirmed_state = ""
    pending_state = ""
    pending_state_days = 0
    last_scan = None
    log = []
    benchmark_history = sorted(prices.get(BENCHMARK_CODE, {}).items())

    for i, date in enumerate(dates):
        has_scan = date in scans
        if has_scan:
            last_scan = scans[date]
        if last_scan is None:
            continue
        sc = last_scan
        rows = sc["rows"]
        bench = 0.0
        if i > 0:
            benchmark_return = calc_ret(prices, BENCHMARK_CODE, dates[i - 1], date)
            if benchmark_return is not None:
                bench = benchmark_return
        nav_before_day = nav
        previous_weights = {
            code: float(item.get("w", 0))
            for code, item in executed_holdings.items()
        }
        execution_weights = {
            code: float(item.get("w", 0))
            for code, item in holdings.items()
        }
        all_execution_codes = set(previous_weights) | set(execution_weights)
        turnover_pct = sum(
            abs(execution_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
            for code in all_execution_codes
        )
        cost_pct = turnover_pct * COST_BPS / 10000

        # 旧仓承担前收盘至今开盘的隔夜波动，盘后信号在今开盘成交，
        # 新仓再承担今开盘至收盘的日内波动。
        overnight_return = 0.0
        intraday_return = 0.0
        missing_prices: list[str] = []
        if i > 0:
            for code, holding in executed_holdings.items():
                previous_close = prices.get(code, {}).get(dates[i - 1])
                current_open = opens.get(code, {}).get(date)
                if not previous_close or not current_open:
                    missing_prices.append(code)
                    holding["missing_days"] = holding.get("missing_days", 0) + 1
                    continue
                holding["missing_days"] = 0
                overnight_return += (
                    holding["w"] / 100 * (current_open / previous_close - 1) * 100
                )

        executed_holdings = deepcopy(holdings)
        for code, holding in executed_holdings.items():
            current_open = opens.get(code, {}).get(date)
            current_close = prices.get(code, {}).get(date)
            if not current_open or not current_close:
                if code not in missing_prices:
                    missing_prices.append(code)
                holding["missing_days"] = holding.get("missing_days", 0) + 1
                continue
            holding["missing_days"] = 0
            if code not in previous_weights:
                holding["peak_price"] = current_open
                holding["entry_date"] = date
            intraday_return += (
                holding["w"] / 100 * (current_close / current_open - 1) * 100
            )
            holding["peak_price"] = max(
                holding.get("peak_price") or current_open,
                current_close,
            )

        nav *= 1 + overnight_return / 100
        nav *= 1 - cost_pct / 100
        nav *= 1 + intraday_return / 100
        dret = (
            (1 + overnight_return / 100) * (1 + intraday_return / 100) - 1
        ) * 100
        peak_nav = max(peak_nav, nav)
        dd = (peak_nav / nav - 1) * 100 if nav > 0 else 100.0

        # 收盘止损只影响下一交易日开盘，不回写当日已实现收益。
        stopped_codes: list[str] = []
        for code, holding in executed_holdings.items():
            current_close = prices.get(code, {}).get(date)
            peak_price = holding.get("peak_price")
            if current_close and peak_price:
                drawdown = (current_close / peak_price - 1) * 100
                if drawdown <= -TRAILING_STOP:
                    stopped_codes.append(code)
        holdings = deepcopy(executed_holdings)
        for code in stopped_codes:
            del holdings[code]

        warmed_benchmark = [
            close
            for history_date, close in benchmark_history
            if history_date <= date
        ][-20:]
        if warmed_benchmark:
            benchmark_nav = warmed_benchmark
        else:
            benchmark_nav.append(
                (benchmark_nav[-1] if benchmark_nav else 1.0) * (1 + bench / 100)
            )
        width = calc_width_ratio(rows)
        raw_market_state = _market_state_from_proxy(benchmark_nav, width, bench)
        if not confirmed_state:
            confirmed_state = raw_market_state
        elif raw_market_state == confirmed_state:
            pending_state = ""
            pending_state_days = 0
        elif raw_market_state == "冰点":
            confirmed_state = raw_market_state
            pending_state = ""
            pending_state_days = 0
        else:
            if raw_market_state != pending_state:
                pending_state = raw_market_state
                pending_state_days = 1
            else:
                pending_state_days += 1
            if pending_state_days >= 2:
                confirmed_state = raw_market_state
                pending_state = ""
                pending_state_days = 0
        market_state = confirmed_state
        target_pos = float(get_target_exposure(market_state))

        # 回撤只降低暴露，不阻断选股；超时解除后设置冷却期并重置高水位。
        if breaker_cooldown > 0:
            breaker_cooldown -= 1
        if not breaker and breaker_cooldown == 0 and dd >= DD_HARD_LIMIT:
            breaker = True
            breaker_days = 0
        if breaker:
            breaker_days += 1
            target_pos = 40.0 if market_state == "冰点" else 60.0
            if dd <= DD_RELEASE or breaker_days >= BREAKER_MAX_DAYS:
                breaker = False
                breaker_days = 0
                breaker_cooldown = BREAKER_COOLDOWN_DAYS
                peak_nav = nav
                dd = 0.0
        elif dd >= DD_SOFT_LIMIT:
            target_pos = min(target_pos, 40.0 if market_state == "冰点" else 70.0)

        # 为旧 scan.json 补齐规则 9 所需的5/20/60日动量。
        for row in rows:
            if not row.ret5:
                row.ret5 = _return_over_days(prices, row.code, dates, i, 5)
            if not row.ret20:
                row.ret20 = _return_over_days(prices, row.code, dates, i, 20)
            if not row.ret60:
                row.ret60 = _return_over_days(prices, row.code, dates, i, 60)

        # 大簇崩盘按不同方向计数，避免同主题多只ETF重复触发。
        cluster_directions: dict[str, dict[str, float]] = {}
        for row in rows:
            major = get_major_sector(row.industry or row.sector)
            if major in {"其他", "宽基", "债券", "货币"}:
                continue
            direction = row.direction or row.name or row.code
            cluster_directions.setdefault(major, {})[direction] = row.pct
        crashing_clusters = {
            major
            for major, direction_pcts in cluster_directions.items()
            if sum(pct < CLUSTER_CRASH_PCT for pct in direction_pcts.values()) >= CLUSTER_MIN_COUNT
        }

        # 只有“跌出TOP30且评分从持有期峰值下降40%”才提前清仓。
        holdable_rows = sorted(
            [
                row for row in rows
                if is_eligible(row, ice_mode=(market_state == "冰点"), for_new_position=False)
            ],
            key=lambda item: -item.score,
        )
        rank_by_code = {row.code: rank for rank, row in enumerate(holdable_rows, 1)}
        row_by_code = {row.code: row for row in rows}
        if has_scan:
            for code in list(holdings):
                holding = holdings[code]
                current = row_by_code.get(code)
                if current is None:
                    if holding.get("missing_days", 0) >= 2:
                        del holdings[code]
                    continue
                holding["peak_score"] = max(holding.get("peak_score", current.score), current.score)
                score_drop = (
                    1 - current.score / holding["peak_score"]
                    if holding["peak_score"] > 0
                    else 0.0
                )
                protected = i - holding.get("entry_index", i) < REBALANCE_DAYS
                if rank_by_code.get(code, 999) > 30 and score_drop >= 0.40 and not protected:
                    del holdings[code]

        rebalance_counter += 1
        is_rebalance = has_scan and rebalance_counter >= REBALANCE_DAYS
        if is_rebalance:
            rebalance_counter = 0

        eligible = [
            row for row in rows
            if row.code not in stopped_codes
            and is_eligible(
                row,
                ice_mode=(market_state == "冰点"),
                crashing_clusters=crashing_clusters,
                for_new_position=True,
            )
        ]

        if has_scan and (is_rebalance or not holdings):
            candidate_by_code = {row.code: row for row in eligible}
            for code in holdings:
                current = row_by_code.get(code)
                if current is None:
                    continue
                major = get_major_sector(current.industry or current.sector)
                if major in crashing_clusters:
                    continue
                if is_eligible(
                    current,
                    ice_mode=(market_state == "冰点"),
                    for_new_position=False,
                ):
                    candidate_by_code[code] = current
            selected = _pick_ranked(
                list(candidate_by_code.values()),
                TARGET_SELECTION_COUNT,
                preferred_codes=set(holdings),
            )
            selected_codes = {row.code for row in selected}
            for code in list(holdings):
                if code not in selected_codes:
                    del holdings[code]
            for rank, row in enumerate(selected, 1):
                if row.code not in holdings:
                    holdings[row.code] = {
                        "w": 0.0,
                        "sector": row.industry or row.sector,
                        "peak_price": prices.get(row.code, {}).get(date, row.close),
                        "entry_date": date,
                        "entry_index": i,
                        "peak_score": row.score,
                        "rank": rank,
                        "missing_days": 0,
                    }
                holdings[row.code]["rank"] = rank
        elif has_scan and len(holdings) < TARGET_SELECTION_COUNT:
            used_codes = set(holdings)
            major_count: dict[str, int] = {}
            for item in holdings.values():
                major = get_major_sector(item.get("sector", ""))
                major_count[major] = major_count.get(major, 0) + 1
            for row in sorted(eligible, key=lambda item: -item.score):
                if row.code in used_codes:
                    continue
                major = get_major_sector(row.industry or row.sector)
                if major_count.get(major, 0) >= 2:
                    continue
                holdings[row.code] = {
                    "w": 0.0,
                    "sector": row.industry or row.sector,
                    "peak_price": prices.get(row.code, {}).get(date, row.close),
                    "entry_date": date,
                    "entry_index": i,
                    "peak_score": row.score,
                    "rank": len(holdings) + 1,
                    "missing_days": 0,
                }
                used_codes.add(row.code)
                major_count[major] = major_count.get(major, 0) + 1
                if len(holdings) >= TARGET_SELECTION_COUNT:
                    break

        # 市场状态变化或持仓变化后必须立刻重算总仓位，禁止残留旧档位。
        ordered_codes = sorted(
            holdings,
            key=lambda code: (
                holdings[code].get("rank", 99),
                -row_by_code.get(code, ETFData()).score,
            ),
        )[:TARGET_SELECTION_COUNT]
        for code in list(holdings):
            if code not in ordered_codes:
                del holdings[code]
        new_weights = _scaled_rank_weights(market_state, target_pos, len(ordered_codes))
        for code, weight in zip(ordered_codes, new_weights):
            holdings[code]["w"] = weight

        # 此处 holdings 是本日盘后生成、供下一交易日开盘执行的目标组合。
        next_weights = {code: float(item.get("w", 0)) for code, item in holdings.items()}
        net_return = (nav / nav_before_day - 1) * 100 if nav_before_day > 0 else 0.0

        actual_exposure = sum(execution_weights.values())
        executed_holdings_text = ", ".join(
            f"{code}@{weight:.1f}%"
            for code, weight in sorted(
                execution_weights.items(),
                key=lambda pair: -pair[1],
            )
        ) or "空仓"
        holdings_text = ", ".join(
            f"{code}@{item['w']:.1f}%"
            for code, item in sorted(holdings.items(), key=lambda pair: -pair[1]["w"])
        ) or "空仓"

        flags = []
        if breaker:
            flags.append("熔断降仓")
        elif dd >= DD_SOFT_LIMIT:
            flags.append("回撤降仓")
        if crashing_clusters:
            flags.append(f"崩盘:{','.join(sorted(crashing_clusters))}")
        if missing_prices:
            flags.append(f"缺价:{len(missing_prices)}")
        if stopped_codes:
            flags.append(f"止损:{','.join(stopped_codes)}")
        if len(holdings) < TARGET_SELECTION_COUNT:
            flags.append(f"候选不足:{len(holdings)}")
        if not has_scan:
            flags.append("无扫描:延续")
        flag_text = "+".join(flags) if flags else "正常"

        log.append({
            "date": date,
            "bench": bench,
            "dret": dret,
            "overnight_return": overnight_return,
            "intraday_return": intraday_return,
            "net_return": net_return,
            "nav": nav,
            "dd": dd,
            "width": width,
            "raw_market_state": raw_market_state,
            "market_state": market_state,
            "pos_target": target_pos,
            "actual_exposure": actual_exposure,
            "turnover": turnover_pct,
            "cost_pct": cost_pct,
            "flags": flag_text,
            "executed_holdings": executed_holdings_text,
            "holdings": holdings_text,
            "is_rebalance": is_rebalance,
            "is_signal_day": has_scan,
        })
    return log


def load_calendar_prices(start_date: str, end_date: str) -> tuple[dict, dict, list[str]]:
    """读取固定ETF池完整交易日行情，用于缺扫描日的持仓收益计算。"""
    from backtest import _fetch_hist_cached, fetch_all_hist
    from etf_analyzer import load_etf_txt

    pool = load_etf_txt()
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=45)
    ).strftime("%Y%m%d")
    benchmark_cfg = next(
        (cfg for cfg in pool.values() if cfg["code"] == BENCHMARK_CODE),
        None,
    )
    if benchmark_cfg is None:
        raise RuntimeError(f"基准 {BENCHMARK_CODE} 不在 scripts/etf.txt 中")

    benchmark = _fetch_hist_cached(
        BENCHMARK_CODE,
        warmup_start,
        end,
        benchmark_cfg.get("type", "ETF"),
    )
    if benchmark.empty:
        raise RuntimeError("完整交易日历获取失败")

    history = fetch_all_hist(pool, start, end)
    prices: dict[str, dict[str, float]] = {}
    opens: dict[str, dict[str, float]] = {}
    for code, frame in history.items():
        for _, row in frame.iterrows():
            date = row["date"].strftime("%Y-%m-%d")
            prices.setdefault(code, {})[date] = float(row["close"])
            opens.setdefault(code, {})[date] = float(row.get("open", row["close"]))
    for _, row in benchmark.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        prices.setdefault(BENCHMARK_CODE, {})[date] = float(row["close"])
        opens.setdefault(BENCHMARK_CODE, {})[date] = float(row.get("open", row["close"]))

    dates = [
        value.strftime("%Y-%m-%d")
        for value in benchmark["date"]
        if start_date <= value.strftime("%Y-%m-%d") <= end_date
    ]
    return prices, opens, sorted(set(dates))


def _summary(log: list[dict]) -> dict:
    if not log:
        return {}
    benchmark_nav = 1.0
    for row in log:
        benchmark_nav *= 1 + row["bench"] / 100
    return {
        "final_return": (log[-1]["nav"] - 1) * 100,
        "benchmark_return": (benchmark_nav - 1) * 100,
        "max_drawdown": max(row["dd"] for row in log),
        "avg_target": sum(row["pos_target"] for row in log) / len(log),
        "avg_actual": sum(row["actual_exposure"] for row in log) / len(log),
        "turnover": sum(row["turnover"] for row in log) / 100,
        "cost_points": sum(row["cost_pct"] for row in log),
        "win_rate": sum(row["net_return"] > 0 for row in log) / len(log) * 100,
    }


def write_daily_report(
    log: list[dict],
    requested_start: str,
    requested_end: str,
    output_path: Path,
) -> Path:
    summary = _summary(log)
    signal_days = sum(row["is_signal_day"] for row in log)
    carry_days = len(log) - signal_days
    lines = [
        f"# 固定ETF池逐日预测回放 — {requested_start} ~ {requested_end}",
        "",
        "## 口径",
        "",
        "- 所有预测标的均来自 `scripts/etf.txt`。",
        "- 有 `scan.json` 的交易日使用当日收盘信号，并在下一交易日开盘执行，杜绝盘后信号按当日收盘价成交。",
        "- 日收益包含旧仓隔夜收益、开盘调仓成本和新仓日内收益；首日只生成预测，不虚构持仓收益。",
        "- 没有扫描快照的交易日不选择新标的，只延续原组合；回撤降仓和止损仍可生效。",
        f"- 实际行情区间：{log[0]['date']} ~ {log[-1]['date']}，共 {len(log)} 个交易日；信号日 {signal_days} 个，延续日 {carry_days} 个。",
        f"- 收益已扣除单边 {COST_BPS:g}bp 换手成本；日线模型不能还原盘中成交顺序和真实滑点。",
        "",
        "## 总结果",
        "",
        f"- 策略总收益：{summary['final_return']:+.2f}%",
        f"- 沪深300ETF：{summary['benchmark_return']:+.2f}%",
        f"- 超额收益：{summary['final_return'] - summary['benchmark_return']:+.2f}%",
        f"- 最大回撤：{summary['max_drawdown']:.2f}%",
        f"- 日胜率：{summary['win_rate']:.1f}%",
        f"- 平均目标/实际仓位：{summary['avg_target']:.1f}% / {summary['avg_actual']:.1f}%",
        f"- 累计单边换手：{summary['turnover']:.2f} 倍",
        f"- 累计成本影响：{summary['cost_points']:.2f} 个百分点（逐日近似和）",
        "",
        "## 每日收益",
        "",
        "| 日期 | 信号 | 沪深300 | 隔夜 | 日内 | 组合毛收益 | 成本 | 组合净收益 | 累计收益 | 回撤 | 市场 | 实际仓位 | 当日执行组合 | 盘后预测/延续组合 | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in log:
        signal = "新扫描" if row["is_signal_day"] else "延续"
        lines.append(
            f"| {row['date']} | {signal} | {row['bench']:+.2f}% "
            f"| {row['overnight_return']:+.2f}% | {row['intraday_return']:+.2f}% "
            f"| {row['dret']:+.2f}% "
            f"| -{row['cost_pct']:.3f}% | {row['net_return']:+.2f}% "
            f"| {(row['nav'] - 1) * 100:+.2f}% | {row['dd']:.2f}% "
            f"| {row['market_state']} | {row['actual_exposure']:.1f}% "
            f"| {row['executed_holdings']} | {row['holdings']} | {row['flags']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="完整策略回测 v2")
    parser.add_argument("--start", default="2026-07-08")
    parser.add_argument("--end", default="2026-07-22")
    parser.add_argument("--days", default="")
    parser.add_argument("--calendar", action="store_true", help="加载完整日线，缺扫描日延续持仓")
    parser.add_argument("--output", default="", help="可选 Markdown 日报输出路径")
    args = parser.parse_args()

    scans = load_all_scans()
    all_dates = sorted(scans.keys())

    start, end = args.start, args.end
    if args.days == "2w":
        start, end = "2026-07-08", "2026-07-22"
    elif args.days == "full":
        start, end = all_dates[0], all_dates[-1]

    if args.calendar:
        prices, opens, dates = load_calendar_prices(start, end)
    else:
        prices = {}
        for date in all_dates:
            for r in scans[date]["rows"]:
                prices.setdefault(r.code, {})[date] = r.close
        opens = prices
        dates = [d for d in all_dates if start <= d <= end]
    if not dates:
        print("无可用数据。可用日期:", all_dates[0], "→", all_dates[-1])
        return

    log = run_backtest(scans, prices, dates, opens=opens)

    print("=" * 120)
    print(f"完整策略回测 v2 | {dates[0]} → {dates[-1]} ({len(dates)}个交易日)")
    print("规则: 1(催化代理) 2(涨跌比仓位) 3(板块分散) 4(3日重排) 5(崩塌清仓)")
    print("      6(回撤熔断) 8(不追高) 9(动量极端) 11(大簇崩盘) 12(假突破)")
    print("      QDII商品豁免 + 15%移动止损 + 冰点:确认/扩散期")
    print("=" * 120)
    print(
        f"{'日期':<11} {'沪深300':>7} {'当日':>8} {'累计':>8} {'DD':>6} "
        f"{'ETF宽度':>7} {'环境':<6} {'目标/实际':>10} {'状态':<22} {'持仓'}"
    )
    print("-" * 120)

    for e in log:
        print(
            f"{e['date']:<11} {e['bench']:+6.2f}% {e['net_return']:+7.2f}% "
            f"{(e['nav']-1)*100:+7.2f}% {e['dd']:5.1f}% "
            f"{e['width']:6.2f} {e['market_state']:<6} "
            f"{e['pos_target']:3.0f}/{e['actual_exposure']:3.0f}% "
            f"{e['flags']:<22} {e['holdings']}"
        )

    summary = _summary(log)

    print("-" * 120)
    print(
        f"策略: {summary['final_return']:+.2f}% | 最大回撤: {summary['max_drawdown']:.2f}% | "
        f"沪深300: {summary['benchmark_return']:+.2f}% | "
        f"超额: {summary['final_return'] - summary['benchmark_return']:+.2f}%"
    )
    print(
        f"平均目标/实际仓位: {summary['avg_target']:.1f}%/{summary['avg_actual']:.1f}% | "
        f"累计单边换手: {summary['turnover']:.2f}x | 成本: {COST_BPS:.0f}bp"
    )
    if args.output:
        report = write_daily_report(log, start, end, Path(args.output))
        print(f"逐日报告: {report}")


if __name__ == "__main__":
    main()
