"""
ETF 三层分析框架回测
====================

无未来函数约束：
- 信号日：调仓月开始前最后一个可交易日。
- 选 ETF：只使用信号日及以前的行情计算阶段和评分。
- 调仓：按月度或周度重排全 ETF 池，但已有强趋势持仓允许续持。
- 持有期收益：从调仓周期第一个可交易日至周期最后一个可交易日。

用法：
  python scripts/backtest.py --start 2025-01-01 --end 2026-06-29
  python scripts/backtest.py --freq weekly --start 2025-01-01 --end 2026-06-29
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from etf_analyzer import (  # noqa: E402
    BENCHMARK_CODE,
    NEW_MONEY_STAGES,
    STAGE_PRIORITY,
    _rank_key,
    calc_auto_score,
    calc_consecutive_strong,
    calc_etf_metrics,
    classify_stage,
    fetch_etf_hist,
    load_etf_txt,
)

HOLDABLE_STAGES = set(NEW_MONEY_STAGES) | {"趋势回撤期", "加速期⚠"}
COMMODITY_CATEGORIES = {"商品", "资源"}
COMMODITY_TRAIL_STOP = -35.0
NORMAL_TRAIL_STOP = -18.0
COMMODITY_PARABOLIC_R20 = 80.0
COMMODITY_PARABOLIC_R60 = 150.0
EXPOSURE_BY_MARKET = {
    "主升": 1.00,
    "震荡": 0.80,
    "退潮": 0.60,
    "退潮末期": 0.40,
    "冰点": 0.30,
    "未知": 0.60,
}
# 追高硬门禁：符合任一条件时该腿仅按试探仓 10% 建仓
# 只在防守市场（退潮/退潮末期/冰点）触发；主升/震荡下真趋势不做机械稀释。
CHASE_5D_RETURN_LIMIT = 25.0
CHASE_SINGLE_DAY_LIMIT = 4.0
PROBE_WEIGHT = 0.10
DEFENSIVE_STATES = {"退潮", "退潮末期", "冰点"}
# 组合最低入选分数：分数低于此的候选不进入主仓；不再机械补足到 TOP N
MIN_ENTRY_SCORE = 45.0
# 月频/周频止损规则：max(固定百分比, N × ATR)
PERIOD_STOP_RULES = {
    "monthly": {
        "normal_initial": 0.12,
        "normal_trailing": 0.18,
        "commodity_initial": 0.22,
        "commodity_trailing": 0.35,
    },
    "weekly": {
        "normal_initial": 0.08,
        "normal_trailing": 0.12,
        "commodity_initial": 0.14,
        "commodity_trailing": 0.20,
    },
}
ATR_MULTIPLIER = 2.5  # ATR 倍率，与固定百分比取较大值
WEEKLY_MIN_AMOUNT_YI = 1.0
WEEKLY_MAX_BROAD = 1
HOLD_BONUS_SCORE_FLOOR = 50.0
HOLD_BONUS_SCORE_STRONG = 55.0
HOLD_BONUS = {
    "weekly": {"weak": 2.0, "normal": 5.0, "strong": 7.0},
    "monthly": {"weak": 3.0, "normal": 6.0, "strong": 8.0},
}


# 简易市场状态判断（回测无实时 market_breadth，用基准趋势替代）
def _simple_market_state(bench_df, signal_date):
    """
    用沪深300 20日均线判断市场状态——保持简单，不过度拟合。

    - close > MA20*1.02 → 主升
    - close > MA20      → 震荡
    - close > MA20*0.98 → 退潮 (-2% ~ 0)
    - close > MA20*0.95 → 退潮末期 (-5% ~ -2%)
    - else              → 冰点 (<-5%)
    """
    td = pd.to_datetime(signal_date)
    recent = bench_df[bench_df["date"] <= td].tail(25)
    if len(recent) < 22:
        return "未知", EXPOSURE_BY_MARKET["未知"]
    ma20 = recent["close"].iloc[-21:].mean()
    close = recent["close"].iloc[-1]
    if close > ma20 * 1.02:
        return "主升", EXPOSURE_BY_MARKET["主升"]
    elif close > ma20:
        return "震荡", EXPOSURE_BY_MARKET["震荡"]
    elif close > ma20 * 0.98:
        return "退潮", EXPOSURE_BY_MARKET["退潮"]
    elif close > ma20 * 0.95:
        return "退潮末期", EXPOSURE_BY_MARKET["退潮末期"]
    else:
        return "冰点", EXPOSURE_BY_MARKET["冰点"]


def _fmt_date(ts) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


def _yyyymmdd(ts) -> str:
    return pd.to_datetime(ts).strftime("%Y%m%d")


def _latest_before(df: pd.DataFrame, dt) -> pd.Timestamp | None:
    rows = df[df["date"] < pd.to_datetime(dt)]
    if rows.empty:
        return None
    return pd.to_datetime(rows.iloc[-1]["date"])


def _first_between(df: pd.DataFrame, start, end) -> pd.Timestamp | None:
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    if rows.empty:
        return None
    return pd.to_datetime(rows.iloc[0]["date"])


def _last_between(df: pd.DataFrame, start, end) -> pd.Timestamp | None:
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    if rows.empty:
        return None
    return pd.to_datetime(rows.iloc[-1]["date"])


def _close_between(df: pd.DataFrame, start, end) -> tuple[float, float, str, str] | None:
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    if len(rows) < 2:
        return None
    first = rows.iloc[0]
    last = rows.iloc[-1]
    return float(first["close"]), float(last["close"]), _fmt_date(first["date"]), _fmt_date(last["date"])


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    peak = cumulative.cummax()
    dd = cumulative / peak - 1
    return round(float(dd.min()) * 100, 2)


def _periods(start_dt: pd.Timestamp, end_dt: pd.Timestamp, freq: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if freq == "monthly":
        starts = pd.date_range(start=start_dt, end=end_dt, freq="MS")
        if not starts.empty and starts[0] > start_dt and start_dt.day == 1:
            starts = starts.insert(0, start_dt)
        periods = []
        for start in starts:
            end = min(start + pd.offsets.MonthEnd(0), end_dt)
            periods.append((start.strftime("%Y-%m"), pd.to_datetime(start), pd.to_datetime(end)))
        return periods

    if freq == "weekly":
        starts = pd.date_range(start=start_dt, end=end_dt, freq="W-MON")
        if starts.empty:
            starts = pd.DatetimeIndex([start_dt])
        periods = []
        for start in starts:
            end = min(start + timedelta(days=6), end_dt)
            iso = pd.to_datetime(start).isocalendar()
            label = f"{iso.year}-W{int(iso.week):02d}"
            periods.append((label, pd.to_datetime(start), pd.to_datetime(end)))
        return periods

    raise ValueError(f"unsupported freq: {freq}")


def _calc_atr(df: pd.DataFrame, period: int = 20) -> float | None:
    """计算 ATR (Average True Range)，用于动态止损宽度。"""
    df2 = df.copy()
    df2["prev_close"] = df2["close"].shift(1)
    df2["tr1"] = df2["high"] - df2["low"]
    df2["tr2"] = abs(df2["high"] - df2["prev_close"])
    df2["tr3"] = abs(df2["low"] - df2["prev_close"])
    df2["tr"] = df2[["tr1", "tr2", "tr3"]].max(axis=1)
    atr_series = df2["tr"].tail(period)
    if len(atr_series) < 5:
        return None
    return float(atr_series.mean())


def _simulate_period_return(
    df: pd.DataFrame,
    start,
    end,
    item: dict,
    freq: str,
) -> tuple[float, str, str, str]:
    """
    用日线 low 模拟周期内止损。
    买入价按周期第一个交易日收盘价，止损从下一交易日开始判断。
    """
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
    if len(rows) < 2:
        return 0.0, "", "", "数据不足"

    rows = rows.sort_values("date").reset_index(drop=True)
    first = rows.iloc[0]
    entry_close = float(first["close"])
    if entry_close <= 0:
        return 0.0, "", "", "入场价无效"

    # ATR(20) 动态调整止损
    atr = _calc_atr(df, 20)
    atr_pct = atr / entry_close if entry_close > 0 and atr else 0
    rules = PERIOD_STOP_RULES[freq]
    if _is_commodity_like(item):
        initial_stop_pct = max(rules["commodity_initial"], ATR_MULTIPLIER * atr_pct)
        trailing_stop_pct = max(rules["commodity_trailing"], ATR_MULTIPLIER * atr_pct * 1.3)
    else:
        initial_stop_pct = max(rules["normal_initial"], ATR_MULTIPLIER * atr_pct)
        trailing_stop_pct = max(rules["normal_trailing"], ATR_MULTIPLIER * atr_pct * 1.3)

    peak = entry_close
    hard_stop = entry_close * (1 - initial_stop_pct)
    for _, row in rows.iloc[1:].iterrows():
        high = float(row["high"]) if "high" in rows.columns and pd.notna(row.get("high")) else float(row["close"])
        low = float(row["low"]) if "low" in rows.columns and pd.notna(row.get("low")) else float(row["close"])
        peak = max(peak, high)
        trail_price = peak * (1 - trailing_stop_pct)
        stop_price = max(hard_stop, trail_price)

        if low <= stop_price:
            ret = (stop_price / entry_close - 1) * 100
            return ret, _fmt_date(first["date"]), _fmt_date(row["date"]), f"止损@{stop_price:.3f}"

    last = rows.iloc[-1]
    ret = (float(last["close"]) / entry_close - 1) * 100
    return ret, _fmt_date(first["date"]), _fmt_date(last["date"]), "持有到期"


def _last_row_on_or_before(df: pd.DataFrame, dt) -> pd.Series | None:
    rows = df[df["date"] <= pd.to_datetime(dt)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _ma_close(df: pd.DataFrame, dt, window: int) -> float | None:
    rows = df[df["date"] <= pd.to_datetime(dt)].tail(window)
    if len(rows) < max(5, window // 2):
        return None
    return float(rows["close"].mean())


def _is_commodity_like(item: dict) -> bool:
    product_type = str(item.get("product_type", "")).upper()
    return product_type == "LOF" or item.get("category") in COMMODITY_CATEGORIES


def _trend_stop_triggered(
    item: dict,
    data: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    holding_since: str,
) -> tuple[bool, str]:
    """
    只用信号日及以前数据判断趋势止损。
    商品/LOF 波动天然更大，使用更宽的最高点回撤阈值。
    """
    df = data.get(item["etf_code"])
    if df is None or df.empty:
        return True, "缺少行情"

    last = _last_row_on_or_before(df, signal_date)
    if last is None:
        return True, "信号日前无行情"

    since = pd.to_datetime(holding_since)
    history = df[(df["date"] >= since) & (df["date"] <= pd.to_datetime(signal_date))]
    if history.empty:
        history = df[df["date"] <= pd.to_datetime(signal_date)].tail(20)
    if history.empty:
        return True, "持仓历史不足"

    close = float(last["close"])
    peak = float(history["close"].max())
    drawdown = (close / peak - 1) * 100 if peak > 0 else 0.0
    ma20 = _ma_close(df, signal_date, 20)
    ret20 = float(item.get("metrics", {}).get("ret_20d") or 0)
    ret60 = float(item.get("metrics", {}).get("ret_60d") or 0)

    if _is_commodity_like(item):
        if ret20 >= COMMODITY_PARABOLIC_R20 and ret60 >= COMMODITY_PARABOLIC_R60:
            return True, f"商品抛物线止盈: 20日{ret20:.1f}%, 60日{ret60:.1f}%"
        if drawdown <= COMMODITY_TRAIL_STOP and ma20 is not None and close < ma20:
            return True, f"商品趋势止损: 高点回撤{drawdown:.1f}%且跌破MA20"
        if ma20 is not None and close < ma20 * 0.90 and ret20 < 0:
            return True, "商品跌破MA20过深且20日转弱"
        if ret20 > 20 and ret60 > 40:
            return False, "商品大趋势仍强"
    else:
        if drawdown <= NORMAL_TRAIL_STOP and ma20 is not None and close < ma20:
            return True, f"趋势止损: 高点回撤{drawdown:.1f}%且跌破MA20"
        if ma20 is not None and close < ma20 * 0.97 and ret20 < 0:
            return True, "跌破MA20且20日转弱"

    return False, "未触发趋势止损"


def _should_hold_position(
    previous: dict,
    ranked_by_code: dict[str, dict],
    data: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
) -> tuple[bool, str, dict | None]:
    current = ranked_by_code.get(previous["etf_code"])
    if current is None:
        return False, "信号日无有效排名", None

    holding_since = previous.get("holding_since") or current.get("signal_date")
    stopped, stop_reason = _trend_stop_triggered(current, data, signal_date, holding_since)
    if stopped:
        return False, stop_reason, current

    stage = current["stage"][0]
    metrics = current.get("metrics", {})
    ret20 = float(metrics.get("ret_20d") or 0)
    ret60 = float(metrics.get("ret_60d") or 0)

    if stage in HOLDABLE_STAGES:
        return True, f"{stage}续持", current
    if _is_commodity_like(current) and ret20 > 20 and ret60 > 40:
        return True, "商品大趋势续持", current
    return False, f"{stage}不再持有", current


def _new_position_allowed(item: dict, market_state: str) -> bool:
    """弱市过滤：冰点不新开仓，退潮/退潮末期提高新开仓门槛。"""
    if market_state == "冰点":
        return False

    metrics = item.get("metrics", {})
    score = float(item.get("score", {}).get("total") or 0)
    stage = item.get("stage", ("", "", ""))[0]
    ret5 = float(metrics.get("ret_5d") or 0)
    ret20 = float(metrics.get("ret_20d") or 0)

    if market_state == "退潮末期":
        # 退潮末期只做主力真流入 + 独立叙事的候选：分数≥60、且短期动量非负
        if score < 60 or stage not in {"扩散期", "加速期", "确认期"}:
            return False
        return ret5 > 0

    if market_state == "退潮":
        if score < 55 or stage not in {"扩散期", "加速期", "确认期"}:
            return False
        if item.get("industry") == "周期" or item.get("category") in COMMODITY_CATEGORIES:
            return ret5 > 0 and ret20 > 0 and score >= 60
        return ret5 > 0 or ret20 > 0

    return True


def _is_chase_high(item: dict, market_state: str = "") -> bool:
    """
    L008 硬门禁：单日 >4% 或 5日累计 >25% 视为追高。
    只在防守市场（退潮/退潮末期/冰点）里触发；主升/震荡下强趋势不做机械稀释。
    """
    if market_state and market_state not in DEFENSIVE_STATES:
        return False
    metrics = item.get("metrics", {})
    pct = float(metrics.get("pct_chg") or 0)
    ret5 = float(metrics.get("ret_5d") or 0)
    return pct > CHASE_SINGLE_DAY_LIMIT or ret5 > CHASE_5D_RETURN_LIMIT


def _selection_priority(item: dict, freq: str) -> tuple:
    """
    最终组合排序：旧仓可以获得很小的续持加分，但必须和新方向同台竞争。
    避免弱旧仓仅因“可续持”就挤掉更强新主线。
    """
    metrics = item.get("metrics", {})
    score = float(item.get("score", {}).get("total") or 0)
    if item.get("entry_type") == "续持":
        bonus_table = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])
        if score >= HOLD_BONUS_SCORE_STRONG:
            score += bonus_table["strong"]
        elif score >= HOLD_BONUS_SCORE_FLOOR:
            score += bonus_table["normal"]
        else:
            score += bonus_table["weak"]
    stage = item.get("stage", ("", "", ""))[0]
    return (
        score,
        STAGE_PRIORITY.get(stage, -99),
        float(metrics.get("ret_20d") or 0),
        float(metrics.get("ret_5d") or 0),
        float(metrics.get("amount_yi") or 0),
    )


def _select_top_candidates(candidate_pool: list[dict], top_n: int, freq: str) -> list[dict]:
    """
    从旧仓和新候选中统一选 TOP。
    - 分数低于 MIN_ENTRY_SCORE 的候选不进入组合（宁少不凑）。
    - 周度更强调主线弹性：最多保留 1 个宽基，避免宽基挤占产业 ETF 名额。
    """
    ordered = sorted(candidate_pool, key=lambda r: _selection_priority(r, freq), reverse=True)
    qualified = [r for r in ordered if float(r.get("score", {}).get("total") or 0) >= MIN_ENTRY_SCORE]
    if freq != "weekly":
        return qualified[:top_n]

    selected = []
    delayed_broad = []
    broad_count = 0
    for r in qualified:
        if len(selected) >= top_n:
            break
        if r.get("industry") == "宽基" and broad_count >= WEEKLY_MAX_BROAD:
            delayed_broad.append(r)
            continue
        selected.append(r)
        if r.get("industry") == "宽基":
            broad_count += 1

    if len(selected) < top_n:
        for r in delayed_broad:
            if len(selected) >= top_n:
                break
            selected.append(r)
    return selected


def fetch_all_hist(etf_pool: dict, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    data = {}
    total = len(etf_pool)
    for i, cfg in enumerate(etf_pool.values(), 1):
        df = fetch_etf_hist(
            cfg["code"],
            start_date,
            end_date,
            cfg.get("type", "ETF"),
        )
        if not df.empty:
            data[cfg["code"]] = df
        if i % 20 == 0 or i == total:
            print(f"  行情已获取 {i}/{total}", flush=True)
    return data


def rank_on_signal_date(
    etf_pool: dict,
    data: dict[str, pd.DataFrame],
    bench_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    min_amount_yi: float,
    allowed_stages: set[str] | None = NEW_MONEY_STAGES,
) -> list[dict]:
    signal_date_str = _fmt_date(signal_date)
    bench_metrics = calc_etf_metrics(bench_df, signal_date_str)
    bench_pct = bench_metrics.get("pct_chg", 0.0) if "error" not in bench_metrics else 0.0

    ranked = []
    for direction, cfg in etf_pool.items():
        if cfg.get("cat") == "货币":
            continue
        df = data.get(cfg["code"])
        if df is None or df.empty:
            continue

        effective_signal = _last_between(df, signal_date - timedelta(days=14), signal_date)
        if effective_signal is None:
            continue
        metrics = calc_etf_metrics(df, _fmt_date(effective_signal))
        if "error" in metrics:
            continue
        if float(metrics.get("amount_yi") or 0) < min_amount_yi:
            continue

        strong_days = calc_consecutive_strong(df, bench_df, _fmt_date(effective_signal))
        stage = classify_stage(metrics, strong_days)
        if allowed_stages is not None and stage[0] not in allowed_stages:
            continue

        score = calc_auto_score(
            metrics,
            stage[0],
            bench_pct,
            is_qdii=cfg.get("is_qdii", False),
            premium_pct=cfg.get("premium_pct"),
            consecutive_strong=strong_days,
        )
        ranked.append({
            "direction": direction,
            "etf_code": cfg["code"],
            "etf_name": cfg["name"],
            "category": cfg["cat"],
            "industry": cfg["ind"],
            "product_type": cfg.get("type", "ETF"),
            "is_qdii": cfg.get("is_qdii", False),
            "metrics": metrics,
            "stage": stage,
            "strong_days": strong_days,
            "score": score,
            "signal_date": _fmt_date(effective_signal),
        })

    return sorted(ranked, key=_rank_key, reverse=True)


def run_backtest(
    start_date: str,
    end_date: str,
    top_n: int = 3,
    min_amount_yi: float = 0.2,
    max_etfs: int | None = None,
    freq: str = "monthly",
) -> pd.DataFrame:
    etf_pool = load_etf_txt()
    if max_etfs:
        full_pool = etf_pool
        etf_pool = dict(list(full_pool.items())[:max_etfs])
        if not any(v["code"] == BENCHMARK_CODE for v in etf_pool.values()):
            bench_item = next(((k, v) for k, v in full_pool.items() if v["code"] == BENCHMARK_CODE), None)
            if bench_item:
                etf_pool[bench_item[0]] = bench_item[1]
    print(f"Loaded {len(etf_pool)} ETF/LOF instruments")

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    warmup_start = start_dt - timedelta(days=90)
    fetch_start = _yyyymmdd(warmup_start)
    fetch_end = _yyyymmdd(end_dt + timedelta(days=5))
    effective_min_amount_yi = max(min_amount_yi, WEEKLY_MIN_AMOUNT_YI) if freq == "weekly" else min_amount_yi

    bench_cfg = next((v for v in etf_pool.values() if v["code"] == BENCHMARK_CODE), None)
    if bench_cfg is None:
        raise RuntimeError(f"基准 {BENCHMARK_CODE} 不在 scripts/etf.txt 中")
    bench_df = fetch_etf_hist(BENCHMARK_CODE, fetch_start, fetch_end, bench_cfg.get("type", "ETF"))
    if bench_df.empty:
        raise RuntimeError("沪深300ETF 基准行情获取失败")

    data = fetch_all_hist(etf_pool, fetch_start, fetch_end)
    periods = _periods(start_dt, end_dt, freq)
    rows = []
    holdings: dict[str, dict] = {}

    for label, period_start, period_end in periods:
        signal_date = _latest_before(bench_df, period_start)
        entry_date = _first_between(bench_df, period_start, period_end)
        exit_date = _last_between(bench_df, period_start, period_end)
        if signal_date is None or entry_date is None or exit_date is None or entry_date >= exit_date:
            continue
        market_state, exposure = _simple_market_state(bench_df, signal_date)

        ranked_all = rank_on_signal_date(
            etf_pool,
            data,
            bench_df,
            signal_date,
            effective_min_amount_yi,
            allowed_stages=None,
        )
        if not ranked_all:
            print(f"  {label}: 无候选标的")
            holdings = {}
            continue

        ranked_by_code = {r["etf_code"]: r for r in ranked_all}
        new_ranked = [
            r for r in ranked_all
            if r["stage"][0] in NEW_MONEY_STAGES
            and _new_position_allowed(r, market_state)
        ]
        if not new_ranked and market_state != "冰点":
            new_ranked = [
                r for r in ranked_all
                if r["stage"][0] in NEW_MONEY_STAGES
            ][:top_n]

        carry_candidates = []
        sold_notes = []
        for previous in holdings.values():
            keep, reason, current = _should_hold_position(previous, ranked_by_code, data, signal_date)
            if keep and current is not None:
                carry_candidates.append({
                    **current,
                    "entry_type": "续持",
                    "hold_reason": reason,
                    "holding_since": previous.get("holding_since") or current.get("signal_date"),
                })
            else:
                sold_notes.append(f"{previous['etf_name']}({reason})")

        candidate_pool = list(carry_candidates)
        selected_codes = {r["etf_code"] for r in candidate_pool}
        for r in new_ranked:
            if r["etf_code"] in selected_codes:
                continue
            candidate_pool.append({
                **r,
                "entry_type": "新开",
                "hold_reason": "当期新排名入选",
                "holding_since": _fmt_date(entry_date),
            })
            selected_codes.add(r["etf_code"])

        selected = _select_top_candidates(candidate_pool, top_n, freq)
        selected_codes = {r["etf_code"] for r in selected}
        for r in carry_candidates:
            if r["etf_code"] not in selected_codes:
                sold_notes.append(f"{r['etf_name']}(被更强方向替换)")

        returns = []
        realized_selected = []
        for r in selected:
            period_ret, real_entry, real_exit, exit_note = _simulate_period_return(
                data[r["etf_code"]],
                entry_date,
                exit_date,
                r,
                freq,
            )
            if not real_entry:
                continue
            realized_selected.append({
                **r,
                "period_return": period_ret,
                "real_entry": real_entry,
                "real_exit": real_exit,
                "exit_note": exit_note,
            })
            returns.append(period_ret)

        if not realized_selected:
            bench_pair = _close_between(bench_df, entry_date, exit_date)
            bench_ret = (bench_pair[1] / bench_pair[0] - 1) * 100 if bench_pair else 0.0
            rows.append({
                "month": label,
                "signal_date": _fmt_date(signal_date),
                "entry_date": _fmt_date(entry_date),
                "exit_date": _fmt_date(exit_date),
                "market_state": market_state,
                "exposure": 0.0,
                "return_pct": 0.0,
                "raw_return_pct": 0.0,
                "benchmark_pct": round(bench_ret, 2),
                "top_names": "现金",
                "top_stages": "—",
                "item_returns": "—",
                "hold_reasons": "市场过滤，无可执行标的",
                "sold": "; ".join(sold_notes) if sold_notes else "—",
            })
            print(f"  {label}: state={market_state}, exposure=0%, 无可持有标的")
            holdings = {}
            continue

        selected = realized_selected
        # 分腿加权：
        # - 追高腿（单日 >4% 或 5日 >25%，仅退潮/冰点触发）按 PROBE_WEIGHT=10% 试探；
        # - 其余腿平摊剩余目标仓位；
        # - 目标仓位 = exposure × min(1, n_qualified / max(1, ceil(top_n/2)))，
        #   即单腿时至少建半仓，两腿及以上打满目标，避免"仅一强腿被机械稀释"。
        n_legs = len(selected)
        min_divisor = max(1, (top_n + 1) // 2)  # top_n=3 → 2
        chase_legs = [r for r in selected if _is_chase_high(r, market_state)]
        non_chase = [r for r in selected if not _is_chase_high(r, market_state)]
        probe_total = PROBE_WEIGHT * len(chase_legs)
        target_deploy = exposure * min(1.0, len(non_chase) / min_divisor)
        remaining = max(0.0, target_deploy - probe_total)
        per_leg = remaining / len(non_chase) if non_chase else 0.0

        weighted_sum = 0.0
        weights = []
        for r in selected:
            if _is_chase_high(r, market_state):
                w = PROBE_WEIGHT
            else:
                w = per_leg
            weights.append(w)
            weighted_sum += w * r["period_return"]
        deployed_exposure = sum(weights)
        raw_return = sum(r["period_return"] for r in selected) / n_legs if n_legs else 0.0
        portfolio_ret = weighted_sum
        bench_pair = _close_between(bench_df, entry_date, exit_date)
        bench_ret = 0.0
        if bench_pair:
            bench_ret = (bench_pair[1] / bench_pair[0] - 1) * 100

        rows.append({
            "month": label,
            "signal_date": _fmt_date(signal_date),
            "entry_date": _fmt_date(entry_date),
            "exit_date": _fmt_date(exit_date),
            "market_state": market_state,
            "exposure": round(deployed_exposure * 100, 1),
            "return_pct": round(portfolio_ret, 2),
            "raw_return_pct": round(raw_return, 2),
            "benchmark_pct": round(bench_ret, 2),
            "top_names": ", ".join(
                f"{r['entry_type']}-{r['etf_name']}({r['score']['total']:.1f}"
                + (",试探" if _is_chase_high(r, market_state) else "")
                + ")"
                for r in selected
            ),
            "top_stages": ", ".join(r["stage"][0] for r in selected),
            "item_returns": ", ".join(
                f"{r['etf_name']} {r['period_return']:+.1f}%({r['exit_note']})"
                for r in selected
            ),
            "hold_reasons": "; ".join(
                f"{r['etf_name']}:{r['hold_reason']}"
                for r in selected
            ),
            "sold": "; ".join(sold_notes) if sold_notes else "—",
        })
        holdings = {
            r["etf_code"]: {
                "etf_code": r["etf_code"],
                "etf_name": r["etf_name"],
                "holding_since": r.get("holding_since") or r["real_entry"],
            }
            for r in selected
        }
        print(
            f"  {label}: signal={_fmt_date(signal_date)}, "
            f"state={market_state}, exposure={deployed_exposure:.0%}, "
            f"top={', '.join((r['entry_type'] + '-' + r['etf_name'][:8]) for r in selected)}, "
            f"ret={portfolio_ret:+.2f}% raw={raw_return:+.2f}%, bench={bench_ret:+.2f}%"
        )

    if not rows:
        empty = pd.DataFrame()
        empty.attrs["universe_size"] = len(etf_pool)
        empty.attrs["max_etfs"] = max_etfs
        empty.attrs["freq"] = freq
        empty.attrs["effective_min_amount_yi"] = effective_min_amount_yi
        empty.attrs["weekly_max_broad"] = WEEKLY_MAX_BROAD if freq == "weekly" else None
        return empty

    df = pd.DataFrame(rows)
    df["cumulative"] = (1 + df["return_pct"] / 100).cumprod()
    df["benchmark_cumulative"] = (1 + df["benchmark_pct"] / 100).cumprod()
    df.attrs["universe_size"] = len(etf_pool)
    df.attrs["max_etfs"] = max_etfs
    df.attrs["freq"] = freq
    df.attrs["effective_min_amount_yi"] = effective_min_amount_yi
    df.attrs["weekly_max_broad"] = WEEKLY_MAX_BROAD if freq == "weekly" else None
    return df


def write_report(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_n: int,
    min_amount_yi: float,
    freq: str,
) -> Path:
    out_name = "backtest_result_weekly.md" if freq == "weekly" else "backtest_result.md"
    out_path = PROJECT_ROOT / "codex" / "stock" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out_path.write_text("# ETF 三层框架回测\n\n无有效回测结果。\n", encoding="utf-8")
        return out_path

    period_label = "周度" if freq == "weekly" else "月度"
    annual_factor = 52 if freq == "weekly" else 12
    period_std = df["return_pct"].std(ddof=0)
    sharpe = 0.0 if period_std == 0 else (df["return_pct"].mean() / period_std) * (annual_factor ** 0.5)
    total_ret = (df["cumulative"].iloc[-1] - 1) * 100
    bench_ret = (df["benchmark_cumulative"].iloc[-1] - 1) * 100
    effective_min_amount_yi = df.attrs.get("effective_min_amount_yi", min_amount_yi)
    weekly_broad_rule = (
        f"- 周度组合最多保留 {df.attrs.get('weekly_max_broad', WEEKLY_MAX_BROAD)} 个宽基，优先把席位留给产业方向。"
        if freq == "weekly"
        else "- 月度组合不限制宽基数量，按趋势、阶段和成交额统一排序。"
    )
    hold_bonus_label = "周度" if freq == "weekly" else "月度"
    hold_bonus = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])

    lines = [
        f"# ETF 三层框架{period_label}回测 — {start_date} ~ {end_date}",
        "",
        "## 回测约束",
        "",
        f"- 调仓频率：{period_label}。",
        "- 信号日为调仓周期开始前最后一个交易日。",
        "- 选 ETF 只使用信号日及以前数据。",
        "- 周期初先判断旧持仓是否仍可续持，再让旧仓和新候选统一竞争 TOP 组合。",
        "- 续持只使用信号日及以前数据；不使用当月未来收益决定是否持有。",
        weekly_broad_rule,
        "- 周期收益从周期第一个交易日计算到周期最后一个交易日。",
        "- 周期内用日线 low 模拟止损，触发后按止损价退出。",
        "- 市场环境过滤会降低暴露比例：主升100%、震荡80%、退潮60%、退潮末期40%、冰点30%。",
        "- 不使用当月已实现收益排序，避免未来函数。",
        "- 商品/资源/LOF 使用更宽的趋势止损，避免强主升浪中被普通行业 ETF 阈值过早洗出。",
        f"- 追高硬门禁：单日 >{CHASE_SINGLE_DAY_LIMIT:g}% 或 5日 >{CHASE_5D_RETURN_LIMIT:g}% → 该腿按试探仓 {PROBE_WEIGHT*100:.0f}%。",
        f"- 最低入选分数：{MIN_ENTRY_SCORE:g}；分数不足宁少不凑，允许仓位低于满仓。",
        "",
        "## 参数",
        "",
        f"- 每期选取：TOP {top_n}",
        f"- 命令最低成交额：{min_amount_yi} 亿",
        f"- 实际最低成交额：{effective_min_amount_yi} 亿",
        f"- 可续持阶段：{', '.join(sorted(HOLDABLE_STAGES))}",
        f"- {hold_bonus_label}续持溢价：弱续持 +{hold_bonus['weak']:.0f}，正常续持 +{hold_bonus['normal']:.0f}，强续持 +{hold_bonus['strong']:.0f}",
        f"- 普通趋势止损：高点回撤 {abs(NORMAL_TRAIL_STOP):.0f}% 且跌破 MA20",
        f"- 商品/LOF 趋势止损：高点回撤 {abs(COMMODITY_TRAIL_STOP):.0f}% 且跌破 MA20",
        f"- 商品/LOF 抛物线止盈：20日涨幅≥{COMMODITY_PARABOLIC_R20:.0f}% 且60日涨幅≥{COMMODITY_PARABOLIC_R60:.0f}%",
        f"- 实际回测标的数：{df.attrs.get('universe_size', '未知')}",
        f"- `--max-etfs`：{df.attrs.get('max_etfs') if df.attrs.get('max_etfs') else '未启用，全量'}",
        "",
        "## 结果",
        "",
        f"- 累计收益：{total_ret:.2f}%",
        f"- 沪深300ETF 基准：{bench_ret:.2f}%",
        f"- 超额收益：{total_ret - bench_ret:.2f}%",
        f"- {period_label}均收益：{df['return_pct'].mean():+.2f}%",
        f"- {period_label}胜率：{(df['return_pct'] > 0).sum() / len(df) * 100:.1f}%",
        f"- {period_label}夏普：{sharpe:.2f}",
        f"- 最大回撤：{_max_drawdown(df['cumulative']):.2f}%",
        "",
        "## 明细",
        "",
        df.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ETF 三层分析框架无未来函数回测")
    parser.add_argument("--start", default="2025-01-01", help="起始日期")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="结束日期")
    parser.add_argument("--top", type=int, default=3, help="每期选股数")
    parser.add_argument("--min-amount-yi", type=float, default=0.2, help="最低成交额，单位亿元")
    parser.add_argument("--max-etfs", type=int, default=None, help="调试用：仅回测前 N 只 ETF/LOF")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="monthly", help="调仓频率")
    args = parser.parse_args()

    df = run_backtest(
        args.start,
        args.end,
        top_n=args.top,
        min_amount_yi=args.min_amount_yi,
        max_etfs=args.max_etfs,
        freq=args.freq,
    )
    out_path = write_report(df, args.start, args.end, args.top, args.min_amount_yi, args.freq)

    if df.empty:
        print("无有效回测结果")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("回测完成")
    print("=" * 60)
    print(f"累计收益: {(df['cumulative'].iloc[-1]-1)*100:.2f}%")
    print(f"基准收益: {(df['benchmark_cumulative'].iloc[-1]-1)*100:.2f}%")
    print(f"详细结果: {out_path}")


if __name__ == "__main__":
    main()
