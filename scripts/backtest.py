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
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    apply_scoring_context,
    calc_consecutive_strong,
    calc_etf_metrics,
    calc_relative_profile,
    classify_stage,
    fetch_etf_hist,
    load_etf_txt,
)
from daily_risk import EXTREME_DAILY_DROP_PCT, cluster_crash_calendar  # noqa: E402
from execution_model import cost_return_pct, simulate_long_with_stop  # noqa: E402
from market_state import breadth_ratio_on_date, classify_from_history  # noqa: E402
from risk_rules import (  # noqa: E402
    CORE_ENTRY_SCORE,
    MAX_NEW_POSITION_ONE_DAY_DOMINANCE,
    MAX_SAME_RISK_CLUSTER,
    TARGET_SELECTION_COUNT,
    allocate_instrument_weights,
    carry_mainline_floor,
    category_root,
    get_target_exposure,
    max_same_risk_cluster,
    max_selection_count,
    new_position_trend_gate,
    risk_cluster,
    score_layers_pass,
)
from universe_history import (  # noqa: E402
    active_codes as active_universe_codes,
    load_registry,
    registry_quality,
)
from strategy_version import load_current_manifest, verify_manifest  # noqa: E402

HOLDABLE_STAGES = set(NEW_MONEY_STAGES) | {"趋势回撤期", "加速期⚠"}
COMMODITY_CATEGORIES = {"商品", "资源"}
COMMODITY_TRAIL_STOP = -35.0
NORMAL_TRAIL_STOP = -18.0
COMMODITY_PARABOLIC_R20 = 80.0
COMMODITY_PARABOLIC_R60 = 150.0
ONE_WAY_COST_BPS = 8.0
STOP_SLIPPAGE_BPS = 10.0
# 追高硬门禁：符合任一条件时该腿仅按试探仓 10% 建仓
# 只在防守市场（退潮/退潮末期/冰点）触发；主升/震荡下真趋势不做机械稀释。
CHASE_5D_RETURN_LIMIT = 25.0
CHASE_SINGLE_DAY_LIMIT = 4.0
DEFENSIVE_STATES = {"退潮", "退潮末期", "冰点"}
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
WEEKLY_MAX_SAME_INDUSTRY = 2
HOLD_BONUS = {
    "weekly": {"weak": 6.0, "normal": 6.0, "strong": 6.0},
    "monthly": {"weak": 6.0, "normal": 6.0, "strong": 6.0},
}
MIN_HOLD_TRADING_DAYS = 10
MIN_HOLD_PRIORITY_BONUS = 100.0
BENCHMARKS = {
    "510300": "沪深300ETF",
    "512500": "中证500ETF",
    "159845": "中证1000ETF",
    "159915": "创业板ETF",
    "588080": "科创50ETF",
}


def _simple_market_state(bench_df, signal_date, data=None):
    """Compatibility wrapper using the same regime rules as live diagnostics."""
    width = breadth_ratio_on_date(data or {}, signal_date)
    state = classify_from_history(
        bench_df,
        signal_date,
        breadth_ratio=width,
    )
    return state, get_target_exposure(state) / 100


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


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    peak = cumulative.cummax().clip(lower=1.0)
    dd = cumulative / peak - 1
    return round(float(dd.min()) * 100, 2)


def _cost_sensitivity(
    df: pd.DataFrame,
    cost_grid_bps: tuple[float, ...] = (8.0, 15.0, 25.0),
    daily_equity: pd.DataFrame | None = None,
    base_cost_bps: float = ONE_WAY_COST_BPS,
) -> pd.DataFrame:
    """Reprice the same signals and fills under several transaction-cost tiers."""
    if df.empty:
        return pd.DataFrame()
    gross = df.get("gross_return_pct", df["return_pct"] + df["cost_pct"])
    rows = []
    for bps in cost_grid_bps:
        if daily_equity is not None and not daily_equity.empty and base_cost_bps > 0:
            repriced_nav: list[float] = []
            linked_nav = 1.0
            ratio = float(bps) / float(base_cost_bps)
            for _, period in daily_equity.groupby("period", sort=False):
                cumulative_stop_cost = 0.0
                entry_cost = float(period["entry_cost_pct"].sum()) * ratio / 100
                for _, day in period.iterrows():
                    cumulative_stop_cost += float(day["stop_cost_pct"]) * ratio / 100
                    factor = float(day["gross_factor"]) - entry_cost - cumulative_stop_cost
                    repriced_nav.append(linked_nav * factor)
                linked_nav = repriced_nav[-1]
            cumulative = pd.Series(repriced_nav)
            cumulative_return = (float(cumulative.iloc[-1]) - 1) * 100
            drawdown = _max_drawdown(cumulative)
        else:
            net_period = gross - df["turnover_pct"] / 100 * float(bps) / 100
            cumulative = (1 + net_period / 100).cumprod()
            cumulative_return = (float(cumulative.iloc[-1]) - 1) * 100
            drawdown = _max_drawdown(cumulative)
        rows.append({
            "单边成本(bp)": float(bps),
            "累计收益%": round(cumulative_return, 2),
            "最大日度回撤%": drawdown,
            "累计成本影响(百分点)": round(
                float((df["turnover_pct"] / 100 * float(bps) / 100).sum()),
                2,
            ),
        })
    return pd.DataFrame(rows)


def _benchmark_comparison(
    data: dict[str, pd.DataFrame],
    daily_equity: pd.DataFrame,
) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(daily_equity["date"])
    start = dates.min()
    end = dates.max()
    rows = []
    for code, name in BENCHMARKS.items():
        frame = data.get(code)
        if frame is None or frame.empty:
            continue
        period = frame[(frame["date"] >= start) & (frame["date"] <= end)].sort_values("date")
        if period.empty:
            continue
        entry = float(period.iloc[0].get("open", period.iloc[0]["close"]))
        nav = period["close"].astype(float) / entry
        rows.append({
            "代码": code,
            "基准": name,
            "累计收益%": round((float(nav.iloc[-1]) - 1) * 100, 2),
            "最大回撤%": _max_drawdown(nav),
            "起始日": _fmt_date(period.iloc[0]["date"]),
            "截止日": _fmt_date(period.iloc[-1]["date"]),
        })

    daily_returns: dict[pd.Timestamp, list[float]] = {}
    target_dates = set(dates.dt.normalize())
    for frame in data.values():
        known = frame[frame["date"] <= end].sort_values("date").copy()
        known["return"] = known["close"].pct_change()
        for _, row in known[known["date"].dt.normalize().isin(target_dates)].iterrows():
            value = row["return"]
            if pd.isna(value):
                continue
            daily_returns.setdefault(pd.to_datetime(row["date"]).normalize(), []).append(float(value))
    equal_weight_returns = pd.Series(
        {
            date: sum(values) / len(values)
            for date, values in daily_returns.items()
            if values
        }
    ).sort_index()
    if not equal_weight_returns.empty:
        equal_weight_nav = (1 + equal_weight_returns).cumprod()
        rows.append({
            "代码": "EW_POOL",
            "基准": "当前固定ETF池等权（日频再平衡）",
            "累计收益%": round((float(equal_weight_nav.iloc[-1]) - 1) * 100, 2),
            "最大回撤%": _max_drawdown(equal_weight_nav),
            "起始日": _fmt_date(equal_weight_nav.index[0]),
            "截止日": _fmt_date(equal_weight_nav.index[-1]),
        })
    return pd.DataFrame(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_loss_row(row: pd.Series) -> tuple[str, str]:
    """Use observable price outcomes to separate market, selection, and mixed losses."""
    strategy_ret = float(row.get("return_pct") or 0)
    benchmark_ret = float(row.get("benchmark_pct") or 0)
    excess_ret = strategy_ret - benchmark_ret

    if benchmark_ret <= -0.5 and excess_ret >= 0:
        conclusion = "市场下跌为主"
        responsibility = "框架非主要责任，组合仍跑赢基准"
    elif benchmark_ret <= -1.0 and excess_ret < 0:
        conclusion = "市场冲击与组合暴露共同作用"
        responsibility = "框架有部分责任，应检查行业集中和仓位"
    elif excess_ret <= -1.0:
        conclusion = "选股或行业暴露为主"
        responsibility = "框架主要责任，不能归因于大盘"
    else:
        conclusion = "组合内部波动为主"
        responsibility = "框架需复核续持、风险暴露和交易成本"

    evidence = str(row.get("risk_flags") or "").strip()
    if not evidence or evidence == "—":
        evidence = "无额外机械风险标记"
    return conclusion, f"{responsibility}；{evidence}"


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


def _calc_atr(df: pd.DataFrame, as_of, period: int = 20) -> float | None:
    """计算 ATR (Average True Range)，用于动态止损宽度。"""
    df2 = df[df["date"] <= pd.to_datetime(as_of)].copy()
    df2["prev_close"] = df2["close"].shift(1)
    df2["tr1"] = df2["high"] - df2["low"]
    df2["tr2"] = abs(df2["high"] - df2["prev_close"])
    df2["tr3"] = abs(df2["low"] - df2["prev_close"])
    df2["tr"] = df2[["tr1", "tr2", "tr3"]].max(axis=1)
    atr_series = df2["tr"].tail(period)
    if len(atr_series) < 5:
        return None
    return float(atr_series.mean())


def _simulate_period_path(
    df: pd.DataFrame,
    start,
    end,
    item: dict,
    freq: str,
    carry_price: float | None = None,
    stop_slippage_bps: float = STOP_SLIPPAGE_BPS,
    cluster_alerts: dict[pd.Timestamp, set[str]] | None = None,
):
    """
    用日线 OHLC 模拟周期内止损并返回逐日路径。

    新仓按周期第一个交易日开盘价买入；续持仓以上一期收盘标记价衔接，
    因而不会漏掉周末/节假日跳空。第 t 日止损只使用 t-1 日以前的
    峰值；跳空跌破止损时按开盘价并扣除滑点成交。
    """
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
    if len(rows) < 2:
        return None

    rows = rows.sort_values("date").reset_index(drop=True)
    first = rows.iloc[0]
    entry_price = (
        float(carry_price)
        if carry_price is not None and carry_price > 0
        else float(first.get("open", first["close"]))
    )
    if entry_price <= 0:
        return None

    # ATR 只能使用入场日前数据，禁止回测终点数据污染历史止损。
    atr = _calc_atr(df, pd.to_datetime(start) - timedelta(days=1), 20)
    atr_pct = atr / entry_price if entry_price > 0 and atr else 0
    rules = PERIOD_STOP_RULES[freq]
    if _is_commodity_like(item):
        initial_stop_pct = max(rules["commodity_initial"], ATR_MULTIPLIER * atr_pct)
        trailing_stop_pct = max(rules["commodity_trailing"], ATR_MULTIPLIER * atr_pct * 1.3)
    else:
        initial_stop_pct = max(rules["normal_initial"], ATR_MULTIPLIER * atr_pct)
        trailing_stop_pct = max(rules["normal_trailing"], ATR_MULTIPLIER * atr_pct * 1.3)

    forced_exit_dates: set[pd.Timestamp] = set()
    item_cluster = risk_cluster(
        item.get("category", "其他"),
        item.get("industry", "其他"),
        item.get("etf_name", ""),
    )
    day_returns: list[float] = []
    prior_close = entry_price
    for _, row in rows.iterrows():
        close = float(row["close"])
        day_returns.append((close / prior_close - 1) * 100)
        prior_close = close
    for exit_index in range(1, len(rows)):
        trigger_index = exit_index - 1
        trigger_date = pd.to_datetime(rows.iloc[trigger_index]["date"])
        crashed_cluster = item_cluster in (cluster_alerts or {}).get(trigger_date, set())
        if day_returns[trigger_index] <= EXTREME_DAILY_DROP_PCT or crashed_cluster:
            forced_exit_dates.add(pd.to_datetime(rows.iloc[exit_index]["date"]))

    return simulate_long_with_stop(
        rows,
        entry_price=entry_price,
        initial_stop_pct=initial_stop_pct,
        trailing_stop_pct=trailing_stop_pct,
        slippage_bps=stop_slippage_bps,
        forced_exit_dates=forced_exit_dates,
    )


def _simulate_period_return(
    df: pd.DataFrame,
    start,
    end,
    item: dict,
    freq: str,
    carry_price: float | None = None,
    stop_slippage_bps: float = STOP_SLIPPAGE_BPS,
    cluster_alerts: dict[pd.Timestamp, set[str]] | None = None,
) -> tuple[float, str, str, str]:
    """Compatibility wrapper around the conservative daily path simulator."""
    result = _simulate_period_path(
        df,
        start,
        end,
        item,
        freq,
        carry_price=carry_price,
        stop_slippage_bps=stop_slippage_bps,
        cluster_alerts=cluster_alerts,
    )
    if result is None:
        return 0.0, "", "", "数据不足"
    return result.return_pct, result.entry_date, result.exit_date, result.exit_note


def _build_daily_period_path(
    *,
    label: str,
    benchmark_rows: pd.DataFrame,
    benchmark_entry: float,
    selected: list[dict],
    weights: list[float],
    entry_turnover: float,
    one_way_cost_bps: float,
    nav_start: float,
    benchmark_nav_start: float,
    market_state: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Aggregate leg paths into reproducible daily equity and position records."""
    calendar = [
        pd.to_datetime(value)
        for value in benchmark_rows.sort_values("date")["date"].tolist()
    ]
    leg_paths: dict[str, pd.DataFrame] = {}
    for item in selected:
        path = item["simulation"].path.copy()
        path["date"] = pd.to_datetime(path["date"])
        leg_paths[item["etf_code"]] = path.set_index("date")

    entry_cost_fraction = cost_return_pct(entry_turnover, one_way_cost_bps) / 100
    cumulative_stop_cost_fraction = 0.0
    daily_rows: list[dict] = []
    position_rows: list[dict] = []
    stop_trades: list[dict] = []

    for day_index, date in enumerate(calendar):
        gross_factor = 1.0
        active_value = 0.0
        active_names: list[str] = []
        stop_cost_today = 0.0
        leg_snapshots: list[tuple[dict, float, pd.Series]] = []

        for item, weight in zip(selected, weights):
            path = leg_paths[item["etf_code"]]
            if date not in path.index:
                prior = path[path.index <= date]
                if prior.empty:
                    continue
                leg_row = prior.iloc[-1]
            else:
                leg_row = path.loc[date]
                if isinstance(leg_row, pd.DataFrame):
                    leg_row = leg_row.iloc[-1]

            factor = float(leg_row["factor"])
            gross_factor += float(weight) * (factor - 1)
            is_active = bool(leg_row["active"])
            if is_active:
                active_value += float(weight) * factor
                active_names.append(item["etf_name"])
            if pd.notna(leg_row.get("stop_fill")):
                stop_cost_today += cost_return_pct(weight, one_way_cost_bps) / 100
                stop_trades.append({
                    "date": _fmt_date(date),
                    "period": label,
                    "code": item["etf_code"],
                    "name": item["etf_name"],
                    "side": "STOP_SELL",
                    "weight_change_pct": round(float(weight) * 100, 2),
                    "execution_price": round(float(leg_row["stop_fill"]), 6),
                    "cost_bps": float(one_way_cost_bps),
                    "reason": str(leg_row.get("stop_reason") or "止损"),
                })
            leg_snapshots.append((item, float(weight), leg_row))

        cumulative_stop_cost_fraction += stop_cost_today
        net_factor = gross_factor - entry_cost_fraction - cumulative_stop_cost_fraction
        nav = nav_start * net_factor
        benchmark_row = benchmark_rows[
            pd.to_datetime(benchmark_rows["date"]) == date
        ].iloc[-1]
        benchmark_factor = float(benchmark_row["close"]) / float(benchmark_entry)
        benchmark_nav = benchmark_nav_start * benchmark_factor
        cash_value = max(net_factor - active_value, 0.0)
        cash_weight = 100.0 if net_factor <= 0 else cash_value / net_factor * 100

        daily_rows.append({
            "date": _fmt_date(date),
            "period": label,
            "market_state": market_state,
            "nav": nav,
            "benchmark_nav": benchmark_nav,
            "gross_factor": gross_factor,
            "entry_cost_pct": entry_cost_fraction * 100 if day_index == 0 else 0.0,
            "stop_cost_pct": stop_cost_today * 100,
            "cash_weight_pct": cash_weight,
            "holdings": ", ".join(active_names) if active_names else "现金",
        })
        for item, weight, leg_row in leg_snapshots:
            if not bool(leg_row["active"]):
                continue
            position_rows.append({
                "date": _fmt_date(date),
                "period": label,
                "code": item["etf_code"],
                "name": item["etf_name"],
                "target_weight_pct": round(weight * 100, 2),
                "mark_price": round(float(leg_row["mark_price"]), 6),
                "market_value_factor": round(weight * float(leg_row["factor"]), 8),
            })

    return daily_rows, position_rows, stop_trades


def _finalize_daily_equity(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    daily = pd.DataFrame(records)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    daily["daily_return_pct"] = daily["nav"].pct_change().fillna(daily["nav"] - 1) * 100
    daily["drawdown_pct"] = (
        daily["nav"] / daily["nav"].cummax().clip(lower=1.0) - 1
    ) * 100
    daily["benchmark_daily_return_pct"] = (
        daily["benchmark_nav"].pct_change().fillna(daily["benchmark_nav"] - 1) * 100
    )
    return daily


def _build_cash_daily_path(
    *,
    label: str,
    benchmark_rows: pd.DataFrame,
    benchmark_entry: float,
    nav_start: float,
    benchmark_nav_start: float,
    exit_turnover: float,
    one_way_cost_bps: float,
    market_state: str,
) -> list[dict]:
    cost_fraction = cost_return_pct(exit_turnover, one_way_cost_bps) / 100
    records = []
    for index, (_, row) in enumerate(benchmark_rows.sort_values("date").iterrows()):
        benchmark_factor = float(row["close"]) / float(benchmark_entry)
        records.append({
            "date": _fmt_date(row["date"]),
            "period": label,
            "market_state": market_state,
            "nav": nav_start * (1 - cost_fraction),
            "benchmark_nav": benchmark_nav_start * benchmark_factor,
            "gross_factor": 1.0,
            "entry_cost_pct": cost_fraction * 100 if index == 0 else 0.0,
            "stop_cost_pct": 0.0,
            "cash_weight_pct": 100.0,
            "holdings": "现金",
        })
    return records


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
    major_category = category_root(item.get("category", ""))
    return product_type == "LOF" or major_category in COMMODITY_CATEGORIES


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
    market_state: str = "未知",
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
    current_score = float(current.get("score", {}).get("total") or 0)
    peak_score = max(float(previous.get("peak_score") or current_score), current_score)
    score_drop = 1 - current_score / peak_score if peak_score > 0 else 0.0
    rank = int(current.get("_rank") or 999)
    score_detail = current.get("score", {})
    mainline_score = float(score_detail.get("mainline") or 0)
    risk_penalty = float(score_detail.get("risk_penalty") or 0)

    if "mainline" in score_detail:
        carry_floor = carry_mainline_floor(market_state)
        if mainline_score < carry_floor:
            return False, f"主线分{mainline_score:.1f}跌破续持门槛{carry_floor:.1f}", current
    if mainline_score < 45 and ret20 < 0:
        return False, f"主线分降至{mainline_score:.1f}且20日趋势转负", current
    if risk_penalty > 30:
        return False, f"风险惩罚升至{risk_penalty:.1f}", current

    if stage in HOLDABLE_STAGES:
        return True, f"{stage}续持", current
    if _is_commodity_like(current) and ret20 > 20 and ret60 > 40:
        return True, "商品大趋势续持", current
    if rank <= 30 and score_drop < 0.40 and ret20 >= 0:
        return True, f"排名{rank}且20日趋势未负，延续持有", current
    if rank > 30 and score_drop >= 0.40:
        return False, f"排名{rank}且评分较峰值下降{score_drop:.0%}", current
    return False, f"{stage}不再持有", current


def _trading_days_held(
    frame: pd.DataFrame,
    holding_since,
    signal_date,
) -> int:
    """Count known trading days since entry without a calendar-day shortcut."""
    if frame.empty or not holding_since:
        return 0
    start = pd.to_datetime(holding_since)
    end = pd.to_datetime(signal_date)
    return int(
        frame.loc[
            (frame["date"] >= start) & (frame["date"] <= end),
            "date",
        ].nunique()
    )


def _new_position_allowed(item: dict, market_state: str) -> bool:
    """新仓必须分别通过主线、买点和风险层，不因席位空缺而降级补位。"""
    if _is_chase_high(item, market_state):
        return False

    metrics = item.get("metrics", {})
    score = float(item.get("score", {}).get("total") or 0)
    layers_ok, _ = score_layers_pass(item.get("score", {}), market_state)
    if not layers_ok:
        return False
    stage = item.get("stage", ("", "", ""))[0]
    ret5 = float(metrics.get("ret_5d") or 0)
    ret20 = float(metrics.get("ret_20d") or 0)
    ret60 = float(metrics.get("ret_60d") or 0)
    trend_ok, _ = new_position_trend_gate(
        category=item.get("category", "其他"),
        market_state=market_state,
        score=score,
        ret_20d=ret20,
        ret_60d=ret60,
    )
    if not trend_ok:
        return False

    if market_state in {"退潮末期", "冰点"}:
        if stage not in {"扩散期", "加速期", "确认期"}:
            return False
        return ret5 > 0

    if market_state == "退潮":
        if stage not in {"扩散期", "加速期", "确认期"}:
            return False
        if item.get("industry") == "周期" or item.get("category") in COMMODITY_CATEGORIES:
            return ret5 > 0 and ret20 > 0
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
    if item.get("entry_tier") == "核心":
        score += 3.0
    if item.get("entry_type") == "续持":
        bonus_table = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])
        score += bonus_table["normal"]
        if freq == "weekly" and item.get("minimum_hold_protected"):
            score += MIN_HOLD_PRIORITY_BONUS
    stage = item.get("stage", ("", "", ""))[0]
    return (
        score,
        STAGE_PRIORITY.get(stage, -99),
        float(metrics.get("ret_20d") or 0),
        float(metrics.get("ret_5d") or 0),
        float(metrics.get("amount_yi") or 0),
    )


def _select_top_candidates(
    candidate_pool: list[dict],
    top_n: int,
    freq: str,
    market_state: str = "未知",
) -> list[dict]:
    """
    从旧仓和新候选中统一选 TOP。
    - 新仓已在上游分别通过主线、买点和风险门禁。
    - 周度更强调主线弹性：最多保留 1 个宽基，避免宽基挤占产业 ETF 名额。
    """
    top_n = min(top_n, max_selection_count(market_state))
    ordered = sorted(candidate_pool, key=lambda r: _selection_priority(r, freq), reverse=True)
    if freq != "weekly":
        return ordered[:top_n]

    selected = []
    delayed_broad = []
    broad_count = 0
    industry_count: dict[str, int] = {}
    cluster_count: dict[str, int] = {}
    cluster_limit = max_same_risk_cluster(market_state)
    for r in ordered:
        if len(selected) >= top_n:
            break
        industry = r.get("industry", "其他")
        cluster = risk_cluster(
            r.get("category", "其他"),
            industry,
            r.get("etf_name", ""),
        )
        if industry == "宽基" and broad_count >= WEEKLY_MAX_BROAD:
            delayed_broad.append(r)
            continue
        if industry != "宽基" and industry_count.get(industry, 0) >= WEEKLY_MAX_SAME_INDUSTRY:
            continue
        if cluster_count.get(cluster, 0) >= cluster_limit:
            continue
        selected.append(r)
        if industry == "宽基":
            broad_count += 1
        industry_count[industry] = industry_count.get(industry, 0) + 1
        cluster_count[cluster] = cluster_count.get(cluster, 0) + 1

    if len(selected) < top_n:
        for r in delayed_broad:
            if len(selected) >= top_n:
                break
            cluster = risk_cluster(
                r.get("category", "其他"),
                r.get("industry", "其他"),
                r.get("etf_name", ""),
            )
            if cluster_count.get(cluster, 0) >= cluster_limit:
                continue
            selected.append(r)
            cluster_count[cluster] = cluster_count.get(cluster, 0) + 1
    return selected


def _fetch_hist_cached(
    code: str,
    start_date: str,
    end_date: str,
    product_type: str,
    refresh_cache: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Use the shared per-instrument cache for every backtest request."""
    frame = fetch_etf_hist(
        code,
        start_date,
        end_date,
        str(product_type or "ETF").upper(),
        refresh_cache=refresh_cache,
        offline=offline,
    )
    if offline and (
        not frame.attrs.get("coverage_start_ok", False)
        or not frame.attrs.get("coverage_end_ok", False)
    ):
        raise RuntimeError(
            f"offline cache coverage incomplete for {code}: "
            f"{start_date}..{end_date}"
        )
    return frame


def fetch_all_hist(
    etf_pool: dict,
    start_date: str,
    end_date: str,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict[str, pd.DataFrame]:
    data = {}
    total = len(etf_pool)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(
                _fetch_hist_cached,
                cfg["code"],
                start_date,
                end_date,
                cfg.get("type", "ETF"),
                refresh_cache,
                offline,
            ): cfg["code"]
            for cfg in etf_pool.values()
        }
        for i, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                df = future.result()
            except Exception as exc:
                print(f"  {code} 行情获取失败: {exc}", flush=True)
                continue
            if not df.empty:
                data[code] = df
            if i % 20 == 0 or i == total:
                print(f"  行情已获取 {i}/{total}", flush=True)
    if offline:
        expected_codes = {cfg["code"] for cfg in etf_pool.values()}
        missing_codes = sorted(expected_codes - set(data))
        if missing_codes:
            preview = ", ".join(missing_codes[:10])
            suffix = "..." if len(missing_codes) > 10 else ""
            raise RuntimeError(
                f"offline cache is incomplete for {len(missing_codes)} instruments: "
                f"{preview}{suffix}"
            )
    return data


def rank_on_signal_date(
    etf_pool: dict,
    data: dict[str, pd.DataFrame],
    bench_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    min_amount_yi: float,
    allowed_stages: set[str] | None = NEW_MONEY_STAGES,
    allowed_codes: set[str] | None = None,
) -> list[dict]:
    signal_date_str = _fmt_date(signal_date)
    bench_metrics = calc_etf_metrics(bench_df, signal_date_str)
    bench_pct = bench_metrics.get("pct_chg", 0.0) if "error" not in bench_metrics else 0.0

    ranked = []
    for direction, cfg in etf_pool.items():
        if allowed_codes is not None and cfg["code"] not in allowed_codes:
            continue
        if category_root(cfg.get("cat", "")) in {"货币", "债券"}:
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
        metrics.update(
            calc_relative_profile(df, bench_df, _fmt_date(effective_signal))
        )
        if float(metrics.get("amount_yi") or 0) < min_amount_yi:
            continue

        strong_days = calc_consecutive_strong(df, bench_df, _fmt_date(effective_signal))
        stage = classify_stage(metrics, strong_days)
        if allowed_stages is not None and stage[0] not in allowed_stages:
            continue

        ranked.append({
            "direction": direction,
            "etf_code": cfg["code"],
            "etf_name": cfg["name"],
            "category": cfg["cat"],
            "industry": cfg["ind"],
            "product_type": cfg.get("type", "ETF"),
            "is_qdii": cfg.get("is_qdii", False),
            "premium_pct": cfg.get("premium_pct"),
            "metrics": metrics,
            "stage": stage,
            "strong_days": strong_days,
            "score": {},
            "signal_date": _fmt_date(effective_signal),
        })

    apply_scoring_context(ranked, bench_pct)
    return sorted(ranked, key=_rank_key, reverse=True)


def run_backtest(
    start_date: str,
    end_date: str,
    top_n: int = 3,
    min_amount_yi: float = 0.2,
    max_etfs: int | None = None,
    freq: str = "monthly",
    refresh_cache: bool = False,
    offline: bool = False,
    one_way_cost_bps: float = ONE_WAY_COST_BPS,
    stop_slippage_bps: float = STOP_SLIPPAGE_BPS,
    min_hold_trading_days: int = MIN_HOLD_TRADING_DAYS,
) -> pd.DataFrame:
    if top_n < 1 or top_n > TARGET_SELECTION_COUNT:
        raise ValueError(f"top_n 必须在 1-{TARGET_SELECTION_COUNT} 之间，实际为 {top_n}")
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
    bench_df = _fetch_hist_cached(
        BENCHMARK_CODE,
        fetch_start,
        fetch_end,
        bench_cfg.get("type", "ETF"),
        refresh_cache,
        offline,
    )
    if bench_df.empty:
        raise RuntimeError("沪深300ETF 基准行情获取失败")

    data = fetch_all_hist(
        etf_pool,
        fetch_start,
        fetch_end,
        refresh_cache,
        offline,
    )
    cluster_alerts = cluster_crash_calendar(etf_pool, data, start_dt, end_dt)
    universe_registry = load_registry()
    universe_status = registry_quality(universe_registry)
    periods = _periods(start_dt, end_dt, freq)
    rows = []
    holdings: dict[str, dict] = {}
    benchmark_mark_price: float | None = None
    portfolio_nav = 1.0
    benchmark_nav = 1.0
    daily_records: list[dict] = []
    position_records: list[dict] = []
    trade_records: list[dict] = []

    for label, period_start, period_end in periods:
        signal_date = _latest_before(bench_df, period_start)
        entry_date = _first_between(bench_df, period_start, period_end)
        exit_date = _last_between(bench_df, period_start, period_end)
        if signal_date is None or entry_date is None or exit_date is None or entry_date >= exit_date:
            continue
        market_state, _ = _simple_market_state(bench_df, signal_date, data)
        benchmark_rows = bench_df[
            (bench_df["date"] >= entry_date) & (bench_df["date"] <= exit_date)
        ].sort_values("date")
        if benchmark_rows.empty:
            continue
        benchmark_entry = (
            benchmark_mark_price
            if benchmark_mark_price is not None
            else float(benchmark_rows.iloc[0].get("open", benchmark_rows.iloc[0]["close"]))
        )
        benchmark_exit = float(benchmark_rows.iloc[-1]["close"])
        bench_ret = (benchmark_exit / benchmark_entry - 1) * 100
        benchmark_mark_price = benchmark_exit

        ranked_all = rank_on_signal_date(
            etf_pool,
            data,
            bench_df,
            signal_date,
            effective_min_amount_yi,
            allowed_stages=None,
            allowed_codes=(
                active_universe_codes(universe_registry, signal_date)
                if not universe_registry.empty
                else None
            ),
        )
        if not ranked_all:
            print(f"  {label}: 无候选标的")
            exit_turnover = sum(float(item.get("weight", 0.0)) for item in holdings.values())
            exit_cost_pct = cost_return_pct(exit_turnover, one_way_cost_bps)
            daily_records.extend(_build_cash_daily_path(
                label=label,
                benchmark_rows=benchmark_rows,
                benchmark_entry=benchmark_entry,
                nav_start=portfolio_nav,
                benchmark_nav_start=benchmark_nav,
                exit_turnover=exit_turnover,
                one_way_cost_bps=one_way_cost_bps,
                market_state=market_state,
            ))
            portfolio_nav *= 1 - exit_cost_pct / 100
            benchmark_nav *= 1 + bench_ret / 100
            rows.append({
                "month": label,
                "signal_date": _fmt_date(signal_date),
                "entry_date": _fmt_date(entry_date),
                "exit_date": _fmt_date(exit_date),
                "market_state": market_state,
                "exposure": 0.0,
                "selection_count": 0,
                "turnover_pct": round(exit_turnover * 100, 1),
                "cost_pct": round(exit_cost_pct, 3),
                "return_pct": round(-exit_cost_pct, 2),
                "raw_return_pct": 0.0,
                "benchmark_pct": round(bench_ret, 2),
                "top_names": "现金",
                "top_stages": "—",
                "item_returns": "—",
                "loss_drivers": "—",
                "risk_flags": "无有效候选",
                "hold_reasons": "信号日无有效排名",
                "sold": "全部退出",
            })
            holdings = {}
            continue

        ranked_by_code = {r["etf_code"]: r for r in ranked_all}
        for rank, ranked_item in enumerate(ranked_all, 1):
            ranked_item["_rank"] = rank
        new_ranked = [
            r for r in ranked_all
            if r["stage"][0] in NEW_MONEY_STAGES
            and _new_position_allowed(r, market_state)
        ]
        carry_candidates = []
        sold_notes = []
        for previous in holdings.values():
            keep, reason, current = _should_hold_position(
                previous,
                ranked_by_code,
                data,
                signal_date,
                market_state,
            )
            if keep and current is not None:
                held_days = _trading_days_held(
                    data[current["etf_code"]],
                    previous.get("holding_since"),
                    signal_date,
                )
                minimum_hold_protected = (
                    freq == "weekly"
                    and held_days < min_hold_trading_days
                )
                carry_candidates.append({
                    **current,
                    "entry_type": "续持",
                    "entry_tier": "持有",
                    "hold_reason": (
                        f"{reason}；最短持有期保护{held_days}/{min_hold_trading_days}日"
                        if minimum_hold_protected
                        else reason
                    ),
                    "holding_since": previous.get("holding_since") or current.get("signal_date"),
                    "carry_price": previous.get("mark_price"),
                    "peak_score": max(
                        float(previous.get("peak_score") or 0),
                        float(current.get("score", {}).get("total") or 0),
                    ),
                    "minimum_hold_protected": minimum_hold_protected,
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
                "entry_tier": "核心",
                "hold_reason": "当期新排名入选",
                "holding_since": _fmt_date(entry_date),
                "carry_price": None,
                "peak_score": float(r.get("score", {}).get("total") or 0),
            })
            selected_codes.add(r["etf_code"])

        selected = _select_top_candidates(
            candidate_pool,
            top_n,
            freq,
            market_state,
        )
        state_max = max_selection_count(market_state)
        if len(selected) < state_max:
            print(
                f"  {label}: 三层门禁和分散约束后选择 {len(selected)}/{state_max} 只，"
                "剩余风险预算保留现金",
                flush=True,
            )
        selected_codes = {r["etf_code"] for r in selected}
        for r in carry_candidates:
            if r["etf_code"] not in selected_codes:
                sold_notes.append(f"{r['etf_name']}(被更强方向替换)")

        returns = []
        realized_selected = []
        for r in selected:
            simulation = _simulate_period_path(
                data[r["etf_code"]],
                entry_date,
                exit_date,
                r,
                freq,
                carry_price=r.get("carry_price"),
                stop_slippage_bps=stop_slippage_bps,
                cluster_alerts=cluster_alerts,
            )
            if simulation is None:
                continue
            realized_selected.append({
                **r,
                "period_return": simulation.return_pct,
                "real_entry": simulation.entry_date,
                "real_exit": simulation.exit_date,
                "exit_note": simulation.exit_note,
                "simulation": simulation,
            })
            returns.append(simulation.return_pct)

        if not realized_selected:
            exit_turnover = sum(float(item.get("weight", 0.0)) for item in holdings.values())
            exit_cost_pct = cost_return_pct(exit_turnover, one_way_cost_bps)
            daily_records.extend(_build_cash_daily_path(
                label=label,
                benchmark_rows=benchmark_rows,
                benchmark_entry=benchmark_entry,
                nav_start=portfolio_nav,
                benchmark_nav_start=benchmark_nav,
                exit_turnover=exit_turnover,
                one_way_cost_bps=one_way_cost_bps,
                market_state=market_state,
            ))
            portfolio_nav *= 1 - exit_cost_pct / 100
            benchmark_nav *= 1 + bench_ret / 100
            rows.append({
                "month": label,
                "signal_date": _fmt_date(signal_date),
                "entry_date": _fmt_date(entry_date),
                "exit_date": _fmt_date(exit_date),
                "market_state": market_state,
                "exposure": 0.0,
                "selection_count": 0,
                "turnover_pct": round(exit_turnover * 100, 1),
                "cost_pct": round(exit_cost_pct, 3),
                "return_pct": round(-exit_cost_pct, 2),
                "raw_return_pct": 0.0,
                "benchmark_pct": round(bench_ret, 2),
                "top_names": "现金",
                "top_stages": "—",
                "item_returns": "—",
                "loss_drivers": "—",
                "risk_flags": "市场过滤后无可执行标的",
                "hold_reasons": "市场过滤，无可执行标的",
                "sold": "; ".join(sold_notes) if sold_notes else "—",
            })
            print(f"  {label}: state={market_state}, exposure=0%, 无可持有标的")
            holdings = {}
            continue

        selected = realized_selected
        # 排名比例按实际通过门禁的标的数量归一化后分配风险预算。
        # QDII/LOF 继续受产品上限约束，所有未使用预算均保留现金。
        n_legs = len(selected)
        rank_weights_pct = allocate_instrument_weights(market_state, selected)
        weights = [weight / 100 for weight in rank_weights_pct]

        weighted_sum = 0.0
        for r, w in zip(selected, weights):
            weighted_sum += w * r["period_return"]
        deployed_exposure = sum(weights)

        old_weights = {
            code: float(item.get("weight", 0.0))
            for code, item in holdings.items()
        }
        new_weights = {
            r["etf_code"]: weight
            for r, weight in zip(selected, weights)
        }
        entry_turnover = sum(
            abs(new_weights.get(code, 0.0) - old_weights.get(code, 0.0))
            for code in set(old_weights) | set(new_weights)
        )
        stopped_codes = {
            r["etf_code"]
            for r in selected
            if str(r.get("exit_note", "")).startswith("止损")
        }
        stop_turnover = sum(
            new_weights.get(code, 0.0)
            for code in stopped_codes
        )
        total_turnover = entry_turnover + stop_turnover
        cost_drag_pct = cost_return_pct(total_turnover, one_way_cost_bps)
        raw_return = sum(r["period_return"] for r in selected) / n_legs if n_legs else 0.0
        gross_return = weighted_sum
        portfolio_ret = gross_return - cost_drag_pct
        period_daily, period_positions, stop_trades = _build_daily_period_path(
            label=label,
            benchmark_rows=benchmark_rows,
            benchmark_entry=benchmark_entry,
            selected=selected,
            weights=weights,
            entry_turnover=entry_turnover,
            one_way_cost_bps=one_way_cost_bps,
            nav_start=portfolio_nav,
            benchmark_nav_start=benchmark_nav,
            market_state=market_state,
        )
        if period_daily:
            daily_records.extend(period_daily)
            position_records.extend(period_positions)
            trade_records.extend(stop_trades)
            period_end_nav = float(period_daily[-1]["nav"])
            portfolio_ret = (period_end_nav / portfolio_nav - 1) * 100
            portfolio_nav = period_end_nav
            benchmark_nav = float(period_daily[-1]["benchmark_nav"])

        for code in sorted(set(old_weights) | set(new_weights)):
            delta = new_weights.get(code, 0.0) - old_weights.get(code, 0.0)
            if abs(delta) < 1e-12:
                continue
            item = next(
                (candidate for candidate in selected if candidate["etf_code"] == code),
                holdings.get(code, {}),
            )
            simulation = item.get("simulation")
            if simulation is not None and delta > 0:
                execution_price = float(simulation.entry_price)
            elif delta < 0 and code in data:
                execution_row = _last_row_on_or_before(data[code], entry_date)
                execution_price = (
                    float(execution_row["open"])
                    if execution_row is not None
                    else float(item.get("mark_price") or 0.0)
                )
            else:
                execution_price = float(item.get("mark_price") or 0.0)
            trade_records.append({
                "date": _fmt_date(entry_date),
                "period": label,
                "code": code,
                "name": item.get("etf_name", code),
                "side": "BUY" if delta > 0 else "SELL",
                "weight_change_pct": round(abs(delta) * 100, 2),
                "execution_price": round(execution_price, 6),
                "cost_bps": float(one_way_cost_bps),
                "reason": "周度/周期调仓",
            })
        contributions = sorted(
            (
                (
                    r["etf_name"],
                    float(r["period_return"]),
                    float(weight) * float(r["period_return"]),
                )
                for r, weight in zip(selected, weights)
            ),
            key=lambda item: item[2],
        )
        negative_drivers = [item for item in contributions if item[2] < 0]
        loss_drivers = "；".join(
            f"{name} {item_ret:+.1f}%（组合贡献{contribution:+.2f}个百分点）"
            for name, item_ret, contribution in negative_drivers[:2]
        ) or "无负贡献标的"

        risk_flags = []
        if len(stopped_codes) >= 2:
            risk_flags.append(f"{len(stopped_codes)}只标的同周止损")
        category_counts: dict[str, int] = {}
        cluster_counts: dict[str, int] = {}
        for r in selected:
            category = str(r.get("category") or "其他")
            category_counts[category] = category_counts.get(category, 0) + 1
            cluster = risk_cluster(
                category,
                r.get("industry", "其他"),
                r.get("etf_name", ""),
            )
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        concentrated = [
            f"{category}类{count}只"
            for category, count in category_counts.items()
            if count >= 2
        ]
        risk_flags.extend(concentrated)
        risk_flags.extend(
            f"{cluster}风险簇{count}只"
            for cluster, count in cluster_counts.items()
            if count >= 2
        )
        high_momentum_count = sum(
            1
            for r in selected
            if float(r.get("metrics", {}).get("ret_60d") or 0) >= 50
        )
        if high_momentum_count >= 2:
            risk_flags.append(f"{high_momentum_count}只标的60日涨幅不低于50%")
        negative_medium_trend = sum(
            1
            for r in selected
            if r.get("entry_type") != "续持"
            and float(r.get("metrics", {}).get("ret_60d") or 0) < 0
        )
        if negative_medium_trend:
            risk_flags.append(f"{negative_medium_trend}只新仓60日趋势为负")
        if total_turnover >= 1.5:
            risk_flags.append(f"单边换手{total_turnover * 100:.0f}%")

        rows.append({
            "month": label,
            "signal_date": _fmt_date(signal_date),
            "entry_date": _fmt_date(entry_date),
            "exit_date": _fmt_date(exit_date),
            "market_state": market_state,
            "exposure": round(deployed_exposure * 100, 1),
            "selection_count": n_legs,
            "turnover_pct": round(total_turnover * 100, 1),
            "cost_pct": round(cost_drag_pct, 3),
            "return_pct": round(portfolio_ret, 2),
            "gross_return_pct": round(gross_return, 3),
            "raw_return_pct": round(raw_return, 2),
            "benchmark_pct": round(bench_ret, 2),
            "top_names": ", ".join(
                f"{r['entry_type']}-{r['etf_name']}({r['score']['total']:.1f}"
                + ")"
                for r in selected
            ),
            "top_stages": ", ".join(r["stage"][0] for r in selected),
            "item_returns": ", ".join(
                f"{r['etf_name']} {r['period_return']:+.1f}%({r['exit_note']})"
                for r in selected
            ),
            "loss_drivers": loss_drivers,
            "risk_flags": "；".join(risk_flags) if risk_flags else "—",
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
                "mark_price": float(
                    _last_row_on_or_before(data[r["etf_code"]], exit_date)["close"]
                ),
                "weight": weight,
                "peak_score": float(r.get("peak_score") or r.get("score", {}).get("total") or 0),
            }
            for r, weight in zip(selected, weights)
            if r["etf_code"] not in stopped_codes
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
        empty.attrs["min_hold_trading_days"] = int(min_hold_trading_days)
        empty.attrs["weekly_max_broad"] = WEEKLY_MAX_BROAD if freq == "weekly" else None
        return empty

    df = pd.DataFrame(rows)
    df["cumulative"] = (1 + df["return_pct"] / 100).cumprod()
    df["benchmark_cumulative"] = (1 + df["benchmark_pct"] / 100).cumprod()
    daily_equity = _finalize_daily_equity(daily_records)
    if not daily_equity.empty:
        # Daily equity is authoritative; period NAV is retained for readable summaries.
        period_nav = daily_equity.groupby("period", sort=False)["nav"].last()
        df["cumulative"] = df["month"].map(period_nav).fillna(df["cumulative"])
        period_benchmark_nav = daily_equity.groupby("period", sort=False)["benchmark_nav"].last()
        df["benchmark_cumulative"] = df["month"].map(period_benchmark_nav).fillna(
            df["benchmark_cumulative"]
        )
    df.attrs["universe_size"] = len(etf_pool)
    df.attrs["max_etfs"] = max_etfs
    df.attrs["freq"] = freq
    df.attrs["effective_min_amount_yi"] = effective_min_amount_yi
    df.attrs["min_hold_trading_days"] = int(min_hold_trading_days)
    df.attrs["weekly_max_broad"] = WEEKLY_MAX_BROAD if freq == "weekly" else None
    df.attrs["one_way_cost_bps"] = float(one_way_cost_bps)
    df.attrs["stop_slippage_bps"] = float(stop_slippage_bps)
    df.attrs["offline"] = bool(offline)
    df.attrs["daily_equity"] = daily_equity
    df.attrs["positions"] = pd.DataFrame(position_records)
    df.attrs["trades"] = pd.DataFrame(trade_records).sort_values(
        ["date", "side", "code"]
    ).reset_index(drop=True) if trade_records else pd.DataFrame()
    df.attrs["universe_quality"] = universe_status
    df.attrs["benchmarks"] = _benchmark_comparison(data, daily_equity)
    return df


def write_report(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    top_n: int,
    min_amount_yi: float,
    freq: str,
    output_path: str | Path | None = None,
) -> Path:
    out_name = "backtest_result_weekly.md" if freq == "weekly" else "backtest_result.md"
    out_path = (
        Path(output_path)
        if output_path
        else PROJECT_ROOT / "codex" / "stock" / out_name
    )
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out_path.write_text("# ETF 三层框架回测\n\n无有效回测结果。\n", encoding="utf-8")
        return out_path

    daily_equity = df.attrs.get("daily_equity", pd.DataFrame())
    positions = df.attrs.get("positions", pd.DataFrame())
    trades = df.attrs.get("trades", pd.DataFrame())
    artifact_base = out_path.with_suffix("")
    daily_path = artifact_base.with_name(artifact_base.name + "_daily_equity.csv")
    positions_path = artifact_base.with_name(artifact_base.name + "_positions.csv")
    trades_path = artifact_base.with_name(artifact_base.name + "_trades.csv")
    run_manifest_path = artifact_base.with_name(artifact_base.name + "_run_manifest.json")
    if isinstance(daily_equity, pd.DataFrame) and not daily_equity.empty:
        daily_equity.to_csv(daily_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    if isinstance(positions, pd.DataFrame):
        positions.to_csv(positions_path, index=False, encoding="utf-8-sig")
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    period_label = "周度" if freq == "weekly" else "月度"
    period_return_label = "本周收益%" if freq == "weekly" else "本月收益%"
    latest_period_label = "最近完成周" if freq == "weekly" else "最近完成月"
    annual_factor = 52 if freq == "weekly" else 12
    period_std = df["return_pct"].std(ddof=0)
    sharpe = 0.0 if period_std == 0 else (df["return_pct"].mean() / period_std) * (annual_factor ** 0.5)
    if isinstance(daily_equity, pd.DataFrame) and not daily_equity.empty:
        daily_std = daily_equity["daily_return_pct"].std(ddof=0)
        daily_sharpe = (
            0.0
            if daily_std == 0
            else daily_equity["daily_return_pct"].mean() / daily_std * (252 ** 0.5)
        )
        max_drawdown = float(daily_equity["drawdown_pct"].min())
        current_drawdown = float(daily_equity.iloc[-1]["drawdown_pct"])
    else:
        daily_sharpe = sharpe
        max_drawdown = _max_drawdown(df["cumulative"])
        current_drawdown = (
            float(df["cumulative"].iloc[-1])
            / float(df["cumulative"].cummax().iloc[-1])
            - 1
        ) * 100
    total_ret = (df["cumulative"].iloc[-1] - 1) * 100
    bench_ret = (df["benchmark_cumulative"].iloc[-1] - 1) * 100
    latest = df.iloc[-1]
    latest_ret = float(latest["return_pct"])
    latest_bench_ret = float(latest["benchmark_pct"])
    actual_start = str(df.iloc[0]["entry_date"])
    actual_end = str(latest["exit_date"])
    ending_nav = float(latest["cumulative"])
    active_cost_bps = float(df.attrs.get("one_way_cost_bps", ONE_WAY_COST_BPS))
    active_slippage_bps = float(df.attrs.get("stop_slippage_bps", STOP_SLIPPAGE_BPS))
    sensitivity = _cost_sensitivity(
        df,
        daily_equity=daily_equity,
        base_cost_bps=active_cost_bps,
    )
    benchmark_table = df.attrs.get("benchmarks", pd.DataFrame())
    strategy_manifest = load_current_manifest()
    strategy_version = "未冻结"
    strategy_hash = "—"
    strategy_integrity = False
    strategy_errors: list[str] = ["缺少策略版本清单"]
    forward_start = None
    minimum_forward_weeks = 26
    if strategy_manifest:
        strategy_version = str(strategy_manifest.get("strategy_version", "未知"))
        strategy_hash = str(strategy_manifest.get("combined_sha256", "—"))
        strategy_integrity, strategy_errors = verify_manifest(strategy_manifest)
        forward_start = pd.to_datetime(strategy_manifest.get("forward_start"), errors="coerce")
        minimum_forward_weeks = int(strategy_manifest.get("minimum_forward_weeks", 26))
    forward_days = 0
    sample_label = "历史回放（非样本外）"
    if (
        forward_start is not None
        and not pd.isna(forward_start)
        and isinstance(daily_equity, pd.DataFrame)
        and not daily_equity.empty
    ):
        forward_days = int((pd.to_datetime(daily_equity["date"]) >= forward_start).sum())
        if forward_days == len(daily_equity):
            sample_label = "冻结后前向记录"
        elif forward_days > 0:
            sample_label = "历史回放与冻结后前向记录混合"
    forward_weeks = forward_days / 5
    universe_quality = str(df.attrs.get("universe_quality", "missing"))

    artifact_hashes = {}
    for artifact in (daily_path, positions_path, trades_path):
        if artifact.exists():
            artifact_hashes[artifact.name] = _file_sha256(artifact)
    run_manifest = {
        "schema_version": 1,
        "strategy_version": strategy_version,
        "strategy_sha256": strategy_hash,
        "strategy_integrity": strategy_integrity,
        "strategy_integrity_errors": strategy_errors,
        "sample_classification": sample_label,
        "forward_start": None if forward_start is None or pd.isna(forward_start) else _fmt_date(forward_start),
        "forward_observed_trading_days": forward_days,
        "minimum_forward_weeks": minimum_forward_weeks,
        "parameters": {
            "start": start_date,
            "end": end_date,
            "frequency": freq,
            "top_n": top_n,
            "minimum_amount_yi": min_amount_yi,
            "offline": bool(df.attrs.get("offline", False)),
            "one_way_cost_bps": active_cost_bps,
            "stop_slippage_bps": active_slippage_bps,
            "min_hold_trading_days": int(
                df.attrs.get("min_hold_trading_days", MIN_HOLD_TRADING_DAYS)
            ),
        },
        "universe_quality": universe_quality,
        "universe_file_sha256": _file_sha256(SCRIPT_DIR / "etf_universe_history.csv")
        if (SCRIPT_DIR / "etf_universe_history.csv").exists()
        else None,
        "artifacts": artifact_hashes,
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    effective_min_amount_yi = df.attrs.get("effective_min_amount_yi", min_amount_yi)
    weekly_broad_rule = (
        f"- 周度组合最多保留 {df.attrs.get('weekly_max_broad', WEEKLY_MAX_BROAD)} 个宽基，优先把席位留给产业方向。"
        if freq == "weekly"
        else "- 月度组合不限制宽基数量，按趋势、阶段和成交额统一排序。"
    )
    hold_bonus_label = "周度" if freq == "weekly" else "月度"
    hold_bonus = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])
    policy_compliant = (
        (df["selection_count"] >= 0)
        & df.apply(
            lambda row: int(row["selection_count"]) <= max_selection_count(row["market_state"]),
            axis=1,
        )
        & df.apply(
            lambda row: 0 <= float(row["exposure"]) <= float(
                get_target_exposure(row["market_state"])
            ) + 0.1,
            axis=1,
        )
    )
    period_summary = df[
        [
            "month",
            "entry_date",
            "exit_date",
            "market_state",
            "exposure",
            "return_pct",
            "benchmark_pct",
            "cumulative",
        ]
    ].copy()
    period_summary.columns = [
        "周期",
        "入场日",
        "截止日",
        "市场状态",
        "仓位%",
        period_return_label,
        "基准收益%",
        "累计净值",
    ]
    period_summary["超额收益%"] = (
        period_summary[period_return_label] - period_summary["基准收益%"]
    ).round(2)
    period_summary["累计收益%"] = (
        (period_summary["累计净值"] - 1) * 100
    ).round(2)
    period_summary["累计净值"] = period_summary["累计净值"].round(4)

    loss_rows = []
    for _, row in df[df["return_pct"] < 0].iterrows():
        conclusion, responsibility = _classify_loss_row(row)
        loss_rows.append({
            "周期": row["month"],
            period_return_label: float(row["return_pct"]),
            "基准收益%": float(row["benchmark_pct"]),
            "超额收益%": round(
                float(row["return_pct"]) - float(row["benchmark_pct"]),
                2,
            ),
            "主要拖累": row.get("loss_drivers", "—"),
            "价格归因": conclusion,
            "责任判断": responsibility,
        })
    loss_table = pd.DataFrame(loss_rows)

    lines = [
        f"# ETF 三层框架{period_label}回测 — {actual_start} ~ {actual_end}",
        "",
        "## 一眼结论",
        "",
        f"- 命令日期范围：{start_date} ~ {end_date}",
        f"- 实际交易区间：{actual_start} ~ {actual_end}",
        f"- {latest_period_label}：{latest['month']}（{latest['entry_date']} ~ {latest['exit_date']}）",
        f"- {latest_period_label}收益：{latest_ret:+.2f}%",
        f"- {latest_period_label}基准收益：{latest_bench_ret:+.2f}%",
        f"- 区间累计收益：{total_ret:+.2f}%",
        f"- 区间基准收益：{bench_ret:+.2f}%",
        f"- 区间超额收益：{total_ret - bench_ret:+.2f}%",
        f"- 期末净值：{ending_nav:.4f}（初始净值 1.0000）",
        f"- 期末距区间净值高点：{current_drawdown:.2f}%",
        f"- 每日净值口径最大回撤：{max_drawdown:.2f}%",
        f"- 样本性质：{sample_label}",
        f"- 策略版本：{strategy_version}（完整性校验：{'通过' if strategy_integrity else '未通过'}）",
        "",
        "## 逐期收益",
        "",
        period_summary.to_markdown(index=False),
        "",
        "## 亏损归因",
        "",
        "以下先做价格与规则归因；突发事件必须另查当期新闻，不能只凭K线臆测。",
        "",
        loss_table.to_markdown(index=False) if not loss_table.empty else "区间内无亏损周期。",
        "",
        "## 多基准比较",
        "",
        benchmark_table.to_markdown(index=False)
        if isinstance(benchmark_table, pd.DataFrame) and not benchmark_table.empty
        else "无可用多基准数据。",
        "",
        "## 可复算产物",
        "",
        f"- 每日权益：`{daily_path.name}`",
        f"- 每日持仓：`{positions_path.name}`",
        f"- 成交与止损：`{trades_path.name}`",
        f"- 运行清单：`{run_manifest_path.name}`",
        "",
        "## 成本敏感性",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "## 回测约束",
        "",
        f"- 调仓频率：{period_label}。",
        "- 信号日为调仓周期开始前最后一个交易日。",
        "- 选 ETF 只使用信号日及以前数据。",
        "- 市场状态由沪深300趋势和ETF池涨跌比共同判断，与实时/日度诊断共用 `market_state.py`。",
        "- 周期初先判断旧持仓是否仍可续持，再让旧仓和新候选统一竞争 TOP 组合。",
        "- 续持只使用信号日及以前数据；不使用当月未来收益决定是否持有。",
        weekly_broad_rule,
        "- 新仓按周期第一个交易日开盘成交；续持仓沿用上一周期收盘标记价，保留周末和节假日跳空收益。",
        "- 第 t 日止损价只使用 t-1 日以前的峰值；盘中触发按止损价减滑点成交，跳空跌破则按开盘价减滑点成交。",
        "- 单标的单日跌幅达到7%或同风险簇至少3个细分方向单日跌超5%，在下一交易日开盘执行日度风控退出，当周不换入新标的。",
        "- 市场风险预算范围为：主升80-100%、震荡70-100%、退潮60-90%、退潮末期/未知60-80%、冰点0-40%。",
        "- 每期允许0–3只；主升最多3只、震荡/退潮最多2只、退潮末期/冰点最多1只，不以低质量候选补位。",
        "- 排名比例按实际合格数量归一化：一只使用全部风险预算，两只按40:35，三只按40:35:25；冰点允许空仓。",
        "- 跨境/QDII允许入选；溢价未知时评分降权且仓位上限15%，溢价可核验且正常时上限25%。",
        f"- 所有仓位变化计入单边 {active_cost_bps:g}bp 交易成本；止损成交另计 {active_slippage_bps:g}bp 滑点。",
        "- 不使用当月已实现收益排序，避免未来函数。",
        "- 商品/资源/LOF 使用更宽的趋势止损，避免强主升浪中被普通行业 ETF 阈值过早洗出。",
        f"- 防守市场追高硬门禁：单日 >{CHASE_SINGLE_DAY_LIMIT:g}% 或 5日 >{CHASE_5D_RETURN_LIMIT:g}% → 不新开仓。",
        (
            f"- 新仓折溢价前总分必须≥{CORE_ENTRY_SCORE:.0f}，"
            f"5日正收益的最大单日贡献必须≤{MAX_NEW_POSITION_ONE_DAY_DOMINANCE:.0%}；"
            "主线强度、买点质量和风险惩罚仍须分别过门槛。"
        ),
        (
            f"- 健康旧仓在前{int(df.attrs.get('min_hold_trading_days', MIN_HOLD_TRADING_DAYS))}"
            "个交易日内获得替换保护；止损、主线失效和风险惩罚超限仍立即退出。"
        ),
        "- 非防守新仓若60日趋势为负，必须同时达到60分和20日涨幅15%的强反转门槛。",
        f"- 主升/震荡最多持有 {MAX_SAME_RISK_CLUSTER} 只同风险簇标的；退潮及更弱状态最多1只，续持仓同样计入。",
        "",
        "## 参数",
        "",
        f"- 每期选取：TOP {top_n}",
        f"- 命令最低成交额：{min_amount_yi} 亿",
        f"- 实际最低成交额：{effective_min_amount_yi} 亿",
        f"- 可续持阶段：{', '.join(sorted(HOLDABLE_STAGES))}",
        f"- 新仓折溢价前总分下限：{CORE_ENTRY_SCORE:.0f}",
        f"- 新仓5日正收益最大单日贡献：{MAX_NEW_POSITION_ONE_DAY_DOMINANCE:.0%}",
        f"- 周度最短持有保护：{int(df.attrs.get('min_hold_trading_days', MIN_HOLD_TRADING_DAYS))} 个交易日",
        f"- {hold_bonus_label}续持溢价：弱续持 +{hold_bonus['weak']:.0f}，正常续持 +{hold_bonus['normal']:.0f}，强续持 +{hold_bonus['strong']:.0f}",
        f"- 普通趋势止损：高点回撤 {abs(NORMAL_TRAIL_STOP):.0f}% 且跌破 MA20",
        f"- 商品/LOF 趋势止损：高点回撤 {abs(COMMODITY_TRAIL_STOP):.0f}% 且跌破 MA20",
        f"- 商品/LOF 抛物线止盈：20日涨幅≥{COMMODITY_PARABOLIC_R20:.0f}% 且60日涨幅≥{COMMODITY_PARABOLIC_R60:.0f}%",
        f"- 实际回测标的数：{df.attrs.get('universe_size', '未知')}",
        f"- 点时ETF池质量：{universe_quality}",
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
        f"- 日度夏普：{daily_sharpe:.2f}",
        f"- 最大回撤（日度权益）：{max_drawdown:.2f}%",
        f"- 平均仓位：{df['exposure'].mean():.1f}%",
        f"- 0–3标的与仓位上限合规率：{policy_compliant.mean() * 100:.1f}%",
        f"- 累计单边换手：{df['turnover_pct'].sum() / 100:.2f} 倍",
        f"- 累计成本影响：{df['cost_pct'].sum():.2f} 个百分点（逐期近似和）",
        "",
        "## 结果限制",
        "",
        "- 初始 `etf_universe_history.csv` 是当前固定池回填，质量标记为 partial；它能阻止未来新增产品回写历史，但无法恢复首次快照前已清盘或已移出池的产品。",
        f"- 冻结后前向样本当前约 {forward_weeks:.1f} 周；未达到 {minimum_forward_weeks} 周前，不用它宣称策略已通过样本外验证。",
        "- 回测只验证可时间点还原的价格、成交额和技术阶段，不包含无法完整归档的历史新闻、公告解读、实时主力资金和折溢价决策。",
        "- 日线 OHLC 仍无法还原完整盘中路径；当前模型使用前一日峰值、跳空开盘成交和独立止损滑点，结果仍应结合更高成本档压力测试。",
        "- 周度与月度结果来自同一历史区间，只是频率敏感性对照，不是彼此独立的样本外验证。",
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
    parser.add_argument("--top", type=int, default=TARGET_SELECTION_COUNT, help="每期最多持有数量（1-3，默认3）")
    parser.add_argument("--min-amount-yi", type=float, default=0.2, help="最低成交额，单位亿元")
    parser.add_argument("--max-etfs", type=int, default=None, help="调试用：仅回测前 N 只 ETF/LOF")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="weekly", help="调仓频率")
    parser.add_argument("--output", help="报告输出路径；默认写入周度/月度标准报告")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略本地行情缓存并重新下载")
    parser.add_argument("--offline", action="store_true", help="只从本地缓存读取行情，禁止网络回退")
    parser.add_argument("--cost-bps", type=float, default=ONE_WAY_COST_BPS, help="单边交易成本，bp")
    parser.add_argument(
        "--stop-slippage-bps",
        type=float,
        default=STOP_SLIPPAGE_BPS,
        help="止损成交额外滑点，bp",
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=MIN_HOLD_TRADING_DAYS,
        help="健康旧仓的最短持有保护交易日数；0用于消融对照",
    )
    args = parser.parse_args()

    df = run_backtest(
        args.start,
        args.end,
        top_n=args.top,
        min_amount_yi=args.min_amount_yi,
        max_etfs=args.max_etfs,
        freq=args.freq,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
        one_way_cost_bps=args.cost_bps,
        stop_slippage_bps=args.stop_slippage_bps,
        min_hold_trading_days=args.min_hold_days,
    )
    out_path = write_report(
        df,
        args.start,
        args.end,
        args.top,
        args.min_amount_yi,
        args.freq,
        output_path=args.output,
    )

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
