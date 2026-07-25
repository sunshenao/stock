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
    calc_auto_score,
    calc_consecutive_strong,
    calc_etf_metrics,
    classify_stage,
    fetch_etf_hist,
    load_etf_txt,
)
from risk_rules import (  # noqa: E402
    CORE_ENTRY_SCORE,
    FALLBACK_ENTRY_SCORE as POLICY_FALLBACK_ENTRY_SCORE,
    MAX_SAME_RISK_CLUSTER,
    TARGET_SELECTION_COUNT,
    allocate_ranked_weights,
    get_target_exposure,
    new_position_trend_gate,
    risk_cluster,
)

HOLDABLE_STAGES = set(NEW_MONEY_STAGES) | {"趋势回撤期", "加速期⚠"}
COMMODITY_CATEGORIES = {"商品", "资源"}
COMMODITY_TRAIL_STOP = -35.0
NORMAL_TRAIL_STOP = -18.0
COMMODITY_PARABOLIC_R20 = 80.0
COMMODITY_PARABOLIC_R60 = 150.0
ONE_WAY_COST_BPS = 8.0
HISTORY_CACHE_DIR = PROJECT_ROOT / "codex" / "stock" / ".cache" / "etf_history"
# 追高硬门禁：符合任一条件时该腿仅按试探仓 10% 建仓
# 只在防守市场（退潮/退潮末期/冰点）触发；主升/震荡下真趋势不做机械稀释。
CHASE_5D_RETURN_LIMIT = 25.0
CHASE_SINGLE_DAY_LIMIT = 4.0
DEFENSIVE_STATES = {"退潮", "退潮末期", "冰点"}
# 组合最低入选分数：分数低于此的候选不进入主仓；不再机械补足到 TOP N
MIN_ENTRY_SCORE = CORE_ENTRY_SCORE
FALLBACK_ENTRY_SCORE = POLICY_FALLBACK_ENTRY_SCORE
FALLBACK_BLOCKED_STAGES = {"衰竭期", "衰弱期", "弱势期", "加速见顶⚠"}
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
        return "未知", get_target_exposure("未知") / 100
    ma20 = recent["close"].iloc[-21:].mean()
    close = recent["close"].iloc[-1]
    if close > ma20 * 1.02:
        return "主升", get_target_exposure("主升") / 100
    elif close > ma20:
        return "震荡", get_target_exposure("震荡") / 100
    elif close > ma20 * 0.98:
        return "退潮", get_target_exposure("退潮") / 100
    elif close > ma20 * 0.95:
        return "退潮末期", get_target_exposure("退潮末期") / 100
    else:
        return "冰点", get_target_exposure("冰点") / 100


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
    peak = cumulative.cummax()
    dd = cumulative / peak - 1
    return round(float(dd.min()) * 100, 2)


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
        responsibility = "框架需复核补位、续持和交易成本"

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


def _simulate_period_return(
    df: pd.DataFrame,
    start,
    end,
    item: dict,
    freq: str,
    carry_price: float | None = None,
) -> tuple[float, str, str, str]:
    """
    用日线 low 模拟周期内止损。
    新仓按周期第一个交易日开盘价买入；续持仓以上一期收盘标记价衔接，
    因而不会漏掉周末/节假日跳空。止损从入场后的第一根日线开始判断。
    """
    rows = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()
    if len(rows) < 2:
        return 0.0, "", "", "数据不足"

    rows = rows.sort_values("date").reset_index(drop=True)
    first = rows.iloc[0]
    entry_price = (
        float(carry_price)
        if carry_price is not None and carry_price > 0
        else float(first.get("open", first["close"]))
    )
    if entry_price <= 0:
        return 0.0, "", "", "入场价无效"

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

    peak = entry_price
    hard_stop = entry_price * (1 - initial_stop_pct)
    for _, row in rows.iterrows():
        high = float(row["high"]) if "high" in rows.columns and pd.notna(row.get("high")) else float(row["close"])
        low = float(row["low"]) if "low" in rows.columns and pd.notna(row.get("low")) else float(row["close"])
        peak = max(peak, high)
        trail_price = peak * (1 - trailing_stop_pct)
        stop_price = max(hard_stop, trail_price)

        if low <= stop_price:
            ret = (stop_price / entry_price - 1) * 100
            return ret, _fmt_date(first["date"]), _fmt_date(row["date"]), f"止损@{stop_price:.3f}"

    last = rows.iloc[-1]
    ret = (float(last["close"]) / entry_price - 1) * 100
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
    current_score = float(current.get("score", {}).get("total") or 0)
    peak_score = max(float(previous.get("peak_score") or current_score), current_score)
    score_drop = 1 - current_score / peak_score if peak_score > 0 else 0.0
    rank = int(current.get("_rank") or 999)

    if stage in HOLDABLE_STAGES:
        return True, f"{stage}续持", current
    if _is_commodity_like(current) and ret20 > 20 and ret60 > 40:
        return True, "商品大趋势续持", current
    if rank <= 30 and score_drop < 0.40 and ret20 >= 0:
        return True, f"排名{rank}且20日趋势未负，延续持有", current
    if rank > 30 and score_drop >= 0.40:
        return False, f"排名{rank}且评分较峰值下降{score_drop:.0%}", current
    return False, f"{stage}不再持有", current


def _new_position_allowed(item: dict, market_state: str) -> bool:
    """弱市过滤：冰点不新开仓，退潮/退潮末期提高新开仓门槛。"""
    if market_state == "冰点":
        return False
    if item.get("is_qdii") and not _is_commodity_like(item):
        return False
    if _is_chase_high(item, market_state):
        return False

    metrics = item.get("metrics", {})
    score = float(item.get("score", {}).get("total") or 0)
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


def _fallback_position_allowed(item: dict, market_state: str) -> bool:
    """三标的补位门禁：允许较早/横盘阶段，但拒绝衰竭、追高和高风险跨境品种。"""
    if item.get("is_qdii") and not _is_commodity_like(item):
        return False
    if _is_chase_high(item, market_state):
        return False

    score = float(item.get("score", {}).get("total") or 0)
    stage = item.get("stage", ("", "", ""))[0]
    if score < FALLBACK_ENTRY_SCORE or stage in FALLBACK_BLOCKED_STAGES:
        return False

    metrics = item.get("metrics", {})
    ret20 = float(metrics.get("ret_20d") or 0)
    if market_state in {"退潮", "退潮末期", "冰点"} and ret20 < -8:
        return False
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
    elif item.get("entry_tier") == "补位":
        score -= 3.0
    elif item.get("entry_tier") == "防守补位":
        score -= 10.0
        if item.get("industry") in {"宽基", "红利", "公用事业", "金融", "债券"}:
            score += 8.0
    if item.get("entry_type") == "续持":
        bonus_table = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])
        score += bonus_table["normal"]
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
    - 主候选和补位候选已在上游分别通过质量门禁。
    - 周度更强调主线弹性：最多保留 1 个宽基，避免宽基挤占产业 ETF 名额。
    """
    ordered = sorted(candidate_pool, key=lambda r: _selection_priority(r, freq), reverse=True)
    if freq != "weekly":
        return ordered[:top_n]

    selected = []
    delayed_broad = []
    broad_count = 0
    industry_count: dict[str, int] = {}
    cluster_count: dict[str, int] = {}
    cluster_limit = (
        TARGET_SELECTION_COUNT
        if market_state == "主升"
        else MAX_SAME_RISK_CLUSTER
    )
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
        if (
            cluster_count.get(cluster, 0) >= cluster_limit
            and r.get("entry_type") != "续持"
        ):
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
            if (
                cluster_count.get(cluster, 0) >= cluster_limit
                and r.get("entry_type") != "续持"
            ):
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
) -> pd.DataFrame:
    """按精确回测区间缓存行情，避免重复下载污染运行时间和可复现性。"""
    product = str(product_type or "ETF").upper()
    cache_path = HISTORY_CACHE_DIR / f"{code}_{product}_{start_date}_{end_date}.csv"
    if cache_path.exists() and not refresh_cache:
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            if not cached.empty:
                return cached
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    df = fetch_etf_hist(code, start_date, end_date, product)
    if not df.empty:
        HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8")
    return df


def fetch_all_hist(
    etf_pool: dict,
    start_date: str,
    end_date: str,
    refresh_cache: bool = False,
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
    refresh_cache: bool = False,
) -> pd.DataFrame:
    if top_n != TARGET_SELECTION_COUNT:
        raise ValueError(f"正式组合固定选择 {TARGET_SELECTION_COUNT} 只标的，不能使用 top_n={top_n}")
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
    )
    if bench_df.empty:
        raise RuntimeError("沪深300ETF 基准行情获取失败")

    data = fetch_all_hist(etf_pool, fetch_start, fetch_end, refresh_cache)
    periods = _periods(start_dt, end_dt, freq)
    rows = []
    holdings: dict[str, dict] = {}
    benchmark_mark_price: float | None = None

    for label, period_start, period_end in periods:
        signal_date = _latest_before(bench_df, period_start)
        entry_date = _first_between(bench_df, period_start, period_end)
        exit_date = _last_between(bench_df, period_start, period_end)
        if signal_date is None or entry_date is None or exit_date is None or entry_date >= exit_date:
            continue
        market_state, _ = _simple_market_state(bench_df, signal_date)
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
        )
        if not ranked_all:
            print(f"  {label}: 无候选标的")
            exit_turnover = sum(float(item.get("weight", 0.0)) for item in holdings.values())
            exit_cost_pct = exit_turnover * ONE_WAY_COST_BPS / 100
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
            and float(r.get("score", {}).get("total") or 0) >= MIN_ENTRY_SCORE
        ]
        primary_codes = {r["etf_code"] for r in new_ranked}
        fallback_ranked = [
            r for r in ranked_all
            if r["etf_code"] not in primary_codes
            and _fallback_position_allowed(r, market_state)
        ]
        fallback_codes = primary_codes | {r["etf_code"] for r in fallback_ranked}
        defensive_ranked = [
            r for r in ranked_all
            if r["etf_code"] not in fallback_codes
            and not (r.get("is_qdii") and not _is_commodity_like(r))
            and not _is_chase_high(r, market_state)
        ]
        carry_candidates = []
        sold_notes = []
        for previous in holdings.values():
            keep, reason, current = _should_hold_position(previous, ranked_by_code, data, signal_date)
            if keep and current is not None:
                carry_candidates.append({
                    **current,
                    "entry_type": "续持",
                    "entry_tier": "持有",
                    "hold_reason": reason,
                    "holding_since": previous.get("holding_since") or current.get("signal_date"),
                    "carry_price": previous.get("mark_price"),
                    "peak_score": max(
                        float(previous.get("peak_score") or 0),
                        float(current.get("score", {}).get("total") or 0),
                    ),
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

        for r in fallback_ranked:
            if r["etf_code"] in selected_codes:
                continue
            candidate_pool.append({
                **r,
                "entry_type": "补位",
                "entry_tier": "补位",
                "hold_reason": "强信号不足三只，按池内质量门禁补位",
                "holding_since": _fmt_date(entry_date),
                "carry_price": None,
                "peak_score": float(r.get("score", {}).get("total") or 0),
            })
            selected_codes.add(r["etf_code"])

        for r in defensive_ranked:
            if r["etf_code"] in selected_codes:
                continue
            candidate_pool.append({
                **r,
                "entry_type": "防守补位",
                "entry_tier": "防守补位",
                "hold_reason": "核心与常规补位不足三只，按固定池防守优先补位",
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
        if len(selected) < TARGET_SELECTION_COUNT:
            print(
                f"  {label}: 数据/分散约束后仅 {len(selected)} 只，"
                "该周期不满足正式三标的口径",
                flush=True,
            )
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
                carry_price=r.get("carry_price"),
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
            exit_turnover = sum(float(item.get("weight", 0.0)) for item in holdings.values())
            exit_cost_pct = exit_turnover * ONE_WAY_COST_BPS / 100
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
        # 唯一仓位口径：三只按40%/35%/25%的强弱结构分配市场目标仓位。
        # 标的不足时保留对应缺口为现金，不把弱候选或单腿机械放大。
        n_legs = len(selected)
        rank_weights_pct = allocate_ranked_weights(market_state, n_legs)
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
        cost_return_pct = total_turnover * ONE_WAY_COST_BPS / 100
        raw_return = sum(r["period_return"] for r in selected) / n_legs if n_legs else 0.0
        portfolio_ret = weighted_sum - cost_return_pct
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
        fallback_count = sum(
            1
            for r in selected
            if r.get("entry_tier") in {"补位", "防守补位"}
        )
        if fallback_count:
            risk_flags.append(f"{fallback_count}只补位标的")
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
            "cost_pct": round(cost_return_pct, 3),
            "return_pct": round(portfolio_ret, 2),
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

    period_label = "周度" if freq == "weekly" else "月度"
    period_return_label = "本周收益%" if freq == "weekly" else "本月收益%"
    latest_period_label = "最近完成周" if freq == "weekly" else "最近完成月"
    annual_factor = 52 if freq == "weekly" else 12
    period_std = df["return_pct"].std(ddof=0)
    sharpe = 0.0 if period_std == 0 else (df["return_pct"].mean() / period_std) * (annual_factor ** 0.5)
    total_ret = (df["cumulative"].iloc[-1] - 1) * 100
    bench_ret = (df["benchmark_cumulative"].iloc[-1] - 1) * 100
    latest = df.iloc[-1]
    latest_ret = float(latest["return_pct"])
    latest_bench_ret = float(latest["benchmark_pct"])
    actual_start = str(df.iloc[0]["entry_date"])
    actual_end = str(latest["exit_date"])
    ending_nav = float(latest["cumulative"])
    current_drawdown = (
        ending_nav / float(df["cumulative"].cummax().iloc[-1]) - 1
    ) * 100
    effective_min_amount_yi = df.attrs.get("effective_min_amount_yi", min_amount_yi)
    weekly_broad_rule = (
        f"- 周度组合最多保留 {df.attrs.get('weekly_max_broad', WEEKLY_MAX_BROAD)} 个宽基，优先把席位留给产业方向。"
        if freq == "weekly"
        else "- 月度组合不限制宽基数量，按趋势、阶段和成交额统一排序。"
    )
    hold_bonus_label = "周度" if freq == "weekly" else "月度"
    hold_bonus = HOLD_BONUS.get(freq, HOLD_BONUS["monthly"])
    policy_compliant = (
        (df["selection_count"] == TARGET_SELECTION_COUNT)
        & df.apply(
            lambda row: abs(
                float(row["exposure"]) - float(get_target_exposure(row["market_state"]))
            ) < 0.1,
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
        "## 回测约束",
        "",
        f"- 调仓频率：{period_label}。",
        "- 信号日为调仓周期开始前最后一个交易日。",
        "- 选 ETF 只使用信号日及以前数据。",
        "- 周期初先判断旧持仓是否仍可续持，再让旧仓和新候选统一竞争 TOP 组合。",
        "- 续持只使用信号日及以前数据；不使用当月未来收益决定是否持有。",
        weekly_broad_rule,
        "- 新仓按周期第一个交易日开盘成交；续持仓沿用上一周期收盘标记价，保留周末和节假日跳空收益。",
        "- 周期内用日线 low 模拟止损，触发后按止损价退出。",
        "- 市场目标仓位统一为：主升100%、震荡90%、退潮70%、退潮末期60%、冰点40%。",
        "- 每期固定三只标的，按排名分配目标仓位的40%/35%/25%。",
        f"- 所有仓位变化计入单边 {ONE_WAY_COST_BPS:g}bp 交易成本。",
        "- 不使用当月已实现收益排序，避免未来函数。",
        "- 商品/资源/LOF 使用更宽的趋势止损，避免强主升浪中被普通行业 ETF 阈值过早洗出。",
        f"- 防守市场追高硬门禁：单日 >{CHASE_SINGLE_DAY_LIMIT:g}% 或 5日 >{CHASE_5D_RETURN_LIMIT:g}% → 不新开仓。",
        f"- 核心门槛 {MIN_ENTRY_SCORE:g} 分，常规补位门槛 {FALLBACK_ENTRY_SCORE:g} 分；不足三只时按固定池防守优先补位。",
        "- 非防守新仓若60日趋势为负，必须同时达到60分和20日涨幅15%的强反转门槛。",
        f"- 震荡/退潮周度组合最多新开 {MAX_SAME_RISK_CLUSTER} 只同风险簇标的；主升期或合格旧强仓不机械去相关。",
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
        f"- 平均仓位：{df['exposure'].mean():.1f}%",
        f"- 三标的与目标仓位合规率：{policy_compliant.mean() * 100:.1f}%",
        f"- 累计单边换手：{df['turnover_pct'].sum() / 100:.2f} 倍",
        f"- 累计成本影响：{df['cost_pct'].sum():.2f} 个百分点（逐期近似和）",
        "",
        "## 结果限制",
        "",
        "- 历史期统一使用当前 `scripts/etf.txt`，可能存在ETF池幸存者偏差；成立较晚的ETF会因无历史行情自然跳过，但历史已清盘或已移出池的产品不会出现。",
        "- 回测只验证可时间点还原的价格、成交额和技术阶段，不包含无法完整归档的历史新闻、公告解读、实时主力资金和折溢价决策。",
        "- 日线止损只能确认当日最高/最低是否触发，无法还原盘中先后顺序；实际滑点也可能高于统一的8bp假设。",
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
    parser.add_argument("--top", type=int, default=TARGET_SELECTION_COUNT, help="固定为3，仅保留参数用于兼容旧命令")
    parser.add_argument("--min-amount-yi", type=float, default=0.2, help="最低成交额，单位亿元")
    parser.add_argument("--max-etfs", type=int, default=None, help="调试用：仅回测前 N 只 ETF/LOF")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="weekly", help="调仓频率")
    parser.add_argument("--output", help="报告输出路径；默认写入周度/月度标准报告")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略本地行情缓存并重新下载")
    args = parser.parse_args()

    df = run_backtest(
        args.start,
        args.end,
        top_n=args.top,
        min_amount_yi=args.min_amount_yi,
        max_etfs=args.max_etfs,
        freq=args.freq,
        refresh_cache=args.refresh_cache,
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
