"""Shared market-regime classification for live scans and backtests."""
from __future__ import annotations

import pandas as pd


MARKET_STATES = ("主升", "震荡", "退潮", "退潮末期", "冰点", "未知")


def classify_market_state(
    benchmark_closes: list[float],
    *,
    breadth_ratio: float | None,
    benchmark_pct: float | None = None,
) -> str:
    """Classify regime from benchmark trend first and market breadth second."""
    clean = [float(value) for value in benchmark_closes if pd.notna(value) and float(value) > 0]
    if len(clean) < 5:
        return "未知"

    nav = clean[-1]
    ma5 = sum(clean[-5:]) / min(5, len(clean))
    ma20 = sum(clean[-20:]) / min(20, len(clean))
    gap20 = nav / ma20 - 1 if ma20 else 0.0
    width = 1.0 if breadth_ratio is None else max(float(breadth_ratio), 0.0)
    daily_pct = 0.0 if benchmark_pct is None else float(benchmark_pct)

    if daily_pct <= -3.0 or (len(clean) >= 20 and gap20 <= -0.05 and width < 0.5):
        return "冰点"
    if len(clean) >= 20 and gap20 <= -0.02:
        return "退潮末期"
    if nav < ma20 or (nav < ma5 and width < 0.5):
        return "退潮"
    if nav >= ma5 and nav >= ma20 * 1.02 and width >= 1.0:
        return "主升"
    return "震荡"


def breadth_ratio_on_date(
    data: dict[str, pd.DataFrame],
    signal_date,
) -> float | None:
    """Compute an ETF-pool advance/decline ratio using only data known by date."""
    target = pd.to_datetime(signal_date)
    up = 0
    down = 0
    for frame in data.values():
        known = frame[frame["date"] <= target].sort_values("date").tail(2)
        if len(known) < 2:
            continue
        change = float(known.iloc[-1]["close"]) / float(known.iloc[-2]["close"]) - 1
        if change > 0:
            up += 1
        elif change < 0:
            down += 1
    if not up and not down:
        return None
    if down == 0:
        return float(up)
    return up / down


def classify_from_history(
    benchmark: pd.DataFrame,
    signal_date,
    *,
    breadth_ratio: float | None,
) -> str:
    target = pd.to_datetime(signal_date)
    known = benchmark[benchmark["date"] <= target].sort_values("date").tail(20)
    if known.empty:
        return "未知"
    closes = known["close"].astype(float).tolist()
    benchmark_pct = None
    if len(closes) >= 2:
        benchmark_pct = (closes[-1] / closes[-2] - 1) * 100
    return classify_market_state(
        closes,
        breadth_ratio=breadth_ratio,
        benchmark_pct=benchmark_pct,
    )
