"""Conservative daily execution primitives shared by backtests.

The model deliberately avoids intraday path assumptions that cannot be known
from daily OHLC data. A trailing stop used on day t is based on the peak known
through day t-1. Gap-down exits fill at the opening price, then pay slippage.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StopSimulation:
    path: pd.DataFrame
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    exit_note: str

    @property
    def stopped(self) -> bool:
        return self.exit_note.startswith("止损")


def simulate_long_with_stop(
    rows: pd.DataFrame,
    *,
    entry_price: float,
    initial_stop_pct: float,
    trailing_stop_pct: float,
    slippage_bps: float = 10.0,
    forced_exit_dates: set[pd.Timestamp] | None = None,
) -> StopSimulation | None:
    """Simulate a long leg from daily OHLC data without optimistic look-ahead."""
    if rows.empty or entry_price <= 0:
        return None

    ordered = rows.sort_values("date").reset_index(drop=True).copy()
    peak_before_day = float(entry_price)
    hard_stop = float(entry_price) * (1 - float(initial_stop_pct))
    stopped = False
    exit_price = float(entry_price)
    exit_date = pd.to_datetime(ordered.iloc[-1]["date"])
    exit_note = "持有到期"
    records: list[dict] = []
    slip_factor = 1 - max(float(slippage_bps), 0.0) / 10_000
    forced_dates = {
        pd.to_datetime(value).normalize()
        for value in (forced_exit_dates or set())
    }

    for _, row in ordered.iterrows():
        date = pd.to_datetime(row["date"])
        open_price = float(row.get("open", row["close"]))
        high = float(row.get("high", row["close"]))
        low = float(row.get("low", row["close"]))
        close = float(row["close"])

        if stopped:
            records.append({
                "date": date,
                "factor": exit_price / entry_price,
                "mark_price": exit_price,
                "active": False,
                "stop_price": None,
                "stop_fill": None,
                "stop_reason": "",
            })
            continue

        # The stop for day t only uses information available before day t.
        trailing_stop = peak_before_day * (1 - float(trailing_stop_pct))
        stop_price = max(hard_stop, trailing_stop)
        stop_fill = None
        stop_reason = ""
        if date.normalize() in forced_dates:
            stop_fill = open_price * slip_factor
            stop_reason = "日度风控次日开盘"
        elif open_price <= stop_price:
            stop_fill = open_price * slip_factor
            stop_reason = "跳空止损"
        elif low <= stop_price:
            stop_fill = stop_price * slip_factor
            stop_reason = "盘中止损"

        if stop_fill is not None:
            stopped = True
            exit_price = float(stop_fill)
            exit_date = date
            exit_note = f"止损-{stop_reason}@{exit_price:.3f}"
            records.append({
                "date": date,
                "factor": exit_price / entry_price,
                "mark_price": exit_price,
                "active": False,
                "stop_price": stop_price,
                "stop_fill": exit_price,
                "stop_reason": stop_reason,
            })
            continue

        # Today's high becomes usable for tomorrow's trailing stop.
        peak_before_day = max(peak_before_day, high)
        exit_price = close
        records.append({
            "date": date,
            "factor": close / entry_price,
            "mark_price": close,
            "active": True,
            "stop_price": stop_price,
            "stop_fill": None,
            "stop_reason": "",
        })

    path = pd.DataFrame(records)
    return StopSimulation(
        path=path,
        entry_date=pd.to_datetime(ordered.iloc[0]["date"]).strftime("%Y-%m-%d"),
        exit_date=exit_date.strftime("%Y-%m-%d"),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        return_pct=(float(exit_price) / float(entry_price) - 1) * 100,
        exit_note=exit_note,
    )


def cost_return_pct(turnover: float, one_way_cost_bps: float) -> float:
    """Return transaction-cost drag in percentage points of portfolio NAV."""
    return max(float(turnover), 0.0) * max(float(one_way_cost_bps), 0.0) / 100
