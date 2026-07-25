"""Shared daily risk rules used by live review and historical simulation."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from risk_rules import risk_cluster


EXTREME_DAILY_DROP_PCT = -7.0
CLUSTER_MEMBER_DROP_PCT = -5.0
CLUSTER_MIN_DISTINCT_DIRECTIONS = 3


def detect_cluster_crashes(rows: list[dict]) -> set[str]:
    """Return risk clusters with broad, independently confirmed daily crashes."""
    weak_members: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if float(row.get("pct") or 0) > CLUSTER_MEMBER_DROP_PCT:
            continue
        cluster = risk_cluster(
            str(row.get("sector") or "").split("/", 1)[0],
            row.get("industry", "其他"),
            row.get("name", ""),
        )
        member = str(row.get("sector") or row.get("name") or row.get("code") or "")
        weak_members[cluster].add(member)
    return {
        cluster
        for cluster, members in weak_members.items()
        if len(members) >= CLUSTER_MIN_DISTINCT_DIRECTIONS
    }


def cluster_crash_calendar(
    etf_pool: dict,
    data: dict[str, pd.DataFrame],
    start,
    end,
) -> dict[pd.Timestamp, set[str]]:
    """Build point-in-time cluster alerts from daily closes across the fixed pool."""
    rows_by_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    for direction, cfg in etf_pool.items():
        frame = data.get(cfg["code"])
        if frame is None or frame.empty:
            continue
        known = frame[frame["date"] <= end_ts].sort_values("date").copy()
        known["pct"] = known["close"].pct_change() * 100
        known = known[known["date"] >= start_ts]
        for _, row in known.iterrows():
            rows_by_date[pd.to_datetime(row["date"])].append({
                "code": cfg["code"],
                "name": cfg["name"],
                "sector": direction,
                "industry": cfg.get("ind", "其他"),
                "pct": float(row["pct"]) if pd.notna(row["pct"]) else 0.0,
            })
    return {
        date: detect_cluster_crashes(rows)
        for date, rows in rows_by_date.items()
        if rows
    }
