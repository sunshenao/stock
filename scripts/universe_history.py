"""Point-in-time ETF universe registry.

The initial snapshot is a current-pool backfill and is explicitly marked
partial. Subsequent additions and removals are timestamped, so forward tests
do not retrospectively rewrite the eligible universe.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "etf_universe_history.csv"
REGISTRY_COLUMNS = (
    "code",
    "name",
    "product_type",
    "effective_from",
    "effective_to",
    "source",
    "quality",
)


def load_registry(path: Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    frame = pd.read_csv(path, dtype={"code": str})
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[list(REGISTRY_COLUMNS)]


def active_codes(registry: pd.DataFrame, as_of) -> set[str]:
    if registry.empty:
        return set()
    target = pd.to_datetime(as_of)
    starts = pd.to_datetime(registry["effective_from"], errors="coerce")
    ends = pd.to_datetime(registry["effective_to"], errors="coerce")
    active = (starts <= target) & (ends.isna() | (ends >= target))
    return set(registry.loc[active, "code"].astype(str))


def sync_registry(
    etf_pool: dict,
    data: dict[str, pd.DataFrame],
    *,
    as_of,
    path: Path = DEFAULT_REGISTRY,
) -> pd.DataFrame:
    as_of_ts = pd.to_datetime(as_of).normalize()
    registry = load_registry(path)
    initial = registry.empty
    current_codes = {str(cfg["code"]) for cfg in etf_pool.values()}

    if not initial:
        active_now = active_codes(registry, as_of_ts)
        removed = active_now - current_codes
        if removed:
            mask = registry["code"].astype(str).isin(removed) & (
                registry["effective_to"].fillna("").astype(str).str.strip() == ""
            )
            registry.loc[mask, "effective_to"] = (
                as_of_ts - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            registry.loc[mask, "source"] = "local_pool_removal"

    known_codes = set(registry["code"].astype(str))
    new_rows = []
    for cfg in etf_pool.values():
        code = str(cfg["code"])
        if code in known_codes:
            continue
        frame = data.get(code)
        first_date = None
        if initial and frame is not None and not frame.empty:
            first_date = pd.to_datetime(frame["date"]).min()
        effective_from = (
            first_date.strftime("%Y-%m-%d")
            if first_date is not None
            else as_of_ts.strftime("%Y-%m-%d")
        )
        new_rows.append({
            "code": code,
            "name": cfg["name"],
            "product_type": cfg.get("type", "ETF"),
            "effective_from": effective_from,
            "effective_to": "",
            "source": "initial_current_pool_backfill" if initial else "local_pool_addition",
            "quality": "partial_survivorship_bias" if initial else "forward_point_in_time",
        })
    if new_rows:
        registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
    registry = registry.sort_values(["effective_from", "code"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False, encoding="utf-8-sig")
    return registry


def registry_quality(registry: pd.DataFrame) -> str:
    if registry.empty:
        return "missing"
    if registry["quality"].astype(str).str.contains("partial").any():
        return "partial"
    return "point_in_time"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the ETF universe registry")
    parser.add_argument("--as-of")
    args = parser.parse_args()
    registry = load_registry()
    if args.as_of:
        print("\n".join(sorted(active_codes(registry, args.as_of))))
    else:
        print(f"rows={len(registry)} quality={registry_quality(registry)}")


if __name__ == "__main__":
    main()
