"""Incremental, local-first market-data storage for the fixed ETF universe.

Each instrument has one canonical CSV file. Requested ranges are served from
local storage whenever covered; only missing edges are fetched from the
network and merged. Derived indicators are trailing-only and therefore safe
for point-in-time research.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "codex" / "stock" / ".cache" / "market_data_v2"
LEGACY_CACHE_ROOT = PROJECT_ROOT / "codex" / "stock" / ".cache" / "etf_history"
EDGE_TOLERANCE_DAYS = 7

RAW_COLUMN_ALIASES = {
    "日期": "date",
    "净值日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude_provider",
    "涨跌幅": "pct_chg_provider",
    "日增长率": "nav_growth_pct",
    "涨跌额": "change_provider",
    "换手率": "turnover_rate",
    "盘后量": "after_hours_volume",
    "盘后额": "after_hours_amount",
    "单位净值": "unit_nav",
    "累计净值": "accumulated_nav",
    "申购状态": "subscription_status",
    "赎回状态": "redemption_status",
    "prevclose": "prev_close_provider",
    "postVol": "after_hours_volume",
    "postAmt": "after_hours_amount",
}

NUMERIC_RAW_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude_provider",
    "pct_chg_provider",
    "change_provider",
    "turnover_rate",
    "after_hours_volume",
    "after_hours_amount",
    "unit_nav",
    "accumulated_nav",
    "nav_growth_pct",
    "prev_close_provider",
)

CORE_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "after_hours_volume",
    "after_hours_amount",
    "source",
)
CANONICAL_FEATURE_COLUMNS = {
    "pct_chg",
    "atr_pct_14",
    "rsi_14",
    "ma_close_250",
    "return_250d_pct",
    "annualized_volatility_20d_pct",
    "obv",
    "amihud_illiquidity_20d",
}

NetworkFetcher = Callable[[str, str, str, str], pd.DataFrame]


def latest_completed_business_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the latest day whose daily bar should be complete."""
    ts = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    day = ts.normalize()
    if ts.weekday() >= 5:
        return (day - pd.offsets.BDay(1)).normalize()
    if ts.hour < 15 or (ts.hour == 15 and ts.minute < 10):
        return (day - pd.offsets.BDay(1)).normalize()
    return day


def _safe_product(product_type: str) -> str:
    product = str(product_type or "ETF").upper()
    return "".join(char for char in product if char.isalnum() or char in {"_", "-"}) or "ETF"


def history_path(
    code: str,
    product_type: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    return Path(cache_root) / "bars" / f"{code}_{_safe_product(product_type)}.csv"


def metadata_path(
    code: str,
    product_type: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    return Path(cache_root) / "metadata" / f"{code}_{_safe_product(product_type)}.json"


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        series = series.replace({"": None, "--": None, "nan": None, "None": None})
    return pd.to_numeric(series, errors="coerce")


def adjust_price_discontinuities(frame: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust price history when a split-like discontinuity is detected."""
    if frame.empty or "close" not in frame.columns:
        return frame
    df = frame.sort_values("date").reset_index(drop=True).copy()
    price_columns = [
        column for column in ("open", "high", "low", "close") if column in df.columns
    ]
    for column in price_columns:
        df[column] = _numeric(df[column])
    for index in range(1, len(df)):
        previous_close = df.at[index - 1, "close"]
        current_reference = (
            df.at[index, "open"] if "open" in df.columns else df.at[index, "close"]
        )
        if (
            pd.isna(previous_close)
            or pd.isna(current_reference)
            or previous_close <= 0
        ):
            continue
        ratio = current_reference / previous_close
        if ratio < 0.65 or ratio > 1.55:
            df.loc[: index - 1, price_columns] *= ratio
    return df


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    losses = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gains / losses.replace(0, math.nan)
    result = 100 - 100 / (1 + rs)
    return result.where(losses.ne(0), 100.0)


def add_trailing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute canonical fields and trailing indicators after every merge."""
    if frame.empty:
        return frame
    df = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True).copy()
    for column in NUMERIC_RAW_COLUMNS:
        if column not in df:
            df[column] = math.nan
        df[column] = _numeric(df[column])

    close = df["close"]
    previous_close = close.shift(1)
    df["prev_close"] = previous_close
    df["change"] = close - previous_close
    df["pct_chg"] = close.pct_change() * 100
    df["log_return"] = (close / previous_close).apply(
        lambda value: math.log(value) if pd.notna(value) and value > 0 else math.nan
    )
    df["amplitude_pct"] = (df["high"] - df["low"]) / previous_close * 100
    df["gap_pct"] = (df["open"] / previous_close - 1) * 100
    df["intraday_return_pct"] = (close / df["open"] - 1) * 100
    df["typical_price"] = (df["high"] + df["low"] + close) / 3

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = true_range
    df["atr_14"] = true_range.rolling(14, min_periods=5).mean()
    df["atr_pct_14"] = df["atr_14"] / close * 100
    df["rsi_14"] = _rsi(close)

    for window in (5, 10, 20, 60, 120, 250):
        min_periods = min(window, 5)
        df[f"ma_close_{window}"] = close.rolling(window, min_periods=min_periods).mean()
        df[f"return_{window}d_pct"] = close.pct_change(window) * 100

    for window in (5, 20, 60):
        min_periods = min(window, 5)
        df[f"volume_ma_{window}"] = df["volume"].rolling(window, min_periods=min_periods).mean()
        df[f"amount_ma_{window}"] = df["amount"].rolling(window, min_periods=min_periods).mean()
        df[f"turnover_ma_{window}"] = (
            df["turnover_rate"].rolling(window, min_periods=min_periods).mean()
        )
        rolling_high = df["high"].rolling(window, min_periods=min_periods).max()
        rolling_low = df["low"].rolling(window, min_periods=min_periods).min()
        df[f"high_{window}d"] = rolling_high
        df[f"low_{window}d"] = rolling_low
        df[f"drawdown_{window}d_pct"] = (close / rolling_high - 1) * 100
        price_range = rolling_high - rolling_low
        df[f"price_position_{window}d"] = (close - rolling_low) / price_range.replace(0, math.nan)

    df["volume_ratio_5d"] = df["volume"] / df["volume_ma_5"].replace(0, math.nan)
    df["amount_ratio_5d"] = df["amount"] / df["amount_ma_5"].replace(0, math.nan)
    df["annualized_volatility_20d_pct"] = (
        close.pct_change().rolling(20, min_periods=10).std() * math.sqrt(252) * 100
    )
    signed_volume = df["volume"].where(df["change"] >= 0, -df["volume"])
    df["obv"] = signed_volume.fillna(0).cumsum()
    df["amihud_illiquidity_20d"] = (
        close.pct_change().abs() / df["amount"].replace(0, math.nan)
    ).rolling(20, min_periods=10).mean()
    return df


def prepare_history(frame: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """Normalize provider columns while retaining useful raw provider fields."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.rename(columns=RAW_COLUMN_ALIASES).copy()
    if "date" not in df or "close" not in df:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    for column in NUMERIC_RAW_COLUMNS:
        if column in df:
            df[column] = _numeric(df[column])
    for column in CORE_COLUMNS:
        if column not in df:
            df[column] = source if column == "source" else math.nan
    df["source"] = df["source"].fillna(source).replace("", source)
    return add_trailing_features(adjust_price_discontinuities(df))


def merge_history(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return add_trailing_features(incoming)
    if incoming.empty:
        return add_trailing_features(existing)
    old = existing.copy().set_index("date")
    new = incoming.copy().set_index("date")
    merged = new.combine_first(old).reset_index()
    return add_trailing_features(adjust_price_discontinuities(merged))


def _read_csv(path: Path, source: str = "local") -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    if source == "local" and CANONICAL_FEATURE_COLUMNS.issubset(frame.columns):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        return (
            frame.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
    return prepare_history(frame, source=source)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    serializable = frame.copy()
    serializable["date"] = pd.to_datetime(serializable["date"]).dt.strftime("%Y-%m-%d")
    serializable.to_csv(temp, index=False, encoding="utf-8")
    os.replace(temp, path)


def _atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(
    code: str,
    product_type: str,
    frame: pd.DataFrame,
    cache_root: Path,
    known_inception_date: pd.Timestamp | None = None,
) -> None:
    path = history_path(code, product_type, cache_root)
    meta_path = metadata_path(code, product_type, cache_root)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    raw_fields = [
        field
        for field in (
            "turnover_rate",
            "amplitude_provider",
            "pct_chg_provider",
            "change_provider",
            "after_hours_volume",
            "after_hours_amount",
            "unit_nav",
            "accumulated_nav",
        )
        if field in frame and frame[field].notna().any()
    ]
    existing_payload = {}
    if meta_path.exists():
        try:
            existing_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
    if known_inception_date is None and existing_payload.get("known_inception_date"):
        known_inception_date = pd.to_datetime(existing_payload["known_inception_date"])
    payload = {
        "schema_version": 2,
        "code": code,
        "product_type": _safe_product(product_type),
        "rows": int(len(frame)),
        "min_date": frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else None,
        "max_date": frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else None,
        "known_inception_date": (
            pd.Timestamp(known_inception_date).strftime("%Y-%m-%d")
            if known_inception_date is not None
            else None
        ),
        "columns": list(frame.columns),
        "raw_optional_fields": raw_fields,
        "sources": sorted(set(frame.get("source", pd.Series(dtype=str)).dropna().astype(str))),
        "missing_core_values": int(
            frame[[column for column in ("open", "high", "low", "close", "volume", "amount") if column in frame]]
            .isna()
            .sum()
            .sum()
        ),
        "duplicate_dates": int(frame["date"].duplicated().sum()) if not frame.empty else 0,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256": _sha256(path),
    }
    temp = meta_path.with_suffix(meta_path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, meta_path)


def _best_legacy_frame(
    code: str,
    product_type: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    legacy_root: Path = LEGACY_CACHE_ROOT,
) -> pd.DataFrame:
    paths = list(Path(legacy_root).glob(f"{code}_{_safe_product(product_type)}_*.csv"))
    if not paths:
        return pd.DataFrame()
    # Exact-range files are mostly duplicates; one largest file normally contains
    # the complete history returned by the Sina fallback.
    path = max(paths, key=lambda item: item.stat().st_size)
    try:
        raw = pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    full_history_source = any(column in raw for column in ("postVol", "postAmt"))
    frame = prepare_history(raw, source="legacy_cache")
    if frame.empty:
        return frame
    provider_min_date = pd.to_datetime(frame["date"]).min()
    requested_starts = []
    for candidate in paths:
        parts = candidate.stem.split("_")
        if len(parts) >= 4:
            try:
                requested_starts.append(pd.to_datetime(parts[-2], format="%Y%m%d"))
            except ValueError:
                continue
    if (
        requested_starts
        and provider_min_date < max(requested_starts) - pd.Timedelta(days=EDGE_TOLERANCE_DAYS)
    ):
        # The old Sina fallback ignored requested ranges and returned all history.
        full_history_source = True
    result = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if full_history_source:
        result.attrs["history_complete_left"] = True
        result.attrs["provider_min_date"] = provider_min_date.strftime("%Y-%m-%d")
    return result


def _read_known_inception(
    code: str,
    product_type: str,
    cache_root: Path,
) -> pd.Timestamp | None:
    path = metadata_path(code, product_type, cache_root)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("known_inception_date")
        return pd.to_datetime(value) if value else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _edge_coverage(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    known_inception_date: pd.Timestamp | None = None,
) -> tuple[bool, bool]:
    if frame.empty:
        return False, False
    minimum = pd.to_datetime(frame["date"]).min()
    maximum = pd.to_datetime(frame["date"]).max()
    start_ok = minimum <= start
    if (
        not start_ok
        and known_inception_date is not None
        and known_inception_date > start
        and minimum <= known_inception_date + pd.Timedelta(days=EDGE_TOLERANCE_DAYS)
    ):
        start_ok = True
    end_ok = maximum >= end
    return start_ok, end_ok


def _normalize_business_edges(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    normalized_start = pd.Timestamp(start).normalize()
    normalized_end = pd.Timestamp(end).normalize()
    if normalized_start.weekday() >= 5:
        normalized_start = (normalized_start + pd.offsets.BDay(1)).normalize()
    if normalized_end.weekday() >= 5:
        normalized_end = (normalized_end - pd.offsets.BDay(1)).normalize()
    return normalized_start, normalized_end


def load_history(
    code: str,
    start: str,
    end: str,
    product_type: str = "ETF",
    *,
    network_fetcher: NetworkFetcher | None = None,
    refresh: bool = False,
    offline: bool = False,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    legacy_root: Path = LEGACY_CACHE_ROOT,
) -> pd.DataFrame:
    """Load a requested range locally and fetch only uncovered edges."""
    requested_start = pd.to_datetime(start)
    requested_end = pd.to_datetime(end)
    if requested_start > requested_end:
        raise ValueError(f"start {start} is after end {end}")
    effective_end = min(requested_end, latest_completed_business_day())
    coverage_start, effective_end = _normalize_business_edges(requested_start, effective_end)
    path = history_path(code, product_type, cache_root)
    stored = _read_csv(path) if path.exists() else pd.DataFrame()
    known_inception_date = _read_known_inception(code, product_type, Path(cache_root))
    metadata_dirty = False
    start_ok, end_ok = _edge_coverage(
        stored,
        coverage_start,
        effective_end,
        known_inception_date,
    )

    if not refresh and not (start_ok and end_ok):
        legacy = _best_legacy_frame(
            code,
            product_type,
            requested_start,
            effective_end,
            legacy_root,
        )
        if not legacy.empty:
            if legacy.attrs.get("history_complete_left") and legacy.attrs.get("provider_min_date"):
                known_inception_date = pd.to_datetime(legacy.attrs["provider_min_date"])
                metadata_dirty = True
            stored = merge_history(stored, legacy)
            _atomic_write_csv(stored, path)
            _write_metadata(
                code,
                product_type,
                stored,
                Path(cache_root),
                known_inception_date,
            )

    start_ok, end_ok = _edge_coverage(
        stored,
        coverage_start,
        effective_end,
        known_inception_date,
    )
    network_calls = 0
    fetch_errors: list[str] = []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if refresh:
        ranges = [(requested_start, effective_end)]
    elif not start_ok:
        missing_end = (
            pd.to_datetime(stored["date"]).min() - pd.Timedelta(days=1)
            if not stored.empty
            else effective_end
        )
        ranges.append((requested_start, min(missing_end, effective_end)))
    if not refresh and not end_ok and not stored.empty:
        missing_start = pd.to_datetime(stored["date"]).max() + pd.Timedelta(days=1)
        ranges.append((max(missing_start, requested_start), effective_end))

    if not offline and network_fetcher is not None:
        for fetch_start, fetch_end in ranges:
            if fetch_start > fetch_end:
                continue
            network_calls += 1
            try:
                fetched = network_fetcher(
                    code,
                    fetch_start.strftime("%Y%m%d"),
                    fetch_end.strftime("%Y%m%d"),
                    _safe_product(product_type),
                )
            except Exception as exc:  # provider failures must not destroy valid local data
                fetch_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if fetched is not None and fetched.attrs.get("history_complete_left"):
                provider_min_date = fetched.attrs.get("provider_min_date")
                if provider_min_date:
                    known_inception_date = pd.to_datetime(provider_min_date)
                    metadata_dirty = True
            prepared = prepare_history(
                fetched,
                source=str(fetched.attrs.get("data_source", "network"))
                if fetched is not None
                else "network",
            )
            if prepared.empty:
                error = fetched.attrs.get("fetch_error") if fetched is not None else "empty response"
                fetch_errors.append(str(error or "empty response"))
                continue
            prepared = prepared[
                (prepared["date"] >= fetch_start) & (prepared["date"] <= fetch_end)
            ]
            stored = merge_history(stored, prepared)

    if not stored.empty and (network_calls or metadata_dirty or not path.exists()):
        _atomic_write_csv(stored, path)
        _write_metadata(
            code,
            product_type,
            stored,
            Path(cache_root),
            known_inception_date,
        )

    result = stored[
        (stored["date"] >= requested_start) & (stored["date"] <= effective_end)
    ].copy() if not stored.empty else pd.DataFrame()
    final_start_ok, final_end_ok = _edge_coverage(
        result,
        coverage_start,
        effective_end,
        known_inception_date,
    )
    result.attrs.update({
        "cache_path": str(path),
        "cache_hit": bool(start_ok and end_ok and not refresh),
        "network_calls": network_calls,
        "coverage_start_ok": final_start_ok,
        "coverage_end_ok": final_end_ok,
        "effective_end": effective_end.strftime("%Y-%m-%d"),
        "fetch_errors": fetch_errors,
        "offline": offline,
    })
    if result.empty and fetch_errors:
        result.attrs["fetch_error"] = "; ".join(fetch_errors)
    return result


def build_catalog(cache_root: Path = DEFAULT_CACHE_ROOT) -> pd.DataFrame:
    records = []
    for path in sorted((Path(cache_root) / "metadata").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            "code": payload.get("code"),
            "product_type": payload.get("product_type"),
            "rows": payload.get("rows"),
            "min_date": payload.get("min_date"),
            "max_date": payload.get("max_date"),
            "column_count": len(payload.get("columns") or []),
            "raw_optional_fields": ",".join(payload.get("raw_optional_fields") or []),
            "sources": ",".join(payload.get("sources") or []),
            "missing_core_values": payload.get("missing_core_values"),
            "duplicate_dates": payload.get("duplicate_dates"),
            "updated_at": payload.get("updated_at"),
            "sha256": payload.get("sha256"),
        })
    catalog = pd.DataFrame(records)
    if not catalog.empty:
        catalog = catalog.sort_values(["code", "product_type"]).reset_index(drop=True)
        _atomic_write_table(catalog, Path(cache_root) / "catalog.csv")
    return catalog
