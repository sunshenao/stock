"""Prefetch and audit one year of local ETF/LOF market data."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from etf_analyzer import fetch_etf_hist, load_etf_txt  # noqa: E402
from market_data_cache import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    build_catalog,
    latest_completed_business_day,
)


def _download_spot_snapshot(codes: set[str], cache_root: Path) -> tuple[Path | None, str]:
    try:
        import akshare as ak

        frames = []
        for product, fetcher in (
            ("ETF", ak.fund_etf_spot_em),
            ("LOF", ak.fund_lof_spot_em),
        ):
            frame = fetcher()
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["产品类型"] = product
            frames.append(frame)
        if not frames:
            return None, "实时快照接口返回空数据"
        snapshot = pd.concat(frames, ignore_index=True)
        code_column = next(
            (column for column in ("代码", "基金代码", "code") if column in snapshot),
            None,
        )
        if code_column:
            snapshot[code_column] = snapshot[code_column].astype(str).str.zfill(6)
            snapshot = snapshot[snapshot[code_column].isin(codes)]
        snapshot_dir = cache_root / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        dated_path = snapshot_dir / f"spot_{pd.Timestamp.now():%Y-%m-%d}.csv"
        latest_path = snapshot_dir / "spot_latest.csv"
        snapshot.to_csv(dated_path, index=False, encoding="utf-8-sig")
        snapshot.to_csv(latest_path, index=False, encoding="utf-8-sig")
        return dated_path, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _write_quality_outputs(
    rows: list[dict],
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    cache_root: Path,
    spot_path: Path | None,
    spot_error: str,
) -> tuple[Path, Path]:
    cache_root.mkdir(parents=True, exist_ok=True)
    quality = pd.DataFrame(rows).sort_values(["status", "code"]).reset_index(drop=True)
    csv_path = cache_root / "quality_report.csv"
    quality.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = quality["status"].value_counts().to_dict()
    payload = {
        "schema_version": 2,
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "requested_end": requested_end.strftime("%Y-%m-%d"),
        "instrument_count": int(len(quality)),
        "status_counts": {str(key): int(value) for key, value in summary.items()},
        "total_rows": int(quality["rows"].sum()),
        "network_calls": int(quality["network_calls"].sum()),
        "cache_hits": int(quality["cache_hit"].sum()),
        "spot_snapshot": str(spot_path) if spot_path else None,
        "spot_error": spot_error or None,
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    json_path = cache_root / "quality_summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = quality[quality["status"].isin({"missing", "stale", "invalid"})]
    inception_limited = quality[quality["status"] == "inception_limited"]
    lines = [
        "# ETF/LOF 本地行情质量报告",
        "",
        f"- 请求区间：{requested_start:%Y-%m-%d} ~ {requested_end:%Y-%m-%d}",
        f"- 固定池标的：{len(quality)}",
        f"- 本地数据行：{int(quality['rows'].sum())}",
        f"- 本次网络请求：{int(quality['network_calls'].sum())}",
        f"- 完整覆盖：{int((quality['status'] == 'ok').sum())}",
        f"- 成立时间不足一年：{len(inception_limited)}",
        f"- 异常标的：{len(failures)}",
        f"- 行情字段数：{int(quality['column_count'].max()) if not quality.empty else 0}",
        f"- 实时快照：{spot_path.name if spot_path else '未获取'}",
    ]
    if spot_error:
        lines.append(f"- 实时快照错误：{spot_error}")
    lines.extend([
        "",
        "## 字段",
        "",
        "除开高低收、成交量和成交额外，本地文件保存可获得的换手率、振幅、涨跌额、盘后量/额，"
        "并计算1/5/10/20/60/120/250日收益、均线、ATR、RSI、年化波动率、量额比、"
        "滚动高低点、价格位置、回撤、OBV和Amihud非流动性指标。",
        "",
        "所有派生指标只使用当日及以前数据。",
        "",
        "## 异常",
        "",
        failures.to_markdown(index=False) if not failures.empty else "无。",
        "",
        "## 成立不足一年",
        "",
        inception_limited[["code", "name", "min_date", "max_date", "rows"]].to_markdown(index=False)
        if not inception_limited.empty
        else "无。",
        "",
    ])
    md_path = cache_root / "quality_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="预下载 etf.txt 全池本地行情")
    parser.add_argument("--days", type=int, default=370, help="预热自然日数，默认370天")
    parser.add_argument("--start", help="开始日期 YYYY-MM-DD，优先于 --days")
    parser.add_argument("--end", help="结束日期 YYYY-MM-DD，默认最近完整交易日")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh-network", action="store_true", help="即使本地已覆盖也刷新网络")
    parser.add_argument("--offline", action="store_true", help="只整合现有本地数据，禁止网络")
    parser.add_argument("--spot-snapshot", action="store_true", help="额外尝试获取ETF/LOF实时全字段快照")
    args = parser.parse_args()

    end = pd.to_datetime(args.end) if args.end else latest_completed_business_day()
    start = pd.to_datetime(args.start) if args.start else end - timedelta(days=max(args.days, 1))
    pool = load_etf_txt()
    total = len(pool)
    rows = []

    def fetch_one(item: tuple[str, dict]) -> dict:
        direction, cfg = item
        frame = fetch_etf_hist(
            cfg["code"],
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            cfg.get("type", "ETF"),
            refresh_cache=args.refresh_network,
            offline=args.offline,
        )
        min_date = pd.to_datetime(frame["date"]).min() if not frame.empty else pd.NaT
        max_date = pd.to_datetime(frame["date"]).max() if not frame.empty else pd.NaT
        if frame.empty:
            status = "missing"
        elif not bool(frame.attrs.get("coverage_end_ok", False)):
            status = "stale"
        elif min_date > start + pd.Timedelta(days=7):
            status = "inception_limited"
        elif frame[["open", "high", "low", "close"]].isna().any().any():
            status = "invalid"
        else:
            status = "ok"
        return {
            "code": cfg["code"],
            "name": cfg["name"],
            "direction": direction,
            "category": cfg.get("cat", ""),
            "industry": cfg.get("ind", ""),
            "product_type": cfg.get("type", "ETF"),
            "status": status,
            "rows": int(len(frame)),
            "min_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "",
            "max_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "",
            "column_count": int(len(frame.columns)),
            "cache_hit": bool(frame.attrs.get("cache_hit", False)),
            "network_calls": int(frame.attrs.get("network_calls", 0)),
            "coverage_start_ok": bool(frame.attrs.get("coverage_start_ok", False)),
            "coverage_end_ok": bool(frame.attrs.get("coverage_end_ok", False)),
            "fetch_errors": " | ".join(frame.attrs.get("fetch_errors") or []),
        }

    workers = max(1, min(args.workers, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_one, item): item[1]["code"]
            for item in pool.items()
        }
        for index, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({
                    "code": code,
                    "name": code,
                    "direction": "",
                    "category": "",
                    "industry": "",
                    "product_type": "",
                    "status": "missing",
                    "rows": 0,
                    "min_date": "",
                    "max_date": "",
                    "column_count": 0,
                    "cache_hit": False,
                    "network_calls": 0,
                    "coverage_start_ok": False,
                    "coverage_end_ok": False,
                    "fetch_errors": f"{type(exc).__name__}: {exc}",
                })
            if index % 20 == 0 or index == total:
                print(f"  已处理 {index}/{total}", flush=True)

    build_catalog(DEFAULT_CACHE_ROOT)
    spot_path = None
    spot_error = ""
    if args.spot_snapshot and not args.offline:
        spot_path, spot_error = _download_spot_snapshot(
            {cfg["code"] for cfg in pool.values()},
            DEFAULT_CACHE_ROOT,
        )
    csv_path, md_path = _write_quality_outputs(
        rows,
        requested_start=start,
        requested_end=end,
        cache_root=DEFAULT_CACHE_ROOT,
        spot_path=spot_path,
        spot_error=spot_error,
    )
    statuses = pd.DataFrame(rows)["status"].value_counts().to_dict()
    print(f"完成: {statuses}")
    print(f"质量明细: {csv_path}")
    print(f"质量报告: {md_path}")


if __name__ == "__main__":
    main()
