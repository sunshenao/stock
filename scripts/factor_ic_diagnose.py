"""Point-in-time weekly factor IC diagnostics for the ETF rotation framework."""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from etf_analyzer import NEW_MONEY_STAGES, load_etf_txt
from risk_rules import score_layers_pass
from universe_history import active_codes, load_registry


FACTOR_COLUMNS = (
    "total",
    "mainline",
    "timing",
    "risk_penalty",
    "r1",
    "r5",
    "r20",
    "r60",
    "ex5",
    "ex20",
    "ex60",
    "beat10",
    "dist20",
    "dist60",
    "dominance5",
    "eff20",
    "vol20",
    "amount_persist",
)


def _forward_return(
    frame: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float | None:
    period = frame[
        (frame["date"] >= entry_date) & (frame["date"] <= exit_date)
    ].sort_values("date")
    if period.empty:
        return None
    entry_price = float(period.iloc[0].get("open", period.iloc[0]["close"]))
    exit_price = float(period.iloc[-1]["close"])
    if entry_price <= 0:
        return None
    return (exit_price / entry_price - 1) * 100


def _forward_horizon_returns(
    frame: pd.DataFrame,
    entry_date: pd.Timestamp,
) -> dict[str, float | None]:
    future = frame.loc[frame["date"] >= entry_date].sort_values("date")
    if future.empty:
        return {"fwd_5d": None, "fwd_10d": None, "fwd_20d": None}
    entry_price = float(future.iloc[0].get("open", future.iloc[0]["close"]))
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"fwd_5d": None, "fwd_10d": None, "fwd_20d": None}

    result: dict[str, float | None] = {}
    for label, bars in (("fwd_5d", 5), ("fwd_10d", 10), ("fwd_20d", 20)):
        if len(future) < bars:
            result[label] = None
            continue
        exit_price = float(future.iloc[bars - 1]["close"])
        result[label] = (
            (exit_price / entry_price - 1) * 100
            if np.isfinite(exit_price) and exit_price > 0
            else None
        )
    return result


def build_panel(
    start_date: str,
    end_date: str,
    *,
    split_date: str,
    offline: bool,
) -> pd.DataFrame:
    pool = load_etf_txt()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    fetch_start = (start - timedelta(days=120)).strftime("%Y%m%d")
    fetch_end = (end + timedelta(days=5)).strftime("%Y%m%d")
    benchmark_cfg = next(
        item for item in pool.values() if item["code"] == bt.BENCHMARK_CODE
    )
    benchmark = bt._fetch_hist_cached(
        bt.BENCHMARK_CODE,
        fetch_start,
        fetch_end,
        benchmark_cfg.get("type", "ETF"),
        False,
        offline,
    )
    data = bt.fetch_all_hist(
        pool,
        fetch_start,
        fetch_end,
        offline=offline,
    )
    registry = load_registry()
    cutoff = pd.Timestamp(split_date)
    rows: list[dict] = []

    for label, period_start, period_end in bt._periods(start, end, "weekly"):
        signal_date = bt._latest_before(benchmark, period_start)
        entry_date = bt._first_between(benchmark, period_start, period_end)
        exit_date = bt._last_between(benchmark, period_start, period_end)
        if (
            signal_date is None
            or entry_date is None
            or exit_date is None
            or entry_date >= exit_date
        ):
            continue
        market_state, _ = bt._simple_market_state(benchmark, signal_date, data)
        ranked = bt.rank_on_signal_date(
            pool,
            data,
            benchmark,
            signal_date,
            1.0,
            allowed_stages=None,
            allowed_codes=active_codes(registry, signal_date),
        )
        for item in ranked:
            forward = _forward_return(
                data[item["etf_code"]],
                entry_date,
                exit_date,
            )
            if forward is None:
                continue
            horizon_returns = _forward_horizon_returns(
                data[item["etf_code"]],
                entry_date,
            )
            metrics = item["metrics"]
            score = item["score"]
            layers_pass, _ = score_layers_pass(score, market_state)
            rows.append({
                "period": label,
                "signal_date": signal_date,
                # The forward outcome of the 2026-W19 signal is realized in May,
                # so a signal exactly on the cutoff belongs to the test sample.
                "split": "train" if signal_date < cutoff else "test",
                "market_state": market_state,
                "code": item["etf_code"],
                "forward_return": forward,
                **horizon_returns,
                "new_money_stage": item["stage"][0] in NEW_MONEY_STAGES,
                "layers_pass": layers_pass,
                "total": score["total"],
                "mainline": score["mainline"],
                "timing": score["timing"],
                "risk_penalty": score["risk_penalty"],
                "r1": metrics["pct_chg"],
                "r5": metrics["ret_5d"],
                "r20": metrics["ret_20d"],
                "r60": metrics["ret_60d"],
                "ex5": metrics["excess_5d"],
                "ex20": metrics["excess_20d"],
                "ex60": metrics["excess_60d"],
                "beat10": metrics["beat_days_10"],
                "dist20": metrics["distance_ma20"],
                "dist60": metrics["distance_ma60"],
                "dominance5": metrics["one_day_dominance_5"],
                "eff20": metrics["trend_efficiency_20"],
                "vol20": metrics["volatility_20"],
                "amount_persist": metrics["amount_persistence_3d"],
            })
    return pd.DataFrame(rows)


def factor_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, split_frame in panel.groupby("split"):
        for scope, scoped in (
            ("all_liquid", split_frame),
            ("new_money_stage", split_frame[split_frame["new_money_stage"]]),
            (
                "layers_pass",
                split_frame[
                    split_frame["new_money_stage"] & split_frame["layers_pass"]
                ],
            ),
        ):
            for horizon in ("forward_return", "fwd_5d", "fwd_10d", "fwd_20d"):
                for factor in FACTOR_COLUMNS:
                    ic_values = []
                    quantile_returns = []
                    for _, group in scoped.groupby("period"):
                        clean = group[[factor, horizon]].dropna()
                        if len(clean) < 10 or clean[factor].nunique() < 3:
                            continue
                        ic_values.append(
                            clean[factor].corr(
                                clean[horizon],
                                method="spearman",
                            )
                        )
                        ranks = clean[factor].rank(method="first", pct=True)
                        quantiles = pd.cut(
                            ranks,
                            [0, 0.2, 0.4, 0.6, 0.8, 1],
                            labels=[1, 2, 3, 4, 5],
                            include_lowest=True,
                        )
                        quantile_returns.append(
                            clean.assign(quantile=quantiles)
                            .groupby("quantile", observed=False)[horizon]
                            .mean()
                        )
                    if not ic_values:
                        continue
                    ic_array = np.asarray(ic_values, dtype=float)
                    qmean = pd.DataFrame(quantile_returns).mean()
                    ic_std = float(np.nanstd(ic_array))
                    rows.append({
                        "split": split,
                        "scope": scope,
                        "horizon": horizon,
                        "factor": factor,
                        "ic_mean": float(np.nanmean(ic_array)),
                        "ic_std": ic_std,
                        "ic_ir": (
                            float(np.nanmean(ic_array)) / ic_std
                            if ic_std > 0
                            else np.nan
                        ),
                        "ic_positive_rate": float(np.nanmean(ic_array > 0)),
                        "q1_return": float(qmean.get(1, np.nan)),
                        "q5_return": float(qmean.get(5, np.nan)),
                        "q5_minus_q1": float(
                            qmean.get(5, np.nan) - qmean.get(1, np.nan)
                        ),
                        "periods": len(ic_array),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF weekly factor IC diagnosis")
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-07-26")
    parser.add_argument("--split", default="2026-04-30")
    parser.add_argument("--output-dir", default="check/factor_ic_v2.3.0")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = bt.PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_panel(
        args.start,
        args.end,
        split_date=args.split,
        offline=args.offline,
    )
    summary = factor_summary(panel)
    panel.to_csv(output_dir / "factor_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    print(f"panel_rows={len(panel)}, periods={panel['period'].nunique()}")
    for split in ("train", "test"):
        print(f"\n=== {split.upper()} / ALL LIQUID ===")
        shown = summary[
            (summary["split"] == split)
            & (summary["scope"] == "all_liquid")
            & (
                summary["factor"].isin(
                    [
                        "total",
                        "mainline",
                        "timing",
                        "r1",
                        "r20",
                        "r60",
                        "ex20",
                        "beat10",
                        "dist60",
                        "dominance5",
                    ]
                )
            )
        ].sort_values("ic_mean", ascending=False)
        print(shown.to_string(index=False))
    print(output_dir)


if __name__ == "__main__":
    main()
