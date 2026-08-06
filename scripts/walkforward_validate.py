"""Expanding-window temporal validation for the frozen weekly strategy.

This is a stability diagnostic, not a substitute for post-freeze forward
performance. Parameters are never tuned inside this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import PROJECT_ROOT, _max_drawdown, run_backtest
from strategy_version import load_current_manifest


def build_folds(
    periods: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    train_periods: int,
    test_periods: int,
) -> pd.DataFrame:
    rows = []
    fold = 1
    for test_start in range(train_periods, len(periods), test_periods):
        test_end = min(test_start + test_periods, len(periods))
        test = periods.iloc[test_start:test_end]
        if test.empty:
            continue
        labels = set(test["month"].astype(str))
        ordered_daily = daily_equity.reset_index(drop=True)
        test_mask = ordered_daily["period"].astype(str).isin(labels)
        test_daily = ordered_daily[test_mask].copy()
        returns = test["return_pct"].astype(float)
        nav = (1 + returns / 100).cumprod()
        benchmark_nav = (1 + test["benchmark_pct"].astype(float) / 100).cumprod()
        if not test_daily.empty:
            first_index = int(test_daily.index.min())
            prior_nav = float(ordered_daily.iloc[first_index - 1]["nav"]) if first_index > 0 else 1.0
            normalized_daily = test_daily["nav"].astype(float) / prior_nav
            max_dd = _max_drawdown(normalized_daily)
        else:
            max_dd = _max_drawdown(nav)
        rows.append({
            "fold": fold,
            "train_start": periods.iloc[0]["entry_date"],
            "train_end": periods.iloc[test_start - 1]["exit_date"],
            "test_start": test.iloc[0]["entry_date"],
            "test_end": test.iloc[-1]["exit_date"],
            "test_periods": len(test),
            "complete_fold": len(test) == test_periods,
            "return_pct": round((float(nav.iloc[-1]) - 1) * 100, 2),
            "benchmark_pct": round((float(benchmark_nav.iloc[-1]) - 1) * 100, 2),
            "excess_pct": round(
                (float(nav.iloc[-1]) - float(benchmark_nav.iloc[-1])) * 100,
                2,
            ),
            "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
            "max_daily_drawdown_pct": round(max_dd, 2),
            "parameter_changes": 0,
        })
        fold += 1
    return pd.DataFrame(rows)


def write_report(folds: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    folds.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest = load_current_manifest() or {}
    if folds.empty:
        body = "可用周数不足，尚不能形成滚动验证折。\n"
    else:
        complete = folds[folds["complete_fold"]].copy()
        positive = int((complete["return_pct"] > 0).sum())
        excess = int((complete["excess_pct"] > 0).sum())
        body = "\n".join([
            f"- 完整测试折：{len(complete)}；不完整观察窗：{len(folds) - len(complete)}",
            f"- 完整折正收益：{positive}/{len(complete)}",
            f"- 完整折跑赢沪深300：{excess}/{len(complete)}",
            (
                f"- 完整折最差收益：{complete['return_pct'].min():+.2f}%"
                if not complete.empty else "- 完整折最差收益：无"
            ),
            (
                f"- 完整折最差日度回撤：{complete['max_daily_drawdown_pct'].min():.2f}%"
                if not complete.empty else "- 完整折最差日度回撤：无"
            ),
            "",
            folds.to_markdown(index=False),
        ])
    lines = [
        "# 固定策略滚动时间验证",
        "",
        f"- 策略版本：{manifest.get('strategy_version', '未冻结')}",
        "- 规则：扩展训练窗、固定测试窗；本脚本不调参，parameter_changes 必须为0。",
        "- 定位：检查历史跨阶段稳定性，不等同于策略冻结后的真实样本外业绩。",
        f"- 明细：`{csv_path.name}`",
        "",
        body,
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly expanding-window validation")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--train-weeks", type=int, default=26)
    parser.add_argument("--test-weeks", type=int, default=13)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "check" / "walkforward_result.md"),
    )
    args = parser.parse_args()
    result = run_backtest(args.start, args.end, freq="weekly")
    folds = build_folds(
        result,
        result.attrs.get("daily_equity", pd.DataFrame()),
        train_periods=args.train_weeks,
        test_periods=args.test_weeks,
    )
    path = write_report(folds, Path(args.output))
    print(path)


if __name__ == "__main__":
    main()
