"""Validate whether the weekly strategy is stable near its chosen hold period.

The script changes one parameter at a time, keeps every other strategy rule
fixed, and reports results for the full history plus pre/post split windows.
It is a robustness diagnostic, not an optimizer and not forward evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from strategy_version import load_current_manifest  # noqa: E402


def _max_drawdown(returns_pct: pd.Series) -> float:
    if returns_pct.empty:
        return 0.0
    nav = (1.0 + returns_pct.astype(float) / 100.0).cumprod()
    return float((nav / nav.cummax() - 1.0).min() * 100.0)


def _window_metrics(
    periods: pd.DataFrame,
    *,
    label: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    frame = periods.copy()
    entries = pd.to_datetime(frame["entry_date"])
    if start is not None:
        frame = frame[entries >= start]
        entries = entries.loc[frame.index]
    if end is not None:
        frame = frame[entries <= end]
    returns = frame["return_pct"].astype(float)
    benchmark = frame["benchmark_pct"].astype(float)
    if frame.empty:
        return {
            "window": label,
            "weeks": 0,
            "return_pct": 0.0,
            "benchmark_pct": 0.0,
            "excess_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "best_week_pct": 0.0,
            "return_without_best_week_pct": 0.0,
            "best_positive_week_share_pct": 0.0,
            "turnover_multiple": 0.0,
            "average_exposure_pct": 0.0,
        }

    total = (float((1.0 + returns / 100.0).prod()) - 1.0) * 100.0
    bench_total = (float((1.0 + benchmark / 100.0).prod()) - 1.0) * 100.0
    best_index = returns.idxmax()
    without_best = returns.drop(index=best_index)
    without_best_total = (
        float((1.0 + without_best / 100.0).prod()) - 1.0
        if not without_best.empty
        else 0.0
    ) * 100.0
    positive = returns[returns > 0]
    positive_share = (
        float(positive.max() / positive.sum() * 100.0)
        if not positive.empty and float(positive.sum()) > 0
        else 0.0
    )
    return {
        "window": label,
        "weeks": int(len(frame)),
        "return_pct": round(total, 2),
        "benchmark_pct": round(bench_total, 2),
        "excess_pct": round(total - bench_total, 2),
        "max_drawdown_pct": round(_max_drawdown(returns), 2),
        "win_rate_pct": round(float((returns > 0).mean() * 100.0), 1),
        "best_week_pct": round(float(returns.max()), 2),
        "return_without_best_week_pct": round(without_best_total, 2),
        "best_positive_week_share_pct": round(positive_share, 1),
        "turnover_multiple": round(float(frame["turnover_pct"].sum() / 100.0), 2),
        "average_exposure_pct": round(float(frame["exposure"].mean()), 1),
    }


def _run_hold_variant(payload: tuple[int, str, str, str, bool]) -> list[dict]:
    hold_days, start, end, split_date, offline = payload
    import backtest

    output = io.StringIO()
    with redirect_stdout(output):
        periods = backtest.run_backtest(
            start,
            end,
            freq="weekly",
            offline=offline,
            min_hold_trading_days=hold_days,
        )
    split = pd.Timestamp(split_date)
    windows = [
        _window_metrics(periods, label="full"),
        _window_metrics(periods, label="pre_split", end=split - pd.Timedelta(days=1)),
        _window_metrics(periods, label="post_split", start=split),
    ]
    for row in windows:
        row["min_hold_trading_days"] = hold_days
    return windows


def _write_report(
    results: pd.DataFrame,
    output: Path,
    *,
    start: str,
    end: str,
    split_date: str,
    baseline_hold_days: int,
) -> Path:
    strategy_version = str(
        (load_current_manifest() or {}).get("strategy_version") or "unfrozen"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")

    post = results[results["window"] == "post_split"].sort_values(
        "min_hold_trading_days"
    )
    pre = results[results["window"] == "pre_split"].sort_values(
        "min_hold_trading_days"
    )
    baseline = results[
        (results["window"] == "post_split")
        & (results["min_hold_trading_days"] == baseline_hold_days)
    ]
    adjacent = results[
        (results["window"] == "post_split")
        & (results["min_hold_trading_days"].isin([8, 12]))
    ]
    plateau = bool(
        len(adjacent) == 2
        and (adjacent["return_pct"] > 0).all()
        and (adjacent["excess_pct"] > 0).all()
    )
    concentration_warning = bool(
        not baseline.empty
        and float(baseline.iloc[0]["best_positive_week_share_pct"]) >= 30.0
    )
    verdict = (
        "不存在仅10日单点有效的迹象，但样本短且收益集中，仍属于高过拟合风险，不能视为已通过样本外验证。"
        if plateau
        else "邻近参数无法保持同方向表现，10日参数存在明显单点过拟合风险，应撤回或简化。"
    )

    lines = [
        f"# {strategy_version} 参数稳健性检查",
        "",
        f"- 连续回放区间：{start} 至 {end}",
        f"- 时间切分：{split_date}；切分前用于跨阶段复核，切分后已参与策略诊断，不属于真正样本外。",
        "- 切分点不清仓、不重置基准或持仓；因此切分后绝对收益与单独启动的正式回测不同，只用于参数间同口径比较。",
        "- 方法：只扰动最短持有期为5/8/10/12/15个交易日，其余评分、仓位、成本和风险规则固定。",
        f"- 邻域平台判断：{'通过' if plateau else '未通过'}。",
        f"- 收益集中警告：{'触发' if concentration_warning else '未触发'}。",
        f"- 结论：{verdict}",
        "",
        "## 5月后参数邻域",
        "",
        post.to_markdown(index=False),
        "",
        "## 5月前跨阶段复核",
        "",
        pre.to_markdown(index=False),
        "",
        "## 限制",
        "",
        "- 参数比较次数会增加数据窥探风险；本报告只用于否决脆弱参数，不能用于证明未来收益。",
        "- 当前ETF历史池质量为partial，仍存在首次快照以前的幸存者偏差。",
        "- 真正的样本外检验从冻结后的2026-07-27开始，核心参数至少26周不调整。",
        f"- 明细数据：`{csv_path.name}`",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    strategy_version = str(
        (load_current_manifest() or {}).get("strategy_version") or "unfrozen"
    )
    parser = argparse.ArgumentParser(description="Weekly strategy hold-period robustness")
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--split-date", default="2026-05-01")
    parser.add_argument("--hold-days", default="5,8,10,12,15")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "check"
            / f"robustness_v{strategy_version}_hold_period.md"
        ),
    )
    args = parser.parse_args()
    hold_days = sorted({int(value) for value in args.hold_days.split(",")})
    payloads = [
        (value, args.start, args.end, args.split_date, args.offline)
        for value in hold_days
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_run_hold_variant, payload): payload[0]
            for payload in payloads
        }
        for future in as_completed(futures):
            hold = futures[future]
            rows.extend(future.result())
            print(f"completed min_hold_trading_days={hold}", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["window", "min_hold_trading_days"]
    )
    path = _write_report(
        results,
        Path(args.output),
        start=args.start,
        end=args.end,
        split_date=args.split_date,
        baseline_hold_days=10,
    )
    print(path)


if __name__ == "__main__":
    main()
