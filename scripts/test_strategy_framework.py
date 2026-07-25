"""Regression tests for the shared selection and backtest policy."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from backtest import _calc_atr, _classify_loss_row, _should_hold_position  # noqa: E402
from backtest_current import ETFData, _market_state_from_proxy, run_backtest  # noqa: E402
from etf_analyzer import pick_formal_portfolio  # noqa: E402
from event_risk import Event, event_snapshot  # noqa: E402
from risk_rules import (  # noqa: E402
    CORE_ENTRY_SCORE,
    FALLBACK_ENTRY_SCORE,
    MAX_SAME_RISK_CLUSTER,
    TARGET_SELECTION_COUNT,
    allocate_ranked_weights,
    get_position_cap,
    new_position_trend_gate,
    risk_cluster,
)
from selection_guard import validate_selection  # noqa: E402


def _selection_text(weights: tuple[float, float, float], cash: float = 10.0) -> str:
    return f"""# Selection

调仓频率：周度
数据截止时间：2026-01-09 15:00
下次常规重排：2026-01-16
市场状态：震荡

## 国际事件复核

新闻截止时间：2026-01-09 20:00
国际事件风险：正常
国际事件动作：不否决现有价格信号

| 标的 | 方向 | 计划仓位 | 证据 |
|---|---|---:|---|
| 半导体设备ETF `159516` | 半导体设备 | {weights[0]}% | 政策催化待落地 |
| 通信ETF `515880` | 通信 | {weights[1]}% | 订单催化待确认 |
| 创新药ETF `515120` | 创新药 | {weights[2]}% | 获批催化待发生 |
| 现金 | — | {cash}% | 风险预算 |
"""


class RiskPolicyTests(unittest.TestCase):
    def test_three_distinct_rank_weights_fill_target(self) -> None:
        self.assertEqual(TARGET_SELECTION_COUNT, 3)
        for state in ("主升", "震荡", "退潮", "退潮末期", "冰点", "未知"):
            weights = allocate_ranked_weights(state)
            self.assertEqual(len(weights), 3)
            self.assertEqual(len(set(weights)), 3)
            self.assertAlmostEqual(sum(weights), get_position_cap(state).high)

    def test_low_etf_width_does_not_override_rising_benchmark(self) -> None:
        benchmark_nav = [1.0 + i * 0.01 for i in range(20)]
        state = _market_state_from_proxy(benchmark_nav, width=0.2, bench_pct=0.5)
        self.assertNotIn(state, {"退潮", "退潮末期", "冰点"})

    def test_negative_medium_trend_needs_strong_reversal_evidence(self) -> None:
        ok, _ = new_position_trend_gate(
            category="军工",
            market_state="主升",
            score=52.7,
            ret_20d=13.4,
            ret_60d=-3.5,
        )
        self.assertFalse(ok)
        ok, _ = new_position_trend_gate(
            category="军工",
            market_state="主升",
            score=62.0,
            ret_20d=18.0,
            ret_60d=-3.5,
        )
        self.assertTrue(ok)

    def test_growth_proxies_share_one_risk_cluster(self) -> None:
        clusters = {
            risk_cluster("科技", "半导体", "半导体ETF"),
            risk_cluster("军工", "机械", "工业母机ETF"),
            risk_cluster("宽基", "宽基", "科创50ETF"),
        }
        self.assertEqual(clusters, {"高弹性成长"})
        self.assertEqual(MAX_SAME_RISK_CLUSTER, 2)

    def test_scanner_fills_three_from_its_fixed_input_pool(self) -> None:
        def candidate(code: str, industry: str, stage: str, score: float) -> dict:
            return {
                "etf_code": code,
                "etf_name": f"ETF-{code}",
                "direction": industry,
                "industry": industry,
                "category": industry,
                "is_qdii": False,
                "stage": (stage, "", ""),
                "score": {"total": score},
                "metrics": {
                    "amount_yi": 5.0,
                    "pct_chg": 0.5,
                    "ret_5d": 1.0,
                    "overheat": False,
                },
            }

        source = [
            candidate("159516", "科技", "扩散期", 60.0),
            candidate("515880", "科技", "观察期", 42.0),
            candidate("515120", "医药", "休眠期", 35.0),
        ]
        selected, _ = pick_formal_portfolio(source)
        self.assertEqual(len(selected), 3)
        self.assertTrue({row["etf_code"] for row in selected}.issubset(
            {row["etf_code"] for row in source}
        ))

    def test_live_scanner_uses_shared_core_and_fallback_score_floors(self) -> None:
        self.assertEqual(CORE_ENTRY_SCORE, 45.0)
        self.assertEqual(FALLBACK_ENTRY_SCORE, 30.0)

        def candidate(code: str, category: str, stage: str, score: float) -> dict:
            return {
                "etf_code": code,
                "etf_name": f"ETF-{code}",
                "direction": category,
                "industry": category,
                "category": category,
                "is_qdii": False,
                "stage": (stage, "", ""),
                "score": {"total": score},
                "metrics": {
                    "amount_yi": 5.0,
                    "pct_chg": 0.5,
                    "ret_5d": 1.0,
                    "ret_20d": 5.0,
                    "ret_60d": 5.0,
                    "overheat": False,
                },
            }

        source = [
            candidate("159516", "科技", "扩散期", 50.0),
            candidate("515120", "医药", "确认期", 46.0),
            candidate("513310", "跨境", "加速期", 10.5),
            candidate("516310", "金融", "休眠期", 40.0),
        ]
        selected, _ = pick_formal_portfolio(source, market_state="退潮")
        codes = {row["etf_code"] for row in selected}
        self.assertNotIn("513310", codes)
        self.assertIn("516310", codes)


class SelectionGuardTests(unittest.TestCase):
    def _validate(self, text: str) -> tuple[bool, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.md"
            path.write_text(text, encoding="utf-8")
            return validate_selection(path)

    def test_valid_three_position_plan_passes(self) -> None:
        ok, errors = self._validate(_selection_text((36.0, 31.5, 22.5)))
        self.assertTrue(ok, errors)

    def test_daily_formal_selection_is_rejected(self) -> None:
        text = _selection_text((36.0, 31.5, 22.5)).replace(
            "调仓频率：周度",
            "调仓频率：日度",
        )
        ok, errors = self._validate(text)
        self.assertFalse(ok)
        self.assertTrue(any("只允许周度调仓" in error for error in errors), errors)

    def test_equal_weights_are_rejected(self) -> None:
        ok, errors = self._validate(_selection_text((30.0, 30.0, 30.0)))
        self.assertFalse(ok)
        self.assertTrue(any("不能机械等权" in error for error in errors), errors)

    def test_non_ice_cash_above_40_is_rejected(self) -> None:
        ok, errors = self._validate(_selection_text((24.0, 21.0, 15.0), cash=40.0))
        self.assertFalse(ok)
        self.assertTrue(any("总仓位应在" in error for error in errors), errors)

    def test_instrument_outside_fixed_pool_is_rejected(self) -> None:
        text = _selection_text((36.0, 31.5, 22.5)).replace(
            "半导体设备ETF `159516`",
            "池外股票 `600001`",
        )
        ok, errors = self._validate(text)
        self.assertFalse(ok)
        self.assertTrue(any("不在 scripts/etf.txt" in error for error in errors), errors)

    def test_midweek_formal_selection_needs_holiday_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-07-22" / "selection.md"
            path.parent.mkdir()
            path.write_text(_selection_text((36.0, 31.5, 22.5)), encoding="utf-8")
            ok, errors = validate_selection(path)
        self.assertFalse(ok)
        self.assertTrue(any("节假日特殊调仓" in error for error in errors), errors)

    def test_red_event_risk_rejects_new_high_beta_position(self) -> None:
        text = _selection_text((36.0, 31.5, 22.5)).replace(
            "国际事件风险：正常",
            "国际事件风险：红色",
        )
        ok, errors = self._validate(text)
        self.assertFalse(ok)
        self.assertTrue(any("禁止新开高弹性仓位" in error for error in errors), errors)


class PointInTimeTests(unittest.TestCase):
    def test_close_signal_executes_at_next_open(self) -> None:
        rows = [
            ETFData(
                code=code,
                name=code,
                sector=sector,
                industry=sector,
                stage="扩散期",
                score=score,
                amount_yi=5.0,
                pct=1.0,
            )
            for code, sector, score in (
                ("159516", "科技", 60.0),
                ("515120", "医药", 55.0),
                ("512400", "资源", 50.0),
            )
        ]
        scans = {
            date: {
                "bench": 0.0,
                "rows": rows,
                "by_code": {row.code: row for row in rows},
            }
            for date in ("2026-01-05", "2026-01-06")
        }
        prices = {
            code: {"2026-01-05": 50.0, "2026-01-06": 110.0}
            for code in ("159516", "515120", "512400")
        }
        prices["510300"] = {"2026-01-05": 100.0, "2026-01-06": 100.0}
        opens = {
            code: {"2026-01-05": 50.0, "2026-01-06": 100.0}
            for code in ("159516", "515120", "512400")
        }
        opens["510300"] = {"2026-01-05": 100.0, "2026-01-06": 100.0}

        log = run_backtest(
            scans,
            prices,
            ["2026-01-05", "2026-01-06"],
            opens=opens,
            verbose=False,
        )

        self.assertAlmostEqual(log[0]["net_return"], 0.0)
        self.assertEqual(log[0]["executed_holdings"], "空仓")
        self.assertAlmostEqual(log[1]["intraday_return"], 9.0)
        expected = (1 - 0.072 / 100) * (1 + 9.0 / 100) - 1
        self.assertAlmostEqual(log[1]["net_return"], expected * 100)

    def test_first_day_market_state_uses_benchmark_warmup(self) -> None:
        signal_date = "2026-01-30"
        rows = [
            ETFData(
                code=code,
                name=code,
                sector=sector,
                industry=sector,
                stage="扩散期",
                score=score,
                amount_yi=5.0,
                pct=1.0,
            )
            for code, sector, score in (
                ("159516", "科技", 60.0),
                ("515120", "医药", 55.0),
                ("512400", "资源", 50.0),
            )
        ]
        scans = {
            signal_date: {
                "bench": 0.0,
                "rows": rows,
                "by_code": {row.code: row for row in rows},
            }
        }
        benchmark_dates = pd.date_range("2026-01-01", periods=30, freq="D")
        prices = {
            code: {signal_date: 100.0}
            for code in ("159516", "515120", "512400")
        }
        prices["510300"] = {
            date.strftime("%Y-%m-%d"): 130.0 - index
            for index, date in enumerate(benchmark_dates)
        }

        log = run_backtest(
            scans,
            prices,
            [signal_date],
            opens=prices,
            verbose=False,
        )

        self.assertEqual(log[0]["market_state"], "退潮末期")

    def test_atr_ignores_rows_after_signal_date(self) -> None:
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        base = pd.DataFrame({
            "date": dates,
            "high": [11.0] * 30,
            "low": [9.0] * 30,
            "close": [10.0] * 30,
        })
        as_of = dates[19]
        before = _calc_atr(base, as_of)
        changed = base.copy()
        changed.loc[changed["date"] > as_of, ["high", "low"]] = [1000.0, 0.1]
        after = _calc_atr(changed, as_of)
        self.assertAlmostEqual(before, after)

    def test_positive_trend_is_not_sold_for_observation_stage_alone(self) -> None:
        dates = pd.date_range("2026-01-01", periods=40, freq="D")
        closes = [10.0 + i * 0.1 for i in range(40)]
        history = pd.DataFrame({
            "date": dates,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
        })
        current = {
            "etf_code": "159516",
            "signal_date": "2026-01-01",
            "stage": ("观察期", "", ""),
            "metrics": {"ret_20d": 8.0, "ret_60d": 12.0},
            "score": {"total": 42.0},
            "_rank": 10,
            "category": "科技",
            "industry": "科技",
            "etf_name": "半导体设备ETF",
        }
        previous = {
            "etf_code": "159516",
            "holding_since": "2026-01-01",
            "peak_score": 45.0,
        }
        keep, reason, _ = _should_hold_position(
            previous,
            {"159516": current},
            {"159516": history},
            dates[-1],
        )
        self.assertTrue(keep, reason)


class BacktestReportTests(unittest.TestCase):
    def test_loss_attribution_does_not_blame_market_when_benchmark_is_flat(self) -> None:
        row = pd.Series({
            "return_pct": -3.48,
            "benchmark_pct": -0.16,
            "risk_flags": "2只标的同周止损",
        })
        conclusion, responsibility = _classify_loss_row(row)
        self.assertEqual(conclusion, "选股或行业暴露为主")
        self.assertIn("框架主要责任", responsibility)

    def test_loss_attribution_marks_benchmark_led_selloff(self) -> None:
        row = pd.Series({
            "return_pct": -4.03,
            "benchmark_pct": -4.97,
            "risk_flags": "—",
        })
        conclusion, responsibility = _classify_loss_row(row)
        self.assertEqual(conclusion, "市场下跌为主")
        self.assertIn("组合仍跑赢基准", responsibility)


class EventRiskTests(unittest.TestCase):
    def test_future_event_is_not_visible(self) -> None:
        events = [
            Event(
                known_at=pd.Timestamp("2026-01-10 20:00").to_pydatetime(),
                event_type="macro",
                scope="global",
                score=-1.0,
                horizon_days=5,
                source="test",
                url="",
                summary="future",
            )
        ]
        snapshot = event_snapshot(
            events,
            pd.Timestamp("2026-01-10 15:00").to_pydatetime(),
        )
        self.assertEqual(snapshot["global_score"], 0.0)

    def test_single_media_report_cannot_turn_global_risk_red(self) -> None:
        events = [
            Event(
                known_at=pd.Timestamp("2026-01-10 14:00").to_pydatetime(),
                event_type="macro",
                scope="global",
                score=-0.8,
                horizon_days=5,
                source="test",
                url="",
                summary="known",
            )
        ]
        snapshot = event_snapshot(
            events,
            pd.Timestamp("2026-01-10 15:00").to_pydatetime(),
        )
        self.assertEqual(snapshot["risk_level"], "黄色")

    def test_two_independent_global_sources_turn_risk_red(self) -> None:
        events = [
            Event(
                known_at=pd.Timestamp(f"2026-01-10 {hour}:00").to_pydatetime(),
                event_type="macro",
                scope="global",
                score=-0.45,
                horizon_days=5,
                source=source,
                url="",
                summary=source,
            )
            for hour, source in ((13, "source-a"), (14, "source-b"))
        ]
        snapshot = event_snapshot(
            events,
            pd.Timestamp("2026-01-10 15:00").to_pydatetime(),
        )
        self.assertEqual(snapshot["risk_level"], "红色")

    def test_positive_news_does_not_cancel_negative_risk(self) -> None:
        events = [
            Event(
                known_at=pd.Timestamp("2026-01-10 13:00").to_pydatetime(),
                event_type="macro",
                scope="高弹性成长",
                score=-0.6,
                horizon_days=5,
                source="risk-source",
                url="",
                summary="risk",
            ),
            Event(
                known_at=pd.Timestamp("2026-01-10 14:00").to_pydatetime(),
                event_type="industry",
                scope="高弹性成长",
                score=0.8,
                horizon_days=5,
                source="positive-source",
                url="",
                summary="positive",
            ),
        ]
        snapshot = event_snapshot(
            events,
            pd.Timestamp("2026-01-10 15:00").to_pydatetime(),
        )
        self.assertLessEqual(snapshot["risk_scores"]["高弹性成长"], -0.55)
        self.assertGreater(snapshot["confirmation_scores"]["高弹性成长"], 0)


if __name__ == "__main__":
    unittest.main()
