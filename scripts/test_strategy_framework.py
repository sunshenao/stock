"""Regression tests for the shared selection and backtest policy."""
from __future__ import annotations

import sys
import tempfile
import unittest
import json
import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from backtest import (  # noqa: E402
    _calc_atr,
    _classify_loss_row,
    _cost_sensitivity,
    _fetch_hist_cached,
    _finalize_daily_equity,
    _selection_priority,
    _should_hold_position,
    _trading_days_held,
)
from backtest_current import (  # noqa: E402
    ETFData,
    _market_state_from_proxy,
    _scaled_rank_weights,
    run_backtest,
)
from daily_risk import detect_cluster_crashes  # noqa: E402
from etf_analyzer import (  # noqa: E402
    _is_cross_border_or_qdii,
    _market_note,
    _portfolio_rank_key,
    _rank_key,
    calc_auto_score,
    calc_etf_metrics,
    pick_formal_portfolio,
)
from execution_model import simulate_long_with_stop  # noqa: E402
from event_risk import Event, event_snapshot  # noqa: E402
from market_data_cache import load_history, prepare_history  # noqa: E402
from risk_rules import (  # noqa: E402
    CORE_ENTRY_SCORE,
    MAX_NEW_POSITION_ONE_DAY_DOMINANCE,
    MAX_SAME_RISK_CLUSTER,
    TARGET_SELECTION_COUNT,
    allocate_instrument_weights,
    allocate_ranked_weights,
    category_root,
    get_position_cap,
    max_same_risk_cluster,
    max_selection_count,
    new_position_trend_gate,
    risk_cluster,
    score_layers_pass,
    single_position_cap,
)
from selection_guard import validate_selection  # noqa: E402
from strategy_version import build_manifest, verify_manifest  # noqa: E402
from universe_history import active_codes  # noqa: E402
from walkforward_validate import build_folds  # noqa: E402


def _selection_text(weights: tuple[float, float, float], cash: float = 10.0) -> str:
    return f"""# Selection

调仓频率：周度
数据截止时间：2026-01-09 15:00
下次常规重排：2026-01-16
市场状态：主升

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


def _cash_only_selection_text() -> str:
    return """# Selection

调仓频率：周度
数据截止时间：2026-01-09 15:00
下次常规重排：2026-01-16
市场状态：冰点

## 国际事件复核

新闻截止时间：2026-01-09 20:00
国际事件风险：正常
国际事件动作：维持空仓

| 标的 | 方向 | 计划仓位 | 证据 |
|---|---|---:|---|
| 现金 | — | 100% | 无合格标的 |
"""


class RiskPolicyTests(unittest.TestCase):
    def test_rank_ties_have_a_stable_code_tiebreaker(self) -> None:
        def candidate(code: str) -> dict:
            return {
                "etf_code": code,
                "direction": code,
                "industry": "科技",
                "metrics": {
                    "pct_chg": 1.0,
                    "ret_5d": 5.0,
                    "amount_yi": 2.0,
                },
                "stage": ("确认期", "", ""),
                "score": {"total": 70.0},
            }

        rows = [candidate("516000"), candidate("159900")]
        self.assertEqual(
            ["159900", "516000"],
            [row["etf_code"] for row in sorted(rows, key=_rank_key, reverse=True)],
        )
        self.assertEqual(
            ["159900", "516000"],
            [
                row["etf_code"]
                for row in sorted(rows, key=_portfolio_rank_key, reverse=True)
            ],
        )

    def test_new_position_requires_core_total_and_distributed_advance(self) -> None:
        score = {
            "total": CORE_ENTRY_SCORE,
            "mainline": 75.0,
            "timing": 65.0,
            "risk_penalty": 5.0,
            "one_day_dominance": MAX_NEW_POSITION_ONE_DAY_DOMINANCE,
            "mainline_pass": True,
            "timing_pass": True,
        }
        passed, reason = score_layers_pass(score, "震荡")
        self.assertTrue(passed, reason)

        weak_total = {**score, "total": CORE_ENTRY_SCORE - 0.1}
        passed, reason = score_layers_pass(weak_total, "震荡")
        self.assertFalse(passed)
        self.assertIn("总分", reason)

        one_day_spike = {
            **score,
            "one_day_dominance": MAX_NEW_POSITION_ONE_DAY_DOMINANCE + 0.01,
        }
        passed, reason = score_layers_pass(one_day_spike, "震荡")
        self.assertFalse(passed)
        self.assertIn("单日贡献", reason)

    def test_minimum_hold_protection_uses_trading_days(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=12)
        frame = pd.DataFrame({"date": dates, "close": range(12)})
        self.assertEqual(
            _trading_days_held(frame, dates[0], dates[7]),
            8,
        )
        protected = {
            "entry_type": "续持",
            "entry_tier": "持有",
            "minimum_hold_protected": True,
            "score": {"total": 70.0},
            "stage": ("确认期", "", ""),
            "metrics": {},
        }
        challenger = {
            "entry_type": "新开",
            "entry_tier": "核心",
            "score": {"total": 95.0},
            "stage": ("确认期", "", ""),
            "metrics": {},
        }
        self.assertGreater(
            _selection_priority(protected, "weekly"),
            _selection_priority(challenger, "weekly"),
        )

    def test_rank_weights_use_full_budget_for_actual_candidate_count(self) -> None:
        self.assertEqual(TARGET_SELECTION_COUNT, 3)
        for state in ("主升", "震荡", "退潮", "退潮末期", "冰点", "未知"):
            full = allocate_ranked_weights(state, 3)
            two = allocate_ranked_weights(state, 2)
            one = allocate_ranked_weights(state, 1)
            none = allocate_ranked_weights(state, 0)
            self.assertAlmostEqual(sum(full), get_position_cap(state).high)
            self.assertAlmostEqual(sum(two), get_position_cap(state).high)
            self.assertAlmostEqual(sum(one), get_position_cap(state).high)
            self.assertEqual(none, [])
        self.assertEqual(allocate_ranked_weights("主升", 1), [100.0])
        self.assertEqual(allocate_ranked_weights("退潮末期", 1), [80.0])

    def test_position_floors_and_single_instrument_caps_match_policy(self) -> None:
        for state in ("主升", "震荡", "退潮", "退潮末期", "未知"):
            self.assertGreaterEqual(get_position_cap(state).low, 60)
        self.assertEqual(get_position_cap("冰点").low, 0)
        self.assertEqual(single_position_cap("stock"), 100)
        self.assertEqual(single_position_cap("ETF"), 100)
        self.assertEqual(single_position_cap("LOF"), 100)
        self.assertEqual(single_position_cap("ETF", is_qdii=True), 25)

    def test_market_state_limits_position_count(self) -> None:
        self.assertEqual(max_selection_count("主升"), 3)
        self.assertEqual(max_selection_count("震荡"), 2)
        self.assertEqual(max_selection_count("退潮"), 2)
        self.assertEqual(max_selection_count("退潮末期"), 1)
        self.assertEqual(max_selection_count("冰点"), 1)
        self.assertEqual(max_same_risk_cluster("震荡"), 2)
        self.assertEqual(max_same_risk_cluster("退潮"), 1)
        self.assertEqual(category_root("债券/地方债"), "债券")

    def test_unknown_qdii_is_eligible_but_capped_at_fifteen_percent(self) -> None:
        weights = allocate_instrument_weights(
            "主升",
            [{"product_type": "ETF", "is_qdii": True, "premium_pct": None}],
        )
        self.assertEqual(weights, [15.0])
        replay_weights = _scaled_rank_weights(
            "主升",
            100.0,
            1,
            [ETFData(code="513350", is_qdii=True)],
        )
        self.assertEqual(replay_weights, [15.0])
        self.assertTrue(_is_cross_border_or_qdii("纳指ETF", "跨境/美股"))

    def test_low_etf_width_does_not_override_rising_benchmark(self) -> None:
        benchmark_nav = [1.0 + i * 0.01 for i in range(20)]
        state = _market_state_from_proxy(benchmark_nav, width=0.2, bench_pct=0.5)
        self.assertNotIn(state, {"退潮", "退潮末期", "冰点"})

    def test_retreat_note_distinguishes_strong_breadth_reversal(self) -> None:
        note = _market_note("退潮", 10.0)
        self.assertIn("宽度强修复", note)
        self.assertNotIn("赚钱效应弱", note)

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
            risk_cluster("科技/半导体", "半导体", "半导体ETF"),
            risk_cluster("军工", "机械", "工业母机ETF"),
            risk_cluster("宽基", "宽基", "科创50ETF"),
        }
        self.assertEqual(clusters, {"高弹性成长"})
        self.assertEqual(MAX_SAME_RISK_CLUSTER, 2)
        self.assertEqual(
            risk_cluster("债券/地方债", "债券", "10年地方债ETF"),
            "防御价值",
        )

    def test_scanner_does_not_fill_empty_slots_with_weak_candidates(self) -> None:
        def candidate(
            code: str,
            industry: str,
            stage: str,
            score: float,
            *,
            passes: bool,
        ) -> dict:
            return {
                "etf_code": code,
                "etf_name": f"ETF-{code}",
                "direction": industry,
                "industry": industry,
                "category": industry,
                "is_qdii": False,
                "stage": (stage, "", ""),
                "score": {
                    "total": score,
                    "mainline": 70.0 if passes else 20.0,
                    "timing": 65.0 if passes else 35.0,
                    "risk_penalty": 5.0,
                    "mainline_pass": passes,
                    "timing_pass": passes,
                    "gate_reason": "" if passes else "主线未确认",
                },
                "metrics": {
                    "amount_yi": 5.0,
                    "pct_chg": 0.5,
                    "ret_5d": 1.0,
                    "ret_20d": 8.0,
                    "ret_60d": 12.0,
                    "overheat": False,
                },
            }

        source = [
            candidate("159516", "科技", "扩散期", 70.0, passes=True),
            candidate("515880", "科技", "观察期", 42.0, passes=False),
            candidate("515120", "医药", "休眠期", 35.0, passes=False),
        ]
        selected, _ = pick_formal_portfolio(source, market_state="主升")
        self.assertEqual([row["etf_code"] for row in selected], ["159516"])

    def test_qdii_can_enter_when_all_score_layers_pass(self) -> None:
        def candidate(code: str, category: str, score: float, is_qdii: bool = False) -> dict:
            return {
                "etf_code": code,
                "etf_name": f"ETF-{code}",
                "direction": category,
                "industry": category,
                "category": category,
                "is_qdii": is_qdii,
                "premium_pct": 1.0 if is_qdii else None,
                "stage": ("确认期", "", ""),
                "score": {
                    "total": score,
                    "mainline": 72.0,
                    "timing": 65.0,
                    "risk_penalty": 5.0,
                    "mainline_pass": True,
                    "timing_pass": True,
                },
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
            candidate("159516", "科技", 75.0),
            candidate("513310", "跨境/美股", 72.0, is_qdii=True),
        ]
        selected, _ = pick_formal_portfolio(source, market_state="退潮")
        codes = {row["etf_code"] for row in selected}
        self.assertIn("513310", codes)

    def test_one_day_rebound_fails_while_persistent_mainline_passes(self) -> None:
        persistent = {
            "pct_chg": 1.0,
            "ret_5d": 6.0,
            "ret_20d": 18.0,
            "ret_60d": 30.0,
            "excess_5d": 4.0,
            "excess_20d": 12.0,
            "excess_60d": 20.0,
            "beat_days_10": 8,
            "trend_efficiency_20": 0.45,
            "ma20_slope_5d": 2.0,
            "distance_ma20": 3.0,
            "distance_ma60": 10.0,
            "one_day_dominance_5": 0.25,
            "positive_days_5": 4,
            "amount_persistence_3d": 1.3,
            "volatility_20": 30.0,
            "theme_breadth": 0.7,
            "theme_confirmations": 3,
            "amount_yi": 5.0,
        }
        rebound = {
            **persistent,
            "ret_5d": 8.0,
            "ret_20d": 2.0,
            "ret_60d": -2.0,
            "excess_20d": -1.0,
            "beat_days_10": 2,
            "ma20_slope_5d": -1.0,
            "one_day_dominance_5": 0.9,
            "positive_days_5": 1,
            "theme_breadth": 0.3,
            "theme_confirmations": 0,
        }
        persistent_score = calc_auto_score(persistent, "确认期", 0.0)
        rebound_score = calc_auto_score(rebound, "萌芽期", 0.0)
        self.assertLessEqual(persistent_score["mainline"], 100.0)
        self.assertTrue(persistent_score["mainline_pass"])
        self.assertFalse(rebound_score["mainline_pass"])
        self.assertGreater(rebound_score["risk_penalty"], persistent_score["risk_penalty"])

    def test_one_day_dominance_detects_spike_before_last_day(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=8)
        closes = [100.0, 99.0, 98.0, 108.0, 107.0, 106.5, 106.8, 107.0]
        frame = pd.DataFrame({
            "date": dates,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
            "amount": [500_000_000] * len(closes),
        })
        metrics = calc_etf_metrics(frame, dates[-1].strftime("%Y-%m-%d"))
        self.assertGreater(metrics["one_day_dominance_5"], 0.5)

    def test_retreat_allows_only_one_candidate_per_risk_cluster(self) -> None:
        def candidate(code: str) -> dict:
            return {
                "etf_code": code,
                "etf_name": f"科技ETF-{code}",
                "direction": "科技",
                "industry": "科技",
                "category": "科技",
                "is_qdii": False,
                "stage": ("确认期", "", ""),
                "score": {
                    "total": 75.0,
                    "mainline": 72.0,
                    "timing": 65.0,
                    "risk_penalty": 5.0,
                    "mainline_pass": True,
                    "timing_pass": True,
                },
                "metrics": {
                    "amount_yi": 5.0,
                    "pct_chg": 0.5,
                    "ret_5d": 2.0,
                    "ret_20d": 8.0,
                    "ret_60d": 12.0,
                    "overheat": False,
                },
            }

        selected, _ = pick_formal_portfolio(
            [candidate("159516"), candidate("515880")],
            market_state="退潮",
        )
        self.assertEqual(len(selected), 1)


class SelectionGuardTests(unittest.TestCase):
    def _validate(
        self,
        text: str,
        *,
        write_evidence: bool = True,
    ) -> tuple[bool, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.md"
            path.write_text(text, encoding="utf-8")
            if write_evidence:
                evidence = {
                    "schema_version": 1,
                    "data_cutoff": "2026-01-09 15:00",
                    "news_cutoff": "2026-01-09 20:00",
                    "holdings": [
                        {
                            "code": code,
                            "action": "new",
                            "thesis": "价格趋势与事件验证同向",
                            "invalidation": "价格跌破计划止损且事件证伪",
                            "catalyst": {
                                "title": "待发生官方数据",
                                "known_at": "2026-01-09 19:00",
                                "event_date": "2026-01-12 10:00",
                                "source_name": "official",
                                "source_url": "https://example.com/official",
                            },
                        }
                        for code in re.findall(r"`(\d{6})`", text)
                    ],
                }
                path.with_name("selection_evidence.json").write_text(
                    json.dumps(evidence, ensure_ascii=False),
                    encoding="utf-8",
                )
            return validate_selection(path, require_decision_contract=False)

    def test_valid_three_position_plan_passes(self) -> None:
        ok, errors = self._validate(_selection_text((36.0, 31.5, 22.5)))
        self.assertTrue(ok, errors)

    def test_missing_structured_evidence_is_rejected(self) -> None:
        ok, errors = self._validate(
            _selection_text((36.0, 31.5, 22.5)),
            write_evidence=False,
        )
        self.assertFalse(ok)
        self.assertTrue(any("selection_evidence.json" in error for error in errors), errors)

    def test_handwritten_selection_has_no_execution_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.md"
            path.write_text(_selection_text((36.0, 31.5, 22.5)), encoding="utf-8")
            ok, errors = validate_selection(path)
        self.assertFalse(ok)
        self.assertTrue(any("selection.generated.md" in error for error in errors), errors)

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

    def test_non_ice_market_rejects_exposure_below_state_floor(self) -> None:
        ok, errors = self._validate(_selection_text((24.0, 21.0, 15.0), cash=40.0))
        self.assertFalse(ok)
        self.assertTrue(any("80-100%" in error for error in errors), errors)

        ok, errors = self._validate(_selection_text((32.0, 28.0, 20.0), cash=20.0))
        self.assertTrue(ok, errors)

    def test_ice_point_cash_only_plan_passes(self) -> None:
        ok, errors = self._validate(_cash_only_selection_text())
        self.assertTrue(ok, errors)

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
            evidence = {
                "schema_version": 1,
                "data_cutoff": "2026-01-09 15:00",
                "news_cutoff": "2026-01-09 20:00",
                "holdings": [],
            }
            path.with_name("selection_evidence.json").write_text(
                json.dumps(evidence),
                encoding="utf-8",
            )
            ok, errors = validate_selection(
                path,
                require_decision_contract=False,
            )
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
        self.assertAlmostEqual(log[1]["intraday_return"], 8.0)
        expected = (1 - 0.064 / 100) * (1 + 8.0 / 100) - 1
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

    def test_weak_mainline_carry_is_sold_in_retreat(self) -> None:
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
            "score": {"total": 60.0, "mainline": 55.0, "risk_penalty": 0.0},
            "_rank": 10,
            "category": "科技",
            "industry": "科技",
            "etf_name": "半导体设备ETF",
        }
        previous = {
            "etf_code": "159516",
            "holding_since": "2026-01-01",
            "peak_score": 65.0,
        }
        keep, reason, _ = _should_hold_position(
            previous,
            {"159516": current},
            {"159516": history},
            dates[-1],
            market_state="退潮",
        )
        self.assertFalse(keep)
        self.assertIn("续持门槛", reason)


class BacktestReportTests(unittest.TestCase):
    def test_trailing_stop_uses_only_prior_day_peak(self) -> None:
        rows = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [100.0, 110.0],
            "high": [120.0, 111.0],
            "low": [95.0, 107.0],
            "close": [110.0, 107.5],
        })
        result = simulate_long_with_stop(
            rows,
            entry_price=100.0,
            initial_stop_pct=0.20,
            trailing_stop_pct=0.10,
            slippage_bps=0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.exit_date, "2026-01-06")
        self.assertAlmostEqual(result.exit_price, 108.0)

    def test_gap_below_stop_fills_at_open_less_slippage(self) -> None:
        rows = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [100.0, 80.0],
            "high": [105.0, 82.0],
            "low": [99.0, 75.0],
            "close": [104.0, 78.0],
        })
        result = simulate_long_with_stop(
            rows,
            entry_price=100.0,
            initial_stop_pct=0.10,
            trailing_stop_pct=0.10,
            slippage_bps=10,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.exit_note.split("@")[0], "止损-跳空止损")
        self.assertAlmostEqual(result.exit_price, 79.92)

    def test_daily_risk_exit_executes_next_open(self) -> None:
        rows = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [100.0, 95.0],
            "high": [101.0, 96.0],
            "low": [99.0, 94.0],
            "close": [100.0, 95.0],
        })
        result = simulate_long_with_stop(
            rows,
            entry_price=100.0,
            initial_stop_pct=0.20,
            trailing_stop_pct=0.20,
            slippage_bps=0,
            forced_exit_dates={pd.Timestamp("2026-01-06")},
        )
        self.assertEqual(result.exit_date, "2026-01-06")
        self.assertEqual(result.exit_price, 95.0)
        self.assertIn("日度风控次日开盘", result.exit_note)

    def test_cluster_crash_requires_three_distinct_directions(self) -> None:
        rows = [
            {
                "code": str(index),
                "name": name,
                "sector": sector,
                "industry": industry,
                "pct": -6.0,
            }
            for index, (name, sector, industry) in enumerate((
                ("芯片ETF", "科技/芯片", "科技"),
                ("通信ETF", "科技/通信", "科技"),
                ("机器人ETF", "科技/机器人", "科技"),
            ))
        ]
        self.assertEqual(detect_cluster_crashes(rows), {"高弹性成长"})

    def test_daily_drawdown_catches_intraperiod_loss(self) -> None:
        daily = _finalize_daily_equity([
            {"date": "2026-01-05", "nav": 1.0, "benchmark_nav": 1.0},
            {"date": "2026-01-06", "nav": 0.8, "benchmark_nav": 1.0},
            {"date": "2026-01-09", "nav": 1.0, "benchmark_nav": 1.0},
        ])
        self.assertAlmostEqual(float(daily["drawdown_pct"].min()), -20.0)

    def test_daily_drawdown_includes_initial_nav(self) -> None:
        daily = _finalize_daily_equity([
            {"date": "2026-01-05", "nav": 0.95, "benchmark_nav": 1.0},
            {"date": "2026-01-06", "nav": 0.97, "benchmark_nav": 1.0},
        ])
        self.assertAlmostEqual(float(daily["drawdown_pct"].min()), -5.0)

    def test_cost_sensitivity_is_monotonic(self) -> None:
        frame = pd.DataFrame({
            "gross_return_pct": [2.0, 1.0],
            "return_pct": [1.9, 0.9],
            "cost_pct": [0.1, 0.1],
            "turnover_pct": [100.0, 100.0],
        })
        table = _cost_sensitivity(frame)
        self.assertTrue(table["累计收益%"].is_monotonic_decreasing)

    def test_strategy_manifest_verifies_current_files(self) -> None:
        manifest = build_manifest(
            "test",
            frozen_at="2026-07-25 23:00",
            forward_start="2026-07-27",
        )
        ok, errors = verify_manifest(manifest)
        self.assertTrue(ok, errors)

    def test_point_in_time_universe_respects_effective_dates(self) -> None:
        registry = pd.DataFrame([
            {
                "code": "510300",
                "effective_from": "2020-01-01",
                "effective_to": "",
            },
            {
                "code": "588080",
                "effective_from": "2020-09-28",
                "effective_to": "2026-01-31",
            },
        ])
        self.assertEqual(active_codes(registry, "2020-06-01"), {"510300"})
        self.assertEqual(
            active_codes(registry, "2025-01-01"),
            {"510300", "588080"},
        )
        self.assertEqual(active_codes(registry, "2026-02-01"), {"510300"})

    def test_walkforward_folds_never_change_parameters(self) -> None:
        periods = pd.DataFrame({
            "month": [f"W{i}" for i in range(8)],
            "entry_date": pd.date_range("2026-01-01", periods=8, freq="W").astype(str),
            "exit_date": pd.date_range("2026-01-05", periods=8, freq="W").astype(str),
            "return_pct": [1.0] * 8,
            "benchmark_pct": [0.0] * 8,
        })
        daily = pd.DataFrame({
            "period": [f"W{i}" for i in range(8)],
            "nav": [1.0 + i * 0.01 for i in range(8)],
        })
        folds = build_folds(periods, daily, train_periods=4, test_periods=2)
        self.assertEqual(len(folds), 2)
        self.assertTrue(folds["complete_fold"].all())
        self.assertTrue((folds["parameter_changes"] == 0).all())

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


class MarketDataCacheTests(unittest.TestCase):
    @staticmethod
    def _bars(start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))
        sequence = pd.Series(range(len(dates)), dtype=float)
        close = 1.0 + sequence * 0.01
        return pd.DataFrame({
            "date": dates,
            "open": close - 0.005,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": 1_000_000 + sequence * 1_000,
            "amount": 10_000_000 + sequence * 10_000,
            "turnover_rate": 1.0 + sequence * 0.01,
        })

    def test_second_read_is_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def fetcher(code: str, start: str, end: str, product: str) -> pd.DataFrame:
                calls.append((code, start, end, product))
                frame = self._bars(start, end)
                frame.attrs["data_source"] = "test_provider"
                return frame

            first = load_history(
                "510300",
                "20260101",
                "20260210",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertFalse(first.empty)
            self.assertEqual(len(calls), 1)

            def forbidden_fetcher(*_args) -> pd.DataFrame:
                raise AssertionError("covered local range must not access the network")

            second = load_history(
                "510300",
                "20260101",
                "20260210",
                network_fetcher=forbidden_fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertTrue(second.attrs["cache_hit"])
            self.assertEqual(second.attrs["network_calls"], 0)
            self.assertEqual(len(second), len(first))

    def test_missing_right_edge_fetches_only_the_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def fetcher(_code: str, start: str, end: str, _product: str) -> pd.DataFrame:
                calls.append((start, end))
                return self._bars(start, end)

            initial = load_history(
                "510300",
                "20260101",
                "20260120",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            extended = load_history(
                "510300",
                "20260101",
                "20260220",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1], ("20260121", "20260220"))
            self.assertEqual(extended.attrs["network_calls"], 1)
            self.assertEqual(extended["date"].max(), pd.Timestamp("2026-02-20"))
            persisted_path = root / "cache" / "bars" / "510300_ETF.csv"
            persisted = pd.read_csv(persisted_path, parse_dates=["date"])
            self.assertEqual(persisted["date"].min(), initial["date"].min())
            self.assertEqual(persisted["date"].max(), extended["date"].max())
            self.assertEqual(int(persisted["date"].duplicated().sum()), 0)

            def forbidden_fetcher(*_args) -> pd.DataFrame:
                raise AssertionError("persisted increment must be available offline")

            offline = load_history(
                "510300",
                "20260101",
                "20260220",
                network_fetcher=forbidden_fetcher,
                offline=True,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertEqual(offline.attrs["network_calls"], 0)
            pd.testing.assert_series_equal(
                offline["close"].reset_index(drop=True),
                extended["close"].reset_index(drop=True),
                check_names=False,
            )

    def test_monday_close_is_not_covered_by_previous_friday(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calls = []

            def fetcher(_code: str, start: str, end: str, _product: str) -> pd.DataFrame:
                calls.append((start, end))
                return self._bars(start, end)

            load_history(
                "510300",
                "20260202",
                "20260220",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            monday = load_history(
                "510300",
                "20260202",
                "20260223",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertEqual(calls[-1], ("20260221", "20260223"))
            self.assertEqual(monday["date"].max(), pd.Timestamp("2026-02-23"))
            self.assertEqual(monday.attrs["network_calls"], 1)

    def test_later_cached_session_covers_holiday_request_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def fetcher(_code: str, start: str, end: str, _product: str) -> pd.DataFrame:
                frame = self._bars(start, end)
                return frame.loc[
                    frame["date"] != pd.Timestamp("2026-01-08")
                ].reset_index(drop=True)

            load_history(
                "510300",
                "20260101",
                "20260109",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            replay = load_history(
                "510300",
                "20260101",
                "20260108",
                network_fetcher=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("later cached bar proves the in-range calendar edge")
                ),
                offline=True,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertTrue(replay.attrs["coverage_end_ok"])
            self.assertEqual(replay.attrs["network_calls"], 0)

    def test_split_on_first_incremental_day_is_back_adjusted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def fetcher(_code: str, start: str, end: str, _product: str) -> pd.DataFrame:
                frame = self._bars(start, end)
                if pd.to_datetime(end) <= pd.Timestamp("2026-01-20"):
                    frame[["open", "high", "low", "close"]] *= 3
                return frame

            load_history(
                "510300",
                "20260102",
                "20260120",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            merged = load_history(
                "510300",
                "20260102",
                "20260206",
                network_fetcher=fetcher,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            boundary_return = merged.loc[
                merged["date"] == pd.Timestamp("2026-01-21"),
                "pct_chg",
            ].iloc[0]
            self.assertLess(abs(boundary_return), 5.0)

    def test_offline_mode_never_calls_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def forbidden_fetcher(*_args) -> pd.DataFrame:
                raise AssertionError("offline mode must not access the network")

            result = load_history(
                "510300",
                "20260101",
                "20260120",
                network_fetcher=forbidden_fetcher,
                offline=True,
                cache_root=root / "cache",
                legacy_root=root / "legacy",
            )
            self.assertTrue(result.empty)
            self.assertEqual(result.attrs["network_calls"], 0)

    def test_backtest_rejects_partial_offline_history(self) -> None:
        partial = self._bars("2026-01-05", "2026-01-20")
        partial.attrs["coverage_start_ok"] = False
        partial.attrs["coverage_end_ok"] = True
        with patch("backtest.fetch_etf_hist", return_value=partial):
            with self.assertRaisesRegex(RuntimeError, "offline cache coverage incomplete"):
                _fetch_hist_cached(
                    "510300",
                    "20260101",
                    "20260120",
                    "ETF",
                    offline=True,
                )

    def test_legacy_full_history_recognizes_post_listing_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "legacy"
            legacy.mkdir()
            self._bars("2025-10-22", "2026-02-20").to_csv(
                legacy / "530100_ETF_20260501_20260725.csv",
                index=False,
            )

            def forbidden_fetcher(*_args) -> pd.DataFrame:
                raise AssertionError("pre-listing dates are not a network gap")

            result = load_history(
                "530100",
                "20250719",
                "20260220",
                network_fetcher=forbidden_fetcher,
                cache_root=root / "cache",
                legacy_root=legacy,
            )
            self.assertTrue(result.attrs["cache_hit"])
            self.assertTrue(result.attrs["coverage_start_ok"])
            self.assertEqual(result.attrs["network_calls"], 0)

    def test_trailing_features_have_no_future_leakage(self) -> None:
        bars = self._bars("2025-01-01", "2026-02-20")
        baseline = prepare_history(bars, source="test")
        mutated_bars = bars.copy()
        mutated_bars.loc[mutated_bars.index[-1], ["open", "high", "low", "close"]] *= 1.1
        mutated = prepare_history(mutated_bars, source="test")

        expected_columns = {
            "return_20d_pct",
            "ma_close_60",
            "atr_pct_14",
            "rsi_14",
            "annualized_volatility_20d_pct",
            "volume_ratio_5d",
            "drawdown_60d_pct",
            "obv",
            "amihud_illiquidity_20d",
        }
        self.assertTrue(expected_columns.issubset(baseline.columns))
        pd.testing.assert_frame_equal(
            baseline.iloc[:-1].reset_index(drop=True),
            mutated.iloc[:-1].reset_index(drop=True),
            check_dtype=False,
        )


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
