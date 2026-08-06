"""Regression tests for model-independent decision and operating contracts."""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import decision_contract as dc  # noqa: E402
import ops_contract as oc  # noqa: E402
import workflow as wf  # noqa: E402
from selection_guard import validate_selection  # noqa: E402


DATE = "2026-01-09"  # Friday


class RecommendationBookingTests(unittest.TestCase):
    def test_ready_formal_recommendation_is_booked_as_assumed_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            decision_path = root / "decision.json"
            account_path = root / "account_state.json"
            decision = {
                "status": "ready",
                "orders_status": "ready",
                "strategy_version": "2.5.7",
                "contract_sha256": "contract-hash",
                "portfolio_fingerprint": "portfolio-hash",
                "as_of": {
                    "signal_date": "2026-07-27",
                    "data_cutoff": "2026-07-27 15:00",
                },
                "portfolio": {
                    "cash_pct": 20.0,
                    "targets": [
                        {
                            "code": "159985",
                            "name": "豆粕ETF华夏",
                            "target_weight_pct": 80.0,
                            "execution": {"reference_close": 2.254},
                        }
                    ],
                },
            }
            decision_path.write_text(
                json.dumps(decision, ensure_ascii=False),
                encoding="utf-8",
            )

            account = wf._record_assumed_formal_recommendation(
                decision_path,
                account_path,
            )

            self.assertEqual(account["cash_pct"], 20.0)
            self.assertEqual(account["positions"][0]["code"], "159985")
            self.assertEqual(account["positions"][0]["entry_price"], 2.254)
            self.assertEqual(
                account["confirmed_by"],
                "user_auto_assume_recommendations",
            )
            self.assertEqual(
                json.loads(account_path.read_text(encoding="utf-8"))["positions"],
                account["positions"],
            )

    def test_blocked_recommendation_cannot_change_account(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            decision_path = root / "decision.json"
            account_path = root / "account_state.json"
            original = {"schema_version": 1, "cash_pct": 100.0, "positions": []}
            account_path.write_text(json.dumps(original), encoding="utf-8")
            decision_path.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "orders_status": "blocked",
                        "portfolio": {"cash_pct": 100.0, "targets": []},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                wf._record_assumed_formal_recommendation(
                    decision_path,
                    account_path,
                )

            self.assertEqual(
                json.loads(account_path.read_text(encoding="utf-8")),
                original,
            )

    def test_confirmed_empty_account_does_not_fall_back_to_stale_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            account_path = root / "account_state.json"
            account_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "confirmed": True,
                        "cash_pct": 100.0,
                        "positions": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(wf, "STOCK_DIR", root):
                positions, authoritative = wf._confirmed_positions()

            self.assertTrue(authoritative)
            self.assertEqual(positions, [])


class ContractFixture(unittest.TestCase):
    def setUp(self) -> None:
        check_dir = PROJECT_ROOT / "check"
        check_dir.mkdir(exist_ok=True)
        self.temp_root = Path(
            tempfile.mkdtemp(prefix="model_contract_", dir=check_dir)
        )
        self.day_dir = self.temp_root / DATE
        self.day_dir.mkdir()
        self.ledger = self.temp_root / "event_ledger.csv"
        self.ledger.write_text(
            "known_at,event_type,scope,score,horizon_days,source,url,summary\n",
            encoding="utf-8",
        )
        self._write_inputs()
        self.patches = [
            patch.object(dc, "DEFAULT_LEDGER", self.ledger),
            patch.object(dc, "load_event_ledger", return_value=[]),
            patch.object(
                dc,
                "event_snapshot",
                return_value={
                    "risk_level": "正常",
                    "action": "按冻结事件台账执行",
                },
            ),
            patch.object(
                dc,
                "load_current_manifest",
                return_value={
                    "strategy_version": "2.4.0",
                    "combined_sha256": "frozen-test-hash",
                },
            ),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        shutil.rmtree(self.temp_root)

    def _write_inputs(self) -> None:
        scan = {
            "schema_version": 2,
            "date": DATE,
            "signal_date": DATE,
            "data_cutoff": f"{DATE} 15:00",
            "strategy_version": "2.4.0",
            "strategy_sha256": "frozen-test-hash",
            "market_state": "主升",
            "formal_candidates": [
                {
                    "rank": 1,
                    "code": "159985",
                    "name": "豆粕ETF华夏",
                    "direction": "豆粕华夏",
                    "product_type": "ETF",
                    "is_qdii": False,
                    "premium_pct": None,
                    "industry": "周期",
                    "stage": "扩散期",
                    "score": 80.0,
                    "mainline_score": 75.0,
                    "timing_score": 70.0,
                    "risk_penalty": 5.0,
                    "scanner_weight_pct": 53.3,
                },
                {
                    "rank": 2,
                    "code": "515180",
                    "name": "红利ETF易方达",
                    "direction": "红利易方达",
                    "product_type": "ETF",
                    "is_qdii": False,
                    "premium_pct": None,
                    "industry": "红利",
                    "stage": "确认期",
                    "score": 75.0,
                    "mainline_score": 70.0,
                    "timing_score": 65.0,
                    "risk_penalty": 4.0,
                    "scanner_weight_pct": 46.7,
                },
            ],
            "cash_pct": 0.0,
            "bench_pct": 0.2,
            "rows": [
                {
                    "code": "159985",
                    "name": "豆粕华夏",
                    "sector": "商品/豆粕",
                    "stage": "扩散期",
                    "score": 80.0,
                    "amount_yi": 8.0,
                    "pct": 1.0,
                    "close": 2.0,
                    "ret_5d": 5.0,
                    "ret_20d": 12.0,
                    "ret_60d": 18.0,
                    "amount_ratio": 1.4,
                    "industry": "周期",
                    "direction": "豆粕华夏",
                    "is_qdii": False,
                    "actual_date": DATE,
                },
                {
                    "code": "515180",
                    "name": "红利易方达",
                    "sector": "红利/综合",
                    "stage": "确认期",
                    "score": 75.0,
                    "amount_yi": 5.0,
                    "pct": 0.5,
                    "close": 1.5,
                    "ret_5d": 3.0,
                    "ret_20d": 8.0,
                    "ret_60d": 10.0,
                    "amount_ratio": 1.2,
                    "industry": "红利",
                    "direction": "红利易方达",
                    "is_qdii": False,
                    "actual_date": DATE,
                },
            ],
        }
        (self.day_dir / "scan.json").write_text(
            json.dumps(scan, ensure_ascii=False),
            encoding="utf-8",
        )
        report = f"""# ETF 三层分析报告

- 数据截止时间：{DATE} 15:00

| 项目 | 结论 |
|---|---|
| 市场状态 | **主升** |

正式池内候选：

| 优先级 | 方向 | 类型 | 行业 | 阶段 | 评分 | 主线 | 买点 | 风险罚分 | 建议仓位 | 池内标的 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | 豆粕华夏 | ETF | 周期 | 扩散期 | 80 | 75 | 70 | 5 | 53.3% | 豆粕ETF华夏 `159985` |
| 2 | 红利易方达 | ETF | 红利 | 确认期 | 75 | 70 | 65 | 4 | 46.7% | 红利ETF易方达 `515180` |
| — | 现金 | 现金 | — | — | — | — | — | — | 0% | — |
"""
        (self.day_dir / "etf_scan.md").write_text(report, encoding="utf-8")
        (self.day_dir / "event_risk.md").write_text(
            f"""# 国际事件风险复核 - {DATE} 16:00

- 风险级别：**正常**
- 风险动作：按冻结事件台账执行
""",
            encoding="utf-8",
        )

    def evidence(
        self,
        *,
        first_status: str = "approved",
        include_second: bool = True,
        narrative: str = "版本甲",
    ) -> dict:
        items = [
            {
                "code": "159985",
                "status": first_status,
                "thesis": f"{narrative}：供需数据继续验证趋势。",
                "invalidation": f"{narrative}：官方数据否定供需缺口。",
                "catalyst": {
                    "title": f"{narrative}官方报告",
                    "known_at": f"{DATE} 15:20",
                    "event_date": "2026-01-12 16:00",
                    "source_name": "官方",
                    "source_url": "https://example.com/a",
                },
                # Extra model-authored execution fields must be ignored.
                "execution": {
                    "entry_low": 999,
                    "entry_high": 1000,
                    "stop": 1,
                    "target": 2000,
                    "risk_reward": 99,
                },
            }
        ]
        if first_status == "rejected":
            items[0] = {
                "code": "159985",
                "status": "rejected",
                "rejection_reason": "证据不足",
            }
        if include_second:
            items.append(
                {
                    "code": "515180",
                    "status": "approved",
                    "thesis": f"{narrative}：红利相对强度保持。",
                    "invalidation": f"{narrative}：趋势和资金同时转弱。",
                    "catalyst": {
                        "title": f"{narrative}成分调整",
                        "known_at": f"{DATE} 15:20",
                        "event_date": "2026-01-13 09:00",
                        "source_name": "交易所",
                        "source_url": "https://example.com/b",
                    },
                }
            )
        return {
            "schema_version": 2,
            "data_cutoff": f"{DATE} 15:00",
            "news_cutoff": f"{DATE} 16:00",
            "approval": {
                "kind": "user",
                "id": "USER_CONFIRMED",
                "approved_at": f"{DATE} 15:50",
            },
            "items": items,
        }

    @staticmethod
    def account(*, confirmed: bool = True, outside_pool: bool = False) -> dict:
        positions = (
            [{"code": "999999", "weight_pct": 20.0}]
            if outside_pool
            else [{"code": "515180", "weight_pct": 20.0}]
        )
        return {
            "schema_version": 1,
            "as_of": f"{DATE} 15:30",
            "confirmed": confirmed,
            "confirmed_by": "USER",
            "cash_pct": 80.0,
            "positions": positions,
        }

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.day_dir / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def build(
        self,
        evidence: dict,
        account: dict | None = None,
    ) -> dict:
        evidence_path = self.write_json("weekly_evidence.json", evidence)
        account_path = (
            self.write_json("account_state.json", account)
            if account is not None
            else None
        )
        return dc.build_decision(
            day_dir=self.day_dir,
            evidence_path=evidence_path,
            account_path=account_path,
        )


class DecisionContractTests(ContractFixture):
    def test_relative_cli_paths_resolve_from_project_root(self) -> None:
        resolved = dc._resolve_project_path(
            "codex/stock/2026-01-09/weekly_evidence.json"
        )
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(
            PROJECT_ROOT / "codex" / "stock" / DATE / "weekly_evidence.json",
            resolved,
        )

    def test_signal_batch_date_does_not_replace_market_data_cutoff(self) -> None:
        previous_day = "2026-01-08"
        scan_path = self.day_dir / "scan.json"
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        scan["data_cutoff"] = f"{previous_day} 15:00"
        for row in scan["rows"]:
            row["actual_date"] = previous_day
        scan_path.write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
        report_path = self.day_dir / "etf_scan.md"
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace(
                f"{DATE} 15:00",
                f"{previous_day} 15:00",
            ),
            encoding="utf-8",
        )
        evidence = self.evidence()
        evidence["data_cutoff"] = f"{previous_day} 15:00"
        decision = self.build(evidence, self.account())
        self.assertEqual(f"{previous_day} 15:00", decision["as_of"]["data_cutoff"])
        rendered = dc._render_selection(decision)
        self.assertIn(f"计划执行时间：{DATE} 开盘", rendered)

    def test_report_cannot_override_machine_formal_candidates(self) -> None:
        report_path = self.day_dir / "etf_scan.md"
        report_path.write_text(
            report_path.read_text(encoding="utf-8").replace(
                "| 1 | 豆粕华夏",
                "| 2 | 豆粕华夏",
            ),
            encoding="utf-8",
        )
        decision = self.build(self.evidence(), self.account())
        self.assertEqual("blocked", decision["status"])
        self.assertTrue(
            any(
                "formal candidates do not equal" in error
                for error in decision["blocking_errors"]
            )
        )

    def test_code_level_event_block_removes_candidate(self) -> None:
        with patch.object(
            dc,
            "event_snapshot",
            return_value={
                "risk_level": "正常",
                "action": "代码级风险否决",
                "blocked_scopes": ["159985"],
            },
        ):
            decision = self.build(self.evidence(), self.account())
        self.assertEqual("ready", decision["status"])
        self.assertEqual(
            ["515180"],
            [row["code"] for row in decision["portfolio"]["targets"]],
        )
        self.assertEqual(
            100.0,
            decision["portfolio"]["targets"][0]["target_weight_pct"],
        )
        rejected = {
            row["code"]: row["rejection_reason"]
            for row in decision["rejected_candidates"]
        }
        self.assertEqual("EVENT_SCOPE_BLOCKED:159985", rejected["159985"])

    def test_non_ice_decision_blocks_when_all_candidates_are_vetoed(self) -> None:
        with patch.object(
            dc,
            "event_snapshot",
            return_value={
                "risk_level": "正常",
                "action": "代码级风险否决",
                "blocked_scopes": ["159985", "515180"],
            },
        ):
            decision = self.build(self.evidence(), self.account())
        self.assertEqual("blocked", decision["status"])
        self.assertEqual([], decision["portfolio"]["targets"])
        self.assertEqual(100.0, decision["portfolio"]["cash_pct"])
        self.assertTrue(
            any("deployment floor" in error for error in decision["blocking_errors"])
        )

    def test_model_narrative_and_fake_execution_do_not_change_portfolio(self) -> None:
        first = self.build(self.evidence(narrative="简短模型"), self.account())
        second = self.build(self.evidence(narrative="详细模型"), self.account())
        self.assertEqual("ready", first["status"])
        self.assertEqual(first["portfolio_fingerprint"], second["portfolio_fingerprint"])
        self.assertEqual(first["portfolio"], second["portfolio"])
        execution = first["portfolio"]["targets"][0]["execution"]
        self.assertEqual(dc.EXECUTION_POLICY_ID, execution["policy_id"])
        self.assertNotEqual(999, execution["entry_low"])

    def test_rejected_candidate_reallocates_only_to_approved_scanner_candidate(self) -> None:
        decision = self.build(
            self.evidence(first_status="rejected"),
            self.account(),
        )
        self.assertEqual("ready", decision["status"])
        self.assertEqual(["515180"], [row["code"] for row in decision["portfolio"]["targets"]])
        self.assertEqual(100.0, decision["portfolio"]["targets"][0]["target_weight_pct"])
        self.assertEqual(0.0, decision["portfolio"]["cash_pct"])

    def test_incomplete_evidence_blocks_entire_decision(self) -> None:
        decision = self.build(
            self.evidence(include_second=False),
            self.account(),
        )
        self.assertEqual("blocked", decision["status"])
        self.assertTrue(
            any("incomplete" in error for error in decision["blocking_errors"])
        )

    def test_unconfirmed_or_outside_pool_account_blocks_orders(self) -> None:
        unconfirmed = self.build(self.evidence(), self.account(confirmed=False))
        self.assertEqual("ready", unconfirmed["status"])
        self.assertEqual("blocked", unconfirmed["orders_status"])
        outside = self.build(self.evidence(), self.account(outside_pool=True))
        self.assertEqual("blocked", outside["orders_status"])
        self.assertTrue(
            any("outside the fixed pool" in error for error in outside["order_blocking_errors"])
        )

    def test_rendered_selection_passes_existing_execution_guard(self) -> None:
        decision = self.build(self.evidence(), self.account())
        selection = self.day_dir / "selection.generated.md"
        selection.write_text(dc._render_selection(decision), encoding="utf-8")
        self.write_json("decision.json", decision)
        self.write_json("selection_evidence.json", dc._selection_evidence_v1(decision))
        ok, errors = validate_selection(selection)
        self.assertTrue(ok, errors)

    def test_tampering_breaks_contract_hash(self) -> None:
        decision = self.build(self.evidence(), self.account())
        path = self.write_json("decision.json", decision)
        ok, errors = dc.verify_decision(path)
        self.assertTrue(ok, errors)
        decision["portfolio"]["cash_pct"] = 99
        self.write_json("decision.json", decision)
        ok, errors = dc.verify_decision(path)
        self.assertFalse(ok)
        self.assertIn("decision contract hash mismatch", errors)


class OperatingContractTests(unittest.TestCase):
    @staticmethod
    def complete_checks(packet: dict) -> None:
        packet["completed_checks"] = sorted(
            oc.PHASE_RULES[packet["phase"]]["checks"]
        )

    def test_intraday_is_risk_only_and_override_needs_authorization(self) -> None:
        packet = oc._base_template("intraday", DATE)
        self.complete_checks(packet)
        packet["data_quality"] = "pass"
        packet["decision"]["action"] = "emergency_override"
        ok, errors = oc.validate_packet(packet)
        self.assertFalse(ok)
        self.assertTrue(any("authorization" in error for error in errors))

        packet["decision"]["codes"] = ["159985"]
        packet["decision"]["reason_codes"] = ["HARD_STOP"]
        packet["decision"]["authorization"] = {
            "kind": "user",
            "id": "USER",
            "authorized_at": f"{DATE} 14:05",
            "action": "emergency_override",
            "codes": ["159985"],
            "reason_codes": ["HARD_STOP"],
        }
        packet["as_of"] = f"{DATE} 14:05"
        ok, errors = oc.validate_packet(packet)
        self.assertTrue(ok, errors)

    def test_execute_plan_is_blocked_when_data_quality_is_not_pass(self) -> None:
        packet = oc._base_template("preopen", DATE)
        self.complete_checks(packet)
        packet["decision"]["action"] = "execute_existing_plan"
        ok, errors = oc.validate_packet(packet)
        self.assertFalse(ok)
        self.assertTrue(any("data quality" in error for error in errors))
        self.assertTrue(any("decision_contract" in error for error in errors))

    def test_weekly_review_reconciles_and_gates_lessons(self) -> None:
        packet = oc._base_template("weekly_review", DATE)
        self.complete_checks(packet)
        packet["data_quality"] = "pass"
        ok, errors = oc.validate_packet(packet)
        self.assertTrue(ok, errors)

        bad = copy.deepcopy(packet)
        bad["lessons"] = [
            {
                "category": "model_weakness",
                "severity": "medium",
                "repeat_count": 1,
                "promote_to_long_term": True,
                "rule": "一次偶发现象",
                "applicability": "所有市场",
                "counterexample": "尚未检查",
                "next_action": "立即改参数",
            }
        ]
        ok, errors = oc.validate_packet(bad)
        self.assertFalse(ok)
        self.assertTrue(any("repeat_count>=2" in error for error in errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
