"""Validate deterministic operating packets for every trading phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from decision_contract import verify_decision  # noqa: E402
from etf_analyzer import load_etf_txt  # noqa: E402
from strategy_version import load_current_manifest  # noqa: E402


SCHEMA_VERSION = 1
PHASE_RULES = {
    "preopen": {
        "checks": {
            "event_ledger_updated",
            "account_state_confirmed",
            "pending_orders_reconciled",
            "tradability_checked",
            "qdii_premium_checked",
        },
        "actions": {
            "no_action",
            "execute_existing_plan",
            "cancel_pending_entry",
            "reduce_next_open",
            "exit_next_open",
            "blocked",
        },
    },
    "intraday": {
        "checks": {
            "positions_checked",
            "cluster_risk_checked",
            "market_liquidity_checked",
            "qdii_premium_checked",
            "official_event_checked",
        },
        "actions": {"observe", "emergency_override", "blocked"},
    },
    "postclose": {
        "checks": {
            "market_data_refreshed",
            "data_quality_passed",
            "stops_checked",
            "invalidation_checked",
            "cluster_risk_checked",
            "event_ledger_updated",
        },
        "actions": {
            "continue_weekly",
            "reduce_next_open",
            "exit_next_open",
            "blocked",
        },
    },
    "weekly_review": {
        "checks": {
            "performance_reconciled",
            "benchmark_compared",
            "attribution_reconciled",
            "execution_reviewed",
            "issues_classified",
            "lessons_gated",
        },
        "actions": {"review_only"},
    },
}
DATA_QUALITY = {"pass", "fail", "unknown"}
AUTHORIZATION_KINDS = {"user", "rules_engine", "data_connector"}
MUTATING_ACTIONS = {
    "cancel_pending_entry",
    "reduce_next_open",
    "exit_next_open",
    "emergency_override",
}
ISSUE_CATEGORIES = {
    "normal_variance",
    "data_issue",
    "execution_error",
    "process_violation",
    "model_weakness",
}
SERIOUS_LESSON_CATEGORIES = {
    "lookahead_bias",
    "unauthorized_trade",
    "data_contamination",
    "account_state_fabrication",
}


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("packet root must be an object")
    return payload


def _parse_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _base_template(phase: str, date: str) -> dict:
    rules = PHASE_RULES[phase]
    manifest = load_current_manifest() or {}
    action = {
        "preopen": "no_action",
        "intraday": "observe",
        "postclose": "continue_weekly",
        "weekly_review": "review_only",
    }[phase]
    phase_time = {
        "preopen": "08:40",
        "intraday": "14:00",
        "postclose": "15:30",
        "weekly_review": "16:30",
    }[phase]
    template = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "date": date,
        "as_of": f"{date} {phase_time}",
        "strategy_version": str(manifest.get("strategy_version") or ""),
        "data_quality": "unknown",
        # Deliberately blank: initialization must not claim that checks ran.
        "completed_checks": [],
        "facts": {
            "hard_risk_triggered": False,
            "notes": [],
        },
        "decision": {
            "action": action,
            "codes": [],
            "reason_codes": [],
            "decision_contract": "",
            "authorization": {
                "kind": "",
                "id": "",
                "authorized_at": "",
                "action": "",
                "codes": [],
                "reason_codes": [],
            },
        },
        "parameter_changes": [],
    }
    if phase == "weekly_review":
        template.update(
            {
                "metrics": {
                    "strategy_return_pct": 0.0,
                    "benchmark_return_pct": 0.0,
                    "excess_return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "average_exposure_pct": 0.0,
                    "turnover_multiple": 0.0,
                    "cost_pct": 0.0,
                },
                "attribution": {
                    "market_exposure_pct": 0.0,
                    "selection_pct": 0.0,
                    "timing_pct": 0.0,
                    "cost_pct": 0.0,
                    "risk_actions_pct": 0.0,
                    "residual_pct": 0.0,
                },
                "issues": [],
                "lessons": [],
                "reflection": {
                    "did_right": [],
                    "did_wrong": [],
                    "luck": [],
                },
                "next_week_actions": [],
            }
        )
    return template

def _validate_weekly_review(packet: dict, errors: list[str]) -> None:
    metrics = packet.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("weekly_review metrics object is missing")
        return
    required_metrics = (
        "strategy_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "max_drawdown_pct",
        "average_exposure_pct",
        "turnover_multiple",
        "cost_pct",
    )
    values: dict[str, float] = {}
    for field in required_metrics:
        value = _number(metrics.get(field))
        if value is None:
            errors.append(f"weekly_review metric {field} must be numeric")
        else:
            values[field] = value
    if len(values) == len(required_metrics):
        expected_excess = (
            values["strategy_return_pct"] - values["benchmark_return_pct"]
        )
        if abs(expected_excess - values["excess_return_pct"]) > 0.05:
            errors.append("weekly_review excess return does not reconcile")
        if not -100 <= values["max_drawdown_pct"] <= 0:
            errors.append("weekly_review max_drawdown_pct must be between -100 and 0")
        if not 0 <= values["average_exposure_pct"] <= 100:
            errors.append("weekly_review average_exposure_pct must be 0..100")
        if values["turnover_multiple"] < 0 or values["cost_pct"] < 0:
            errors.append("weekly_review turnover and cost cannot be negative")

    attribution = packet.get("attribution")
    attribution_fields = (
        "market_exposure_pct",
        "selection_pct",
        "timing_pct",
        "cost_pct",
        "risk_actions_pct",
        "residual_pct",
    )
    if not isinstance(attribution, dict):
        errors.append("weekly_review attribution object is missing")
    else:
        attribution_values = [_number(attribution.get(field)) for field in attribution_fields]
        if any(value is None for value in attribution_values):
            errors.append("weekly_review attribution fields must be numeric")
        elif "strategy_return_pct" in values:
            attributed = sum(float(value) for value in attribution_values if value is not None)
            if abs(attributed - values["strategy_return_pct"]) > 0.15:
                errors.append("weekly_review attribution does not reconcile to return")

    issues = packet.get("issues")
    if not isinstance(issues, list):
        errors.append("weekly_review issues must be an array")
    else:
        for issue in issues:
            if not isinstance(issue, dict):
                errors.append("weekly_review issue must be an object")
                continue
            if issue.get("category") not in ISSUE_CATEGORIES:
                errors.append("weekly_review issue category is invalid")
            if not str(issue.get("evidence") or "").strip():
                errors.append("weekly_review issue needs evidence")

    lessons = packet.get("lessons")
    if not isinstance(lessons, list):
        errors.append("weekly_review lessons must be an array")
    else:
        for lesson in lessons:
            if not isinstance(lesson, dict):
                errors.append("weekly_review lesson must be an object")
                continue
            promote = lesson.get("promote_to_long_term") is True
            repeat_count = int(_number(lesson.get("repeat_count")) or 0)
            category = str(lesson.get("category") or "")
            severity = str(lesson.get("severity") or "")
            serious = category in SERIOUS_LESSON_CATEGORIES and severity == "critical"
            if promote and repeat_count < 2 and not serious:
                errors.append(
                    "lesson promotion requires repeat_count>=2 or a critical serious error"
                )
            for field in ("rule", "applicability", "counterexample", "next_action"):
                if promote and not str(lesson.get(field) or "").strip():
                    errors.append(f"promoted lesson missing {field}")

    reflection = packet.get("reflection")
    if not isinstance(reflection, dict):
        errors.append("weekly_review reflection object is missing")
    else:
        for field in ("did_right", "did_wrong", "luck"):
            if not isinstance(reflection.get(field), list):
                errors.append(f"weekly_review reflection.{field} must be an array")
    actions = packet.get("next_week_actions")
    if not isinstance(actions, list) or len(actions) > 3:
        errors.append("weekly_review next_week_actions must be an array of at most 3")


def validate_packet(packet: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    phase = str(packet.get("phase") or "")
    if phase not in PHASE_RULES:
        return False, errors + ["phase is invalid"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(packet.get("date") or "")):
        errors.append("date must be YYYY-MM-DD")
    as_of = _parse_datetime(packet.get("as_of"))
    if not as_of:
        errors.append("as_of must be an ISO datetime")
    elif as_of.strftime("%Y-%m-%d") != str(packet.get("date") or ""):
        errors.append("as_of date must equal packet date")
    if packet.get("data_quality") not in DATA_QUALITY:
        errors.append("data_quality must be pass/fail/unknown")

    completed = packet.get("completed_checks")
    if not isinstance(completed, list):
        errors.append("completed_checks must be an array")
        completed_set: set[str] = set()
    else:
        completed_set = {str(value) for value in completed}
    missing_checks = PHASE_RULES[phase]["checks"] - completed_set
    unknown_checks = completed_set - PHASE_RULES[phase]["checks"]
    if missing_checks:
        errors.append("missing required checks: " + ", ".join(sorted(missing_checks)))
    if unknown_checks:
        errors.append("unknown checks: " + ", ".join(sorted(unknown_checks)))

    decision = packet.get("decision")
    if not isinstance(decision, dict):
        return False, errors + ["decision object is missing"]
    action = str(decision.get("action") or "")
    if action not in PHASE_RULES[phase]["actions"]:
        errors.append(f"action {action or '<missing>'} is not allowed in {phase}")
    codes = decision.get("codes")
    if not isinstance(codes, list):
        errors.append("decision codes must be an array")
        codes = []
    elif len(list(map(str, codes))) != len(set(map(str, codes))):
        errors.append("decision codes contain duplicates")
    allowed_codes = {cfg["code"] for cfg in load_etf_txt().values()}
    for code in codes:
        if str(code) not in allowed_codes:
            errors.append(f"decision code {code} is outside the fixed pool")

    if phase == "intraday" and action not in {"observe", "emergency_override", "blocked"}:
        errors.append("intraday cannot rotate or open a new position")

    reason_codes = decision.get("reason_codes")
    if not isinstance(reason_codes, list):
        errors.append("decision reason_codes must be an array")
        reason_codes = []
    authorization = decision.get("authorization")
    if action in MUTATING_ACTIONS:
        if not isinstance(authorization, dict):
            errors.append(f"{action} requires structured non-model authorization")
        else:
            kind = str(authorization.get("kind") or "")
            if kind not in AUTHORIZATION_KINDS:
                errors.append(
                    f"{action} authorization.kind must be "
                    "user/rules_engine/data_connector"
                )
            if action == "emergency_override" and kind != "user":
                errors.append("emergency_override requires explicit user authorization")
            for field in ("id", "authorized_at", "action"):
                if not str(authorization.get(field) or "").strip():
                    errors.append(f"{action} authorization missing {field}")
            if not _parse_datetime(authorization.get("authorized_at")):
                errors.append(f"{action} authorization authorized_at is invalid")
            elif as_of and _parse_datetime(authorization.get("authorized_at")) > as_of:
                errors.append(f"{action} authorization is later than packet as_of")
            if authorization.get("action") != action:
                errors.append(f"{action} authorization action does not match")
            auth_codes = authorization.get("codes")
            if not isinstance(auth_codes, list) or sorted(map(str, auth_codes)) != sorted(
                map(str, codes)
            ):
                errors.append(f"{action} authorization codes do not match")
            auth_reasons = authorization.get("reason_codes")
            if (
                not isinstance(auth_reasons, list)
                or sorted(map(str, auth_reasons))
                != sorted(map(str, reason_codes))
            ):
                errors.append(f"{action} authorization reason_codes do not match")
            if not reason_codes:
                errors.append(f"{action} requires at least one reason_code")

    if packet.get("data_quality") != "pass" and action == "execute_existing_plan":
        errors.append("cannot execute a new plan when data quality is not pass")
    if (
        phase in {"postclose", "weekly_review"}
        and packet.get("data_quality") != "pass"
        and action != "blocked"
    ):
        errors.append(f"{phase} requires passing data quality or a blocked action")
    if action in {"no_action", "observe", "continue_weekly", "review_only"} and codes:
        errors.append(f"{action} must not contain instrument codes")
    decision_path = str(decision.get("decision_contract") or "").strip()
    if action == "execute_existing_plan":
        if not decision_path:
            errors.append("execute_existing_plan requires decision_contract")
        else:
            path = PROJECT_ROOT / decision_path
            ok, decision_errors = verify_decision(path)
            if not ok:
                errors.extend(
                    f"decision_contract: {error}" for error in decision_errors
                )
            else:
                decision_payload = _read_json(path)
                if decision_payload.get("status") != "ready":
                    errors.append("decision contract target is not ready")
                if decision_payload.get("orders_status") != "ready":
                    errors.append("decision contract orders are not ready")
                expected_codes = sorted(
                    str(row.get("code"))
                    for row in decision_payload.get("orders", [])
                    if row.get("action") != "hold"
                )
                if sorted(map(str, codes)) != expected_codes:
                    errors.append(
                        "execute_existing_plan codes do not match actionable orders"
                    )

    parameter_changes = packet.get("parameter_changes")
    if not isinstance(parameter_changes, list):
        errors.append("parameter_changes must be an array")
    elif parameter_changes:
        errors.append(
            "operating packets cannot change frozen parameters; create a candidate version"
        )
    if phase == "weekly_review":
        _validate_weekly_review(packet, errors)
    return not errors, errors


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize or verify operating contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--phase", choices=sorted(PHASE_RULES), required=True)
    init_parser.add_argument("--date", required=True)
    init_parser.add_argument("--output", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("packet")
    verify_parser.add_argument("--output")

    args = parser.parse_args()
    if args.command == "init":
        payload = _base_template(args.phase, args.date)
        _write_json(Path(args.output), payload)
        print(args.output)
        return

    packet_path = Path(args.packet)
    try:
        packet = _read_json(packet_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"- packet cannot be read: {exc}", file=sys.stderr)
        raise SystemExit(2)
    ok, errors = validate_packet(packet)
    if not ok:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    verified = dict(packet)
    verified["verified_at"] = datetime.now().isoformat(timespec="seconds")
    verified["validator_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    verified["contract_sha256"] = _canonical_hash(packet)
    output = (
        Path(args.output)
        if args.output
        else packet_path.with_name(packet_path.stem + ".verified.json")
    )
    _write_json(output, verified)
    print(f"OK: {output}")


if __name__ == "__main__":
    main()
