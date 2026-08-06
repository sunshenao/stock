"""Compile a model-independent weekly target and verify its provenance.

The scanner remains the only authority for ranking, eligibility and the market
risk budget.  After deterministic evidence and event vetoes, this compiler
normalizes that budget across the surviving scanner candidates.  A language
model may summarize evidence, but it cannot reorder candidates, resize
positions, create orders, or approve its own evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from risk_rules import (  # noqa: E402
    allocate_instrument_weights,
    get_position_cap,
    risk_cluster,
)
from strategy_version import load_current_manifest, verify_manifest  # noqa: E402
from event_risk import DEFAULT_LEDGER, event_snapshot, load_event_ledger  # noqa: E402


SCHEMA_VERSION = 1
SCAN_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 2
ACCOUNT_SCHEMA_VERSION = 1
ALLOWED_APPROVAL_KINDS = {"user", "rules_engine", "data_connector"}
ALLOWED_EVIDENCE_STATUS = {"approved", "rejected"}
EXECUTION_POLICY_ID = "weekly_close_band_v1"
ENTRY_LOW_MULTIPLIER = 0.995
ENTRY_HIGH_MULTIPLIER = 1.01
NORMAL_INITIAL_STOP_PCT = 8.0
COMMODITY_INITIAL_STOP_PCT = 14.0
TARGET_RISK_REWARD = 2.0
MODEL_AUTHORITY = {
    "may": [
        "summarize frozen inputs",
        "draft thesis and invalidation text",
        "report missing fields",
    ],
    "must_not": [
        "reorder scanner candidates",
        "change scanner target weights",
        "set entry, stop, or target prices",
        "approve evidence",
        "create instruments outside the fixed pool",
        "convert missing data into a trade",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"decision input must be inside project: {resolved}") from exc


def _parse_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group()) if match else None


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]


def _extract_market_state(report_text: str) -> str | None:
    patterns = (
        r"\|\s*市场状态\s*\|\s*\*{0,2}(主升|震荡|退潮末期|退潮|冰点|未知)",
        r"市场状态\s*[：:]\s*\*{0,2}(主升|震荡|退潮末期|退潮|冰点|未知)",
    )
    for pattern in patterns:
        match = re.search(pattern, report_text)
        if match:
            return match.group(1)
    return None


def _extract_report_cutoff(report_text: str) -> datetime | None:
    match = re.search(r"数据截止时间\s*[：:]\s*([^\n|]+)", report_text)
    return _parse_datetime(match.group(1).strip()) if match else None


def _extract_formal_candidates(report_text: str) -> tuple[list[dict], float | None]:
    lines = report_text.splitlines()
    header_index = None
    headers: list[str] = []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_markdown_row(line)
        normalized = [re.sub(r"\s+", "", value) for value in cells]
        required = {"优先级", "方向", "阶段", "建议仓位", "池内标的"}
        if required.issubset(set(normalized)):
            if not {"总分", "评分"}.intersection(normalized):
                continue
            header_index = index
            headers = normalized
            break
    if header_index is None:
        if "当前没有同时通过主线、买点和风险门禁的标的" in report_text:
            return [], 100.0
        raise ValueError("formal candidate table not found in etf_scan.md")

    def col(name: str) -> int:
        return headers.index(name)

    candidates: list[dict] = []
    cash_pct: float | None = None
    for line in lines[header_index + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = _split_markdown_row(line)
        if len(cells) != len(headers):
            continue
        direction = cells[col("方向")]
        weight = _number(cells[col("建议仓位")])
        if direction == "现金":
            cash_pct = weight
            continue
        instrument = cells[col("池内标的")]
        code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", instrument)
        if not code_match:
            raise ValueError(f"candidate code missing: {instrument}")
        instrument_name = re.sub(r"`?\d{6}`?", "", instrument).strip()
        product_type = cells[col("类型")] if "类型" in headers else "ETF"
        industry = cells[col("行业")] if "行业" in headers else "其他"
        score_column = "总分" if "总分" in headers else "评分"
        candidates.append(
            {
                "rank": int(_number(cells[col("优先级")]) or len(candidates) + 1),
                "code": code_match.group(1),
                "name": instrument_name,
                "direction": direction,
                "product_type": product_type,
                "industry": industry,
                "stage": cells[col("阶段")],
                "score": float(_number(cells[col(score_column)]) or 0.0),
                "mainline_score": float(
                    _number(cells[col("主线")]) or 0.0
                )
                if "主线" in headers
                else None,
                "timing_score": float(
                    _number(cells[col("买点")]) or 0.0
                )
                if "买点" in headers
                else None,
                "risk_penalty": float(
                    _number(cells[col("风险罚分")]) or 0.0
                )
                if "风险罚分" in headers
                else None,
                "scanner_weight_pct": float(weight or 0.0),
            }
        )
    return sorted(candidates, key=lambda row: row["rank"]), cash_pct


def _extract_event_snapshot(event_text: str) -> dict:
    level_match = re.search(
        r"(?:风险级别|国际事件风险)\s*[：:]\s*\*{0,2}(正常|黄色|红色)",
        event_text,
    )
    time_match = re.search(
        r"国际事件风险复核\s*[-—]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
        event_text,
    )
    action_match = re.search(r"(?:风险动作|国际事件动作)\s*[：:]\s*([^\n]+)", event_text)
    return {
        "risk_level": level_match.group(1) if level_match else None,
        "news_cutoff": time_match.group(1) if time_match else None,
        "action": action_match.group(1).strip() if action_match else "",
    }


def _validate_evidence_packet(
    payload: dict,
    *,
    candidates: list[dict],
    data_cutoff: datetime,
    news_cutoff: datetime,
) -> tuple[dict[str, dict], list[str]]:
    errors: list[str] = []
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append(
            f"evidence schema_version must be {EVIDENCE_SCHEMA_VERSION}"
        )
    packet_data_cutoff = _parse_datetime(payload.get("data_cutoff"))
    packet_news_cutoff = _parse_datetime(payload.get("news_cutoff"))
    if packet_data_cutoff != data_cutoff:
        errors.append("evidence data_cutoff does not equal scanner cutoff")
    if packet_news_cutoff != news_cutoff:
        errors.append("evidence news_cutoff does not equal event cutoff")

    approval = payload.get("approval")
    if not isinstance(approval, dict):
        errors.append("evidence approval object is missing")
    else:
        kind = str(approval.get("kind") or "")
        approver = str(approval.get("id") or "").strip()
        approved_at = _parse_datetime(approval.get("approved_at"))
        if kind not in ALLOWED_APPROVAL_KINDS:
            errors.append(
                "evidence approval.kind must be user/rules_engine/data_connector; "
                "a language model cannot self-approve"
            )
        if not approver:
            errors.append("evidence approval.id is missing")
        if not approved_at or approved_at > news_cutoff:
            errors.append("evidence approval time is invalid or later than cutoff")

    items = payload.get("items")
    if not isinstance(items, list):
        errors.append("evidence items must be an array")
        return {}, errors
    by_code: dict[str, dict] = {}
    allowed_codes = {candidate["code"] for candidate in candidates}
    for item in items:
        if not isinstance(item, dict):
            errors.append("every evidence item must be an object")
            continue
        code = str(item.get("code") or "")
        if not re.fullmatch(r"\d{6}", code):
            errors.append("evidence item code must be six digits")
            continue
        if code not in allowed_codes:
            errors.append(f"evidence contains non-candidate code {code}")
            continue
        if code in by_code:
            errors.append(f"duplicate evidence item {code}")
            continue
        status = str(item.get("status") or "")
        if status not in ALLOWED_EVIDENCE_STATUS:
            errors.append(f"{code} evidence status must be approved/rejected")
        if status == "rejected":
            if not str(item.get("rejection_reason") or "").strip():
                errors.append(f"{code} rejected evidence needs rejection_reason")
            by_code[code] = item
            continue

        for field in ("thesis", "invalidation"):
            if not str(item.get(field) or "").strip():
                errors.append(f"{code} approved evidence missing {field}")
        catalyst = item.get("catalyst")
        if not isinstance(catalyst, dict):
            errors.append(f"{code} approved evidence missing catalyst")
        else:
            required = (
                "title",
                "known_at",
                "event_date",
                "source_name",
                "source_url",
            )
            missing = [
                field
                for field in required
                if not str(catalyst.get(field) or "").strip()
            ]
            if missing:
                errors.append(f"{code} catalyst missing {', '.join(missing)}")
            known_at = _parse_datetime(catalyst.get("known_at"))
            event_date = _parse_datetime(catalyst.get("event_date"))
            if not known_at or known_at > news_cutoff:
                errors.append(f"{code} catalyst known_at is after cutoff")
            if not event_date or event_date.date() < data_cutoff.date():
                errors.append(f"{code} catalyst is not a future validation event")
            if not str(catalyst.get("source_url") or "").startswith(
                ("https://", "http://")
            ):
                errors.append(f"{code} catalyst source_url is not traceable")

        by_code[code] = item

    missing_codes = sorted(allowed_codes - set(by_code))
    if missing_codes:
        errors.append(
            "evidence packet is incomplete for scanner candidates: "
            + ", ".join(missing_codes)
        )
    return by_code, errors


def _validate_account_state(
    payload: dict | None,
    *,
    allowed_codes: set[str],
    maximum_as_of: datetime,
) -> tuple[dict[str, float], list[str]]:
    if payload is None:
        return {}, ["account state is missing; target may be reviewed but orders are blocked"]
    errors: list[str] = []
    if payload.get("schema_version") != ACCOUNT_SCHEMA_VERSION:
        errors.append(f"account schema_version must be {ACCOUNT_SCHEMA_VERSION}")
    if payload.get("confirmed") is not True:
        errors.append("account state is not user-confirmed")
    if not str(payload.get("confirmed_by") or "").strip():
        errors.append("account confirmed_by is missing")
    account_as_of = _parse_datetime(payload.get("as_of"))
    if not account_as_of or account_as_of > maximum_as_of:
        errors.append("account as_of is invalid or later than the decision cutoff")
    positions = payload.get("positions")
    if not isinstance(positions, list):
        return {}, errors + ["account positions must be an array"]
    weights: dict[str, float] = {}
    for row in positions:
        if not isinstance(row, dict):
            errors.append("account position must be an object")
            continue
        code = str(row.get("code") or "")
        weight = _number(row.get("weight_pct"))
        if not re.fullmatch(r"\d{6}", code) or weight is None or weight < 0:
            errors.append("account position has invalid code or weight")
            continue
        if code in weights:
            errors.append(f"duplicate account position {code}")
            continue
        weights[code] = float(weight)
    cash_pct = _number(payload.get("cash_pct"))
    if cash_pct is None or cash_pct < 0:
        errors.append("account cash_pct is invalid")
    elif abs(sum(weights.values()) + cash_pct - 100.0) > 0.2:
        errors.append("account positions plus cash must equal 100%")
    outside_pool = sorted(set(weights) - allowed_codes)
    if outside_pool:
        errors.append(
            "account contains positions outside the fixed pool: "
            + ", ".join(outside_pool)
        )
    return weights, errors


def _deterministic_execution(candidate: dict, scan_row: dict) -> dict:
    """Derive all price levels from scanner data and a versioned fixed rule."""
    close = _number(scan_row.get("close"))
    if close is None or close <= 0:
        raise ValueError(f"{candidate['code']} scanner close is missing or invalid")
    sector = str(scan_row.get("sector") or "")
    is_commodity = sector.split("/")[0] in {"商品", "资源"}
    stop_pct = (
        COMMODITY_INITIAL_STOP_PCT
        if is_commodity
        else NORMAL_INITIAL_STOP_PCT
    )
    entry_low = round(close * ENTRY_LOW_MULTIPLIER, 3)
    entry_high = round(close * ENTRY_HIGH_MULTIPLIER, 3)
    stop = round(entry_high * (1 - stop_pct / 100), 3)
    target = round(
        entry_high + TARGET_RISK_REWARD * (entry_high - stop),
        3,
    )
    return {
        "policy_id": EXECUTION_POLICY_ID,
        "reference_close": round(close, 3),
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_condition": "next_open_in_band_else_cancel",
        "initial_stop_pct": stop_pct,
        "stop": stop,
        "target": target,
        "risk_reward": TARGET_RISK_REWARD,
    }


def _next_weekday(value: datetime) -> datetime:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _derive_orders(
    current: dict[str, float],
    targets: list[dict],
) -> list[dict]:
    target_map = {row["code"]: float(row["target_weight_pct"]) for row in targets}
    orders = []
    for code in sorted(set(current) | set(target_map)):
        before = float(current.get(code, 0.0))
        after = float(target_map.get(code, 0.0))
        delta = round(after - before, 2)
        if abs(delta) < 0.05:
            action = "hold"
        elif delta > 0:
            action = "buy"
        elif after <= 0:
            action = "exit"
        else:
            action = "reduce"
        orders.append(
            {
                "code": code,
                "action": action,
                "current_weight_pct": round(before, 2),
                "target_weight_pct": round(after, 2),
                "delta_weight_pct": delta,
                "execution": "next_trading_day_open",
            }
        )
    return orders


def build_decision(
    *,
    day_dir: Path,
    evidence_path: Path,
    account_path: Path | None = None,
) -> dict:
    day_dir = day_dir.resolve()
    evidence_path = evidence_path.resolve()
    account_path = account_path.resolve() if account_path else None
    scan_path = day_dir / "scan.json"
    report_path = day_dir / "etf_scan.md"
    event_path = day_dir / "event_risk.md"
    required = (scan_path, report_path, event_path, evidence_path, DEFAULT_LEDGER)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing decision input: " + ", ".join(missing))

    scan = _read_json(scan_path)
    report_text = report_path.read_text(encoding="utf-8-sig")
    evidence = _read_json(evidence_path)
    account = _read_json(account_path) if account_path and account_path.exists() else None

    errors: list[str] = []
    signal_date = str(scan.get("signal_date") or scan.get("date") or day_dir.name)
    if signal_date != day_dir.name:
        errors.append("scan date does not match directory date")
    if scan.get("schema_version") != SCAN_SCHEMA_VERSION:
        errors.append(f"scan schema_version must be {SCAN_SCHEMA_VERSION}")
    market_state = _extract_market_state(report_text)
    if not market_state:
        errors.append("market state missing from scanner report")
        market_state = "未知"
    scan_market_state = str(scan.get("market_state") or "")
    if scan_market_state != market_state:
        errors.append("scanner report market state does not equal scan.json")
    try:
        report_candidates, report_cash = _extract_formal_candidates(report_text)
    except ValueError as exc:
        report_candidates, report_cash = [], None
        errors.append(str(exc))
    machine_candidates = scan.get("formal_candidates")
    if not isinstance(machine_candidates, list):
        candidates = []
        errors.append("scan formal_candidates must be an array")
    else:
        candidates = [
            dict(candidate)
            for candidate in machine_candidates
            if isinstance(candidate, dict)
        ]
        if len(candidates) != len(machine_candidates):
            errors.append("every scan formal candidate must be an object")
        candidates.sort(key=lambda row: int(row.get("rank") or 999))
    scanner_cash = _number(scan.get("cash_pct"))
    report_core = [
        (
            int(row.get("rank") or 0),
            str(row.get("code") or ""),
            round(float(row.get("score") or 0), 2),
            round(float(row.get("scanner_weight_pct") or 0), 2),
        )
        for row in report_candidates
    ]
    machine_core = [
        (
            int(row.get("rank") or 0),
            str(row.get("code") or ""),
            round(float(row.get("score") or 0), 2),
            round(float(row.get("scanner_weight_pct") or 0), 2),
        )
        for row in candidates
    ]
    if report_core != machine_core:
        errors.append("scanner report formal candidates do not equal scan.json")
    if (
        scanner_cash is None
        or report_cash is None
        or abs(scanner_cash - report_cash) > 0.05
    ):
        errors.append("scanner report cash weight does not equal scan.json")

    raw_scan_rows = [
        row
        for row in scan.get("rows", [])
        if isinstance(row, dict) and row.get("code")
    ]
    raw_scan_codes = [str(row.get("code")) for row in raw_scan_rows]
    if len(raw_scan_codes) != len(set(raw_scan_codes)):
        errors.append("scan rows contain duplicate codes")
    scan_rows = {str(row.get("code")): row for row in raw_scan_rows}
    for candidate in candidates:
        row = scan_rows.get(candidate["code"])
        if row is None:
            errors.append(f"scanner candidate {candidate['code']} missing from scan.json")
            continue
        row_score = _number(row.get("score"))
        if row_score is None or abs(row_score - candidate["score"]) > 0.11:
            errors.append(f"scanner score mismatch for {candidate['code']}")

    data_cutoff = _parse_datetime(scan.get("data_cutoff"))
    news_cutoff = _parse_datetime(evidence.get("news_cutoff"))
    if data_cutoff is None:
        errors.append("scanner data_cutoff is missing or invalid")
        data_cutoff = datetime.min
    report_cutoff = _extract_report_cutoff(report_text)
    if report_cutoff != data_cutoff:
        errors.append("scanner report data cutoff does not equal scan.json")
    signal_day = _parse_datetime(signal_date)
    if signal_day is None or data_cutoff.date() > signal_day.date():
        errors.append("scanner data cutoff is later than signal date")
    for code, row in scan_rows.items():
        row_date = _parse_datetime(row.get("actual_date"))
        if row_date is None:
            errors.append(f"scan row {code} actual_date is missing or invalid")
        elif row_date.date() > data_cutoff.date():
            errors.append(f"scan row {code} is later than scanner data cutoff")
    if news_cutoff is None:
        errors.append("evidence news cutoff is invalid")
        news_cutoff = datetime.min
    elif news_cutoff < data_cutoff:
        errors.append("evidence news cutoff is earlier than scanner cutoff")
    event = event_snapshot(load_event_ledger(DEFAULT_LEDGER), news_cutoff)
    if event.get("risk_level") not in {"正常", "黄色", "红色"}:
        errors.append("event risk level missing or invalid")
    event_review = _extract_event_snapshot(
        event_path.read_text(encoding="utf-8-sig")
    )
    if _parse_datetime(event_review.get("news_cutoff")) != news_cutoff:
        errors.append("event review news cutoff does not equal evidence cutoff")
    if event_review.get("risk_level") != event.get("risk_level"):
        errors.append("event review risk level does not equal event ledger snapshot")

    evidence_by_code, evidence_errors = _validate_evidence_packet(
        evidence,
        candidates=candidates,
        data_cutoff=data_cutoff,
        news_cutoff=news_cutoff,
    )
    errors.extend(evidence_errors)

    target_rows: list[dict] = []
    rejected_rows: list[dict] = []
    explanations: list[dict] = []
    blocked_scopes = set(event.get("blocked_scopes") or [])
    for candidate in candidates:
        item = evidence_by_code.get(candidate["code"], {})
        status = str(item.get("status") or "missing")
        rejection_reason = str(item.get("rejection_reason") or "")
        cluster = risk_cluster(
            candidate["direction"],
            candidate["industry"],
            candidate["name"],
        )
        candidate_scopes = {
            candidate["code"],
            candidate["direction"],
            candidate["industry"],
            cluster,
        }
        matched_blocked_scopes = sorted(candidate_scopes & blocked_scopes)
        if matched_blocked_scopes:
            status = "rejected"
            rejection_reason = (
                "EVENT_SCOPE_BLOCKED:"
                + ",".join(matched_blocked_scopes)
            )
        if event.get("risk_level") == "红色" and cluster == "高弹性成长":
            status = "rejected"
            rejection_reason = "EVENT_RED_HIGH_BETA"
        if status != "approved":
            rejected_rows.append(
                {
                    **candidate,
                    "evidence_status": status,
                    "rejection_reason": rejection_reason or "EVIDENCE_NOT_APPROVED",
                }
            )
            continue
        row = scan_rows.get(candidate["code"], {})
        try:
            execution = _deterministic_execution(candidate, row)
        except ValueError as exc:
            errors.append(str(exc))
            rejected_rows.append(
                {
                    **candidate,
                    "evidence_status": "rejected",
                    "rejection_reason": "EXECUTION_INPUT_INVALID",
                }
            )
            continue
        target_rows.append(
            {
                **candidate,
                "risk_cluster": cluster,
                "target_weight_pct": candidate["scanner_weight_pct"],
                "evidence_status": "approved",
                "execution": execution,
            }
        )
        explanations.append(
            {
                "code": candidate["code"],
                "thesis": str(item.get("thesis") or ""),
                "invalidation": str(item.get("invalidation") or ""),
                "catalyst": item.get("catalyst"),
            }
        )
    target_rows.sort(key=lambda row: row["rank"])
    normalized_weights = allocate_instrument_weights(market_state, target_rows)
    for row, weight in zip(target_rows, normalized_weights):
        row["target_weight_pct"] = float(weight)
    target_weight = round(sum(row["target_weight_pct"] for row in target_rows), 2)
    cash_pct = round(100.0 - target_weight, 2)
    if cash_pct < -0.05:
        errors.append("target weights exceed 100%")
    exposure_floor = float(get_position_cap(market_state).low)
    if target_weight + 0.05 < exposure_floor:
        errors.append(
            f"{market_state} approved exposure {target_weight:g}% is below "
            f"the {exposure_floor:g}% deployment floor"
        )

    account_weights, account_errors = _validate_account_state(
        account,
        allowed_codes=set(scan_rows),
        maximum_as_of=news_cutoff,
    )
    order_blockers = account_errors.copy()
    orders = [] if order_blockers else _derive_orders(account_weights, target_rows)

    manifest = load_current_manifest() or {}
    if not manifest:
        errors.append("current frozen strategy manifest is missing")
    else:
        manifest_ok, manifest_errors = verify_manifest(manifest)
        if not manifest_ok:
            errors.extend(
                f"frozen strategy: {error}" for error in manifest_errors
            )
        if scan.get("strategy_version") != manifest.get("strategy_version"):
            errors.append("scan strategy version does not equal current frozen strategy")
        if scan.get("strategy_sha256") != manifest.get("combined_sha256"):
            errors.append("scan strategy hash does not equal current frozen strategy")
    input_paths = {
        "scan": scan_path,
        "scanner_report": report_path,
        "event_ledger": DEFAULT_LEDGER,
        "evidence": evidence_path,
    }
    if event_path.exists():
        input_paths["event_review"] = event_path
    if account_path and account_path.exists():
        input_paths["account_state"] = account_path
    input_hashes = {
        label: {
            "path": _project_relative(path),
            "sha256": _sha256(path),
        }
        for label, path in input_paths.items()
    }
    portfolio_core = {
        "strategy_version": manifest.get("strategy_version"),
        "signal_date": signal_date,
        "data_cutoff": data_cutoff.strftime("%Y-%m-%d %H:%M"),
        "news_cutoff": news_cutoff.strftime("%Y-%m-%d %H:%M"),
        "targets": [
            {
                "rank": row["rank"],
                "code": row["code"],
                "weight_pct": row["target_weight_pct"],
                "entry": row["execution"],
            }
            for row in target_rows
        ],
        "cash_pct": cash_pct,
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "decision_type": "weekly_target",
        "status": "blocked" if errors else "ready",
        "blocking_errors": errors,
        "orders_status": "blocked" if order_blockers or errors else "ready",
        "order_blocking_errors": order_blockers,
        "strategy_version": manifest.get("strategy_version"),
        "strategy_sha256": manifest.get("combined_sha256"),
        "decision_engine_sha256": _sha256(Path(__file__)),
        "model_authority": MODEL_AUTHORITY,
        "as_of": {
            "signal_date": signal_date,
            "data_cutoff": data_cutoff.strftime("%Y-%m-%d %H:%M"),
            "news_cutoff": news_cutoff.strftime("%Y-%m-%d %H:%M"),
        },
        "market": {
            "state": market_state,
            "event_risk": event.get("risk_level"),
            "event_action": event.get("action"),
        },
        "scanner_candidates": candidates,
        "rejected_candidates": rejected_rows,
        "explanations": sorted(
            explanations,
            key=lambda row: next(
                item["rank"] for item in target_rows if item["code"] == row["code"]
            ),
        ),
        "portfolio": {
            "targets": target_rows,
            "cash_pct": cash_pct,
            "scanner_cash_pct_before_evidence": scanner_cash,
        },
        "orders": orders,
        "input_hashes": input_hashes,
        "portfolio_fingerprint": _canonical_hash(portfolio_core),
    }
    unsigned = dict(decision)
    decision["contract_sha256"] = _canonical_hash(unsigned)
    return decision


def _render_selection(decision: dict) -> str:
    if decision.get("status") != "ready":
        raise ValueError("blocked decision cannot be rendered as executable selection")
    data_cutoff = _parse_datetime(decision["as_of"]["data_cutoff"])
    news_cutoff = _parse_datetime(decision["as_of"]["news_cutoff"])
    if not data_cutoff or not news_cutoff:
        raise ValueError("decision cutoffs are invalid")
    execution_date = _next_weekday(data_cutoff)
    next_rebalance = data_cutoff + timedelta(days=7)
    market = decision["market"]
    targets = decision["portfolio"]["targets"]
    explanations = {
        row["code"]: row for row in decision.get("explanations", [])
    }
    lines = [
        f"# {data_cutoff:%Y-%m-%d} 周度ETF方案（确定性生成）",
        "",
        "- 调仓频率：周度",
        f"- 信号批次日期：{decision['as_of'].get('signal_date')}",
        f"- 数据截止时间：{data_cutoff:%Y-%m-%d %H:%M}",
        f"- 新闻截止时间：{news_cutoff:%Y-%m-%d %H:%M}",
        f"- 计划执行时间：{execution_date:%Y-%m-%d} 开盘",
        f"- 下次常规重排：{next_rebalance:%Y-%m-%d} 盘后",
        f"- 市场状态：{market['state']}",
        f"- 决策契约：`{decision['contract_sha256']}`",
        f"- 组合指纹：`{decision['portfolio_fingerprint']}`",
        "",
        "## 国际事件复核",
        "",
        f"- 新闻截止时间：{news_cutoff:%Y-%m-%d %H:%M}",
        f"- 国际事件风险：{market['event_risk']}",
        f"- 国际事件动作：{market.get('event_action') or '按结构化事件快照执行'}",
        "",
        "## 最终组合",
        "",
        "| 标的 | 方向 | 计划仓位 | 阶段 | 总分 | 买入区间 | 止损 | 目标 | 预期盈亏比 | 证据 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in targets:
        execution = row["execution"]
        explanation = explanations.get(row["code"], {})
        catalyst = explanation.get("catalyst") or {}
        source = (
            f"[{catalyst.get('source_name', '来源')}]"
            f"({catalyst.get('source_url', '')})"
        )
        lines.append(
            f"| {row['name']} `{row['code']}` | {row['direction']} "
            f"| {row['target_weight_pct']:g}% | {row['stage']} | {row['score']:.1f} "
            f"| {execution['entry_low']:g}-{execution['entry_high']:g} "
            f"| {execution['stop']:g} | {execution['target']:g} "
            f"| {execution['risk_reward']:.1f}:1 | {source} |"
        )
    lines.append(
        f"| 现金 | — | {decision['portfolio']['cash_pct']:g}% "
        "| — | — | — | — | — | — | 风险预算 |"
    )
    lines.extend(
        [
            "",
            "## 执行与失效",
            "",
        ]
    )
    for index, row in enumerate(targets, 1):
        explanation = explanations.get(row["code"], {})
        catalyst = explanation.get("catalyst") or {}
        lines.append(
            f"{index}. **{row['name']} `{row['code']}`**："
            f"{explanation.get('thesis', '')}；"
            f"催化/待发生事件：{catalyst.get('title', '')}"
            f"（{catalyst.get('event_date', '')}）；"
            f"失效条件：{explanation.get('invalidation', '')}"
        )
    if not targets:
        lines.append("1. 没有获得批准的候选，保持100%现金。")
    lines.extend(
        [
            "",
            "## 模型权限",
            "",
            "- 本文件由决策契约确定性渲染；模型不能改排名、仓位、代码或订单。",
            "- 任一输入哈希变化后，本文件立即失效，必须重新编译和校验。",
            "- 日度只处理硬退出或系统性降仓，不根据日度排名换票。",
            "",
        ]
    )
    return "\n".join(lines)


def _selection_evidence_v1(decision: dict) -> dict:
    explanations = {
        row["code"]: row for row in decision.get("explanations", [])
    }
    return {
        "schema_version": 1,
        "data_cutoff": decision["as_of"]["data_cutoff"],
        "news_cutoff": decision["as_of"]["news_cutoff"],
        "holdings": [
            {
                "code": row["code"],
                "action": "new",
                "thesis": explanations.get(row["code"], {}).get("thesis", ""),
                "invalidation": explanations.get(row["code"], {}).get(
                    "invalidation", ""
                ),
                "catalyst": explanations.get(row["code"], {}).get("catalyst"),
            }
            for row in decision["portfolio"]["targets"]
        ],
    }


def verify_decision(decision_path: Path, *, verify_inputs: bool = True) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        decision = _read_json(decision_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"decision cannot be read: {exc}"]
    if decision.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"decision schema_version must be {SCHEMA_VERSION}")
    if decision.get("status") != "ready":
        errors.append("decision status is not ready")
    if decision.get("orders_status") != "ready":
        errors.append("decision orders_status is not ready")
    if decision.get("decision_engine_sha256") != _sha256(Path(__file__)):
        errors.append("decision engine changed after compilation")
    manifest = load_current_manifest() or {}
    if not manifest:
        errors.append("current frozen strategy manifest is missing")
    else:
        if decision.get("strategy_version") != manifest.get("strategy_version"):
            errors.append("decision strategy version is no longer current")
        if decision.get("strategy_sha256") != manifest.get("combined_sha256"):
            errors.append("decision strategy hash is no longer current")
        manifest_ok, manifest_errors = verify_manifest(manifest)
        if not manifest_ok:
            errors.extend(
                f"frozen strategy: {error}" for error in manifest_errors
            )
    expected_contract = str(decision.get("contract_sha256") or "")
    unsigned = dict(decision)
    unsigned.pop("contract_sha256", None)
    if _canonical_hash(unsigned) != expected_contract:
        errors.append("decision contract hash mismatch")

    portfolio = decision.get("portfolio")
    if not isinstance(portfolio, dict):
        return False, errors + ["portfolio object is missing"]
    targets = portfolio.get("targets")
    if not isinstance(targets, list):
        return False, errors + ["portfolio targets must be an array"]
    codes = [str(row.get("code") or "") for row in targets if isinstance(row, dict)]
    if len(codes) != len(set(codes)):
        errors.append("duplicate target codes")
    total = sum(float(row.get("target_weight_pct") or 0) for row in targets)
    cash = float(portfolio.get("cash_pct") or 0)
    if abs(total + cash - 100.0) > 0.05:
        errors.append("target weights plus cash do not equal 100%")
    portfolio_core = {
        "strategy_version": decision.get("strategy_version"),
        "signal_date": decision.get("as_of", {}).get("signal_date"),
        "data_cutoff": decision.get("as_of", {}).get("data_cutoff"),
        "news_cutoff": decision.get("as_of", {}).get("news_cutoff"),
        "targets": [
            {
                "rank": row.get("rank"),
                "code": row.get("code"),
                "weight_pct": row.get("target_weight_pct"),
                "entry": row.get("execution"),
            }
            for row in targets
        ],
        "cash_pct": cash,
    }
    if _canonical_hash(portfolio_core) != decision.get("portfolio_fingerprint"):
        errors.append("portfolio fingerprint mismatch")

    if verify_inputs:
        for label, item in (decision.get("input_hashes") or {}).items():
            if not isinstance(item, dict):
                errors.append(f"input hash entry {label} is invalid")
                continue
            path = PROJECT_ROOT / str(item.get("path") or "")
            if not path.is_file():
                errors.append(f"input {label} no longer exists")
            elif _sha256(path) != item.get("sha256"):
                errors.append(f"input {label} changed after decision compilation")
    return not errors, errors


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile/verify weekly decision contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    build_parser.add_argument("--evidence", required=True)
    build_parser.add_argument("--account-state")
    build_parser.add_argument("--output")
    build_parser.add_argument("--selection-output")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("decision")
    verify_parser.add_argument("--skip-input-hashes", action="store_true")

    args = parser.parse_args()
    if args.command == "verify":
        ok, errors = verify_decision(
            _resolve_project_path(args.decision),
            verify_inputs=not args.skip_input_hashes,
        )
        if ok:
            print(f"OK: {args.decision}")
            return
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    day_dir = (PROJECT_ROOT / "codex" / "stock" / args.date).resolve()
    output = (
        _resolve_project_path(args.output)
        if args.output
        else day_dir / "decision.json"
    )
    evidence_path = _resolve_project_path(args.evidence)
    account_path = (
        _resolve_project_path(args.account_state)
        if args.account_state
        else None
    )
    decision = build_decision(
        day_dir=day_dir,
        evidence_path=evidence_path,
        account_path=account_path,
    )
    _write_json(output, decision)
    print(f"decision: {output}")
    print(f"status: {decision['status']}; orders: {decision['orders_status']}")
    if decision["blocking_errors"]:
        for error in decision["blocking_errors"]:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(2)

    selection_output = (
        _resolve_project_path(args.selection_output)
        if args.selection_output
        else day_dir / "selection.generated.md"
    )
    selection_output.write_text(_render_selection(decision), encoding="utf-8")
    normalized_evidence = selection_output.with_name("selection_evidence.json")
    _write_json(normalized_evidence, _selection_evidence_v1(decision))
    print(f"selection: {selection_output}")
    print(f"evidence sidecar: {normalized_evidence}")
    if decision["orders_status"] != "ready":
        for error in decision["order_blocking_errors"]:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
