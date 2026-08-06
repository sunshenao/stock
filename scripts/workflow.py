"""Unified monthly, weekly, and daily ETF workflow.

Monthly reviews set the direction map, weekly runs create the only executable
selection, and daily runs monitor risk without rotating into new instruments.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from event_risk import DEFAULT_LEDGER, write_event_review
from daily_risk import detect_cluster_crashes
from risk_rules import get_target_exposure, risk_cluster
from decision_contract import verify_decision
from strategy_version import load_current_manifest, verify_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STOCK_DIR = PROJECT_ROOT / "codex" / "stock"

MODE_OUTPUTS = {
    "monthly": "monthly_review.md",
    "weekly": "selection.generated.md",
    "daily": "risk_review.md",
}


def _record_assumed_formal_recommendation(
    decision_path: Path,
    account_path: Path,
) -> dict:
    """Persist a verified executable recommendation as the assumed account."""
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    if decision.get("status") != "ready" or decision.get("orders_status") != "ready":
        raise ValueError("only ready executable decisions may update account state")

    portfolio = decision.get("portfolio") or {}
    targets = portfolio.get("targets")
    if not isinstance(targets, list):
        raise ValueError("decision portfolio targets must be an array")

    cutoff = str((decision.get("as_of") or {}).get("data_cutoff") or "")
    contract_hash = str(decision.get("contract_sha256") or "")
    positions = []
    for row in targets:
        execution = row.get("execution") or {}
        reference_close = execution.get("reference_close")
        positions.append(
            {
                "code": str(row["code"]),
                "name": str(row.get("name") or ""),
                "weight_pct": float(row["target_weight_pct"]),
                "entry_price": (
                    float(reference_close) if reference_close is not None else None
                ),
                "entry_price_source": "decision_reference_close",
                "assumed_fill_time": cutoff,
                "execution_status": "assumed_filled_on_recommendation",
                "recommendation_contract": contract_hash,
            }
        )

    account = {
        "schema_version": 1,
        "as_of": cutoff,
        "confirmed": True,
        "confirmed_by": "user_auto_assume_recommendations",
        "assumption_mode": "auto_assume_formal_recommendations_filled",
        "execution_price_policy": (
            "formal executable recommendations are booked immediately at the "
            "decision reference close; user-reported broker fills take precedence"
        ),
        "source_instruction_date": "2026-07-27",
        "cash_pct": float(portfolio.get("cash_pct") or 0.0),
        "positions": positions,
        "last_recommendation": {
            "signal_date": str((decision.get("as_of") or {}).get("signal_date") or ""),
            "strategy_version": str(decision.get("strategy_version") or ""),
            "decision_contract": contract_hash,
            "portfolio_fingerprint": str(
                decision.get("portfolio_fingerprint") or ""
            ),
            "decision_path": str(decision_path.resolve()),
        },
    }
    account_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = account_path.with_suffix(account_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(account_path)
    return account


def _pool_codes() -> set[str]:
    text = (SCRIPT_DIR / "etf.txt").read_text(encoding="utf-8-sig")
    return set(re.findall(r"(?<!\d)(\d{6})(?!\d)", text))


def _validate_scan(scan_path: Path, target_date: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        data = json.loads(scan_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"扫描文件无法读取: {exc}"]

    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("扫描结果为空")
        return False, errors

    scan_date = str(data.get("date") or data.get("scan_date") or "")
    if scan_date and scan_date != target_date:
        errors.append(f"扫描日期 {scan_date} 与目标日期 {target_date} 不一致")
    if data.get("schema_version") != 2:
        errors.append("scan.json schema_version 必须为 2")
    cutoff_text = str(data.get("data_cutoff") or "")
    try:
        cutoff = datetime.fromisoformat(cutoff_text)
        target = datetime.strptime(target_date, "%Y-%m-%d")
        if cutoff.date() > target.date():
            errors.append("扫描数据截止时间晚于目标日期")
    except ValueError:
        errors.append("scan.json 缺少有效 data_cutoff")

    manifest = load_current_manifest() or {}
    manifest_ok, manifest_errors = verify_manifest(manifest)
    if not manifest or not manifest_ok:
        errors.extend(f"冻结策略无效: {error}" for error in manifest_errors)
        if not manifest:
            errors.append("当前冻结策略清单缺失")
    else:
        if data.get("strategy_version") != manifest.get("strategy_version"):
            errors.append("扫描策略版本不是当前冻结版本")
        if data.get("strategy_sha256") != manifest.get("combined_sha256"):
            errors.append("扫描策略哈希不是当前冻结版本")

    allowed = _pool_codes()
    codes = {str(row.get("code") or "") for row in rows if row.get("code")}
    outside = sorted(code for code in codes if code not in allowed)
    if outside:
        errors.append(f"扫描出现池外代码: {', '.join(outside)}")
    if len(codes) < 3:
        errors.append(f"有效标的不足3只，当前 {len(codes)} 只")
    return not errors, errors


def _market_state_from_report(report_path: Path) -> str:
    text = report_path.read_text(encoding="utf-8-sig")
    match = re.search(r"\|\s*市场状态\s*\|\s*\*{0,2}([^|*\r\n]+)", text)
    return match.group(1).strip() if match else "未知"


def _confirmed_positions() -> tuple[list[dict], bool]:
    account_path = STOCK_DIR / "account_state.json"
    if account_path.exists():
        try:
            account = json.loads(account_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            account = None
        if isinstance(account, dict) and account.get("confirmed") is True:
            rows = account.get("positions")
            if isinstance(rows, list):
                positions = [
                    {
                        "code": str(row.get("code") or ""),
                        "name": str(row.get("name") or ""),
                        "position": f"{float(row.get('weight_pct') or 0):g}%",
                        "source": "结构化账户",
                    }
                    for row in rows
                    if (
                        isinstance(row, dict)
                        and re.fullmatch(r"\d{6}", str(row.get("code") or ""))
                        and float(row.get("weight_pct") or 0) > 0
                    )
                ]
                return positions, True

    path = STOCK_DIR / "current_positions.md"
    if not path.exists():
        return [], False
    positions = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[1] != "持有":
            continue
        code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", cells[0])
        if not code_match:
            continue
        positions.append({
            "code": code_match.group(1),
            "name": re.sub(r"`?\d{6}`?", "", cells[0]).strip(),
            "position": cells[2],
            "source": "实际账户",
        })
    return positions, bool(positions)


def _latest_weekly_model(target_date: str, max_age_days: int = 8) -> tuple[Path | None, list[dict]]:
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    candidates = []
    for path in STOCK_DIR.glob("????-??-??/decision.json"):
        try:
            plan_date = datetime.strptime(path.parent.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        age = (target - plan_date).days
        if 0 <= age <= max_age_days:
            candidates.append((plan_date, path))
    for _, path in sorted(candidates, reverse=True):
        ok, _ = verify_decision(path)
        if not ok:
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if (
            payload.get("status") != "ready"
            or payload.get("orders_status") != "ready"
        ):
            continue
        holdings = [
            {
                "code": str(holding.get("code") or ""),
                "name": str(holding.get("name") or ""),
                "position": f"{float(holding.get('target_weight_pct') or 0):g}%",
                "source": "周度模型",
            }
            for holding in payload.get("portfolio", {}).get("targets", [])
            if float(holding.get("target_weight_pct") or 0) > 0
        ]
        if holdings:
            return path, holdings
    return None, []


def _write_daily_risk_review(
    target_date: str,
    scan_path: Path,
    report_path: Path,
    output_path: Path,
    event_snapshot: dict,
) -> Path:
    scan = json.loads(scan_path.read_text(encoding="utf-8-sig"))
    rows = scan.get("rows") or []
    by_code = {str(row.get("code")): row for row in rows if row.get("code")}
    market_state = _market_state_from_report(report_path)
    positions, account_authoritative = _confirmed_positions()
    selection_path = None
    if not positions and not account_authoritative:
        selection_path, positions = _latest_weekly_model(target_date)
    cluster_crashes = detect_cluster_crashes(rows)
    blocked_scopes = set(event_snapshot.get("blocked_scopes") or [])
    global_red = event_snapshot.get("risk_level") == "红色"

    decisions = []
    hard_action = False
    for position in positions:
        row = by_code.get(position["code"])
        if not row:
            decisions.append({
                **position,
                "stage": "数据缺失",
                "pct": None,
                "decision": "人工核验",
                "reason": "当日扫描无该标的，不能凭缺失数据换仓",
            })
            continue
        cluster = risk_cluster(
            str(row.get("sector") or "").split("/", 1)[0],
            row.get("industry", "其他"),
            row.get("name", ""),
        )
        pct = float(row.get("pct") or 0)
        ret20 = float(row.get("ret_20d") or 0)
        stage = str(row.get("stage") or "")
        reasons = []
        decision = "延续周度组合"
        if pct <= -7:
            decision = "硬退出复核"
            reasons.append(f"单日{pct:+.2f}%达到极端下跌阈值")
            hard_action = True
        if stage in {"衰弱期", "弱势期", "衰竭期"} and ret20 < 0:
            decision = "硬退出复核"
            reasons.append(f"{stage}且20日趋势{ret20:+.1f}%")
            hard_action = True
        if cluster in cluster_crashes:
            decision = "降仓/退出复核"
            reasons.append(f"{cluster}至少3个独立细分方向单日跌超5%")
            hard_action = True
        if cluster in blocked_scopes and (pct <= -3 or stage in {"衰弱期", "弱势期", "衰竭期"}):
            decision = "降仓/退出复核"
            reasons.append(f"国际事件风险簇{cluster}被否决且价格确认走弱")
            hard_action = True
        elif cluster in blocked_scopes:
            reasons.append(f"国际事件风险簇{cluster}被否决，禁止加仓")
        if global_red:
            reasons.append("全球事件风险为红色，禁止新增高弹性仓位")
        decisions.append({
            **position,
            "stage": stage,
            "pct": pct,
            "decision": decision,
            "reason": "；".join(reasons) if reasons else "未触发日度硬条件",
        })

    target_exposure = get_target_exposure(market_state)
    if market_state == "冰点":
        hard_action = True
    source_text = (
        "结构化账户（用户授权：正式推荐视为成交）"
        if account_authoritative
        else (
            f"周度模型 {selection_path.relative_to(PROJECT_ROOT)}"
            if selection_path
            else "无已确认实际持仓，且近8日无通过校验的周度组合"
        )
    )
    lines = [
        f"# 日度风险复核 - {target_date}",
        "",
        "- 决策频率：日度风控",
        "- 调仓权限：仅硬退出或降仓，禁止日度换票",
        "- 执行时点：盘后复核，动作最早在下一可交易时点执行",
        f"- 持仓来源：{source_text}",
        f"- 市场状态：{market_state}",
        f"- 市场目标仓位上限：{target_exposure}%",
        f"- 国际事件风险：{event_snapshot.get('risk_level', '未知')}",
        "",
        "## 结论",
        "",
    ]
    if not positions:
        lines.append("没有可核验持仓，不能生成虚假的止损或换仓指令。")
    elif hard_action:
        lines.append("存在硬风控触发项；只处理下表原持仓，不引入新标的。")
    else:
        lines.append("延续周度组合；当日排名变化不构成换仓理由。")
    lines.extend(["", "## 持仓检查", ""])
    if decisions:
        lines.extend([
            "| 标的 | 来源 | 原计划仓位 | 当日涨跌 | 阶段 | 决策 | 原因 |",
            "|---|---|---:|---:|---|---|---|",
        ])
        for item in decisions:
            pct_text = "—" if item["pct"] is None else f"{item['pct']:+.2f}%"
            lines.append(
                f"| {item['name']} `{item['code']}` | {item['source']} "
                f"| {item['position']} | {pct_text} | {item['stage']} "
                f"| {item['decision']} | {item['reason']} |"
            )
    else:
        lines.append("无持仓可检查。")
    lines.extend([
        "",
        "## 日度权限边界",
        "",
        "- 允许：执行既定止损、催化证伪退出、冰点降仓、大风险簇降仓。",
        "- 禁止：根据今日TOP排名卖旧买新、重新生成0–3只组合、提高未列入周度方案的标的仓位。",
        "- 国际利好只能验证周度候选；国际利空必须同时得到价格或资金走弱确认后才触发持仓动作，全球红色风险除外。",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _run_scan(args: argparse.Namespace, target_date: str) -> int:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "etf_analyzer.py"),
        "--date",
        target_date,
        "--top",
        str(args.top),
        "--workers",
        str(args.workers),
    ]
    if not args.no_hithink:
        command.append("--hithink")
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def _print_next_step(mode: str, target_date: str, output_path: Path) -> None:
    relative = output_path.relative_to(PROJECT_ROOT)
    if mode == "monthly":
        print("\n月度职责：只更新未来1至3个月的产业方向、核心催化和反证。")
        print(f"记录位置: {relative}")
        print("不得在月度复核中直接生成可执行调仓。")
    elif mode == "weekly":
        print("\n周度职责：扫描器和决策契约是唯一候选、仓位与订单口径。")
        print(f"正式方案: {relative}")
        print("模型只能起草证据；selection.generated.md 必须由 decision.json 确定性生成。")
        print(f"校验命令: python scripts/selection_guard.py {relative}")
    else:
        print("\n日度职责：只检查持仓止损、催化证伪、大簇崩盘和市场冰点。")
        print(f"风险记录: {relative}")
        print("未触发硬退出或降仓条件时，必须延续周度组合，不得按日换票。")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF 月度方向、周度选股、日度风控统一入口")
    parser.add_argument("--mode", choices=MODE_OUTPUTS, default="daily")
    parser.add_argument("--date", help="目标日期 YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=30, help="扫描报告保留前N名")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-hithink", action="store_true")
    parser.add_argument(
        "--event-ledger",
        default=str(DEFAULT_LEDGER),
        help="时点化国际事件台账CSV",
    )
    parser.add_argument(
        "--news-cutoff",
        help="国际事件截止时间（ISO datetime）；历史复盘默认目标日23:59",
    )
    parser.add_argument(
        "--reuse-scan",
        action="store_true",
        help="已有同日 scan.json 时直接校验并复用",
    )
    parser.add_argument(
        "--evidence",
        help="周度证据JSON，默认 codex/stock/YYYY-MM-DD/weekly_evidence.json",
    )
    parser.add_argument(
        "--account-state",
        default=str(STOCK_DIR / "account_state.json"),
        help="用户确认的结构化账户状态",
    )
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    day_dir = STOCK_DIR / target_date
    scan_path = day_dir / "scan.json"
    report_path = day_dir / "etf_scan.md"

    can_reuse = args.reuse_scan and scan_path.exists() and report_path.exists()
    if not can_reuse:
        return_code = _run_scan(args, target_date)
        if return_code:
            raise SystemExit(return_code)

    if not report_path.exists() or report_path.stat().st_size == 0:
        print(f"扫描报告缺失或为空: {report_path}", file=sys.stderr)
        raise SystemExit(2)

    ok, errors = _validate_scan(scan_path, target_date)
    if not ok:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(3)

    output_path = day_dir / MODE_OUTPUTS[args.mode]
    if args.news_cutoff:
        news_cutoff = datetime.fromisoformat(args.news_cutoff)
    else:
        target_day = datetime.strptime(target_date, "%Y-%m-%d")
        now = datetime.now().replace(second=0, microsecond=0)
        news_cutoff = (
            target_day + timedelta(hours=23, minutes=59)
            if target_day.date() < now.date()
            else now
        )
    event_path, event_snapshot = write_event_review(
        news_cutoff,
        day_dir / "event_risk.md",
        Path(args.event_ledger),
    )
    print(
        f"国际事件复核: {event_snapshot['risk_level']} "
        f"(global={event_snapshot['global_score']:+.2f}) -> "
        f"{event_path.relative_to(PROJECT_ROOT)}"
    )
    if args.mode == "daily":
        _write_daily_risk_review(
            target_date,
            scan_path,
            report_path,
            output_path,
            event_snapshot,
        )
    elif args.mode == "weekly":
        evidence_path = (
            Path(args.evidence)
            if args.evidence
            else day_dir / "weekly_evidence.json"
        )
        if not evidence_path.is_absolute():
            evidence_path = PROJECT_ROOT / evidence_path
        account_path = Path(args.account_state)
        if not account_path.is_absolute():
            account_path = PROJECT_ROOT / account_path
        if not evidence_path.exists():
            print(
                "周度流程失败关闭：缺少 weekly_evidence.json。"
                "模型手写 selection.md 不会被采用。",
                file=sys.stderr,
            )
            print(f"需要证据文件: {evidence_path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
            raise SystemExit(4)
        command = [
            sys.executable,
            str(SCRIPT_DIR / "decision_contract.py"),
            "build",
            "--date",
            target_date,
            "--evidence",
            str(evidence_path),
            "--account-state",
            str(account_path),
        ]
        compiled = subprocess.run(command, cwd=PROJECT_ROOT)
        if compiled.returncode:
            raise SystemExit(compiled.returncode)
    print(f"扫描校验通过: {scan_path.relative_to(PROJECT_ROOT)}")
    _print_next_step(args.mode, target_date, output_path)

    if args.mode == "weekly" and not output_path.exists():
        print(
            f"周度流程未完成：缺少正式方案 {output_path.relative_to(PROJECT_ROOT)}。"
            " etf_scan.md 只是全池扫描，不能冒充周度选股结果。",
            file=sys.stderr,
        )
        raise SystemExit(4)
    if args.mode == "weekly":
        decision_path = day_dir / "decision.json"
        contract = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "decision_contract.py"),
                "verify",
                str(decision_path),
            ],
            cwd=PROJECT_ROOT,
        )
        if contract.returncode:
            raise SystemExit(contract.returncode)
        guard = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "selection_guard.py"), str(output_path)],
            cwd=PROJECT_ROOT,
        )
        if guard.returncode == 0:
            _record_assumed_formal_recommendation(decision_path, account_path)
            print(
                "正式推荐已按用户授权视为成交并写入账户: "
                f"{account_path.relative_to(PROJECT_ROOT)}"
            )
        raise SystemExit(guard.returncode)


if __name__ == "__main__":
    main()
