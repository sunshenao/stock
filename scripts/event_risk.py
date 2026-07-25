"""Point-in-time international event risk layer for weekly selection and daily control.

The ledger is maintained separately from price signals. Positive news may confirm
an existing candidate, but only negative events can veto or reduce risk. This
prevents subjective headlines from manufacturing a buy signal.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LEDGER = PROJECT_ROOT / "codex" / "stock" / "event_ledger.csv"
REQUIRED_COLUMNS = {
    "known_at",
    "event_type",
    "scope",
    "score",
    "horizon_days",
    "source",
    "url",
    "summary",
}
GLOBAL_VETO_SCORE = -0.65
SCOPE_VETO_SCORE = -0.55
WATCH_SCORE = -0.30
HARD_EVENT_TYPES = {
    "official_systemic",
    "emergency_policy",
    "war_escalation",
    "market_disruption",
}


@dataclass(frozen=True)
class Event:
    known_at: datetime
    event_type: str
    scope: str
    score: float
    horizon_days: int
    source: str
    url: str
    summary: str


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def load_event_ledger(path: Path = DEFAULT_LEDGER) -> list[Event]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"事件台账缺少字段: {', '.join(sorted(missing))}")
        events = []
        seen = set()
        for line_no, row in enumerate(reader, 2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            known_at = _parse_datetime(row["known_at"])
            score = float(row["score"])
            horizon_days = int(row["horizon_days"])
            if not -1.0 <= score <= 1.0:
                raise ValueError(f"事件台账第{line_no}行 score 必须在[-1,1]")
            if horizon_days <= 0:
                raise ValueError(f"事件台账第{line_no}行 horizon_days 必须>0")
            key = (
                known_at.isoformat(),
                row["scope"].strip(),
                row["summary"].strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(Event(
                known_at=known_at,
                event_type=row["event_type"].strip(),
                scope=row["scope"].strip() or "global",
                score=score,
                horizon_days=horizon_days,
                source=row["source"].strip(),
                url=row["url"].strip(),
                summary=row["summary"].strip(),
            ))
    return sorted(events, key=lambda event: event.known_at)


def event_snapshot(events: list[Event], as_of: datetime) -> dict:
    active = []
    risk_scores: dict[str, float] = {}
    confirmation_scores: dict[str, float] = {}
    negative_sources: dict[str, set[str]] = {}
    hard_scopes: set[str] = set()
    for event in events:
        if event.known_at > as_of:
            continue
        age_days = max((as_of - event.known_at).total_seconds() / 86400, 0.0)
        if age_days > event.horizon_days:
            continue
        decay = math.exp(-math.log(2) * age_days / max(event.horizon_days / 2, 1))
        effective_score = event.score * decay
        active.append((event, effective_score))
        if effective_score < 0:
            risk_scores[event.scope] = max(
                -1.0,
                risk_scores.get(event.scope, 0.0) + effective_score,
            )
            source_key = event.source or event.url or event.summary
            negative_sources.setdefault(event.scope, set()).add(source_key)
            if event.event_type in HARD_EVENT_TYPES:
                hard_scopes.add(event.scope)
        elif effective_score > 0:
            confirmation_scores[event.scope] = min(
                1.0,
                confirmation_scores.get(event.scope, 0.0) + effective_score,
            )

    corroborated_scopes = {
        scope
        for scope, sources in negative_sources.items()
        if len(sources) >= 2 or scope in hard_scopes
    }
    global_score = risk_scores.get("global", 0.0)
    blocked_scopes = sorted(
        scope
        for scope, score in risk_scores.items()
        if (
            scope != "global"
            and score <= SCOPE_VETO_SCORE
            and scope in corroborated_scopes
        )
    )
    watch_scopes = sorted(
        scope
        for scope, score in risk_scores.items()
        if scope != "global" and score <= WATCH_SCORE and scope not in blocked_scopes
    )
    if global_score <= GLOBAL_VETO_SCORE and "global" in corroborated_scopes:
        risk_level = "红色"
        action = "禁止新增高弹性仓位；现有仓位进入日度硬风控复核"
    elif global_score <= WATCH_SCORE or blocked_scopes or watch_scopes:
        risk_level = "黄色"
        action = "不提高总风险预算；被否决风险簇禁止新开，观察项须价格与资金双确认"
    else:
        risk_level = "正常"
        action = "事件层不否决，但利好不能单独产生买入信号"

    return {
        "as_of": as_of,
        "active": active,
        "scope_scores": risk_scores,
        "risk_scores": risk_scores,
        "confirmation_scores": confirmation_scores,
        "negative_sources": {
            scope: sorted(sources)
            for scope, sources in negative_sources.items()
        },
        "corroborated_scopes": sorted(corroborated_scopes),
        "global_score": global_score,
        "blocked_scopes": blocked_scopes,
        "watch_scopes": watch_scopes,
        "risk_level": risk_level,
        "action": action,
    }


def render_event_review(snapshot: dict) -> str:
    as_of = snapshot["as_of"]
    lines = [
        f"# 国际事件风险复核 - {as_of:%Y-%m-%d %H:%M}",
        "",
        f"- 风险级别：**{snapshot['risk_level']}**",
        f"- 全球风险分：{snapshot['global_score']:+.2f}",
        f"- 风险动作：{snapshot['action']}",
        "- 使用原则：负面事件可以否决或降级；正面事件只能验证已有价格信号。",
        "- 红色门槛：至少两个独立来源交叉验证，或属于官方/系统性硬事件；单一普通报道最多判黄色。",
        "",
        "## 有效事件",
        "",
    ]
    if not snapshot["active"]:
        lines.append("没有处于有效期内的已归档事件；视为新闻覆盖不足，不视为风险为零。")
    else:
        lines.extend([
            "| 可知时间 | 范围 | 原始分 | 衰减分 | 有效期 | 来源 | 摘要 |",
            "|---|---|---:|---:|---:|---|---|",
        ])
        for event, effective_score in snapshot["active"]:
            source = (
                f"[{event.source}]({event.url})"
                if event.url
                else event.source
            )
            lines.append(
                f"| {event.known_at:%Y-%m-%d %H:%M} | {event.scope} "
                f"| {event.score:+.2f} | {effective_score:+.2f} "
                f"| {event.horizon_days}日 | {source} | {event.summary} |"
            )
    lines.extend(["", "## 负面风险分数", ""])
    if snapshot["risk_scores"]:
        corroborated = set(snapshot["corroborated_scopes"])
        for scope, score in sorted(snapshot["risk_scores"].items()):
            sources = len(snapshot["negative_sources"].get(scope, []))
            status = "已交叉验证" if scope in corroborated else "单一来源观察"
            lines.append(f"- {scope}: {score:+.2f}（{sources}个独立来源，{status}）")
    else:
        lines.append("- 无")
    lines.extend(["", "## 正面验证分数", ""])
    if snapshot["confirmation_scores"]:
        for scope, score in sorted(snapshot["confirmation_scores"].items()):
            lines.append(f"- {scope}: {score:+.2f}（只作验证，不抵消负面风险）")
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def write_event_review(
    as_of: datetime,
    output_path: Path,
    ledger_path: Path = DEFAULT_LEDGER,
) -> tuple[Path, dict]:
    snapshot = event_snapshot(load_event_ledger(ledger_path), as_of)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_event_review(snapshot), encoding="utf-8")
    return output_path, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="国际事件风险台账的时点化复核")
    parser.add_argument("--date", required=True, help="复核时间 YYYY-MM-DD 或 ISO datetime")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    as_of = _parse_datetime(args.date)
    if len(args.date.strip()) == 10:
        as_of = as_of.replace(hour=23, minute=59, second=59)
    path, snapshot = write_event_review(
        as_of,
        Path(args.output),
        Path(args.ledger),
    )
    print(f"事件风险: {snapshot['risk_level']} / global={snapshot['global_score']:+.2f}")
    print(f"报告: {path}")


if __name__ == "__main__":
    main()
