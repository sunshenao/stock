"""
Selection execution guard.

Checks hard portfolio constraints in a generated selection.md before it can be
treated as an executable plan.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from risk_rules import (  # noqa: E402
    TARGET_SELECTION_COUNT,
    get_position_cap,
    requires_aggressive_deployment,
    risk_cluster,
    single_position_cap,
)
from etf_analyzer import load_etf_txt  # noqa: E402


MARKET_STATES = ("主升", "震荡", "退潮", "退潮末期", "冰点", "未知")
FORMAL_FREQUENCY = "周度"
MAIN_POSITION_MIN = 20
CONSERVATIVE_POSITION_CAP = 15
PROBE_POSITION_CAP = 10
MIN_MAIN_ETF_AMOUNT_YI = 1.0
CHASE_5D_RETURN_LIMIT = 25.0
CHASE_SINGLE_DAY_LIMIT = 7.0
RETREAT_SINGLE_DAY_NO_CATALYST_LIMIT = 5.0
RETREAT_RET20_NEW_POSITION_LIMIT = 15.0
RETREAT_RET20_HOLD_ONLY_LIMIT = 25.0
RETREAT_RET20_HIGH_RISK_LIMIT = 35.0
WEEKLY_ALLOWED_WEEKDAYS = {0, 4, 5, 6}  # 周一盘前，或周五至周末
EVIDENCE_FILENAME = "selection_evidence.json"


@dataclass(frozen=True)
class Holding:
    instrument: str
    direction: str
    position_low: float
    position_high: float
    row_text: str

    @property
    def is_cash(self) -> bool:
        return "现金" in self.instrument

    @property
    def code(self) -> str | None:
        match = re.search(r"`?(\d{6})`?", self.instrument)
        return match.group(1) if match else None

    @property
    def is_etf_like(self) -> bool:
        return _is_etf_like(self.instrument)

    @property
    def is_qdii_like(self) -> bool:
        return _is_qdii_like(self.instrument, self.direction)


def _extract_market_state(text: str) -> str | None:
    patterns = [
        r"\|\s*市场状态\s*\|\s*\*{0,2}(主升|震荡|退潮末期|退潮|冰点|未知)\*{0,2}\s*\|",
        r"(?:市场状态|状态)\s*[：:]\s*\*{0,2}(主升|震荡|退潮末期|退潮|冰点|未知)\*{0,2}",
        r"-\s*(?:市场状态|状态)\s*[：:]\s*\*{0,2}(主升|震荡|退潮末期|退潮|冰点|未知)\*{0,2}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    for state in MARKET_STATES:
        if f"市场状态 | **{state}**" in text:
            return state
    return None


def _extract_frequency(text: str) -> str | None:
    match = re.search(r"(?:调仓频率|决策频率|推荐频率)\s*[：:|]\s*\*{0,2}(日度|周度|月度)\*{0,2}", text)
    return match.group(1) if match else None


def _cash_range_from_line(line: str) -> tuple[float, float] | None:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:[-~至]\s*(\d+(?:\.\d+)?))?\s*%", line)
    if not matches:
        return None
    ranges = []
    for low, high in matches:
        lo = float(low)
        hi = float(high) if high else lo
        ranges.append((lo, hi))
    return max(ranges, key=lambda item: item[1])


def _extract_cash_range(text: str) -> tuple[float, float] | None:
    for holding in _extract_portfolio_holdings(text):
        if holding.is_cash:
            return holding.position_low, holding.position_high

    for line in text.splitlines():
        if "|" not in line or "现金" not in line:
            continue
        if "---" in line:
            continue
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] != "现金":
            continue
        cash_range = _cash_range_from_line(line)
        if cash_range:
            return cash_range
    return None


def _split_row(line: str) -> list[str]:
    return [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _normalize_header(header: str) -> str:
    return re.sub(r"\s+", "", header.strip().strip("*"))


def _is_portfolio_header(headers: list[str]) -> bool:
    normalized = [_normalize_header(header) for header in headers]
    if "标的" not in normalized:
        return False
    if any(header in normalized for header in ("原仓位", "移出原因")):
        return False
    return any(header in normalized for header in ("计划仓位", "最终仓位", "目标仓位", "建议仓位"))


def _position_column(headers: list[str]) -> int | None:
    normalized = [_normalize_header(header) for header in headers]
    for name in ("计划仓位", "最终仓位", "目标仓位", "建议仓位"):
        if name in normalized:
            return normalized.index(name)
    return None


def _column(headers: list[str], name: str) -> int | None:
    normalized = [_normalize_header(header) for header in headers]
    return normalized.index(name) if name in normalized else None


def _extract_portfolio_holdings(text: str) -> list[Holding]:
    holdings = []
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        if not lines[i].lstrip().startswith("|") or not _is_separator(lines[i + 1]):
            i += 1
            continue

        headers = _split_row(lines[i])
        if not _is_portfolio_header(headers):
            i += 1
            continue

        instrument_idx = _column(headers, "标的")
        direction_idx = _column(headers, "方向")
        position_idx = _position_column(headers)
        if instrument_idx is None or position_idx is None:
            i += 1
            continue

        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            if _is_separator(lines[i]):
                i += 1
                continue
            cells = _split_row(lines[i])
            if len(cells) <= max(instrument_idx, position_idx):
                i += 1
                continue
            position = _cash_range_from_line(cells[position_idx])
            if not position:
                i += 1
                continue
            direction = cells[direction_idx] if direction_idx is not None and len(cells) > direction_idx else ""
            holdings.append(
                Holding(
                    instrument=cells[instrument_idx],
                    direction=direction,
                    position_low=position[0],
                    position_high=position[1],
                    row_text=lines[i],
                )
            )
            i += 1
        continue

    return holdings


def _is_etf_like(value: str) -> bool:
    code_match = re.search(r"`?(\d{6})`?", value)
    code = code_match.group(1) if code_match else ""
    if "ETF" in value.upper() or "LOF" in value.upper():
        return True
    return code.startswith(("15", "51", "56", "58"))


def _is_qdii_like(instrument: str, direction: str) -> bool:
    value = instrument + direction
    qdii_words = ("QDII", "跨境", "纳指", "标普", "恒生", "港股", "日经", "德国", "法国", "沙特", "海外")
    return any(word in value for word in qdii_words)


def _extract_code_context(text: str, code: str | None) -> str:
    if not code:
        return ""
    return "\n".join(line for line in text.splitlines() if code in line)


def _tokens(*values: str) -> list[str]:
    """从标的名/方向里抽出可用于对齐 TOP 排名的关键词。"""
    tokens: list[str] = []
    for value in values:
        if not value:
            continue
        cleaned = re.sub(r"[`\d/\\，、,\s—\-—()（）\[\]]", " ", value)
        cleaned = cleaned.replace("ETF", " ").replace("LOF", " ").replace("QDII", " ")
        for token in cleaned.split():
            token = token.strip()
            if len(token) < 2:
                continue
            if token in tokens:
                continue
            tokens.append(token)
    return tokens


def _extract_holding_context(text: str, holding: "Holding") -> str:
    """
    构建候选行文本：包含代码的行 + 包含标的/方向关键词的表格行。
    覆盖"代码在最终组合表、涨跌幅在 TOP 排名表"这种跨表场景。
    """
    parts: list[str] = []
    seen = set()

    def _add(line: str) -> None:
        if line in seen:
            return
        seen.add(line)
        parts.append(line)

    code_ctx = _extract_code_context(text, holding.code)
    if code_ctx:
        for line in code_ctx.splitlines():
            _add(line)

    tokens = _tokens(holding.instrument, holding.direction)
    if tokens:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if not any(token in line for token in tokens):
                continue
            _add(line)

    return "\n".join(parts) if parts else holding.row_text


def _has_unknown_qdii_premium(context: str) -> bool:
    return "溢价未知" in context or ("溢价" not in context and "折溢价" not in context)


def _has_clear_catalyst(context: str) -> bool:
    if "无事件催化" in context or "无明确事件催化" in context or "无催化" in context:
        return False
    catalyst_words = (
        "政策", "业绩", "财报", "订单", "产品", "发布", "获批", "回购",
        "涨价", "供给", "减产", "制裁", "美股", "纳指", "费半", "XBI", "中概",
        "催化", "指引",
    )
    return any(word in context for word in catalyst_words)


def _has_major_catalyst(context: str) -> bool:
    if not _has_clear_catalyst(context):
        return False
    major_words = (
        "重磅", "重大", "超预期", "政策", "业绩", "财报", "订单", "获批",
        "制裁", "禁令", "指引", "涨价", "减产",
    )
    return any(word in context for word in major_words)


def _is_existing_or_hold_only(context: str) -> bool:
    hold_words = ("已持有", "持有", "保留", "减仓", "清仓", "不新开", "观察")
    new_words = ("新进", "新开", "建仓", "买入", "加仓")
    if any(word in context for word in new_words):
        return False
    return any(word in context for word in hold_words)


def _extract_amount_yi(context: str) -> float | None:
    match = re.search(r"成交额\s*(\d+(?:\.\d+)?)\s*亿", context)
    if match:
        return float(match.group(1))
    if re.search(r"成交额\s*(?:<|＜|不足|低于)\s*1\s*亿", context):
        return 0.0
    return None


_SINGLE_DAY_HEADERS = ("今日涨跌", "当日涨跌", "单日涨跌", "涨跌幅", "今日涨幅", "当日涨幅")
_RET5_HEADERS = ("5日动量", "5日涨幅", "5日收益", "近5日", "5日涨跌")
_RET20_HEADERS = ("20日动量", "20日涨幅", "20日涨跌")
_OVERSOLD_REBOUND_THRESHOLD = -10.0


def _iter_tables(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and _is_separator(lines[i + 1]):
            headers = [_normalize_header(cell) for cell in _split_row(lines[i])]
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                if _is_separator(lines[j]):
                    j += 1
                    continue
                rows.append(_split_row(lines[j]))
                j += 1
            yield headers, rows
            i = j
        else:
            i += 1


def _row_matches(cells: list[str], holding: "Holding") -> bool:
    joined = " ".join(cells)
    if holding.code and holding.code in joined:
        return True
    tokens = _tokens(holding.instrument, holding.direction)
    return any(token in joined for token in tokens)


def _cell_number(cell: str) -> float | None:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", cell)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _column_index(headers: list[str], candidates: tuple[str, ...]) -> int | None:
    for name in candidates:
        if name in headers:
            return headers.index(name)
    return None


def _lookup_metric(text: str, holding: "Holding", header_candidates: tuple[str, ...]) -> float | None:
    for headers, rows in _iter_tables(text):
        col = _column_index(headers, header_candidates)
        if col is None:
            continue
        for cells in rows:
            if len(cells) <= col:
                continue
            if not _row_matches(cells, holding):
                continue
            value = _cell_number(cells[col])
            if value is not None:
                return value
    return None


def _extract_5d_return(context: str) -> float | None:
    for pattern in (r"5\s*日\s*(?:动量|涨幅|收益)\s*([+-]?\d+(?:\.\d+)?)\s*%", r"近5日\s*([+-]?\d+(?:\.\d+)?)\s*%"):
        match = re.search(pattern, context)
        if match:
            return float(match.group(1))
    return None


def _extract_single_day_change(context: str) -> float | None:
    for pattern in (
        r"今日\s*涨跌\s*([+-]?\d+(?:\.\d+)?)\s*%",
        r"单日\s*(?:涨跌|涨幅)\s*([+-]?\d+(?:\.\d+)?)\s*%",
        r"当日\s*涨跌?\s*([+-]?\d+(?:\.\d+)?)\s*%",
    ):
        match = re.search(pattern, context)
        if match:
            return float(match.group(1))
    return None


def _validate_holdings(
    text: str,
    market_state: str | None,
    allowed_codes: set[str],
    errors: list[str],
) -> None:
    holdings = _extract_portfolio_holdings(text)
    if not holdings:
        errors.append("未找到最终组合表（需包含 标的 + 计划/最终/目标/建议仓位 列）")
        return

    active_holdings = [holding for holding in holdings if not holding.is_cash and holding.position_high > 0]
    active_codes = [holding.code for holding in active_holdings if holding.code]
    if len(active_codes) != len(active_holdings):
        errors.append("正式组合每个标的都必须写出 scripts/etf.txt 中的六位代码")
    if len(set(active_codes)) != len(active_codes):
        errors.append("正式组合的三个标的代码必须互不相同")
    for holding in active_holdings:
        if holding.code and holding.code not in allowed_codes:
            errors.append(
                f"{holding.instrument} 不在 scripts/etf.txt 固定池中，禁止进入正式预测"
            )
    if market_state:
        if len(active_holdings) != TARGET_SELECTION_COUNT:
            errors.append(
                f"{market_state}正式组合必须有 {TARGET_SELECTION_COUNT} 个非现金标的，"
                f"当前为 {len(active_holdings)} 个"
            )

        exposure_low = sum(holding.position_low for holding in active_holdings)
        exposure_high = sum(holding.position_high for holding in active_holdings)
        cap = get_position_cap(market_state)
        if exposure_low < cap.low or exposure_high > cap.high:
            errors.append(
                f"{market_state}总仓位应在 {cap.text}，当前为 "
                f"{exposure_low:g}-{exposure_high:g}%"
            )

        if len(active_holdings) == TARGET_SELECTION_COUNT:
            weights = {round(holding.position_high, 2) for holding in active_holdings}
            if len(weights) != TARGET_SELECTION_COUNT:
                errors.append("三只标的必须按强弱设置不同仓位，不能机械等权")

    direction_exposure: dict[str, float] = {}
    for holding in active_holdings:
        cap = single_position_cap("etf", holding.is_qdii_like)
        if holding.position_high > cap:
            errors.append(f"{holding.instrument} 仓位 {holding.position_high:g}% 超过单标的上限 {cap}%")

        if holding.direction and holding.direction != "—":
            direction_exposure[holding.direction] = direction_exposure.get(holding.direction, 0) + holding.position_high

        context = _extract_holding_context(text, holding)
        if (
            holding.position_high > PROBE_POSITION_CAP
            and not _is_existing_or_hold_only(context)
            and not _has_clear_catalyst(context)
        ):
            errors.append(
                f"{holding.instrument} 仓位 >{PROBE_POSITION_CAP:g}%，"
                "但未找到明确催化或待发生事件"
            )

        if holding.is_qdii_like and holding.position_high > CONSERVATIVE_POSITION_CAP:
            context = context or holding.row_text
            if _has_unknown_qdii_premium(context):
                errors.append(f"QDII/跨境 {holding.instrument} 溢价未知，仓位不得超过15%")

        if holding.is_etf_like and holding.position_high >= MAIN_POSITION_MIN:
            amount = _extract_amount_yi(context or holding.row_text)
            if amount is not None and amount < MIN_MAIN_ETF_AMOUNT_YI:
                errors.append(f"主仓 ETF {holding.instrument} 成交额 {amount:g}亿 < 1亿，不能作为正式主仓")

        # 追高硬门禁（规则 5：单日≥7%次日不追，看位置不只看幅度）
        # 默认：单日>7% → 仓位≤10%。例外：20日动量<-10%（超跌反弹）可放宽至标准仓位。
        # 例外：扩散期方向不适用追高仓位限制（规则 11 优先）
        chase_ctx = context or holding.row_text
        is_diffusion = "扩散期" in chase_ctx
        ret5 = _extract_5d_return(chase_ctx)
        if ret5 is None:
            ret5 = _lookup_metric(text, holding, _RET5_HEADERS)
        chg1 = _extract_single_day_change(chase_ctx)
        if chg1 is None:
            chg1 = _lookup_metric(text, holding, _SINGLE_DAY_HEADERS)
        ret20 = _lookup_metric(text, holding, _RET20_HEADERS)
        if not is_diffusion:
            if holding.position_high > PROBE_POSITION_CAP:
                if ret5 is not None and ret5 > CHASE_5D_RETURN_LIMIT:
                    errors.append(
                        f"{holding.instrument} 5日动量 {ret5:+g}%>{CHASE_5D_RETURN_LIMIT:g}%，追高禁令要求仓位≤{PROBE_POSITION_CAP}%"
                    )
                if chg1 is not None and chg1 > CHASE_SINGLE_DAY_LIMIT:
                    # 规则 5 超跌反弹例外：20日动量 < -10% 可放宽
                    if ret20 is not None and ret20 < _OVERSOLD_REBOUND_THRESHOLD:
                        pass
                    else:
                        errors.append(
                            f"{holding.instrument} 单日 {chg1:+g}%>{CHASE_SINGLE_DAY_LIMIT:g}%，追高禁令要求仓位≤{PROBE_POSITION_CAP}%"
                        )

        if market_state in {"退潮", }:  # 仅退潮，退潮末期不触发此限制（规则 9）
            has_catalyst = _has_clear_catalyst(chase_ctx)
            has_major_catalyst = _has_major_catalyst(chase_ctx)
            hold_only = _is_existing_or_hold_only(chase_ctx)
            if chg1 is not None and chg1 > RETREAT_SINGLE_DAY_NO_CATALYST_LIMIT and not has_catalyst:
                errors.append(
                    f"退潮期 {holding.instrument} 单日 {chg1:+g}%>{RETREAT_SINGLE_DAY_NO_CATALYST_LIMIT:g}% 且无明确催化，禁止进入最终组合，只能观察"
                )
            if chg1 is not None and chg1 > CHASE_SINGLE_DAY_LIMIT and not has_major_catalyst:
                errors.append(
                    f"退潮期 {holding.instrument} 单日 {chg1:+g}%>{CHASE_SINGLE_DAY_LIMIT:g}% 且无重磅新催化，禁止进入最终组合"
                )
            if ret20 is not None and ret20 > RETREAT_RET20_HIGH_RISK_LIMIT:
                errors.append(
                    f"退潮期 {holding.instrument} 20日动量 {ret20:+g}%>{RETREAT_RET20_HIGH_RISK_LIMIT:g}%，默认高位风险，不得作为执行标的"
                )
            elif ret20 is not None and ret20 > RETREAT_RET20_HOLD_ONLY_LIMIT and not hold_only:
                errors.append(
                    f"退潮期 {holding.instrument} 20日动量 {ret20:+g}%>{RETREAT_RET20_HOLD_ONLY_LIMIT:g}%，只允许持有/减仓，不允许新开"
                )
            elif ret20 is not None and ret20 > RETREAT_RET20_NEW_POSITION_LIMIT and not has_catalyst and not hold_only:
                errors.append(
                    f"退潮期 {holding.instrument} 20日动量 {ret20:+g}%>{RETREAT_RET20_NEW_POSITION_LIMIT:g}% 且无新催化，不允许新开仓"
                )

    for direction, exposure in direction_exposure.items():
        if exposure > 60:
            errors.append(f"单一方向 {direction} 合计仓位 {exposure:g}% 超过 60%")


def _parse_evidence_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _text_cutoff(text: str, field: str) -> datetime | None:
    match = re.search(rf"{re.escape(field)}\s*[：:]\s*([^\n|]+)", text)
    if not match:
        return None
    value = match.group(1).strip().strip("*")
    return _parse_evidence_time(value)


def _validate_structured_evidence(
    selection_path: Path,
    text: str,
    errors: list[str],
) -> None:
    evidence_path = selection_path.with_name(EVIDENCE_FILENAME)
    if not evidence_path.exists():
        errors.append(f"缺少 {EVIDENCE_FILENAME}，无法审计催化来源和已知时间")
        return
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{EVIDENCE_FILENAME} 无法解析: {exc}")
        return

    if payload.get("schema_version") != 1:
        errors.append(f"{EVIDENCE_FILENAME} schema_version 必须为 1")
    data_cutoff = _text_cutoff(text, "数据截止时间")
    news_cutoff = _text_cutoff(text, "新闻截止时间")
    payload_data_cutoff = _parse_evidence_time(payload.get("data_cutoff", ""))
    payload_news_cutoff = _parse_evidence_time(payload.get("news_cutoff", ""))
    if not payload_data_cutoff or not data_cutoff or payload_data_cutoff != data_cutoff:
        errors.append("结构化证据 data_cutoff 必须与 selection 数据截止时间完全一致")
    if not payload_news_cutoff or not news_cutoff or payload_news_cutoff != news_cutoff:
        errors.append("结构化证据 news_cutoff 必须与 selection 新闻截止时间完全一致")

    holdings = [
        holding
        for holding in _extract_portfolio_holdings(text)
        if not holding.is_cash and holding.position_high > 0
    ]
    evidence_rows = payload.get("holdings")
    if not isinstance(evidence_rows, list):
        errors.append("结构化证据 holdings 必须为数组")
        return
    by_code = {
        str(row.get("code")): row
        for row in evidence_rows
        if isinstance(row, dict) and row.get("code")
    }
    active_codes = {holding.code for holding in holdings if holding.code}
    extra_codes = set(by_code) - active_codes
    if extra_codes:
        errors.append(f"结构化证据包含非组合代码: {', '.join(sorted(extra_codes))}")

    for holding in holdings:
        if not holding.code:
            continue
        row = by_code.get(holding.code)
        if row is None:
            errors.append(f"{holding.instrument} 缺少结构化证据")
            continue
        action = str(row.get("action") or "")
        if action not in {"new", "hold", "reduce"}:
            errors.append(f"{holding.instrument} action 必须为 new/hold/reduce")
        thesis = str(row.get("thesis") or "").strip()
        invalidation = str(row.get("invalidation") or "").strip()
        if not thesis:
            errors.append(f"{holding.instrument} 缺少可证伪 thesis")
        if not invalidation:
            errors.append(f"{holding.instrument} 缺少 invalidation")

        if action != "new" or holding.position_high <= PROBE_POSITION_CAP:
            continue
        catalyst = row.get("catalyst")
        if not isinstance(catalyst, dict):
            errors.append(f"{holding.instrument} 新仓超过{PROBE_POSITION_CAP:g}%但缺少 catalyst")
            continue
        required = ("title", "known_at", "event_date", "source_name", "source_url")
        missing = [field for field in required if not str(catalyst.get(field) or "").strip()]
        if missing:
            errors.append(f"{holding.instrument} catalyst 缺少字段: {', '.join(missing)}")
            continue
        known_at = _parse_evidence_time(catalyst["known_at"])
        event_date = _parse_evidence_time(catalyst["event_date"])
        if not known_at or not news_cutoff or known_at > news_cutoff:
            errors.append(f"{holding.instrument} catalyst known_at 晚于新闻截止时间或格式无效")
        if not event_date or not data_cutoff or event_date.date() < data_cutoff.date():
            errors.append(f"{holding.instrument} catalyst event_date 不是数据截止后的待发生事件")
        if not str(catalyst["source_url"]).startswith(("https://", "http://")):
            errors.append(f"{holding.instrument} catalyst source_url 必须是可追溯链接")


def validate_selection(path: Path) -> tuple[bool, list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    errors = []
    allowed_codes = {cfg["code"] for cfg in load_etf_txt().values()}
    if not allowed_codes:
        errors.append("scripts/etf.txt 为空或读取失败，禁止生成正式预测")

    frequency = _extract_frequency(text)
    if frequency != FORMAL_FREQUENCY:
        shown = frequency or "未填写"
        errors.append(
            f"正式 selection 只允许周度调仓，当前频率为 {shown}；"
            "日度只能写 risk_review.md，月度只能写 monthly_review.md"
        )

    try:
        plan_date = datetime.strptime(path.parent.name, "%Y-%m-%d")
    except (TypeError, ValueError):
        plan_date = None
    if plan_date is not None and plan_date.weekday() not in WEEKLY_ALLOWED_WEEKDAYS:
        if "节假日特殊调仓：是" not in text:
            errors.append(
                "正式周度方案只能在周五至周末生成或周一盘前执行；"
                "节假日导致的周中方案必须明确写“节假日特殊调仓：是”"
            )

    if not re.search(r"数据截止时间\s*[：:]\s*\S+", text):
        errors.append("周度方案缺少“数据截止时间”，无法验证时点")
    if not re.search(r"下次常规重排\s*[：:]\s*\S+", text):
        errors.append("周度方案缺少“下次常规重排”，无法锁定周度频率")
    if "国际事件复核" not in text:
        errors.append("周度方案缺少“国际事件复核”")
    if not re.search(r"新闻截止时间\s*[：:]\s*\S+", text):
        errors.append("国际事件复核缺少“新闻截止时间”")
    event_risk_match = re.search(
        r"国际事件风险\s*[：:]\s*\*{0,2}(正常|黄色|红色)",
        text,
    )
    if not event_risk_match:
        errors.append("国际事件复核缺少标准风险结论：正常/黄色/红色")
    if not re.search(r"国际事件动作\s*[：:]\s*\S+", text):
        errors.append("国际事件复核缺少“国际事件动作”，无法确认新闻是否实际影响组合")

    market_state = _extract_market_state(text)
    if not market_state:
        errors.append("未找到市场状态，无法判断现金硬约束")

    cash_range = _extract_cash_range(text)
    if not cash_range:
        errors.append("未找到现金仓位行，selection 不可执行")

    if market_state and cash_range:
        cash_low, cash_high = cash_range
        if requires_aggressive_deployment(market_state) and cash_high > 40:
            errors.append(
                f"{market_state}要求现金≤40%，当前现金区间为{cash_low:g}-{cash_high:g}%"
            )

        holdings = _extract_portfolio_holdings(text)
        active = [holding for holding in holdings if not holding.is_cash]
        invested_low = sum(holding.position_low for holding in active)
        invested_high = sum(holding.position_high for holding in active)
        total_low = invested_low + cash_low
        total_high = invested_high + cash_high
        if total_low < 99.5 or total_high > 100.5:
            errors.append(
                f"组合仓位与现金之和必须为100%，当前为{total_low:g}-{total_high:g}%"
            )

    _validate_holdings(text, market_state, allowed_codes, errors)
    _validate_structured_evidence(path, text, errors)

    if event_risk_match and event_risk_match.group(1) == "红色":
        for holding in _extract_portfolio_holdings(text):
            if holding.is_cash or holding.position_high <= 0:
                continue
            context = _extract_holding_context(text, holding)
            cluster = risk_cluster(
                holding.direction,
                holding.direction,
                holding.instrument,
            )
            if cluster == "高弹性成长" and not _is_existing_or_hold_only(context):
                errors.append(
                    f"全球事件风险为红色，禁止新开高弹性仓位 {holding.instrument}"
                )

    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 selection.md 的硬性执行约束")
    parser.add_argument("selection", help="selection.md 路径")
    args = parser.parse_args()

    path = Path(args.selection)
    if not path.exists():
        print(f"selection 文件不存在: {path}", file=sys.stderr)
        sys.exit(2)

    ok, errors = validate_selection(path)
    if ok:
        print(f"OK: {path} 通过硬约束校验")
        return

    print(f"FAIL: {path} 未通过硬约束校验", file=sys.stderr)
    for err in errors:
        print(f"- {err}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
