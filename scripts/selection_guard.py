"""
Selection execution guard.

Checks hard portfolio constraints in a generated selection.md before it can be
treated as an executable plan.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from risk_rules import requires_aggressive_deployment  # noqa: E402


MARKET_STATES = ("主升", "震荡", "退潮", "冰点", "未知")


def _extract_market_state(text: str) -> str | None:
    patterns = [
        r"\|\s*市场状态\s*\|\s*\*{0,2}(主升|震荡|退潮|冰点|未知)\*{0,2}\s*\|",
        r"(?:市场状态|状态)\s*[：:]\s*\*{0,2}(主升|震荡|退潮|冰点|未知)\*{0,2}",
        r"-\s*(?:市场状态|状态)\s*[：:]\s*\*{0,2}(主升|震荡|退潮|冰点|未知)\*{0,2}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    for state in MARKET_STATES:
        if f"市场状态 | **{state}**" in text:
            return state
    return None


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


def validate_selection(path: Path) -> tuple[bool, list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    errors = []

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
                f"非冰点市场({market_state})现金上限为40%，当前现金区间为{cash_low:g}-{cash_high:g}%"
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
