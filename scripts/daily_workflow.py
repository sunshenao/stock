"""
每日选股工作流入口
=================
运行 ETF 三层分析：市场环境 → 板块阶段 → 方向建议

用法：
  python scripts/daily_workflow.py --date 2026-06-29
"""
import sys
import os
import subprocess
import argparse
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(SCRIPT_DIR).parent


def main():
    parser = argparse.ArgumentParser(description="每日 ETF 三层分析")
    parser.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=30, help="ETF 摘要输出前 N 名")
    parser.add_argument("--workers", type=int, default=8, help="ETF 行情并发线程数")
    parser.add_argument("--no-hithink", action="store_true", help="关闭同花顺实时增强")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    etf_script = os.path.join(SCRIPT_DIR, "etf_analyzer.py")
    cmd = [
        sys.executable,
        etf_script,
        "--date",
        target_date,
        "--top",
        str(args.top),
        "--workers",
        str(args.workers),
    ]
    if not args.no_hithink:
        cmd.append("--hithink")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"ETF 扫描失败，已停止后续流程。退出码: {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    report_path = PROJECT_ROOT / "codex" / "stock" / target_date / "etf_scan.md"
    if not report_path.exists() or report_path.stat().st_size == 0:
        print(f"ETF 扫描报告缺失或为空: {report_path}", file=sys.stderr)
        sys.exit(2)

    report_text = report_path.read_text(encoding="utf-8-sig")
    rank_rows = len(re.findall(r"^\|\s*\d+\s*\|", report_text, flags=re.MULTILINE))
    if rank_rows == 0:
        print(f"ETF 扫描报告没有有效排名行，已停止后续流程: {report_path}", file=sys.stderr)
        sys.exit(3)

    print(f"\n报告: codex/stock/{target_date}/etf_scan.md")
    print("\n后续步骤 (Claude Code 对话中):")
    print("  1. 查看 etf_scan.md 的方向建议")
    print("  2. 用 news-search + announcement-search 验证催化")
    print("  3. 用 hithink-market-query 验证个股技术面")
    print("  4. 填写 selection.md")
    print(f"  5. 运行 python scripts/selection_guard.py codex/stock/{target_date}/selection.md；未通过则不可执行")


if __name__ == "__main__":
    main()
