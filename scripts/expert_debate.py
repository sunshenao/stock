"""
专家辩论脚本
============
调用 stock-analyzer-skill 的 7 位专家对个股打分，输出共识结论。

用法：python scripts/expert_debate.py 603259
"""
import sys, os, re, json, argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from skill_paths import find_skill_root, find_skill_scripts

SKILL_ROOT = find_skill_root("stock-analyzer-skill")
SKILL_SCRIPTS = find_skill_scripts("stock-analyzer-skill")
if SKILL_SCRIPTS:
    sys.path.insert(0, str(SKILL_SCRIPTS))
if SKILL_ROOT:
    sys.path.insert(0, str(SKILL_ROOT))

from experts.scoring import (sector_specialist, momentum_trader, risk_manager,
                              value_anchor, institution, emotion_tech, topic_leader)
from data import get_quote, get_finance
from common import normalize_finance_code

EXPERTS = [
    ("价值双锚", value_anchor),
    ("行业专家", sector_specialist),
    ("动量派", momentum_trader),
    ("风险管理", risk_manager),
    ("机构派", institution),
    ("情绪技术", emotion_tech),
    ("题材龙头", topic_leader),
]


def extract_dim_scores(reasoning):
    """从推理文本中提取维度分数"""
    if not isinstance(reasoning, list):
        return []
    scores = []
    for item in reasoning:
        if isinstance(item, str):
            m = re.search(r'(\d+)/100', item)
            if m:
                scores.append(int(m.group(1)))
    return scores


def debate(code: str) -> dict:
    """运行专家辩论，返回共识结果"""
    prefix = "sh" if code.startswith(("6", "5")) else "sz"
    q = get_quote(f"{prefix}{code}")
    fin = get_finance(normalize_finance_code(f"{prefix}{code}"))
    f = fin[0] if fin else None

    stock_data = {
        "code": code, "name": q.name, "price": q.price,
        "pe": q.pe, "pb": q.pb, "change_pct": q.change_pct,
        "turnover": q.turnover, "total_cap": q.total_cap,
    }
    if f:
        stock_data.update({
            "roe": f.roe, "eps": f.eps, "revenue_yoy": f.revenue_yoy,
            "net_profit_yoy": f.net_profit_yoy, "gross_margin": f.gross_margin,
            "debt_ratio": f.debt_ratio,
        })

    results = []
    for name, mod in EXPERTS:
        try:
            r = mod.score_with_reasoning(stock_data)
        except Exception:
            try:
                r = mod.score(stock_data)
            except Exception:
                results.append({"expert": name, "score": 50, "dims": [], "detail": ["评分出错"]})
                continue

        total = r.get("total", 0)
        reasoning = r.get("reasoning", [])
        dims = extract_dim_scores(reasoning)
        if total == 0 and dims:
            total = sum(dims) / len(dims)

        results.append({
            "expert": name,
            "score": round(total, 1),
            "dims": dims,
            "detail": [str(x) for x in (reasoning if isinstance(reasoning, list) else [reasoning])],
        })

    results.sort(key=lambda x: -x["score"])
    consensus = sum(r["score"] for r in results) / len(results)

    return {
        "code": code,
        "name": q.name,
        "pe": q.pe,
        "roe": stock_data.get("roe", 0),
        "revenue_yoy": stock_data.get("revenue_yoy", 0),
        "results": results,
        "consensus": round(consensus, 1),
        "verdict": "看多" if consensus >= 60 else "中性偏多" if consensus >= 55 else "中性" if consensus >= 45 else "中性偏空" if consensus >= 35 else "看空",
    }


def format_markdown(result: dict) -> str:
    """生成 markdown 报告"""
    lines = [
        f"# 专家辩论 — {result['name']} {result['code']}",
        "",
        f"| 指标 | 数值 |",
        f"|---|---|",
        f"| PE | {result['pe']}x |",
        f"| ROE | {result['roe']:.1f}% |",
        f"| 营收增速 | {result['revenue_yoy']:.1f}% |",
        "",
        "---",
        "",
        "| 专家 | 评分 | 判断 | 关键维度 |",
        "|---|---|---|---|",
    ]
    for r in result["results"]:
        dim_str = ", ".join(str(d) for d in r["dims"][:4])
        view = "看多" if r["score"] >= 65 else "中性偏多" if r["score"] >= 55 else "中性" if r["score"] >= 45 else "中性偏空" if r["score"] >= 35 else "看空"
        lines.append(f"| {r['expert']} | {r['score']:.0f} | {view} | {dim_str} |")

    lines.append(f"\n**共识均分: {result['consensus']:.0f}/100 ({result['verdict']})**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 各专家详细意见")
    lines.append("")
    for r in result["results"]:
        lines.append(f"### {r['expert']} ({r['score']:.0f}分)")
        lines.append("")
        for d in r["detail"][:6]:
            lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="专家辩论评分")
    parser.add_argument("code", help="股票代码，例如 603259")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="输出日期 YYYY-MM-DD")
    args = parser.parse_args()

    code = args.code
    result = debate(code)

    out_dir = SCRIPT_DIR.parent / "codex" / "stock" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"expert_debate_{code}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(format_markdown(result))

    print(f"共识: {result['consensus']:.0f}/100 ({result['verdict']})")
    for r in result["results"]:
        print(f"  {r['expert']}: {r['score']:.0f}")
    print(f"\n报告: {out_path}")


if __name__ == "__main__":
    main()
