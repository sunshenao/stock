"""
个股三分钟体检脚本
==================
对候选标的快速输出：营收增速、PE分位、ROE、毛利率、主力资金、技术位置

用法：
  python scripts/stock_checkup.py 603259
  python scripts/stock_checkup.py 603259 512010 560800

数据源：stock-analyzer-skill (quote/finance) + hithink-market-query (northbound)
"""
import sys
import os
import subprocess
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from skill_paths import find_hithink_cli, find_skill_scripts

SKILL_SCRIPTS = find_skill_scripts("stock-analyzer-skill")
if SKILL_SCRIPTS:
    sys.path.insert(0, str(SKILL_SCRIPTS))

IWENCAI_CLI = find_hithink_cli()

# 行业差异化阈值（统一维护于 risk_rules.py）
from risk_rules import INDUSTRY_THRESHOLDS as THRESHOLDS


def _to_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def get_quote_info(code: str) -> dict:
    """获取实时行情（PE/PB/市值/涨跌幅）"""
    from data import get_quote
    prefix = "sh" if code.startswith(("6","5")) else "sz"
    q = get_quote(f"{prefix}{code}")
    return {
        "name": q.name,
        "price": q.price,
        "pe": q.pe,
        "pb": q.pb,
        "change_pct": q.change_pct,
        "turnover": q.turnover,
        "total_cap": q.total_cap,
    }


def get_finance_info(code: str) -> dict:
    """获取最新财务数据"""
    from data import get_finance
    from common import normalize_finance_code
    prefix = "sh" if code.startswith(("6","5")) else "sz"
    records = get_finance(normalize_finance_code(f"{prefix}{code}"))
    if not records:
        return {}
    f = records[0]
    return {
        "roe": f.roe,
        "eps": f.eps,
        "revenue_yoy": f.revenue_yoy,
        "net_profit_yoy": f.net_profit_yoy,
        "gross_margin": f.gross_margin,
        "debt_ratio": f.debt_ratio,
    }


def get_flow_info(code: str) -> dict:
    """获取主力资金流向（通过 hithink CLI）"""
    if not IWENCAI_CLI:
        return {"main_flow": None, "fund_flow": "N/A", "data_ok": False, "error": "未找到 hithink-market-query CLI"}
    env = os.environ.copy()
    settings_path = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings_path) as f:
            s = json.load(f)
        for k in ("IWENCAI_API_KEY", "IWENCAI_BASE_URL"):
            if k in s.get("env", {}):
                env[k] = s["env"][k]
    except Exception:
        print("[WARN] 无法读取 settings.json 中的同花顺 API Key，资金流向数据将缺失", file=sys.stderr)

    last_error = ""
    try:
        result = subprocess.run(
            [sys.executable, IWENCAI_CLI, "--query", f"{code} 主力资金流向 北向资金", "--limit", "1", "--timeout", "20"],
            capture_output=True, text=True, env=env, timeout=25,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get("datas"):
                d = data["datas"][0]
                flow = d.get("主力资金流向", 0) or 0
                try:
                    flow = float(flow)
                except (ValueError, TypeError):
                    flow = 0.0
                fund_flow = d.get("资金流向", "N/A")
                return {
                    "main_flow": flow,
                    "fund_flow": str(fund_flow)[:50] if fund_flow else "N/A",
                    "data_ok": True,
                    "error": "",
                }
        else:
            last_error = (result.stderr or result.stdout or "").strip()[:120]
    except Exception as exc:
        last_error = str(exc)[:120]
    return {"main_flow": None, "fund_flow": "N/A", "data_ok": False, "error": last_error or "未返回资金数据"}


def get_technical_info(code: str) -> dict:
    """获取技术面数据：均线位置、MACD、KDJ"""
    if not IWENCAI_CLI:
        return {
            "macd": None,
            "kdj": None,
            "ma5_dist": None,
            "ma20_dist": None,
            "data_ok": False,
            "error": "未找到 hithink-market-query CLI",
        }
    env = os.environ.copy()
    settings_path = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings_path) as f:
            s = json.load(f)
        for k in ("IWENCAI_API_KEY", "IWENCAI_BASE_URL"):
            if k in s.get("env", {}):
                env[k] = s["env"][k]
    except Exception:
        print("[WARN] 无法读取 settings.json 中的同花顺 API Key，技术指标数据将缺失", file=sys.stderr)

    last_error = ""
    try:
        result = subprocess.run(
            [sys.executable, IWENCAI_CLI, "--query",
             f"{code} MACD KDJ 均线 距5日线 距20日线", "--limit", "1", "--timeout", "20"],
            capture_output=True, text=True, env=env, timeout=25,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get("datas"):
                d = data["datas"][0]
                # hithink返回的key带日期后缀如 macd[20260629]，需模糊匹配
                macd = kdj_val = ma5 = ma20 = sentiment = None
                for k, v in d.items():
                    if 'macd' in k.lower():
                        macd = v
                    elif 'kdj' in k.lower():
                        kdj_val = v
                    elif '距5日' in k or 'ma5' in k.lower():
                        ma5 = v
                    elif '距20日' in k or 'ma20' in k.lower():
                        ma20 = v
                return {
                    "macd": _to_float(macd),
                    "kdj": _to_float(kdj_val),
                    "ma5_dist": _to_float(ma5),
                    "ma20_dist": _to_float(ma20),
                    "data_ok": True,
                    "error": "",
                }
        else:
            last_error = (result.stderr or result.stdout or "").strip()[:120]
    except Exception as exc:
        last_error = str(exc)[:120]
    return {
        "macd": None,
        "kdj": None,
        "ma5_dist": None,
        "ma20_dist": None,
        "data_ok": False,
        "error": last_error or "未返回技术数据",
    }


# ---- 评分与输出 ----
def evaluate(code: str, industry: str) -> dict:
    """对个股做三分钟体检并打分
    权重：基本面45 + 估值15 + 技术面25 + 资金面15 = 100

    industry 必须显式传入，不设默认值：
      医药 / 科技 / 消费 / 周期 / 金融 / 新能源 / 军工 / 公用事业 / 红利
    """
    quote = get_quote_info(code)
    finance = get_finance_info(code)
    flow = get_flow_info(code)
    tech = get_technical_info(code)

    if industry not in THRESHOLDS:
        raise ValueError(f"不支持的行业类型: {industry}，可选: {list(THRESHOLDS.keys())}")
    thresh = THRESHOLDS[industry]

    checks = []
    score = 0

    # ===== 基本面 (45分) =====
    # 1. 营收增速 (20)
    rev = finance.get("revenue_yoy", 0)
    rev_min = thresh.get("growth_min", 20) or 20
    if rev >= rev_min * 1.5:
        checks.append(("[OK]", f"营收增速 {rev:.1f}%（优秀 >{rev_min*1.5:.0f}%）"))
        score += 20
    elif rev >= rev_min:
        checks.append(("[OK]", f"营收增速 {rev:.1f}%（合格 ≥{rev_min}%）"))
        score += 14
    elif rev > 0:
        checks.append(("[WARN]", f"营收增速 {rev:.1f}%（低于门槛 {rev_min}%）"))
        score += 6
    else:
        checks.append(("[FAIL]", f"营收增速 {rev:.1f}%（负增长）"))
        score += 0

    # 2. ROE (15)
    roe = finance.get("roe", 0)
    roe_min = thresh.get("roe_min", 12) or 12
    if roe >= roe_min:
        checks.append(("[OK]", f"ROE {roe:.1f}%（合格 ≥{roe_min}%）"))
        score += 15
    elif roe >= roe_min * 0.6:
        checks.append(("[WARN]", f"ROE {roe:.1f}%（偏低，门槛 {roe_min}%）"))
        score += 7
    else:
        checks.append(("[FAIL]", f"ROE {roe:.1f}%（不达标）"))
        score += 0

    # 3. 毛利率 (10)
    gross = finance.get("gross_margin", 0)
    gross_min = thresh.get("gross_min", 40) or 40
    if gross >= gross_min:
        checks.append(("[OK]", f"毛利率 {gross:.1f}%（有壁垒 ≥{gross_min}%）"))
        score += 10
    elif gross >= gross_min * 0.7:
        checks.append(("[WARN]", f"毛利率 {gross:.1f}%（一般）"))
        score += 5
    else:
        checks.append(("[FAIL]", f"毛利率 {gross:.1f}%（薄利）"))
        score += 0

    # ===== 估值 (15分) =====
    # 4. PE (15)
    pe = quote.get("pe", 0)
    if industry == "医药":
        if pe <= 25:
            checks.append(("[OK]", f"PE {pe:.1f}x（便宜 ≤25x）"))
            score += 15
        elif pe <= 40:
            checks.append(("[OK]", f"PE {pe:.1f}x（合理 ≤40x）"))
            score += 11
        elif pe <= 60:
            checks.append(("[WARN]", f"PE {pe:.1f}x（偏贵 >40x）"))
            score += 5
        else:
            checks.append(("[FAIL]", f"PE {pe:.1f}x（很贵 >60x）"))
            score += 0
    elif industry == "科技":
        if pe <= 40:
            checks.append(("[OK]", f"PE {pe:.1f}x（便宜 ≤40x）"))
            score += 15
        elif pe <= 60:
            checks.append(("[OK]", f"PE {pe:.1f}x（合理 ≤60x）"))
            score += 11
        elif pe <= 100:
            checks.append(("[WARN]", f"PE {pe:.1f}x（偏贵）"))
            score += 5
        else:
            checks.append(("[FAIL]", f"PE {pe:.1f}x（很贵）"))
            score += 0
    elif industry == "消费":
        # 消费：PE<30 合理，白酒可到 35
        if pe <= 25:
            checks.append(("[OK]", f"PE {pe:.1f}x（便宜 ≤25x）"))
            score += 15
        elif pe <= 35:
            checks.append(("[OK]", f"PE {pe:.1f}x（合理 ≤35x）"))
            score += 11
        elif pe <= 50:
            checks.append(("[WARN]", f"PE {pe:.1f}x（偏贵）"))
            score += 5
        else:
            checks.append(("[FAIL]", f"PE {pe:.1f}x（很贵 >50x）"))
            score += 0
    elif industry == "金融":
        # 金融：银行看 PB<0.7，券商看 ROE 分位。PE 参考意义有限
        pb = quote.get("pb", 0)
        if pb < 0.8:
            checks.append(("[OK]", f"PB {pb:.2f}x（破净/低估）"))
            score += 15
        elif pb < 1.2:
            checks.append(("[OK]", f"PB {pb:.2f}x（合理）"))
            score += 11
        elif pb < 2.0:
            checks.append(("[WARN]", f"PB {pb:.2f}x（偏贵）"))
            score += 5
        else:
            checks.append(("[FAIL]", f"PB {pb:.2f}x（很贵）"))
            score += 0
    elif industry in ("周期", "新能源", "军工", "公用事业", "红利"):
        # 周期/新能源/军工: PE 波动大，主要看绝对值和行业分位
        if pe <= 20:
            checks.append(("[OK]", f"PE {pe:.1f}x（低位 ≤20x）"))
            score += 15
        elif pe <= 35:
            checks.append(("[OK]", f"PE {pe:.1f}x（合理 ≤35x）"))
            score += 11
        elif pe <= 55:
            checks.append(("[WARN]", f"PE {pe:.1f}x（偏高）"))
            score += 5
        else:
            checks.append(("[FAIL]", f"PE {pe:.1f}x（很贵 >55x）"))
            score += 0
    else:
        # 兜底：按中等标准
        if pe <= 30:
            checks.append(("[OK]", f"PE {pe:.1f}x（低位）"))
            score += 11
        elif pe <= 50:
            checks.append(("[WARN]", f"PE {pe:.1f}x（中性）"))
            score += 5
        else:
            checks.append(("[WARN]", f"PE {pe:.1f}x（偏高）"))
            score += 0

    # ===== 技术面 (25分) =====
    # 5. MACD (10)
    macd = tech.get("macd")
    if macd is not None:
        if macd > 0:
            checks.append(("[OK]", f"MACD {macd:.2f}（金叉/多头）"))
            score += 10
        else:
            checks.append(("[WARN]", f"MACD {macd:.2f}（死叉/空头）"))
            score += 3
    else:
        checks.append(("[MISS]", "MACD 数据缺失（不加分，需补查）"))
        score += 0

    # 6. KDJ 超买风险 (5)
    kdj = tech.get("kdj")
    if kdj is not None:
        if kdj > 85:
            checks.append(("[FAIL]", f"KDJ {kdj:.0f}（严重超买，追高风险极大）"))
            score += 0
        elif kdj > 70:
            checks.append(("[WARN]", f"KDJ {kdj:.0f}（偏高，等回调）"))
            score += 2
        else:
            checks.append(("[OK]", f"KDJ {kdj:.0f}（正常区间）"))
            score += 5
    else:
        checks.append(("[MISS]", "KDJ 数据缺失（不加分，需补查）"))
        score += 0

    # 7. 距 20 日线 (10) — 判断是否追高
    ma20_dist = tech.get("ma20_dist")
    if ma20_dist is not None:
        if ma20_dist > 15:
            checks.append(("[FAIL]", f"距20日线 {ma20_dist:+.0f}%（严重偏离，等回踩）"))
            score += 0
        elif ma20_dist > 8:
            checks.append(("[WARN]", f"距20日线 {ma20_dist:+.0f}%（偏高）"))
            score += 4
        elif ma20_dist < -5:
            checks.append(("[WARN]", f"距20日线 {ma20_dist:+.0f}%（破位风险）"))
            score += 3
        else:
            checks.append(("[OK]", f"距20日线 {ma20_dist:+.0f}%（合理区间）"))
            score += 10
    else:
        checks.append(("[MISS]", "均线数据缺失（不加分，需补查）"))
        score += 0

    # ===== 资金面 (15分) =====
    # 8. 主力资金 (15)
    main_flow = flow.get("main_flow")
    if flow.get("data_ok") and main_flow is not None and main_flow > 1e8:
        checks.append(("[OK]", f"主力净流入 {main_flow/1e8:.1f}亿（机构积极买入）"))
        score += 15
    elif flow.get("data_ok") and main_flow is not None and main_flow > 0:
        checks.append(("[OK]", f"主力净流入 {main_flow/1e8:.1f}亿（机构在买）"))
        score += 12
    elif flow.get("data_ok") and main_flow is not None and main_flow < -1e8:
        checks.append(("[FAIL]", f"主力净流出 {abs(main_flow)/1e8:.1f}亿（机构在跑）"))
        score += 0
    elif flow.get("data_ok") and main_flow is not None and main_flow < 0:
        checks.append(("[WARN]", f"主力净流出 {abs(main_flow)/1e8:.1f}亿"))
        score += 4
    else:
        checks.append(("[MISS]", f"主力资金数据缺失（不加分，需补查；{flow.get('error','未知原因')}）"))
        score += 0

    return {
        "code": code,
        "industry": industry,
        "score": score,
        "max_score": 100,
        "grade": "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D",
        "quote": quote,
        "finance": finance,
        "flow": flow,
        "tech": tech,
        "checks": checks,
    }


def format_report(result: dict) -> str:
    """格式化输出"""
    q = result["quote"]
    f = result["finance"]
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"  {q['name']} ({result['code']}) — {result['industry']}")
    lines.append(f"{'='*50}")
    lines.append(f"  最新价: {q['price']:.2f}  |  涨跌幅: {q['change_pct']:+.2f}%  |  换手率: {q['turnover']:.2f}%")
    lines.append(f"  PE: {q['pe']:.1f}x  |  PB: {q['pb']:.2f}x  |  总市值: {q['total_cap']:.0f}亿")
    lines.append("")
    lines.append(f"  --- 财务面 ---")
    lines.append(f"  营收增速: {f.get('revenue_yoy',0):+.1f}%  |  净利增速: {f.get('net_profit_yoy',0):+.1f}%")
    lines.append(f"  ROE: {f.get('roe',0):.2f}%  |  毛利率: {f.get('gross_margin',0):.1f}%")
    lines.append(f"  EPS: {f.get('eps',0):.2f}  |  负债率: {f.get('debt_ratio',0):.1f}%")
    lines.append("")
    lines.append(f"  --- 体检项 ---")
    for icon, text in result["checks"]:
        lines.append(f"  {icon} {text}")
    lines.append("")
    lines.append(f"  综合评分: {result['score']}/{result['max_score']}  ({result['grade']}级)")
    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="个股三分钟体检")
    parser.add_argument("codes", nargs="*", default=["603259"], help="股票代码")
    parser.add_argument("--industry", "-i", help="行业: 医药/科技/消费/周期/金融/新能源/军工")
    args = parser.parse_args()

    codes = args.codes
    forced_industry = args.industry

    # Auto-detect industry for common codes（--industry 参数可覆盖）
    code_industry = {
        # 医药
        "603259": "医药",
        "600276": "医药",
        "300759": "医药",
        "300347": "医药",
        "688336": "医药",
        "688180": "医药",
        # 科技
        "688120": "科技",
        "002371": "科技",
        "688012": "科技",
        "300308": "科技",
        "300502": "科技",
        "002463": "科技",
        "601138": "科技",
        "002475": "科技",
        "688041": "科技",
        "688981": "科技",
        # 消费
        "600519": "消费",
        "000858": "消费",
        "603288": "消费",
        "000333": "消费",
        # 周期
        "601899": "周期",
        "603993": "周期",
        "601088": "周期",
        # 金融
        "600036": "金融",
        "601318": "金融",
        "600030": "金融",
        # 新能源
        "300750": "新能源",
        "601012": "新能源",
        # 军工
        "600760": "军工",
        "600893": "军工",
    }

    for code in codes:
        if forced_industry:
            industry = forced_industry
        else:
            industry = code_industry.get(code)
            if industry is None:
                print(f"错误: 未识别 {code} 的行业，请用 --industry 参数指定")
                print(f"可用行业: 医药 | 科技 | 消费 | 周期 | 金融 | 新能源 | 军工 | 公用事业 | 红利")
                print(f"已知代码: {sorted(code_industry.keys())}")
                sys.exit(1)
        result = evaluate(code, industry)
        print(format_report(result))


if __name__ == "__main__":
    main()
