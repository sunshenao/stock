"""
ETF 板块三层分析脚本
====================
第一层：市场环境（市场宽度 → 总仓位上限）
第二层：板块阶段（ETF动量 + 行业差异化阈值 → 板块生命周期 + 动作建议）
第三层：个股映射（输出方向，个股选择在 Claude Code 对话中完成）

用法：
  python scripts/etf_analyzer.py --date 2026-06-29

数据源：AKShare + stock-analyzer-skill (market_breadth, sector_specialist)
"""
import sys
import os
import re
import argparse
from datetime import datetime, timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import akshare as ak

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from risk_rules import get_position_cap, premium_discount_factor

# Stock-analyzer-skill 路径
SKILL_SCRIPTS = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "npm", "node_modules", "stock-analyzer-skill", "scripts"
)
if os.path.isdir(SKILL_SCRIPTS):
    sys.path.insert(0, SKILL_SCRIPTS)
ETF_TXT_PATH = SCRIPT_DIR / "etf.txt"
BENCHMARK_CODE = "510300"  # 沪深300ETF 基准代码（需在 etf.txt 中存在）

STAGE_PRIORITY = {
    "扩散期": 7,
    "加速期": 6,
    "确认期": 5,
    "萌芽期": 4,
    "趋势回撤期": 3,
    "观察期": 2,
    "加速期⚠": 1,
    "加速见顶⚠": 0,
    "衰弱期": -1,
    "弱势期": -2,
    "休眠期": -3,
    "衰竭期": -4,
}

NEW_MONEY_STAGES = {"扩散期", "加速期", "确认期", "萌芽期"}
EXCLUDED_STAGES = {"衰弱期", "弱势期", "休眠期", "衰竭期", "加速见顶⚠", "加速期⚠"}

# 行业差异化阈值（统一维护于 risk_rules.py）
from risk_rules import INDUSTRY_THRESHOLDS


def _infer_industry(category: str, direction: str, name: str) -> str:
    """根据分类和名称推导行业大类（txt 文件中没有 ind 列时兜底用）。"""
    if any(word in direction + name for word in ("白银", "黄金", "原油", "豆粕", "商品")):
        return "周期"
    if any(word in direction + name for word in ("芯片", "半导体", "信息科技", "互联网", "数据", "数字")):
        return "科技"
    if any(word in direction + name for word in ("创新药", "生物医药", "生物科技", "医药", "医疗", "中药")):
        return "医药"
    if category in INDUSTRY_THRESHOLDS:
        if category in {"资源", "商品"}:
            return "周期"
        if category == "其他":
            tech_words = ("数据", "信息", "专精特新", "绿色能源")
            return "科技" if any(word in direction + name for word in tech_words) else "消费"
        if category == "货币":
            return "债券"
        return category
    return "其他"


def _infer_product_type(name: str, category: str) -> str:
    if category == "LOF" or "LOF" in name.upper():
        return "LOF"
    return "ETF"


def _is_cross_border_or_qdii(name: str, category: str) -> bool:
    if category == "跨境":
        return True
    qdii_words = (
        "港", "美", "纳指", "标普", "日经", "德国", "法国", "沙特",
        "全球", "海外", "互联网LOF", "信息科技LOF",
    )
    return any(word in name for word in qdii_words)


def load_etf_txt(path: Path = ETF_TXT_PATH) -> OrderedDict:
    """
    从 etf.txt 读取 ETF 池（主数据源）。

    格式：name,code,category
    半导体设备ETF国泰,159516,科技

    规则：同代码只保留第一条，自动跳过空行
    """
    if not path.exists():
        return OrderedDict()

    etfs = OrderedDict()
    seen_codes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        name, code, cat = parts[0], parts[1], parts[2]
        if not code.isdigit() or code in seen_codes:
            continue
        seen_codes.add(code)
        # Use short name without ETF/LOF suffix as key
        key = name.replace('ETF','').replace('LOF','').strip()
        etfs[key] = {
            "code": code,
            "name": name,
            "cat": cat or "其他",
            "ind": _infer_industry(cat or "其他", key, name),
            "type": _infer_product_type(name, cat or "其他"),
            "is_qdii": _is_cross_border_or_qdii(name, cat or "其他"),
        }
    return etfs


def get_market_environment() -> dict:
    """获取市场宽度数据，判断市场状态，给出仓位上限建议"""
    try:
        from market_breadth import get_market_breadth, get_market_state
        breadth = get_market_breadth()
        state_info = get_market_state(breadth)
        market_state = state_info["state"]
    except Exception:
        breadth = {"up_count": 0, "down_count": 0, "up_ratio": 0}
        market_state = "未知"

    cap = get_position_cap(market_state).text

    up_ratio = breadth.get("up_ratio", 0)

    return {
        "market_state": market_state,
        "up_count": breadth.get("up_count", 0),
        "down_count": breadth.get("down_count", 0),
        "up_ratio": round(up_ratio, 2),
        "position_cap": cap,
        "note": _market_note(market_state, up_ratio),
    }


def _market_note(state: str, up_ratio: float) -> str:
    if state == "主升":
        return "市场赚钱效应强，可积极持仓，弹性仓位可适当放大"
    elif state == "震荡":
        return "市场分歧，优先保留主线仓位，弹性仓不追高"
    elif state == "退潮":
        return "赚钱效应弱但非冰点，精选强方向，现金控制在40%以内"
    elif state == "冰点":
        return "极度恐慌，以现金为主，只保留最强方向的试探仓"
    else:
        if up_ratio > 1.2:
            return "数据不完整但偏强，按震荡对待"
        elif up_ratio > 0.8:
            return "数据不完整但中性，按退潮对待"
        else:
            return "数据不完整但偏弱，谨慎操作"


# ============================================================
# Layer 1.5: 同花顺实时资金增强（可选，AKShare 不可用时做主要数据源）
# ============================================================
HITHINK_CLI = None  # 延迟初始化


def _get_hithink_cli():
    global HITHINK_CLI
    if HITHINK_CLI is None:
        skill_dir = os.path.join(os.path.expanduser("~"), ".claude", "skills", "hithink-market-query", "scripts")
        HITHINK_CLI = os.path.join(skill_dir, "cli.py") if os.path.isdir(skill_dir) else None
    return HITHINK_CLI


def _get_iwencai_env() -> dict:
    """读取同花顺 API Key"""
    env = os.environ.copy()
    settings_path = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings_path, encoding="utf-8") as f:
            s = json.load(f)
        for k in ("IWENCAI_API_KEY", "IWENCAI_BASE_URL"):
            if k in s.get("env", {}):
                env[k] = s["env"][k]
    except Exception:
        pass
    return env


def fetch_hithink_money_flow(codes: list[str], timeout: int = 25) -> dict[str, dict]:
    """批量查询 ETF 主力资金流向（同花顺）"""
    cli = _get_hithink_cli()
    if not cli:
        return {}
    env = _get_iwencai_env()
    if "IWENCAI_API_KEY" not in env:
        return {}

    import json as _json
    result = {}
    code_str = " ".join(codes)
    try:
        r = subprocess.run(
            [sys.executable, cli, "--query", f"{code_str} 最新价 涨跌幅 成交额 主力资金流向",
             "--limit", str(len(codes) + 2), "--timeout", str(timeout)],
            capture_output=True, text=True, env=env, timeout=timeout + 5,
        )
        if r.returncode != 0:
            return {}
        data = _json.loads(r.stdout)
        for item in data.get("datas", []):
            code = None
            for k, v in item.items():
                if "代码" in k and v:
                    code = str(v).split(".")[0] if "." in str(v) else str(v)
                    break
            if not code:
                continue
            flow = _extract_float(item, "主力") or 0
            chg = _extract_float(item, "涨跌")
            amt = _extract_float(item, "成交")
            result[code] = {
                "main_flow": flow,
                "hithink_chg": chg if chg is not None else None,
                "hithink_amt": amt if amt is not None else None,
            }
    except Exception:
        pass
    return result


def _extract_float(item: dict, keyword: str) -> float | None:
    for k, v in item.items():
        if keyword in k and v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def fetch_market_sentiment() -> set[str]:
    """查询当前市场热议方向（同花顺热搜/热门板块），返回行业关键词集合。"""
    cli = _get_hithink_cli()
    if not cli:
        return set()
    env = _get_iwencai_env()
    if "IWENCAI_API_KEY" not in env:
        return set()
    try:
        r = subprocess.run(
            [sys.executable, cli, "--query", "今日A股热门板块 资金关注方向", "--limit", "10", "--timeout", "20"],
            capture_output=True, text=True, env=env, timeout=25,
        )
        if r.returncode != 0:
            return set()
        data = json.loads(r.stdout)
        keywords = set()
        for item in data.get("datas", []):
            name = str(item.get("股票简称", "") or "")
            for kw in ["半导体", "芯片", "AI", "人工智能", "医药", "创新药", "新能源", "光伏",
                       "机器人", "军工", "通信", "光模块", "CPO", "消费", "白酒", "有色",
                       "黄金", "煤炭", "电力", "银行", "券商", "红利"]:
                if kw in name:
                    keywords.add(kw)
        return keywords
    except Exception:
        return set()


def calc_sentiment_bonus(etf_name: str, category: str, hot_keywords: set[str]) -> float:
    """如果 ETF 属于当前热议方向，给予加分。"""
    if not hot_keywords:
        return 0.0
    for kw in hot_keywords:
        if kw in etf_name or kw in category:
            return 5.0
    return 0.0


def enrich_with_hithink(results: list[dict], top_n: int = 15) -> dict:
    """用同花顺实时资金数据增强 ETF 排名结果（分批查询，避免超长 query）"""
    codes = [r["etf_code"] for r in results[:top_n] if r.get("etf_code")]
    if not codes:
        return {"data": {}, "available": False}
    all_data = {}
    batch_size = 8  # 每批最多 8 个 code，避免 query 超长
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        batch_data = fetch_hithink_money_flow(batch)
        all_data.update(batch_data)
    return {"data": all_data, "available": len(all_data) > 0}


def calc_money_flow_score(main_flow_yi: float | None) -> tuple[float, str]:
    """主力资金流入评分（总分 10 分）"""
    if main_flow_yi is None:
        return 5.0, "数据缺失"
    if main_flow_yi > 3:
        return 10.0, f"主力大幅流入 {main_flow_yi:.1f}亿"
    if main_flow_yi > 1:
        return 8.0, f"主力流入 {main_flow_yi:.1f}亿"
    if main_flow_yi > 0:
        return 6.0, f"主力微幅流入 {main_flow_yi:.1f}亿"
    if main_flow_yi > -1:
        return 4.0, f"主力微幅流出 {main_flow_yi:.1f}亿"
    if main_flow_yi > -3:
        return 2.0, f"主力流出 {main_flow_yi:.1f}亿"
    return 0.0, f"主力大幅流出 {main_flow_yi:.1f}亿"

import json  # noqa: E402
import subprocess  # noqa: E402

# ============================================================
# Layer 2: ETF 数据获取 + 阶段判断
# ============================================================
def _adjust_price_discontinuities(df: pd.DataFrame) -> pd.DataFrame:
    """
    修正 ETF/LOF 因折算、拆分导致的价格断裂。

    新浪等免费源有时返回不复权价格，例如前一日收盘 3.x、次日开盘 1.x。
    这种跳变不是投资亏损，若直接回测会制造虚假的 -60% 月收益。
    """
    if df.empty or "close" not in df.columns:
        return df
    df = df.sort_values("date").reset_index(drop=True).copy()
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for i in range(1, len(df)):
        prev_close = df.at[i - 1, "close"]
        today_ref = df.at[i, "open"] if "open" in df.columns else df.at[i, "close"]
        if pd.isna(prev_close) or pd.isna(today_ref) or prev_close <= 0:
            continue
        ratio = today_ref / prev_close
        if ratio < 0.65 or ratio > 1.55:
            df.loc[: i - 1, price_cols] = df.loc[: i - 1, price_cols] * ratio

    df["pct_chg"] = df["close"].pct_change().fillna(0) * 100
    return df


def fetch_etf_hist(code: str, start: str, end: str, product_type: str = "ETF") -> pd.DataFrame:
    """获取 ETF 历史日线。优先用东方财富，失败时回退新浪。"""
    # 1. 先试东方财富
    try:
        fetcher = ak.fund_lof_hist_em if str(product_type).upper() == "LOF" else ak.fund_etf_hist_em
        df = fetcher(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
        if len(df) > 0:
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "pct_chg",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            return _adjust_price_discontinuities(df)
    except Exception:
        pass

    # 2. 东方财富失败 → 回退新浪（不需要代理）
    try:
        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
        if len(df) > 0:
            col_map = {"date": "date", "open": "open", "high": "high",
                       "low": "low", "close": "close", "volume": "volume", "amount": "amount"}
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            # 新浪不直接给涨跌幅，后续 calc_etf_metrics 会自己算
            return _adjust_price_discontinuities(df)
    except Exception:
        pass

    empty = pd.DataFrame()
    empty.attrs["fetch_error"] = "东方财富和新浪均不可用"
    return empty


def calc_etf_metrics(df: pd.DataFrame, target_date: str) -> dict:
    if df.empty:
        fetch_error = df.attrs.get("fetch_error")
        if fetch_error:
            return {"error": f"数据源失败: {fetch_error}"}
        return {"error": "无数据"}
    target_ts = pd.to_datetime(target_date).normalize()
    td = target_ts
    today = df[df["date"] == td]
    # 若精确日期无数据，回退到 <= target_date 的最近交易日
    actual_date = target_date
    if today.empty:
        fallback = df[df["date"] <= td]
        if fallback.empty:
            return {"error": f"无 {target_date} 及之前数据"}
        td = pd.to_datetime(fallback.iloc[-1]["date"]).normalize()
        today = fallback.iloc[-1:]
        actual_date = td.strftime("%Y-%m-%d")
    today = today.iloc[0]
    stale_days = max(0, (target_ts - td).days)
    is_stale = stale_days > 0 and target_ts <= pd.Timestamp.today().normalize()
    close_today = today["close"]
    volume_today = float(today.get("volume", 0) or 0)
    amount_today = float(today.get("amount", 0) or 0)

    past_5 = df[df["date"] <= td].tail(6)
    past_20 = df[df["date"] <= td].tail(21)
    last_5 = df[df["date"] <= td].tail(5)

    ret_5d = (close_today / past_5.iloc[0]["close"] - 1) * 100 if len(past_5) >= 2 else None
    ret_20d = (close_today / past_20.iloc[0]["close"] - 1) * 100 if len(past_20) >= 2 else None

    # 5日动量加速度：最近5日涨跌 - 前一个5日涨跌
    if len(past_5) >= 6 and ret_5d is not None:
        prev_5 = df[df["date"] <= past_5.iloc[0]["date"]].tail(6)
        if len(prev_5) >= 2:
            prev_ret_5 = (past_5.iloc[0]["close"] / prev_5.iloc[0]["close"] - 1) * 100
            accel_5d = round(ret_5d - prev_ret_5, 1)
        else:
            accel_5d = None
    else:
        accel_5d = None

    vol_5d_avg = last_5["volume"].mean() if len(last_5) >= 2 else volume_today
    vol_ratio = volume_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
    amount_5d_avg = last_5["amount"].mean() if "amount" in last_5 and len(last_5) >= 2 else amount_today
    amount_ratio = amount_today / amount_5d_avg if amount_5d_avg > 0 else 1.0

    pct_chg = today.get("pct_chg", None)
    if pct_chg is None or pd.isna(pct_chg):
        prev = df[df["date"] < td].tail(1)
        pct_chg = (close_today / prev.iloc[0]["close"] - 1) * 100 if not prev.empty else 0.0

    # 60日动量
    past_60 = df[df["date"] <= td].tail(61)
    ret_60d = (close_today / past_60.iloc[0]["close"] - 1) * 100 if len(past_60) >= 2 else None

    # 创 N 日新高检测
    lookback_20 = df[df["date"] <= td].tail(21)
    is_20d_high = close_today >= lookback_20["close"].max() if len(lookback_20) >= 5 else False
    lookback_60 = df[df["date"] <= td].tail(61)
    is_60d_high = close_today >= lookback_60["close"].max() if len(lookback_60) >= 10 else False

    # 连涨天数
    consecutive_up = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["close"] > df.iloc[i - 1]["close"] if i > 0 else False:
            consecutive_up += 1
        else:
            break
    consecutive_up = min(consecutive_up, 15)  # 上限 15 天

    return {
        "close": close_today,
        "pct_chg": round(float(pct_chg), 2),
        "ret_5d": round(ret_5d, 2) if ret_5d is not None else None,
        "ret_20d": round(ret_20d, 2) if ret_20d is not None else None,
        "ret_60d": round(ret_60d, 2) if ret_60d is not None else None,
        "accel_5d": accel_5d,
        "vol_ratio": round(vol_ratio, 2),
        "amount_ratio": round(amount_ratio, 2),
        "amount_today": round(amount_today, 2),
        "amount_yi": round(amount_today / 100000000, 2),
        "volume_today": int(volume_today),
        "overheat": (ret_20d is not None and ret_20d > 30),
        "actual_date": actual_date,
        "target_date": target_date,
        "is_stale": is_stale,
        "stale_days": stale_days,
        "realtime_patched": False,
        "is_20d_high": is_20d_high,
        "is_60d_high": is_60d_high,
        "consecutive_up": consecutive_up,
    }


def _amount_to_yi(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    # 同花顺有时返回“亿元”数值，有时返回“元”数值。
    if abs(amount) > 10000:
        return amount / 100000000
    return amount


def patch_metrics_with_hithink(metrics: dict, hithink_item: dict, target_date: str) -> bool:
    """用同花顺实时涨跌幅/成交额补齐 AKShare 当日尚未更新的 ETF。"""
    chg = hithink_item.get("hithink_chg")
    if chg is None:
        return False
    try:
        metrics["pct_chg"] = round(float(chg), 2)
    except (TypeError, ValueError):
        return False

    amount_yi = _amount_to_yi(hithink_item.get("hithink_amt"))
    if amount_yi is not None and amount_yi >= 0:
        metrics["amount_yi"] = round(amount_yi, 2)
        metrics["amount_today"] = round(amount_yi * 100000000, 2)

    metrics["actual_date"] = target_date
    metrics["is_stale"] = False
    metrics["stale_days"] = 0
    metrics["realtime_patched"] = True
    return True


def mark_unpatched_stale_results(results: list[dict], target_date: str) -> int:
    """目标日扫描禁止旧行情混入主排名；未实时补全的旧行情直接熔断。"""
    count = 0
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        if m.get("is_stale") and not m.get("realtime_patched"):
            actual = m.get("actual_date", "未知")
            m["error"] = f"旧行情未补全: 行情日期 {actual} < 目标日期 {target_date}"
            r["stage"] = ("数据滞后", "禁止使用", "0%")
            r["score"] = {
                "total": 0.0,
                "raw_total": 0.0,
                "relative": 0.0,
                "volume": 0.0,
                "momentum": 0.0,
                "trend": 0.0,
                "liquidity": 0.0,
                "money_flow": 0.0,
                "catalyst": 0.0,
                "risk_factor": 1.0,
                "risk_note": "旧行情未补全，禁止进入主排名",
                "flow_note": "",
            }
            count += 1
    return count


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calc_auto_score(
    m: dict,
    stage: str,
    bench_pct: float,
    *,
    is_qdii: bool = False,
    premium_pct: float | None = None,
    consecutive_strong: int = 0,
) -> dict:
    """
    自动评分总分 100。

    权重设计原则：短线动量权重 > 长线（捕捉主线切换），
    连续强于指数是核心主线确认信号。
    """
    if "error" in m:
        return {
            "total": 0.0, "raw_total": 0.0, "relative": 0.0, "volume": 0.0,
            "momentum": 0.0, "liquidity": 0.0, "trend": 0.0,
            "catalyst": 0.0, "risk_factor": 1.0, "risk_note": "",
        }

    pct = float(m.get("pct_chg") or 0)
    r5 = float(m.get("ret_5d") or 0)
    r20 = float(m.get("ret_20d") or 0)
    amount_ratio = float(m.get("amount_ratio") or m.get("vol_ratio") or 1)
    amount_yi = float(m.get("amount_yi") or 0)
    relative = pct - bench_pct

    # 相对涨幅: 20 分
    relative_score = _clip((relative + 2) / 10 * 20, 0, 20)
    # 成交额放大: 20 分
    volume_score = _clip((amount_ratio - 0.7) / 1.5 * 20, 0, 20)

    # 动量: 短线(5日)15分 + 长线(20日)5分 = 20 分（短线优先捕捉主线切换）
    r5_score = _clip((r5 + 3) / 13 * 15, 0, 15)
    if r20 > 30:
        r20_score = _clip(5 - (r20 - 30) * 0.15, 0, 5)
    else:
        r20_score = _clip((r20 + 5) / 25 * 5, 0, 5)
    momentum_score = r5_score + r20_score

    # 反弹识别：5日强但20日弱 → 大概率只是脉冲，不是主线
    bounce_penalty = 1.0
    if r5 > 3 and r20 < 2:
        bounce_penalty = 0.7
    if r5 > 3 and r20 < 0:
        bounce_penalty = 0.5
    if r5 > 5 and r20 < -2:
        bounce_penalty = 0.3

    # 连续强于指数: 10 分（核心主线信号）
    trend_score = min(10.0, consecutive_strong * 1.2)

    # 流动性: 5 分
    if amount_yi >= 5:
        liquidity_score = 5
    elif amount_yi >= 1:
        liquidity_score = 4
    elif amount_yi >= 0.2:
        liquidity_score = 2
    else:
        liquidity_score = 0

    # 主力资金流: 10 分（同花顺增强）
    extra = m.get("_hithink_extra") or {}
    main_flow_yi = extra.get("main_flow_yi")
    flow_score, flow_note = calc_money_flow_score(main_flow_yi)

    catalyst_hints = {
        "扩散期": 10,
        "加速期": 8,
        "确认期": 6,
        "萌芽期": 4,
        "观察期": 2,
        "加速期⚠": 1,
        "加速见顶⚠": 0,
    }
    catalyst_score = catalyst_hints.get(stage, 0)

    # 市场热度加分（来自同花顺热搜/热门板块）
    hot_keywords = m.get("_hot_keywords", set())
    hot_name = m.get("_etf_name", "")
    hot_cat = m.get("_etf_category", "")
    sentiment_bonus = calc_sentiment_bonus(hot_name, hot_cat, hot_keywords)

    raw_total = relative_score + volume_score + momentum_score + trend_score + liquidity_score + flow_score + catalyst_score + sentiment_bonus
    risk_factor, risk_note = premium_discount_factor(is_qdii=is_qdii, premium_pct=premium_pct)
    total = raw_total * risk_factor * bounce_penalty

    return {
        "total": round(total, 1),
        "raw_total": round(raw_total, 1),
        "relative": round(relative_score, 1),
        "volume": round(volume_score, 1),
        "momentum": round(momentum_score, 1),
        "trend": round(trend_score, 1),
        "liquidity": round(liquidity_score, 1),
        "money_flow": round(flow_score, 1),
        "catalyst": round(catalyst_score, 1),
        "risk_factor": round(risk_factor, 2),
        "risk_note": risk_note,
        "flow_note": flow_note,
    }


# ============================================================
# Layer 2: 板块生命周期判断
# ============================================================
def classify_stage(m: dict, consecutive_strong: int = 0) -> tuple:
    """
    根据 ETF 数据判断板块所处阶段。
    返回 (阶段, 动作, 仓位建议)

    阶段定义：
      - 萌芽：首次放量+涨幅显著，但5日动量刚转正
      - 确认：连续2天+强于指数，量比>1.2
      - 扩散：5日动量>3%且量比>1，板块内多票共振
      - 加速：20日动量>30%（过热），或连续5天+强但量比下降
      - 趋势回撤：20/60日大趋势仍强，但短线剧烈回撤，适合持有观察而非直接清仓
      - 衰竭：量比<0.8且涨幅<0.5%，或连续弱于指数
    """
    pct = m.get("pct_chg", 0)
    r5 = m.get("ret_5d") or 0
    r20 = m.get("ret_20d") or 0
    r60 = m.get("ret_60d") or 0
    vol = m.get("vol_ratio", 1.0)
    accel = m.get("accel_5d") or 0
    amount_yi = m.get("amount_yi") or 0

    # 大趋势未破的剧烈洗盘：商品、资源、强趋势主题经常出现。
    # 先判为趋势回撤期，交给持仓规则用移动止损决定是否离场。
    if r20 > 25 and r60 > 40 and r5 < -8 and amount_yi >= 1:
        return ("趋势回撤期", "持有观察，跌破趋势止损再清仓", "10-20%")

    # 衰竭：连续走弱
    if r5 < -3 and vol < 0.8:
        return ("衰竭期", "清仓或大幅减仓", "0-10%")
    if r5 < -2 and pct < -1:
        return ("衰弱期", "减仓观望", "5-15%")

    # 过热判断（加入量价结构）：
    # - 缩量+减速 → 加速见顶⚠
    # - 放量+加速 → 正常加速期（趋势延续，不过早减仓）
    # - 放量+减速 → 加速期⚠（持有但不加仓）
    if r20 > 30:
        if vol < 1.0 and accel < 0:
            return ("加速见顶⚠", "逐步减仓，不追", "10-20%")
        if vol >= 1.2 and accel >= 0:
            return ("加速期", "趋势延续，持有", "20-30%")
        return ("加速期⚠", "持有但不加仓，设紧止损", "15-25%")

    # 加速：5日动量>8%且加速中
    if r5 > 8 and accel > 0:
        return ("加速期", "持有，逐步提止损", "20-30%")

    # 扩散：5日>3%且放量
    if r5 > 3 and vol > 1.2:
        return ("扩散期", "核心仓位，可加仓", "30-40%")

    # 确认：连续强势或放量上涨
    if (r5 > 2 and vol > 1.0) or consecutive_strong >= 2:
        return ("确认期", "试探建仓/加仓", "15-25%")
    if pct > 2 and vol > 1.2:
        return ("确认期", "试探建仓", "10-20%")

    # 萌芽：首次放量
    if pct > 1.5 and vol > 1.5:
        return ("萌芽期", "加入观察，等待回踩", "0-10%")

    # 震荡/休眠
    if abs(pct) < 1:
        return ("休眠期", "不操作，保持观察", "0%")
    if r5 < 0 and vol < 1:
        return ("弱势期", "不参与", "0%")

    return ("观察期", "等待信号", "0%")


def calc_consecutive_strong(df: pd.DataFrame, bench_df: pd.DataFrame, target_date: str) -> int:
    """计算 ETF 连续强于基准的天数"""
    if df.empty or bench_df.empty:
        return 0
    td = pd.to_datetime(target_date)
    # 取最近10天的数据
    recent = df[df["date"] <= td].tail(11)
    bench_recent = bench_df[bench_df["date"] <= td].tail(11)
    if len(recent) < 2:
        return 0

    count = 0
    for i in range(len(recent) - 1, 0, -1):
        etf_chg = (recent.iloc[i]["close"] / recent.iloc[i-1]["close"] - 1) * 100
        # 找对应日期的基准
        d = recent.iloc[i]["date"]
        b_row = bench_recent[bench_recent["date"] == d]
        b_prev = bench_recent[bench_recent["date"] == recent.iloc[i-1]["date"]]
        if b_row.empty or b_prev.empty:
            continue
        bench_chg = (b_row.iloc[0]["close"] / b_prev.iloc[0]["close"] - 1) * 100
        if etf_chg > bench_chg:
            count += 1
        else:
            break
    return count


# ============================================================
# 输出生成
# ============================================================
def _rank_key(r: dict) -> tuple:
    m = r.get("metrics", {})
    score = r.get("score", {}).get("total", 0)
    stage = r.get("stage", ("", "", ""))[0]
    return (
        score,
        STAGE_PRIORITY.get(stage, -99),
        m.get("pct_chg", 0) or 0,
        m.get("ret_5d", 0) or 0,
        m.get("amount_yi", 0) or 0,
    )


def _is_dominant_theme(candidates: list[dict]) -> tuple[bool, str | None]:
    """主线判定：如果评分 TOP2 同属一个行业，该行业即为主线。"""
    if len(candidates) < 2:
        return False, None
    i1 = candidates[0].get("industry", "")
    i2 = candidates[1].get("industry", "")
    if i1 and i1 == i2 and i1 not in ("宽基", "债券", "货币"):
        return True, i1
    # 备选：如果榜首得分比第二名高 10 分以上，也判定为主线
    if len(candidates) >= 2:
        s1 = candidates[0]["score"]["total"]
        s2 = candidates[1]["score"]["total"]
        if s1 - s2 > 10 and i1 not in ("宽基", "债券", "货币"):
            return True, i1
    return False, None


def _pick_diversified(candidates: list[dict], target_count: int = 3) -> list[dict]:
    # 若第 3 名评分比第 2 名低 8 分以上 → 只选 2 只，不硬凑
    if len(candidates) >= 3:
        s2 = candidates[1]["score"]["total"]
        s3 = candidates[2]["score"]["total"]
        if s2 - s3 > 8:
            target_count = 2
    """
    主线判定：TOP2 同行业 → 主线确认 → 允许集中 2 只 + 1 只分散。
    无主线 → 强制去相关（每行业最多 1 只）。
    """
    is_dominant, dom_industry = _is_dominant_theme(candidates)
    picks = []
    used_industries = set()
    skip = {"宽基", "债券", "货币"}
    dom_count = 0

    for r in candidates:
        ind = r.get("industry", "其他")
        if ind in skip:
            continue
        if is_dominant and ind == dom_industry and dom_count < 2:
            picks.append(r)
            dom_count += 1
            if dom_count >= 2:
                used_industries.add(ind)
            if len(picks) >= target_count:
                return picks
            continue
        if ind in used_industries:
            continue
        if is_dominant and ind != dom_industry:
            picks.append(r)
            used_industries.add(ind)
            if len(picks) >= target_count:
                return picks
            continue
        if not is_dominant:
            picks.append(r)
            used_industries.add(ind)
            if len(picks) >= target_count:
                return picks

    for r in candidates:
        if r in picks:
            continue
        picks.append(r)
        if len(picks) >= target_count:
            break
    return picks


def generate_report(results: list, market_env: dict, bench_pct: float, target_date: str, hithink_used: bool = False) -> str:
    """三层分析报告"""
    source_tag = " | 同花顺资金增强已启用" if hithink_used else ""

    lines = []
    lines.append(f"# ETF 三层分析报告 — {target_date}{source_tag}")
    lines.append("")

    # ---- 第一层：市场环境 ----
    lines.append("## 第一层：市场环境")
    lines.append("")
    me = market_env
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 市场状态 | **{me['market_state']}** |")
    lines.append(f"| 上涨家数 | {me['up_count']} |")
    lines.append(f"| 下跌家数 | {me['down_count']} |")
    lines.append(f"| 涨跌比 | {me['up_ratio']} |")
    lines.append(f"| 建议仓位上限 | **{me['position_cap']}** |")
    lines.append("")
    lines.append(f"> {me['note']}")
    lines.append("")

    stale_errors = [
        r for r in results
        if "error" in r.get("metrics", {}) and "旧行情未补全" in str(r.get("metrics", {}).get("error", ""))
    ]
    realtime_patched = [
        r for r in results
        if r.get("metrics", {}).get("realtime_patched")
    ]
    if stale_errors or realtime_patched:
        lines.append("## 数据质量")
        lines.append("")
        lines.append(f"- 同花顺实时补全：{len(realtime_patched)} 只。")
        lines.append(f"- 旧行情剔除：{len(stale_errors)} 只。")
        if stale_errors:
            sample = "、".join(
                f"{r['direction']}({r.get('metrics', {}).get('actual_date', '未知')})"
                for r in stale_errors[:8]
            )
            lines.append(f"- 剔除样本：{sample}")
        lines.append("")

    # ---- 第二层：板块阶段 ----
    lines.append("## 第二层：板块生命周期")
    lines.append("")
    lines.append(f"沪深300基准涨跌: **{bench_pct:+.2f}%**")
    lines.append("")
    valid_results = [r for r in results if "error" not in r["metrics"]]
    ranked_results = sorted(valid_results, key=_rank_key, reverse=True)

    lines.append("| 排名 | 方向 | 类型 | 分类 | 行情日期 | 涨跌幅 | 5日动量 | 20日动量 | 量比 | 成交额(亿) | 连强 | 阶段 | 评分 | 风控 | 建议动作 | 仓位 |")
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|")

    actionable = []  # 可操作的方向

    for rank, r in enumerate(ranked_results, 1):
        m = r["metrics"]
        stage, action, weight = r["stage"]
        oh = " ⚠" if m.get("overheat") else ""
        r5 = f"{m['ret_5d']:+.1f}%" if m.get("ret_5d") is not None else "-"
        r20 = f"{m['ret_20d']:+.1f}%" if m.get("ret_20d") is not None else "-"
        aliases = r.get("aliases") or []
        alias_note = f"（含代理：{'、'.join(aliases)}）" if aliases else ""
        risk_note = r.get("score", {}).get("risk_note") or "—"
        product_note = r.get("product_type", "ETF")
        if r.get("is_qdii"):
            product_note = f"{product_note}/QDII"

        lines.append(
            f"| {rank} | {r['direction']}{oh}{alias_note} | {product_note} | {r['category']} "
            f"| {m.get('actual_date', target_date)} | {m['pct_chg']:+.2f}% | {r5} | {r20} | {m['amount_ratio']:.1f}x "
            f"| {m['amount_yi']:.2f} | {r.get('strong_days', 0)} | **{stage}** "
            f"| {r['score']['total']:.1f} | {risk_note} | {action} | {weight} |"
        )

        # 收集非休眠/非衰竭的方向，后续仍按评分排序
        if stage not in EXCLUDED_STAGES and stage != "观察期":
            actionable.append(r)
        elif stage == "观察期" and m.get("pct_chg", 0) > 1:
            actionable.append(r)

    lines.append("")
    lines.append("> 阶段逻辑来自 stock-analyzer-skill 的 sector_specialist + market_breadth 方法论")
    lines.append("> 行业差异化阈值：医药(ROE≥12%,增速≥20%) / 科技(ROE≥10%,增速≥30%) / 消费(ROE≥15%,增速≥10%) / 周期(看商品价格分位)")
    lines.append("")

    if not ranked_results:
        error_samples = [r for r in results if "error" in r["metrics"]][:5]
        lines.append("> **数据熔断**：本次没有任何 ETF/LOF 有效行情，禁止继续生成正式 selection。")
        if error_samples:
            lines.append("")
            lines.append("数据源错误样本：")
            for r in error_samples:
                lines.append(f"- {r['direction']}: {r['metrics'].get('error')}")
        lines.append("")

    # ---- 第三层：方向建议 ----
    lines.append("## 第三层：方向建议")
    lines.append("")

    # 按阶段分组
    stages_order = ["扩散期", "加速期", "确认期", "萌芽期", "观察期", "加速见顶⚠"]
    stage_groups = OrderedDict()
    for s in stages_order:
        stage_groups[s] = sorted([r for r in actionable if r["stage"][0] == s], key=_rank_key, reverse=True)

    for stage, items in stage_groups.items():
        if not items:
            continue
        lines.append(f"### {stage}")
        lines.append("")
        for r in items:
            ind = r["industry"]
            thresh = INDUSTRY_THRESHOLDS.get(ind, {})
            risk = thresh.get("risk_note", "—")
            risk_note = r.get("score", {}).get("risk_note") or "—"
            product_note = r.get("product_type", "ETF")
            if r.get("is_qdii"):
                product_note = f"{product_note}/QDII"
            lines.append(f"- **{r['direction']}**（{product_note}，{r['category']}，评分 {r['score']['total']:.1f}）→ {r['stage'][1]}，仓位 {r['stage'][2]}")
            lines.append(f"  - 行业风险点：{risk}")
            lines.append(f"  - 产品风控：{risk_note}")
            lines.append(f"  - ETF: {r['etf_name']} `{r['etf_code']}`")
        lines.append("")

    # 汇总
    lines.append("### 组合建议")
    lines.append("")
    candidate_picks = sorted(
        [
            r for r in actionable
            if r["stage"][0] in NEW_MONEY_STAGES and r["category"] != "货币"
        ],
        key=_rank_key,
        reverse=True,
    )
    raw_picks = candidate_picks[:5]
    picks = _pick_diversified(candidate_picks, target_count=3)

    if raw_picks:
        lines.append("原始强度前三：")
        lines.append("")
        lines.append("| 排名 | 方向 | 类型 | 行业 | 阶段 | 评分 | ETF/LOF |")
        lines.append("|---:|---|---|---|---|---:|---|")
        for i, r in enumerate(raw_picks[:3], 1):
            product_note = r.get("product_type", "ETF")
            if r.get("is_qdii"):
                product_note = f"{product_note}/QDII"
            lines.append(
                f"| {i} | {r['direction']} | {product_note} | {r['industry']} | {r['stage'][0]} | {r['score']['total']:.1f} | {r['etf_name']} `{r['etf_code']}` |"
            )
        lines.append("")
        lines.append("去相关组合候选：")
        lines.append("")
        lines.append("| 优先级 | 方向 | 类型 | 行业 | 阶段 | 评分 | 建议仓位 | ETF或候选标的 |")
        lines.append("|---:|---|---|---|---|---:|---|---|")
        for i, r in enumerate(picks[:3], 1):
            product_note = r.get("product_type", "ETF")
            if r.get("is_qdii"):
                product_note = f"{product_note}/QDII"
            lines.append(
                f"| {i} | {r['direction']} | {product_note} | {r['industry']} | {r['stage'][0]} | {r['score']['total']:.1f} | {r['stage'][2]} "
                f"| {r['etf_name']} `{r['etf_code']}` |"
            )
        lines.append("")
        lines.append("> 原始强度用于确认主线，去相关候选用于避免三个标的押同一政策/情绪周期；正式 selection 仍需做催化验证、个股买点、ETF/LOF 折溢价风险和盈亏比检查。")
    else:
        lines.append("> 当前无扩散期或确认期方向，建议以现金为主或仅保留试探仓")

    return "\n".join(lines)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ETF 三层分析")
    parser.add_argument("--date", default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=30, help="输出前 N 名 (默认30)")
    parser.add_argument("--workers", type=int, default=8, help="并发抓取 ETF 行情的线程数 (默认8)")
    parser.add_argument("--hithink", action="store_true", help="启用同花顺资金增强（AKShare 不可用时强制回退）")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")

    # Layer 1: 市场环境
    print("Layer 1: 获取市场宽度...", file=sys.stderr)
    market_env = get_market_environment()
    print(f"  状态={market_env['market_state']}, 涨跌比={market_env['up_ratio']}, 仓位上限={market_env['position_cap']}", file=sys.stderr)

    etf_pool = load_etf_txt()
    if not etf_pool:
        print("错误: scripts/etf.txt 为空或不存在", file=sys.stderr)
        sys.exit(1)
    
    print(f"Layer 2: ETF 全量扫描 ({len(etf_pool)} 只，来源=scripts/etf.txt)...", file=sys.stderr)
    start_date = (pd.to_datetime(target_date) - timedelta(days=45)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    # 基准：查找沪深300 ETF (code=510300)
    bench_cfg = None
    for v in etf_pool.values():
        if v["code"] == BENCHMARK_CODE:
            bench_cfg = v
            break
    bench_df = fetch_etf_hist(
        bench_cfg["code"],
        start_date,
        end_date,
        bench_cfg.get("type", "ETF"),
    ) if bench_cfg else pd.DataFrame()
    bench_m = calc_etf_metrics(bench_df, target_date)
    bench_pct = bench_m.get("pct_chg", 0.0)

    def analyze_one(item: tuple[str, dict]) -> dict:
        direction, cfg = item
        df = fetch_etf_hist(cfg["code"], start_date, end_date, cfg.get("type", "ETF"))
        metrics = calc_etf_metrics(df, target_date)
        strong_days = calc_consecutive_strong(df, bench_df, target_date)
        stage = classify_stage(metrics, strong_days)
        score = calc_auto_score(
            metrics,
            stage[0],
            bench_pct,
            is_qdii=cfg.get("is_qdii", False),
            premium_pct=cfg.get("premium_pct"),
            consecutive_strong=strong_days,
        )
        return {
            "direction": direction,
            "etf_code": cfg["code"],
            "etf_name": cfg["name"],
            "category": cfg["cat"],
            "industry": cfg["ind"],
            "product_type": cfg.get("type", "ETF"),
            "is_qdii": cfg.get("is_qdii", False),
            "aliases": cfg.get("aliases", []),
            "metrics": metrics,
            "stage": stage,
            "strong_days": strong_days,
            "score": score,
        }

    total = len(etf_pool)
    results = []
    workers = max(1, min(args.workers, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(analyze_one, item): item[0] for item in etf_pool.items()}
        for done_count, future in enumerate(as_completed(future_map), 1):
            direction = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "direction": direction,
                    "etf_code": "",
                    "etf_name": "",
                    "category": "未知",
                    "industry": "其他",
                    "product_type": "未知",
                    "is_qdii": False,
                    "aliases": [],
                    "metrics": {"error": str(exc)},
                    "stage": ("未知", "跳过", "0%"),
                    "strong_days": 0,
                    "score": {"total": 0, "relative": 0, "volume": 0, "momentum": 0, "liquidity": 0, "catalyst": 0},
                }
            results.append(result)
            if done_count % 10 == 0 or done_count == total:
                print(f"  已完成 {done_count}/{total}: {direction}", file=sys.stderr)

    # Layer 2.5: 同花顺资金增强 + 热度查询
    valid_count = sum(1 for r in results if "error" not in r["metrics"])
    hithink_used = False
    hot_keywords = set()
    if args.hithink or valid_count == 0:
        hot_keywords = fetch_market_sentiment()
        if hot_keywords:
            print(f"  市场热度: 热议方向 -> {', '.join(sorted(hot_keywords)[:6])}", file=sys.stderr)

    if args.hithink or valid_count == 0:
        enrichment = enrich_with_hithink(sorted(results, key=_rank_key, reverse=True), top_n=50)
        if enrichment["available"]:
            hithink_used = True
            print(f"  同花顺资金增强: 获取了 {len(enrichment['data'])} 只 ETF 的主力资金数据", file=sys.stderr)
            hf = enrichment["data"]
            for r in results:
                code = r.get("etf_code", "")
                if code in hf:
                    flow_yi = hf[code].get("main_flow", 0) or 0
                    try:
                        flow_yi = float(flow_yi) / 1e8
                    except (ValueError, TypeError):
                        flow_yi = 0
                    patch_metrics_with_hithink(r["metrics"], hf[code], target_date)
                    r["metrics"]["_hot_keywords"] = hot_keywords
                    r["metrics"]["_etf_name"] = r.get("etf_name", "")
                    r["metrics"]["_etf_category"] = r.get("category", "")
                    r["metrics"]["_hithink_extra"] = {
                        "main_flow_yi": flow_yi,
                        "hithink_chg": hf[code].get("hithink_chg"),
                        "hithink_amt": hf[code].get("hithink_amt"),
                    }
                    # Re-score with money flow
                    r["score"] = calc_auto_score(
                        r["metrics"], r["stage"][0], bench_pct,
                        is_qdii=r.get("is_qdii", False),
                        premium_pct=r.get("premium_pct"),
                    )
        elif valid_count == 0:
            print(f"错误: {target_date} 没有任何有效行情且同花顺数据也不可用", file=sys.stderr)
            print("提示: 请检查 AKShare 网络连接，或手动运行 hithink CLI 确认 API 配置", file=sys.stderr)
            sys.exit(2)

    stale_excluded = mark_unpatched_stale_results(results, target_date)
    if stale_excluded:
        print(f"  数据新鲜度门禁: 剔除 {stale_excluded} 只未实时补全的旧行情 ETF/LOF", file=sys.stderr)

    # Layer 3: 生成报告
    out_path = PROJECT_ROOT / "codex" / "stock" / target_date / "etf_scan.md"
    os.makedirs(out_path.parent, exist_ok=True)
    mode_note = " + 同花顺资金增强" if hithink_used else ""
    report = generate_report(results, market_env, bench_pct, target_date, hithink_used=hithink_used)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    display_path = f"codex/stock/{target_date}/etf_scan.md"
    print(f"  报告: {display_path} (data_source={target_date}{mode_note})", file=sys.stderr)

    # 终端摘要
    valid_count = sum(1 for r in results if "error" not in r["metrics"])
    if valid_count == 0 and not hithink_used:
        print(f"错误: {target_date} 没有任何 ETF/LOF 有效行情；已写入空报告用于排查: {display_path}", file=sys.stderr)
        sys.exit(2)

    ranked = sorted(
        [r for r in results if "error" not in r["metrics"] and r["stage"][0] not in ("休眠期","衰弱期","衰竭期","弱势期")],
        key=_rank_key,
        reverse=True,
    )
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"市场状态: {market_env['market_state']} | 仓位上限: {market_env['position_cap']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in ranked[:args.top]:
        m = r["metrics"]
        oh = " ⚠过热" if m.get("overheat") else ""
        print(
            f"  {r['direction']:<14} {m['pct_chg']:>+7.2f}% | 评分 {r['score']['total']:>5.1f} | "
            f"{r['stage'][0]:<8} | {r['stage'][1]:<20} | {r['stage'][2]}{oh}",
            file=sys.stderr,
        )
    print(f"\n完整报告: {display_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
