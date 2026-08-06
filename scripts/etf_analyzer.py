"""
ETF 板块三层分析脚本
====================
第一层：市场环境（市场宽度 → 总仓位上限）
第二层：板块阶段（ETF动量 + 行业差异化阈值 → 板块生命周期 + 动作建议）
第三层：池内标的比较（所有正式候选均来自 scripts/etf.txt）

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

from risk_rules import (
    MAX_NEW_POSITION_ONE_DAY_DOMINANCE,
    allocate_instrument_weights,
    category_root,
    get_position_cap,
    max_same_risk_cluster,
    max_selection_count,
    new_position_trend_gate,
    premium_discount_factor,
    risk_cluster,
    score_layers_pass,
)
from market_state import classify_market_state
from market_data_cache import adjust_price_discontinuities, load_history
from skill_paths import find_hithink_cli, find_skill_scripts
from strategy_version import load_current_manifest, verify_manifest

# Stock-analyzer-skill 路径（兼容 .claude/.codex/npm 全局安装）
SKILL_SCRIPTS = find_skill_scripts("stock-analyzer-skill")
if SKILL_SCRIPTS:
    sys.path.insert(0, str(SKILL_SCRIPTS))
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
PORTFOLIO_MIN_AMOUNT_YI = 1.0
PORTFOLIO_MAX_SAME_INDUSTRY = 2
PORTFOLIO_MAX_BROAD = 1
PORTFOLIO_THEME_TOLERANCE = 6.0
GENERIC_DIRECTION_WORDS = ("科技", "信息技术", "科创50", "科创100", "科创综指", "创业板", "中证")
SPECIFIC_THEME_WORDS = (
    "创新药", "生物医药", "半导体设备", "科创新材料", "机器人", "通信",
    "消费电子", "证券保险", "卫星", "工业母机", "科创200", "战略新兴",
)

# 行业差异化阈值（统一维护于 risk_rules.py）
from risk_rules import INDUSTRY_THRESHOLDS


def _infer_industry(category: str, direction: str, name: str) -> str:
    """根据分类和名称推导行业大类（txt 文件中没有 ind 列时兜底用）。

    支持两种 category 格式：
    - 旧格式: "科技", "医药", "消费" 等
    - 新格式: "科技/半导体", "医药/创新药" 等（大类/细分）
    """
    # 提取大类（兼容 大类/细分 格式）
    major_cat = category.split("/")[0] if "/" in category else category

    # 名称关键词优先
    if any(word in direction + name for word in ("白银", "黄金", "原油", "豆粕", "商品")):
        return "周期"
    if any(word in direction + name for word in ("芯片", "半导体", "信息科技", "互联网", "数据", "数字")):
        return "科技"
    if any(word in direction + name for word in ("创新药", "生物医药", "生物科技", "医药", "医疗", "中药")):
        return "医药"

    # 大类映射
    major_to_ind = {
        "科技": "科技", "医药": "医药", "消费": "消费", "金融": "金融",
        "资源": "周期", "商品": "周期", "新能源": "新能源", "军工": "军工",
        "公用事业": "公用事业", "红利": "红利", "债券": "债券", "货币": "债券",
        "跨境": "跨境", "宽基": "宽基", "LOF": "其他", "其他": "其他",
    }
    if major_cat in major_to_ind:
        mapped = major_to_ind[major_cat]
        if mapped == "其他":
            tech_words = ("数据", "信息", "专精特新", "绿色能源")
            return "科技" if any(word in direction + name for word in tech_words) else "消费"
        return mapped

    # 兜底：原来 INDUSTRY_THRESHOLDS 的兼容逻辑
    if category in INDUSTRY_THRESHOLDS:
        if category in {"资源", "商品"}:
            return "周期"
        if category == "货币":
            return "债券"
        return category
    return "其他"


def _infer_product_type(name: str, category: str) -> str:
    if category == "LOF" or "LOF" in name.upper():
        return "LOF"
    return "ETF"


def _is_cross_border_or_qdii(name: str, category: str) -> bool:
    if str(category).startswith("跨境"):
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


def get_market_environment(*, offline: bool = False) -> dict:
    """获取市场宽度数据，判断市场状态，给出仓位上限建议"""
    fallback_state = "未知"
    try:
        from market_breadth import get_market_breadth, get_market_state
        breadth = get_market_breadth()
        state_info = get_market_state(breadth)
        fallback_state = state_info["state"]
    except Exception:
        breadth = {"up_count": 0, "down_count": 0, "up_ratio": 0}
    up_ratio = breadth.get("up_ratio")
    try:
        end = pd.Timestamp.today().normalize()
        start = end - timedelta(days=45)
        benchmark = fetch_etf_hist(
            BENCHMARK_CODE,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            "ETF",
            offline=offline,
        )
        closes = benchmark.sort_values("date")["close"].astype(float).tail(20).tolist()
        benchmark_pct = (
            (closes[-1] / closes[-2] - 1) * 100
            if len(closes) >= 2
            else None
        )
        market_state = classify_market_state(
            closes,
            breadth_ratio=float(up_ratio) if up_ratio is not None else None,
            benchmark_pct=benchmark_pct,
        )
        if market_state == "未知":
            market_state = fallback_state
    except Exception:
        market_state = fallback_state

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


def get_replay_market_environment(target_date: str, *, offline: bool = False) -> dict:
    """历史复盘禁止引用当前实时市场宽度，避免未来数据污染。"""
    target = pd.to_datetime(target_date)
    state = "未知"
    try:
        benchmark = fetch_etf_hist(
            BENCHMARK_CODE,
            (target - timedelta(days=45)).strftime("%Y%m%d"),
            target.strftime("%Y%m%d"),
            "ETF",
            offline=offline,
        )
        closes = benchmark.sort_values("date")["close"].astype(float).tail(20).tolist()
        benchmark_pct = (
            (closes[-1] / closes[-2] - 1) * 100
            if len(closes) >= 2
            else None
        )
        state = classify_market_state(
            closes,
            breadth_ratio=None,
            benchmark_pct=benchmark_pct,
        )
    except Exception:
        pass
    cap = get_position_cap(state).text
    return {
        "market_state": state,
        "up_count": "不使用实时宽度",
        "down_count": "不使用实时宽度",
        "up_ratio": "不使用实时宽度",
        "position_cap": cap,
        "note": (
            f"{target_date} 为历史目标日，ETF 排名只使用目标日及以前行情；"
            "实时市场宽度不参与复盘判断；市场状态使用同一基准趋势规则。"
        ),
    }


def get_latest_realtime_trade_ts(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """同花顺实时快照对应的最新交易日，盘后跨午夜仍归属前一交易日。"""
    ts = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    day = ts.normalize()

    if ts.weekday() >= 5:
        day = day - pd.offsets.BDay(1)
    elif ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
        day = day - pd.offsets.BDay(1)

    return pd.Timestamp(day).normalize()


def _market_note(state: str, up_ratio: float) -> str:
    if state == "主升":
        return "市场赚钱效应强，可积极持仓，弹性仓位可适当放大"
    elif state == "震荡":
        return "市场分歧，优先保留主线仓位，弹性仓不追高"
    elif state == "退潮":
        if up_ratio >= 2.0:
            return "市场宽度强修复，但基准尚未收复中期趋势；按反弹确认期处理，不把单日普涨直接判为主升"
        return "赚钱效应弱，ETF 排名只作候选；不强制三标的，允许高现金等待主力确认"
    elif state == "退潮末期":
        if up_ratio >= 2.0:
            return "超跌后的宽度修复正在发生，中期趋势仍弱；等待连续性和主线强度共同确认"
        return "中期趋势处于退潮末期，只保留最强方向并等待宽度修复"
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
        HITHINK_CLI = find_hithink_cli()
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
    result = adjust_price_discontinuities(df)
    if not result.empty:
        result["pct_chg"] = result["close"].pct_change().fillna(0) * 100
    return result


def _fetch_etf_hist_network(
    code: str,
    start: str,
    end: str,
    product_type: str = "ETF",
) -> pd.DataFrame:
    """获取 ETF 历史日线。优先用东方财富，失败时回退新浪。"""
    # 1. 先试东方财富
    try:
        fetcher = ak.fund_lof_hist_em if str(product_type).upper() == "LOF" else ak.fund_etf_hist_em
        df = fetcher(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
        if len(df) > 0:
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude_provider",
                "涨跌幅": "pct_chg_provider", "涨跌额": "change_provider",
                "换手率": "turnover_rate",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            result = _adjust_price_discontinuities(df)
            result.attrs["data_source"] = "eastmoney"
            return result
    except Exception:
        pass

    # 2. 东方财富失败 → 回退新浪（不需要代理）
    try:
        prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
        df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
        if len(df) > 0:
            col_map = {"date": "date", "open": "open", "high": "high",
                       "low": "low", "close": "close", "volume": "volume",
                       "amount": "amount", "postVol": "after_hours_volume",
                       "postAmt": "after_hours_amount"}
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            provider_min_date = df["date"].min()
            # 新浪不直接给涨跌幅，后续 calc_etf_metrics 会自己算
            requested_start = pd.to_datetime(start)
            requested_end = pd.to_datetime(end)
            df = df[(df["date"] >= requested_start) & (df["date"] <= requested_end)]
            result = _adjust_price_discontinuities(df)
            result.attrs["data_source"] = "sina"
            result.attrs["history_complete_left"] = True
            result.attrs["provider_min_date"] = provider_min_date.strftime("%Y-%m-%d")
            return result
    except Exception:
        pass

    empty = pd.DataFrame()
    empty.attrs["fetch_error"] = "东方财富和新浪均不可用"
    return empty


def fetch_etf_hist(
    code: str,
    start: str,
    end: str,
    product_type: str = "ETF",
    *,
    refresh_cache: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Load ETF history locally first and fetch only missing date edges."""
    return load_history(
        code,
        start,
        end,
        product_type,
        network_fetcher=_fetch_etf_hist_network,
        refresh=refresh_cache,
        offline=offline,
    )


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
    known = df[df["date"] <= td].sort_values("date").copy()
    stale_days = max(0, (target_ts - td).days)
    is_stale = stale_days > 0 and target_ts <= pd.Timestamp.today().normalize()
    close_today = today["close"]
    volume_today = float(today.get("volume", 0) or 0)
    amount_today = float(today.get("amount", 0) or 0)

    past_5 = known.tail(6)
    past_20 = known.tail(21)
    last_5 = known.tail(5)

    ret_5d = (close_today / past_5.iloc[0]["close"] - 1) * 100 if len(past_5) >= 2 else None
    ret_20d = (close_today / past_20.iloc[0]["close"] - 1) * 100 if len(past_20) >= 2 else None

    # 5日动量加速度：最近5日涨跌 - 前一个5日涨跌
    if len(past_5) >= 6 and ret_5d is not None:
        prev_5 = known[known["date"] <= past_5.iloc[0]["date"]].tail(6)
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
        prev = known[known["date"] < td].tail(1)
        pct_chg = (close_today / prev.iloc[0]["close"] - 1) * 100 if not prev.empty else 0.0

    # 60日动量
    past_60 = known.tail(61)
    ret_60d = (close_today / past_60.iloc[0]["close"] - 1) * 100 if len(past_60) >= 2 else None

    close_series = pd.to_numeric(known["close"], errors="coerce").dropna()
    daily_returns = close_series.pct_change().dropna()
    ma20 = float(close_series.tail(20).mean()) if len(close_series) >= 5 else float(close_today)
    ma60 = float(close_series.tail(60).mean()) if len(close_series) >= 10 else ma20
    previous_ma20 = (
        float(close_series.iloc[:-5].tail(20).mean())
        if len(close_series) >= 25
        else ma20
    )
    ma20_slope_5d = (ma20 / previous_ma20 - 1) * 100 if previous_ma20 > 0 else 0.0
    distance_ma20 = (float(close_today) / ma20 - 1) * 100 if ma20 > 0 else 0.0
    distance_ma60 = (float(close_today) / ma60 - 1) * 100 if ma60 > 0 else 0.0

    recent_20_returns = daily_returns.tail(20)
    net_20 = abs(float(close_today) / float(past_20.iloc[0]["close"]) - 1) if len(past_20) >= 2 else 0.0
    path_20 = float(recent_20_returns.abs().sum())
    trend_efficiency_20 = net_20 / path_20 if path_20 > 0 else 0.0
    volatility_20 = (
        float(recent_20_returns.std(ddof=0) * (252 ** 0.5) * 100)
        if len(recent_20_returns) >= 10
        else 0.0
    )

    recent_5_returns = daily_returns.tail(5)
    positive_5_returns = recent_5_returns.clip(lower=0)
    one_day_dominance_5 = (
        float(positive_5_returns.max() / positive_5_returns.sum())
        if not positive_5_returns.empty and positive_5_returns.sum() > 0
        else 0.0
    )
    positive_days_5 = int((recent_5_returns > 0).sum())

    amount_series = (
        pd.to_numeric(known["amount"], errors="coerce")
        if "amount" in known
        else pd.Series(dtype=float)
    )
    if len(amount_series.dropna()) >= 5:
        recent_amount = float(amount_series.tail(3).mean())
        base_amount = float(amount_series.tail(20).mean())
        amount_persistence_3d = recent_amount / base_amount if base_amount > 0 else 1.0
    else:
        amount_persistence_3d = 1.0

    # 创 N 日新高检测
    lookback_20 = known.tail(21)
    is_20d_high = close_today >= lookback_20["close"].max() if len(lookback_20) >= 5 else False
    lookback_60 = known.tail(61)
    is_60d_high = close_today >= lookback_60["close"].max() if len(lookback_60) >= 10 else False

    # 连涨天数
    consecutive_up = 0
    for i in range(len(known) - 1, -1, -1):
        if known.iloc[i]["close"] > known.iloc[i - 1]["close"] if i > 0 else False:
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
        "ma20": round(ma20, 6),
        "ma60": round(ma60, 6),
        "ma20_slope_5d": round(ma20_slope_5d, 2),
        "distance_ma20": round(distance_ma20, 2),
        "distance_ma60": round(distance_ma60, 2),
        "trend_efficiency_20": round(trend_efficiency_20, 3),
        "volatility_20": round(volatility_20, 2),
        "one_day_dominance_5": round(one_day_dominance_5, 3),
        "positive_days_5": positive_days_5,
        "amount_persistence_3d": round(amount_persistence_3d, 2),
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


def calc_relative_profile(
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
    target_date: str,
) -> dict:
    """计算与基准同交易日对齐的相对强度和持续跑赢天数。"""
    empty_profile = {
        "excess_5d": 0.0,
        "excess_20d": 0.0,
        "excess_60d": 0.0,
        "beat_days_10": 0,
    }
    if df.empty or bench_df.empty:
        return empty_profile

    cutoff = pd.to_datetime(target_date).normalize()
    left = df.loc[df["date"] <= cutoff, ["date", "close"]].copy()
    right = bench_df.loc[bench_df["date"] <= cutoff, ["date", "close"]].copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    aligned = left.merge(right, on="date", how="inner", suffixes=("_etf", "_bench"))
    aligned = aligned.sort_values("date").dropna().drop_duplicates("date", keep="last")
    if len(aligned) < 2:
        return empty_profile

    def excess_return(days: int) -> float:
        sample = aligned.tail(days + 1)
        if len(sample) < 2:
            return 0.0
        etf_return = float(sample.iloc[-1]["close_etf"]) / float(sample.iloc[0]["close_etf"]) - 1
        bench_return = float(sample.iloc[-1]["close_bench"]) / float(sample.iloc[0]["close_bench"]) - 1
        return ((1 + etf_return) / (1 + bench_return) - 1) * 100

    daily = aligned[["close_etf", "close_bench"]].pct_change().dropna()
    return {
        "excess_5d": round(excess_return(5), 2),
        "excess_20d": round(excess_return(20), 2),
        "excess_60d": round(excess_return(60), 2),
        "beat_days_10": int(
            (daily["close_etf"].tail(10) > daily["close_bench"].tail(10)).sum()
        ),
    }


def apply_scoring_context(results: list[dict], bench_pct: float) -> list[dict]:
    """加入去重后的方向宽度，再统一计算三层评分。"""
    valid = [r for r in results if "error" not in r.get("metrics", {})]
    theme_rows = []
    for r in valid:
        metrics = r["metrics"]
        theme_rows.append({
            "major": str(r.get("category") or "其他").split("/")[0],
            "theme": str(r.get("category") or r.get("direction") or "其他"),
            "ret_20d": float(metrics.get("ret_20d") or 0),
            "excess_5d": float(metrics.get("excess_5d") or 0),
            "excess_20d": float(metrics.get("excess_20d") or 0),
        })

    theme_stats: dict[str, tuple[float, int]] = {}
    theme_confirmation: dict[tuple[str, str], tuple[float, float, bool]] = {}
    if theme_rows:
        theme_df = pd.DataFrame(theme_rows)
        deduped = (
            theme_df.groupby(["major", "theme"], as_index=False)
            [["ret_20d", "excess_5d", "excess_20d"]]
            .median()
        )
        for major, group in deduped.groupby("major"):
            positive = (group["ret_20d"] > 0) & (group["excess_20d"] > 0)
            breadth = float(positive.mean()) if len(group) >= 2 else 0.5
            confirmations = int(
                (positive & (group["excess_5d"] >= -1)).sum()
            )
            theme_stats[str(major)] = (breadth, confirmations)
        for row in deduped.itertuples(index=False):
            confirmed = (
                float(row.ret_20d) > 0
                and float(row.excess_20d) > 0
            )
            theme_confirmation[(str(row.major), str(row.theme))] = (
                float(row.ret_20d),
                float(row.excess_20d),
                confirmed,
            )

    for r in results:
        metrics = r.get("metrics", {})
        if "error" in metrics:
            continue
        major = str(r.get("category") or "其他").split("/")[0]
        theme = str(r.get("category") or r.get("direction") or "其他")
        breadth, confirmations = theme_stats.get(major, (0.5, 0))
        theme_ret20, theme_excess20, theme_confirmed = theme_confirmation.get(
            (major, theme),
            (0.0, 0.0, False),
        )
        metrics["theme_breadth"] = round(breadth, 3)
        metrics["theme_confirmations"] = confirmations
        metrics["theme_ret_20d"] = round(theme_ret20, 2)
        metrics["theme_excess_20d"] = round(theme_excess20, 2)
        metrics["theme_confirmed"] = theme_confirmed
        r["score"] = calc_auto_score(
            metrics,
            r.get("stage", ("未知", "", ""))[0],
            bench_pct,
            is_qdii=bool(r.get("is_qdii", False)),
            premium_pct=r.get("premium_pct"),
            consecutive_strong=int(r.get("strong_days") or 0),
        )
    return results


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


def mark_unpatched_stale_results(results: list[dict], expected_data_date: str) -> int:
    """禁止落后于本次A股数据截止日的旧行情混入主排名。"""
    count = 0
    expected = pd.to_datetime(expected_data_date).normalize()
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        actual_value = m.get("actual_date")
        actual = (
            pd.to_datetime(actual_value).normalize()
            if actual_value
            else pd.Timestamp.min
        )
        m["is_stale"] = bool(actual < expected)
        m["stale_days"] = max(0, (expected - actual).days)
        if actual < expected and not m.get("realtime_patched"):
            actual = m.get("actual_date", "未知")
            m["error"] = (
                f"旧行情未补全: 行情日期 {actual} < 数据截止日 {expected_data_date}"
            )
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
    """分层评分：主线强度、买点质量和风险惩罚分别计算。

    总分只用于排序，不能替代三层门禁。主线分主要由20/60日相对强度、
    持续跑赢次数、趋势效率和去重后的方向宽度组成；单日涨幅不会再通过
    多个同源指标重复放大。
    """
    if "error" in m:
        return {
            "total": 0.0, "raw_total": 0.0, "relative": 0.0, "volume": 0.0,
            "momentum": 0.0, "liquidity": 0.0, "trend": 0.0,
            "catalyst": 0.0, "risk_factor": 1.0, "risk_note": "",
            "mainline": 0.0, "timing": 0.0, "risk_penalty": 100.0,
            "mainline_pass": False, "timing_pass": False,
            "gate_reason": str(m.get("error") or "数据无效"),
        }

    pct = float(m.get("pct_chg") or 0)
    r5 = float(m.get("ret_5d") or 0)
    r20 = float(m.get("ret_20d") or 0)
    r60 = float(m.get("ret_60d") or 0)
    excess5 = float(m.get("excess_5d") or 0)
    excess20 = float(m.get("excess_20d") or 0)
    excess60 = float(m.get("excess_60d") or 0)
    beat_days_10 = float(m.get("beat_days_10") or consecutive_strong)
    efficiency = float(m.get("trend_efficiency_20") or 0)
    ma20_slope = float(m.get("ma20_slope_5d") or 0)
    distance_ma20 = float(m.get("distance_ma20") or 0)
    distance_ma60 = float(m.get("distance_ma60") or 0)
    one_day_dominance = float(m.get("one_day_dominance_5") or 0)
    positive_days_5 = float(m.get("positive_days_5") or 0)
    amount_persistence = float(
        m.get("amount_persistence_3d")
        or m.get("amount_ratio")
        or m.get("vol_ratio")
        or 1
    )
    volatility_20 = float(m.get("volatility_20") or 0)
    theme_breadth = float(m.get("theme_breadth") or 0.5)
    theme_confirmations = float(m.get("theme_confirmations") or 1)
    theme_confirmed = bool(m.get("theme_confirmed", True))
    amount_yi = float(m.get("amount_yi") or 0)

    # 主线强度 100 分：慢变量优先，避免单日脉冲重复计分。
    relative_5_score = _clip((excess5 + 2) / 10 * 15, 0, 15)
    relative_20_score = _clip((excess20 + 5) / 20 * 20, 0, 20)
    relative_60_score = _clip((excess60 + 8) / 33 * 15, 0, 15)
    persistence_score = _clip(beat_days_10 / 8 * 15, 0, 15)
    efficiency_score = _clip((efficiency - 0.12) / 0.43 * 15, 0, 15)
    structure_score = 0.0
    if distance_ma20 >= 0:
        structure_score += 4.0
    if distance_ma60 >= 0:
        structure_score += 3.0
    if ma20_slope > 0:
        structure_score += 3.0
    breadth_score = _clip((theme_breadth - 0.25) / 0.60 * 10, 0, 10)
    breadth_score += _clip(theme_confirmations / 3 * 5, 0, 5)
    mainline_score = _clip(
        relative_5_score
        + relative_20_score
        + relative_60_score
        + persistence_score
        + efficiency_score
        + structure_score
        + breadth_score,
        0,
        100,
    )

    # 买点质量 100 分：强趋势可买，但过度延伸或单日贡献过高会降分。
    if r5 <= 10:
        r5_timing = _clip((r5 + 2) / 12 * 25, 0, 25)
    else:
        r5_timing = _clip(25 - (r5 - 10) * 1.2, 4, 25)

    if -3 <= distance_ma20 <= 8:
        distance_score = _clip(25 - abs(distance_ma20 - 2.5) * 2.3, 10, 25)
    elif distance_ma20 < -3:
        distance_score = _clip(10 + distance_ma20, 0, 10)
    else:
        distance_score = _clip(18 - (distance_ma20 - 8) * 2, 0, 18)

    dominance_score = _clip((0.80 - one_day_dominance) / 0.60 * 20, 0, 20)
    positive_days_score = _clip(positive_days_5 / 4 * 15, 0, 15)
    if 0.85 <= amount_persistence <= 2.2:
        volume_score = _clip(15 - abs(amount_persistence - 1.35) * 7, 7, 15)
    elif amount_persistence < 0.85:
        volume_score = _clip(amount_persistence / 0.85 * 7, 0, 7)
    else:
        volume_score = _clip(10 - (amount_persistence - 2.2) * 2, 3, 10)
    timing_score = (
        r5_timing
        + distance_score
        + dominance_score
        + positive_days_score
        + volume_score
    )

    # 风险惩罚单列，不能由其他高分抵消。
    risk_penalty = 0.0
    risk_reasons = []
    if one_day_dominance > 0.65 and r20 < 8:
        risk_penalty += 12
        risk_reasons.append("5日涨幅过度依赖单日")
    if one_day_dominance > 0.80:
        risk_penalty += 8
        risk_reasons.append("单日脉冲占比极高")
    if r20 < 0:
        risk_penalty += 10
        risk_reasons.append("20日趋势为负")
    if r60 < 0:
        risk_penalty += 6
        risk_reasons.append("60日趋势为负")
    if distance_ma20 > 12:
        risk_penalty += min(12, (distance_ma20 - 12) * 1.2)
        risk_reasons.append("偏离MA20过远")
    if r5 > 15:
        risk_penalty += 8
        risk_reasons.append("5日涨幅过热")
    if r20 > 35:
        risk_penalty += 5
        risk_reasons.append("20日涨幅过热")
    if r60 > 50:
        risk_penalty += min(12, (r60 - 50) * 0.3)
        risk_reasons.append("60日趋势拥挤")
    if volatility_20 > 45:
        risk_penalty += min(10, (volatility_20 - 45) * 0.3)
        risk_reasons.append("波动率过高")
    if "加速见顶" in stage or "加速期⚠" in stage:
        risk_penalty += 8
        risk_reasons.append(stage)

    # 流动性: 5 分
    if amount_yi >= 5:
        liquidity_score = 5
    elif amount_yi >= 1:
        liquidity_score = 4
    elif amount_yi >= 0.2:
        liquidity_score = 2
    else:
        liquidity_score = 0

    # 实时资金流只展示，不混入历史可复算的核心技术总分。
    extra = m.get("_hithink_extra") or {}
    main_flow_yi = extra.get("main_flow_yi")
    flow_score_raw, flow_note = calc_money_flow_score(main_flow_yi)
    flow_score = flow_score_raw

    raw_total = 0.65 * mainline_score + 0.35 * timing_score - risk_penalty
    risk_factor, risk_note = premium_discount_factor(is_qdii=is_qdii, premium_pct=premium_pct)
    total = _clip(raw_total * risk_factor, 0, 100)

    established_trend = (
        r20 > 0
        and excess20 > 0
        and distance_ma20 >= -2
        and ma20_slope >= 0
    )
    strong_reversal = r20 >= 15 and beat_days_10 >= 6 and theme_breadth >= 0.55
    one_day_trap = one_day_dominance > MAX_NEW_POSITION_ONE_DAY_DOMINANCE
    mainline_pass = (
        (established_trend or strong_reversal)
        and (beat_days_10 >= 5 or theme_breadth >= 0.60)
        and theme_confirmed
        and not one_day_trap
    )
    late_extension = r60 > 60 and r20 > 20 and distance_ma20 > 6
    timing_pass = (
        timing_score >= 50
        and distance_ma20 <= 15
        and r5 <= 25
        and not late_extension
    )
    gate_reason = ""
    if one_day_trap:
        gate_reason = (
            "5日上涨过度依赖单日脉冲"
            f"（>{MAX_NEW_POSITION_ONE_DAY_DOMINANCE:.0%}）"
        )
    elif not (established_trend or strong_reversal):
        gate_reason = "20日相对趋势或MA20结构未确认"
    elif not (beat_days_10 >= 5 or theme_breadth >= 0.60):
        gate_reason = "持续跑赢次数和方向宽度均不足"
    elif not theme_confirmed:
        gate_reason = "所属子主题20日收益或相对收益未确认"
    elif late_extension:
        gate_reason = "60日趋势拥挤且价格偏离MA20"
    elif not timing_pass:
        gate_reason = "买点过热、偏离趋势或质量不足"

    return {
        "total": round(total, 1),
        "raw_total": round(raw_total, 1),
        "mainline": round(mainline_score, 1),
        "timing": round(timing_score, 1),
        "risk_penalty": round(risk_penalty, 1),
        "one_day_dominance": round(one_day_dominance, 3),
        "mainline_pass": mainline_pass,
        "timing_pass": timing_pass,
        "gate_reason": gate_reason,
        "relative": round(relative_5_score + relative_20_score + relative_60_score, 1),
        "volume": round(volume_score, 1),
        "momentum": round(relative_20_score + relative_60_score, 1),
        "trend": round(persistence_score + efficiency_score + structure_score, 1),
        "breadth": round(breadth_score, 1),
        "liquidity": round(liquidity_score, 1),
        "money_flow": round(flow_score, 1),
        "catalyst": 0.0,
        "risk_factor": round(risk_factor, 2),
        "risk_note": "；".join(filter(None, [risk_note, *risk_reasons])),
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
        return ("趋势回撤期", "持有观察，跌破趋势止损再清仓", "仅诊断")

    # 衰竭：连续走弱
    if r5 < -3 and vol < 0.8:
        return ("衰竭期", "清仓或大幅减仓", "仅诊断")
    if r5 < -2 and pct < -1:
        return ("衰弱期", "减仓观望", "仅诊断")

    # 过热判断（加入量价结构）：
    # - 缩量+减速 → 加速见顶⚠
    # - 放量+加速 → 正常加速期（趋势延续，不过早减仓）
    # - 放量+减速 → 加速期⚠（持有但不加仓）
    if r20 > 30:
        if vol < 1.0 and accel < 0:
            return ("加速见顶⚠", "逐步减仓，不追", "仅诊断")
        if vol >= 1.2 and accel >= 0:
            return ("加速期", "趋势延续，持有", "组合层计算")
        return ("加速期⚠", "持有但不加仓，设紧止损", "组合层计算")

    # 加速：5日动量>8%且加速中
    if r5 > 8 and accel > 0:
        return ("加速期", "持有，逐步提止损", "组合层计算")

    # 扩散：5日>3%且放量
    if r5 > 3 and vol > 1.2:
        return ("扩散期", "核心仓位，可加仓", "组合层计算")

    # 确认：连续强势或放量上涨
    if (r5 > 2 and vol > 1.0) or consecutive_strong >= 2:
        return ("确认期", "试探建仓/加仓", "组合层计算")
    if pct > 2 and vol > 1.2:
        return ("确认期", "试探建仓", "组合层计算")

    # 萌芽：首次放量
    if pct > 1.5 and vol > 1.5:
        return ("萌芽期", "加入观察，等待回踩", "仅诊断")

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
    code = str(r.get("etf_code") or "")
    code_tiebreaker = -int(code) if code.isdigit() else -999999
    return (
        score,
        STAGE_PRIORITY.get(stage, -99),
        m.get("pct_chg", 0) or 0,
        m.get("ret_5d", 0) or 0,
        m.get("amount_yi", 0) or 0,
        code_tiebreaker,
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


def _specificity_score(r: dict) -> float:
    direction = str(r.get("direction", ""))
    name = str(r.get("etf_name", ""))
    text = direction + name
    if r.get("industry") == "宽基":
        return -1.0
    if any(word in text for word in SPECIFIC_THEME_WORDS):
        return 2.0
    if any(word in text for word in GENERIC_DIRECTION_WORDS):
        return 0.0
    return 1.0


def _portfolio_rank_key(r: dict) -> tuple:
    m = r.get("metrics", {})
    score = float(r.get("score", {}).get("total") or 0)
    adjusted = score + _specificity_score(r)
    code = str(r.get("etf_code") or "")
    code_tiebreaker = -int(code) if code.isdigit() else -999999
    return (
        adjusted,
        STAGE_PRIORITY.get(r.get("stage", ("", "", ""))[0], -99),
        float(m.get("ret_5d") or 0),
        float(m.get("amount_yi") or 0),
        code_tiebreaker,
    )


def _portfolio_exclusion_reason(r: dict, market_state: str = "未知") -> str | None:
    m = r.get("metrics", {})
    stage = r.get("stage", ("", "", ""))[0]
    score = float(r.get("score", {}).get("total") or 0)
    if "error" in m:
        return str(m.get("error"))
    if stage not in NEW_MONEY_STAGES:
        return f"{stage}不适合作为新组合主仓"
    layers_ok, layer_reason = score_layers_pass(r.get("score", {}), market_state)
    if not layers_ok:
        return layer_reason
    if category_root(r.get("category", "")) in {"货币", "债券"}:
        return "现金/债券工具不进入进攻组合"
    if float(m.get("amount_yi") or 0) < PORTFOLIO_MIN_AMOUNT_YI:
        return f"成交额<{PORTFOLIO_MIN_AMOUNT_YI:.0f}亿，不作为主仓"
    if m.get("overheat") and float(m.get("pct_chg") or 0) >= 7:
        return "20日过热且单日高潮，不新开主仓"
    trend_ok, trend_reason = new_position_trend_gate(
        category=r.get("category", "其他"),
        market_state=market_state,
        score=score,
        ret_20d=float(m.get("ret_20d") or 0),
        ret_60d=float(m.get("ret_60d") or 0),
    )
    if not trend_ok:
        return trend_reason
    return None


def _better_theme_candidate(current: dict, challenger: dict) -> dict:
    """分数接近时，用更具体的主题 ETF 替代泛科技/泛宽基。"""
    current_score = float(current.get("score", {}).get("total") or 0)
    challenger_score = float(challenger.get("score", {}).get("total") or 0)
    if current_score - challenger_score > PORTFOLIO_THEME_TOLERANCE:
        return current
    if _specificity_score(challenger) > _specificity_score(current):
        return challenger
    return current


def pick_formal_portfolio(
    candidates: list[dict],
    target_count: int = 3,
    market_state: str = "未知",
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """
    正式组合选择器。
    它比“强度排名”更严格：流动性、同主线集中度、宽基数量和过热新开都在这里统一处理。
    """
    excluded = []
    eligible = []
    for r in candidates:
        reason = _portfolio_exclusion_reason(r, market_state)
        if reason:
            excluded.append((r, reason))
        else:
            eligible.append(r)

    ordered = sorted(eligible, key=_portfolio_rank_key, reverse=True)
    selected = []
    industry_count: dict[str, int] = {}
    cluster_count: dict[str, int] = {}
    broad_count = 0
    target_count = min(target_count, max_selection_count(market_state))
    cluster_limit = max_same_risk_cluster(market_state)

    for r in ordered:
        ind = r.get("industry", "其他")
        selected_codes = {x.get("etf_code") for x in selected}

        if ind == "宽基" and len(selected) >= 2:
            selected_non_broad_industries = [
                x.get("industry") for x in selected
                if x.get("industry") not in {"宽基", "债券", "货币"}
            ]
            # 前两只已经压在同一行业时，第三只优先拿具体主题，不用宽基凑数。
            if len(set(selected_non_broad_industries)) == 1:
                broad_score = _portfolio_rank_key(r)[0]
                alternatives = [
                    x for x in ordered
                    if x.get("etf_code") not in selected_codes
                    and x.get("industry") not in {"宽基", "债券", "货币"}
                    and industry_count.get(x.get("industry", "其他"), 0) < PORTFOLIO_MAX_SAME_INDUSTRY
                    and broad_score - _portfolio_rank_key(x)[0] <= PORTFOLIO_THEME_TOLERANCE
                ]
                if alternatives:
                    r = alternatives[0]
                    ind = r.get("industry", "其他")
        if ind == "宽基":
            if broad_count >= PORTFOLIO_MAX_BROAD:
                continue
        elif industry_count.get(ind, 0) >= PORTFOLIO_MAX_SAME_INDUSTRY:
            continue
        cluster = risk_cluster(r.get("category", "其他"), ind, r.get("etf_name", ""))
        if cluster_count.get(cluster, 0) >= cluster_limit:
            continue

        # 若同一行业已有泛主题，且新候选分数接近但更具体，替换泛主题。
        replaced = False
        for idx, current in enumerate(selected):
            if current.get("industry") != ind or ind == "宽基":
                continue
            better = _better_theme_candidate(current, r)
            if better is r:
                selected[idx] = r
                replaced = True
                break
        if replaced:
            continue

        if len(selected) >= target_count:
            continue

        selected.append(r)
        if ind == "宽基":
            broad_count += 1
        industry_count[ind] = industry_count.get(ind, 0) + 1
        cluster_count[cluster] = cluster_count.get(cluster, 0) + 1

    selected_codes = {r.get("etf_code") for r in selected}
    excluded = [
        (r, reason)
        for r, reason in excluded
        if r.get("etf_code") not in selected_codes
    ]
    return selected, excluded


def build_formal_snapshot(results: list[dict], market_state: str) -> dict:
    """Build the machine-authoritative portfolio shortlist from scanner results."""
    valid = [r for r in results if "error" not in r.get("metrics", {})]
    actionable = [
        r
        for r in valid
        if (
            r.get("stage", ("", "", ""))[0] not in EXCLUDED_STAGES
            and r.get("stage", ("", "", ""))[0] != "观察期"
        )
        or (
            r.get("stage", ("", "", ""))[0] == "观察期"
            and float(r.get("metrics", {}).get("pct_chg") or 0) > 1
        )
    ]
    pool_candidates = sorted(
        [r for r in actionable if r.get("category") != "货币"],
        key=_rank_key,
        reverse=True,
    )
    picks, _ = pick_formal_portfolio(
        pool_candidates,
        target_count=3,
        market_state=market_state,
    )
    picks = picks[:3]
    weights = allocate_instrument_weights(market_state, picks)
    candidates = []
    for rank, (row, weight) in enumerate(zip(picks, weights), 1):
        score = row.get("score") or {}
        candidates.append(
            {
                "rank": rank,
                "code": str(row.get("etf_code") or ""),
                "name": str(row.get("etf_name") or ""),
                "direction": str(row.get("direction") or ""),
                "product_type": str(row.get("product_type") or "ETF"),
                "is_qdii": bool(row.get("is_qdii", False)),
                "premium_pct": row.get("premium_pct"),
                "industry": str(row.get("industry") or "其他"),
                "stage": str(row.get("stage", ("未知", "", ""))[0]),
                "score": float(score.get("total") or 0),
                "mainline_score": float(score.get("mainline") or 0),
                "timing_score": float(score.get("timing") or 0),
                "risk_penalty": float(score.get("risk_penalty") or 0),
                "scanner_weight_pct": float(weight),
            }
        )
    exposure = round(sum(weights), 2)
    exposure_floor = float(get_position_cap(market_state).low)
    return {
        "market_state": market_state,
        "candidates": candidates,
        "cash_pct": round(100.0 - exposure, 2),
        "exposure_floor_pct": exposure_floor,
        "deployment_status": (
            "ready"
            if exposure + 0.05 >= exposure_floor
            else "blocked_below_exposure_floor"
        ),
    }


def generate_report(
    results: list,
    market_env: dict,
    bench_pct: float,
    target_date: str,
    hithink_used: bool = False,
    data_cutoff: str | None = None,
) -> str:
    """三层分析报告"""
    source_tag = " | 同花顺资金增强已启用" if hithink_used else ""

    lines = []
    lines.append(f"# ETF 三层分析报告 — {target_date}{source_tag}")
    lines.append("")
    if data_cutoff:
        lines.append(f"- 数据截止时间：{data_cutoff}")
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
    lines.append(f"| 风险预算范围 | **{me['position_cap']}** |")
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

    lines.append("| 排名 | 方向 | 类型 | 分类 | 行情日期 | 涨跌幅 | 5日动量 | 20日动量 | 量比 | 成交额(亿) | 连强 | 阶段 | 评分 | 风控 | 建议动作 | 阶段仓位提示 |")
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
    strong_picks = sorted(
        [
            r for r in actionable
            if r["stage"][0] in NEW_MONEY_STAGES and r["category"] != "货币"
        ],
        key=_rank_key,
        reverse=True,
    )
    pool_candidates = sorted(
        [r for r in actionable if r["category"] != "货币"],
        key=_rank_key,
        reverse=True,
    )
    raw_picks = strong_picks[:5]
    picks = _pick_diversified(strong_picks, target_count=3)
    market_state = market_env.get("market_state", "未知")
    formal_picks, formal_excluded = pick_formal_portfolio(
        pool_candidates,
        target_count=3,
        market_state=market_state,
    )

    if formal_picks:
        if raw_picks:
            lines.append("原始强度前三（诊断参考，非执行组合）：")
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
        lines.append("正式池内候选（唯一方向口径，仍需完成催化、买点和盈亏比检查）：")
        lines.append("")
        lines.append("| 优先级 | 方向 | 类型 | 行业 | 阶段 | 总分 | 主线 | 买点 | 风险罚分 | 建议仓位 | 池内标的 |")
        lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---|")
        formal_weights = allocate_instrument_weights(market_state, formal_picks[:3])
        for i, r in enumerate(formal_picks[:3], 1):
            product_note = r.get("product_type", "ETF")
            if r.get("is_qdii"):
                product_note = f"{product_note}/QDII"
            score = r.get("score", {})
            lines.append(
                f"| {i} | {r['direction']} | {product_note} | {r['industry']} | {r['stage'][0]} | {r['score']['total']:.1f} "
                f"| {float(score.get('mainline') or 0):.1f} | {float(score.get('timing') or 0):.1f} "
                f"| {float(score.get('risk_penalty') or 0):.1f} | {formal_weights[i - 1]:g}% "
                f"| {r['etf_name']} `{r['etf_code']}` |"
            )
        cash_weight = round(100 - sum(formal_weights), 1)
        lines.append(f"| — | 现金 | 现金 | — | — | — | — | — | — | {cash_weight:g}% | — |")
        exposure_floor = get_position_cap(market_state).low
        if sum(formal_weights) + 0.05 < exposure_floor:
            lines.append("")
            lines.append(
                f"> 部署阻断：正式候选合计仓位 {sum(formal_weights):g}% "
                f"低于{market_state}最低风险预算 {exposure_floor:g}%；"
                "不得把该结果标记为可执行组合。"
            )
        if len(formal_picks) < max_selection_count(market_state):
            lines.append("")
            lines.append("> 正式组合允许 0–3 只；未使用的风险预算保留为现金，不以低质量候选补位。")
        if formal_excluded:
            shown = 0
            lines.append("")
            lines.append("主要剔除原因：")
            for r, reason in formal_excluded:
                if shown >= 6:
                    break
                if float(r.get("score", {}).get("total") or 0) <= 0:
                    continue
                lines.append(f"- {r['direction']}：{reason}")
                shown += 1
        lines.append("")
        lines.append("去相关组合参考（诊断参考，非执行组合）：")
        lines.append("")
        lines.append("| 优先级 | 方向 | 类型 | 行业 | 阶段 | 评分 | 阶段仓位提示 | 池内标的 |")
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
        lines.append("> 原始强度和去相关组合只用于解释资金主线；周度正式 selection 只能从“正式池内候选”继续做催化、买点、折溢价风险和盈亏比检查。")
    else:
        lines.append("> 当前没有同时通过主线、买点和风险门禁的标的；正式动作是空仓等待。")

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
    parser.add_argument("--offline", action="store_true", help="只使用本地行情，禁止网络回退")
    parser.add_argument("--refresh-cache", action="store_true", help="强制刷新请求区间并合并到本地缓存")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    target_ts = pd.to_datetime(target_date).normalize()
    strategy_manifest = load_current_manifest() or {}
    manifest_ok, manifest_errors = verify_manifest(strategy_manifest)
    if not strategy_manifest or not manifest_ok:
        details = "; ".join(manifest_errors) if manifest_errors else "未找到当前策略清单"
        print(f"错误: 冻结策略清单无效，禁止生成扫描: {details}", file=sys.stderr)
        sys.exit(3)
    latest_realtime_ts = get_latest_realtime_trade_ts()
    use_hithink_realtime = bool(args.hithink and target_ts == latest_realtime_ts)
    if args.hithink and target_ts != latest_realtime_ts:
        print(
            f"  同花顺实时增强已禁用: target_date={target_date} 不是最新实时交易日"
            f" {latest_realtime_ts.strftime('%Y-%m-%d')}，避免实时数据污染历史复盘",
            file=sys.stderr,
        )

    # Layer 1: 市场环境
    print("Layer 1: 获取市场宽度...", file=sys.stderr)
    if target_ts == latest_realtime_ts:
        market_env = get_market_environment(offline=args.offline)
    else:
        market_env = get_replay_market_environment(target_date, offline=args.offline)
        print("  历史复盘模式: 不使用实时市场宽度", file=sys.stderr)
    print(f"  状态={market_env['market_state']}, 涨跌比={market_env['up_ratio']}, 仓位上限={market_env['position_cap']}", file=sys.stderr)

    etf_pool = load_etf_txt()
    if not etf_pool:
        print("错误: scripts/etf.txt 为空或不存在", file=sys.stderr)
        sys.exit(1)
    
    print(f"Layer 2: ETF 全量扫描 ({len(etf_pool)} 只，来源=scripts/etf.txt)...", file=sys.stderr)
    start_date = (pd.to_datetime(target_date) - timedelta(days=120)).strftime("%Y%m%d")
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
        refresh_cache=args.refresh_cache,
        offline=args.offline,
    ) if bench_cfg else pd.DataFrame()
    bench_m = calc_etf_metrics(bench_df, target_date)
    bench_pct = bench_m.get("pct_chg", 0.0)
    benchmark_data_date = str(bench_m.get("actual_date") or target_date)

    def analyze_one(item: tuple[str, dict]) -> dict:
        direction, cfg = item
        df = fetch_etf_hist(
            cfg["code"],
            start_date,
            end_date,
            cfg.get("type", "ETF"),
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        metrics = calc_etf_metrics(df, target_date)
        metrics.update(calc_relative_profile(df, bench_df, target_date))
        strong_days = calc_consecutive_strong(df, bench_df, target_date)
        stage = classify_stage(metrics, strong_days)
        return {
            "direction": direction,
            "etf_code": cfg["code"],
            "etf_name": cfg["name"],
            "category": cfg["cat"],
            "industry": cfg["ind"],
            "product_type": cfg.get("type", "ETF"),
            "is_qdii": cfg.get("is_qdii", False),
            "premium_pct": cfg.get("premium_pct"),
            "aliases": cfg.get("aliases", []),
            "metrics": metrics,
            "stage": stage,
            "strong_days": strong_days,
            "score": {},
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

    results.sort(key=lambda row: (str(row.get("etf_code") or ""), str(row.get("direction") or "")))
    if not bench_m.get("actual_date"):
        available_dates = sorted(
            {
                str(row.get("metrics", {}).get("actual_date"))
                for row in results
                if row.get("metrics", {}).get("actual_date")
                and "error" not in row.get("metrics", {})
            }
        )
        if available_dates:
            benchmark_data_date = available_dates[-1]
    apply_scoring_context(results, bench_pct)

    # Layer 2.5: 同花顺资金增强 + 热度查询
    valid_count = sum(1 for r in results if "error" not in r["metrics"])
    hithink_used = False
    hot_keywords = set()
    if use_hithink_realtime or (valid_count == 0 and target_ts == latest_realtime_ts):
        hot_keywords = fetch_market_sentiment()
        if hot_keywords:
            print(f"  市场热度: 热议方向 -> {', '.join(sorted(hot_keywords)[:6])}", file=sys.stderr)

    if use_hithink_realtime or (valid_count == 0 and target_ts == latest_realtime_ts):
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
            apply_scoring_context(results, bench_pct)
        elif valid_count == 0:
            print(f"错误: {target_date} 没有任何有效行情且同花顺数据也不可用", file=sys.stderr)
            print("提示: 请检查 AKShare 网络连接，或手动运行 hithink CLI 确认 API 配置", file=sys.stderr)
            sys.exit(2)
    elif valid_count == 0:
        print(f"错误: {target_date} 没有任何历史有效行情；历史日期禁止用同花顺实时数据回填", file=sys.stderr)
        sys.exit(2)

    stale_excluded = mark_unpatched_stale_results(results, benchmark_data_date)
    if stale_excluded:
        print(f"  数据新鲜度门禁: 剔除 {stale_excluded} 只未实时补全的旧行情 ETF/LOF", file=sys.stderr)

    # Layer 3: 生成报告
    out_path = PROJECT_ROOT / "codex" / "stock" / target_date / "etf_scan.md"
    os.makedirs(out_path.parent, exist_ok=True)
    mode_note = " + 同花顺资金增强" if hithink_used else ""
    data_cutoff = f"{benchmark_data_date} 15:00"
    formal_snapshot = build_formal_snapshot(
        results,
        str(market_env.get("market_state") or "未知"),
    )
    report = generate_report(
        results,
        market_env,
        bench_pct,
        target_date,
        hithink_used=hithink_used,
        data_cutoff=data_cutoff,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    display_path = f"codex/stock/{target_date}/etf_scan.md"
    print(f"  报告: {display_path} (data_source={target_date}{mode_note})", file=sys.stderr)

    # 机器可读侧车 scan.json：供走前向引擎读取（含代码+收盘价，免手动补价）
    import json as _json
    scan_rows = []
    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        stage = r.get("stage", ["", "", ""])
        scan_rows.append({
            "code": r.get("etf_code"),
            "name": r.get("direction") or r.get("etf_name"),
            "sector": r.get("category"),
            "stage": stage[0] if stage else "",
            "score": (r.get("score") or {}).get("total", 0.0),
            "amount_yi": m.get("amount_yi"),
            "pct": m.get("pct_chg"),
            "close": m.get("close"),
            "ret_5d": m.get("ret_5d"),
            "ret_20d": m.get("ret_20d"),
            "ret_60d": m.get("ret_60d"),
            "amount_ratio": m.get("amount_ratio"),
            "industry": r.get("industry"),
            "direction": r.get("direction"),
            "is_qdii": r.get("is_qdii", False),
            "actual_date": m.get("actual_date"),
            "raw_score": (r.get("score") or {}).get("raw_total", 0.0),
            "mainline_score": (r.get("score") or {}).get("mainline", 0.0),
            "timing_score": (r.get("score") or {}).get("timing", 0.0),
            "risk_penalty": (r.get("score") or {}).get("risk_penalty", 0.0),
            "one_day_dominance": (r.get("score") or {}).get(
                "one_day_dominance",
                0.0,
            ),
        })
    scan_rows.sort(
        key=lambda row: (
            -float(row.get("score") or 0),
            str(row.get("code") or ""),
        )
    )
    scan_payload = {
        "schema_version": 2,
        "date": target_date,
        "signal_date": target_date,
        "data_cutoff": data_cutoff,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "strategy_version": strategy_manifest.get("strategy_version"),
        "strategy_sha256": strategy_manifest.get("combined_sha256"),
        "market_state": formal_snapshot["market_state"],
        "formal_candidates": formal_snapshot["candidates"],
        "cash_pct": formal_snapshot["cash_pct"],
        "exposure_floor_pct": formal_snapshot["exposure_floor_pct"],
        "deployment_status": formal_snapshot["deployment_status"],
        "bench_pct": bench_pct,
        "rows": scan_rows,
    }
    with open(out_path.parent / "scan.json", "w", encoding="utf-8") as jf:
        _json.dump(scan_payload, jf, ensure_ascii=False, indent=2)
        jf.write("\n")

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
