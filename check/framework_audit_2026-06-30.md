# 选股框架修复审计 — 2026-06-30

## 已修复问题

| 问题 | 风险 | 修复位置 |
|---|---|---|
| 回测用本月收益选择本月 TOP，存在未来函数 | 回测收益无效，容易误判策略有效 | `scripts/backtest.py` 重写为信号日只看历史数据 |
| 市场仓位上限口径冲突 | selection、ETF 扫描、手册互相矛盾 | 新增 `scripts/risk_rules.py`，统一为主升80-100%、震荡60-80%、退潮60%、冰点20-40%、未知60%；除冰点外现金≤40% |
| 凯利仓位被归一化放大 | 小优势被硬塞成大仓位 | `scripts/kelly_sizer.py` 改为真实止损距离 + 单笔风险预算，不再补满 |
| 专家共识规则被绕过 | 共识偏空股票仍可能重仓 | `risk_rules.py` 和 `kelly_sizer.py` 增加共识 <40 仓位上限 15% |
| 缺失技术/资金数据仍加分 | 接口失败被伪装成中性偏多 | `scripts/stock_checkup.py` 改为缺失不加分并输出 MISS |
| ETF 池加入 LOF/QDII 但未特殊处理 | LOF 接口错误、跨境溢价风险被忽略 | `scripts/etf_analyzer.py` 区分 ETF/LOF/QDII 并对跨境溢价未知降权 |
| ETF 扫描空表仍算成功 | 后续可能基于空数据生成组合 | `scripts/etf_analyzer.py` 有效行情为 0 时返回失败码；`daily_workflow.py` 遇失败即停止 |
| 专家辩论输出日期硬编码 | 每天结果写错目录，多个股票互相覆盖 | `scripts/expert_debate.py` 支持 `--date`，输出 `expert_debate_<代码>.md` |
| 数据源说明不一致 | 候选池仍引用旧 `etf_universe.md` 作为来源 | `candidate_pool.md` 改为 ETF/LOF 必须来自 `scripts/etf.txt` 和当日 `etf_scan.md` |
| 当前持仓和 selection 仓位矛盾 | 18%、25%、22%、保留30% 同时存在 | `current_positions.md` 和 `2026-06-29/selection.md` 按新风控折算并标注待复核 |
| ETF 可能被擅自新增 | 池外 ETF 未经用户确认进入扫描或组合 | `stock_selection_logic.md`、`candidate_pool.md`、`current_positions.md`、`CLAUDE.md` 增加禁止自动新增 ETF 规则；个股可从全 A 股按热门 ETF 主线挖掘 |

## 当前强制规则

1. 每次选股前必须先跑 `python scripts/daily_workflow.py --date YYYY-MM-DD`。
2. ETF 扫描失败、空表、有效行情为 0 时，不得生成 selection。
3. 每日最多选 3 个标的；ETF/LOF 必须来自 `scripts/etf.txt`，不得自动新增 ETF。
4. 个股可以从全 A 股选择，必须先由热门 ETF/板块验证主线，再证明个股与主线的产业链或催化关联。
5. 池外 ETF 只能作为主线验证信息汇报给用户确认，不得写入 ETF 池、持仓台账或 selection。
6. QDII/跨境 ETF 必须检查折溢价；溢价未知先降权，溢价 >5% 原则上不新开仓。
7. 凯利自然仓位只做起点；除冰点外必须用合格标的补足到至少60%，但不得突破单票/ETF硬上限。
8. 专家共识 <40 分的股票仓位 ≤15%；若同时技术追高或主线无效，不开新仓。

## 回归检查命令

```powershell
python -m py_compile scripts\risk_rules.py scripts\etf_analyzer.py scripts\kelly_sizer.py scripts\stock_checkup.py scripts\expert_debate.py scripts\daily_workflow.py scripts\backtest.py
python scripts\kelly_sizer.py
python scripts\backtest.py --start 2026-05-01 --end 2026-06-29 --top 3 --max-etfs 10
```
