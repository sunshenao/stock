# A股ETF周度轮动框架

当前正式策略版本为 **2.5.7**。框架以周度选取ETF/LOF为核心，月度信息只提供方向背景，日度流程只处理止损、事件证伪和系统性降仓。

日常执行时间、命令与允许动作见 `DAILY_RUNBOOK.md`。

## 唯一规则源

- 标的池：`scripts/etf.txt`
- 历史可用池：`scripts/etf_universe_history.csv`
- 评分：`scripts/etf_analyzer.py`
- 仓位与门禁：`scripts/risk_rules.py`
- 完整方法：`codex/stock_selection_logic.md`
- 当前冻结版本：`codex/strategy_versions/current.json`
- 用户确认的结构化账户：`codex/stock/account_state.json`
- 模型权限与机器契约：`codex/contracts/README.md`

`current_positions.md` 只作为账户的可读镜像。其他报告只能解释规则，不能另行定义参数、改排名、改仓位或创建订单。

## 框架结构

| 层级 | 主要文件 | 职责 |
|---|---|---|
| 数据 | `market_data_cache.py`、`prefetch_market_data.py`、`universe_history.py` | 点时标的池、日线缓存和数据质量 |
| 信号 | `etf_analyzer.py`、`market_state.py` | 主线、买点、风险及市场状态 |
| 组合 | `risk_rules.py` | 0至3只、风险预算、集中度和QDII上限 |
| 决策契约 | `decision_contract.py`、`codex/contracts/` | 冻结输入、证据批准、确定性执行价格、目标组合及订单 |
| 执行 | `DAILY_RUNBOOK.md`、`execution_model.py`、`daily_risk.py`、`event_risk.py` | 盘前/盘中/盘后审查、次日开盘、成本、止损和事件风险 |
| 工作流 | `workflow.py`、`selection_guard.py`、`ops_contract.py` | 月/周/日输出、阶段动作白名单和正式方案校验 |
| 回测 | `backtest.py`、`backtest_current.py` | 周度主回测与日度风险回放 |
| 验证 | `test_strategy_framework.py`、`walkforward_validate.py`、`factor_ic_diagnose.py`、`robustness_validate.py` | 回归、时序、因子衰减和参数邻域检查 |
| 版本 | `strategy_version.py`、`codex/strategy_versions/` | 规则文件哈希、冻结日期和前向起点 |

数据流：

```text
固定池与点时历史
  -> 本地行情缓存
  -> ETF三层评分
  -> 市场状态与组合门禁
  -> 非模型主体批准证据
  -> 决策编译器生成周度方案与组合指纹
  -> selection_guard校验
  -> 盘前操作契约校验
  -> 用户确认成交与账户状态
  -> 盘中/盘后风险操作契约
```

## 当前核心约束

- 正式组合允许0至3只；非冰点风险预算最低60%，冰点允许0仓位。单只境内ETF/LOF最高可使用100%仓位，但必须先通过全部质量门禁。
- 排名权重按实际获批候选归一化；事件或证据否决后只能在原正式候选内重算，不允许模型临时补票。
- 非冰点仓位下限用于周度目标；周中硬止损可以降到下限以下，日度流程不为补仓而换票。
- QDII不因分类被排除，使用折溢价降权和15%/25%仓位上限控制风险。
- 新仓折溢价前原始总分至少65，5日正收益的最大单日贡献不超过60%。
- 健康持仓前10个交易日不因普通周度排名变化被替换。
- 止损、主线失效、风险惩罚和集中度约束始终可以覆盖最短持有保护。
- 信号使用当时可得数据，按下一交易日开盘执行；默认单边成本8bp，并检查15bp和25bp。
- `scan.json` 固化真实行情截止日、策略哈希和机器候选；模型手写方案不能替代确定性决策。

详细口径以 `codex/stock_selection_logic.md` 和代码常量为准。

## 常用命令

```powershell
# 预热并检查固定池行情
python scripts/prefetch_market_data.py --days 370

# 周度历史回放
python scripts/backtest.py --freq weekly --start YYYY-MM-DD --end YYYY-MM-DD --offline

# 月度敏感性对照
python scripts/backtest.py --freq monthly --start YYYY-MM-DD --end YYYY-MM-DD --offline

# 参数邻域检查
python scripts/robustness_validate.py --offline

# 回归测试
python scripts/test_strategy_framework.py -v
python scripts/test_model_independence.py -v

# 编译并验证唯一周度机器决策
python scripts/decision_contract.py build --date YYYY-MM-DD --evidence codex/stock/YYYY-MM-DD/weekly_evidence.json --account-state codex/stock/account_state.json
python scripts/decision_contract.py verify codex/stock/YYYY-MM-DD/decision.json

# 初始化和验证某个运行阶段
python scripts/ops_contract.py init --phase intraday --date YYYY-MM-DD --output codex/stock/YYYY-MM-DD/intraday_packet.json
python scripts/ops_contract.py verify codex/stock/YYYY-MM-DD/intraday_packet.json

# 月度、周度、日度工作流
python scripts/workflow.py --mode monthly --date YYYY-MM-DD
python scripts/workflow.py --mode weekly --date YYYY-MM-DD
python scripts/workflow.py --mode daily --date YYYY-MM-DD
```

## 目录纪律

- `codex/stock/YYYY-MM-DD/`：真实时间顺序下的扫描、选择和风险记录。
- `codex/contracts/`：模型权限、输入模式和操作契约模板。
- `codex/templates/`：盘中审查和周度复盘的固定模板。
- `codex/stock/.cache/`：可再生行情缓存，不进入版本控制。
- `codex/strategy_versions/`：策略版本清单，保留历史版本。
- `check/`：只保留当前正式回测、稳健性报告和必要审计记录。
- 临时候选回测、消融明细、标准输出日志和 `__pycache__` 不长期保留。

## 验证边界

历史回放只能用于排错和否决脆弱规则。2026年5月以后的数据已经参与2.4.0诊断，不再属于独立样本外；真正前向记录从2026-07-27开始，核心参数至少冻结26周。当前历史ETF池质量为 `partial`，首次快照以前仍可能存在幸存者偏差。
