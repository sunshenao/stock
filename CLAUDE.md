# 项目协作约束

开始修改前先阅读：

1. `README.md`：项目结构、命令和目录纪律。
2. `DAILY_RUNBOOK.md`：盘前、盘后和周末的执行顺序。
3. `codex/stock_selection_logic.md`：正式策略的唯一文字规则。
4. `codex/strategy_versions/current.json`：当前冻结版本。
5. `codex/contracts/README.md`：模型权限、批准主体和失败关闭规则。

必须遵守以下约束：

- 正式标的只能来自 `scripts/etf.txt`，历史回放同时受 `scripts/etf_universe_history.csv` 约束。
- 正式组合允许0至3只，不得为了凑数降低门槛；无合格标的允许空仓。
- QDII/跨境产品不得因分类直接排除，只能按折溢价、流动性和仓位上限控制。
- 月度只提供方向背景，周度是唯一常规换仓频率，日度只处理硬退出或系统性降仓。
- 盘中审查只做风险预警，不得按盘中排名换票；人工紧急处置必须与规则策略收益分开记录。
- 历史信号只使用当时可得数据，成交使用下一交易日开盘，并计入成本和止损滑点。
- 用户观点、当前热点或旧持仓不能自动加分。
- 改动正式信号、仓位、执行、数据或标的池后必须运行测试，并创建新的策略版本；不得修改已冻结版本来拼接前向业绩。
- 历史回测、滚动切片和参数扰动都不是冻结后的真实样本外证据。
- `codex/stock/current_positions.md` 只有在用户确认实际成交后才能更新。
- `codex/stock/account_state.json` 是账户级订单的唯一结构化事实源；未确认时必须阻断订单，不得猜测。
- 模型只能起草证据摘要和解释，不得改扫描器候选、排名、仓位、执行价格或订单。
- `scan.json` 的实际行情截止日、策略哈希、正式候选和仓位是机器权威输入；目录日期和模型文字不得覆盖。
- `weekly_evidence.json` 必须由 `user`、`rules_engine` 或 `data_connector` 批准；模型不能自批。
- 正式周度结果只能由 `decision_contract.py` 生成；模型手写的 `selection.md` 不具备执行权限。
- 所有会取消、减仓、退出或盘中紧急成交的运行操作都必须携带与动作、代码和原因完全一致的非模型授权。
- 输入缺失、数据质量失败、哈希变化或授权不匹配时一律 fail closed，不得用自然语言补齐。

常用入口：

```powershell
python scripts/workflow.py --mode weekly --date YYYY-MM-DD
python scripts/decision_contract.py build --date YYYY-MM-DD --evidence codex/stock/YYYY-MM-DD/weekly_evidence.json --account-state codex/stock/account_state.json
python scripts/decision_contract.py verify codex/stock/YYYY-MM-DD/decision.json
python scripts/selection_guard.py codex/stock/YYYY-MM-DD/selection.generated.md
python scripts/ops_contract.py verify codex/stock/YYYY-MM-DD/intraday_packet.json
python scripts/backtest.py --freq weekly --start YYYY-MM-DD --end YYYY-MM-DD --offline
python scripts/robustness_validate.py --offline
python scripts/test_strategy_framework.py -v
python scripts/test_model_independence.py -v
```
