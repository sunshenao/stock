# ETF策略每日执行手册

适用版本：2.5.7  
交易市场：A股ETF/LOF  
原则：盘前只复核，盘后才更新信号；周度是唯一常规换仓频率。

执行权限原则：模型只整理事实和解释，不能自行改变候选、仓位、价格或订单。真正可执行的周度结果以 `decision.json` 为准，每个运行阶段以通过校验的 `*.verified.json` 操作包为准；自然语言报告本身没有交易权限。

## 一、每天必须执行的三次审查

### 1. 开盘前：08:40–09:15

目的：确认隔夜是否出现足以取消原计划或触发风险处置的新信息，不使用当天尚未形成的行情重新选票。

#### 必做输入

1. `codex/stock/account_state.json`：用户已确认的真实账户；`confirmed` 不是 `true` 时不得生成账户级订单。
2. 最近一份通过哈希校验的 `decision.json`：尚待执行或正在持有的唯一机器方案。
3. 上一交易日的 `risk_review.md`：已经确定的止损或降仓指令。
4. `codex/stock/event_ledger.csv`：截至当前时点可核验的隔夜事件。

先把经过来源核验的隔夜事件写入事件台账，再运行：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
$cutoff = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
python scripts/event_risk.py --date $cutoff --output "codex/stock/$tradeDate/preopen_event_risk.md"
python scripts/ops_contract.py init --phase preopen --date $tradeDate --output "codex/stock/$tradeDate/preopen_packet.json"
```

完成实际检查并填写 `completed_checks`、事实和动作后，必须运行：

```powershell
python scripts/ops_contract.py verify "codex/stock/$tradeDate/preopen_packet.json"
```

只有 `execute_existing_plan` 可以直接引用已验证且订单状态为 `ready` 的 `decision.json`。取消待买、减仓或退出必须有用户、规则引擎或数据连接器的结构化授权，授权中的动作、代码和原因必须逐项一致。

#### 盘前必须回答

- 今天是否为交易日，标的是否停牌或存在明显不可交易风险。
- 是否有上一交易日盘后已经确认、应在今天开盘执行的退出或降仓。
- 周度计划中的新仓是否仍有效，催化是否被证伪。
- 国际事件风险是正常、黄色还是红色，是否命中持仓或待买风险簇。
- QDII折溢价是否可核验：绝对折溢价低于2%为正常，2%至5%进入观察并降权，达到5%原则上不新开；无法核验时仓位上限15%。
- 实际账户现金、持仓和待执行订单是否与记录一致。

#### 盘前允许的动作

- 执行上一份合格周度方案中明确安排在今天开盘的订单。
- 执行上一交易日盘后已经触发的止损、催化证伪或系统性降仓。
- 因停牌、重大官方硬事件、全球红色风险或QDII高溢价取消相关新开仓。

#### 盘前禁止的动作

- 根据隔夜标题临时买入一个周度方案外的热门ETF。
- 使用集合竞价涨跌幅重新排名并换票。
- 因某只ETF昨天跌出前三就卖出健康持仓。
- 在没有用户确认成交时修改 `current_positions.md`。

若没有待执行订单和新增硬风险，盘前结论应明确写为：**维持周度方案，今日不开新决策单。**

### 2. 盘中审查：14:00–14:15

目的：发现持仓和市场的异常风险，不利用盘中噪声重新选票。

盘中审查使用 `codex/templates/intraday_review.md`，输出到：

```text
codex/stock/YYYY-MM-DD/intraday_review.md
```

同时生成机器操作包：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
python scripts/ops_contract.py init --phase intraday --date $tradeDate --output "codex/stock/$tradeDate/intraday_packet.json"
# 完成真实检查后填写，再校验
python scripts/ops_contract.py verify "codex/stock/$tradeDate/intraday_packet.json"
```

#### 盘中必须检查

- 真实持仓相对开盘价、昨收和预设止损的位置，但不把尚未收盘的普通跌破当成最终信号。
- 同一风险簇是否有至少3个独立细分方向同步急跌。
- 全市场是否出现流动性异常、指数快速下跌、涨跌停失衡或交易中断。
- 持仓是否停牌、临时公告、无法成交、异常放量或价格偏离净值。
- QDII实时折溢价是否进入2%至5%观察区或达到5%高风险区。
- 是否出现经过官方或两个独立来源确认的系统性硬事件。

#### 盘中处理规则

- 默认动作是“记录并等待收盘确认”，不调整周度组合。
- 盘中TOP排名、热点异动、单一新闻标题和普通浮亏都不能产生新订单。
- 普通结构/ATR/移动止损仍在收盘后确认，最早下一交易日开盘执行。
- 只有交易中断、产品净值/价格严重异常、重大官方系统性事件或账户不可承受风险，才允许人工紧急处置。
- 所有盘中紧急成交必须标记为“策略外人工覆盖”，单独记录时间、价格、原因和授权，不得混入规则策略收益冒充模型表现。
- `emergency_override` 只接受用户结构化授权；模型、新闻摘要或普通数据源不能授权盘中成交。

若没有异常，盘中结论固定为：**无硬风险升级，等待收盘数据，不调整周度组合。**

### 3. 收盘后：15:20–16:30

目的：补齐完整收盘数据，生成当天唯一正式的日度风险结论。

第一步，更新并检查全池行情：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
python scripts/prefetch_market_data.py --days 370 --end $tradeDate --workers 4
```

检查 `codex/stock/.cache/market_data_v2/quality_report.md`：

- 最新日期必须覆盖当天；数据源尚未更新时，不得把缺失数据解释成空仓信号。
- OHLC、重复日期和核心字段检查不得出现未解释异常。
- 若数据源延迟，可在18:00后重跑；在数据完整前沿用上一份有效结论。

第二步，更新当天事件台账后运行日度风险流程：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
python scripts/workflow.py --mode daily --date $tradeDate --news-cutoff "${tradeDate}T15:30:00"
python scripts/ops_contract.py init --phase postclose --date $tradeDate --output "codex/stock/$tradeDate/postclose_packet.json"
```

输出位于：

```text
codex/stock/YYYY-MM-DD/scan.json
codex/stock/YYYY-MM-DD/etf_scan.md
codex/stock/YYYY-MM-DD/event_risk.md
codex/stock/YYYY-MM-DD/risk_review.md
```

#### 盘后必须回答

- 持仓是否触发结构、ATR或移动止损。
- 催化和投资逻辑是否被公告或事实证伪。
- 市场是否进入冰点或发生至少3个独立方向同步下跌的大簇风险。
- 持仓是否出现流动性、折溢价或不可交易风险。
- 是否需要在下一交易日开盘退出或降仓。

没有硬触发时，`risk_review.md` 必须写明：**延续周度组合，不因日度排名变化换票。**

盘后产生的是“下一交易日订单”，不得假设按当天收盘价成交。

填写盘后检查和动作后必须校验：

```powershell
python scripts/ops_contract.py verify "codex/stock/$tradeDate/postclose_packet.json"
```

任何减仓或退出都必须由非模型主体明确授权；数据不完整时动作应为 `blocked`，不能让模型把“缺数据”解释为卖出或买入信号。

## 二、开盘执行与成交记录

### 09:25–09:35

- 只处理盘前确认后的订单。
- 常规订单按开盘成交模型执行；跳空越过止损价时按真实可成交价格处理，不能使用理想止损价。
- 不追逐开盘后突然拉升的计划外标的。

账户采用用户在2026-07-27确认的“正式推荐视为成交”模式。每次生成通过全部校验的正式可执行推荐后，必须立即按目标仓位更新结构化事实源；风险否决、观察名单和阻断结果不计为成交：

```text
codex/stock/account_state.json
```

周度工作流会在决策契约和执行校验全部通过后自动写入，按决策参考收盘价视为成交，并记录推荐批次、目标仓位、成交时间和价格来源。`current_positions.md` 仅作为兼容性可读镜像，日度风控优先读取结构化账户。用户提供券商真实成交时，以真实成交覆盖假设值。

## 三、周五或本周最后一个交易日

完成日度流程后，先使用 `codex/templates/weekly_review.md` 生成：

```text
codex/stock/YYYY-MM-DD/weekly_review.md
```

同时从结构化模板生成 `weekly_review_packet.json`，填入与报告一致的指标、归因、问题和教训，然后验证：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
python scripts/ops_contract.py init --phase weekly_review --date $tradeDate --output "codex/stock/$tradeDate/weekly_review_packet.json"
python scripts/ops_contract.py verify "codex/stock/$tradeDate/weekly_review_packet.json"
```

周复盘必须包含：

1. 本周策略收益、沪深300及其他适用基准收益、超额收益、最大回撤、平均仓位、换手和成本。
2. 收益来源：市场暴露、方向/标的选择、仓位时机、交易成本、止损与事件处置；无法精确分解时保留残差，不伪造精确归因。
3. 每个持仓的收益贡献，以及最大盈利和最大亏损来源。
4. 周度计划与真实成交的差异：成交价、滑点、未成交、漏执行和人工覆盖。
5. 问题分类：正常波动、数据问题、执行错误、流程违规或模型弱点。
6. 本周做对、做错、纯属运气各是什么，并为每项给出证据。
7. 下周最多3项可执行改进，但不得直接修改冻结参数。

经验沉淀纪律：

- 单周盈亏只写入 `weekly_review.md`，不能直接升级为永久规则。
- 同类问题重复出现至少2次，或单次属于未来函数、越权交易、数据污染等严重错误，才写入 `codex/lessons_learned.md`。
- 新经验必须写清适用条件、反例和可验证动作，不能写成“下次一定买涨得最好的”一类结果导向规则。
- 涉及评分、仓位、止损、持有期或标的池的修改必须另建候选版本，经过训练/验证切分和参数邻域检查后才能冻结。

完成周复盘后，再编译周度正式决策：

1. 使用当天完整 `scan.json`、`etf_scan.md`、`event_risk.md` 和真实持仓。
2. 读取本周 `weekly_review.md` 和 `codex/lessons_learned.md`，只纠正执行与流程错误，不因单周结果临时改参数。
3. 根据扫描器的全部正式候选起草 `weekly_evidence.json`；每只必须明确批准或否决，不能漏项。
4. 证据必须由 `user`、`rules_engine` 或 `data_connector` 批准。模型可以写论述，但不能自批，也不能填写买入区间、止损、目标价或仓位。
5. `decision_contract.py` 只保留批准候选；否决后只在剩余的原扫描正式候选中重新归一化风险预算，禁止加入弱候选或模型临时替代品。非冰点获批仓位不足60%时决策阻断；冰点允许0仓位。执行价格由扫描收盘价和固定公式生成。
6. 编译并运行三重校验：

```powershell
$tradeDate = Get-Date -Format "yyyy-MM-dd"
python scripts/workflow.py --mode weekly --date $tradeDate --reuse-scan --news-cutoff "${tradeDate}T15:30:00" --evidence "codex/stock/$tradeDate/weekly_evidence.json" --account-state "codex/stock/account_state.json"
python scripts/decision_contract.py verify "codex/stock/$tradeDate/decision.json"
python scripts/selection_guard.py "codex/stock/$tradeDate/selection.generated.md"
```

只有三项校验都通过且 `orders_status=ready` 的 `decision.json` 才可在下一交易日开盘执行。`selection.generated.md` 是其确定性可读版本，手写 `selection.md` 不具备执行权限。周末出现重大事件时，周一盘前只做事件复核和取消/降级，不重新用未来行情回填周五信号。

`$tradeDate` 是运行批次目录，不等于行情截止日。`scan.json.data_cutoff` 必须来自基准实际最后交易日；当 `$tradeDate` 是今天或未来日期且未显式传入 `--news-cutoff` 时，工作流只能使用真实运行时刻，不能默认到未来当日23:59。

## 四、每月最后一个交易日

周度和日度流程照常执行，另做一次未来1至3个月的方向复核：

- 产业景气、政策落地、订单/价格/库存和盈利预期。
- 尚未发生的催化及其日期。
- 估值、拥挤、产能和需求不及预期等反证。

月度结论写入：

```text
codex/stock/YYYY-MM-DD/monthly_review.md
```

月度报告只提供方向背景，不直接产生交易订单。

## 五、异常情况

| 情况 | 处理 |
|---|---|
| 数据未更新或质量报告失败 | 沿用上一份有效信号，不新开仓，数据完整后重跑 |
| 网络或资讯源不可用 | 标记覆盖不足，不把“没有搜到”解释为“没有风险” |
| 休市日 | 不运行行情和选股；重大事件可更新台账，但不生成交易订单 |
| 盘中普通波动 | 不重新选票；只执行已经定义的硬止损和风险规则 |
| 停牌、无法成交、涨跌停封单 | 记录未成交，下一可交易时点重新评估，不伪造成交 |
| 实际持仓记录不确定 | 停止生成账户级订单，先由用户确认真实持仓 |
| 操作包授权缺失或与动作不一致 | 阻断动作，不能由模型补写授权身份 |
| 决策输入哈希变化 | 原决策立即失效，重新冻结输入并编译 |

## 六、每日最简清单

```text
盘前：
[ ] 隔夜事件已核验并写入台账
[ ] 已运行 preopen_event_risk
[ ] account_state 已由用户确认，已核对待执行订单、停牌和QDII折溢价
[ ] preopen_packet 已通过校验

盘中：
[ ] 14:00已检查持仓异常、大簇风险、市场流动性和QDII折溢价
[ ] 未因盘中排名或普通波动重新选票
[ ] intraday_packet 已通过校验；紧急成交具有用户结构化授权

盘后：
[ ] 全池行情已更新且质量报告通过
[ ] 已运行 daily workflow
[ ] 已检查止损、证伪、冰点、大簇风险和流动性
[ ] postclose_packet 已通过校验

周末追加：
[ ] weekly_review.md 已完成收益归因和执行复盘
[ ] weekly_review_packet 已对账并通过经验升级门槛
[ ] weekly_evidence 已由非模型主体批准且覆盖全部正式候选
[ ] decision.json、selection.generated.md 已确定性生成
[ ] decision verify 与 selection_guard 均已通过
[ ] 周一开盘前仍需做隔夜事件复核
```
