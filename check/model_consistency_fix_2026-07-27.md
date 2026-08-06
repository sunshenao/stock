# 2026-07-27 周度结果一致性修复

## 结论

Claude 手写的 `selection.md` 与此前 Codex 自然语言答案出现差异，根因不是同一算法合理地产生了两个组合，而是项目同时存在旧的手写方案入口和新的机器决策契约。旧 `workflow.py` 仍读取 `selection.md`，与文档规定的 `decision.json -> selection.generated.md` 相冲突。

修复后，模型只允许起草证据。唯一组合口径是：

`scan.json.formal_candidates -> weekly_evidence.json -> decision.json -> selection.generated.md`

## 已修复

1. `workflow.py --mode weekly` 直接调用 `decision_contract.py`，不再认可手写 `selection.md`。
2. `selection_guard.py` 必须找到同目录 `decision.json`，验证输入哈希，并逐字复算 `selection.generated.md`。
3. `scan.json` schema升级为2，记录真实行情截止日、策略版本/哈希、市场状态、正式候选和现金比例。
4. 决策编译器从 `scan.json.formal_candidates` 读取候选，报告只能交叉校验，不能覆盖机器候选。
5. 批次目录日期不再冒充行情截止日。2026-07-27批次使用的实际行情截止日为2026-07-24 15:00，计划执行日为2026-07-27开盘。
6. 未来/当天批次的默认新闻截止时间不再推进到当日23:59，只能使用真实运行时刻。
7. 并列评分增加六位代码稳定排序；1线程和8线程扫描的规范哈希一致。
8. `decision_contract.py` 和 `ops_contract.py` 已纳入冻结策略哈希。

## 本次复跑

- 策略版本：2.4.1
- 数据截止：2026-07-24 15:00
- 市场状态：退潮末期
- 机器正式候选：豆粕ETF华夏 `159985`，24%
- 扫描器现金：76%
- 事件风险（截至2026-07-26 23:59）：黄色
- 正式状态：blocked

阻断原因是 Claude 侧车仍为旧的 evidence schema 1，缺少非模型批准；账户状态也未由用户确认。因此当前只有机器候选，没有可执行订单。

## 仓位政策冲突

2.4.0起采用“质量优先、不为资金利用率降低门槛”，允许非冰点时现金超过40%。这与更早提出的“除冰点外现金不得超过40%”冲突。本次没有通过塞入低质量候选或放松QDII上限来掩盖冲突，因为2.4.0诊断显示强制补位会增加无效换手并降低历史收益。

当前2.4.1继续沿用2.4.0的质量优先政策，所以76%现金是策略输出，不是计算错误。若恢复现金硬上限，必须建立新的候选策略版本，明确第二/第三席位的降级规则并重新做训练/验证切分，不能由模型临时补票。

## 验证

- `python scripts/test_strategy_framework.py -v`：54项通过。
- `python scripts/test_model_independence.py -v`：12项通过。
- 1线程与8线程扫描规范哈希：`ee3da1a22752bce99bdd8997cd36372d86ac420d12e37d7b974d7a9e67edbe25`。
- 手写 `codex/stock/2026-07-27/selection.md` 已被正式校验器拒绝。
