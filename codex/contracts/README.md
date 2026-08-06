# 模型无关执行契约

目标不是让不同模型写出相同文字，而是让相同输入产生相同的标的、仓位、订单和风险动作。

## 权限边界

| 内容 | 唯一权威 | 模型权限 |
|---|---|---|
| 全池排名、正式候选、市场风险预算 | `etf_analyzer.py` | 只读 |
| 市场状态、评分门禁、集中度 | 冻结策略代码 | 只读 |
| 事件风险 | 截止时点的 `event_ledger.csv` | 可摘要，不可改分 |
| 催化与否决证据 | 非模型主体批准的 evidence packet | 可起草，不可自批 |
| 买入区间、止损、目标价 | `decision_contract.py` 固定执行公式 | 无权限 |
| 真实账户 | 用户确认的 account state | 不可猜测 |
| 否决后归一化仓位、目标组合与订单 | `decision_contract.py` | 只可解释 |
| 盘前/盘中/盘后动作 | `ops_contract.py` 允许值与硬门禁 | 只可填写事实 |

任何缺失、冲突或哈希变化都必须 fail closed：阻断新订单并报告缺口，不能由模型自由补全。

操作层同样采用白名单：盘中只有 `observe`、`emergency_override`、`blocked`。取消待买、减仓、退出和紧急处置必须带有结构化授权，且授权中的动作、代码和原因必须与操作包完全一致；盘中紧急处置只接受用户授权。初始化操作包时 `completed_checks` 故意为空，避免模板冒充已经完成检查。

## 周度决策

1. 扫描器生成 `scan.json` 和 `etf_scan.md`。
2. 证据收集器或模型可起草 `weekly_evidence.json`。
3. 用户、规则引擎或数据连接器批准证据；模型不能填写批准身份，也不能填写执行价格。
4. 用户确认真实账户并形成 `account_state.json`。
5. 编译器剔除被事件或证据否决的正式候选，只在剩余正式候选内按冻结规则重新归一化仓位；非冰点最终仓位不足60%时阻断。
6. 编译决策：

```powershell
python scripts/decision_contract.py build `
  --date YYYY-MM-DD `
  --evidence codex/stock/YYYY-MM-DD/weekly_evidence.json `
  --account-state codex/stock/account_state.json
```

7. 校验决策及全部输入哈希：

```powershell
python scripts/decision_contract.py verify codex/stock/YYYY-MM-DD/decision.json
```

输出：

- `decision.json`：唯一机器决策。
- `selection.generated.md`：由决策确定性渲染的可读报告。
- `selection_evidence.json`：兼容正式执行校验器的标准侧车。

模型手写的 `selection.md` 不再是权威。需要正式执行时，应先验证生成文件，再将其作为正式方案。

## 每日与周度操作包

```powershell
python scripts/ops_contract.py init `
  --phase preopen `
  --date YYYY-MM-DD `
  --output codex/stock/YYYY-MM-DD/preopen_packet.json

# 完成真实检查、填写事实和允许动作后
python scripts/ops_contract.py verify codex/stock/YYYY-MM-DD/preopen_packet.json
```

`--phase` 只允许 `preopen`、`intraday`、`postclose`、`weekly_review`。通过校验后会生成 `.verified.json`，未通过的包没有执行权限。周复盘还会核对策略收益、基准、超额、归因和经验升级门槛，避免一个模型因单周输赢随意修改长期规则。

## 相同结果的定义

下列字段形成 `portfolio_fingerprint`：

- 策略版本。
- 数据和新闻截止时间。
- 标的代码及固定排名。
- 目标仓位。
- 由收盘价和固定公式生成的买入区间、止损、目标和盈亏比。
- 现金比例。

叙述措辞、摘要长短和模型风格不进入组合指纹。只要冻结输入与批准状态相同，不同模型必须得到相同指纹。

## 无法消除的差异

若让不同模型各自搜索新闻，它们可能找到不同证据，因此输入本身已经不同。解决方法是先冻结统一证据快照并由非模型主体批准，再让模型解释；不能宣称在不同输入下也能得到相同结果。
