---
name: prediction-to-file
description: 月周日三类结论写入不同文件
metadata:
  type: feedback
---

结果必须写入对应日期目录：

1. 月度方向：`monthly_review.md`
2. 周度正式组合：`selection.md`，并通过 `selection_guard.py`
3. 日度风险复核：`risk_review.md`
4. 用户确认真实成交后更新 `current_positions.md`

不要把日度风险复核写成新的选股方案，也不要仅在对话中保留正式周度结论。
