---
name: feedback-frequency-and-scan
description: 周度正式选股必须全池扫描，日度只做风险复核
metadata:
  type: feedback
---

正式推荐采用周度频率。每次周度选股前必须运行 `scripts/workflow.py --mode weekly`，完整扫描 `scripts/etf.txt`，不能只盯旧持仓、熟悉标的或用户提到的方向。

月度只更新产业方向和催化背景；日度只检查止损、催化证伪、冰点和大簇崩盘。日度排名变化不得生成新的正式组合。

固定的是ETF池，不是关注名单。池内缺少重要方向时只报告缺口，由用户决定是否修改 `etf.txt`。
