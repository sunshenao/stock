"""走前向选股引擎 v2（真数据回放，读 scan.json 侧车）。

数据源全部真实：
- 榜单+收盘价：codex/stock/<date>/scan.json（由 etf_analyzer.py 生成，含代码/收盘价/评分）
策略规则写成可编辑函数，改规则→重跑→出数。用于验证是否稳健跑赢旧策略(-3.58%)。

用法:
  python scripts/walkforward.py                # 默认全窗口 6/01-7/10 明细
  python scripts/walkforward.py --sweep        # 参数扫描(样本外 6/01-6/26 + 样本内 6/29-7/10)
"""
import re, os, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK = os.path.join(ROOT, 'codex', 'stock')

def all_dates():
    ds = []
    for p in sorted(glob.glob(os.path.join(STOCK, '2026-*'))):
        d = os.path.basename(p)
        if os.path.exists(os.path.join(p, 'scan.json')):
            ds.append(d)
    return ds

# 加载所有 scan.json
SCANS = {}   # date -> {'bench':float,'rows':[...],'by_code':{code:row}}
PRICE = {}   # code -> {date: close}
def load():
    for d in all_dates():
        with open(os.path.join(STOCK, d, 'scan.json'), encoding='utf-8') as f:
            data = json.load(f)
        rows = [r for r in data['rows'] if r.get('code')]
        SCANS[d] = {'bench': data.get('bench_pct', 0.0), 'rows': rows,
                    'by_code': {r['code']: r for r in rows}}
        for r in rows:
            if r.get('close') is not None:
                PRICE.setdefault(r['code'], {})[d] = r['close']
load()

def ret(code, prev, cur):
    p = PRICE.get(code, {})
    if prev in p and cur in p and p[prev]:
        return (p[cur] - p[prev]) / p[prev] * 100.0
    return None

# ============================================================
# 策略规则（可编辑区）
# ============================================================
LIQ_MIN = 1.0
EXIT_TOPN = 4
EXIT_DAYS = 1
DD_BREAK = 8.0
DD_RELEASE = 3.0
COST_BPS = 8.0    # 单边成本(基点): 佣金~2-3bp + 滑点~5bp, 0=不计成本
BOND_SECTORS = ('债券', '货币')
BAD_STAGES = ('观察期', '萌芽期', '衰弱期', '弱势期', '休眠期', '衰竭期')

def eligible(c):
    if not c.get('code'): return False
    if c.get('is_qdii'): return False
    if c.get('sector') in BOND_SECTORS: return False
    st = c.get('stage', '')
    if st in BAD_STAGES: return False
    if '加速' in st: return False
    if '⚠' in st: return False
    if (c.get('amount_yi') or 0) < LIQ_MIN: return False
    return True

def pick_diverse(rows, n, exclude_sectors):
    out, used = [], set(exclude_sectors)
    for c in rows:
        if len(out) >= n: break
        if not eligible(c): continue
        if c['sector'] in used: continue
        out.append(c); used.add(c['sector'])
    return out

def _run(dates, verbose=True):
    holdings = {}
    nav, peak, maxdd, trades = 1.0, 1.0, 0.0, 0
    breaker = False
    log = []
    prev = None
    prev_w = {}
    for i, date in enumerate(dates):
        sc = SCANS[date]
        rows = sorted(sc['rows'], key=lambda x: -(x['score'] or 0))
        topn = set(c['code'] for c in rows[:EXIT_TOPN] if c.get('code'))

        # 当日P&L
        dret = 0.0
        if prev is not None:
            for code, h in holdings.items():
                r = ret(code, prev, date)
                if r is None: r = 0.0
                dret += h['w'] / 100.0 * r
        nav *= (1 + dret / 100.0)
        peak = max(peak, nav)
        dd = (peak - nav) / peak * 100
        maxdd = max(maxdd, dd)

        # 走弱计数
        for code, h in holdings.items():
            h['weak'] = 0 if code in topn else h.get('weak', 0) + 1

        # 熔断状态机
        if dd > DD_BREAK: breaker = True
        if breaker and dd < DD_RELEASE: breaker = False

        acts = []
        if breaker and holdings:
            best = max(holdings, key=lambda c: PRICE.get(c, {}).get(date, -9))
            for code in list(holdings):
                if code != best:
                    holdings.pop(code); trades += 1; acts.append('熔断砍'+code)
        else:
            for code in list(holdings):
                if holdings[code].get('weak', 0) >= EXIT_DAYS:
                    holdings.pop(code); trades += 1; acts.append('走弱换'+code)

        # 建仓/补仓(末日不建)
        if not breaker and i < len(dates) - 1:
            held_sec = set(h['sector'] for h in holdings.values())
            need = 2 - len(holdings)
            if need > 0:
                for p in pick_diverse(rows, need, held_sec):
                    holdings[p['code']] = {'w': 0, 'sector': p['sector'], 'weak': 0}
                    trades += 1; acts.append('建'+p['code'])
            codes = list(holdings)
            if len(codes) == 2:
                scr = {c: sc['by_code'].get(c, {}).get('score', 0) for c in codes}
                codes.sort(key=lambda c: -scr[c])
                holdings[codes[0]]['w'] = 60; holdings[codes[1]]['w'] = 40
            elif len(codes) == 1:
                holdings[codes[0]]['w'] = 60

        log.append((date, sc['bench'], dret, nav, dd, list((c, h['w']) for c, h in holdings.items()), acts))
        # 换手成本: 与昨日权重比,变动的部分按单边成本扣
        cur_w = {c: h['w'] for c, h in holdings.items()}
        turnover = 0.0
        for c in set(list(cur_w) + list(prev_w)):
            turnover += abs(cur_w.get(c, 0) - prev_w.get(c, 0))
        cost = turnover / 100.0 * (COST_BPS / 10000.0)
        nav *= (1 - cost)
        if log:
            log[-1] = log[-1][:3] + (nav,) + log[-1][4:]
        prev_w = cur_w
        prev = date

    if verbose:
        print('='*100)
        print('走前向 v2 | 窗口 %s→%s (%d日) | TOPN=%d DAYS=%d DDbrk=%.0f' %
              (dates[0], dates[-1], len(dates), EXIT_TOPN, EXIT_DAYS, DD_BREAK))
        print('='*100)
        print('%-11s %7s %8s %9s %6s  %s' % ('日期', '沪深300', '当日', '累计', 'DD', '持仓/动作'))
        print('-'*100)
        for date, bench, dr, nv, dd, hold, acts in log:
            hs = ', '.join('%s@%d' % (c, w) for c, w in hold) or '空仓'
            ac = (' | ' + '; '.join(acts)) if acts else ''
            print('%-11s %+6.2f%% %+7.2f%% %+8.2f%% %5.1f%%  %s%s' % (date, bench, dr, (nv-1)*100, dd, hs, ac))
        print('-'*100)
        print('最终 %+.2f%% | 最大回撤 %.2f%% | 交易 %d | 旧策略 -3.58%%' % ((nav-1)*100, maxdd, trades))
        print('='*100)
    return (nav-1)*100, maxdd, trades

def run(verbose=True):
    return _run(all_dates(), verbose)

# ============================================================
# 趋势跟踪策略: 按真实动量排名(无透支惩罚) 集中押最强 移动止损
# ============================================================
def _mom(code, date, lookback, dseq, dpos):
    """从PRICE算 code 在 date 的 lookback 日动量%。dseq=全交易日列表, dpos=date索引。"""
    p = PRICE.get(code, {})
    j = dpos - lookback
    if j < 0: return None
    d0 = dseq[j]
    if d0 in p and date in p and p[d0]:
        return (p[date] / p[d0] - 1) * 100
    return None

def _run_trend(dates, lookback=10, nhold=2, rebal=3, tstop=12.0,
               regime='off', crash=-2.0, low_expo=0.0, cost_bps=8.0,
               same_sector=True, lev=1.0, fin_daily=0.025, mm_dd=40.0,
               cool=0, verbose=False, logpath=None):
    ALL = all_dates()
    pos_of = {d: ALL.index(d) for d in dates}
    holdings = {}   # code -> {w, peak_price, sector}
    nav, peak, maxdd, trades = 1.0, 1.0, 0.0, 0
    idx = 1.0; idxh = []
    prev, prev_w = None, {}
    cooldown = 0
    log = []
    for i, date in enumerate(dates):
        sc = SCANS[date]; bench = sc['bench']
        # P&L
        dret = 0.0
        if prev is not None:
            for code, h in holdings.items():
                r = ret(code, prev, date)
                dret += h['w']/100.0 * (r if r is not None else 0.0)
        nav *= (1 + dret/100.0)
        # 融资成本: 借入部分(总敞口-100%)按日息扣
        gross = sum(h['w'] for h in holdings.values())
        borrowed = max(0.0, gross - 100.0)
        nav *= (1 - borrowed/100.0 * fin_daily/100.0)
        peak = max(peak, nav); dd = (peak-nav)/peak*100; maxdd = max(maxdd, dd)
        idx *= (1 + bench/100.0); idxh.append(idx)
        # 强平: 杠杆下回撤击穿维持保证金 → 全部清仓避免爆仓
        if dd >= mm_dd and holdings:
            for code in list(holdings):
                holdings.pop(code); trades += 1
        # 更新持仓峰值(移动止损用)
        for code, h in holdings.items():
            cp = PRICE.get(code, {}).get(date)
            if cp: h['peak_price'] = max(h.get('peak_price', cp), cp)
        # 择时: 默认满仓, 只在确认转弱时才降. maN=指数跌破N日均线
        risk_on = True
        if regime.startswith('ma'):
            n = int(regime[2:])
            if len(idxh) >= n:
                risk_on = idxh[-1] >= sum(idxh[-n:])/n
        elif regime == 'mom5' and len(idxh) > 5:
            risk_on = idxh[-1] >= idxh[-6]
        if bench <= crash:
            risk_on = False
            cooldown = cool          # 崩盘后冷静期: 之后 cool 天不回满仓
        if cooldown > 0:
            risk_on = False
            cooldown -= 1
        target = (100.0 * lev) if risk_on else low_expo

        acts = []
        if i < len(dates)-1:
            dpos = pos_of[date]
            # 移动止损: 从持仓峰值回撤 > tstop → 清出
            for code in list(holdings):
                cp = PRICE.get(code, {}).get(date); pk = holdings[code].get('peak_price')
                if cp and pk and (cp/pk - 1)*100 <= -tstop:
                    holdings.pop(code); trades += 1; acts.append('移损'+code)
            # 再平衡: 按动量排名重选
            if (i % rebal == 0) or (not holdings):
                ranked = []
                for c in sc['rows']:
                    if not eligible(c): continue
                    m = _mom(c['code'], date, lookback, ALL, dpos)
                    if m is not None:
                        ranked.append((m, c))
                ranked.sort(key=lambda x: -x[0])
                picks, used = [], set()
                for m, c in ranked:
                    if len(picks) >= nhold: break
                    if (not same_sector) and c['sector'] in used: continue
                    picks.append(c); used.add(c['sector'])
                # 清掉不在新picks里的
                pickset = set(c['code'] for c in picks)
                for code in list(holdings):
                    if code not in pickset:
                        holdings.pop(code); trades += 1; acts.append('换出'+code)
                for c in picks:
                    if c['code'] not in holdings:
                        cp = PRICE.get(c['code'], {}).get(date)
                        holdings[c['code']] = {'w':0, 'peak_price':cp, 'sector':c['sector']}
                        trades += 1; acts.append('建'+c['code'])
            # 权重: 集中. nhold=1→target; nhold=2→ target*0.6/0.4
            codes = list(holdings)
            if len(codes) >= 2:
                # 按动量定主次
                dpos = pos_of[date]
                mm = {c: (_mom(c, date, lookback, ALL, dpos) or -99) for c in codes}
                codes.sort(key=lambda c: -mm[c])
                holdings[codes[0]]['w'] = target*0.6; holdings[codes[1]]['w'] = target*0.4
                for c in codes[2:]: holdings[c]['w'] = 0
            elif len(codes) == 1:
                holdings[codes[0]]['w'] = target

        log.append([date, bench, dret, nav, dd, list((c,round(h['w'],1)) for c,h in holdings.items()), acts, 'ON' if risk_on else 'off'])
        cur_w = {c: h['w'] for c, h in holdings.items()}
        turn = sum(abs(cur_w.get(c,0)-prev_w.get(c,0)) for c in set(list(cur_w)+list(prev_w)))
        nav *= (1 - turn/100.0*(cost_bps/10000.0)); log[-1][3] = nav
        prev_w = cur_w; prev = date

    if verbose:
        print('='*104)
        print('趋势跟踪 | %s→%s | 动量=%d日 持仓=%d 再平衡=%d日 移损=%.0f%% 择时=%s 同板块=%s 成本=%.0fbp' %
              (dates[0],dates[-1],lookback,nhold,rebal,tstop,regime,same_sector,cost_bps))
        print('='*104)
        print('%-11s %7s %5s %8s %9s %6s  %s' % ('日期','沪深300','择时','当日','累计','DD','持仓/动作'))
        print('-'*104)
        for date,bench,dr,nv,dd,hold,acts,reg in log:
            hs = ', '.join('%s@%g' % (c,wt) for c,wt in hold) or '空仓'
            ac = (' | '+'; '.join(acts)) if acts else ''
            print('%-11s %+6.2f%% %5s %+7.2f%% %+8.2f%% %5.1f%%  %s%s' % (date,bench,reg,dr,(nv-1)*100,dd,hs,ac))
        print('-'*104)
        print('最终 %+.2f%% | 最大回撤 %.2f%% | 交易 %d' % ((nav-1)*100,maxdd,trades))
        print('='*104)
    if logpath:
        lines = ['# 主策略逐日选股日志\n',
                 '策略: 集中2只强趋势 + 默认满仓 + 弱市降70%% + %.0f%%移损 | 动量%d日 再平衡%d日 择时%s 成本%.0fbp\n' % (tstop, lookback, rebal, regime, cost_bps),
                 '窗口: %s → %s | 最终 %+.2f%% | 最大回撤 %.2f%% | 交易 %d\n' % (dates[0], dates[-1], (nav-1)*100, maxdd, trades),
                 '\n> 持仓栏=次日实际持有(代码 名称@仓位%%); 动作栏=当日收盘调仓; 当日=该日持仓收益; 择时ON=满仓/off=降70%%\n\n',
                 '| 日期 | 沪深300 | 择时 | 持仓(次日) | 当日收盘动作 | 当日 | 累计 | DD |\n',
                 '|---|---:|:--:|---|---|---:|---:|---:|\n']
        for date, bench, dr, nv, dd, hold, acts, reg in log:
            byc = SCANS[date]['by_code']
            hs = ', '.join('%s %s@%g%%' % (c, byc.get(c, {}).get('name', ''), wt) for c, wt in hold) or '空仓'
            ac = '; '.join(acts) if acts else '—'
            lines.append('| %s | %+.2f%% | %s | %s | %s | %+.2f%% | %+.2f%% | %.1f%% |\n' %
                         (date, bench, reg, hs, ac, dr, (nv-1)*100, dd))
        with open(logpath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    return (nav-1)*100, maxdd, trades

# ============================================================
# 低换手策略 v3: 周度再平衡 + 大盘择时 + 个股止损
# ============================================================
def _run_lt(dates, rebal=5, regime='ma5', stop=8.0, low_expo=40.0,
            crash=-2.0, cost_bps=8.0, pick_mode='mom', verbose=False):
    holdings = {}   # code -> {w, entry, sector}
    nav, peak, maxdd, trades = 1.0, 1.0, 0.0, 0
    idx = 1.0; idxh = []
    prev, prev_w = None, {}
    log = []
    for i, date in enumerate(dates):
        sc = SCANS[date]; bench = sc['bench']
        rows = sorted(sc['rows'], key=lambda x: -(x['score'] or 0))
        # 当日P&L
        dret = 0.0
        if prev is not None:
            for code, h in holdings.items():
                r = ret(code, prev, date)
                dret += h['w']/100.0 * (r if r is not None else 0.0)
        nav *= (1 + dret/100.0)
        peak = max(peak, nav); dd = (peak-nav)/peak*100; maxdd = max(maxdd, dd)
        # 大盘指数(累积沪深300)
        idx *= (1 + bench/100.0); idxh.append(idx)
        # 择时信号(收盘可知 → 决定次日仓位)
        risk_on = True
        if regime == 'mom5' and len(idxh) > 5:
            risk_on = idxh[-1] >= idxh[-6]
        elif regime == 'ma5' and len(idxh) >= 5:
            risk_on = idxh[-1] >= sum(idxh[-5:])/5
        if bench <= crash:      # 崩盘日 → 次日避险
            risk_on = False
        target = 100.0 if risk_on else low_expo

        acts = []
        if i < len(dates)-1:
            # 止损: 任何日,较入场跌破 stop → 清出
            if stop > 0:
                for code in list(holdings):
                    ep = holdings[code]['entry']; cp = PRICE.get(code,{}).get(date)
                    if ep and cp and (cp/ep-1)*100 <= -stop:
                        holdings.pop(code); trades += 1; acts.append('止损'+code)
            # 再平衡日: 重选前2分散
            if (i % rebal == 0) or not holdings:
                # 剔除走弱旧持仓: 追热看评分掉出前12; 反转看是否已冲高(今日大涨=兑现)
                if pick_mode == 'rev':
                    for code in list(holdings):
                        r = sc['by_code'].get(code, {})
                        if (r.get('pct') or 0) > 5:      # 已大涨→反转获利了结
                            holdings.pop(code); trades += 1; acts.append('兑现'+code)
                else:
                    topcodes = set(c['code'] for c in rows[:12] if c.get('code'))
                    for code in list(holdings):
                        if code not in topcodes:
                            holdings.pop(code); trades += 1; acts.append('换出'+code)
                held_sec = set(h['sector'] for h in holdings.values())
                # 选股口径: mom=评分最高(追热) / rev=合格池里今日跌最多(逢低反转)
                if pick_mode == 'rev':
                    pool = sorted([c for c in rows if eligible(c)], key=lambda x: x.get('pct', 0))
                else:
                    pool = rows
                for p in pick_diverse(pool, 2-len(holdings), held_sec):
                    holdings[p['code']] = {'w':0,'entry':PRICE.get(p['code'],{}).get(date),
                                           'sector':p['sector']}
                    trades += 1; acts.append('建'+p['code'])
            # 设权重 = target 分配(risk_on=100, 否则low_expo)
            codes = list(holdings)
            if len(codes) == 2:
                scr = {c: sc['by_code'].get(c,{}).get('score',0) for c in codes}
                codes.sort(key=lambda c: -scr[c])
                holdings[codes[0]]['w'] = target*0.6; holdings[codes[1]]['w'] = target*0.4
            elif len(codes) == 1:
                holdings[codes[0]]['w'] = target*0.6

        log.append([date, bench, dret, nav, dd, list((c,round(h['w'],1)) for c,h in holdings.items()), acts, 'ON' if risk_on else 'off'])
        cur_w = {c: h['w'] for c, h in holdings.items()}
        turn = sum(abs(cur_w.get(c,0)-prev_w.get(c,0)) for c in set(list(cur_w)+list(prev_w)))
        nav *= (1 - turn/100.0*(cost_bps/10000.0))
        log[-1][3] = nav
        prev_w = cur_w; prev = date

    if verbose:
        print('='*104)
        print('低换手v3 | %s→%s | 再平衡=%d日 择时=%s 止损=%.0f%% 弱市仓=%.0f%% 成本=%.0fbp' %
              (dates[0],dates[-1],rebal,regime,stop,low_expo,cost_bps))
        print('='*104)
        print('%-11s %7s %5s %8s %9s %6s  %s' % ('日期','沪深300','择时','当日','累计','DD','持仓/动作'))
        print('-'*104)
        for date,bench,dr,nv,dd,hold,acts,reg in log:
            hs = ', '.join('%s@%g' % (c,w) for c,w in hold) or '空仓'
            ac = (' | '+'; '.join(acts)) if acts else ''
            print('%-11s %+6.2f%% %5s %+7.2f%% %+8.2f%% %5.1f%%  %s%s' % (date,bench,reg,dr,(nv-1)*100,dd,hs,ac))
        print('-'*104)
        print('最终 %+.2f%% | 最大回撤 %.2f%% | 交易 %d | 旧策略 -3.58%%' % ((nav-1)*100,maxdd,trades))
        print('='*104)
    return (nav-1)*100, maxdd, trades

if __name__ == '__main__':
    import sys
    ADS = all_dates()
    OOS = [d for d in ADS if d <= '2026-06-26']       # 样本外
    INS = [d for d in ADS if d >= '2026-06-29']       # 样本内

    # 主策略参数(与 CLAUDE.md「核心策略」一致): 集中2只强趋势+默认满仓+弱市降70%+15%移损
    def MAIN(win, verbose=False):
        return _run_trend(win, lookback=5, nhold=2, rebal=3, tstop=15.0,
                          regime='ma5', low_expo=70.0, crash=-2.0,
                          cost_bps=8.0, same_sector=True, verbose=verbose)

    if '--sweep' in sys.argv:
        for label, win in [('样本外 6/01-6/26', OOS), ('样本内 6/29-7/10', INS), ('全窗口', ADS)]:
            print('\n### %s (%d日) 扣费后净收益 (COST=%.0fbp单边) ###' % (label, len(win), COST_BPS))
            print('%-18s %8s %8s %6s' % ('TOPN/DAYS', '净收益', '回撤', '交易'))
            print('-'*44)
            for days in (1, 2, 3):
                EXIT_DAYS = days
                r, dd, tr = _run(win, verbose=False)
                print('%-18s %+7.2f%% %6.2f%% %5d' % ('4 / %d' % days, r, dd, tr))
        print('\n### EXIT_DAYS=1 成本敏感性 ###')
        print('%-12s %10s %10s %10s' % ('单边成本', '样本外', '样本内', '全窗口'))
        print('-'*46)
        EXIT_DAYS = 1
        for cb in (0.0, 5.0, 8.0, 12.0, 20.0):
            COST_BPS = cb
            ro = _run(OOS, verbose=False)[0]
            ri = _run(INS, verbose=False)[0]
            ra = _run(ADS, verbose=False)[0]
            print('%-9.0fbp %+9.2f%% %+9.2f%% %+9.2f%%' % (cb, ro, ri, ra))
        print('-'*46)
        print('旧策略(无此对比): -3.58%')
    elif '--oos' in sys.argv:
        _run(OOS, verbose=True)
    elif '--lt' in sys.argv:
        print('低换手 v3 参数扫描 (扣8bp成本, 两窗口都测, 按"较差窗口"排序找稳健配置)')
        print('%-34s %8s %8s %8s %6s' % ('再平衡/择时/止损/弱市仓','样本外','样本内','全窗口','较差'))
        print('-'*76)
        results = []
        for rebal in (5, 3):
            for regime in ('off', 'mom5', 'ma5'):
                for stop in (0.0, 6.0, 10.0):
                    for low in (0.0, 40.0):
                        ro = _run_lt(OOS, rebal, regime, stop, low, cost_bps=8.0)[0]
                        ri = _run_lt(INS, rebal, regime, stop, low, cost_bps=8.0)[0]
                        ra = _run_lt(ADS, rebal, regime, stop, low, cost_bps=8.0)[0]
                        worse = min(ro, ri)
                        results.append((worse, ro, ri, ra, rebal, regime, stop, low))
        results.sort(reverse=True)
        for worse, ro, ri, ra, rebal, regime, stop, low in results[:12]:
            tag = '%d日/%s/%.0f%%/%.0f%%' % (rebal, regime, stop, low)
            print('%-34s %+7.2f%% %+7.2f%% %+7.2f%% %+6.2f%%' % (tag, ro, ri, ra, worse))
        print('-'*76)
        b = results[0]
        print('最稳健: 再平衡%d日 择时%s 止损%.0f%% 弱市仓%.0f%% → 样本外%+.2f%% 样本内%+.2f%% (旧策略-3.58%%)'
              % (b[4], b[5], b[6], b[7], b[1], b[2]))
    else:
        # 默认: 跑主策略, 全程明细 + 两窗口验证
        def bh(win):
            x = 1.0
            for d in win: x *= (1 + SCANS[d]['bench']/100)
            return (x-1)*100
        MAIN(ADS, verbose=True)
        print('\n主策略两窗口验证 (扣8bp成本):')
        for lbl, win in [('样本外 %s-%s' % (OOS[0][5:], OOS[-1][5:]), OOS),
                         ('样本内 %s-%s' % (INS[0][5:], INS[-1][5:]), INS),
                         ('全程   %s-%s' % (ADS[0][5:], ADS[-1][5:]), ADS)]:
            r, dd, tr = MAIN(win)
            print('  %-18s 收益 %+7.2f%% | 回撤 %5.2f%% | 沪深300 %+6.2f%%' % (lbl, r, dd, bh(win)))
