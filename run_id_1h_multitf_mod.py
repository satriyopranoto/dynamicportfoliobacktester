"""
IDX — Intraday 1h, Single TF vs Multi TF Moderat
Multi TF = existing conditions + PDI_d > MDI_d (daily only, tanpa close>basisd)
1 year, 116 stocks
"""
import os, sys, numpy as np, pandas as pd, yfinance as yf, time, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(OUTPUT_DIR, 'id_liquid.csv')
if not os.path.exists(csv_path):
    print(f"id_liquid.csv not found at {csv_path}")
    sys.exit(1)

with open(csv_path) as f:
    tickers = [row.strip() for row in f if row.strip() and not row.startswith('Symbol')]
print(f"Tickers: {len(tickers)}")

PERIOD = "1y"
CAPITAL = 100000000
MAX_POS = 10; RISK_PCT = 1.0; MIN_ADX = 20

def calc_i(c, h, l):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr = pd.Series(tr).rolling(14).mean().values
    sp = pd.Series(pdm).rolling(14).mean().values; sm = pd.Series(mdm).rolling(14).mean().values
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(14).mean().values
    return adx, pdi, mdi

def load(ticker):
    try:
        time.sleep(0.3)
        h1 = yf.download(ticker, period=PERIOD, interval='1h', progress=False)
        if h1.empty or len(h1) < 250: return None
        if isinstance(h1.columns, pd.MultiIndex): h1.columns = h1.columns.get_level_values(0)
        c = h1['Close'].values.astype(float); h = h1['High'].values.astype(float); l = h1['Low'].values.astype(float)
        sma = pd.Series(c).rolling(20).mean().values
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        sl = pd.Series(c).rolling(28).max().values - 2*pd.Series(tr).rolling(10).mean().values*(2.8-1)/2.8
        adx, pdi, mdi = calc_i(c, h, l)
        r = {'ticker':ticker,'dates':h1.index,'close':c,'high':h,'low':l,'sma20':sma,'sl':sl,'adx':adx,'pdi':pdi,'mdi':mdi}
        
        d = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
        if d.empty or len(d) < 30: return None
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        dc = d['Close'].values.astype(float); dh = d['High'].values.astype(float); dl = d['Low'].values.astype(float)
        _, dp, dm = calc_i(dc, dh, dl)
        d_dates = d.index; d_map = {}
        for i, dd in enumerate(d_dates): d_map[dd] = {'pdi': dp[i], 'mdi': dm[i]}
        def get_d(dt):
            for dd in reversed(d_dates):
                if dd <= dt.tz_localize(None): return d_map.get(dd, {})
            return {}
        hp = np.full(len(c), np.nan); hm = np.full(len(c), np.nan)
        for i, dt in enumerate(h1.index):
            w = get_d(dt); hp[i] = w.get('pdi', np.nan); hm[i] = w.get('mdi', np.nan)
        r['d_pdi'] = hp; r['d_mdi'] = hm
        return r
    except:
        return None

def run(stocks, multi=False):
    lbl = "Multi TF Moderat" if multi else "Single TF"
    print(f"\n  {lbl}...")
    all_d = set()
    for t, s in stocks.items():
        for d in s['dates']: all_d.add(pd.Timestamp(d).normalize())
    tl = sorted(all_d)
    print(f"  Bars: {len(tl)}")
    si = {}
    for t, s in stocks.items():
        im = {}
        for i, d in enumerate(s['dates']): im[pd.Timestamp(d).normalize()] = i
        si[t] = im
    cash = float(CAPITAL); pos = []; trades = []; slx=0; tpx=0
    for di, td in enumerate(tl):
        if di % 500 == 0: print(f"    {di}/{len(tl)}...")
        i = 0
        while i < len(pos):
            p = pos[i]; s = stocks[p['ticker']]; b = si[p['ticker']].get(td)
            if b is not None:
                cl = float(s['close'][b])
                if cl < p['es']:
                    cash += p['sz']*cl; trades.append((cl/p['ep']-1)*100); slx+=1; pos.pop(i); continue
                fl = (cl-p['ep'])/p['ep']*100
                if fl > p['tp']:
                    cash += p['sz']*cl; trades.append((cl/p['ep']-1)*100); tpx+=1; pos.pop(i); continue
            i += 1
        if len(pos) < MAX_POS:
            cand = []
            for t, s in stocks.items():
                if any(p['ticker']==t for p in pos): continue
                b = si[t].get(td)
                if b is None or b < 20: continue
                if not (float(s['low'][b]) > float(s['sl'][b]) and float(s['close'][b]) > float(s['sma20'][b])
                    and float(s['adx'][b]) > MIN_ADX and float(s['pdi'][b]) > float(s['mdi'][b])
                    and float(s['pdi'][b]) > float(s['pdi'][max(0,b-5)])):
                    continue
                if multi:
                    dp = float(s['d_pdi'][b]) if not np.isnan(s['d_pdi'][b]) else None
                    dm = float(s['d_mdi'][b]) if not np.isnan(s['d_mdi'][b]) else None
                    if dp is None or dm is None or not (dp > dm): continue
                cl = float(s['close'][b]); slv = float(s['sl'][b])
                sc = sum(1 for j in range(max(0,b-100),b+1) if not np.isnan(s['adx'][j]) and not np.isnan(s['sma20'][j]) and s['adx'][j]>25 and s['close'][j]>s['sma20'][j])
                cand.append({'t':t,'sc':sc,'cl':cl,'sl':slv})
            cand.sort(key=lambda x: x['sc'], reverse=True)
            for c in cand:
                if len(pos) >= MAX_POS: break
                sd = abs(c['cl']-c['sl'])
                if sd <= 0: continue
                sz = max(1, min(int(cash*(RISK_PCT/100)/sd), int((cash*0.95)/c['cl'])))
                cost = sz * c['cl']
                if cost > cash*0.95 or sz <= 0: continue
                pos.append({'ticker':c['t'],'ep':c['cl'],'es':c['sl'],'sz':sz,'cost':cost,'tp':0.4*sd/c['cl']*100})
                cash -= cost
    for p in pos:
        s = stocks[p['ticker']]; cl = float(s['close'][-1]); cash += p['sz']*cl
        trades.append((cl/p['ep']-1)*100)
    fe = cash
    r = {'ret':(fe-CAPITAL)/CAPITAL*100,'n':len(trades)}
    if trades:
        ta = np.array(trades); w=ta[ta>0]; l_=ta[ta<=0]
        r['wr']=len(w)/len(ta)*100; r['best']=ta.max(); r['worst']=ta.min()
        r['pf']=abs(w.sum()/l_.sum()) if len(l_)>0 and l_.sum()!=0 else float('inf')
    return r

print(f"\nLoading {len(tickers)} stocks (1h + daily)...")
stocks = {}
for i, t in enumerate(tickers):
    s = load(t)
    if s is not None: stocks[t] = s
    if (i+1)%30==0: print(f"  [{i+1}/{len(tickers)}] ({len(stocks)} valid)")
print(f"Loaded: {len(stocks)} stocks")

results = []
for multi in [False, True]:
    r = run(stocks, multi)
    results.append(r)
    lbl = "Multi TF Moderat" if multi else "Single TF"
    print(f"\n{'='*40}")
    print(f"  IDX 1h — {lbl}")
    print(f"{'='*40}")
    print(f"  Return : {r['ret']:+.2f}%")
    print(f"  Trades : {r['n']} | WR: {r.get('wr',0):.1f}%")
    print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")

print(f"\n\n{'='*45}")
print(f"  PERBANDINGAN IDX 1h")
print(f"{'='*45}")
print(f"  {'Metrik':18s} {'Single TF':14s} {'Multi TF Mod':14s}")
print('  ' + '-'*46)
for m in ['Return','Trades','WR']:
    s = results[0].get(m.lower(), 0)
    m2 = results[1].get(m.lower(), 0)
    if m in ['Return','WR']:
        print(f"  {m:18s} {s:>+9.2f}%      {m2:>+9.2f}%")
    else:
        print(f"  {m:18s} {s:>9.0f}      {m2:>9.0f}")
