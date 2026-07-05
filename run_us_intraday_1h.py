"""
US Market — Intraday 1h, Single TF vs Multi TF (1h + Daily PDI>MDI)
1 year data
"""
import os, sys, numpy as np, pandas as pd, yfinance as yf, time, warnings
warnings.filterwarnings('ignore')

CAPITAL = 6000; MAX_POS = 6; RISK_PCT = 1.0; MIN_ADX = 20; PERIOD = "1y"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","XOM","PG","KO","PEP","HD","DIS","NFLX","MA",
    "UNH","BAC","ABBV","PFE","TMO","AVGO","CVX","LLY","COST",
    "MRK","ABT","ACN","DHR","LIN","NKE","WFC","TXN","QCOM","UPS",
    "RTX","LOW","SPGI","INTU","GS","MS","C","BLK","SCHW","PLD"]

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
        time.sleep(0.2)
        h1 = yf.download(ticker, period=PERIOD, interval='1h', progress=False)
        if h1.empty or len(h1) < 250: return None
        if isinstance(h1.columns, pd.MultiIndex): h1.columns = h1.columns.get_level_values(0)
        c1 = h1['Close'].values.astype(float); hh1 = h1['High'].values.astype(float); l1 = h1['Low'].values.astype(float)
        sma20 = pd.Series(c1).rolling(20).mean().values
        tr1 = np.maximum(hh1-l1, np.maximum(np.abs(hh1-np.roll(c1,1)), np.abs(l1-np.roll(c1,1))))
        slv = pd.Series(c1).rolling(28).max().values - 2*pd.Series(tr1).rolling(10).mean().values*(2.8-1)/2.8
        adx, pdi, mdi = calc_i(c1, hh1, l1)
        r = {'ticker':ticker,'dates':h1.index,'close':c1,'high':hh1,'low':l1,'sma20':sma20,'sl':slv,'adx':adx,'pdi':pdi,'mdi':mdi}
        
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
        h1d = h1.index; hp = np.full(len(h1d), np.nan); hm = np.full(len(h1d), np.nan)
        for i, dt in enumerate(h1d):
            w = get_d(dt); hp[i] = w.get('pdi', np.nan); hm[i] = w.get('mdi', np.nan)
        r['d_pdi'] = hp; r['d_mdi'] = hm
        return r
    except:
        return None

def buy_sig(s, bar, multi=False):
    if bar < 20 or bar >= len(s['close']): return False
    for k in ['adx','pdi','mdi','sma20','sl']:
        if np.isnan(s[k][bar]): return False
    c, l, sm, sl = float(s['close'][bar]), float(s['low'][bar]), float(s['sma20'][bar]), float(s['sl'][bar])
    a, p, m = float(s['adx'][bar]), float(s['pdi'][bar]), float(s['mdi'][bar])
    if not (l > sl and c > sm and a > MIN_ADX and p > m): return False
    if bar >= 5 and not (p > float(s['pdi'][bar-5])): return False
    if multi:
        dp = float(s['d_pdi'][bar]) if not np.isnan(s['d_pdi'][bar]) else None
        dm = float(s['d_mdi'][bar]) if not np.isnan(s['d_mdi'][bar]) else None
        if dp is None or dm is None or not (dp > dm): return False
    return True

def run(stocks, multi=False):
    lbl = "Multi TF" if multi else "Single TF"
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
    cash = float(CAPITAL); pos = []; eq = []; trades = []; slx=0; tpx=0
    for di, td in enumerate(tl):
        if di % 500 == 0: print(f"    {di}/{len(tl)}...")
        i = 0
        while i < len(pos):
            p = pos[i]; s = stocks[p['ticker']]; b = si[p['ticker']].get(td)
            if b is not None:
                cl = float(s['close'][b])
                if cl < p['es']:
                    cash += p['sz']*cl; trades.append({'r':(cl/p['ep']-1)*100,'ex':'SL'}); slx+=1; pos.pop(i); continue
                fl = (cl-p['ep'])/p['ep']*100
                if fl > p['tp']:
                    cash += p['sz']*cl; trades.append({'r':(cl/p['ep']-1)*100,'ex':'TP'}); tpx+=1; pos.pop(i); continue
            i += 1
        if len(pos) < MAX_POS:
            cand = []
            for t, s in stocks.items():
                if any(p['ticker']==t for p in pos): continue
                b = si[t].get(td)
                if b is None or b < 20: continue
                if not buy_sig(s, b, multi): continue
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
        eq.append(cash + sum(p['sz']*float(stocks[p['ticker']]['close'][si[p['ticker']].get(td,-1)]) for p in pos if si[p['ticker']].get(td,-1)>=0))
    for p in pos:
        s = stocks[p['ticker']]; cl = float(s['close'][-1]); cash += p['sz']*cl
        trades.append({'r':(cl/p['ep']-1)*100,'ex':'END'})
    fe = eq[-1] if eq else CAPITAL
    r = {'ret':(fe-CAPITAL)/CAPITAL*100,'n':len(trades),'sl':slx,'tp':tpx}
    if trades:
        d = pd.DataFrame(trades); w = d[d['r']>0]; l_ = d[d['r']<=0]
        r['wr'] = len(w)/len(d)*100; r['best'] = d['r'].max(); r['worst'] = d['r'].min()
        r['pf'] = abs(w['r'].sum()/l_['r'].sum()) if len(l_)>0 and l_['r'].sum()!=0 else float('inf')
        ea = np.array(eq); pk = np.maximum.accumulate(ea); r['dd'] = ((pk-ea)/pk*100).max()
        dr = pd.Series(eq).pct_change().dropna()
        r['sharpe'] = np.sqrt(252*7)*dr.mean()/dr.std() if dr.std()>0 else 0
    return r

print("US — Intraday 1h: Single vs Multi TF (1 year)")
print(f"Loading {len(tickers)} stocks...")
stocks = {}
for i, t in enumerate(tickers):
    s = load(t)
    if s is not None: stocks[t] = s
    if (i+1)%20==0: print(f"  [{i+1}/{len(tickers)}] ({len(stocks)} valid)")
print(f"Loaded: {len(stocks)} stocks")

for multi in [False, True]:
    r = run(stocks, multi)
    label = "Multi TF (1h + Daily PDI>MDI)" if multi else "Single TF (1h only)"
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Return   : {r['ret']:+.2f}%")
    print(f"  Sharpe   : {r.get('sharpe',0):.2f}")
    print(f"  Max DD   : -{r.get('dd',0):.2f}%")
    print(f"  Trades   : {r['n']} | WR: {r.get('wr',0):.1f}%")
    print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")
    print()
