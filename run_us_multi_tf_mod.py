"""
US Market — Single TF vs Multi TF Moderat (hanya PDI_d > MDI_d, tanpa close>basisd)
10 years, 49 stocks
"""
import os, numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings('ignore')

CAPITAL = 6000; MAX_POS = 6; RISK_PCT = 1.0; PERIOD = "10y"; MIN_ADX = 20
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","XOM","PG","KO","PEP","HD","DIS","NFLX","MA",
    "UNH","BAC","ABBV","PFE","TMO","AVGO","CVX","LLY","COST",
    "MRK","ABT","ACN","DHR","LIN","NKE","WFC","TXN","QCOM","UPS",
    "RTX","LOW","SPGI","INTU","GS","MS","C","BLK","SCHW","PLD"]

def load_stock(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
        if df.empty or len(df) < 250: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        c = df['Close'].values.astype(float); h = df['High'].values.astype(float); l = df['Low'].values.astype(float)
        sma20 = pd.Series(c).rolling(20).mean().values
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr = pd.Series(tr).rolling(14).mean().values
        sp = pd.Series(pdm).rolling(14).mean().values
        sm = pd.Series(mdm).rolling(14).mean().values
        pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        adx = pd.Series(dx).rolling(14).mean().values
        sl = pd.Series(c).rolling(28).max().values - 2*pd.Series(tr).rolling(10).mean().values*(2.8-1)/2.8
        
        result = {'ticker':ticker,'dates':df.index,'close':c,'high':h,'low':l,'sma20':sma20,'sl':sl,'adx':adx,'pdi':pdi,'mdi':mdi}
        
        # Weekly
        wk = yf.download(ticker, period=PERIOD, interval='1wk', progress=False)
        if wk.empty or len(wk) < 30: return None
        if isinstance(wk.columns, pd.MultiIndex): wk.columns = wk.columns.get_level_values(0)
        wk_c = wk['Close'].values.astype(float); wk_h = wk['High'].values.astype(float); wk_l = wk['Low'].values.astype(float)
        tr_w = np.maximum(wk_h-wk_l, np.maximum(np.abs(wk_h-np.roll(wk_c,1)), np.abs(wk_l-np.roll(wk_c,1))))
        up_w, dn_w = np.diff(wk_h, prepend=wk_h[0]), np.diff(wk_l, prepend=wk_l[0])
        pdm_w = np.where((up_w>dn_w)&(up_w>0), up_w, 0); mdm_w = np.where((dn_w>up_w)&(dn_w>0), dn_w, 0)
        atr_w = pd.Series(tr_w).rolling(14).mean().values
        sp_w = pd.Series(pdm_w).rolling(14).mean().values
        sm_w = pd.Series(mdm_w).rolling(14).mean().values
        wk_pdi = np.where(atr_w>0, 100*sp_w/atr_w, 0); wk_mdi = np.where(atr_w>0, 100*sm_w/atr_w, 0)
        
        wk_dates = wk.index; wk_map = {}
        for i, wd in enumerate(wk_dates): wk_map[wd] = {'pdi': wk_pdi[i], 'mdi': wk_mdi[i]}
        def get_w(d):
            for wd in reversed(wk_dates):
                if wd <= d: return wk_map.get(wd, {})
            return {}
        dd = df.index
        wp = np.full(len(dd), np.nan); wm = np.full(len(dd), np.nan)
        for i, d in enumerate(dd):
            w = get_w(d); wp[i] = w.get('pdi', np.nan); wm[i] = w.get('mdi', np.nan)
        result['wk_pdi'] = wp; result['wk_mdi'] = wm
        return result
    except:
        return None

def buy(stock, bar, multi=False):
    if bar < 20 or bar >= len(stock['close']): return False
    for k in ['adx','pdi','mdi','sma20','sl']:
        if np.isnan(stock[k][bar]): return False
    c, l, s, sl = float(stock['close'][bar]), float(stock['low'][bar]), float(stock['sma20'][bar]), float(stock['sl'][bar])
    a, p, m = float(stock['adx'][bar]), float(stock['pdi'][bar]), float(stock['mdi'][bar])
    if not (l > sl and c > s and a > MIN_ADX and p > m): return False
    if bar >= 5 and not (p > float(stock['pdi'][bar-5])): return False
    if multi:
        wp = float(stock['wk_pdi'][bar]) if not np.isnan(stock['wk_pdi'][bar]) else None
        wm = float(stock['wk_mdi'][bar]) if not np.isnan(stock['wk_mdi'][bar]) else None
        if wp is None or wm is None or not (wp > wm): return False
    return True

def run(stocks, multi=False):
    all_d = set()
    for t, s in stocks.items():
        for d in s['dates']: all_d.add(pd.Timestamp(d).normalize())
    tl = sorted(all_d)
    si = {}
    for t, s in stocks.items():
        im = {}
        for i, d in enumerate(s['dates']): im[pd.Timestamp(d).normalize()] = i
        si[t] = im
    cash = float(CAPITAL); pos = []; eq = []; trades = []; sl_x = 0; tp_x = 0
    for di, td in enumerate(tl):
        if di % 500 == 0: print(f"    {di}/{len(tl)}... (${cash:,.0f})")
        i = 0
        while i < len(pos):
            p = pos[i]; s = stocks[p['ticker']]; b = si[p['ticker']].get(td)
            if b is not None:
                cl = float(s['close'][b])
                if cl < p['es']:
                    cash += p['sz']*cl; trades.append({'pnl':p['sz']*cl-p['cost'],'r':(cl/p['ep']-1)*100,'ex':'SL'}); sl_x+=1; pos.pop(i); continue
                fl = (cl-p['ep'])/p['ep']*100
                if fl > p['tp']:
                    cash += p['sz']*cl; trades.append({'pnl':p['sz']*cl-p['cost'],'r':(cl/p['ep']-1)*100,'ex':'TP'}); tp_x+=1; pos.pop(i); continue
            i += 1
        if len(pos) < MAX_POS:
            cand = []
            for t, s in stocks.items():
                if any(p['ticker']==t for p in pos): continue
                b = si[t].get(td)
                if b is None or b < 20: continue
                if not buy(s, b, multi): continue
                cl = float(s['close'][b]); slv = float(s['sl'][b])
                score = np.sum([s['adx'][b]>25 and s['close'][b]>s['sma20'][b] for _ in range(1)])
                cand.append({'t':t,'sc':score,'cl':cl,'sl':slv})
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
        trades.append({'pnl':p['sz']*cl-p['cost'],'r':(cl/p['ep']-1)*100,'ex':'END'})
    fe = eq[-1] if eq else CAPITAL; yrs = len(tl)/252
    r = {'ret':(fe-CAPITAL)/CAPITAL*100,'final':fe,'n':len(trades),'sl':sl_x,'tp':tp_x}
    if trades:
        d = pd.DataFrame(trades); w = d[d['pnl']>0]; l = d[d['pnl']<=0]
        r['wr'] = len(w)/len(d)*100; r['best'] = d['r'].max(); r['worst'] = d['r'].min()
        r['pf'] = abs(w['pnl'].sum()/l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else float('inf')
        ea = np.array(eq); pk = np.maximum.accumulate(ea); r['dd'] = ((pk-ea)/pk*100).max()
        dr = pd.Series(eq).pct_change().dropna()
        r['sharpe'] = np.sqrt(252)*dr.mean()/dr.std() if dr.std()>0 else 0
        r['cagr'] = ((fe/CAPITAL)**(1/yrs)-1)*100 if yrs>0 else 0
    return r

print("US Market — Single TF vs Multi TF Moderat (10 yr)")
print(f"Loading {len(tickers)} stocks...")
stocks = {}
for i, t in enumerate(tickers):
    s = load_stock(t)
    if s is not None: stocks[t] = s
    if (i+1)%20==0: print(f"  [{i+1}/{len(tickers)}] ({len(stocks)} valid)")
print(f"Loaded: {len(stocks)} stocks\n")

for multi in [False, True]:
    lbl = "Multi TF Moderat (wk PDI>MDI)" if multi else "Single TF (Daily)"
    print(f"Running {lbl}...")
    r = run(stocks, multi)
    print(f"  Return: {r['ret']:+.2f}% | CAGR: {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe: {r.get('sharpe',0):.2f} | DD: -{r.get('dd',0):.2f}%")
    print(f"  PF: {r.get('pf',0):.2f} | WR: {r.get('wr',0):.1f}%")
    print(f"  Trades: {r['n']} | Worst: {r.get('worst',0):.2f}%")
    print()
