"""
Dynamic Portfolio — ID Market (116 stocks), 10 years.
Compare: Single TF vs Multi TF Moderat (hanya PDI_d > MDI_d, tanpa close>basisd)
"""
import os, sys, csv, json, numpy as np, pandas as pd, yfinance as yf
import warnings
warnings.filterwarnings('ignore')

CAPITAL = 100_000_000
MAX_POSITIONS = 10
RISK_PCT = 1.0
PERIOD = "10y"
MIN_ADX = 20

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(OUTPUT_DIR, 'id_liquid.csv')
with open(csv_path) as f:
    reader = csv.DictReader(f)
    tickers_list = []
    for row in reader:
        t = row['Symbol'].strip().upper().replace('.JK','') + '.JK'
        tickers_list.append(t)

def load_stock(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
        if df.empty or len(df) < 250: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df['Close'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        
        sma20 = pd.Series(c).rolling(20).mean().values
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(14).mean().values
        sp = pd.Series(pdm).rolling(14).mean().values
        sm = pd.Series(mdm).rolling(14).mean().values
        pdi = np.where(atr_s>0, 100*sp/atr_s, 0); mdi = np.where(atr_s>0, 100*sm/atr_s, 0)
        dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
        adx = pd.Series(dx).rolling(14).mean().values
        sl = pd.Series(c).rolling(28).max().values - 2 * pd.Series(tr).rolling(10).mean().values * (2.8-1)/2.8
        
        result = {
            'ticker': ticker, 'dates': df.index, 'close': c, 'high': h, 'low': l,
            'sma20': sma20, 'sl': sl, 'adx': adx, 'pdi': pdi, 'mdi': mdi,
        }
        
        # Weekly
        wk = yf.download(ticker, period=PERIOD, interval='1wk', progress=False)
        if wk.empty or len(wk) < 30: return None
        if isinstance(wk.columns, pd.MultiIndex):
            wk.columns = wk.columns.get_level_values(0)
        wk_c = wk['Close'].values.astype(float)
        wk_h = wk['High'].values.astype(float)
        wk_l = wk['Low'].values.astype(float)
        
        tr_w = np.maximum(wk_h-wk_l, np.maximum(np.abs(wk_h-np.roll(wk_c,1)), np.abs(wk_l-np.roll(wk_c,1))))
        up_w, dn_w = np.diff(wk_h, prepend=wk_h[0]), np.diff(wk_l, prepend=wk_l[0])
        pdm_w = np.where((up_w>dn_w)&(up_w>0), up_w, 0); mdm_w = np.where((dn_w>up_w)&(dn_w>0), dn_w, 0)
        atr_w = pd.Series(tr_w).rolling(14).mean().values
        sp_w = pd.Series(pdm_w).rolling(14).mean().values
        sm_w = pd.Series(mdm_w).rolling(14).mean().values
        wk_pdi = np.where(atr_w>0, 100*sp_w/atr_w, 0); wk_mdi = np.where(atr_w>0, 100*sm_w/atr_w, 0)
        dx_w = np.where((wk_pdi+wk_mdi)>0, 100*np.abs(wk_pdi-wk_mdi)/(wk_pdi+wk_mdi), 0)
        wk_adx = pd.Series(dx_w).rolling(14).mean().values
        
        wk_dates = wk.index
        wk_map = {}
        for i, wd in enumerate(wk_dates):
            wk_map[wd] = {'pdi': wk_pdi[i], 'mdi': wk_mdi[i]}
        
        def get_weekly(d):
            for wd in reversed(wk_dates):
                if wd <= d: return wk_map.get(wd, {})
            return {}
        
        daily_dates = df.index
        wk_pdi_arr = np.full(len(daily_dates), np.nan)
        wk_mdi_arr = np.full(len(daily_dates), np.nan)
        for i, d in enumerate(daily_dates):
            w = get_weekly(d)
            wk_pdi_arr[i] = w.get('pdi', np.nan)
            wk_mdi_arr[i] = w.get('mdi', np.nan)
        
        result['wk_pdi'] = wk_pdi_arr
        result['wk_mdi'] = wk_mdi_arr
        
        return result
    except:
        return None

def check_buy_signal(stock, bar_idx, multi_tf=False):
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    for k in ['adx','pdi','mdi','sma20','sl']:
        if np.isnan(stock[k][bar_idx]): return False
    close, low = float(stock['close'][bar_idx]), float(stock['low'][bar_idx])
    sma20, sl = float(stock['sma20'][bar_idx]), float(stock['sl'][bar_idx])
    adx, pdi, mdi = float(stock['adx'][bar_idx]), float(stock['pdi'][bar_idx]), float(stock['mdi'][bar_idx])
    
    if not (low > sl and close > sma20): return False
    if not (adx > MIN_ADX and pdi > mdi): return False
    if bar_idx >= 5 and not (pdi > float(stock['pdi'][bar_idx-5])): return False
    
    if multi_tf:
        wk_pdi = float(stock['wk_pdi'][bar_idx]) if not np.isnan(stock['wk_pdi'][bar_idx]) else None
        wk_mdi = float(stock['wk_mdi'][bar_idx]) if not np.isnan(stock['wk_mdi'][bar_idx]) else None
        if wk_pdi is None or wk_mdi is None: return False
        if not (wk_pdi > wk_mdi): return False  # ONLY PDI > MDI
    
    return True

def calc_trend_score(close, sma20, adx, n=100):
    start = max(0, len(close) - n)
    total = 0; bull = 0
    for i in range(start, len(close)):
        if i < 20 or np.isnan(adx[i]) or np.isnan(sma20[i]): continue
        total += 1
        if float(adx[i]) > 25 and float(close[i]) > float(sma20[i]): bull += 1
    return round(bull / total * 100, 1) if total > 0 else 0

def run_dynamic_portfolio(stocks_dict, multi_tf=False):
    all_dates = set()
    for ticker, s in stocks_dict.items():
        for d in s['dates']: all_dates.add(pd.Timestamp(d).normalize())
    timeline = sorted(all_dates)
    
    stock_idx = {}
    for ticker, s in stocks_dict.items():
        idx_map = {}
        for i, d in enumerate(s['dates']): idx_map[pd.Timestamp(d).normalize()] = i
        stock_idx[ticker] = idx_map
    
    cash = float(CAPITAL); positions = []
    equity_history = []; all_trades = []
    sl_exits = 0; tp_exits = 0; total_signals = 0
    
    for day_idx, today in enumerate(timeline):
        if day_idx % 400 == 0:
            print(f"    {day_idx}/{len(timeline)}... ({len(positions)} pos, Rp{cash:,.0f})")
        
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(today)
            if bar is not None:
                close = float(s['close'][bar])
                if close < pos['entry_sl']:
                    cash += pos['size'] * close
                    all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
                        'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'SL'})
                    sl_exits += 1; positions.pop(i); continue
                floating = ((close-pos['entry_price'])/pos['entry_price'])*100
                if floating > pos['tp']:
                    cash += pos['size'] * close
                    all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
                        'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'TP'})
                    tp_exits += 1; positions.pop(i); continue
            i += 1
        
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker']==ticker for p in positions): continue
                bar = stock_idx[ticker].get(today)
                if bar is None or bar < 20: continue
                if not check_buy_signal(s, bar, multi_tf): continue
                close = float(s['close'][bar]); sl_val = float(s['sl'][bar])
                score = calc_trend_score(s['close'], s['sma20'], s['adx'])
                candidates.append({'ticker':ticker,'score':score,'close':close,'sl_val':sl_val})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            total_signals += len(candidates)
            for cand in candidates:
                if len(positions) >= MAX_POSITIONS: break
                stop_dist = abs(cand['close']-cand['sl_val'])
                if stop_dist <= 0: continue
                risk = cash * (RISK_PCT/100.0)
                size = int(risk / stop_dist)
                max_s = int((cash*0.95)/cand['close'])
                if max_s <= 0: continue
                size = max(1, min(size, max_s))
                cost = size * cand['close']
                if cost > cash * 0.95: continue
                positions.append({'ticker':cand['ticker'],'entry_price':cand['close'],
                    'entry_sl':cand['sl_val'],'size':size,'cost':cost,
                    'tp':0.4*stop_dist/cand['close']*100})
                cash -= cost
        
        equity_history.append(cash + sum(
            p['size']*float(stocks_dict[p['ticker']]['close'][stock_idx[p['ticker']].get(today,-1)])
            for p in positions if stock_idx[p['ticker']].get(today,-1)>=0))
    
    for pos in positions:
        s = stocks_dict[pos['ticker']]; close = float(s['close'][-1])
        cash += pos['size'] * close
        all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
            'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'END'})
    
    final_eq = equity_history[-1] if equity_history else CAPITAL
    yrs = len(timeline)/252
    r = {'return':(final_eq-CAPITAL)/CAPITAL*100,'final':final_eq,'trades':len(all_trades),
         'sl':sl_exits,'tp':tp_exits,'signals':total_signals}
    
    if len(all_trades) > 0:
        df = pd.DataFrame(all_trades)
        w = df[df['pnl']>0]; l = df[df['pnl']<=0]
        r['wr'] = len(w)/len(df)*100
        r['best'] = df['return_pct'].max(); r['worst'] = df['return_pct'].min()
        r['pf'] = abs(w['pnl'].sum()/l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else float('inf')
        eq_arr = np.array(equity_history)
        peak = np.maximum.accumulate(eq_arr)
        r['dd'] = ((peak-eq_arr)/peak*100).max()
        daily_r = pd.Series(equity_history).pct_change().dropna()
        r['sharpe'] = np.sqrt(252)*daily_r.mean()/daily_r.std() if daily_r.std()>0 else 0
        r['cagr'] = ((final_eq/CAPITAL)**(1/yrs)-1)*100 if yrs>0 else 0
    return r

# ══ MAIN ══
print("IDX Multi-TF Comparison (10 years, 116 stocks)")
print("1) Single TF (Daily only)")
print("2) Multi TF Moderat (daily + weekly PDI>MDI only)")
print()

print(f"Loading {len(tickers_list)} stocks...")
stocks = {}
for i, t in enumerate(tickers_list):
    s = load_stock(t)
    if s is not None: stocks[t] = s
    if (i+1)%30==0: print(f"  [{i+1}/{len(tickers_list)}] ({len(stocks)} valid)")
print(f"Loaded: {len(stocks)} stocks\n")

# Define the three versions
versions = [
    ("Single TF (Daily only)", lambda bar: check_buy_signal(bar, False)),
    ("Multi TF Moderat (wk+DI>-DI only)", lambda bar: check_buy_signal(bar, True)),
]

# We need a different approach since check_buy_signal signature differs
# Let me just run one at a time
for multi_tf in [False, True]:
    label = "Multi TF Moderat (PDI_d > MDI_d only)" if multi_tf else "Single TF (Daily only)"
    print(f"Running {label}...")
    r = run_dynamic_portfolio(stocks, multi_tf)
    print(f"  Return: {r['return']:+.2f}% | CAGR: {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe: {r.get('sharpe',0):.2f} | DD: -{r.get('dd',0):.2f}%")
    print(f"  PF: {r.get('pf',0):.2f} | WR: {r.get('wr',0):.1f}%")
    print(f"  Trades: {r['trades']} | SL/TP: {r['sl']}/{r['tp']}")
    print(f"  Best/Worst: +{r.get('best',0):.2f}% / {r.get('worst',0):.2f}%")
    print()
