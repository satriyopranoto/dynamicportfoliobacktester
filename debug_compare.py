"""
Debug: Compare scan vs main engine on identical data.
Runs BOTH and prints results side by side with detailed signal breakdown.
"""
import yfinance as yf, pandas as pd, numpy as np, csv

# Load tickers
tickers = []
with open('id_liquid.csv') as f:
    for row in csv.DictReader(f): tickers.append(row['Symbol'].strip())

# Download data ONCE
raw = {}
for t in tickers:
    df = yf.download(t, period='2mo', interval='1h', progress=False, auto_adjust=True)
    if df.empty or len(df) < 60: continue
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    raw[t] = df
print(f'Loaded {len(raw)} stocks')

CAPITAL = 100_000_000
MAX_POS = 10
RISK_PCT = 1.0

# ── SCAN ENGINE ──
def engine_scan(raw):
    cfg = {'adx_p':7,'sl_p':5,'bb_p':10,'sl_m':2.2,'min_adx':20,'pdi_bars':3}
    
    stocks = {}
    for t, df in raw.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        if np.mean(c * v) < 5_000_000_000: continue
        
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
        sma = pd.Series(c).rolling(10).mean().values
        sl = pd.Series(h).rolling(11).max().values - 2 * pd.Series(tr).rolling(5).mean().values * 1.2/2.2
        up = np.diff(h, prepend=h[0]); dn = np.diff(l, prepend=l[0])
        pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
        atr_s = pd.Series(tr).rolling(7).mean().values
        pdi = np.where(atr_s>0, 100*pd.Series(pdm).rolling(7).mean().values/atr_s, np.nan)
        mdi = np.where(atr_s>0, 100*pd.Series(mdm).rolling(7).mean().values/atr_s, np.nan)
        adx = pd.Series(np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), np.nan)).rolling(7).mean().values
        
        d = {'close':c,'high':h,'low':l,'volume':v,'sma':sma,'sl':sl,'adx':adx,'pdi':pdi,'mdi':mdi,'timestamps':df.index}
        stocks[t] = d
    print(f'  Scan: {len(stocks)} stocks')
    
    all_ts = set()
    for s in stocks.values():
        for ts in s['timestamps']: all_ts.add(pd.Timestamp(ts))
    timeline = sorted(all_ts)
    
    stock_idx = {}
    for t, s in stocks.items():
        idx_map = {}
        for i, ts in enumerate(s['timestamps']): idx_map[pd.Timestamp(ts)] = i
        stock_idx[t] = idx_map
    
    cash = float(CAPITAL); positions = []
    sig_total=0; entries=0; skipped=0; sl_exits=0; tp_exits=0
    
    for now in timeline:
        i = 0
        while i < len(positions):
            pos = positions[i]; s = stocks[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(now)
            if bar is not None:
                close = float(s['close'][bar])
                if close < pos['entry_sl']:
                    cash += pos['size']*close; sl_exits+=1; positions.pop(i); continue
                elif pos['tp_th'] is not None:
                    fl = (close-pos['entry_price'])/pos['entry_price']*100
                    if fl > pos['tp_th']:
                        cash += pos['size']*close; tp_exits+=1; positions.pop(i); continue
            i += 1
        
        if len(positions) < MAX_POS:
            sigs = []
            for t, s in stocks.items():
                if any(p['ticker']==t for p in positions): continue
                bar = stock_idx[t].get(now)
                if bar is None or bar < 30: continue
                if np.isnan(s['adx'][bar]) or np.isnan(s['pdi'][bar]) or np.isnan(s['mdi'][bar]): continue
                if np.isnan(s['sma'][bar]) or np.isnan(s['sl'][bar]): continue
                close=float(s['close'][bar]); lo=float(s['low'][bar])
                if not (lo>float(s['sl'][bar]) and close>float(s['sma'][bar])): continue
                if not (float(s['adx'][bar])>20): continue
                if not (float(s['pdi'][bar])>float(s['mdi'][bar])): continue
                if bar>=3 and not (float(s['pdi'][bar])>float(s['pdi'][bar-3])): continue
                
                slv = float(s['sl'][bar])
                sd = abs(close-slv)
                if sd <= 0: continue
                eq = cash + sum(p['cost'] for p in positions)
                size = int(eq*(RISK_PCT/100.0)/sd)
                max_sz = int(eq*0.95/close)
                if max_sz <= 0: continue
                size = max(1, min(size, max_sz))
                cost = size*close
                if size > 0:
                    sigs.append({'ticker':t,'close':close,'size':size,'cost':cost,'sl':slv,'tp_th':(sd/close)*100*0.4})
            sig_total += len(sigs)
            for sig in sigs:
                if len(positions)>=MAX_POS: break
                if cash >= sig['cost']:
                    positions.append({'ticker':sig['ticker'],'entry_price':sig['close'],'entry_sl':sig['sl'],'size':sig['size'],'cost':sig['cost'],'tp_th':sig['tp_th']})
                    cash -= sig['cost']; entries+=1
                else: skipped+=1
    
    fe = cash + sum(p['cost'] for p in positions)
    return (fe/CAPITAL-1)*100, sl_exits+tp_exits, sl_exits, tp_exits, sig_total, entries, skipped

# ── MAIN ENGINE (using run_id_intraday.py logic) ──
def calc_indicators_main(close, high, low, volume):
    SL_M=2.2; SL_P=5; ADX_P=7; BB_P=10
    sma = pd.Series(close).rolling(BB_P).mean().values
    ero = int(SL_M * SL_P)
    tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    atr = pd.Series(tr).rolling(SL_P).mean().values
    sl = pd.Series(high).rolling(ero).max().values - 2*atr*(SL_M-1)/SL_M
    up_move = np.diff(high, prepend=high[0]); down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move>down_move)&(up_move>0), up_move, 0)
    minus_dm = np.where((down_move>up_move)&(down_move>0), down_move, 0)
    atr_smooth = pd.Series(tr).rolling(ADX_P).mean().values
    sp = pd.Series(plus_dm).rolling(ADX_P).mean().values
    sm = pd.Series(minus_dm).rolling(ADX_P).mean().values
    pdi = np.where(atr_smooth>0, 100*sp/atr_smooth, np.nan)
    mdi = np.where(atr_smooth>0, 100*sm/atr_smooth, np.nan)
    dm_sum = pdi + mdi
    dx = np.where(dm_sum>0, 100*np.abs(pdi-mdi)/dm_sum, np.nan)
    adx = pd.Series(dx).rolling(ADX_P).mean().values
    return {'sma':sma,'sl':sl,'adx':adx,'pdi':pdi,'mdi':mdi}

def check_buy_signal(stock, bar_idx):
    MIN_ADX=20
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    if np.isnan(stock['adx'][bar_idx]) or np.isnan(stock['pdi'][bar_idx]) or np.isnan(stock['mdi'][bar_idx]): return False
    if np.isnan(stock['sma'][bar_idx]) or np.isnan(stock['sl'][bar_idx]): return False
    close=float(stock['close'][bar_idx]); low=float(stock['low'][bar_idx])
    sma=float(stock['sma'][bar_idx]); sl_val=float(stock['sl'][bar_idx])
    adx=float(stock['adx'][bar_idx]); pdi=float(stock['pdi'][bar_idx]); mdi=float(stock['mdi'][bar_idx])
    if not (low>sl_val and close>sma): return False
    if not (adx>MIN_ADX): return False
    if not (pdi>mdi): return False
    if bar_idx>=3:
        pdi_3ago = float(stock['pdi'][bar_idx-3])
        if not (pdi>pdi_3ago): return False
    return True

def calc_position_size(equity, close, sl_val):
    stop_dist = abs(close-sl_val)
    if stop_dist <= 0: return 0
    size = int(equity*(RISK_PCT/100.0)/stop_dist)
    max_sz = int(equity*0.95/close)
    if max_sz <= 0: return 0
    return max(1, min(size, max_sz))

def engine_main(raw):
    stocks = {}
    for t, df in raw.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        if np.mean(c * v) < 5_000_000_000: continue
        ind = calc_indicators_main(c, h, l, v)
        stocks[t] = {'close':c,'high':h,'low':l,'volume':v,'sma':ind['sma'],'sl':ind['sl'],'adx':ind['adx'],'pdi':ind['pdi'],'mdi':ind['mdi'],'timestamps':df.index}
    print(f'  Main: {len(stocks)} stocks')
    
    all_ts = set()
    for s in stocks.values():
        for ts in s['timestamps']: all_ts.add(pd.Timestamp(ts))
    timeline = sorted(all_ts)
    stock_idx = {}
    for t, s in stocks.items():
        idx_map = {}
        for i, ts in enumerate(s['timestamps']): idx_map[pd.Timestamp(ts)] = i
        stock_idx[t] = idx_map
    
    cash = float(CAPITAL); positions = []
    sig_total=0; entries=0; skipped=0; sl_exits=0; tp_exits=0
    
    for now in timeline:
        i = 0
        while i < len(positions):
            pos = positions[i]; s = stocks[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(now)
            if bar is not None:
                close = float(s['close'][bar])
                if close < pos['entry_sl']:
                    cash += pos['size']*close; sl_exits+=1; positions.pop(i); continue
                elif pos['tp_th'] is not None:
                    fl = (close-pos['entry_price'])/pos['entry_price']*100
                    if fl > pos['tp_th']:
                        cash += pos['size']*close; tp_exits+=1; positions.pop(i); continue
            i += 1
        
        if len(positions) < MAX_POS:
            sigs = []
            for t, s in stocks.items():
                if any(p['ticker']==t for p in positions): continue
                bar = stock_idx[t].get(now)
                if bar is None or bar < 30: continue
                if not check_buy_signal(s, bar): continue
                close = float(s['close'][bar]); sl_val = float(s['sl'][bar])
                eq = cash + sum(p['cost'] for p in positions)
                size = calc_position_size(eq, close, sl_val)
                cost = size*close
                if size > 0:
                    sd = abs(close-sl_val)
                    tp_th = (sd/close)*100*0.4 if sd>0 else None
                    sigs.append({'ticker':t,'close':close,'size':size,'cost':cost,'sl':sl_val,'tp_th':tp_th})
            sig_total += len(sigs)
            for sig in sigs:
                if len(positions)>=MAX_POS: break
                if cash >= sig['cost']:
                    positions.append({'ticker':sig['ticker'],'entry_price':sig['close'],'entry_sl':sig['sl'],'size':sig['size'],'cost':sig['cost'],'tp_th':sig['tp_th']})
                    cash -= sig['cost']; entries+=1
                else: skipped+=1
    
    fe = cash + sum(p['cost'] for p in positions)
    return (fe/CAPITAL-1)*100, sl_exits+tp_exits, sl_exits, tp_exits, sig_total, entries, skipped

# Run both
print()
print("=== SCAN ENGINE ===")
r1 = engine_scan(raw)
print(f"  Return={r1[0]:+.2f}% Trades={r1[1]} SL/TP={r1[2]}/{r1[3]} Signals={r1[4]} Entries={r1[5]} Skipped={r1[6]}")

print()
print("=== MAIN ENGINE ===")
r2 = engine_main(raw)
print(f"  Return={r2[0]:+.2f}% Trades={r2[1]} SL/TP={r2[2]}/{r2[3]} Signals={r2[4]} Entries={r2[5]} Skipped={r2[6]}")
