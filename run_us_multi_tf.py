"""
Dynamic Portfolio — US Market, 10 years.
Compare: Single TF (Daily) vs Multi TF (Daily+Weekly)
"""
import os, sys, csv, json, numpy as np, pandas as pd, yfinance as yf
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── CONFIG ──
CAPITAL = 6000
MAX_POSITIONS = 6
RISK_PCT = 1.0
COMMISSION = 0.001
PERIOD = "10y"
MIN_ADX = 20
MIN_TREND_SCORE = 0

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

tickers_list = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","XOM","PG","KO","PEP","HD","DIS","NFLX","MA",
    "UNH","BAC","ABBV","PFE","TMO","AVGO","CVX","LLY","COST",
    "MRK","ABT","ACN","DHR","LIN","NKE","WFC","TXN","QCOM","UPS",
    "RTX","LOW","SPGI","INTU","GS","MS","C","BLK","SCHW","PLD",
]

def calc_i(c, h, l):
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    up, dn = np.diff(h, prepend=h[0]), np.diff(l, prepend=l[0])
    pdm = np.where((up>dn)&(up>0), up, 0); mdm = np.where((dn>up)&(dn>0), dn, 0)
    atr = pd.Series(tr).rolling(14).mean().values
    sp = pd.Series(pdm).rolling(14).mean().values
    sm = pd.Series(mdm).rolling(14).mean().values
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    dx = np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), 0)
    adx = pd.Series(dx).rolling(14).mean().values
    return sma20, sl, adx, pdi, mdi

def calc_trend_score(close, sma20, adx, n=100):
    start = max(0, len(close) - n)
    total = 0; bull = 0
    for i in range(start, len(close)):
        if i < 20 or np.isnan(adx[i]) or np.isnan(sma20[i]): continue
        total += 1
        if float(adx[i]) > 25 and float(close[i]) > float(sma20[i]): bull += 1
    return round(bull / total * 100, 1) if total > 0 else 0

def load_stock(ticker, use_weekly=False):
    """Load daily stock data, optionally also load weekly for HTF."""
    try:
        df = yf.download(ticker, period=PERIOD, interval='1d', progress=False)
        if df.empty or len(df) < 250: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df['Close'].values.astype(float)
        h = df['High'].values.astype(float)
        l = df['Low'].values.astype(float)
        v = df['Volume'].values.astype(float) if 'Volume' in df.columns else np.zeros(len(df))
        
        # Daily indicators
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
            'volume': v, 'sma20': sma20, 'sl': sl,
            'adx': adx, 'pdi': pdi, 'mdi': mdi,
        }
        
        # Weekly HTF data
        if use_weekly:
            wk = yf.download(ticker, period=PERIOD, interval='1wk', progress=False)
            if wk.empty or len(wk) < 30:
                return None
            if isinstance(wk.columns, pd.MultiIndex):
                wk.columns = wk.columns.get_level_values(0)
            wk_c = wk['Close'].values.astype(float)
            wk_h = wk['High'].values.astype(float)
            wk_l = wk['Low'].values.astype(float)
            
            wk_sma20 = pd.Series(wk_c).rolling(20).mean().values
            tr_w = np.maximum(wk_h-wk_l, np.maximum(np.abs(wk_h-np.roll(wk_c,1)), np.abs(wk_l-np.roll(wk_c,1))))
            up_w, dn_w = np.diff(wk_h, prepend=wk_h[0]), np.diff(wk_l, prepend=wk_l[0])
            pdm_w = np.where((up_w>dn_w)&(up_w>0), up_w, 0); mdm_w = np.where((dn_w>up_w)&(dn_w>0), dn_w, 0)
            atr_w = pd.Series(tr_w).rolling(14).mean().values
            sp_w = pd.Series(pdm_w).rolling(14).mean().values
            sm_w = pd.Series(mdm_w).rolling(14).mean().values
            wk_pdi = np.where(atr_w>0, 100*sp_w/atr_w, 0); wk_mdi = np.where(atr_w>0, 100*sm_w/atr_w, 0)
            dx_w = np.where((wk_pdi+wk_mdi)>0, 100*np.abs(wk_pdi-wk_mdi)/(wk_pdi+wk_mdi), 0)
            wk_adx = pd.Series(dx_w).rolling(14).mean().values
            
            # Map weekly to daily
            wk_dates = wk.index
            wk_map = {}
            for i, wd in enumerate(wk_dates):
                wk_map[wd] = {'sma': wk_sma20[i], 'pdi': wk_pdi[i], 'mdi': wk_mdi[i]}
            
            def get_weekly(d):
                for wd in reversed(wk_dates):
                    if wd <= d:
                        return wk_map.get(wd, {})
                return {}
            
            daily_dates = df.index
            wk_sma_arr = np.full(len(daily_dates), np.nan)
            wk_pdi_arr = np.full(len(daily_dates), np.nan)
            wk_mdi_arr = np.full(len(daily_dates), np.nan)
            for i, d in enumerate(daily_dates):
                w = get_weekly(d)
                wk_sma_arr[i] = w.get('sma', np.nan)
                wk_pdi_arr[i] = w.get('pdi', np.nan)
                wk_mdi_arr[i] = w.get('mdi', np.nan)
            
            result['wk_sma'] = wk_sma_arr
            result['wk_pdi'] = wk_pdi_arr
            result['wk_mdi'] = wk_mdi_arr
        
        return result
    except Exception as e:
        print(f"  Error loading {ticker}: {e}")
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
    
    # Multi-TF: weekly confirmation
    if multi_tf:
        wk_sma = float(stock['wk_sma'][bar_idx]) if 'wk_sma' in stock and not np.isnan(stock['wk_sma'][bar_idx]) else None
        wk_pdi = float(stock['wk_pdi'][bar_idx]) if 'wk_pdi' in stock and not np.isnan(stock['wk_pdi'][bar_idx]) else None
        wk_mdi = float(stock['wk_mdi'][bar_idx]) if 'wk_mdi' in stock and not np.isnan(stock['wk_mdi'][bar_idx]) else None
        if wk_sma is None or wk_pdi is None or wk_mdi is None: return False
        if not (close > wk_sma and wk_pdi > wk_mdi): return False
    
    return True

def run_dynamic_portfolio(stocks_dict, multi_tf=False):
    name = "Multi TF" if multi_tf else "Single TF"
    print(f"\n  Running {name}...")
    all_dates = set()
    for ticker, s in stocks_dict.items():
        for d in s['dates']: all_dates.add(pd.Timestamp(d).normalize())
    timeline = sorted(all_dates)
    print(f"  Timeline: {len(timeline)} days | {timeline[0].date()} → {timeline[-1].date()}")
    
    stock_idx = {}
    for ticker, s in stocks_dict.items():
        idx_map = {}
        for i, d in enumerate(s['dates']): idx_map[pd.Timestamp(d).normalize()] = i
        stock_idx[ticker] = idx_map
    
    cash = float(CAPITAL)
    positions = []
    equity_history = []; all_trades = []
    total_signals = 0; sl_exits = 0; tp_exits = 0
    
    for day_idx, today in enumerate(timeline):
        if day_idx % 500 == 0:
            print(f"    Day {day_idx}/{len(timeline)}... ({len(positions)} open, ${cash:,.0f})")
        
        # Exits
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(today)
            if bar is not None:
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                if close < pos['entry_sl']:
                    cash += pos['size'] * close
                    all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
                        'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'SL'})
                    sl_exits += 1; positions.pop(i); continue
                floating = ((close-pos['entry_price'])/pos['entry_price'])*100
                if floating > pos['tp_threshold'] and close < sl_val:
                    cash += pos['size'] * close
                    all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
                        'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'TP'})
                    tp_exits += 1; positions.pop(i); continue
            i += 1
        
        # Entries
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker'] == ticker for p in positions): continue
                bar = stock_idx[ticker].get(today)
                if bar is None or bar < 20: continue
                if not check_buy_signal(s, bar, multi_tf): continue
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                score = calc_trend_score(s['close'], s['sma20'], s['adx'])
                if score < MIN_TREND_SCORE: continue
                candidates.append({'ticker':ticker,'score':score,'close':close,'sl_val':sl_val})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            total_signals += len(candidates)
            for cand in candidates:
                if len(positions) >= MAX_POSITIONS: break
                stop_dist = abs(cand['close']-cand['sl_val'])
                if stop_dist <= 0: continue
                risk = cash * (RISK_PCT/100.0)
                size = int(risk / stop_dist)
                max_by_cash = int((cash*0.95)/cand['close'])
                if max_by_cash <= 0: continue
                size = max(1, min(size, max_by_cash))
                cost = size * cand['close']
                if cost > cash * 0.95: continue
                positions.append({'ticker':cand['ticker'],'entry_price':cand['close'],
                    'entry_sl':cand['sl_val'],'size':size,'cost':cost,'close':cand['close'],'entry_date':today,
                    'tp_threshold':0.4*stop_dist/cand['close']*100})
                cash -= cost
        
        equity_history.append(cash + sum(p['size']*float(stocks_dict[p['ticker']]['close'][stock_idx[p['ticker']].get(today, -1)]) for p in positions if stock_idx[p['ticker']].get(today, -1) >= 0))
    
    for pos in positions:
        s = stocks_dict[pos['ticker']]
        close = float(s['close'][-1])
        cash += pos['size'] * close
        all_trades.append({'ticker':pos['ticker'],'pnl':pos['size']*close-pos['cost'],
            'return_pct':(close/pos['entry_price']-1)*100,'exit_reason':'END'})
    
    # Results
    final_eq = equity_history[-1] if equity_history else CAPITAL
    returns = {}
    returns['return_pct'] = (final_eq-CAPITAL)/CAPITAL*100
    returns['final_equity'] = final_eq
    returns['total_trades'] = len(all_trades)
    returns['sl_exits'] = sl_exits
    returns['tp_exits'] = tp_exits
    returns['total_signals'] = total_signals
    
    if len(all_trades) > 0:
        df = pd.DataFrame(all_trades)
        winners = df[df['pnl']>0]; losers = df[df['pnl']<=0]
        returns['win_rate'] = len(winners)/len(df)*100
        returns['best_trade'] = df['return_pct'].max()
        returns['worst_trade'] = df['return_pct'].min()
        returns['profit_factor'] = abs(winners['pnl'].sum()/losers['pnl'].sum()) if len(losers)>0 and losers['pnl'].sum()!=0 else float('inf')
        eq = np.array(equity_history)
        peak = np.maximum.accumulate(eq)
        returns['max_dd'] = ((peak-eq)/peak*100).max()
        daily_r = pd.Series(equity_history).pct_change().dropna()
        returns['sharpe'] = np.sqrt(252)*daily_r.mean()/daily_r.std() if daily_r.std()>0 else 0
        cagr_yrs = len(timeline)/252
        returns['cagr'] = ((final_eq/CAPITAL)**(1/cagr_yrs)-1)*100 if cagr_yrs>0 else 0
    
    return returns, equity_history, timeline, all_trades

# ══ MAIN ══
print("="*60)
print("  DYNAMIC PORTFOLIO — US MARKET (10 Years)")
print("  Basis+ADX: Single TF vs Multi TF (Daily+Weekly)")
print(f"  Capital: ${CAPITAL:,}, Max Pos: {MAX_POSITIONS}")
print("="*60)

print(f"\nLoading {len(tickers_list)} US stocks...")
stocks = {}
for i, t in enumerate(tickers_list):
    s = load_stock(t, use_weekly=True)
    if s is not None: stocks[t] = s
    if (i+1)%20==0: print(f"  [{i+1}/{len(tickers_list)}] ({len(stocks)} valid)")

print(f"\nSuccessfully loaded: {len(stocks)} stocks")

for multi_tf in [False, True]:
    label = "Multi TF (Daily+Weekly)" if multi_tf else "Single TF (Daily only)"
    r, eq, tl, trades = run_dynamic_portfolio(stocks, multi_tf)
    
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Return       : {r['return_pct']:+.2f}%")
    print(f"  CAGR         : {r.get('cagr',0):+.2f}%/yr")
    print(f"  Sharpe       : {r.get('sharpe',0):.2f}")
    print(f"  Max DD       : -{r.get('max_dd',0):.2f}%")
    print(f"  Profit Factor: {r.get('profit_factor',0):.2f}")
    print(f"  Trades       : {r['total_trades']}")
    print(f"  Win Rate     : {r.get('win_rate',0):.1f}%")
    print(f"  Best/Worst   : +{r.get('best_trade',0):.2f}% / {r.get('worst_trade',0):.2f}%")

print("\nDone!")
