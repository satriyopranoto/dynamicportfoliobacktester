"""
Dynamic Portfolio Backtester — ID Market (Rp 100jt), Hybrid Strategy
Basis+ADX entry + Mean Rev ADX exit. 116 liquid IDX stocks, 7 years.
"""
import os, sys, csv, json, numpy as np, pandas as pd, yfinance as yf
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────
CAPITAL = 100_000_000        # Rp 100 juta
MAX_POSITIONS = 10
RISK_PCT = 1.0
COMMISSION = 0.001
PERIOD = "7y"
SL_MULTIPLE = 2.8
SL_PERIOD = 10
ADX_PERIOD = 14
BB_PERIOD = 20
TP_RATIO = 0.4                # 0.4R minimum before hybrid exit logic
MIN_ADX = 20                  # ADX threshold for entry
MIN_TREND_SCORE = 0           # Minimum trend score (0 = no filter)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def calc_indicators(close, high, low, volume):
    n = len(close)
    sma20 = pd.Series(close).rolling(20).mean().values
    ero = int(SL_MULTIPLE * SL_PERIOD)
    tr_arr = np.maximum(high - low,
        np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr_arr).rolling(SL_PERIOD).mean().values
    highest_high = pd.Series(high).rolling(ero).max().values
    lowest_low = pd.Series(low).rolling(ero).min().values
    sl = highest_high - 2 * atr * (SL_MULTIPLE - 1) / SL_MULTIPLE
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_smooth = pd.Series(tr_arr).rolling(ADX_PERIOD).mean().values
    sp = pd.Series(plus_dm).rolling(ADX_PERIOD).mean().values
    sm = pd.Series(minus_dm).rolling(ADX_PERIOD).mean().values
    pdi = np.where(atr_smooth > 0, 100 * sp / atr_smooth, np.nan)
    mdi = np.where(atr_smooth > 0, 100 * sm / atr_smooth, np.nan)
    dm_sum = pdi + mdi
    dx = np.where(dm_sum > 0, 100 * np.abs(pdi - mdi) / dm_sum, np.nan)
    adx = pd.Series(dx).rolling(ADX_PERIOD).mean().values
    
    return {'sma20': sma20, 'sl': sl, 'adx': adx, 'pdi': pdi, 'mdi': mdi}

def calc_trend_score(close, sma20, adx, n_bars=100):
    start = max(0, len(close) - n_bars)
    total = 0; bull = 0
    for i in range(start, len(close)):
        if i < 20 or np.isnan(adx[i]) or np.isnan(sma20[i]): continue
        total += 1
        if float(adx[i]) > 25 and float(close[i]) > float(sma20[i]): bull += 1
    return round(bull / total * 100, 1) if total > 0 else 0

def load_stock(ticker):
    try:
        df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(len(df))
        ind = calc_indicators(c, h, l, v)
        return {'ticker': ticker, 'dates': df.index, 'close': c, 'high': h, 'low': l,
                'volume': v, 'sma20': ind['sma20'], 'sl': ind['sl'],
                'adx': ind['adx'], 'pdi': ind['pdi'], 'mdi': ind['mdi']}
    except Exception:
        return None

def check_buy_signal(stock, bar_idx):
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    if np.isnan(stock['adx'][bar_idx]) or np.isnan(stock['pdi'][bar_idx]) or np.isnan(stock['mdi'][bar_idx]): return False
    if np.isnan(stock['sma20'][bar_idx]) or np.isnan(stock['sl'][bar_idx]): return False
    close = float(stock['close'][bar_idx])
    low = float(stock['low'][bar_idx])
    sma20 = float(stock['sma20'][bar_idx])
    sl_val = float(stock['sl'][bar_idx])
    adx = float(stock['adx'][bar_idx])
    pdi = float(stock['pdi'][bar_idx])
    mdi = float(stock['mdi'][bar_idx])
    if not (low > sl_val and close > sma20): return False
    if not (adx > MIN_ADX and not np.isnan(adx)): return False
    if not (pdi > mdi): return False
    if bar_idx >= 5:
        pdi_5ago = float(stock['pdi'][bar_idx - 5])
        if not (pdi > pdi_5ago): return False
    return True

def check_hybrid_exit(stock, bar_idx, entry_price):
    """Mean Rev ADX exit conditions after minimum 0.4R profit."""
    if bar_idx < 5 or bar_idx >= len(stock['close']): return False
    close = float(stock['close'][bar_idx])
    adx = float(stock['adx'][bar_idx])
    pdi = float(stock['pdi'][bar_idx])
    mdi = float(stock['mdi'][bar_idx])
    adx_5 = float(stock['adx'][bar_idx - 5])
    pdi_5 = float(stock['pdi'][bar_idx - 5])
    if np.isnan(adx) or np.isnan(pdi) or np.isnan(mdi): return False
    # Hybrid exit: ADX declining, PDI fading, MDI overtaking PDI
    if not (adx < 25): return False           # ADX low (no trend)
    if not (adx < adx_5): return False         # ADX declining
    if not (pdi < pdi_5): return False         # PDI fading (bullish momentum fading)
    if not (mdi > pdi): return False           # MDI > PDI (bearish cross)
    return True

def calc_position_size(equity, close, sl_val):
    stop_dist = abs(close - sl_val)
    if stop_dist <= 0: return 0
    risk_amount = equity * (RISK_PCT / 100.0)
    size = int(risk_amount / stop_dist)
    max_by_cash = int((equity * 0.95) / close)
    if max_by_cash <= 0: return 0
    size = max(1, min(size, max_by_cash))
    return size

def run_dynamic_portfolio(stocks_dict):
    print(f"  Stocks loaded: {len(stocks_dict)}")
    all_dates = set()
    for ticker, s in stocks_dict.items():
        for d in s['dates']: all_dates.add(pd.Timestamp(d).normalize())
    timeline = sorted(all_dates)
    print(f"  Timeline: {len(timeline)} days ({timeline[0].date()} → {timeline[-1].date()})")
    
    stock_idx = {}
    for ticker, s in stocks_dict.items():
        idx_map = {}
        for i, d in enumerate(s['dates']): idx_map[pd.Timestamp(d).normalize()] = i
        stock_idx[ticker] = idx_map
    
    cash = float(CAPITAL)
    positions = []
    equity_history = []; cash_history = []; positions_history = []
    all_trades = []
    total_signals = 0; total_entries = 0; total_skipped = 0; sl_exits = 0; tp_exits = 0
    
    total_days = len(timeline)
    for day_idx, today in enumerate(timeline):
        if day_idx % 200 == 0:
            print(f"  Day {day_idx}/{total_days}... ({len(positions)} open, Rp{cash:,.0f} cash)")
        
        # ── EXITS ──
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(today)
            if bar is not None:
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                
                # SL exit
                if close < pos['entry_sl']:
                    exit_val = pos['size'] * close
                    pnl = exit_val - pos['cost']
                    cash += exit_val
                    all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                        'exit_date': today.date(), 'entry_price': pos['entry_price'],
                        'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                        'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'SL'})
                    sl_exits += 1; positions.pop(i); continue
                
                # Hybrid TP exit: min 0.4R profit + trend exhaustion
                floating = ((close - pos['entry_price']) / pos['entry_price']) * 100
                if floating > pos['tp_threshold']:
                    if check_hybrid_exit(s, bar, pos['entry_price']):
                        exit_val = pos['size'] * close
                        pnl = exit_val - pos['cost']
                        cash += exit_val
                        all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                            'exit_date': today.date(), 'entry_price': pos['entry_price'],
                            'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'HYBRID_TP'})
                        tp_exits += 1; positions.pop(i); continue
            i += 1
        
        # ── ENTRIES ──
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker'] == ticker for p in positions): continue
                bar = stock_idx[ticker].get(today)
                if bar is None or bar < 20: continue
                if not check_buy_signal(s, bar): continue
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                score = calc_trend_score(s['close'], s['sma20'], s['adx'])
                if score < MIN_TREND_SCORE: continue
                stop_dist = abs(close - sl_val)
                r_multiple = stop_dist / close if close > 0 else 0
                candidates.append({'ticker': ticker, 'score': score, 'close': close, 'sl_val': sl_val,
                    'stop_dist': stop_dist, 'r_multiple': r_multiple, 'bar': bar})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            total_signals += len(candidates)
            
            for cand in candidates:
                if len(positions) >= MAX_POSITIONS: break
                remaining = cash * 0.95
                size = calc_position_size(cash + sum(p['size'] * float(p['close']) for p in positions), cand['close'], cand['sl_val'])
                if size == 0: continue
                cost = size * cand['close']
                if cost > remaining: continue
                total_entries += 1
                positions.append({
                    'ticker': cand['ticker'], 'entry_price': cand['close'], 'entry_sl': cand['sl_val'],
                    'size': size, 'cost': cost, 'close': cand['close'],
                    'entry_date': today,
                    'tp_threshold': TP_RATIO * cand['stop_dist'] / cand['close'] * 100
                })
                cash -= cost
        
        # Record equity
        pos_value = sum(p['size'] * float(p['close']) for p in positions)
        equity = cash + pos_value
        equity_history.append(equity)
        cash_history.append(cash)
        positions_history.append(len(positions))
    
    # Close remaining positions at last price
    for pos in positions:
        s = stocks_dict[pos['ticker']]
        close = float(s['close'][-1])
        exit_val = pos['size'] * close
        cash += exit_val
        all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
            'exit_date': s['dates'][-1].date(), 'entry_price': pos['entry_price'],
            'exit_price': close, 'size': pos['size'], 'pnl': exit_val - pos['cost'],
            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'END'})
    
    # Results
    result = {}
    result['trades'] = pd.DataFrame(all_trades)
    result['equity'] = pd.Series(equity_history, index=timeline)
    result['cash'] = pd.Series(cash_history, index=timeline)
    result['positions'] = pd.Series(positions_history, index=timeline)
    result['total_signals'] = total_signals
    result['total_entries'] = total_entries
    result['sl_exits'] = sl_exits
    result['tp_exits'] = tp_exits
    
    final_equity = equity_history[-1] if equity_history else CAPITAL
    total_return = ((final_equity - CAPITAL) / CAPITAL) * 100
    result['total_return_pct'] = total_return
    
    if len(all_trades) > 0:
        df = pd.DataFrame(all_trades)
        winners = df[df['pnl'] > 0]
        losers = df[df['pnl'] <= 0]
        result['win_rate'] = len(winners) / len(df) * 100 if len(df) > 0 else 0
        result['total_trades'] = len(df)
        result['best_trade'] = df['return_pct'].max()
        result['worst_trade'] = df['return_pct'].min()
        result['profit_factor'] = abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf')
        # Max drawdown
        eq = np.array(equity_history)
        peak = np.maximum.accumulate(eq)
        dd = ((peak - eq) / peak) * 100
        result['max_dd_pct'] = dd.max()
        # Sharpe (daily)
        daily_returns = pd.Series(equity_history).pct_change().dropna()
        result['sharpe'] = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    else:
        result['win_rate'] = 0; result['total_trades'] = 0
        result['best_trade'] = 0; result['worst_trade'] = 0
        result['profit_factor'] = 0; result['max_dd_pct'] = 0; result['sharpe'] = 0
    
    return result

# ════════════ MAIN ════════════
print("=" * 60)
print("  HYBRID DYNAMIC PORTFOLIO — ID Market (116 stocks)")
print("  Basis+ADX Entry + Mean Rev ADX Exit")
print(f"  Capital: Rp{CAPITAL:,.0f}, Max Positions: {MAX_POSITIONS}")
print("=" * 60)

csv_path = os.path.join(OUTPUT_DIR, 'id_liquid.csv')
with open(csv_path) as f:
    reader = csv.DictReader(f)
    raw_tickers = [row['Symbol'].strip() for row in reader]

# Remove any .JK if already present, add .JK
tickers = []
for t in raw_tickers:
    t = t.upper().replace('.JK', '') + '.JK'
    tickers.append(t)

print(f"\nTickers loaded: {len(tickers)}")
print(f"Sample: {tickers[:5]}")

# Load all stocks
print("\nLoading stocks from Yahoo Finance...")
stocks = {}
for i, t in enumerate(tickers):
    if i % 20 == 0:
        print(f"  [{i}/{len(tickers)}] Loading...")
    s = load_stock(t)
    if s is not None:
        stocks[t] = s

print(f"\nStocks successfully loaded: {len(stocks)}/{len(tickers)}")

# Run portfolio simulation
print("\nRunning dynamic portfolio simulation...")
result = run_dynamic_portfolio(stocks)

# Print results
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Total Return : {result['total_return_pct']:+,.2f}%")
print(f"  Final Equity : Rp{result['equity'].iloc[-1]:,.0f}")
print(f"  CAGR         : {((result['equity'].iloc[-1]/CAPITAL)**(1/7)-1)*100:+.2f}%/yr")
print(f"  Sharpe       : {result['sharpe']:.2f}")
print(f"  Max DD       : -{result['max_dd_pct']:.2f}%")
print(f"  Profit Factor: {result['profit_factor']:.2f}")
print(f"  Total Trades : {result['total_trades']}")
print(f"  Total Signals: {result['total_signals']}")
print(f"  Win Rate     : {result['win_rate']:.1f}%")
print(f"  Best Trade   : +{result['best_trade']:.2f}%")
print(f"  Worst Trade  : {result['worst_trade']:.2f}%")
print(f"  SL Exits     : {result['sl_exits']}")
print(f"  TP Exits     : {result['tp_exits']}")
print()

# Save results
result['trades'].to_csv(os.path.join(OUTPUT_DIR, 'id_hybrid_trades.csv'), index=False)
result['equity'].to_csv(os.path.join(OUTPUT_DIR, 'id_hybrid_equity.csv'))
print("Trades saved to id_hybrid_trades.csv")
print("Equity saved to id_hybrid_equity.csv")

# Summary stats
print("\n" + "=" * 60)
print("  TRADE STATS")
print("=" * 60)
if len(result['trades']) > 0:
    by_reason = result['trades'].groupby('exit_reason').agg(
        count=('pnl', 'count'), avg_return=('return_pct', 'mean'),
        total_pnl=('pnl', 'sum'))
    print(by_reason)
print("\nDone!")
