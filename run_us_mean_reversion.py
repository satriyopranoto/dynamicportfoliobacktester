"""
Dynamic Portfolio Backtester — US Market (Pure Mean Reversion)
RSI < 30 + Close < Lower BB → Buy
RSI > 70 + Close > Upper BB → Sell
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
PERIOD = "7y"
SL_MULTIPLE = 2.8
SL_PERIOD = 10
ADX_PERIOD = 14
BB_PERIOD = 20
TP_RATIO = 0.4
RSI_ENTRY = 30
RSI_EXIT = 70
MIN_VOLUME = 0

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

tickers = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","XOM","PG","KO","PEP","HD","DIS","NFLX","MA",
    "UNH","BAC","ABBV","PFE","TMO","AVGO","CVX","LLY","COST",
    "MRK","ABT","ACN","DHR","LIN","NKE","WFC","TXN","QCOM","UPS",
    "RTX","LOW","SPGI","INTU","GS","MS","C","BLK","SCHW","PLD",
]

def calc_indicators(close, high, low, volume):
    n = len(close)
    sma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    ero = int(SL_MULTIPLE * SL_PERIOD)
    tr_arr = np.maximum(high - low,
        np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr_arr).rolling(SL_PERIOD).mean().values
    highest_high = pd.Series(high).rolling(ero).max().values
    sl = highest_high - 2 * atr * (SL_MULTIPLE - 1) / SL_MULTIPLE
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return {'sma20': sma20, 'sl': sl, 'upper_bb': upper_bb, 'lower_bb': lower_bb, 'rsi': rsi}

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
                'upper_bb': ind['upper_bb'], 'lower_bb': ind['lower_bb'], 'rsi': ind['rsi']}
    except Exception:
        return None

def check_buy_signal(stock, bar_idx):
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    close = float(stock['close'][bar_idx])
    low = float(stock['low'][bar_idx])
    lower_bb = float(stock['lower_bb'][bar_idx])
    rsi = float(stock['rsi'][bar_idx])
    if np.isnan(rsi) or np.isnan(lower_bb): return False
    # Mean reversion entry: oversold + price below lower BB
    if not (rsi < RSI_ENTRY): return False
    if not (close < lower_bb): return False
    return True

def calc_position_size(equity, close, sl_val):
    if np.isnan(sl_val) or sl_val <= 0:
        # Fallback: use 5% of close as stop distance
        stop_dist = close * 0.05
    else:
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
    total_signals = 0; sl_exits = 0; tp_exits = 0
    
    total_days = len(timeline)
    for day_idx, today in enumerate(timeline):
        if day_idx % 300 == 0:
            print(f"  Day {day_idx}/{total_days}... ({len(positions)} open, ${cash:,.0f} cash)")
        
        # EXITS
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(today)
            if bar is not None:
                close = float(s['close'][bar])
                high = float(s['high'][bar])
                sl_val = float(s['sl'][bar])
                rsi = float(s['rsi'][bar])
                upper_bb = float(s['upper_bb'][bar])
                
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
                
                # Mean reversion exit: overbought + above upper BB
                if not np.isnan(rsi) and not np.isnan(upper_bb):
                    if rsi > RSI_EXIT and high > upper_bb:
                        exit_val = pos['size'] * close
                        pnl = exit_val - pos['cost']
                        cash += exit_val
                        all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                            'exit_date': today.date(), 'entry_price': pos['entry_price'],
                            'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'MR_TP'})
                        tp_exits += 1; positions.pop(i); continue
                i += 1
        
        # ENTRIES
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker'] == ticker for p in positions): continue
                bar = stock_idx[ticker].get(today)
                if bar is None or bar < 20: continue
                if not check_buy_signal(s, bar): continue
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                candidates.append({'ticker': ticker, 'close': close, 'sl_val': sl_val, 'bar': bar})
            
            total_signals += len(candidates)
            for cand in candidates:
                if len(positions) >= MAX_POSITIONS: break
                remaining = cash * 0.95
                size = calc_position_size(cash + sum(p['size'] * float(p['close']) for p in positions), cand['close'], cand['sl_val'])
                if size == 0: continue
                cost = size * cand['close']
                if cost > remaining: continue
                positions.append({
                    'ticker': cand['ticker'], 'entry_price': cand['close'], 'entry_sl': cand['sl_val'],
                    'size': size, 'cost': cost, 'close': cand['close'], 'entry_date': today,
                    'tp_threshold': TP_RATIO * abs(cand['close'] - cand['sl_val']) / cand['close'] * 100
                })
                cash -= cost
        
        pos_value = sum(p['size'] * float(p['close']) for p in positions)
        equity_history.append(cash + pos_value)
        cash_history.append(cash)
        positions_history.append(len(positions))
    
    for pos in positions:
        s = stocks_dict[pos['ticker']]
        close = float(s['close'][-1])
        cash += pos['size'] * close
        all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
            'exit_date': s['dates'][-1].date(), 'entry_price': pos['entry_price'],
            'exit_price': close, 'size': pos['size'], 'pnl': pos['size'] * close - pos['cost'],
            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'END'})
    
    result = {}
    result['trades'] = pd.DataFrame(all_trades)
    result['equity'] = pd.Series(equity_history, index=timeline)
    result['timeline'] = timeline
    result['total_signals'] = total_signals
    result['sl_exits'] = sl_exits
    result['tp_exits'] = tp_exits
    final_eq = equity_history[-1] if equity_history else CAPITAL
    result['total_return_pct'] = (final_eq - CAPITAL) / CAPITAL * 100
    result['final_equity'] = final_eq
    
    if len(all_trades) > 0:
        df = pd.DataFrame(all_trades)
        winners = df[df['pnl'] > 0]; losers = df[df['pnl'] <= 0]
        result['win_rate'] = len(winners) / len(df) * 100
        result['total_trades'] = len(df)
        result['best_trade'] = df['return_pct'].max()
        result['worst_trade'] = df['return_pct'].min()
        result['profit_factor'] = abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else float('inf')
        eq_arr = np.array(equity_history)
        peak = np.maximum.accumulate(eq_arr)
        result['max_dd_pct'] = ((peak - eq_arr) / peak * 100).max()
        daily_ret = pd.Series(equity_history).pct_change().dropna()
        result['sharpe'] = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
        result['cagr'] = ((final_eq / CAPITAL) ** (1/7) - 1) * 100
    else:
        result['win_rate'] = 0; result['total_trades'] = 0
        result['best_trade'] = 0; result['worst_trade'] = 0
        result['profit_factor'] = 0; result['max_dd_pct'] = 0
        result['sharpe'] = 0; result['cagr'] = 0
    return result

# ══ MAIN ══
print("=" * 60)
print("  PURE MEAN REVERSION — US Market (49 stocks)")
print(f"  Entry: RSI<{RSI_ENTRY} + Close<LowerBB")
print(f"  Exit:  RSI>{RSI_EXIT} + High>UpperBB or SL")
print(f"  Capital: ${CAPITAL:,}, Max Pos: {MAX_POSITIONS}")
print("=" * 60)

print(f"\n  Loading {len(tickers)} US stocks...")
stocks = {}
for i, t in enumerate(tickers):
    s = load_stock(t)
    if s is not None: stocks[t] = s
    if (i+1) % 20 == 0: print(f"  [{i+1}/{len(tickers)}] ({len(stocks)} valid)")

print(f"  Successfully loaded: {len(stocks)} stocks\n")
result = run_dynamic_portfolio(stocks)

print("\n" + "=" * 60)
print("  RESULTS — PURE MEAN REVERSION")
print("=" * 60)
print(f"  Total Return : {result['total_return_pct']:+,.2f}%")
print(f"  Final Equity : ${result['final_equity']:,.2f}")
print(f"  CAGR         : {result['cagr']:+.2f}%/yr")
print(f"  Sharpe       : {result['sharpe']:.2f}")
print(f"  Max DD       : -{result['max_dd_pct']:.2f}%")
print(f"  Profit Factor: {result['profit_factor']:.2f}")
print(f"  Total Trades : {result['total_trades']}")
print(f"  Win Rate     : {result['win_rate']:.1f}%")
print(f"  Best Trade   : +{result['best_trade']:.2f}%")
print(f"  Worst Trade  : {result['worst_trade']:.2f}%")
print(f"  SL Exits     : {result['sl_exits']}")
print(f"  TP Exits     : {result['tp_exits']}")

if not result['trades'].empty:
    result['trades'].to_csv(os.path.join(OUTPUT_DIR, 'us_mean_rev_trades.csv'), index=False)
eq_df = pd.DataFrame({'Date': result['timeline'], 'Equity': result['equity']})
eq_df.to_csv(os.path.join(OUTPUT_DIR, 'us_mean_rev_equity.csv'), index=False)
print(f"\n  📁 Output saved")
