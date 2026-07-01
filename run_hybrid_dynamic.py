"""
Dynamic Portfolio — Hybrid Strategy (Basis+ADX entry + Mean Rev ADX exit)
AmiBroker-style rolling stock selection on US market.
"""
import os, sys, csv, json, numpy as np, pandas as pd, yfinance as yf
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

sys.path.insert(0, r"C:\Users\Acer\code\backtester")

# ── CONFIG ──
CAPITAL = 100000
MAX_POSITIONS = 6
RISK_PCT = 1.0
COMMISSION = 0.001
PERIOD = "5y"
SL_MULTIPLE = 2.8
SL_PERIOD = 10
ADX_PERIOD = 14
BB_PERIOD = 20
TP_RATIO = 0.4
MIN_ADX = 20
MIN_TREND_SCORE = 30

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
                'volume': v, **ind}
    except Exception:
        return None

def check_buy_signal(stock, bar_idx):
    """Basis+ADX entry conditions."""
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    for k in ['adx','pdi','mdi','sma20','sl']:
        if np.isnan(stock[k][bar_idx]): return False
    close = float(stock['close'][bar_idx])
    low = float(stock['low'][bar_idx])
    sma20 = float(stock['sma20'][bar_idx])
    sl_val = float(stock['sl'][bar_idx])
    adx = float(stock['adx'][bar_idx])
    pdi = float(stock['pdi'][bar_idx])
    mdi = float(stock['mdi'][bar_idx])
    if bar_idx >= 5:
        pdi_5ago = float(stock['pdi'][bar_idx - 5])
        if not (pdi > pdi_5ago): return False
    return (low > sl_val and close > sma20 and
            adx > MIN_ADX and not np.isnan(adx) and
            pdi > mdi)

def check_hybrid_exit(stock, bar_idx, entry_price, entry_sl, tp_threshold):
    """Mean Rev ADX exit — trend exhaustion check."""
    if bar_idx >= len(stock['close']): return None
    close = float(stock['close'][bar_idx])
    low = float(stock['low'][bar_idx])
    
    # Cut loss
    if low < entry_sl:
        return 'SL'
    
    # Take profit: 0.4R minimum + trend exhaustion
    if entry_price > 0 and tp_threshold is not None:
        floating = ((close - entry_price) / entry_price) * 100
        if floating > tp_threshold:
            adx = float(stock['adx'][bar_idx])
            pdi = float(stock['pdi'][bar_idx])
            mdi = float(stock['mdi'][bar_idx])
            if bar_idx >= 5:
                adx_5ago = float(stock['adx'][bar_idx - 5])
                pdi_5ago = float(stock['pdi'][bar_idx - 5])
                if (adx < 25 and adx < adx_5ago and
                    pdi < pdi_5ago and mdi > pdi):
                    return 'TP'
    return None

def run_dynamic_portfolio(stocks_dict):
    print(f"  Stocks loaded: {len(stocks_dict)}")
    all_dates = set()
    for s in stocks_dict.values():
        for d in s['dates']: all_dates.add(pd.Timestamp(d).normalize())
    timeline = sorted(all_dates)
    print(f"  Timeline: {len(timeline)} days ({timeline[0].date()} → {timeline[-1].date()})")
    
    stock_idx = {}
    for ticker, s in stocks_dict.items():
        idx_map = {}
        for i, d in enumerate(s['dates']): idx_map[pd.Timestamp(d).normalize()] = i
        stock_idx[ticker] = idx_map
    
    cash = float(CAPITAL)
    positions = []  # {ticker, entry_price, entry_sl, size, cost, tp_threshold, entry_date, entry_score}
    equity_history = []; cash_history = []; positions_history = []
    all_trades = []
    total_signals = 0; total_entries = 0; total_skipped = 0; sl_exits = 0; tp_exits = 0
    
    for day_idx, today in enumerate(timeline):
        if day_idx % 200 == 0:
            print(f"  Day {day_idx}/{len(timeline)}... ({len(positions)} open, ${cash:,.0f} cash)")
        
        # ── EXITS (Hybrid logic) ──
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(today)
            if bar is not None:
                reason = check_hybrid_exit(
                    s, bar, pos['entry_price'], pos['entry_sl'], pos['tp_threshold'])
                if reason:
                    close = float(s['close'][bar])
                    exit_val = pos['size'] * close
                    pnl = exit_val - pos['cost']
                    cash += exit_val
                    all_trades.append({
                        'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                        'exit_date': today.date(), 'entry_price': pos['entry_price'],
                        'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                        'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': reason})
                    if reason == 'SL': sl_exits += 1
                    else: tp_exits += 1
                    positions.pop(i); continue
            i += 1
        
        # ── NEW SIGNALS ──
        if len(positions) < MAX_POSITIONS:
            new_signals = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker'] == ticker for p in positions): continue
                bar = stock_idx[ticker].get(today)
                if bar is None or bar < 30: continue
                if check_buy_signal(s, bar):
                    close = float(s['close'][bar]); sl_val = float(s['sl'][bar])
                    score = calc_trend_score(s['close'], s['sma20'], s['adx'], 100)
                    if score < MIN_TREND_SCORE: continue
                    stop_dist = abs(close - sl_val)
                    if stop_dist <= 0: continue
                    risk_amount = cash + sum(p['cost'] for p in positions)
                    size = int(risk_amount * (RISK_PCT/100) / stop_dist)
                    max_cash = int((risk_amount * 0.95) / close)
                    if max_cash <= 0: continue
                    size = max(1, min(size, max_cash))
                    cost = size * close
                    if size > 0:
                        new_signals.append({'ticker': ticker, 'score': score,
                            'close': close, 'sl': sl_val, 'size': size, 'cost': cost, 'bar': bar})
            total_signals += len(new_signals)
            new_signals.sort(key=lambda x: x['score'], reverse=True)
            for sig in new_signals:
                if len(positions) >= MAX_POSITIONS: break
                if cash >= sig['cost']:
                    stop_dist = abs(sig['close'] - sig['sl'])
                    tp_th = (stop_dist / sig['close']) * 100 * TP_RATIO if stop_dist > 0 else None
                    positions.append({
                        'ticker': sig['ticker'], 'entry_price': sig['close'],
                        'entry_sl': sig['sl'], 'size': sig['size'], 'cost': sig['cost'],
                        'tp_threshold': tp_th, 'entry_date': today, 'entry_score': sig['score']})
                    cash -= sig['cost']; total_entries += 1
                else: total_skipped += 1
        
        invested = sum(p['cost'] for p in positions)
        equity_history.append(cash + invested)
        cash_history.append(cash)
        positions_history.append(len(positions))
    
    # ── RESULTS ──
    final_equity = equity_history[-1]
    total_return = (final_equity / CAPITAL - 1) * 100
    days_elapsed = max((timeline[-1] - timeline[0]).days, 1)
    cagr = ((final_equity / CAPITAL) ** (365.0 / days_elapsed) - 1) * 100
    eq_series = pd.Series(equity_history)
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak * 100
    max_dd = dd.min()
    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) > 0:
        n_win = (trades_df['pnl'] > 0).sum()
        wr = n_win / len(trades_df) * 100
        gp = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gl = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')
        aw = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if n_win > 0 else 0
        al = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df) > n_win else 0
    else: wr = 0; pf = 0; aw = 0; al = 0
    dr = pd.Series(equity_history).pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    
    return {
        'equity': equity_history, 'cash': cash_history, 'positions_count': positions_history,
        'timeline': timeline, 'total_return': total_return, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'profit_factor': pf, 'total_trades': len(all_trades), 'win_rate': wr,
        'avg_win': aw, 'avg_loss': al, 'max_concurrent': max(positions_history) if positions_history else 0,
        'total_signals': total_signals, 'total_entries': total_entries, 'total_skipped': total_skipped,
        'sl_exits': sl_exits, 'tp_exits': tp_exits, 'trades': trades_df}

def print_results(r, capital):
    print(f"\n{'='*65}")
    print("  DYNAMIC PORTFOLIO — HYBRID STRATEGY")
    print("  (Entry: Basis+ADX || Exit: Mean Rev ADX)")
    print(f"{'='*65}")
    print(f"  Capital       : ${capital:,.0f}")
    print(f"  Final Equity  : ${r['equity'][-1]:,.0f}")
    print(f"  Total Return  : {r['total_return']:+.2f}%")
    print(f"  CAGR          : {r['cagr']:+.2f}%")
    print(f"  Max Drawdown  : {r['max_dd']:.2f}%")
    print(f"  Sharpe Ratio  : {r['sharpe']:.2f}")
    print(f"  Profit Factor : {r['profit_factor']:.2f}")
    print(f"  Total Trades  : {r['total_trades']}")
    print(f"  Win Rate      : {r['win_rate']:.1f}%")
    print(f"  Max Positions : {MAX_POSITIONS}")
    print(f"  Total Signals : {r['total_signals']}")
    print(f"  Total Entries : {r['total_entries']}")
    print(f"  Exits SL/TP   : {r['sl_exits']} / {r['tp_exits']}")
    print(f"{'='*65}")

if __name__ == '__main__':
    print("="*65)
    print("  DYNAMIC PORTFOLIO — HYBRID (US Market)")
    print("  Rolling stock selection with Basis+ADX entry")
    print("  and Mean Rev ADX (trend exhaustion) exit")
    print("="*65)
    
    # Top 50 US stocks by market cap (simplified list)
    tickers = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
        "JNJ","WMT","XOM","PG","KO","PEP","HD","DIS","NFLX","MA",
        "UNH","HD","BAC","ABBV","PFE","TMO","AVGO","CVX","LLY","COST",
        "MRK","ABT","ACN","DHR","LIN","NKE","WFC","TXN","QCOM","UPS",
        "RTX","LOW","SPGI","INTU","GS","MS","C","BLK","SCHW","PLD",
    ]
    
    print(f"\n  Loading {len(tickers)} US stocks...")
    stocks = {}
    for i, ticker in enumerate(tickers):
        result = load_stock(ticker)
        if result is not None:
            stocks[ticker] = result
        if (i+1) % 20 == 0:
            print(f"  Loaded {i+1}/{len(tickers)}... ({len(stocks)} valid)")
    
    print(f"  Successfully loaded: {len(stocks)} stocks")
    
    result = run_dynamic_portfolio(stocks)
    print_results(result, CAPITAL)
    
    # Save equity curve
    eq_df = pd.DataFrame({'Date': result['timeline'], 'Equity': result['equity'],
        'Cash': result['cash'], 'OpenPositions': result['positions_count']})
    eq_df.to_csv(os.path.join(OUTPUT_DIR, 'hybrid_equity.csv'), index=False)
    if not result['trades'].empty:
        result['trades'].to_csv(os.path.join(OUTPUT_DIR, 'hybrid_trades.csv'), index=False)
    
    # Plot chart
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import matplotlib.dates as mdates
        dates = result['timeline']; equity = result['equity']
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(14,9),gridspec_kw={'height_ratios':[3,1,1]})
        fig.patch.set_facecolor('#1a1612')
        ax1.plot(dates, equity, color='#d4af37', linewidth=1.5)
        ax1.axhline(y=CAPITAL, color='#666', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.fill_between(dates, equity, CAPITAL, where=[e>=CAPITAL for e in equity], alpha=0.1, color='#4ade80')
        ax1.fill_between(dates, equity, CAPITAL, where=[e<CAPITAL for e in equity], alpha=0.1, color='#f87171')
        for s in ax1.spines.values(): s.set_color('#333')
        ax1.tick_params(colors='#99907c'); ax1.set_ylabel('Equity ($)', color='#99907c')
        ax1.set_facecolor('#1a1612'); ax1.grid(True, alpha=0.08, color='#d4af37')
        ax1.set_title('Hybrid Dynamic Portfolio — Equity Curve', color='#d4af37', fontsize=14, fontweight='bold')
        eq_s = pd.Series(equity); peak = eq_s.expanding().max(); dd = (eq_s-peak)/peak*100
        ax2.fill_between(dates, 0, dd.values, color='#f87171', alpha=0.4)
        ax2.plot(dates, dd.values, color='#f87171', linewidth=0.8)
        for s in ax2.spines.values(): s.set_color('#333')
        ax2.tick_params(colors='#99907c'); ax2.set_ylabel('Drawdown %', color='#99907c')
        ax2.set_facecolor('#1a1612'); ax2.grid(True, alpha=0.08, color='#d4af37')
        ax3.fill_between(dates, 0, result['positions_count'], color='#60a5fa', alpha=0.3)
        ax3.plot(dates, result['positions_count'], color='#60a5fa', linewidth=0.8)
        ax3.axhline(y=MAX_POSITIONS, color='#d4af37', linestyle='--', linewidth=0.5, alpha=0.5)
        for s in ax3.spines.values(): s.set_color('#333')
        ax3.tick_params(colors='#99907c'); ax3.set_ylabel('Positions', color='#99907c')
        ax3.set_facecolor('#1a1612'); ax3.grid(True, alpha=0.08, color='#d4af37')
        for ax in [ax1,ax2,ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, 'hybrid_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1612'); plt.close()
        print(f"\n  🖼️  Chart saved: {chart_path}")
    except ImportError:
        pass
    
    print(f"\n  ✅ Done")
