"""
Dynamic Portfolio Backtester — ID Market INTRADAY (1h)
Winning parameters: ADX=7, SL=5, SMA=10, SLx=2.2 (Aggressive variant)
+24.86% in ~35 days, Sharpe 3.46, PF 1.50
"""
import os, sys, csv, json
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────
# Winning params from scan: C_Aggressive(7/5/10)
CAPITAL = 100_000_000        # Rp
MAX_POSITIONS = 10
RISK_PCT = 1.0
COMMISSION = 0.001
PERIOD = "2mo"               # Intraday 1h max ~2-3mo dari Yahoo
INTERVAL = "1h"
SL_MULTIPLE = 2.2            # Tight trailing
SL_PERIOD = 5                # Faster SL (5 bars = ~half day)
ADX_PERIOD = 7               # Faster ADX detection (~1 day)
BB_PERIOD = 10               # Shorter SMA basis
TP_RATIO = 0.4               # Same 0.4R
MIN_ADX = 20                 # Same threshold
MIN_TREND_SCORE = 0
MIN_VOLUME = 5_000_000_000   # Min avg daily volume (Rp) — filter saham illiquid

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def calc_indicators(close, high, low, volume):
    """Calculate all needed indicators. Uses config BB_PERIOD for SMA."""
    n = len(close)
    # SMA (configurable via BB_PERIOD)
    sma = pd.Series(close).rolling(BB_PERIOD).mean().values
    
    # Donchian SL
    ero = int(SL_MULTIPLE * SL_PERIOD)
    tr = np.maximum(high - low,
        np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).rolling(SL_PERIOD).mean().values
    sl = pd.Series(high).rolling(ero).max().values - 2 * atr * (SL_MULTIPLE - 1) / SL_MULTIPLE
    
    # ADX
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_smooth = pd.Series(tr).rolling(ADX_PERIOD).mean().values
    sp = pd.Series(plus_dm).rolling(ADX_PERIOD).mean().values
    sm = pd.Series(minus_dm).rolling(ADX_PERIOD).mean().values
    pdi = np.where(atr_smooth > 0, 100 * sp / atr_smooth, np.nan)
    mdi = np.where(atr_smooth > 0, 100 * sm / atr_smooth, np.nan)
    dm_sum = pdi + mdi
    dx = np.where(dm_sum > 0, 100 * np.abs(pdi - mdi) / dm_sum, np.nan)
    adx = pd.Series(dx).rolling(ADX_PERIOD).mean().values
    
    return {'sma': sma, 'sl': sl, 'adx': adx, 'pdi': pdi, 'mdi': mdi}


def calc_trend_score(close, sma, adx, n_bars=100):
    """Calculate trend strength score from last N bars."""
    start = max(0, len(close) - n_bars)
    total = 0; bull = 0
    for i in range(start, len(close)):
        if i < 20 or np.isnan(adx[i]) or np.isnan(sma[i]): continue
        total += 1
        if float(adx[i]) > 25 and float(close[i]) > float(sma[i]): bull += 1
    return round(bull / total * 100, 1) if total > 0 else 0


def load_stock(ticker):
    """Download 1h intraday data and precompute indicators."""
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(len(df))
        
        ind = calc_indicators(c, h, l, v)
        
        return {'ticker': ticker, 'timestamps': df.index, 'close': c, 'high': h, 'low': l,
                'volume': v, 'sma': ind['sma'], 'sl': ind['sl'],
                'adx': ind['adx'], 'pdi': ind['pdi'], 'mdi': ind['mdi']}
    except Exception as e:
        return None


def check_buy_signal(stock, bar_idx):
    """Check entry conditions. Same logic as daily version."""
    if bar_idx < 20 or bar_idx >= len(stock['close']): return False
    if np.isnan(stock['adx'][bar_idx]) or np.isnan(stock['pdi'][bar_idx]) or np.isnan(stock['mdi'][bar_idx]): return False
    if np.isnan(stock['sma'][bar_idx]) or np.isnan(stock['sl'][bar_idx]): return False
    
    close = float(stock['close'][bar_idx])
    low = float(stock['low'][bar_idx])
    sma = float(stock['sma'][bar_idx])
    sl_val = float(stock['sl'][bar_idx])
    adx = float(stock['adx'][bar_idx])
    pdi = float(stock['pdi'][bar_idx])
    mdi = float(stock['mdi'][bar_idx])
    
    if not (low > sl_val and close > sma): return False
    if not (adx > MIN_ADX): return False
    if not (pdi > mdi): return False
    if bar_idx >= 3:
        pdi_3ago = float(stock['pdi'][bar_idx - 3])
        if not (pdi > pdi_3ago): return False
    return True


def calc_position_size(equity, close, sl_val):
    """Calculate position size based on risk %."""
    stop_dist = abs(close - sl_val)
    if stop_dist <= 0: return 0
    risk_amount = equity * (RISK_PCT / 100.0)
    size = int(risk_amount / stop_dist)
    max_by_cash = int((equity * 0.95) / close)
    if max_by_cash <= 0: return 0
    size = max(1, min(size, max_by_cash))
    return size


def run_dynamic_portfolio(stocks_dict):
    """Core dynamic portfolio simulation — intraday (hourly bars)."""
    print(f"  Stocks loaded: {len(stocks_dict)}")
    
    # Build unified timeline (all unique hourly timestamps)
    all_ts = set()
    for ticker, s in stocks_dict.items():
        for ts in s['timestamps']:
            all_ts.add(pd.Timestamp(ts))
    timeline = sorted(all_ts)
    print(f"  Timeline: {len(timeline)} bars ({timeline[0]} → {timeline[-1]})")
    
    # Map stock timestamps to bar indices
    stock_idx = {}
    for ticker, s in stocks_dict.items():
        idx_map = {}
        for i, ts in enumerate(s['timestamps']):
            idx_map[pd.Timestamp(ts)] = i
        stock_idx[ticker] = idx_map
    
    cash = float(CAPITAL)
    positions = []
    equity_history = []; cash_history = []; positions_history = []
    all_trades = []
    total_signals = 0; total_entries = 0; total_skipped = 0; sl_exits = 0; tp_exits = 0
    
    total_bars = len(timeline)
    for bar_idx, now in enumerate(timeline):
        if bar_idx % 100 == 0:
            print(f"  Bar {bar_idx}/{total_bars}... ({len(positions)} open, Rp {cash:,.0f} cash)")
        
        # ── EXITS ──
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks_dict[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(now)
            
            if bar is not None:
                close = float(s['close'][bar])
                sl_val = float(s['sl'][bar])
                
                # Stop loss
                if close < pos['entry_sl']:
                    exit_val = pos['size'] * close
                    pnl = exit_val - pos['cost']
                    cash += exit_val
                    all_trades.append({'ticker': pos['ticker'],
                        'entry_time': str(pos['entry_time']), 'exit_time': str(now),
                        'entry_price': pos['entry_price'], 'exit_price': close,
                        'size': pos['size'], 'pnl': pnl,
                        'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'SL'})
                    sl_exits += 1; positions.pop(i); continue
                
                # Take profit
                elif pos['tp_threshold'] is not None:
                    floating = ((close - pos['entry_price']) / pos['entry_price']) * 100
                    if floating > pos['tp_threshold']:
                        exit_val = pos['size'] * close
                        pnl = exit_val - pos['cost']
                        cash += exit_val
                        all_trades.append({'ticker': pos['ticker'],
                            'entry_time': str(pos['entry_time']), 'exit_time': str(now),
                            'entry_price': pos['entry_price'], 'exit_price': close,
                            'size': pos['size'], 'pnl': pnl,
                            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'TP'})
                        tp_exits += 1; positions.pop(i); continue
            i += 1
        
        # ── NEW SIGNALS ──
        if len(positions) < MAX_POSITIONS:
            new_signals = []
            for ticker, s in stocks_dict.items():
                if any(p['ticker'] == ticker for p in positions): continue
                bar = stock_idx[ticker].get(now)
                if bar is None or bar < 30: continue
                
                if check_buy_signal(s, bar):
                    close = float(s['close'][bar]); sl_val = float(s['sl'][bar])
                    size = calc_position_size(cash + sum(p['cost'] for p in positions), close, sl_val)
                    cost = size * close
                    if size > 0:
                        new_signals.append({'ticker': ticker, 'close': close,
                            'sl': sl_val, 'size': size, 'cost': cost, 'bar': bar})
            
            total_signals += len(new_signals)
            for sig in new_signals:
                if len(positions) >= MAX_POSITIONS: break
                if cash >= sig['cost']:
                    stop_dist = abs(sig['close'] - sig['sl'])
                    tp_th = (stop_dist / sig['close']) * 100 * TP_RATIO if stop_dist > 0 else None
                    positions.append({'ticker': sig['ticker'], 'entry_price': sig['close'],
                        'entry_sl': sig['sl'], 'size': sig['size'], 'cost': sig['cost'],
                        'tp_threshold': tp_th, 'entry_time': now})
                    cash -= sig['cost']; total_entries += 1
                else: total_skipped += 1
        
        invested = sum(p['cost'] for p in positions)
        equity_history.append(cash + invested)
        cash_history.append(cash)
        positions_history.append(len(positions))
    
    # ── RESULTS ──
    final_equity = equity_history[-1]
    total_return = (final_equity / CAPITAL - 1) * 100
    hours_elapsed = len(timeline)  # in hours
    days_elapsed = hours_elapsed / 7  # ~7 bars per trading day
    cagr = ((final_equity / CAPITAL) ** (365.0 / max(days_elapsed, 1)) - 1) * 100 if days_elapsed > 0 else 0
    
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
    sharpe = dr.mean() / dr.std() * np.sqrt(252 * 7) if dr.std() > 0 else 0  # ~7 bars/day
    
    return {'equity': equity_history, 'cash': cash_history, 'positions_count': positions_history,
        'timeline': timeline, 'total_return': total_return, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'profit_factor': pf, 'total_trades': len(all_trades), 'win_rate': wr,
        'avg_win': aw, 'avg_loss': al, 'max_concurrent': max(positions_history) if positions_history else 0,
        'total_signals': total_signals, 'total_entries': total_entries, 'total_skipped': total_skipped,
        'sl_exits': sl_exits, 'tp_exits': tp_exits, 'trades': trades_df}


def print_results(result, capital):
    print()
    print("=" * 65)
    print("  DYNAMIC PORTFOLIO — ID MARKET INTRADAY (1h)")
    print("  C_Aggressive: ADX=7 SL=5 SMA=10 SLx=2.2")
    print("=" * 65)
    print(f"  Period        : {str(result['timeline'][0])} → {str(result['timeline'][-1])}")
    print(f"  Total Bars    : {len(result['timeline'])} bars (~{len(result['timeline'])//7} days)")
    print(f"  Initial Capital: Rp {capital:,.0f}")
    print(f"  Final Equity   : Rp {result['equity'][-1]:,.0f}")
    print(f"  Total Return   : {result['total_return']:+.2f}%")
    print(f"  CAGR           : {result['cagr']:+.2f}%")
    print(f"  Max Drawdown   : {result['max_dd']:.2f}%")
    print(f"  Sharpe Ratio   : {result['sharpe']:.2f}")
    print(f"  Profit Factor  : {result['profit_factor']:.2f}")
    print(f"  Total Trades   : {result['total_trades']}")
    print(f"  Win Rate       : {result['win_rate']:.1f}%")
    print(f"  Max Concurrent : {result['max_concurrent']} / {MAX_POSITIONS}")
    print(f"  Risk/Trade     : {RISK_PCT}%")
    print(f"  TP Ratio       : {TP_RATIO}R")
    print(f"  Avg Win        : Rp {result['avg_win']:,.0f}")
    print(f"  Avg Loss       : Rp {result['avg_loss']:,.0f}")
    print(f"  Total Signals  : {result['total_signals']}")
    print(f"  Total Entries  : {result['total_entries']}")
    print(f"  Skipped (cash) : {result['total_skipped']}")
    print(f"  Exits (SL/TP)  : {result['sl_exits']} / {result['tp_exits']}")
    print("=" * 65)


def plot_results(result):
    """Save equity curve chart."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import matplotlib.dates as mdates
        
        dates = result['timeline']
        equity = result['equity']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9),
                                             gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.patch.set_facecolor('#1a1612')
        
        ax1.plot(dates, equity, color='#d4af37', linewidth=1.5, label='Equity')
        ax1.axhline(y=CAPITAL, color='#666', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.fill_between(dates, equity, CAPITAL, where=[e >= CAPITAL for e in equity],
                         alpha=0.1, color='#4ade80')
        ax1.fill_between(dates, equity, CAPITAL, where=[e < CAPITAL for e in equity],
                         alpha=0.1, color='#f87171')
        for spine in ax1.spines.values(): spine.set_color('#333')
        ax1.tick_params(colors='#99907c')
        ax1.set_ylabel('Equity (Rp)', color='#99907c')
        ax1.set_facecolor('#1a1612')
        ax1.grid(True, alpha=0.08, color='#d4af37')
        ax1.set_title('ID Intraday (1h) Dynamic Portfolio — Equity Curve', 
                      color='#d4af37', fontsize=14, fontweight='bold')
        
        eq_s = pd.Series(equity)
        peak = eq_s.expanding().max()
        dd = (eq_s - peak) / peak * 100
        ax2.fill_between(dates, 0, dd.values, color='#f87171', alpha=0.4)
        ax2.plot(dates, dd.values, color='#f87171', linewidth=0.8)
        for spine in ax2.spines.values(): spine.set_color('#333')
        ax2.tick_params(colors='#99907c')
        ax2.set_ylabel('Drawdown %', color='#99907c')
        ax2.set_facecolor('#1a1612')
        ax2.grid(True, alpha=0.08, color='#d4af37')
        
        ax3.fill_between(dates, 0, result['positions_count'], color='#60a5fa', alpha=0.3)
        ax3.plot(dates, result['positions_count'], color='#60a5fa', linewidth=0.8)
        ax3.axhline(y=MAX_POSITIONS, color='#d4af37', linestyle='--', linewidth=0.5, alpha=0.5)
        for spine in ax3.spines.values(): spine.set_color('#333')
        ax3.tick_params(colors='#99907c')
        ax3.set_ylabel('Open Positions', color='#99907c')
        ax3.set_xlabel('Time', color='#99907c')
        ax3.set_facecolor('#1a1612')
        ax3.grid(True, alpha=0.08, color='#d4af37')
        ax3.set_ylim(bottom=0)
        
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, 'dynamic_id_intraday_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1612')
        plt.close()
        return chart_path
    except ImportError: return None


if __name__ == '__main__':
    print("=" * 65)
    print("  DYNAMIC PORTFOLIO — ID MARKET INTRADAY (1h)")
    print("  C_Aggressive: ADX=7 SL=5 SMA=10 SLx=2.2")
    print("  +24.86% in scan — AmiBroker-style rolling stock selection")
    print("=" * 65)
    print(f"  Capital: Rp {CAPITAL:,}")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Risk/Trade: {RISK_PCT}%")
    print(f"  TP Ratio: {TP_RATIO}R")
    print(f"  Min ADX: {MIN_ADX}")
    print(f"  Period: {PERIOD} ({INTERVAL})")
    print()
    
    tickers = []
    csv_path = os.path.join(OUTPUT_DIR, 'id_liquid.csv')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader: tickers.append(row['Symbol'].strip())
    
    print(f"  Loading {len(tickers)} liquid ID stocks (1h intraday)...")
    stocks = {}
    skipped_low_volume = 0
    for i, ticker in enumerate(tickers):
        result = load_stock(ticker)
        if result is not None:
            avg_volume_rp = np.mean(result['close'] * result['volume'])
            if avg_volume_rp < MIN_VOLUME:
                skipped_low_volume += 1
                continue
            stocks[ticker] = result
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(tickers)}... ({len(stocks)} valid, {skipped_low_volume} low vol)")
    
    print(f"  Successfully loaded: {len(stocks)} stocks (skipped {skipped_low_volume} illiquid)")
    print()
    
    result = run_dynamic_portfolio(stocks)
    print_results(result, CAPITAL)
    
    chart_path = plot_results(result)
    if not result['trades'].empty:
        result['trades'].to_csv(os.path.join(OUTPUT_DIR, 'id_intraday_trades.csv'), index=False)
    eq_df = pd.DataFrame({'Timestamp': result['timeline'], 'Equity': result['equity'],
        'Cash': result['cash'], 'OpenPositions': result['positions_count']})
    eq_df.to_csv(os.path.join(OUTPUT_DIR, 'id_intraday_equity.csv'), index=False)
    report = {k: v for k, v in result.items() if k not in ['equity','cash','positions_count','timeline','trades']}
    with open(os.path.join(OUTPUT_DIR, 'id_intraday_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n  📁 Output saved")
    if chart_path: print(f"  🖼️  MEDIA:{chart_path}")
