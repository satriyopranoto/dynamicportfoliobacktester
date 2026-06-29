"""
Dynamic Portfolio Backtester — ID Market (Rp 100jt)
"""
import os, sys, csv, json, numpy as np, pandas as pd, yfinance as yf
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────
CAPITAL = 100_000_000        # Rp
MAX_POSITIONS = 10
RISK_PCT = 1.0
COMMISSION = 0.001
PERIOD = "5y"
SL_MULTIPLE = 2.8
SL_PERIOD = 10
ADX_PERIOD = 14
BB_PERIOD = 20
TP_RATIO = 0.4                # 0.4R
MIN_ADX = 20                  # ADX threshold for entry
MIN_TREND_SCORE = 0           # Minimum trend score (0 = no filter)
MIN_VOLUME = 5_000_000_000    # Minimum avg daily volume (Rp) — filter saham illiquid

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
        if day_idx % 300 == 0:
            print(f"  Day {day_idx}/{total_days}... ({len(positions)} open, ${cash:,.0f} cash)")
        
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
                    exit_val = pos['size'] * close
                    pnl = exit_val - pos['cost']
                    cash += exit_val
                    all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                        'exit_date': today.date(), 'entry_price': pos['entry_price'],
                        'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                        'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'SL'})
                    sl_exits += 1; positions.pop(i); continue
                elif pos['tp_threshold'] is not None:
                    floating = ((close - pos['entry_price']) / pos['entry_price']) * 100
                    if floating > pos['tp_threshold'] and close < sl_val:
                        exit_val = pos['size'] * close
                        pnl = exit_val - pos['cost']
                        cash += exit_val
                        all_trades.append({'ticker': pos['ticker'], 'entry_date': pos['entry_date'].date(),
                            'exit_date': today.date(), 'entry_price': pos['entry_price'],
                            'exit_price': close, 'size': pos['size'], 'pnl': pnl,
                            'return_pct': (close/pos['entry_price']-1)*100, 'exit_reason': 'TP'})
                        tp_exits += 1; positions.pop(i); continue
            i += 1
        
        # New signals
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
                    size = calc_position_size(cash + sum(p['cost'] for p in positions), close, sl_val)
                    cost = size * close
                    if size > 0:
                        new_signals.append({'ticker': ticker, 'score': score, 'close': close,
                            'sl': sl_val, 'size': size, 'cost': cost, 'bar': bar})
            total_signals += len(new_signals)
            new_signals.sort(key=lambda x: x['score'], reverse=True)
            for sig in new_signals:
                if len(positions) >= MAX_POSITIONS: break
                if cash >= sig['cost']:
                    stop_dist = abs(sig['close'] - sig['sl'])
                    tp_th = (stop_dist / sig['close']) * 100 * TP_RATIO if stop_dist > 0 else None
                    positions.append({'ticker': sig['ticker'], 'entry_price': sig['close'],
                        'entry_sl': sig['sl'], 'size': sig['size'], 'cost': sig['cost'],
                        'tp_threshold': tp_th, 'entry_date': today, 'entry_score': sig['score']})
                    cash -= sig['cost']; total_entries += 1
                else: total_skipped += 1
        
        invested = sum(p['cost'] for p in positions)
        equity_history.append(cash + invested)
        cash_history.append(cash)
        positions_history.append(len(positions))
    
    final_equity = equity_history[-1]
    total_return = (final_equity / CAPITAL - 1) * 100
    days_elapsed = (timeline[-1] - timeline[0]).days if len(timeline) > 1 else 1
    cagr = ((final_equity / CAPITAL) ** (365.0 / days_elapsed) - 1) * 100 if days_elapsed > 0 else 0
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
    
    return {'equity': equity_history, 'cash': cash_history, 'positions_count': positions_history,
        'timeline': timeline, 'total_return': total_return, 'cagr': cagr, 'max_dd': max_dd,
        'sharpe': sharpe, 'profit_factor': pf, 'total_trades': len(all_trades), 'win_rate': wr,
        'avg_win': aw, 'avg_loss': al, 'max_concurrent': max(positions_history) if positions_history else 0,
        'total_signals': total_signals, 'total_entries': total_entries, 'total_skipped': total_skipped,
        'sl_exits': sl_exits, 'tp_exits': tp_exits, 'trades': trades_df}

def print_results(result, capital):
    print()
    print("=" * 65)
    print("  DYNAMIC PORTFOLIO — ID MARKET RESULTS")
    print("=" * 65)
    curr = "Rp"
    print(f"  Initial Capital : {curr} {capital:,.0f}")
    print(f"  Final Equity    : {curr} {result['equity'][-1]:,.0f}")
    print(f"  Total Return    : {result['total_return']:+.2f}%")
    print(f"  CAGR            : {result['cagr']:+.2f}%")
    print(f"  Max Drawdown    : {result['max_dd']:.2f}%")
    print(f"  Sharpe Ratio    : {result['sharpe']:.2f}")
    print(f"  Profit Factor   : {result['profit_factor']:.2f}")
    print(f"  Total Trades    : {result['total_trades']}")
    print(f"  Win Rate        : {result['win_rate']:.1f}%")
    print(f"  Max Concurrent  : {result['max_concurrent']}")
    print(f"  Max Positions   : {MAX_POSITIONS}")
    print(f"  Risk/Trade      : {RISK_PCT}%")
    print(f"  TP Ratio        : {TP_RATIO}R")
    print(f"  Avg Win         : {curr} {result['avg_win']:,.0f}")
    print(f"  Avg Loss        : {curr} {result['avg_loss']:,.0f}")
    print(f"  Total Signals   : {result['total_signals']}")
    print(f"  Total Entries   : {result['total_entries']}")
    print(f"  Skipped (no cash): {result['total_skipped']}")
    print(f"  Exits (SL/TP)   : {result['sl_exits']} / {result['tp_exits']}")
    print("=" * 65)

def plot_results(result):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt; import matplotlib.dates as mdates
        dates = result['timeline']; equity = result['equity']
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.patch.set_facecolor('#1a1612')
        ax1.plot(dates, equity, color='#d4af37', linewidth=1.5, label='Equity')
        ax1.axhline(y=CAPITAL, color='#666', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.fill_between(dates, equity, CAPITAL, where=[e >= CAPITAL for e in equity], alpha=0.1, color='#4ade80')
        ax1.fill_between(dates, equity, CAPITAL, where=[e < CAPITAL for e in equity], alpha=0.1, color='#f87171')
        for spine in ax1.spines.values(): spine.set_color('#333')
        ax1.tick_params(colors='#99907c'); ax1.set_ylabel('Equity (Rp)', color='#99907c')
        ax1.set_facecolor('#1a1612'); ax1.grid(True, alpha=0.08, color='#d4af37')
        ax1.set_title('ID Dynamic Portfolio — Equity Curve', color='#d4af37', fontsize=14, fontweight='bold')
        eq_s = pd.Series(equity); peak = eq_s.expanding().max()
        dd = (eq_s - peak) / peak * 100
        ax2.fill_between(dates, 0, dd.values, color='#f87171', alpha=0.4)
        ax2.plot(dates, dd.values, color='#f87171', linewidth=0.8)
        for spine in ax2.spines.values(): spine.set_color('#333')
        ax2.tick_params(colors='#99907c'); ax2.set_ylabel('Drawdown %', color='#99907c')
        ax2.set_facecolor('#1a1612'); ax2.grid(True, alpha=0.08, color='#d4af37')
        ax3.fill_between(dates, 0, result['positions_count'], color='#60a5fa', alpha=0.3)
        ax3.plot(dates, result['positions_count'], color='#60a5fa', linewidth=0.8)
        ax3.axhline(y=MAX_POSITIONS, color='#d4af37', linestyle='--', linewidth=0.5, alpha=0.5)
        for spine in ax3.spines.values(): spine.set_color('#333')
        ax3.tick_params(colors='#99907c'); ax3.set_ylabel('Open Positions', color='#99907c')
        ax3.set_xlabel('Date', color='#99907c'); ax3.set_facecolor('#1a1612')
        ax3.grid(True, alpha=0.08, color='#d4af37'); ax3.set_ylim(bottom=0)
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, 'dynamic_id_chart.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#1a1612'); plt.close()
        return chart_path
    except ImportError: return None

if __name__ == '__main__':
    print("=" * 65)
    print("  DYNAMIC PORTFOLIO — ID MARKET")
    print("  AmiBroker-style rolling stock selection")
    print("=" * 65)
    print(f"  Capital: Rp {CAPITAL:,}")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Risk/Trade: {RISK_PCT}%")
    print(f"  TP Ratio: {TP_RATIO}R")
    print(f"  Min ADX: {MIN_ADX}")
    print(f"  Period: {PERIOD}")
    print()
    
    tickers = []
    csv_path = os.path.join(OUTPUT_DIR, 'id_liquid.csv')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader: tickers.append(row['Symbol'].strip())
    
    print(f"  Loading {len(tickers)} liquid ID stocks (volume > Rp 5M/hari)...")
    stocks = {}
    skipped_low_volume = 0
    for i, ticker in enumerate(tickers):
        result = load_stock(ticker)
        if result is not None:
            # Volume filter — skip illiquid stocks
            avg_volume_rp = np.mean(result['close'] * result['volume'])
            if avg_volume_rp < MIN_VOLUME:
                skipped_low_volume += 1
                continue
            stocks[ticker] = result
        if (i + 1) % 50 == 0: print(f"  Loaded {i+1}/{len(tickers)}... ({len(stocks)} valid, {skipped_low_volume} low vol)")
    
    print(f"  Successfully loaded: {len(stocks)} stocks (skipped {skipped_low_volume} illiquid)")
    print()
    
    result = run_dynamic_portfolio(stocks)
    print_results(result, CAPITAL)
    
    chart_path = plot_results(result)
    if not result['trades'].empty:
        result['trades'].to_csv(os.path.join(OUTPUT_DIR, 'id_trades.csv'), index=False)
    eq_df = pd.DataFrame({'Date': result['timeline'], 'Equity': result['equity'],
        'Cash': result['cash'], 'OpenPositions': result['positions_count']})
    eq_df.to_csv(os.path.join(OUTPUT_DIR, 'id_equity.csv'), index=False)
    report = {k: v for k, v in result.items() if k not in ['equity','cash','positions_count','timeline','trades']}
    with open(os.path.join(OUTPUT_DIR, 'id_report.json'), 'w') as f: json.dump(report, f, indent=2, default=str)
    
    print(f"\n  📁 Output saved")
    if chart_path: print(f"  🖼️  MEDIA:{chart_path}")
