"""
Parameter Scan — Intraday 1h Dynamic Portfolio
Download data ONCE, recalc indicators per variant.
"""
import os, sys, json, csv
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPITAL = 100_000_000
MAX_POSITIONS = 10
RISK_PCT = 1.0
PERIOD = "2mo"
INTERVAL = "1h"
MIN_VOLUME = 5_000_000_000

VARIANTS = {
    "A_Daily(14/10/20)": {"adx_p": 14, "sl_p": 10, "bb_p": 20, "sl_m": 2.8, "min_adx": 20, "pdi_bars": 5},
    "B_Moderate(10/7/14)": {"adx_p": 10, "sl_p": 7,  "bb_p": 14, "sl_m": 2.5, "min_adx": 20, "pdi_bars": 3},
    "C_Aggressive(7/5/10)": {"adx_p": 7,  "sl_p": 5,  "bb_p": 10, "sl_m": 2.2, "min_adx": 20, "pdi_bars": 3},
    "D_SLfaster(14/7/20)": {"adx_p": 14, "sl_p": 7,  "bb_p": 20, "sl_m": 2.5, "min_adx": 20, "pdi_bars": 5},
    "E_ADXfaster(7/10/20)": {"adx_p": 7,  "sl_p": 10, "bb_p": 20, "sl_m": 2.8, "min_adx": 25, "pdi_bars": 3},
}

# ── Load raw data once ──
print("Downloading data for all tickers...")
tickers = []
with open(os.path.join(OUTPUT_DIR, 'id_liquid.csv')) as f:
    for row in csv.DictReader(f): tickers.append(row['Symbol'].strip())

raw_data = {}
for t in tickers:
    try:
        df = yf.download(t, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        raw_data[t] = df
    except: pass

print(f"  Loaded {len(raw_data)} stocks with raw data")

# ── Run each variant ──
def calc_indicators(close, high, low, volume, cfg):
    sma = pd.Series(close).rolling(cfg['bb_p']).mean().values
    ero = int(cfg['sl_m'] * cfg['sl_p'])
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).rolling(cfg['sl_p']).mean().values
    sl = pd.Series(high).rolling(ero).max().values - 2 * atr * (cfg['sl_m'] - 1) / cfg['sl_m']
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_s = pd.Series(tr).rolling(cfg['adx_p']).mean().values
    sp = pd.Series(plus_dm).rolling(cfg['adx_p']).mean().values
    sm = pd.Series(minus_dm).rolling(cfg['adx_p']).mean().values
    pdi = np.where(atr_s > 0, 100 * sp / atr_s, np.nan)
    mdi = np.where(atr_s > 0, 100 * sm / atr_s, np.nan)
    dx = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), np.nan)
    adx = pd.Series(dx).rolling(cfg['adx_p']).mean().values
    return {'sma': sma, 'sl': sl, 'adx': adx, 'pdi': pdi, 'mdi': mdi}


def run_variant(vname, cfg):
    print(f"\n  ── {vname} ──")
    print(f"     ADX={cfg['adx_p']} SL={cfg['sl_p']} SMA={cfg['bb_p']} SLx={cfg['sl_m']} PDIb={cfg['pdi_bars']} minADX={cfg['min_adx']}")
    
    # Prepare stocks
    stocks = {}
    for t, df in raw_data.items():
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        avg_vol = np.mean(c * v)
        if avg_vol < MIN_VOLUME: continue
        ind = calc_indicators(c, h, l, v, cfg)
        stocks[t] = {'close': c, 'high': h, 'low': l, 'sma': ind['sma'], 'sl': ind['sl'],
                     'adx': ind['adx'], 'pdi': ind['pdi'], 'mdi': ind['mdi'],
                     'timestamps': df.index}
    print(f"     Stocks: {len(stocks)}")
    
    # Timeline
    all_ts = set()
    for s in stocks.values():
        for ts in s['timestamps']: all_ts.add(pd.Timestamp(ts))
    timeline = sorted(all_ts)
    
    stock_idx = {}
    for t, s in stocks.items():
        idx_map = {}
        for i, ts in enumerate(s['timestamps']): idx_map[pd.Timestamp(ts)] = i
        stock_idx[t] = idx_map
    
    # Simulate
    cash = CAPITAL
    positions = []
    eq_hist = []; trades = []
    sig_total = 0; entries = 0; sl_exits = 0; tp_exits = 0
    
    for now in timeline:
        # Exits
        i = 0
        while i < len(positions):
            pos = positions[i]
            s = stocks[pos['ticker']]
            bar = stock_idx[pos['ticker']].get(now)
            if bar is not None:
                close = float(s['close'][bar])
                if close < pos['entry_sl']:
                    ev = pos['size'] * close; cash += ev
                    trades.append({'pnl': ev - pos['cost'], 'ret': (close/pos['entry_price']-1)*100, 'reason': 'SL'})
                    sl_exits += 1; positions.pop(i); continue
                elif pos['tp_th'] is not None:
                    fl = (close - pos['entry_price'])/pos['entry_price']*100
                    if fl > pos['tp_th']:
                        ev = pos['size'] * close; cash += ev
                        trades.append({'pnl': ev - pos['cost'], 'ret': (close/pos['entry_price']-1)*100, 'reason': 'TP'})
                        tp_exits += 1; positions.pop(i); continue
            i += 1
        
        # Entries
        if len(positions) < MAX_POSITIONS:
            signals = []
            for t, s in stocks.items():
                if any(p['ticker'] == t for p in positions): continue
                bar = stock_idx[t].get(now)
                if bar is None or bar < 30: continue
                
                if cfg['min_adx'] > 0 and np.isnan(s['adx'][bar]): continue
                if cfg['min_adx'] > 0 and s['adx'][bar] <= cfg['min_adx']: continue
                if np.isnan(sadx := s['adx'][bar]): continue
                if np.isnan(spdi := s['pdi'][bar]): continue
                if np.isnan(smdi := s['mdi'][bar]): continue
                if np.isnan(ssma := s['sma'][bar]): continue
                if np.isnan(ssl := s['sl'][bar]): continue
                
                close = float(s['close'][bar])
                low = float(s['low'][bar])
                if not (low > ssl and close > ssma): continue
                if not (spdi > smdi): continue
                if bar >= cfg['pdi_bars']:
                    pdi_ago = float(s['pdi'][bar - cfg['pdi_bars']])
                    if not (spdi > pdi_ago): continue
                
                slv = ssl
                sd = abs(close - slv)
                if sd <= 0: continue
                size = int((cash + sum(p['cost'] for p in positions)) * (RISK_PCT/100.0) / sd)
                max_sz = int((cash + sum(p['cost'] for p in positions)) * 0.95 / close)
                if max_sz <= 0: continue
                size = max(1, min(size, max_sz))
                cost = size * close
                if size > 0:
                    tp_th = (sd / close) * 100 * 0.4
                    signals.append({'ticker': t, 'close': close, 'sl': slv, 'size': size, 'cost': cost, 'tp_th': tp_th})
            
            sig_total += len(signals)
            for sig in signals:
                if len(positions) >= MAX_POSITIONS: break
                if cash >= sig['cost']:
                    positions.append({'ticker': sig['ticker'], 'entry_price': sig['close'],
                        'entry_sl': sig['sl'], 'size': sig['size'], 'cost': sig['cost'],
                        'tp_th': sig['tp_th']})
                    cash -= sig['cost']; entries += 1
        
        eq_hist.append(cash + sum(p['cost'] for p in positions))
    
    # Metrics
    fe = eq_hist[-1]
    tr = (fe / CAPITAL - 1) * 100
    days = len(timeline) / 7
    cagr = ((fe / CAPITAL) ** (365.0 / max(days, 1)) - 1) * 100 if days > 0 else 0
    
    eq_s = pd.Series(eq_hist)
    peak = eq_s.expanding().max()
    mdd = ((eq_s - peak) / peak * 100).min()
    
    tdf = pd.DataFrame(trades)
    n_win = (tdf['pnl'] > 0).sum() if len(tdf) > 0 else 0
    wr = n_win / len(tdf) * 100 if len(tdf) > 0 else 0
    gp = tdf[tdf['pnl'] > 0]['pnl'].sum() if len(tdf) > 0 else 0
    gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum()) if len(tdf) > 0 else 0
    pf = gp / gl if gl > 0 else 0
    aw = tdf[tdf['pnl'] > 0]['pnl'].mean() if n_win > 0 else 0
    al = tdf[tdf['pnl'] < 0]['pnl'].mean() if len(tdf) > n_win else 0
    sharpe = (pd.Series(eq_hist).pct_change().dropna().mean() / 
              pd.Series(eq_hist).pct_change().dropna().std() * np.sqrt(252*7)
              if pd.Series(eq_hist).pct_change().dropna().std() > 0 else 0)
    
    print(f"     → Return: {tr:+.2f}% | CAGR: {cagr:+.2f}% | DD: {mdd:.2f}% | "
          f"Sharpe: {sharpe:.2f} | PF: {pf:.2f} | Trades: {len(trades)} | "
          f"WR: {wr:.1f}% | SL/TP: {sl_exits}/{tp_exits}")
    
    return {'name': vname, 'total_return': tr, 'cagr': cagr, 'max_dd': mdd,
            'sharpe': sharpe, 'profit_factor': pf, 'trades': len(trades), 'win_rate': wr,
            'max_conc': MAX_POSITIONS, 'entries': entries, 'signals': sig_total,
            'sl_exits': sl_exits, 'tp_exits': tp_exits, 'avg_win': aw, 'avg_loss': al}

results = []
for vname, vcfg in VARIANTS.items():
    results.append(run_variant(vname, vcfg))

# Summary
print(f"\n{'='*80}")
print(f"  {'Variant':<20} {'Return':>8} {'CAGR':>8} {'DD':>7} {'Sharpe':>7} {'PF':>6} {'Trades':>7} {'WR%':>6} {'SL/TP':>8}")
print(f"{'='*80}")
for r in results:
    print(f"  {r['name']:<20} {r['total_return']:>+7.2f}% {r['cagr']:>+7.2f}% "
          f"{r['max_dd']:>6.2f}% {r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
          f"{r['trades']:>6} {r['win_rate']:>5.1f}% "
          f"{r['sl_exits']:>3}/{r['tp_exits']:<3}")
print(f"{'='*80}")

with open(os.path.join(OUTPUT_DIR, 'id_intraday_variants.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  📁 Saved to id_intraday_variants.json")
