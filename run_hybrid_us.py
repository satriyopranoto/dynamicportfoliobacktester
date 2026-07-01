"""
Test Hybrid Strategy on multiple US stocks — last 5 years.
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
import sys
sys.path.insert(0, r"C:\Users\Acer\code\backtester")
from pathlib import Path
from backtesting import Backtest
from strategies.hybrid_basis_adx_strategy import HybridBasisAdxStrategy
warnings.filterwarnings("ignore")

out = Path(r'C:\Users\Acer\code\backtester\reports')
out.mkdir(exist_ok=True)

# Major US stocks across sectors
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "TSLA", "JPM", "V", "JNJ", "WMT", "XOM",
    "PG", "KO", "PEP", "HD", "DIS", "NFLX",
]

print("=" * 65)
print("  HYBRID STRATEGY — US Stocks (5 tahun)")
print("  (Entry: Basis+ADX || Exit: Mean Rev ADX)")
print("=" * 65)

results = []

for ticker in tickers:
    try:
        data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna(subset=["Open","High","Low","Close"])
        
        if len(data) < 100:
            print(f"  {ticker:6s}: data terlalu sedikit ({len(data)}) — skip")
            continue

        bt = Backtest(data, HybridBasisAdxStrategy, cash=100_000, commission=0.001, finalize_trades=True)
        stats = bt.run()
        bh = stats['Buy & Hold Return [%]']
        ret = stats['Return [%]']
        sharpe = stats['Sharpe Ratio']
        mdd = stats['Max. Drawdown [%]']
        trades = stats['# Trades']
        wr = stats.get('Win Rate [%]', 0)
        
        print(f"  {ticker:6s}: Return {ret:+6.2f}% | B&H {bh:+6.2f}% | "
              f"Sharpe {sharpe:.2f} | DD {mdd:.2f}% | Trades {trades:2d} | Win% {wr:.1f}%")
        
        results.append({
            'ticker': ticker, 'return': ret, 'buy_hold': bh,
            'sharpe': sharpe, 'mdd': mdd, 'trades': trades, 'win_rate': wr
        })
        
        # Save individual HTML for best performers
        if ret > 5:
            bt.plot(filename=str(out / f"{ticker}_hybrid.html"), open_browser=False)
            
    except Exception as e:
        print(f"  {ticker:6s}: ERROR — {e}")

# Summary
print(f"\n{'=' * 65}")
print(f"  SUMMARY — {len(results)} stocks")
print(f"{'=' * 65}")
df = pd.DataFrame(results)
print(f"  Avg Return      : {df['return'].mean():+.2f}%")
print(f"  Avg Buy&Hold    : {df['buy_hold'].mean():+.2f}%")
print(f"  Avg Sharpe      : {df['sharpe'].mean():.2f}")
print(f"  Avg Max DD      : {df['mdd'].mean():.2f}%")
print(f"  Avg Trades      : {df['trades'].mean():.0f}")
print(f"  Avg Win Rate    : {df['win_rate'].mean():.1f}%")
print(f"\n  {'Ticker':6s} | {'Return':>7s} | {'B&H':>7s} | {'Sharpe':>6s} | {'DD':>6s} | {'Trades':>6s} | {'Win%':>5s}")
print(f"  {'-'*6} | {'-'*7} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*5}")
for _, r in df.sort_values('return', ascending=False).iterrows():
    print(f"  {r['ticker']:6s} | {r['return']:+6.2f}% | {r['buy_hold']:+6.2f}% | "
          f"{r['sharpe']:5.2f} | {r['mdd']:5.2f}% | {r['trades']:5.0f} | {r['win_rate']:4.1f}%")

print(f"\n{'=' * 65}")
print("  ✅ Done")
print(f"{'=' * 65}")
