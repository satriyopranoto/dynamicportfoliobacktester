import pandas as pd

df = pd.read_csv(r'C:\Users\satri\code\dynamicportfolio\id_trades.csv')
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])

# Q4 2025
q4 = df[(df['exit_date'] >= '2025-10-01') & (df['exit_date'] <= '2025-12-31')]
q4 = q4.sort_values('pnl', ascending=False)

print(f"Trades in Q4 2025: {len(q4)}")
print()

# Big trades (PnL > 50jt)
big = q4[q4['pnl'].abs() > 50_000_000]
if len(big) > 0:
    print("=== BIG TRADES (PnL > Rp 50jt) ===")
    print(f"{'Ticker':<10} {'Entry':>12} {'Exit':>12} {'Entry':>10} {'Exit':>10} {'PnL':>15} {'Return%':>8} {'Reason':>6}")
    print("-" * 75)
    for _, r in big.iterrows():
        print(f"{r['ticker']:<10} {str(r['entry_date'])[:10]:>12} {str(r['exit_date'])[:10]:>12} {r['entry_price']:>9,.0f} {r['exit_price']:>9,.0f} {r['pnl']:>+14,.0f} {r['return_pct']:>+7.2f}% {r['exit_reason']:>6}")

print()
print(f"Total PnL Q4 2025: Rp {q4['pnl'].sum():,.0f}")
print(f"Trades: {len(q4)}  |  Wins: {(q4['pnl']>0).sum()}  |  Losses: {(q4['pnl']<0).sum()}")

# Monthly breakdown 2025
print()
print("=== MONTHLY PnL 2025 ===")
df['month'] = df['exit_date'].dt.to_period('M')
monthly = df[df['exit_date'].dt.year == 2025].groupby('month')['pnl'].agg(['sum', 'count', 'mean'])
monthly.columns = ['Total PnL', 'Trades', 'Avg']
monthly['Total PnL'] = monthly['Total PnL'].map('Rp {:,.0f}'.format)
monthly['Avg'] = monthly['Avg'].map('Rp {:,.0f}'.format)
print(monthly)

# Check what stock had the biggest single trade
print()
print("=== TOP 3 TRADES EVER ===")
top3 = df.sort_values('pnl', ascending=False).head(3)
for _, r in top3.iterrows():
    print(f"{r['ticker']:<10} PnL: Rp {r['pnl']:>+14,.0f}  Return: {r['return_pct']:>+7.2f}%  Entry: {str(r['entry_date'])[:10]}  Exit: {str(r['exit_date'])[:10]}  {r['exit_reason']}")
