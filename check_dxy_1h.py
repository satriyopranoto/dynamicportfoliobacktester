"""DXY 1H — 3 Month Trend Analysis"""
import yfinance as yf, pandas as pd, numpy as np

print('='*65)
print('  DXY 1H — Last 3 Months')
print('='*65)

df = yf.download('DX-Y.NYB', period='3mo', interval='1h', progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df.columns = [c.lower() for c in df.columns]
c = df['close'].values.astype(float); h = df['high'].values.astype(float); l = df['low'].values.astype(float)

tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
up=np.diff(h, prepend=h[0]); dn=np.diff(l, prepend=l[0])
pdm=np.where((up>dn)&(up>0), up, 0); mdm=np.where((dn>up)&(dn>0), dn, 0)
atr_s=pd.Series(tr).rolling(14).mean().values
sp=pd.Series(pdm).rolling(14).mean().values; sm=pd.Series(mdm).rolling(14).mean().values
pdi=np.where(atr_s>0, 100*sp/atr_s, np.nan); mdi=np.where(atr_s>0, 100*sm/atr_s, np.nan)
dx=np.where((pdi+mdi)>0, 100*np.abs(pdi-mdi)/(pdi+mdi), np.nan)
adx=pd.Series(dx).rolling(14).mean().values
sma20=pd.Series(c).rolling(20).mean().values

total=0; bt=0; bnt=0; wt=0; wnt=0
for i in range(len(c)):
    if i<20 or np.isnan(adx[i]) or np.isnan(sma20[i]): continue
    total+=1
    if adx[i]>25 and c[i]>sma20[i]: bt+=1
    elif adx[i]>25 and c[i]<=sma20[i]: bnt+=1
    elif adx[i]<=25 and c[i]>sma20[i]: wt+=1
    else: wnt+=1

print(f'Bars: {len(df)} ({len(df)//24:.0f} days)')
print(f'Price: {c[0]:.2f} -> {c[-1]:.2f} ({((c[-1]/c[0])-1)*100:+.2f}%)')
print(f'Rata-rata ADX: {np.nanmean(adx[20:]):.1f}')
print(f'ADX sekarang: {adx[-1]:.1f}')
print()
print('Quadrant (ADX=14, SMA=20):')
print(f'  Bullish Trend (ADX>25, Close>SMA): {bt:>4} bars ({bt/total*100:.1f}%)')
print(f'  Bearish Trend (ADX>25, Close<SMA): {bnt:>4} bars ({bnt/total*100:.1f}%)')
print(f'  Bullish Weak  (ADX<=25, Close>SMA): {wt:>4} bars ({wt/total*100:.1f}%)')
print(f'  Bearish Weak  (ADX<=25, Close<SMA): {wnt:>4} bars ({wnt/total*100:.1f}%)')
print(f'  Total: {total} bars')
print()

# Monthly breakdown
print('Monthly Trend Progression:')
df2 = df.copy()
df2['adx'] = adx
df2['sma'] = sma20
df2['bull'] = (df2['adx'] > 25) & (df2['close'] > df2['sma'])
df2['bear'] = (df2['adx'] > 25) & (df2['close'] <= df2['sma'])
monthly = df2.resample('ME').agg({'close':'last','high':'max','low':'min','adx':'mean','bull':'sum','bear':'sum'})
for idx, row in monthly.iterrows():
    tot = int(row['bull'])+int(row['bear'])
    print(f'  {idx.strftime("%b %Y"):>8}  Close:{row["close"]:>6.1f}  H:{row["high"]:>6.1f} L:{row["low"]:>6.1f}  ADXu:{row["adx"]:>5.1f}  Trend:{tot:>3}h  Bull:{int(row["bull"]):>2}h Bear:{int(row["bear"]):>2}h  Net:{int(row["bull"]-row["bear"]):>+3}h')
