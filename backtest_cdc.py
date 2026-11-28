"""Technical-only comparison, signals at close, executions next open."""
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from cdc_hybrid import signals


def evaluate(hist, sig, prefix, start, cost):
    # Signal state is persistent; next open execution avoids look-ahead.
    target=(sig[prefix+'_trend']==1).astype(int)
    pos=target.shift(1,fill_value=0).loc[start:]
    opens=hist.Open.reindex(pos.index)
    pos=pos.iloc[:-1]; ret=opens.shift(-1).div(opens).sub(1).iloc[:-1]
    changes=pos.diff().abs(); changes.iloc[0]=pos.iloc[0]
    net=(1-cost*changes)*(1+pos*ret)
    net.iloc[-1]*=1-cost*pos.iloc[-1]
    equity=net.cumprod(); peak=equity.cummax().clip(lower=1)
    buys=sig[prefix+'_buy']; sells=sig[prefix+'_sell'].to_numpy()
    idx=np.flatnonzero(buys.to_numpy() & (hist.index>=start))
    result={'return_pct':100*(equity.iloc[-1]-1),'max_drawdown_pct':100*(equity/peak-1).min(),'buy_signals':len(idx)}
    for k in (5,10):
        valid=[i for i in idx if i+k<len(hist)]
        result[f'reversal_{k}d_pct']=100*np.mean([sells[i+1:i+k+1].any() for i in valid]) if valid else np.nan
        result[f'reversal_{k}d_n']=len(valid)
    for k in (20,60):
        values=[hist.Open.iloc[i+k+1]/hist.Open.iloc[i+1]-1 for i in idx if i+k+1<len(hist)]
        result[f'forward_{k}d_pct']=100*np.mean(values) if values else np.nan
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('tickers',nargs='*',default=['ACN','ADBE','DECK','VZ','INTU','AOS','PHM']); p.add_argument('--cost-bps',type=float,default=10); args=p.parse_args()
    rows=[]
    for ticker in args.tickers:
        try:
            h=yf.Ticker(ticker).history(start=(pd.Timestamp.now()-pd.DateOffset(years=6)).strftime('%Y-%m-%d'),auto_adjust=True,interval='1d')[['Open','High','Low','Close','Volume']].dropna().sort_index()
            # Exclude current exchange-local date conservatively.
            h=h[h.index.date < pd.Timestamp.now(tz=h.index.tz).date()]
            if len(h)<300: raise ValueError('Insufficient history')
            start=h.index[-1]-pd.DateOffset(years=5)
            for source in ('Close','Open'):
                s=signals(h,source)
                for prefix in (('normal','ha','hybrid') if source=='Close' else ('ha','hybrid')):
                    rows.append({'ticker':ticker,'strategy':prefix if prefix=='normal' else prefix+'_'+source,**evaluate(h,s,prefix,start,args.cost_bps/10000)})
        except Exception as e: print(ticker, type(e).__name__, str(e))
    if not rows: raise SystemExit('No results')
    df=pd.DataFrame(rows); df.to_csv('cdc_backtest_results.csv',index=False); print(df.to_string(index=False))

if __name__=='__main__': main()
