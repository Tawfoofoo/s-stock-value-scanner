"""Technical research only; fixed window, next-open execution, adjusted OHLC."""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from cdc_hybrid import cdc, heikin_ashi, signals

START='2021-11-26'; END='2026-11-27'; WARMUP='2020-11-26'

def evaluate(h, s, start, end, cost):
    target=s.trend.eq(1).astype(int)
    pos=target.shift(1,fill_value=0).loc[start:end]
    opens=h.Open.reindex(pos.index)
    pos=pos.iloc[:-1]; ret=opens.shift(-1).div(opens).sub(1).iloc[:-1]
    changes=pos.diff().abs(); changes.iloc[0]=pos.iloc[0]
    net=(1-cost*changes)*(1+pos*ret); net.iloc[-1]*=1-cost*pos.iloc[-1]
    equity=net.cumprod(); peak=equity.cummax().clip(lower=1)
    idx=np.flatnonzero(s.buy.to_numpy() & (h.index>=start) & (h.index<=end))
    sells=s.sell.to_numpy(); out=dict(return_pct=100*(equity.iloc[-1]-1),max_drawdown_pct=100*(equity/peak-1).min(),buy_signals=len(idx),exposure_pct=100*pos.mean())
    for k in (5,10):
        valid=[i for i in idx if i+k<len(h)]
        n=sum(bool(sells[i+1:i+k+1].any()) for i in valid)
        out.update({f'reversal_{k}d_n':len(valid),f'reversal_{k}d_count':n,f'reversal_{k}d_pct':100*n/len(valid) if valid else np.nan})
    for k in (20,60):
        vals=[100*(h.Open.iloc[i+k+1]/h.Open.iloc[i+1]-1) for i in idx if i+k+1<len(h)]
        out.update({f'forward_{k}d_pct':np.mean(vals) if vals else np.nan,f'forward_{k}d_n':len(vals)})
    return out

def strategies(h):
    yield 'Normal Close',cdc(h)
    yield 'Normal Open',cdc(h,'Open')
    for source in ('Close','Open'):
        yield 'HA '+source,cdc(heikin_ashi(h),source)
        x=signals(h,source); yield 'Hybrid '+source,x[['hybrid_trend','hybrid_buy','hybrid_sell']].rename(columns=lambda x:x.removeprefix('hybrid_'))
    yield 'Buy & Hold',pd.DataFrame({'trend':1,'buy':False,'sell':False},index=h.index)

def delays(base, other, start):
    b=np.flatnonzero(base.buy.to_numpy() & (base.index>=start)); o=np.flatnonzero(other.buy.to_numpy()); used=set(); ds=[]
    for i in b:
        candidates=[j for j in o if abs(j-i)<=20 and j not in used]
        if candidates:
            j=min(candidates,key=lambda j:(abs(j-i),j)); used.add(j); ds.append(j-i)
    return dict(matched_signals=len(ds),normal_signals=len(b),median_delay_days=np.median(ds) if ds else np.nan,mean_delay_days=np.mean(ds) if ds else np.nan)

def main():
    import yfinance as yf
    p=argparse.ArgumentParser(); p.add_argument('--universe',required=True); p.add_argument('--cost-bps',type=float,default=10); a=p.parse_args()
    tickers=list(dict.fromkeys(pd.read_csv(a.universe)['Symbol'].astype(str).str.strip().str.replace('.','-',regex=False)))
    rows=[]; errors=[]
    for offset in range(0,len(tickers),25):
        batch=tickers[offset:offset+25]
        try: data=yf.download(batch,start=WARMUP,end='2026-11-28',auto_adjust=True,group_by='ticker',threads=4,progress=False)
        except Exception as e:
            errors.extend(dict(ticker=t,error=str(e)) for t in batch); continue
        for t in batch:
            try:
                h=data[t][['Open','High','Low','Close']].dropna().sort_index(); h.index=pd.to_datetime(h.index).tz_localize(None)
                if len(h.loc[:START])<200 or len(h.loc[START:END])<1200 or h.loc[START:END].index[0].strftime('%Y-%m-%d')!=START or h.index[-1].strftime('%Y-%m-%d')!=END: raise ValueError('Incomplete fixed window or warmup')
                if not np.isfinite(h.to_numpy()).all() or (h<=0).any().any(): raise ValueError('Invalid prices')
                base=cdc(h); local=[]
                for name,s in strategies(h):
                    local.append(dict(ticker=t,strategy=name,start=START,end=END,**evaluate(h,s,START,END,a.cost_bps/10000),**(delays(base,s,START) if name!='Buy & Hold' else {})))
                rows.extend(local)
            except Exception as e: errors.append(dict(ticker=t,error=str(e)))
        print(f'Processed {min(offset+25,len(tickers))}/{len(tickers)}',flush=True); time.sleep(1)
    pd.DataFrame(errors,columns=['ticker','error']).to_csv('cdc_v2_errors.csv',index=False)
    if not rows: raise SystemExit('No complete results')
    df=pd.DataFrame(rows); df.to_csv('cdc_v2_results.csv',index=False)
    summary=df.groupby('strategy').agg(stocks=('ticker','nunique'),return_pct=('return_pct','mean'),max_drawdown_pct=('max_drawdown_pct','mean'),buy_signals=('buy_signals','sum'),median_delay_days=('median_delay_days','median'))
    for k in (5,10):
        g=df.groupby('strategy'); summary[f'reversal_{k}d_pct']=100*g[f'reversal_{k}d_count'].sum()/g[f'reversal_{k}d_n'].sum().replace(0,np.nan)
    for k in (20,60):
        n=df[f'forward_{k}d_n']; weighted=df[f'forward_{k}d_pct']*n
        summary[f'forward_{k}d_pct']=weighted.groupby(df.strategy).sum()/n.groupby(df.strategy).sum().replace(0,np.nan)
    summary.to_csv('cdc_v2_summary.csv'); print(summary.to_string()); print(f'Universe {len(tickers)}; complete {df.ticker.nunique()}; errors {len(errors)}')
if __name__=='__main__': main()
