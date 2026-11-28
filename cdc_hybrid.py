"""Experimental CDC research; no production integration."""
import numpy as np
import pandas as pd


def heikin_ashi(hist):
    d = hist.copy()
    o,h,l,c = (d[k].astype(float).to_numpy() for k in ('Open','High','Low','Close'))
    if not len(o) or not np.isfinite([o,h,l,c]).all():
        raise ValueError('Invalid OHLC')
    hc = (o+h+l+c)/4
    ho = np.empty(len(o)); ho[0]=(o[0]+c[0])/2
    for i in range(1,len(o)):
        ho[i]=(ho[i-1]+hc[i-1])/2
    d['Open'],d['Close']=ho,hc
    d['High']=np.maximum.reduce([h,ho,hc]); d['Low']=np.minimum.reduce([l,ho,hc])
    return d


def cdc(hist, source='Close'):
    c=hist[source].astype(float)
    f=c.ewm(span=12,adjust=False).mean(); s=c.ewm(span=26,adjust=False).mean()
    bull, bear=f>s,f<s
    g=bull & (c>f); r=bear & (c<f)
    zone=np.select([g,bear&(c>f)&(c>s),bear&(c>f)&(c<s),r,bull&(c<f)&(c<s),bull&(c<f)&(c>s)],['Green','Blue','Light Blue','Red','Orange','Yellow'],default='Neutral')
    fg=g & ~g.shift(1,fill_value=False); fr=r & ~r.shift(1,fill_value=False)
    state=pd.Series(np.select([fg,fr],[1.,-1.],default=np.nan),index=c.index).ffill().fillna(0).astype(int)
    prev=state.shift(1,fill_value=0)
    return pd.DataFrame({'zone':zone,'trend':state,'buy':fg & (prev==-1),'sell':fr & (prev==1)},index=c.index)


def signals(hist, ha_source='Close'):
    n=cdc(hist); a=cdc(heikin_ashi(hist),ha_source)
    out=n.add_prefix('normal_').join(a.add_prefix('ha_'))
    both=(n.trend==1)&(a.trend==1)
    out['hybrid_trend']=both.astype(int)
    out['hybrid_buy']=both & ~both.shift(1,fill_value=False)
    out['hybrid_sell']=~both & both.shift(1,fill_value=False)
    out['status']=np.select([out.hybrid_buy,both,(n.trend==1)^(a.trend==1)],['Confirmed Buy','Confirmed Trend','Early / Weak'],default='Avoid')
    return out
