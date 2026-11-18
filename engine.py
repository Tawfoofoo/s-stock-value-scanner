"""V6 valuation engine. Indicative FCF DCF, not investment advice."""
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

SPECIAL = {'Financials', 'Real Estate', 'Utilities'}

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None

def first(*xs):
    return next((v for x in xs if (v := num(x)) is not None), None)

def dated_values(frame, names):
    if frame is None or frame.empty:
        return {}
    dates = pd.to_datetime(frame.columns, utc=True, errors='coerce')
    if dates.isna().any():
        return {}
    for name in names:
        if name in frame.index:
            s = frame.loc[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return {d.strftime('%Y-%m-%d'): num(v) for d, v in zip(dates, s)}
    return {}

def latest(frame, names):
    values = dated_values(frame, names)
    return next((v for _, v in sorted(values.items(), reverse=True) if v is not None), None)

@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(symbol):
    t = yf.Ticker(symbol)
    out = []
    for name in ('info', 'fast_info', 'cashflow', 'balance_sheet', 'income_stmt'):
        try:
            x = getattr(t, name)
            out.append(dict(x) if name == 'fast_info' else x)
        except Exception:
            out.append({} if name in ('info', 'fast_info') else pd.DataFrame())
    return tuple(out)

@st.cache_data(ttl=1800, show_spinner=False)
def history(symbol):
    d = yf.Ticker(symbol).history(period='2y', auto_adjust=True)
    if d is None or d.empty or len(d) < 210:
        raise ValueError('Insufficient daily price history')
    return d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close']).copy()

def indicators(hist):
    d = hist.copy()
    c, h, l, v = (d[k].astype(float) for k in ('Close','High','Low','Volume'))
    d['EMA12'] = c.ewm(span=12, adjust=False).mean()
    d['EMA26'] = c.ewm(span=26, adjust=False).mean()
    d['EMA50'] = c.ewm(span=50, adjust=False).mean()
    d['EMA200'] = c.ewm(span=200, adjust=False).mean()
    bull, bear = d.EMA12 > d.EMA26, d.EMA12 < d.EMA26
    green = bull & (c > d.EMA12)
    blue = bear & (c > d.EMA12) & (c > d.EMA26)
    aqua = bear & (c > d.EMA12) & (c < d.EMA26)
    red = bear & (c < d.EMA12)
    orange = bull & (c < d.EMA12) & (c < d.EMA26)
    yellow = bull & (c < d.EMA12) & (c > d.EMA26)
    d['CDC_zone'] = np.select([green,blue,aqua,red,orange,yellow], ['Green','Blue','Light Blue','Red','Orange','Yellow'], default='Neutral')
    first_green = green & ~green.shift(1, fill_value=False)
    first_red = red & ~red.shift(1, fill_value=False)
    last_g = last_r = None
    buys, sells, states = [], [], []
    for i, (g,r) in enumerate(zip(first_green, first_red)):
        previous = 1 if last_g is not None and (last_r is None or last_g > last_r) else -1 if last_r is not None and (last_g is None or last_r > last_g) else 0
        buys.append(bool(g and previous == -1))
        sells.append(bool(r and previous == 1))
        if g: last_g = i
        if r: last_r = i
        states.append(1 if last_g is not None and (last_r is None or last_g > last_r) else -1 if last_r is not None and (last_g is None or last_r > last_g) else 0)
    d['CDC_buy'], d['CDC_sell'], d['CDC_trend'] = buys, sells, states
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['RSI'] = 100 - 100/(1+gain/loss.replace(0,np.nan))
    d.loc[(loss==0)&(gain>0),'RSI'] = 100
    d.loc[(loss==0)&(gain==0),'RSI'] = 50
    macd = d.EMA12-d.EMA26
    d['MACD_hist'] = macd-macd.ewm(span=9,adjust=False).mean()
    typical = (h+l+c)/3
    positive = (typical*v).where(typical.diff()>0,0).rolling(14).sum()
    negative = (typical*v).where(typical.diff()<0,0).rolling(14).sum()
    d['MFI'] = 100-100/(1+positive/negative.replace(0,np.nan))
    d.loc[(negative==0)&(positive>0),'MFI'] = 100
    spread = (h-l).replace(0,np.nan)
    d['CMF'] = (((c-l-h+c)/spread).fillna(0)*v).rolling(20).sum()/v.rolling(20).sum().replace(0,np.nan)
    d['VolRatio'] = v/v.rolling(20).mean().replace(0,np.nan)
    return d

def technical(d):
    a,b = d.iloc[-1], d.iloc[-2]
    rsi,mfi,cmf,vr = [num(a[k]) for k in ('RSI','MFI','CMF','VolRatio')]
    proxy = 'Accumulation' if cmf is not None and mfi is not None and cmf>.05 and mfi>=50 else 'Distribution' if cmf is not None and cmf<-.05 else 'Neutral'
    score = sum([a.CDC_zone=='Green',bool(a.Close>a.EMA50),bool(a.MACD_hist>b.MACD_hist),rsi is not None and 40<=rsi<=70,proxy=='Accumulation',vr is not None and vr>=1])
    return {'CDC Zone':str(a.CDC_zone),'CDC Buy today':bool(a.CDC_buy),'CDC Sell today':bool(a.CDC_sell),'CDC Trend':'Bullish' if a.CDC_trend==1 else 'Bearish' if a.CDC_trend==-1 else 'Undefined','Smart Money Proxy':proxy,'RSI':rsi,'MFI':mfi,'CMF':cmf,'Volume / 20d':vr,'Technical Score':score}


def dcf(fcf, cash, debt, shares, growth, terminal, rate):
    """
    Indicative equity cash-flow DCF.

    Assumes the supplied FCF approximates cash flow
    available to equity holders after interest expense.

    Uses an assumed cost of equity, NOT WACC.
    Cash and debt are not adjusted again to avoid
    double-counting financing claims.

    This is a screening estimate, not a full FCFE model.
    """
    values = (fcf, shares, growth, terminal, rate)

    if not all(math.isfinite(x) for x in values):
        return None

    if fcf <= 0 or shares <= 0:
        return None

    if growth <= -1 or terminal <= -1:
        return None

    if rate <= terminal:
        return None

    projected_fcf = [
        fcf * (1 + growth) ** year
        for year in range(1, 6)
    ]

    present_value = sum(
        flow / (1 + rate) ** year
        for year, flow in enumerate(projected_fcf, 1)
    )

    terminal_value = (
        projected_fcf[-1] * (1 + terminal)
        / (rate - terminal)
    )

    discounted_terminal = terminal_value / (1 + rate) ** 5

    equity_value = present_value + discounted_terminal

    if not math.isfinite(equity_value) or equity_value <= 0:
        return None

    return equity_value / shares

    if not all(math.isfinite(x) for x in (fcf,cash,debt,shares,growth,terminal,rate)) or fcf<=0 or shares<=0 or rate<=terminal:
        return None
    flows=[fcf*(1+growth)**i for i in range(1,6)]
    pv=sum(f/(1+rate)**i for i,f in enumerate(flows,1))
    tv=flows[-1]*(1+terminal)/(rate-terminal)
    return (pv+tv/(1+rate)**5+cash-debt)/shares

def quality_score(roe, de, fcf_values, revenue_values, earnings_values, cash_conversion):
    """Conservative evidence-based score; missing components receive zero points."""
    score=0
    if roe is not None:
        score += 15 if roe>=.20 else 10 if roe>=.12 else 5 if roe>0 else 0
    if cash_conversion is not None:
        score += 10 if cash_conversion>=1 else 6 if cash_conversion>=.7 else 0
    if de is not None:
        score += 20 if 0<=de<=.5 else 14 if de<=1 else 7 if de<=2 else 0
    valid_fcf=[v for v in fcf_values if v is not None]
    if len(valid_fcf)>=3:
        score += 15 if all(v>0 for v in valid_fcf) else 7 if sum(v>0 for v in valid_fcf)>=len(valid_fcf)-1 else 0
        score += 10 if valid_fcf[0]>=valid_fcf[-1]>0 else 5 if valid_fcf[0]>0 else 0
    for values in (revenue_values,earnings_values):
        valid=[v for v in values if v is not None]
        if len(valid)>=3:
            score += 10 if valid[0]>valid[-1]>0 else 5 if valid[0]>0 else 0
    if len(valid_fcf)>=4 and all(v>0 for v in valid_fcf):
        score += 10
    return min(100,score)

def evaluate(symbol,growth_cap=.10,rate=.10,terminal=.025,min_upside=20,min_quality=70,unused=1.0,sector='Unknown'):
    if sector in SPECIAL:
        raise ValueError('Special Review: sector-specific valuation required')
    if rate<=terminal+.01:
        raise ValueError('Discount rate must exceed terminal growth by >1 percentage point')
    d=indicators(history(symbol)); tech=technical(d)
    info,fast,cf,bs,inc=fundamentals(symbol)
    price=first(d.Close.iloc[-1],info.get('currentPrice'),fast.get('lastPrice'))
    shares=first(info.get('sharesOutstanding'),fast.get('shares'),latest(bs,['Ordinary Shares Number','Share Issued']))
    if shares is None:
        cap=first(info.get('marketCap'),fast.get('marketCap'))
        shares=cap/price if cap and price else None
    cash=first(info.get('totalCash'),latest(bs,['Cash Cash Equivalents And Short Term Investments','Cash And Cash Equivalents','Cash And Short Term Investments']))
    debt=first(info.get('totalDebt'),latest(bs,['Total Debt']))
    fcf_map=dated_values(cf,['Free Cash Flow','FreeCashFlow'])
    if not fcf_map:
        operating=dated_values(cf,['Operating Cash Flow','Cash Flow From Continuing Operating Activities'])
        capex=dated_values(cf,['Capital Expenditure','Capital Expenditures'])
        fcf_map={date: operating[date]-abs(capex[date]) for date in operating.keys()&capex.keys() if operating[date] is not None and capex[date] is not None}
    if not fcf_map:
        fallback=num(info.get('freeCashflow'))
        if fallback is not None: fcf_map={'fallback':fallback}
    missing=[k for k,v in [('price',price),('shares',shares),('cash',cash),('debt',debt)] if v is None]
    if not fcf_map: missing.append('FCF')
    if missing: raise ValueError('Missing '+', '.join(missing))
    if price<=0 or shares<=0: raise ValueError('Invalid price/shares')
    recent=[v for _,v in sorted(fcf_map.items(),reverse=True)[:5]]
    valid=[v for v in recent if v is not None]
    if len(valid)<3: raise ValueError('Insufficient annual FCF history (<3 years)')
    normalized=float(np.median(valid))
    if normalized<=0: raise ValueError('Non-positive normalized FCF')
    if len(recent)>=3 and all(v is not None for v in (recent[0],recent[-1])) and recent[0]>0 and recent[-1]>0:
        observed=(recent[0]/recent[-1])**(1/(len(recent)-1))-1
        growth=float(np.clip(observed,-.03,growth_cap))
    else: growth=min(.03,growth_cap)
    bear=dcf(normalized,cash,debt,shares,max(-.05,growth-.05),terminal,rate+.01)
    base=dcf(normalized,cash,debt,shares,growth,terminal,rate)
    bull=dcf(normalized,cash,debt,shares,min(growth_cap+.05,growth+.05),terminal,max(terminal+.005,rate-.01))
    if base is None or base<=0: raise ValueError('Non-positive/invalid Base Fair Value')
    roe=first(info.get('returnOnEquity'))
    equity=latest(bs,['Stockholders Equity','Common Stock Equity','Total Equity Gross Minority Interest'])
    if roe is None:
        earnings=latest(inc,['Net Income','Net Income Common Stockholders'])
        roe=earnings/equity if earnings is not None and equity and equity>0 else None
    de=first(info.get('debtToEquity'))
    de=de/100 if de is not None else debt/equity if equity and equity>0 else None
    rev=[v for _,v in sorted(dated_values(inc,['Total Revenue','Operating Revenue']).items(),reverse=True)[:5]]
    earnings=[v for _,v in sorted(dated_values(inc,['Net Income','Net Income Common Stockholders']).items(),reverse=True)[:5]]
    op=latest(cf,['Operating Cash Flow','Cash Flow From Continuing Operating Activities'])
    net=latest(inc,['Net Income','Net Income Common Stockholders'])
    conversion=op/net if op is not None and net is not None and net>0 else None
    score=quality_score(roe,de,recent,rev,earnings,conversion)
    # Confidence measures evidence completeness, not statistical accuracy of the DCF.
    evidence=sum([len(valid)>=4,len(valid)>=5,roe is not None,de is not None,len([x for x in rev if x is not None])>=3,len([x for x in earnings if x is not None])>=3,conversion is not None])
    confidence='High' if evidence>=6 else 'Medium' if evidence>=4 else 'Low'
    upside=(base/price-1)*100
    eligible=upside>=min_upside and score>=min_quality and confidence!='Low'
    decision='พิจารณาจังหวะซื้อ' if eligible and tech['CDC Zone']=='Green' and tech['Technical Score']>=4 and not tech['CDC Sell today'] else 'หุ้นผ่านเกณฑ์ — รอสัญญาณ' if eligible else 'ยังไม่ผ่านเกณฑ์'
    return {'Ticker':symbol,'Company':info.get('shortName') or symbol,'Price ($)':price,'Bear FV ($)':bear,'Base FV ($)':base,'Bull FV ($)':bull,'Fair Price ($)':base,'Upside (%)':upside,'MOS (%)':(1-price/base)*100,'Quality Score':score,'Valuation Confidence':confidence,'Quality Pass':score>=min_quality,'Value Pass':upside>=min_upside,'Eligible':eligible,'Sector':sector,'Model':'FCF DCF (indicative)','Special Review':False,'ROE (%)':roe*100 if roe is not None else None,'D/E (x)':de,'FCF ($B)':normalized/1e9,'FCF years':len(valid),'Growth (%)':growth*100,'Decision':decision,**tech},d
