"""US Stock Value Scanner V6 — final maintenance fix.

Indicative equity cash-flow proxy DCF, not a full FCFE/FCFF valuation.
Annual operating cash flow less capex is used as an equity cash-flow proxy;
principal borrowing/repayment is not modeled. Cost of equity discounts this
proxy; cash and debt are not added/subtracted again. Screening estimates only.
Public evaluate() signature and result columns remain V6-compatible.
"""
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
    """Read annual-statement rows by year-end date; never use quarterly data."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    dates = pd.to_datetime(frame.columns, utc=True, errors='coerce')
    if dates.isna().any() or len(set(dates)) != len(dates):
        return {}
    result = {}
    # Fill gaps across aliases; preserve the first available valid observation.
    for name in names:
        if name not in frame.index:
            continue
        rows = frame.loc[[name]]
        for _, row in rows.iterrows():
            for date, value in zip(dates, row):
                key = date.strftime('%Y-%m-%d')
                candidate = num(value)
                if key not in result or (result[key] is None and candidate is not None):
                    result[key] = candidate
    return result


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
    if d is None or d.empty:
        raise ValueError('Insufficient daily price history')
    d = d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close']).copy()
    if len(d) < 210:
        raise ValueError('Insufficient daily price history')
    return d


def indicators(hist):
    d = hist.copy()
    c, h, l, v = (d[k].astype(float) for k in ('Close', 'High', 'Low', 'Volume'))
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
    d['CDC_zone'] = np.select([green, blue, aqua, red, orange, yellow],
                              ['Green', 'Blue', 'Light Blue', 'Red', 'Orange', 'Yellow'], default='Neutral')
    first_green = green & ~green.shift(1, fill_value=False)
    first_red = red & ~red.shift(1, fill_value=False)
    last_g = last_r = None
    buys, sells, states = [], [], []
    for i, (g, r) in enumerate(zip(first_green, first_red)):
        previous = (1 if last_g is not None and (last_r is None or last_g > last_r)
                    else -1 if last_r is not None and (last_g is None or last_r > last_g) else 0)
        buys.append(bool(g and previous == -1))
        sells.append(bool(r and previous == 1))
        if g:
            last_g = i
        if r:
            last_r = i
        states.append(1 if last_g is not None and (last_r is None or last_g > last_r)
                      else -1 if last_r is not None and (last_g is None or last_r > last_g) else 0)
    d['CDC_buy'], d['CDC_sell'], d['CDC_trend'] = buys, sells, states
    change = c.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d['RSI'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d.loc[(loss == 0) & (gain > 0), 'RSI'] = 100
    d.loc[(loss == 0) & (gain == 0), 'RSI'] = 50
    macd = d.EMA12 - d.EMA26
    d['MACD_hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    typical = (h + l + c) / 3
    positive = (typical * v).where(typical.diff() > 0, 0).rolling(14).sum()
    negative = (typical * v).where(typical.diff() < 0, 0).rolling(14).sum()
    d['MFI'] = 100 - 100 / (1 + positive / negative.replace(0, np.nan))
    d.loc[(negative == 0) & (positive > 0), 'MFI'] = 100
    spread = (h - l).replace(0, np.nan)
    d['CMF'] = (((2 * c - l - h) / spread).fillna(0) * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    d['VolRatio'] = v / v.rolling(20).mean().replace(0, np.nan)
    return d


def technical(d):
    a, b = d.iloc[-1], d.iloc[-2]
    rsi, mfi, cmf, vr = [num(a[k]) for k in ('RSI', 'MFI', 'CMF', 'VolRatio')]
    proxy = ('Accumulation' if cmf is not None and mfi is not None and cmf > .05 and mfi >= 50
             else 'Distribution' if cmf is not None and cmf < -.05 else 'Neutral')
    score = sum([a.CDC_zone == 'Green', bool(a.Close > a.EMA50), bool(a.MACD_hist > b.MACD_hist),
                 rsi is not None and 40 <= rsi <= 70, proxy == 'Accumulation', vr is not None and vr >= 1])
    return {'CDC Zone': str(a.CDC_zone), 'CDC Buy today': bool(a.CDC_buy),
            'CDC Sell today': bool(a.CDC_sell),
            'CDC Trend': 'Bullish' if a.CDC_trend == 1 else 'Bearish' if a.CDC_trend == -1 else 'Undefined',
            'Smart Money Proxy': proxy, 'RSI': rsi, 'MFI': mfi, 'CMF': cmf,
            'Volume / 20d': vr, 'Technical Score': int(score)}


def dcf(fcf, cash, debt, shares, growth, terminal, rate):
    """Five-year equity FCF proxy DCF; cash/debt API args not added/subtracted."""
    vals = [num(x) for x in (fcf, shares, growth, terminal, rate)]
    if any(x is None for x in vals):
        return None
    fcf, shares, growth, terminal, rate = vals
    if fcf <= 0 or shares <= 0 or growth <= -1 or terminal <= -1 or rate <= terminal or rate <= -1:
        return None
    if any(x is not None and num(x) is None for x in (cash, debt)):
        return None
    try:
        flows = [fcf * (1 + growth) ** year for year in range(1, 6)]
        pv = sum(flow / (1 + rate) ** year for year, flow in enumerate(flows, 1))
        tv = flows[-1] * (1 + terminal) / (rate - terminal)
        value = (pv + tv / (1 + rate) ** 5) / shares
    except (OverflowError, ZeroDivisionError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def quality_score(roe, de, fcf_values, revenue_values, earnings_values, cash_conversion):
    """V6 scoring unchanged; annual arrays newest first."""
    score = 0
    if roe is not None:
        score += 15 if roe >= .20 else 10 if roe >= .12 else 5 if roe > 0 else 0
    if cash_conversion is not None:
        score += 10 if cash_conversion >= 1 else 6 if cash_conversion >= .7 else 0
    if de is not None and de >= 0:
        score += 20 if de <= .5 else 14 if de <= 1 else 7 if de <= 2 else 0
    if len(fcf_values) >= 3 and all(x is not None for x in fcf_values):
        score += 15 if all(v > 0 for v in fcf_values) else 7 if sum(v > 0 for v in fcf_values) >= len(fcf_values) - 1 else 0
        score += 10 if fcf_values[0] >= fcf_values[-1] > 0 else 5 if fcf_values[0] > 0 else 0
        if len(fcf_values) >= 4 and all(v > 0 for v in fcf_values):
            score += 10
    for values in (revenue_values, earnings_values):
        if len(values) >= 3 and all(x is not None for x in values):
            score += 10 if values[0] > values[-1] > 0 else 5 if values[0] > 0 else 0
    return min(100, int(score))


def annual_series(values, count=5):
    """Select latest contiguous annual observations, allowing 330–400 days."""
    pairs = sorted(((pd.Timestamp(k), v) for k, v in values.items() if k != 'fallback'), reverse=True)
    if not pairs:
        return []
    result = [pairs[0]]
    for pair in pairs[1:]:
        gap = (result[-1][0] - pair[0]).days
        if not 330 <= gap <= 400:
            break
        result.append(pair)
        if len(result) >= count:
            break
    return result


def annual_fcf_map(cf):
    """Fill missing direct FCF observations with CFO minus absolute capex, by date.

    A zero or negative FCF is a real observation, not a missing value.
    No annual observations are fabricated or borrowed across years.
    """
    direct = dated_values(cf, ['Free Cash Flow', 'FreeCashFlow'])
    operating = dated_values(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
    capex = dated_values(cf, ['Capital Expenditure', 'Capital Expenditures'])
    result = dict(direct)
    for date in operating.keys() & capex.keys():
        if result.get(date) is None and operating[date] is not None and capex[date] is not None:
            result[date] = operating[date] - abs(capex[date])
    return result


def fcf_history_or_error(fcf_map):
    """Fail with actionable diagnostics, without relaxing three-year requirement."""
    if not fcf_map:
        raise ValueError('Missing annual FCF (direct and CFO/capex unavailable)')
    series = annual_series(fcf_map)
    if len(series) < 3 or any(v is None for _, v in series):
        dates = ','.join(f'{d.date()}={"NA" if v is None else "OK"}' for d, v in series)
        raise ValueError(f'Insufficient consecutive annual FCF history (<3 years); found={len(series)}; dates={dates}')
    last_date = series[0][0]
    if (pd.Timestamp.now(tz='UTC').tz_localize(None) - last_date).days > 550:
        raise ValueError('Stale annual FCF statements')
    return series


def evaluate(symbol, growth_cap=.10, rate=.10, terminal=.025, min_upside=20,
             min_quality=70, unused=1.0, sector='Unknown'):
    """V6-compatible positional arguments, return columns and technical signals."""
    if sector in SPECIAL:
        raise ValueError('Special Review: sector-specific valuation required')
    if any(num(x) is None for x in (growth_cap, rate, terminal, min_upside, min_quality)):
        raise ValueError('Invalid valuation parameters')
    if growth_cap < 0 or growth_cap > .50 or rate <= terminal + .01 or rate <= 0 or terminal < 0:
        raise ValueError('Invalid discount/growth parameters')
    d = indicators(history(symbol))
    tech = technical(d)
    info, fast, cf, bs, inc = fundamentals(symbol)
    info = info if isinstance(info, dict) else {}
    fast = fast if isinstance(fast, dict) else {}
    price = first(d.Close.iloc[-1], info.get('currentPrice'), fast.get('lastPrice'))
    shares = first(info.get('sharesOutstanding'), fast.get('shares'),
                   latest(bs, ['Ordinary Shares Number', 'Share Issued']))
    if shares is None:
        cap = first(info.get('marketCap'), fast.get('marketCap'))
        shares = cap / price if cap and price and price > 0 else None
    cash = first(info.get('totalCash'), latest(bs, ['Cash Cash Equivalents And Short Term Investments',
                                                  'Cash And Cash Equivalents', 'Cash And Short Term Investments']))
    debt = first(info.get('totalDebt'), latest(bs, ['Total Debt']))
    fcf_map = annual_fcf_map(cf)
    missing = [k for k, v in [('price', price), ('shares', shares)] if v is None]
    if not fcf_map:
        missing.append('annual FCF')
    if missing:
        raise ValueError('Missing ' + ', '.join(missing))
    if price <= 0 or shares <= 0:
        raise ValueError('Invalid price/shares')
    series = fcf_history_or_error(fcf_map)
    recent = [v for _, v in series]
    normalized = float(np.median(recent))
    if normalized <= 0:
        raise ValueError('Non-positive normalized FCF')
    if recent[0] > 0 and recent[-1] > 0:
        observed = (recent[0] / recent[-1]) ** (1 / (len(recent) - 1)) - 1
        growth = float(np.clip(observed, -.03, growth_cap))
    else:
        growth = min(.03, growth_cap)
    bear = dcf(normalized, cash, debt, shares, max(-.05, growth - .05), terminal, rate + .01)
    base = dcf(normalized, cash, debt, shares, growth, terminal, rate)
    bull = dcf(normalized, cash, debt, shares, min(growth_cap + .05, growth + .05),
               terminal, max(terminal + .005, rate - .01))
    if base is None:
        raise ValueError('Non-positive/invalid Base Fair Value')
    roe = first(info.get('returnOnEquity'))
    equity = latest(bs, ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'])
    if roe is None:
        net_income = latest(inc, ['Net Income', 'Net Income Common Stockholders'])
        roe = net_income / equity if net_income is not None and equity is not None and equity > 0 else None
    de = first(info.get('debtToEquity'))
    de = de / 100 if de is not None else debt / equity if debt is not None and equity is not None and equity > 0 else None
    rev = [v for _, v in sorted(dated_values(inc, ['Total Revenue', 'Operating Revenue']).items(), reverse=True)[:5]]
    earnings = [v for _, v in sorted(dated_values(inc, ['Net Income', 'Net Income Common Stockholders']).items(), reverse=True)[:5]]
    op = latest(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
    net = latest(inc, ['Net Income', 'Net Income Common Stockholders'])
    conversion = op / net if op is not None and net is not None and net > 0 else None
    score = quality_score(roe, de, recent, rev, earnings, conversion)
    evidence = sum([len(recent) >= 4, len(recent) >= 5, roe is not None, de is not None and de >= 0,
                    len([x for x in rev if x is not None]) >= 3,
                    len([x for x in earnings if x is not None]) >= 3, conversion is not None])
    confidence = 'High' if evidence >= 6 else 'Medium' if evidence >= 4 else 'Low'
    upside = (base / price - 1) * 100
    eligible = bool(upside >= min_upside and score >= min_quality and confidence != 'Low')
    decision = ('พิจารณาจังหวะซื้อ' if eligible and tech['CDC Zone'] == 'Green'
                and tech['Technical Score'] >= 4 and not tech['CDC Sell today']
                else 'หุ้นผ่านเกณฑ์ — รอสัญญาณ' if eligible else 'ยังไม่ผ่านเกณฑ์')
    return {'Ticker': symbol, 'Company': info.get('shortName') or symbol,
            'Price ($)': price, 'Bear FV ($)': bear, 'Base FV ($)': base,
            'Bull FV ($)': bull, 'Fair Price ($)': base, 'Upside (%)': upside,
            'MOS (%)': (1 - price / base) * 100, 'Quality Score': score,
            'Valuation Confidence': confidence, 'Quality Pass': bool(score >= min_quality),
            'Value Pass': bool(upside >= min_upside), 'Eligible': eligible,
            'Sector': sector, 'Model': 'FCF DCF (indicative)', 'Special Review': False,
            'ROE (%)': roe * 100 if roe is not None else None, 'D/E (x)': de,
            'FCF ($B)': normalized / 1e9, 'FCF years': len(recent),
            'Growth (%)': growth * 100, 'Decision': decision, **tech}, d
