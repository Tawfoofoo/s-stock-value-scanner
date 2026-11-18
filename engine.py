import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

def num(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError, OverflowError):
        return None

def first(*values):
    for x in values:
        v = num(x)
        if v is not None:
            return v
    return None


def row_values(frame, names):
    if frame is None or frame.empty:
        return []

    # Sort annual statements: newest first.
    frame = frame.copy()
    dates = pd.to_datetime(
        frame.columns, errors="coerce", utc=True
    )

    if dates.isna().any():
        raise ValueError(
            "Cannot verify financial statement dates"
        )

    order = np.argsort(dates.asi8)[::-1]
    frame = frame.iloc[:, order]

    for name in names:
        if name in frame.index:
            s = frame.loc[name]

            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]

            # Preserve missing years as NaN rather than
            # silently shifting the remaining observations.
            return [
                num(v)
                for v in pd.to_numeric(
                    s, errors="coerce"
                )
            ]

    return []


def latest(frame, names):
    v = row_values(frame, names)
    return v[0] if v else None

@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(symbol):
    t = yf.Ticker(symbol)
    out = []
    for attr in ("info", "fast_info", "cashflow", "balance_sheet", "income_stmt"):
        try:
            value = getattr(t, attr)
            out.append(dict(value) if attr == "fast_info" else value)
        except Exception:
            out.append({} if attr in ("info", "fast_info") else pd.DataFrame())
    return tuple(out)

@st.cache_data(ttl=1800, show_spinner=False)
def history(symbol):
    d = yf.Ticker(symbol).history(period="2y", auto_adjust=True)
    if d is None or d.empty:
        raise ValueError("No historical prices")
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    if len(d) < 210:
        raise ValueError("Need at least 210 daily price bars")
    return d

def indicators(hist):
    d = hist.copy()
    c, h, l, v = (d[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    # Pine: xPrice = ema(close, smoothing=1), FastMA = ema(xPrice,12), SlowMA=ema(xPrice,26)
    xprice = c.ewm(span=1, adjust=False).mean()
    d["EMA12"] = xprice.ewm(span=12, adjust=False).mean()
    d["EMA26"] = xprice.ewm(span=26, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    fast, slow = d["EMA12"], d["EMA26"]
    bull, bear = fast > slow, fast < slow
    green = bull & (xprice > fast)
    blue = bear & (xprice > fast) & (xprice > slow)
    aqua = bear & (xprice > fast) & (xprice < slow)
    red = bear & (xprice < fast)
    orange = bull & (xprice < fast) & (xprice < slow)
    yellow = bull & (xprice < fast) & (xprice > slow)
    d["CDC_zone"] = np.select([green, blue, aqua, red, orange, yellow],
                               ["Green", "Blue", "Light Blue", "Red", "Orange", "Yellow"], default="Neutral")
    first_green = green & ~green.shift(1, fill_value=False)
    first_red = red & ~red.shift(1, fill_value=False)
    # Pine buy = bearish[1] and first_green; sell = bullish[1] and first_red.
    # Before both events have appeared, the state is undefined; do not manufacture a signal.
    last_green, last_red = None, None
    buys, sells, states = [], [], []
    for i, (g, r) in enumerate(zip(first_green.to_numpy(), first_red.to_numpy())):
        previous = (1 if last_green is not None and (last_red is None or last_green > last_red)
                    else -1 if last_red is not None and (last_green is None or last_red > last_green)
                    else 0)
        buys.append(bool(g and previous == -1))
        sells.append(bool(r and previous == 1))
        if g:
            last_green = i
        if r:
            last_red = i
        states.append(1 if last_green is not None and (last_red is None or last_green > last_red)
                      else -1 if last_red is not None and (last_green is None or last_red > last_green) else 0)
    d["CDC_buy"], d["CDC_sell"], d["CDC_trend"] = buys, sells, states
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)
    d.loc[(loss == 0) & (gain > 0), "RSI"] = 100
    d.loc[(loss == 0) & (gain == 0), "RSI"] = 50
    macd = d["EMA12"] - d["EMA26"]
    d["MACD_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    typical = (h+l+c)/3
    raw = typical*v
    positive = raw.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = raw.where(typical.diff() < 0, 0).rolling(14).sum()
    d["MFI"] = 100 - 100/(1+positive/negative.replace(0, np.nan))
    d.loc[(negative == 0) & (positive > 0), "MFI"] = 100
    spread = (h-l).replace(0, np.nan)
    d["CMF"] = ((((c-l)-(h-c))/spread).fillna(0)*v).rolling(20).sum()/v.rolling(20).sum().replace(0, np.nan)
    d["VolRatio"] = v/v.rolling(20).mean().replace(0, np.nan)
    return d

def technical(d):
    a, b = d.iloc[-1], d.iloc[-2]
    rsi, mfi, cmf, vr = [num(a[k]) for k in ("RSI", "MFI", "CMF", "VolRatio")]
    proxy = ("Accumulation" if cmf is not None and mfi is not None and cmf > .05 and mfi >= 50
             else "Distribution" if cmf is not None and cmf < -.05 else "Neutral")
    improving = bool(a["MACD_hist"] > b["MACD_hist"])
    score = sum([a["CDC_zone"] == "Green", bool(a["Close"] > a["EMA50"]), improving,
                 rsi is not None and 40 <= rsi <= 70, proxy == "Accumulation", vr is not None and vr >= 1])
    return {"CDC Zone": str(a["CDC_zone"]), "CDC Buy today": bool(a["CDC_buy"]),
            "CDC Sell today": bool(a["CDC_sell"]), "CDC Trend": "Bullish" if a["CDC_trend"] == 1 else "Bearish" if a["CDC_trend"] == -1 else "Undefined",
            "Smart Money Proxy": proxy, "RSI": rsi, "MFI": mfi, "CMF": cmf, "Volume / 20d": vr,
            "Technical Score": score}

def dcf(fcf, cash, debt, shares, growth, terminal, rate):
    if not (fcf > 0 and shares > 0 and rate > terminal):
        return None
    flows = [fcf*(1+growth)**i for i in range(1, 6)]
    pv = sum(f/(1+rate)**i for i, f in enumerate(flows, 1))
    tv = flows[-1]*(1+terminal)/(rate-terminal)
    return (pv + tv/(1+rate)**5 + cash - debt)/shares

def evaluate(symbol, growth_cap, rate, terminal, mos_min, min_roe, max_de, sector="Unknown"):
    if sector in ("Financials", "Real Estate", "Utilities"):
        raise ValueError("Special Review: requires sector-specific valuation; generic FCF DCF skipped")
    d = indicators(history(symbol))
    tech = technical(d)
    info, fast, cf, bs, inc = fundamentals(symbol)
    price = first(d["Close"].iloc[-1], info.get("currentPrice"), fast.get("lastPrice"))
    shares = first(info.get("sharesOutstanding"), fast.get("shares"), latest(bs, ["Ordinary Shares Number", "Share Issued"]))
    if shares is None:
        cap = first(info.get("marketCap"), fast.get("marketCap"))
        shares = cap/price if cap and price else None
    cash = first(info.get("totalCash"), latest(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash And Short Term Investments"]))
    debt = first(info.get("totalDebt"), latest(bs, ["Total Debt"]))
    fcfs = row_values(cf, ["Free Cash Flow", "FreeCashFlow"])
    if not fcfs:
        op = row_values(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        capex = row_values(cf, ["Capital Expenditure", "Capital Expenditures"])
        if op and capex:
            fcfs = [op[i] - abs(capex[i]) for i in range(min(len(op), len(capex)))]
    if not fcfs:
        f = num(info.get("freeCashflow"))
        fcfs = [f] if f is not None else []
    missing = [name for name, val in (("price", price), ("shares", shares), ("cash", cash), ("debt", debt)) if val is None]
    if not fcfs:
        missing.append("FCF")
    if missing:
        raise ValueError("Missing " + ", ".join(missing))
    if price <= 0 or shares <= 0:
        raise ValueError("Invalid price or shares")
    # Do not discard negative years when calculating a normalized FCF.
    recent = fcfs[:5]
    normalized = float(np.median(recent))
    if normalized <= 0:
        raise ValueError("Non-positive normalized FCF; DCF unsuitable")
    if len(recent) >= 3 and recent[0] > 0 and recent[-1] > 0:
        observed = (recent[0]/recent[-1])**(1/(len(recent)-1)) - 1
        base_growth = float(np.clip(observed, -.03, growth_cap))
    else:
        base_growth = min(.05, growth_cap)
    bear = dcf(normalized, cash, debt, shares, max(-.05, base_growth-.05), terminal, rate+.01)
    base = dcf(normalized, cash, debt, shares, base_growth, terminal, rate)
    bull = dcf(normalized, cash, debt, shares, min(growth_cap+.05, base_growth+.05), terminal, max(terminal+.005, rate-.01))
    if base is None:
        raise ValueError("Invalid DCF assumptions")
    roe = first(info.get("returnOnEquity"))
    equity = latest(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    if roe is None:
        earnings = latest(inc, ["Net Income", "Net Income Common Stockholders"])
        roe = earnings/equity if earnings is not None and equity and equity > 0 else None
    de = first(info.get("debtToEquity"))
    de = de/100 if de is not None else (debt/equity if equity and equity > 0 else None)
    upside = (base/price-1)*100
    # Sector-specific provisional quality rules. Financials/REITs/utilities are
    # not assigned a generic FCF DCF fair value as if comparable to industrial firms.
    special = sector in ("Financials", "Real Estate", "Utilities")
    thresholds = {
        "Information Technology": (12, 2.0), "Communication Services": (10, 3.0),
        "Consumer Discretionary": (10, 2.5), "Consumer Staples": (10, 3.0),
        "Industrials": (10, 3.0), "Health Care": (10, 2.5),
        "Energy": (8, 2.5), "Materials": (8, 2.5),
    }
    roe_threshold, de_threshold = thresholds.get(sector, (min_roe, max_de))
    # D/E is a fallback leverage proxy, NOT net debt/EBITDA or ROIC.
    quality = (not special and roe is not None and de is not None
               and roe * 100 >= roe_threshold and 0 <= de <= de_threshold)
    mos = (1 - price / base) * 100 if base > 0 else None
    value = mos is not None and mos >= mos_min
    if quality and value and tech["CDC Zone"] == "Green" and tech["Technical Score"] >= 4 and not tech["CDC Sell today"]:
        decision = "พิจารณาจังหวะซื้อ"
    elif quality and value:
        decision = "หุ้นคุณค่า — รอสัญญาณ"
    else:
        decision = "ยังไม่ผ่านเกณฑ์"
    return {"Ticker": symbol, "Company": info.get("shortName") or symbol, "Price ($)": price,
            "Bear FV ($)": bear, "Base FV ($)": base, "Bull FV ($)": bull,
            "Upside (%)": upside, "MOS (%)": mos, "Sector": sector, "Model": "FCF DCF (indicative)", "Special Review": special,
            "ROE (%)": roe*100 if roe is not None else None, "D/E (x)": de,
            "FCF ($B)": normalized/1e9, "FCF years": len(recent), "Growth (%)": base_growth*100,
            "Quality Pass": quality, "Value Pass": value, "Decision": decision, **tech}, d

