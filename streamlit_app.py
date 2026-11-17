import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="US Stock Value Scanner V3", page_icon="📈", layout="wide")
st.title("📈 US Stock Value Scanner V3")
st.caption("DCF • CDC ActionZone V3 (daily, standard mode) • Smart Money Proxy | เครื่องมือวิจัย ไม่ใช่คำแนะนำลงทุน")
DEFAULT = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JNJ,PG,KO,V,MA,COST,HD,ADBE,CRM,PEP,ABBV,TXN,LOW"

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
    for name in names:
        if name in frame.index:
            s = frame.loc[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            vals = pd.to_numeric(s, errors="coerce").dropna()
            return [float(v) for v in vals if num(v) is not None]
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

def evaluate(symbol, growth_cap, rate, terminal, upside_min, min_roe, max_de):
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
    quality = roe is not None and de is not None and roe*100 >= min_roe and 0 <= de <= max_de
    value = upside >= upside_min
    if quality and value and tech["CDC Zone"] == "Green" and tech["Technical Score"] >= 4 and not tech["CDC Sell today"]:
        decision = "พิจารณาจังหวะซื้อ"
    elif quality and value:
        decision = "หุ้นคุณค่า — รอสัญญาณ"
    else:
        decision = "ยังไม่ผ่านเกณฑ์"
    return {"Ticker": symbol, "Company": info.get("shortName") or symbol, "Price ($)": price,
            "Bear FV ($)": bear, "Base FV ($)": base, "Bull FV ($)": bull,
            "Upside (%)": upside, "MOS (%)": (1-price/base)*100 if base > 0 else None,
            "ROE (%)": roe*100 if roe is not None else None, "D/E (x)": de,
            "FCF ($B)": normalized/1e9, "FCF years": len(recent), "Growth (%)": base_growth*100,
            "Quality Pass": quality, "Value Pass": value, "Decision": decision, **tech}, d

with st.sidebar:
    st.header("⚙️ ตั้งค่าการสแกน")
    ticker_text = st.text_area("รายชื่อหุ้น (คั่นด้วยจุลภาค)", DEFAULT, height=130)
    upside_min = st.slider("Upside ขั้นต่ำ (%)", 0, 60, 20)
    growth_cap = st.slider("เพดาน FCF growth (%)", 0, 25, 10)/100
    rate = st.slider("Discount rate (%)", 7.0, 18.0, 10.0, .5)/100
    terminal = st.slider("Terminal growth (%)", 0.0, 4.0, 2.5, .5)/100
    min_roe = st.slider("ROE ขั้นต่ำ (%)", 0, 40, 15)
    max_de = st.slider("D/E สูงสุด (เท่า)", 0.0, 5.0, 1.0, .25)
    run = st.button("🔎 Scan stocks", type="primary", use_container_width=True)
    st.caption("สูงสุด 50 หุ้นต่อครั้ง • Yahoo Finance อาจจำกัดการเข้าถึง")

if run:
    symbols = list(dict.fromkeys(s.strip().upper() for s in ticker_text.replace("\n", ",").split(",") if s.strip()))[:50]
    if not symbols:
        st.warning("กรุณาใส่ชื่อหุ้น")
    elif rate <= terminal+.01:
        st.error("Discount rate ต้องสูงกว่า terminal growth มากกว่า 1%")
    else:
        rows, errors, charts = [], [], {}
        progress = st.progress(0)
        for i, symbol in enumerate(symbols):
            try:
                result, chart = evaluate(symbol, growth_cap, rate, terminal, upside_min, min_roe, max_de)
                rows.append(result)
                charts[symbol] = chart
            except Exception as exc:
                errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            progress.progress((i+1)/len(symbols))
        st.session_state["v3_rows"] = rows
        st.session_state["v3_errors"] = errors
        st.session_state["v3_charts"] = charts
        st.session_state["v3_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

if "v3_rows" not in st.session_state:
    st.info("เปิดเมนูด้านซ้าย เลือกหุ้น แล้วกด Scan stocks")
else:
    df = pd.DataFrame(st.session_state["v3_rows"])
    if df.empty:
        st.error("ยังไม่มีข้อมูลที่ใช้คำนวณได้ โปรดดูข้อผิดพลาดด้านล่าง")
    else:
        shortlist = df[df["Quality Pass"] & df["Value Pass"]]
        a, b, c = st.columns(3)
        a.metric("วิเคราะห์แล้ว", len(df))
        b.metric("ผ่าน Value + Quality", len(shortlist))
        c.metric("จังหวะซื้อที่น่าศึกษา", int((df["Decision"] == "พิจารณาจังหวะซื้อ").sum()))
        tab1, tab2, tab3 = st.tabs(["🎯 หุ้นน่าสนใจ", "📊 วิเคราะห์รายตัว", "📋 ข้อมูลทั้งหมด"])
        with tab1:
            st.subheader("หุ้นผ่านเกณฑ์มูลค่าและคุณภาพ")
            if shortlist.empty:
                st.info("ยังไม่มีหุ้นผ่านเกณฑ์ ลองเพิ่มรายชื่อหรือทบทวนสมมติฐาน")
            else:
                for _, r in shortlist.sort_values(["Technical Score", "Upside (%)"], ascending=False).iterrows():
                    with st.container(border=True):
                        st.markdown(f"### {r['Ticker']} · {r['Company']}")
                        st.write(f"**{r['Decision']}** | CDC: {r['CDC Zone']} | Smart Money Proxy: {r['Smart Money Proxy']}")
                        p, fv, up = st.columns(3)
                        p.metric("ราคา", f"${r['Price ($)']:,.2f}")
                        fv.metric("Fair Value", f"${r['Base FV ($)']:,.2f}")
                        up.metric("Upside", f"{r['Upside (%)']:+.1f}%")
                        st.caption(f"ROE {r['ROE (%)']:.1f}% · D/E {r['D/E (x)']:.2f}x · Technical {r['Technical Score']}/6")
        with tab2:
            selected = st.selectbox("เลือกหุ้น", df["Ticker"].tolist())
            r = df.loc[df["Ticker"] == selected].iloc[0]
            st.subheader(f"{selected} · {r['Company']}")
            st.write(f"**ผลประเมิน: {r['Decision']}**")
            p, fv, up = st.columns(3)
            p.metric("ราคาล่าสุด", f"${r['Price ($)']:,.2f}")
            fv.metric("Fair Value Base", f"${r['Base FV ($)']:,.2f}")
            up.metric("Upside", f"{r['Upside (%)']:+.1f}%")
            st.write(f"DCF Bear: **${r['Bear FV ($)']:,.2f}** · Base: **${r['Base FV ($)']:,.2f}** · Bull: **${r['Bull FV ($)']:,.2f}**" if pd.notna(r["Bear FV ($)"]) and pd.notna(r["Bull FV ($)"]) else "DCF scenarios incomplete")
            st.write(f"CDC: **{r['CDC Zone']}** · Trend: **{r['CDC Trend']}** · Buy signal today: **{r['CDC Buy today']}** · Sell signal today: **{r['CDC Sell today']}**")
            st.write(f"Smart Money **Proxy**: **{r['Smart Money Proxy']}** · RSI: **{r['RSI']:.1f}** · Score: **{r['Technical Score']}/6**" if pd.notna(r["RSI"]) else "Technical indicators incomplete")
            d = st.session_state["v3_charts"][selected].tail(180)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"))
            for col in ("EMA12", "EMA26", "EMA50"):
                fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, mode="lines"))
            buys = d[d["CDC_buy"]]
            sells = d[d["CDC_sell"]]
            fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"]*.99, mode="markers", name="CDC BUY", marker=dict(symbol="triangle-up", size=12, color="green")))
            fig.add_trace(go.Scatter(x=sells.index, y=sells["High"]*1.01, mode="markers", name="CDC SELL", marker=dict(symbol="triangle-down", size=12, color="red")))
            fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=15, b=5))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("CDC คำนวณบนกราฟรายวันแบบปกติ; สัญญาณเกิดหลังปิดแท่ง และ Pine อาจต่างในช่วง warm-up")
        with tab3:
            st.dataframe(df.sort_values("Upside (%)", ascending=False), hide_index=True, use_container_width=True)
            st.download_button("⬇️ ดาวน์โหลด CSV", df.to_csv(index=False).encode("utf-8-sig"), "us_stock_scanner_v3.csv", "text/csv")
    errors = st.session_state.get("v3_errors", [])
    if errors:
        with st.expander(f"Skipped / unavailable ({len(errors)})"):
            for error in errors:
                st.write(error)
    st.caption("Last scan: " + st.session_state.get("v3_time", ""))

with st.expander("วิธีคำนวณและข้อจำกัดสำคัญ"):
    st.markdown("""**DCF:** ใช้ median FCF ย้อนหลังสูงสุด 5 ปี โดยไม่ตัดปีติดลบทิ้ง; คาดการณ์ 5 ปีและ Terminal Value จากนั้นบวกเงินสด หักหนี้ หารจำนวนหุ้น ค่า Growth เป็นสมมติฐานจาก FCF CAGR ที่จำกัดเพดาน ไม่ใช่ประมาณการนักวิเคราะห์ และไม่เหมาะกับทุกอุตสาหกรรม

**CDC ActionZone:** แปลงตรรกะ 6 สีและเงื่อนไข buy/sell จาก Pine Script ที่ผู้ใช้ส่งมา สำหรับ Source=Close, Smoothing=1, EMA12/26, กราฟรายวันแบบไม่ Fixed TF เท่านั้น ไม่รวม StochRSI optional; ควรตรวจเทียบกับ TradingView ก่อนใช้งานจริง

**Smart Money Proxy:** ใช้ CMF20, MFI14 และ Volume ratio ไม่ใช่ SmartMCDX ต้นฉบับ และไม่ได้ตรวจจับคำสั่งซื้อขายของสถาบันโดยตรง

**ข้อมูล:** Yahoo Finance อาจล่าช้า/ขาดหาย ราคาปิดปรับปรุงอาจไม่ตรงกับข้อมูลหุ้นและงบในวันเดียวกัน ROE/D/E จากผู้ให้บริการอาจมีวิธีนิยามต่างกัน การคำนวณยังไม่ใช่ backtest หรือคำแนะนำซื้อขาย ควรตรวจงบและสมมติฐานด้วยตนเอง""")

