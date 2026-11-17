import math
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='US Stock Value Scanner V2', page_icon='📈', layout='wide')
st.title('📈 US Stock Value Scanner V2')
st.caption('DCF + CDC Action Zone approximation + Smart Money Proxy | Educational research, not a trading recommendation')

DEFAULT = 'AAPL,MSFT,GOOGL,AMZN,META,NVDA,JNJ,PG,KO,V,MA,COST,HD,ADBE,CRM,PEP,ABBV,TXN,LOW'

def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None

def first(*values):
    return next((v for x in values if (v := num(x)) is not None), None)

def row_values(frame, names):
    if frame is None or frame.empty:
        return []
    for name in names:
        if name in frame.index:
            series = frame.loc[name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[0]
            vals = pd.to_numeric(series, errors='coerce').dropna()
            return [float(v) for v in vals.values if num(v) is not None]
    return []

def statement_latest(frame, names):
    values = row_values(frame, names)
    return values[0] if values else None

@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    try:
        fast = dict(ticker.fast_info)
    except Exception:
        fast = {}
    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()
    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        income = ticker.income_stmt
    except Exception:
        income = pd.DataFrame()
    return info, fast, cashflow, balance, income

@st.cache_data(ttl=1800, show_spinner=False)
def history(symbol):
    hist = yf.Ticker(symbol).history(period='2y', auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError('No historical OHLCV prices')
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close']).copy()
    if len(hist) < 60:
        raise ValueError('Fewer than 60 daily prices')
    return hist

def indicators(hist):
    d = hist.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume'].astype(float)
    d['EMA12'] = c.ewm(span=12, adjust=False).mean()
    d['EMA26'] = c.ewm(span=26, adjust=False).mean()
    d['EMA50'] = c.ewm(span=50, adjust=False).mean()
    d['EMA200'] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100 - 100 / (1 + rs)
    macd = d['EMA12'] - d['EMA26']
    signal = macd.ewm(span=9, adjust=False).mean()
    d['MACD_hist'] = macd - signal
    typical = (h + l + c) / 3
    raw = typical * v
    positive = raw.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = raw.where(typical.diff() < 0, 0).rolling(14).sum()
    d['MFI'] = 100 - 100 / (1 + positive / negative.replace(0, np.nan))
    spread = (h - l).replace(0, np.nan)
    multiplier = ((c - l) - (h - c)) / spread
    money_volume = multiplier.fillna(0) * v
    d['CMF'] = money_volume.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    d['VolRatio'] = v / v.rolling(20).mean().replace(0, np.nan)
    # Transparent EMA-based CDC approximation; not the proprietary/original CDC script.
    d['CDC_zone'] = np.select(
        [(c > d['EMA12']) & (d['EMA12'] > d['EMA26']),
         (c < d['EMA12']) & (d['EMA12'] < d['EMA26'])],
        ['Green', 'Red'], default='Yellow')
    return d

def technical(d):
    a, b = d.iloc[-1], d.iloc[-2]
    zone = str(a['CDC_zone'])
    rsi, mfi, cmf, vr = [num(a[k]) for k in ['RSI','MFI','CMF','VolRatio']]
    macd_up = bool(a['MACD_hist'] > b['MACD_hist'])
    trend = bool(a['Close'] > a['EMA50'])
    long_trend = bool(a['Close'] > a['EMA200']) if len(d) >= 200 else None
    proxy = 'Accumulation' if cmf is not None and mfi is not None and cmf > 0.05 and mfi >= 50 else ('Distribution' if cmf is not None and cmf < -0.05 else 'Neutral')
    score = sum([zone == 'Green', trend, macd_up,
                 rsi is not None and 40 <= rsi <= 70,
                 proxy == 'Accumulation',
                 vr is not None and vr >= 1.0])
    return {'CDC zone (proxy)':zone,'Smart money (proxy)':proxy,
            'RSI':rsi,'MFI':mfi,'CMF':cmf,'Vol/20d':vr,
            'MACD improving':macd_up,'Above EMA50':trend,
            'Above EMA200':long_trend,'Technical score /6':score}

def dcf(fcf, cash, debt, shares, growth, terminal_growth, discount_rate):
    if not (fcf > 0 and shares > 0 and discount_rate > terminal_growth):
        return None
    projections = [fcf * (1 + growth)**year for year in range(1,6)]
    pv = sum(f / (1 + discount_rate)**i for i,f in enumerate(projections,1))
    terminal = projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return (pv + terminal / (1 + discount_rate)**5 + cash - debt) / shares

def evaluate(symbol, growth_cap, rate, terminal, upside_min, min_roe, max_de):
    d = indicators(history(symbol))
    tech = technical(d)
    info, fast, cf, bs, inc = fundamentals(symbol)
    price = first(d['Close'].iloc[-1], info.get('currentPrice'), fast.get('lastPrice'))
    shares = first(info.get('sharesOutstanding'), fast.get('shares'),
                   statement_latest(bs, ['Ordinary Shares Number','Share Issued']))
    if shares is None:
        market_cap = first(info.get('marketCap'), fast.get('marketCap'))
        shares = market_cap / price if market_cap and price else None
    cash = first(info.get('totalCash'), statement_latest(bs, [
        'Cash Cash Equivalents And Short Term Investments','Cash And Cash Equivalents',
        'Cash And Short Term Investments']))
    debt = first(info.get('totalDebt'), statement_latest(bs, ['Total Debt']))
    fcfs = row_values(cf, ['Free Cash Flow','FreeCashFlow'])
    if not fcfs:
        operating = row_values(cf, ['Operating Cash Flow','Cash Flow From Continuing Operating Activities'])
        capex = row_values(cf, ['Capital Expenditure','Capital Expenditures'])
        if operating and capex:
            fcfs = [operating[i] + capex[i] for i in range(min(len(operating),len(capex)))]
    if not fcfs:
        val = num(info.get('freeCashflow'))
        fcfs = [val] if val is not None else []
    if not price or not shares or shares <= 0 or not fcfs or cash is None or debt is None:
        missing = [k for k,v in [('price',price),('shares',shares),('FCF',fcfs),('cash',cash),('debt',debt)] if v is None or v == []]
        raise ValueError('Missing '+', '.join(missing))
    positive = [v for v in fcfs[:5] if v > 0]
    if not positive:
        raise ValueError('Non-positive historical FCF; DCF not applicable')
    normalized_fcf = float(np.median(positive))
    if len(positive) >= 3:
        # Observed CAGR is noisy; bound the inferred growth assumption.
        oldest = positive[-1]
        observed = (positive[0] / oldest)**(1/(len(positive)-1)) - 1 if oldest > 0 else 0
        base_g = float(np.clip(observed, -0.03, growth_cap))
    else:
        base_g = min(0.05, growth_cap)
    bear_g = max(-0.05, base_g - 0.05)
    bull_g = min(growth_cap + 0.05, base_g + 0.05)
    fair_bear = dcf(normalized_fcf, cash, debt, shares, bear_g, terminal, rate + 0.01)
    fair_base = dcf(normalized_fcf, cash, debt, shares, base_g, terminal, rate)
    fair_bull = dcf(normalized_fcf, cash, debt, shares, bull_g, terminal, max(terminal + 0.005,rate - 0.01))
    if fair_base is None:
        raise ValueError('Invalid DCF assumptions')
    roe = first(info.get('returnOnEquity'))
    if roe is None:
        earnings = statement_latest(inc, ['Net Income','Net Income Common Stockholders'])
        equity = statement_latest(bs, ['Stockholders Equity','Total Equity Gross Minority Interest','Common Stock Equity'])
        roe = earnings / equity if earnings is not None and equity and equity > 0 else None
    de = first(info.get('debtToEquity'))
    if de is not None:
        de = de / 100
    else:
        equity = statement_latest(bs, ['Stockholders Equity','Common Stock Equity'])
        de = debt / equity if equity and equity > 0 else None
    upside = (fair_base / price - 1) * 100
    quality = roe is not None and de is not None and roe * 100 >= min_roe and 0 <= de <= max_de
    value_pass = upside >= upside_min
    if quality and value_pass and tech['Technical score /6'] >= 4 and tech['CDC zone (proxy)'] == 'Green':
        decision = 'Buy candidate'
    elif quality and value_pass:
        decision = 'Value watchlist'
    else:
        decision = 'Wait / avoid'
    return {'Ticker':symbol,'Company':info.get('shortName') or symbol,'Price ($)':price,
            'Bear FV ($)':fair_bear,'Base FV ($)':fair_base,'Bull FV ($)':fair_bull,
            'Upside (%)':upside,'ROE (%)':roe*100 if roe is not None else None,
            'D/E (x)':de,'Normalized FCF ($B)':normalized_fcf/1e9,
            'FCF years':len(fcfs),'Base growth (%)':base_g*100,
            'Quality pass':quality,'Value pass':value_pass,'Decision':decision,
            **tech}, d

with st.sidebar:
    st.header('Scan settings')
    tickers_text = st.text_area('US tickers (comma/newline-separated)',DEFAULT,height=130)
    upside_min = st.slider('Minimum DCF upside (%)',0,60,15)
    growth_cap = st.slider('Maximum inferred FCF growth (%)',0,25,12) / 100
    rate = st.slider('Base discount rate (%)',7.0,18.0,10.0,0.5) / 100
    terminal = st.slider('Terminal growth (%)',0.0,4.0,2.5,0.5) / 100
    min_roe = st.slider('Minimum ROE (%)',0,40,15)
    max_de = st.slider('Maximum D/E (x)',0.0,5.0,2.0,0.25)
    run = st.button('🔎 Scan stocks',type='primary',use_container_width=True)
    st.caption('Max 50 tickers per run. Free Yahoo data can be incomplete or rate-limited.')

if run:
    symbols = list(dict.fromkeys(s.strip().upper() for s in tickers_text.replace('\n',',').split(',') if s.strip()))[:50]
    if not symbols:
        st.warning('Enter at least one ticker')
    elif rate <= terminal + 0.01:
        st.error('Discount rate must exceed terminal growth by more than 1 percentage point')
    else:
        rows, errors, charts = [], [], {}
        bar = st.progress(0)
        for i, symbol in enumerate(symbols):
            try:
                row, hist = evaluate(symbol,growth_cap,rate,terminal,upside_min,min_roe,max_de)
                rows.append(row)
                charts[symbol] = hist
            except Exception as exc:
                errors.append(f'{symbol}: {type(exc).__name__}: {exc}')
            bar.progress((i+1)/len(symbols))
        st.session_state['v2rows'] = rows
        st.session_state['v2errors'] = errors
        st.session_state['v2charts'] = charts
        st.session_state['v2time'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

if 'v2rows' in st.session_state:
    df = pd.DataFrame(st.session_state['v2rows'])
    if df.empty:
        st.error('No usable data. Open Skipped / unavailable for diagnostic details.')
    else:
        passing = df[(df['Quality pass']) & (df['Value pass'])].sort_values('Technical score /6',ascending=False)
        a,b,c = st.columns(3)
        a.metric('Analyzed',len(df))
        b.metric('Value + quality pass',len(passing))
        c.metric('Buy candidates',int((df['Decision']=='Buy candidate').sum()))
        st.subheader('Value + quality shortlist')
        st.dataframe(passing,hide_index=True,use_container_width=True)
        st.subheader('All analyzed stocks')
        st.dataframe(df.sort_values('Upside (%)',ascending=False),hide_index=True,use_container_width=True)
        st.download_button('Download CSV',df.to_csv(index=False).encode('utf-8-sig'),'us_scanner_v2.csv','text/csv')
        st.subheader('DCF valuation scenarios')
        fig = go.Figure()
        for col in ['Price ($)','Bear FV ($)','Base FV ($)','Bull FV ($)']:
            fig.add_bar(name=col,x=df['Ticker'],y=df[col])
        fig.update_layout(barmode='group',height=400,margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.subheader('Technical entry chart')
        selected = st.selectbox('Choose ticker',df['Ticker'].tolist())
        d = st.session_state['v2charts'].get(selected)
        if d is not None:
            d = d.tail(180)
            fig2 = go.Figure()
            for key in ['Close','EMA12','EMA26','EMA50']:
                fig2.add_scatter(x=d.index,y=d[key],name=key,mode='lines')
            fig2.update_layout(height=420,margin=dict(l=10,r=10,t=20,b=20))
            st.plotly_chart(fig2,use_container_width=True)
            selected_row = df.loc[df['Ticker']==selected].iloc[0]
            st.write(f"**CDC EMA proxy:** {selected_row['CDC zone (proxy)']} | **Smart money proxy:** {selected_row['Smart money (proxy)']} | **RSI:** {selected_row['RSI']:.1f} | **Technical score:** {int(selected_row['Technical score /6'])}/6" if pd.notna(selected_row['RSI']) else 'Technical indicators incomplete')
    errors = st.session_state.get('v2errors',[])
    if errors:
        with st.expander(f'Skipped / unavailable ({len(errors)})'):
            for error in errors:
                st.write(error)
    st.caption('Last scan: '+st.session_state.get('v2time',''))
else:
    st.info('Open the sidebar (»), choose settings, and tap Scan stocks.')

with st.expander('Methodology and important limitations'):
    st.markdown('''**DCF:** normalized FCF = median of available positive annual FCF values (up to five); base growth inferred from available FCF history and capped; bear/base/bull vary growth and discount rate. Equity value adds cash and subtracts debt. Missing financial data are never replaced with zero.\n\n**CDC Action Zone proxy:** EMA12/EMA26 + price relationship. This is a transparent approximation, **not** a verified reproduction of a proprietary CDC Action Zone script.\n\n**Smart Money Proxy:** CMF(20), MFI(14), and volume relative to 20-day average. This is **not Smart MCDX**, does not identify institutional trades, and should not be interpreted as actual smart-money ownership.\n\n**Quality:** requires ROE and D/E; missing values fail the quality filter. Financial companies, negative FCF businesses, share-class complexities, and high-growth businesses may require other models. Price data are adjusted daily closes and can be delayed. No guarantee of profitable signals. Verify all data before trading.''')
