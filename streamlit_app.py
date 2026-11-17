import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title='US Stock Value Scanner', page_icon='📈', layout='wide')
st.title('📈 US Stock Value Scanner')
st.caption('Educational DCF screener • Yahoo Finance data may be delayed or incomplete')

DEFAULT = 'AAPL,MSFT,GOOGL,AMZN,META,NVDA,JNJ,PG,KO,V,MA,COST,HD,UNH,ADBE,CRM,PEP,ABBV,TXN,LOW'
with st.sidebar:
    st.header('Scanner settings')
    tickers_text = st.text_area('US tickers (comma-separated)', DEFAULT, height=130)
    discount = st.slider('Required upside (%)', 0, 60, 15)
    growth = st.slider('FCF growth for years 1–5 (%)', -5, 20, 6)
    terminal_growth = st.slider('Terminal growth (%)', 0.0, 4.0, 2.5, 0.5)
    discount_rate = st.slider('Discount rate (%)', 6.0, 18.0, 10.0, 0.5)
    min_roe = st.slider('Minimum ROE (%)', 0, 40, 10)
    max_de = st.slider('Maximum debt/equity (x)', 0.0, 5.0, 2.0, 0.25)
    run = st.button('🔎 Scan stocks', type='primary')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(symbol):
    t = yf.Ticker(symbol)
    info = t.info
    cashflow = t.cashflow
    return info, cashflow

def number(x):
    try:
        x = float(x)
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None

def get_fcf(info, cashflow):
    # Prefer annual cash-flow statement; use TTM metadata only as fallback.
    if cashflow is not None and not cashflow.empty:
        for row in ('Free Cash Flow', 'FreeCashFlow'):
            if row in cashflow.index:
                vals = pd.to_numeric(cashflow.loc[row], errors='coerce').dropna()
                if len(vals):
                    return number(vals.iloc[0])
    return number(info.get('freeCashflow'))

def evaluate(symbol, g, tg, r):
    info, cashflow = fetch(symbol)
    price = number(info.get('currentPrice')) or number(info.get('regularMarketPrice'))
    fcf = get_fcf(info, cashflow)
    shares = number(info.get('sharesOutstanding'))
    cash = number(info.get('totalCash'))
    debt = number(info.get('totalDebt'))
    roe = number(info.get('returnOnEquity'))
    de_pct = number(info.get('debtToEquity'))
    if not price or not fcf or fcf <= 0 or not shares or shares <= 0:
        raise ValueError('Missing price/shares or non-positive free cash flow')
    if cash is None or debt is None:
        raise ValueError('Missing cash or debt')
    if r <= tg:
        raise ValueError('Discount rate must exceed terminal growth')
    forecast = [fcf * (1 + g) ** year for year in range(1, 6)]
    pv = sum(value / (1 + r) ** year for year, value in enumerate(forecast, 1))
    terminal = forecast[-1] * (1 + tg) / (r - tg)
    equity_value = pv + terminal / (1 + r) ** 5 + cash - debt
    fair = equity_value / shares
    upside = (fair / price - 1) * 100
    de = de_pct / 100 if de_pct is not None else None
    quality = roe is not None and de is not None and roe >= min_roe / 100 and de <= max_de
    return {'Ticker': symbol, 'Company': info.get('shortName', symbol), 'Price ($)': price,
            'Fair value ($)': fair, 'Upside (%)': upside, 'ROE (%)': roe * 100 if roe is not None else None,
            'D/E (x)': de, 'FCF ($B)': fcf / 1e9, 'Quality pass': quality,
            'Screen pass': quality and upside >= discount}

if run:
    tickers = list(dict.fromkeys(x.strip().upper() for x in tickers_text.replace('\n', ',').split(',') if x.strip()))[:50]
    if not tickers:
        st.warning('Enter at least one ticker.')
    elif discount_rate <= terminal_growth:
        st.error('Discount rate must be greater than terminal growth.')
    else:
        results, errors = [], []
        progress = st.progress(0)
        with st.spinner('Fetching financial data and valuing stocks...'):
            for i, ticker in enumerate(tickers):
                try:
                    results.append(evaluate(ticker, growth / 100, terminal_growth / 100, discount_rate / 100))
                except Exception as exc:
                    errors.append(f'{ticker}: {exc}')
                progress.progress((i + 1) / len(tickers))
        st.session_state['results'] = results
        st.session_state['errors'] = errors

if 'results' in st.session_state:
    df = pd.DataFrame(st.session_state['results'])
    if df.empty:
        st.error('No usable data returned. Try a smaller ticker list or retry later.')
    else:
        matches = df[df['Screen pass']].sort_values('Upside (%)', ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric('Analyzed', len(df))
        c2.metric('Passed screen', len(matches))
        c3.metric('Highest upside', f"{df['Upside (%)'].max():.1f}%")
        st.subheader('Stocks passing all filters')
        st.dataframe(matches, hide_index=True, use_container_width=True)
        st.subheader('All analyzed stocks')
        st.dataframe(df.sort_values('Upside (%)', ascending=False), hide_index=True, use_container_width=True)
        st.download_button('Download CSV', df.to_csv(index=False).encode('utf-8-sig'), 'us_stock_scan.csv', 'text/csv')
        st.subheader('Price vs estimated fair value')
        chart = df.set_index('Ticker')[['Price ($)', 'Fair value ($)']]
        st.bar_chart(chart)
    if st.session_state['errors']:
        with st.expander(f"Skipped / unavailable ({len(st.session_state['errors'])})"):
            for error in st.session_state['errors']:
                st.write(error)
else:
    st.info('Choose your assumptions in the sidebar, then tap Scan stocks.')

st.caption('DCF uses constant five-year FCF growth, terminal value, and net cash adjustment. Financial firms and negative-FCF businesses require other valuation methods. Not investment advice.')
