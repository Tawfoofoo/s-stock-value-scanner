from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import evaluate, history, indicators
from universe import load_sp500

st.set_page_config(page_title='US Stock Value Scanner V5', page_icon='📈', layout='wide')
st.title('📈 US Stock Value Scanner V5')
st.caption('S&P 500 • Quality + Fair Value • MOS ≥ 15% • Upside • CDC ActionZone • เพื่อการศึกษาเท่านั้น')


@st.cache_data(ttl=86400, show_spinner=False)
def universe():
    return load_sp500()


def boolean_column(frame, name):
    if name not in frame:
        return pd.Series(False, index=frame.index)
    return frame[name].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def numeric_column(frame, name):
    if name not in frame:
        return pd.Series(float('nan'), index=frame.index)
    return pd.to_numeric(frame[name], errors='coerce')


def shortlist(frame, threshold):
    if frame.empty:
        return frame.copy()
    mos = numeric_column(frame, 'MOS (%)')
    mask = boolean_column(frame, 'Quality Pass') & boolean_column(frame, 'Value Pass') & mos.ge(threshold)
    return frame.loc[mask].sort_values('MOS (%)', ascending=False).copy()


def money(value):
    number = pd.to_numeric(value, errors='coerce')
    return f'${number:,.2f}' if pd.notna(number) else 'N/A'


def percent(value):
    number = pd.to_numeric(value, errors='coerce')
    return f'{number:,.1f}%' if pd.notna(number) else 'N/A'


try:
    members, source = universe()
except Exception as exc:
    st.error(f'โหลดรายชื่อ S&P 500 ไม่สำเร็จ: {exc}')
    st.stop()

with st.sidebar:
    st.header('⚙️ ตัวกรอง V5')
    st.caption(f'Universe: {len(members)} tickers • {source}')
    sector_options = ['All'] + sorted(members['Sector'].dropna().astype(str).unique().tolist())
    sector = st.selectbox('Sector', sector_options)
    candidates = members if sector == 'All' else members[members['Sector'] == sector]
    mode = st.radio('วิธีเลือกหุ้น', ['S&P 500 (แบ่งชุด)', 'กำหนด Ticker เอง'])
    if mode == 'S&P 500 (แบ่งชุด)':
        batch = st.number_input('ลำดับชุด (เริ่มที่ 1)', min_value=1, value=1, step=1)
        size = st.slider('จำนวนหุ้นต่อชุด', 5, 30, 10, 5)
        start = (batch - 1) * size
        selection = candidates.iloc[start:start + size]
        st.caption(f'ชุดนี้: {len(selection)} หุ้น จากทั้งหมด {len(candidates)}')
    else:
        symbols = st.text_area('Tickers', 'AAPL,MSFT,ADBE')
        requested = list(dict.fromkeys(
            s.strip().upper().replace('.', '-')
            for s in symbols.replace('\n', ',').split(',') if s.strip()
        ))[:30]
        selection = pd.DataFrame({'Symbol': requested}).merge(members, on='Symbol', how='left')
        selection['Sector'] = selection['Sector'].fillna('Unknown')

    mos_min = st.slider('MOS ขั้นต่ำ (%)', 0, 50, 15)
    growth_cap = st.slider('เพดาน FCF growth (%)', 0, 25, 10) / 100
    rate = st.slider('Discount rate (%)', 7.0, 18.0, 10.0, 0.5) / 100
    terminal = st.slider('Terminal growth (%)', 0.0, 4.0, 2.5, 0.5) / 100
    run = st.button('🔎 สแกนชุดนี้', type='primary', use_container_width=True)
    st.caption('Financials, Real Estate และ Utilities ยังใช้ Special Review จนกว่าจะมีโมเดลเฉพาะ')

if run:
    if rate <= terminal + .01:
        st.error('Discount rate ต้องสูงกว่า terminal growth มากกว่า 1 จุดเปอร์เซ็นต์')
    elif selection.empty:
        st.warning('ชุดนี้ไม่มีหุ้น กรุณาเลือกชุดอื่น')
    else:
        rows, errors = [], []
        progress = st.progress(0)
        for i, item in enumerate(selection.itertuples(index=False)):
            try:
                row, _ = evaluate(item.Symbol, growth_cap, rate, terminal, mos_min, 15, 1.0, str(item.Sector))
                rows.append(row)
            except Exception as exc:
                errors.append({'Ticker': item.Symbol, 'Sector': item.Sector, 'Reason': str(exc)})
            progress.progress((i + 1) / len(selection))
        st.session_state['v5_rows'] = rows
        st.session_state['v5_errors'] = errors
        st.session_state['v5_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        st.session_state['v5_mos'] = mos_min

scheduled = Path('data/latest_scan.csv')
if scheduled.exists():
    with st.expander('📅 ผลสแกนอัตโนมัติล่าสุด'):
        try:
            auto = pd.read_csv(scheduled)
            st.write(f'มีผลประเมิน {len(auto)} แถว — ไม่ใช่การยืนยันว่าสแกนสำเร็จครบทุกหุ้น')
            if 'Scanned UTC' in auto.columns:
                dates = pd.to_datetime(auto['Scanned UTC'], errors='coerce', utc=True)
                if dates.notna().any():
                    st.caption(f'เวลาสแกนในไฟล์: {dates.min():%Y-%m-%d %H:%M} ถึง {dates.max():%Y-%m-%d %H:%M} UTC')
            auto_passed = shortlist(auto, mos_min)
            st.metric(f'ผ่าน Quality + MOS ≥ {mos_min}%', len(auto_passed))
            st.dataframe(auto, use_container_width=True, hide_index=True)
            st.download_button('⬇️ ดาวน์โหลดผลอัตโนมัติ', auto.to_csv(index=False).encode('utf-8-sig'), 'scheduled_scan.csv')
        except Exception as exc:
            st.warning(f'อ่านผลสแกนอัตโนมัติไม่ได้: {exc}')

if 'v5_rows' not in st.session_state:
    st.info('เลือกหุ้นจากเมนูด้านซ้าย แล้วกดสแกน เริ่มจากชุดละ 10 หุ้นเพื่อลดปัญหา Yahoo Rate Limit')
else:
    df = pd.DataFrame(st.session_state['v5_rows'])
    errors = st.session_state['v5_errors']
    active_mos = st.session_state.get('v5_mos', 15)
    if not df.empty:
        passed = shortlist(df, active_mos)
        a, b, c = st.columns(3)
        a.metric('ประเมินมูลค่าได้', len(df))
        b.metric(f'ผ่าน Quality + MOS ≥ {active_mos}%', len(passed))
        c.metric('Special Review / ข้อมูลไม่ครบ', len(errors))
        st.caption('Quality Pass ยังอิงเกณฑ์ ROE และ D/E จาก Engine V4; Quality Score 100 คะแนนจะเพิ่มใน Engine V5')
        t1, t2, t3 = st.tabs(['🎯 Shortlist', '📊 วิเคราะห์รายตัว', '📋 ทั้งหมด'])
        with t1:
            if passed.empty:
                st.info('ยังไม่มีหุ้นผ่านเกณฑ์ในชุดนี้')
            for _, r in passed.iterrows():
                with st.container(border=True):
                    st.subheader(f"{r['Ticker']} · {r.get('Company', '')}")
                    st.caption(f"{r.get('Sector', 'Unknown')} | {r.get('Decision', '')} | CDC {r.get('CDC Zone', 'N/A')}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric('Price', money(r.get('Price ($)')))
                    c2.metric('Base Fair Value', money(r.get('Base FV ($)')))
                    c3.metric('Upside', percent(r.get('Upside (%)')))
                    st.write(f"**MOS {percent(r.get('MOS (%)'))}** · Technical {r.get('Technical Score', 'N/A')}/6")
        with t2:
            selected = st.selectbox('Ticker', df['Ticker'].astype(str).tolist())
            r = df.loc[df['Ticker'].astype(str) == selected].iloc[0]
            st.write(f"**{r.get('Decision', '')}** · {r.get('Sector', '')} · {r.get('Model', '')}")
            c1, c2, c3 = st.columns(3)
            c1.metric('MOS', percent(r.get('MOS (%)')))
            c2.metric('Upside', percent(r.get('Upside (%)')))
            c3.metric('Base Fair Value', money(r.get('Base FV ($)')))
            st.write(f"DCF Bear {money(r.get('Bear FV ($)'))} / Base {money(r.get('Base FV ($)'))} / Bull {money(r.get('Bull FV ($)'))}")
            st.write(f"CDC {r.get('CDC Zone', 'N/A')} | Buy today {r.get('CDC Buy today', 'N/A')} | Sell today {r.get('CDC Sell today', 'N/A')} | Proxy {r.get('Smart Money Proxy', 'N/A')}")
            try:
                d = indicators(history(selected)).tail(180)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name='Price'))
                for col in ['EMA12', 'EMA26']:
                    fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col))
                for signal, marker, color in [('CDC_buy', 'triangle-up', 'green'), ('CDC_sell', 'triangle-down', 'red')]:
                    s = d[d[signal].fillna(False)]
                    y = s['Low'] * .99 if signal == 'CDC_buy' else s['High'] * 1.01
                    fig.add_trace(go.Scatter(x=s.index, y=y, mode='markers', name=signal,
                                             marker=dict(symbol=marker, color=color, size=12)))
                fig.update_layout(height=430, xaxis_rangeslider_visible=False,
                                  margin=dict(l=5, r=5, t=10, b=5))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f'กราฟโหลดไม่ได้: {exc}')
        with t3:
            st.dataframe(df.sort_values('MOS (%)', ascending=False), use_container_width=True, hide_index=True)
            st.download_button('⬇️ CSV', df.to_csv(index=False).encode('utf-8-sig'), 'scanner_v5.csv')
    else:
        st.warning('ไม่มีหุ้นที่ประเมินมูลค่าได้ในชุดนี้')
    if errors:
        with st.expander(f'⚠️ Special Review / ข้อมูลไม่ครบ ({len(errors)})'):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)
    st.caption('สแกนเมื่อ ' + st.session_state['v5_at'])

with st.expander('วิธีคำนวณ / ข้อจำกัด'):
    st.markdown('''
- **MOS = (Fair Value − Price) / Fair Value × 100**
- **Upside = (Fair Value − Price) / Price × 100**
- MOS 15% เท่ากับ Upside ประมาณ 17.65% เมื่อใช้ Fair Value เดียวกัน
- ใช้ MOS เป็นเงื่อนไขคัดหุ้น และแสดง Upside เพื่อการตีความ ไม่ใช้เป็นตัวกรองซ้ำซ้อน
- Engine ปัจจุบันยังใช้ DCF และ Quality Pass แบบ V4; ยังไม่ได้เพิ่ม Quality Score และ Valuation Confidence
- Financials, Real Estate และ Utilities ต้องใช้โมเดลเฉพาะ จึงแยก Special Review
- ผลสแกนใน CSV ไม่ยืนยันว่า Universe ถูกสแกนครบทั้งหมด
- Yahoo Finance อาจจำกัดคำขอหรือมีข้อมูลไม่ครบ; Fair Value เป็นค่าประเมินตามสมมติฐาน ไม่ใช่ราคาเป้าหมายที่รับประกัน
- ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
''')
