from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import evaluate, history, indicators
from universe import load_sp500

st.set_page_config(page_title='US Stock Value Scanner V4', page_icon='📈', layout='wide')
st.title('📈 US Stock Value Scanner V4')
st.caption('S&P 500 → S&P Composite 1500 (phased) • MOS ≥ 20% • CDC ActionZone • educational only')

@st.cache_data(ttl=86400, show_spinner=False)
def universe():
    return load_sp500()

try:
    members, source = universe()
except Exception as exc:
    st.error(f'โหลดรายชื่อ S&P 500 ไม่สำเร็จ: {exc}')
    st.stop()

with st.sidebar:
    st.header('⚙️ ตัวกรอง V4')
    st.caption(f'Universe: {len(members)} tickers • {source}')
    sector = st.selectbox('Sector', ['All'] + sorted(members['Sector'].unique().tolist()))
    candidates = members if sector == 'All' else members[members['Sector'] == sector]
    mode = st.radio('วิธีเลือกหุ้น', ['S&P 500 (แบ่งชุด)', 'กำหนด Ticker เอง'])
    if mode == 'S&P 500 (แบ่งชุด)':
        batch = st.number_input('ลำดับชุด (เริ่มที่ 1)', min_value=1, value=1, step=1)
        size = st.slider('จำนวนหุ้นต่อชุด', 5, 30, 10, 5)
        start = (batch - 1) * size
        selection = candidates.iloc[start:start+size]
        st.caption(f'ชุดนี้: {len(selection)} หุ้น จากทั้งหมด {len(candidates)}')
    else:
        symbols = st.text_area('Tickers', 'AAPL,MSFT,ADBE')
        requested = list(dict.fromkeys(s.strip().upper().replace('.', '-') for s in symbols.replace('\n', ',').split(',') if s.strip()))[:30]
        selection = pd.DataFrame({'Symbol': requested}).merge(members, on='Symbol', how='left').fillna({'Sector':'Unknown'})
    mos_min = st.slider('MOS ขั้นต่ำ (%)', 0, 50, 20)
    growth_cap = st.slider('เพดาน FCF growth (%)', 0, 25, 10)/100
    rate = st.slider('Discount rate (%)', 7.0, 18.0, 10.0, .5)/100
    terminal = st.slider('Terminal growth (%)', 0.0, 4.0, 2.5, .5)/100
    run = st.button('🔎 สแกนชุดนี้', type='primary', use_container_width=True)
    st.caption('Financials, Real Estate และ Utilities แยก Special Review จนกว่าจะมีโมเดลเฉพาะ')

if run:
    if rate <= terminal + .01:
        st.error('Discount rate ต้องสูงกว่า terminal growth มากกว่า 1 จุดเปอร์เซ็นต์')
    elif selection.empty:
        st.warning('ชุดนี้ไม่มีหุ้น กรุณาเลือกชุดอื่น')
    else:
        rows, errors = [], []
        bar = st.progress(0)
        for i, item in enumerate(selection.itertuples(index=False)):
            try:
                row, _ = evaluate(item.Symbol, growth_cap, rate, terminal, mos_min, 15, 1.0, str(item.Sector))
                rows.append(row)
            except Exception as exc:
                errors.append({'Ticker': item.Symbol, 'Sector': item.Sector, 'Reason': str(exc)})
            bar.progress((i+1)/len(selection))
        st.session_state['v4_rows'] = rows
        st.session_state['v4_errors'] = errors
        st.session_state['v4_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

# Optionally display the latest scheduled result committed by GitHub Actions.
scheduled = Path('data/latest_scan.csv')
if scheduled.exists():
    with st.expander('📅 ผลสแกนอัตโนมัติล่าสุด'):
        try:
            auto = pd.read_csv(scheduled)
            st.write(f'บันทึกผล {len(auto)} แถว (อาจไม่ครอบคลุมทั้ง S&P 500)')
            st.dataframe(auto, use_container_width=True, hide_index=True)
            st.download_button('ดาวน์โหลดผลอัตโนมัติ', auto.to_csv(index=False).encode('utf-8-sig'), 'scheduled_scan.csv')
        except Exception as exc:
            st.warning(str(exc))

if 'v4_rows' not in st.session_state:
    st.info('เลือกชุดหุ้นจากเมนูด้านซ้าย แล้วกดสแกน เริ่มจากชุดละ 10 หุ้นเพื่อลดปัญหา Yahoo Rate Limit')
else:
    df = pd.DataFrame(st.session_state['v4_rows'])
    errors = st.session_state['v4_errors']
    if not df.empty:
        passed = df[df['Quality Pass'] & df['Value Pass']].sort_values('MOS (%)', ascending=False)
        a,b = st.columns(2)
        a.metric('ประเมินมูลค่าได้', len(df))
        b.metric('ผ่าน Quality + MOS', len(passed))
        st.caption('หุ้น Special Review และหุ้นข้อมูลไม่ครบไม่นับเป็นหุ้นไม่ผ่านเกณฑ์')
        t1,t2,t3 = st.tabs(['🎯 Shortlist','📊 วิเคราะห์รายตัว','📋 ทั้งหมด'])
        with t1:
            if passed.empty:
                st.info('ยังไม่มีหุ้นผ่านเกณฑ์ในชุดนี้')
            for _,r in passed.iterrows():
                with st.container(border=True):
                    st.subheader(f"{r['Ticker']} · {r['Company']}")
                    st.write(f"{r['Sector']} | {r['Decision']} | CDC {r['CDC Zone']} | Smart Money Proxy {r['Smart Money Proxy']}")
                    c1,c2 = st.columns(2)
                    c1.metric('ราคา',f"${r['Price ($)']:,.2f}")
                    c2.metric('Fair Value',f"${r['Base FV ($)']:,.2f}")
                    st.write(f"**MOS {r['MOS (%)']:.1f}%** · Upside {r['Upside (%)']:.1f}% · Technical {r['Technical Score']}/6")
        with t2:
            selected = st.selectbox('Ticker',df['Ticker'].tolist())
            r = df.loc[df['Ticker']==selected].iloc[0]
            st.write(f"**{r['Decision']}** · {r['Sector']} · {r['Model']}")
            c1,c2 = st.columns(2)
            c1.metric('MOS',f"{r['MOS (%)']:.1f}%")
            c2.metric('Upside',f"{r['Upside (%)']:.1f}%")
            st.write(f"DCF Bear ${r['Bear FV ($)']:,.2f} / Base ${r['Base FV ($)']:,.2f} / Bull ${r['Bull FV ($)']:,.2f}")
            st.write(f"CDC {r['CDC Zone']} | Buy today {r['CDC Buy today']} | Sell today {r['CDC Sell today']} | Proxy {r['Smart Money Proxy']}")
            try:
                d=indicators(history(selected)).tail(180)
                fig=go.Figure()
                fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='Price'))
                for col in ['EMA12','EMA26']:
                    fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col))
                for signal,marker,color in [('CDC_buy','triangle-up','green'),('CDC_sell','triangle-down','red')]:
                    s=d[d[signal]]
                    fig.add_trace(go.Scatter(x=s.index,y=s['Low']*.99 if signal=='CDC_buy' else s['High']*1.01,mode='markers',name=signal,marker=dict(symbol=marker,color=color,size=12)))
                fig.update_layout(height=430,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=10,b=5))
                st.plotly_chart(fig,use_container_width=True)
            except Exception as exc:
                st.warning(f'กราฟโหลดไม่ได้: {exc}')
        with t3:
            st.dataframe(df.sort_values('MOS (%)',ascending=False),use_container_width=True,hide_index=True)
            st.download_button('⬇️ CSV',df.to_csv(index=False).encode('utf-8-sig'),'scanner_v4.csv')
    else:
        st.warning('ไม่มีหุ้นที่ประเมินมูลค่าได้ในชุดนี้')
    if errors:
        with st.expander(f'⚠️ Special Review / ข้อมูลไม่ครบ ({len(errors)})'):
            st.dataframe(pd.DataFrame(errors),hide_index=True,use_container_width=True)
    st.caption('สแกนเมื่อ '+st.session_state['v4_at'])

with st.expander('วิธีคำนวณ / ข้อจำกัด'):
    st.markdown('''- **MOS = (Fair Value − Price) / Fair Value × 100**; MOS 20% เท่ากับ Upside 25%\n- DCF: ใช้ median FCF ย้อนหลังสูงสุด 5 ปี คาดการณ์ 5 ปีและ terminal value ตามสมมติฐานที่กำหนด\n- Sector quality V4 ระยะแรกใช้ ROE และ D/E เป็น **ตัวแทนชั่วคราว** โดยเกณฑ์ต่างกันตาม Sector; ยังไม่ได้คำนวณ ROIC หรือ Net Debt/EBITDA ตามเกณฑ์ระยะถัดไป\n- Financials, Real Estate, Utilities ถูกแยก Special Review ไม่ใช้ DCF ทั่วไป\n- CDC: สูตร 6 สีจาก Pine ที่ผู้ใช้ให้มา โหมด Daily มาตรฐาน ไม่รวม StochRSI optional; Smart Money เป็น Proxy ไม่ใช่ SmartMCDX\n- S&P 500 เป็นรายชื่อจากแหล่งข้อมูลสาธารณะซึ่งอาจคลาดเคลื่อนจากรายชื่อทางการ; S&P 400/600 ยังไม่เปิดใช้ในระยะ A\n- Yahoo Finance อาจจำกัดคำขอ ข้อมูลขาดหายหรือช้า; ไม่ใช่สัญญาณซื้อขายหรือการรับรองมูลค่าหุ้น''')
