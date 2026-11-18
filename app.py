from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import evaluate,history,indicators
from universe import load_sp500

st.set_page_config(page_title='US Stock Value Scanner V6 Final',page_icon='📈',layout='wide')
st.title('📈 US Stock Value Scanner V6 Final')
st.caption('S&P 500 • Upside ≥ 20% • Quality ≥ 70/100 • Fair Price • CDC • เพื่อการศึกษาเท่านั้น')

@st.cache_data(ttl=86400)
def universe(): return load_sp500()

def numeric(df,col): return pd.to_numeric(df[col],errors='coerce') if col in df else pd.Series(float('nan'),index=df.index)
def shortlist(df,up,quality):
    if df.empty: return df.copy()
    mask=(numeric(df,'Upside (%)')>=up)&(numeric(df,'Quality Score')>=quality)&df.get('Valuation Confidence',pd.Series('',index=df.index)).isin(['Medium','High'])
    return df.loc[mask].sort_values('Upside (%)',ascending=False).copy()
def money(v):
    n=pd.to_numeric(v,errors='coerce');return f'${n:,.2f}' if pd.notna(n) else 'N/A'
def pct(v):
    n=pd.to_numeric(v,errors='coerce');return f'{n:,.1f}%' if pd.notna(n) else 'N/A'

try: members,source=universe()
except Exception as exc: st.error(f'โหลด Universe ไม่ได้: {exc}');st.stop()
with st.sidebar:
    st.header('⚙️ V6 Final')
    st.caption(f'{len(members)} tickers • {source}')
    sectors=['All']+sorted(members.Sector.dropna().astype(str).unique())
    sector=st.selectbox('Sector',sectors)
    candidates=members if sector=='All' else members[members.Sector==sector]
    mode=st.radio('เลือกหุ้น',['S&P 500 (แบ่งชุด)','Ticker เอง'])
    if mode=='S&P 500 (แบ่งชุด)':
        batch=st.number_input('ชุดที่',min_value=1,value=1)
        size=st.slider('หุ้นต่อชุด',5,30,10,5)
        selection=candidates.iloc[(batch-1)*size:batch*size]
    else:
        symbols=st.text_area('Tickers','AAPL,MSFT,ADBE')
        requested=list(dict.fromkeys(x.strip().upper().replace('.','-') for x in symbols.replace('\n',',').split(',') if x.strip()))[:30]
        selection=pd.DataFrame({'Symbol':requested}).merge(members,on='Symbol',how='left')
        selection['Sector']=selection['Sector'].fillna('Unknown')
    min_up=st.slider('Upside ขั้นต่ำ (%)',0,60,20,5)
    min_quality=st.slider('Quality Score ขั้นต่ำ',0,100,70,5)
    growth=st.slider('FCF growth cap (%)',0,25,10)/100
    rate=st.slider('Discount rate (%)',7.0,18.0,10.0,.5)/100
    terminal=st.slider('Terminal growth (%)',0.0,4.0,2.5,.5)/100
    run=st.button('🔎 สแกนชุดนี้',type='primary',use_container_width=True)
if run:
    rows=[];errors=[]
    if rate<=terminal+.01: st.error('Discount rate ต้องสูงกว่า Terminal growth มากกว่า 1%')
    else:
        progress=st.progress(0)
        for i,item in enumerate(selection.itertuples(index=False)):
            try:
                row,_=evaluate(item.Symbol,growth,rate,terminal,min_up,min_quality,1,str(item.Sector));rows.append(row)
            except Exception as exc: errors.append({'Ticker':item.Symbol,'Reason':str(exc)})
            progress.progress((i+1)/max(1,len(selection)))
        st.session_state['rows']=rows;st.session_state['errors']=errors
        st.session_state['scanned_at']=pd.Timestamp.now(tz='UTC').isoformat()

scheduled=Path('data/latest_scan.csv')
if scheduled.exists():
    with st.expander('📅 ผลสแกนอัตโนมัติล่าสุด',expanded=True):
        try:
            auto=pd.read_csv(scheduled)
            report_path=Path('data/scan_report.json')
            if report_path.exists():
                report=json.loads(report_path.read_text())
                st.caption(f"Coverage: {report['processed']}/{report['universe']} | Evaluated {report['evaluated']} | Special {report['special_review']} | Errors {report['errors']} | Complete {report['complete']}")
            else: st.warning('ไม่พบรายงานยืนยันความครบถ้วน')
            passed=shortlist(auto,min_up,min_quality)
            a,b,c=st.columns(3)
            a.metric('ประเมินได้',len(auto));b.metric('ผ่านเกณฑ์',len(passed));c.metric('Upside ขั้นต่ำ',f'{min_up}%')
            st.dataframe(passed,use_container_width=True,hide_index=True)
            st.download_button('⬇️ ดาวน์โหลดผลทั้งหมด',auto.to_csv(index=False).encode('utf-8-sig'),'v6_scheduled.csv')
        except Exception as exc: st.warning(str(exc))
if 'rows' not in st.session_state:
    st.info('เลือกหุ้นใน Sidebar แล้วกดสแกน เพื่อวิเคราะห์รายตัว')
else:
    df=pd.DataFrame(st.session_state['rows']);errors=st.session_state['errors']
    if not df.empty:
        passed=shortlist(df,min_up,min_quality)
        t1,t2,t3=st.tabs(['🎯 Shortlist','📊 วิเคราะห์รายตัว','📋 ทั้งหมด'])
        with t1:
            st.dataframe(passed,use_container_width=True,hide_index=True)
        with t2:
            symbol=st.selectbox('Ticker',df.Ticker.tolist())
            row=df.loc[df.Ticker==symbol].iloc[0]
            c1,c2,c3,c4=st.columns(4)
            c1.metric('Current Price',money(row.get('Price ($)')))
            c2.metric('Fair Price',money(row.get('Fair Price ($)')))
            c3.metric('Upside',pct(row.get('Upside (%)')))
            c4.metric('Quality',f"{row.get('Quality Score')}/100")
            st.write(f"Bear {money(row.get('Bear FV ($)'))} • Base {money(row.get('Base FV ($)'))} • Bull {money(row.get('Bull FV ($)'))}")
            st.write(f"Confidence **{row.get('Valuation Confidence')}** • CDC **{row.get('CDC Zone')}** • Technical {row.get('Technical Score')}/6")
            try:
                d=indicators(history(symbol)).tail(180)
                fig=go.Figure([go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='Price')])
                for col in ['EMA12','EMA26']: fig.add_trace(go.Scatter(x=d.index,y=d[col],name=col))
                for col,shape,color,anchor in [('CDC_buy','triangle-up','green','Low'),('CDC_sell','triangle-down','red','High')]:
                    x=d[d[col].fillna(False)]
                    fig.add_trace(go.Scatter(x=x.index,y=x[anchor],mode='markers',name=col,marker=dict(symbol=shape,color=color,size=11)))
                fig.update_layout(height=430,xaxis_rangeslider_visible=False)
                st.plotly_chart(fig,use_container_width=True)
            except Exception as exc: st.warning(f'กราฟโหลดไม่ได้: {exc}')
        with t3:
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button('⬇️ CSV',df.to_csv(index=False).encode('utf-8-sig'),'v6_scan.csv')
    if errors:
        with st.expander(f'⚠️ Special Review / Errors ({len(errors)})'):
            st.dataframe(pd.DataFrame(errors),hide_index=True)
st.caption('Fair Price เป็นการประเมินจากสมมติฐาน FCF DCF ไม่ใช่ราคาเป้าหมายที่รับประกัน; Confidence เป็นความครบถ้วนของข้อมูล ไม่ใช่ความแม่นยำเชิงสถิติ')
