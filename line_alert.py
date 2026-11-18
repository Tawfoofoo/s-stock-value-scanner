"""V6 LINE daily report + new eligible and CDC signals."""
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

CSV=Path(os.getenv('SCAN_CSV_PATH','data/latest_scan.csv'))
STATE=Path(os.getenv('LINE_ALERT_STATE_PATH','data/line_alert_state.json'))
REPORT=Path('data/scan_report.json')
API='https://api.line.me/v2/bot/message/push'
MIN_UP=float(os.getenv('LINE_MIN_UPSIDE','20'))
MIN_Q=float(os.getenv('LINE_MIN_QUALITY','70'))

def fmt(v):
    n=pd.to_numeric(v,errors='coerce')
    return f'{n:,.2f}' if pd.notna(n) else 'N/A'

def is_true(v): return str(v).strip().lower() in ('true','1','yes')

def line(row):
    return (f"{row['Ticker']} | Price ${fmt(row.get('Price ($)'))} | Fair ${fmt(row.get('Fair Price ($)'))} | "
            f"Upside {fmt(row.get('Upside (%)'))}% | Quality {fmt(row.get('Quality Score'))}/100 | "
            f"Confidence {row.get('Valuation Confidence','N/A')} | CDC {row.get('CDC Zone','N/A')}\n"
            f"Bear/Base/Bull: ${fmt(row.get('Bear FV ($)'))} / ${fmt(row.get('Base FV ($)'))} / ${fmt(row.get('Bull FV ($)'))}")

def chunks(lines,limit=4500):
    result=[]; current=''
    for line_ in lines:
        if len(line_)>limit: raise ValueError('Oversized LINE line')
        candidate=current+'\n'+line_ if current else line_
        if len(candidate)>limit: result.append(current);current=line_
        else: current=candidate
    if current: result.append(current)
    return result

def main():
    token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN','').strip()
    uid=os.getenv('LINE_TO_USER_ID','').strip()
    if not token or not uid: raise RuntimeError('LINE secrets missing')
    if not REPORT.exists(): raise RuntimeError('Missing scan_report.json')
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    if not report.get('complete'): raise RuntimeError('Scan not complete')
    if not CSV.exists(): raise RuntimeError('No evaluated stocks')
    df=pd.read_csv(CSV)
    required={'Ticker','Upside (%)','Quality Score','Fair Price ($)','CDC Buy today','Scanned UTC','Valuation Confidence'}
    if required-set(df.columns): raise RuntimeError(f'Missing columns: {required-set(df.columns)}')
    df['Ticker']=df.Ticker.astype(str).str.strip().str.upper()
    df=df[df.Ticker.str.fullmatch(r'[A-Z0-9.\-]+',na=False)].drop_duplicates('Ticker',keep='last')
    if len(df)!=report['evaluated']: raise RuntimeError('CSV/report mismatch')
    scan_times=pd.to_datetime(df['Scanned UTC'],utc=True,errors='coerce')
    if scan_times.isna().any() or ((pd.Timestamp.now(tz='UTC')-scan_times).max()>pd.Timedelta(hours=6)):
        raise RuntimeError('Stale scan rows')
    eligible=(pd.to_numeric(df['Upside (%)'],errors='coerce')>=MIN_UP)&(pd.to_numeric(df['Quality Score'],errors='coerce')>=MIN_Q)&df['Valuation Confidence'].isin(['Medium','High'])
    good=df.loc[eligible].sort_values('Upside (%)',ascending=False)
    current=set(good.Ticker)
    buys=set(good.loc[good['CDC Buy today'].map(is_true),'Ticker'])
    state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    prev=set(state.get('value_alerted',[])); prev_buy=set(state.get('buy_alerted',[]))
    new=current-prev; new_buy=buys-prev_buy
    stamp=report['timestamp_utc']
    lines=['📈 US Stock Value Scanner V6 Final',f'📅 Daily Scan Report | {stamp}',f"Universe {report['universe']} | Evaluated {report['evaluated']} | Special {report['special_review']} | Errors {report['errors']}",f'Upside ≥ {MIN_UP:g}% + Quality ≥ {MIN_Q:g}/100',f'Qualified {len(good)} | New Value {len(new)} | New CDC Buy {len(new_buy)}','','💎 QUALIFIED STOCKS']
    lines += [line(r) for _,r in good.iterrows()] or ['None']
    lines += ['','🆕 NEW VALUE ALERT']+[line(r) for _,r in good[good.Ticker.isin(new)].iterrows()] if new else ['','🆕 NEW VALUE ALERT','None']
    lines += ['','🟢 NEW CDC BUY ALERT']+[line(r) for _,r in good[good.Ticker.isin(new_buy)].iterrows()] if new_buy else ['','🟢 NEW CDC BUY ALERT','None']
    lines += ['','Educational only; not investment advice.']
    # Retry keys are stable per report chunk, enabling LINE API deduplication within its retry-key window.
    for i,message in enumerate(chunks(lines)):
        retry_key=hashlib.sha256(f'{stamp}:{i}:{message}'.encode()).hexdigest()[:32]
        resp=requests.post(API,headers={'Authorization':f'Bearer {token}','Content-Type':'application/json','X-Line-Retry-Key':retry_key},json={'to':uid,'messages':[{'type':'text','text':message}]},timeout=25)
        if resp.status_code not in (200, 409):
    print("LINE API status:", resp.status_code)
    print("LINE API response:", resp.text)
    resp.raise_for_status()
        print('Sent chunk',i+1,flush=True)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps({'value_alerted':sorted(current),'buy_alerted':sorted(buys),'last_report':stamp},indent=2),encoding='utf-8')
    tmp.replace(STATE)

if __name__=='__main__': main()
