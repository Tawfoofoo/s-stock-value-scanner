"""One resumable batch; workflow resets state before each full run."""
import json, os, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from universe import load_sp500
from engine import evaluate

OUT=Path('data'); OUT.mkdir(exist_ok=True)
CSV=OUT/'latest_scan.csv'; STATE=OUT/'scan_state.txt'; ERRORS=OUT/'scan_errors.csv'
SIZE=max(1,min(100,int(os.getenv('BATCH_SIZE','25'))))
DELAY=max(0,float(os.getenv('SCAN_DELAY','0.5')))
members,source=load_sp500(); total=len(members)
start=int(STATE.read_text().strip()) if STATE.exists() else 0
if not 0<=start<total: start=0
batch=members.iloc[start:start+SIZE]
rows=[]; errors=[]
for item in batch.itertuples(index=False):
    ticker=str(item.Symbol)
    for attempt in range(3):
        try:
            row,_=evaluate(ticker,.10,.10,.025,20,70,1.0,str(item.Sector))
            row['Scanned UTC']=datetime.now(timezone.utc).isoformat()
            rows.append(row)
            print('OK',ticker,flush=True)
            break
        except Exception as exc:
            message=str(exc)
            if message.startswith('Special Review:') or attempt==2:
                errors.append({'Ticker':ticker,'Sector':str(item.Sector),'Status':'Special Review' if message.startswith('Special Review:') else 'Error','Reason':message,'Scanned UTC':datetime.now(timezone.utc).isoformat()})
                print('SKIP',ticker,message,flush=True)
                break
            time.sleep(2*(attempt+1))
    time.sleep(DELAY)

def append_unique(path,records):
    if not records: return
    frame=pd.DataFrame(records)
    if path.exists(): frame=pd.concat([pd.read_csv(path),frame],ignore_index=True)
    frame.drop_duplicates('Ticker',keep='last').to_csv(path,index=False)
append_unique(CSV,rows)
append_unique(ERRORS,errors)
next_index=start+len(batch)
complete=next_index>=total
STATE.write_text(str(0 if complete else next_index),encoding='utf-8')
print(f'Processed {len(batch)}, success {len(rows)}, special/errors {len(errors)}, complete={complete}, source={source}',flush=True)
if complete:
    successful=pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    failed=pd.read_csv(ERRORS) if ERRORS.exists() else pd.DataFrame()
    observed=set(successful.get('Ticker',pd.Series(dtype=str)).astype(str))|set(failed.get('Ticker',pd.Series(dtype=str)).astype(str))
    universe=set(members.Symbol.astype(str))
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'universe':len(universe),'evaluated':len(successful),'special_review':int((failed.get('Status',pd.Series(dtype=str))=='Special Review').sum()),'errors':int((failed.get('Status',pd.Series(dtype=str))=='Error').sum()),'processed':len(observed),'missing':sorted(universe-observed),'source':str(source),'complete':observed==universe}
    (OUT/'scan_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['complete']: raise RuntimeError('Scan coverage incomplete: '+str(report['missing'][:10]))
