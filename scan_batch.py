"""Scheduled small-batch scan. GitHub Actions rotates through S&P 500 members."""
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from universe import load_sp500
from engine import evaluate

out = Path('data'); out.mkdir(exist_ok=True)
constituents, source = load_sp500()
size = max(1, min(25, int(os.getenv('BATCH_SIZE', '10'))))
state = out / 'scan_state.txt'
start = int(state.read_text().strip()) if state.exists() else 0
if start >= len(constituents): start = 0
batch = constituents.iloc[start:start+size]
rows = []
for item in batch.itertuples(index=False):
    try:
        row, _ = evaluate(item.Symbol, .10, .10, .025, 20, 15, 1, str(item.Sector))
        row['Scanned UTC'] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        print('OK', item.Symbol)
    except Exception as exc:
        print('SKIP', item.Symbol, str(exc))
if rows:
    path = out / 'latest_scan.csv'
    fresh = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        fresh = pd.concat([old, fresh], ignore_index=True).drop_duplicates('Ticker',keep='last')
    fresh.to_csv(path,index=False)
state.write_text(str((start+len(batch))%len(constituents)))
print(f'Processed {len(batch)}; saved {len(rows)}; next index {state.read_text()}; source {source}')
