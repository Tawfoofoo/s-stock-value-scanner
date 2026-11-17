
"""Resumable S&P 500 batch scanner."""

import os
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from universe import load_sp500
from engine import evaluate

OUT = Path("data")
OUT.mkdir(exist_ok=True)

CSV = OUT / "latest_scan.csv"
STATE = OUT / "scan_state.txt"
ERRORS = OUT / "scan_errors.csv"

BATCH_SIZE = max(
    1, min(100, int(os.getenv("BATCH_SIZE", "25")))
)
DELAY = max(
    0, float(os.getenv("SCAN_DELAY", "0.5"))
)

constituents, source = load_sp500()
total = len(constituents)

start = 0
if STATE.exists():
    try:
        start = int(STATE.read_text().strip())
    except ValueError:
        start = 0

if start < 0 or start >= total:
    start = 0

batch = constituents.iloc[start:start + BATCH_SIZE]

rows = []
errors = []

print(
    f"Scanning {start + 1} to "
    f"{start + len(batch)} of {total}",
    flush=True
)

for item in batch.itertuples(index=False):
    ticker = str(item.Symbol)

    try:
        row, _ = evaluate(
            ticker,
            .10, .10, .025,
            15, 15, 1,
            str(item.Sector)
        )

        row["Scanned UTC"] = (
            datetime.now(timezone.utc).isoformat()
        )

        rows.append(row)
        print("OK", ticker, flush=True)

    except Exception as exc:
        errors.append({
            "Ticker": ticker,
            "Reason": str(exc),
            "Scanned UTC": (
                datetime.now(timezone.utc).isoformat()
            )
        })

        print(
            "SKIP", ticker, str(exc),
            flush=True
        )

    time.sleep(DELAY)

if rows:
    fresh = pd.DataFrame(rows)

    if CSV.exists():
        old = pd.read_csv(CSV)
        fresh = pd.concat(
            [old, fresh],
            ignore_index=True
        )

    fresh = fresh.drop_duplicates(
        "Ticker", keep="last"
    )

    fresh.to_csv(CSV, index=False)

if errors:
    fresh_errors = pd.DataFrame(errors)

    if ERRORS.exists():
        old_errors = pd.read_csv(ERRORS)
        fresh_errors = pd.concat(
            [old_errors, fresh_errors],
            ignore_index=True
        )

    fresh_errors = fresh_errors.drop_duplicates(
        "Ticker", keep="last"
    )

    fresh_errors.to_csv(ERRORS, index=False)

next_index = start + len(batch)

if next_index >= total:
    next_index = 0
    print("Completed one full S&P 500 cycle")

STATE.write_text(
    str(next_index),
    encoding="utf-8"
)

print(
    f"Processed {len(batch)}; "
    f"saved {len(rows)}; "
    f"errors {len(errors)}; "
    f"next index {next_index}; "
    f"source {source}",
    flush=True
)
