# US Stock Value Scanner V4 — Phase A

Streamlit Community Cloud app, S&P 500 constituent list, batch scan, MOS >=20%, CDC ActionZone, and a scheduled GitHub Actions batch job.

## Files
- `streamlit_app.py` — Streamlit dashboard (set as main file on Community Cloud)
- `engine.py` — DCF + CDC + proxy + preliminary sector checks
- `universe.py` — S&P 500 constituents (community dataset, Wikipedia fallback)
- `scan_batch.py` — scheduled batch job
- `requirements.txt` — dependencies
- `.github/workflows/daily-scan.yml` — scheduled GitHub Actions job

## Important
1. **Phase A covers S&P 500 only.** S&P 400 and 600 will be added in later phases after sourcing a validated constituent list.
2. GitHub Actions runs a **10-symbol batch** per scheduled run, not all 500 at once. It rotates over the universe; approximately 50 scheduled trading-day runs are needed to cover 500 symbols, assuming no data failures. CSV is incremental, and entries may be stale; inspect `Scanned UTC`.
3. Sectors Financials, Real Estate, Utilities are **Special Review**; no sector-specific valuation is implemented yet. Sector thresholds for other groups are **ROE/D-E proxies**, not true ROIC or net-debt/EBITDA.
4. Scheduled results are committed to the repo; this requires GitHub Actions permission to write repository contents. Enable Settings > Actions > General > Workflow permissions > Read and write permissions, if allowed. GitHub may pause scheduled workflows on inactive public repos; rate limits may occur.
5. For an existing app using `app.py`, either change the Streamlit main file to `streamlit_app.py` in the app settings, or rename `streamlit_app.py` to `app.py` in GitHub.
6. The app's manual scan is per-batch and results stay in the current browser session; the scheduled CSV persists in GitHub. The app does **not** automatically scan all 500 on each visit.
7. Data from yfinance and third-party constituent lists are unofficial, may be delayed, and should be independently verified. This is research software, not financial advice.
