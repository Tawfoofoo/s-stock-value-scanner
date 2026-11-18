# US Stock Value Scanner V6 Final

Replace `app.py`, `engine.py`, `scan_batch.py`, `line_alert.py`, and `.github/workflows/scheduled_scan.yml` in the existing repository with these files. Keep existing `universe.py` and `requirements.txt` (ensure `streamlit`, `yfinance`, `numpy`, `pandas`, `plotly`, `requests` are installed). Workflow assumes the default branch is `main`.

Criteria: Base Fair Price DCF Upside >=20%; Quality Score >=70/100; Confidence Medium/High. MOS is displayed for reference only. Special sectors are excluded from generic FCF DCF and explicitly reported. Quality and confidence are heuristic scores, not verified investment-grade ratings. This package has not been tested against live Yahoo Finance, GitHub Actions, or LINE credentials.

**Important:** The workflow resets prior scan outputs at the start of a run; only completed runs are committed. A completed run means every ticker is accounted for (including Special Review / errors), not that every ticker has a fair value. LINE deduplication uses state and LINE retry keys; exactly-once delivery across failures cannot be guaranteed. Historical state from V4 is reused for eligibility tracking. To avoid suppressing first V6 notifications, optionally archive the previous `data/line_alert_state.json` before first run.

**Financial limitations:** Generic FCF DCF is unsuitable for some companies and industries. DCF is highly sensitive to normalized FCF, discount rate, terminal growth, net debt and diluted shares. Yahoo statements may have missing/changed labels; results require human verification. `Quality Score` uses simple available-data metrics and assigns zero for missing evidence; it is not a professional fundamental rating. CDC is a timing aid only.
