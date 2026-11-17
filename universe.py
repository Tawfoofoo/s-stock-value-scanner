"""S&P 500 constituent loader; data sources are best-effort, not official S&P feeds."""
from io import StringIO
import pandas as pd
import requests

DATASET = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def load_sp500():
    errors = []
    for name, url in (("GitHub community dataset", DATASET), ("Wikipedia", WIKI)):
        try:
            response = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (research scanner; educational)"}, timeout=20)
            response.raise_for_status()
            if name == "Wikipedia":
                frame = pd.read_html(StringIO(response.text))[0]
                frame = frame.rename(columns={"Symbol":"Symbol", "GICS Sector":"Sector", "Security":"Security"})
            else:
                frame = pd.read_csv(StringIO(response.text))
                frame = frame.rename(columns={"GICS Sector":"Sector", "GICS_Sector":"Sector"})
            if not {"Symbol", "Sector"}.issubset(frame.columns):
                raise ValueError("Missing Symbol or Sector columns")
            frame = frame[["Symbol", "Sector"]].dropna().drop_duplicates("Symbol")
            frame["Symbol"] = frame["Symbol"].astype(str).str.replace(".", "-", regex=False)
            frame["Sector"] = frame["Sector"].astype(str)
            if len(frame) < 450:
                raise ValueError("Unexpectedly short S&P 500 list")
            return frame.sort_values("Symbol").reset_index(drop=True), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Cannot load current S&P 500 constituents: " + " | ".join(errors))
