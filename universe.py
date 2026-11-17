
"""S&P 500 constituent loader with validation."""

from io import StringIO
import pandas as pd
import requests

DATASET = (
    "https://raw.githubusercontent.com/"
    "datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)

WIKI = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)


def load_sp500():
    errors = []

    sources = [
        ("GitHub community dataset", DATASET),
        ("Wikipedia", WIKI),
    ]

    for name, url in sources:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
                timeout=25,
            )
            response.raise_for_status()

            if name == "Wikipedia":
                frame = pd.read_html(
                    StringIO(response.text)
                )[0]
            else:
                frame = pd.read_csv(
                    StringIO(response.text)
                )

            frame = frame.rename(columns={
                "GICS Sector": "Sector",
                "GICS_Sector": "Sector",
            })

            required = {"Symbol", "Sector"}
            if not required.issubset(frame.columns):
                raise ValueError(
                    "Missing required columns"
                )

            frame = frame[
                ["Symbol", "Sector"]
            ].dropna().copy()

            frame["Symbol"] = (
                frame["Symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(
                    ".", "-", regex=False
                )
            )

            frame["Sector"] = (
                frame["Sector"]
                .astype(str)
                .str.strip()
            )

            frame = frame.drop_duplicates(
                subset="Symbol",
                keep="first"
            )

            if not 450 <= len(frame) <= 550:
                raise ValueError(
                    f"Unexpected constituent count: "
                    f"{len(frame)}"
                )

            frame = (
                frame.sort_values("Symbol")
                .reset_index(drop=True)
            )

            print(
                f"Loaded {len(frame)} symbols "
                f"from {name}",
                flush=True
            )

            return frame, name

        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError(
        "Cannot load S&P 500: "
        + " | ".join(errors)
    )
