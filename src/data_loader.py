import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/yahoo_raw")

def load_series(name):
    path = DATA_DIR / f"{name}.csv"

    # Skip the first 3 junk rows (Price, Ticker, Date)
    df = pd.read_csv(path, header=None, skiprows=3)

    # Assign correct column names
    df.columns = ["date", "close", "high", "low", "open", "volume"]

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Sort and return
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_all():
    cl2 = load_series("CL2")
    dxy = load_series("DXY")
    return cl2, dxy

