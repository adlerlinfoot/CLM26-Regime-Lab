import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data/yahoo_raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = {
    "CL1": "CL=F",
    "CL2": "CL=F?offset=1",
    "DXY": "DX-Y.NYB"
}

def download_symbol(name, ticker):
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start="1990-01-01", progress=False)
    df.to_csv(DATA_DIR / f"{name}.csv")
    print(f"Saved → {DATA_DIR / f'{name}.csv'}")

def download_raw_data():
    for name, ticker in SYMBOLS.items():
        download_symbol(name, ticker)
