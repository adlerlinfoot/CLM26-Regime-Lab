import pandas as pd

def load_cl_data(path: str) -> pd.DataFrame:
    """
    Load CLM26 historical data from CSV.

    Expected columns:
    Date, Open, High, Low, Close, Volume, Open Interest
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['return'] = df['Close'].pct_change()
    return df