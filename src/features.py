import pandas as pd

def add_rolling_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add rolling mean, volatility, and slope features.
    """
    df['ma'] = df['Close'].rolling(window).mean()
    df['vol'] = df['return'].rolling(window).std()
    df['slope'] = df['ma'].diff()
    return df