import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def add_returns(df):
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_21d"] = df["close"].pct_change(21)
    return df

def add_volatility(df):
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    df["vol_63d"] = df["ret_1d"].rolling(63).std()
    return df

def add_momentum(df):
    df["mom_20d"] = df["close"].pct_change(20)
    df["mom_63d"] = df["close"].pct_change(63)
    return df

def add_sma(df):
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["price_vs_sma"] = df["close"] / df["sma_20"] - 1
    return df

def rolling_slope(series, window):
    """
    Compute rolling slope using linear regression over a moving window.
    """
    slopes = []
    X = np.arange(window).reshape(-1, 1)
    lr = LinearRegression()

    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window:i].values.reshape(-1, 1)
            lr.fit(X, y)
            slopes.append(lr.coef_[0][0])
    return pd.Series(slopes, index=series.index)

def add_slopes(df):
    df["slope_20"] = rolling_slope(df["close"], 20)
    df["slope_50"] = rolling_slope(df["close"], 50)
    return df

def add_vol_of_vol(df):
    df["vol_of_vol"] = df["vol_21d"].rolling(21).std()
    return df

def add_features(df):
    df = add_returns(df)
    df = add_volatility(df)
    df = add_momentum(df)
    df = add_sma(df)
    df = add_slopes(df)
    df = add_vol_of_vol(df)
    return df