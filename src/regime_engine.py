def classify_trend(row):
    """
    Classify trend regime based on price relative to moving average.
    """
    if row['Close'] > row['ma']:
        return "uptrend"
    elif row['Close'] < row['ma']:
        return "downtrend"
    return "sideways"

def classify_vol(row, low=0.01, high=0.03):
    """
    Classify volatility regime based on rolling volatility.
    """
    if row['vol'] < low:
        return "low"
    elif row['vol'] > high:
        return "high"
    return "medium"

def classify_timing(date):
    """
    Classify timing regime based on day of week.
    Wednesday = pre-weekend (game-specific logic)
    """
    weekday = date.weekday()
    if weekday == 2:  # Wednesday
        return "pre-weekend"
    return "normal"