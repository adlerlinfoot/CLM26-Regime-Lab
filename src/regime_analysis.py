import pandas as pd

def compute_transition_matrix(y):
    return pd.crosstab(y.shift(), y, normalize="index")

def compute_expected_durations(y):
    segments = (y != y.shift()).cumsum()
    durations = y.groupby(segments).size()
    regime_of_segment = y.groupby(segments).first()
    return durations.groupby(regime_of_segment).mean()