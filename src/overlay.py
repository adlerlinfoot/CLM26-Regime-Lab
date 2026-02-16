import pandas as pd

regime_to_position = {
    0: 0.0,
    1: 1.0,
    2: -1.0,
    3: 0.5
}

def build_overlay(df, preds, test_index):
    df.loc[test_index, "regime_pred"] = preds
    df["position"] = df["regime_pred"].map(regime_to_position)

    # Strategy returns
    df["strategy_ret"] = df["position"].shift(1) * df["ret_1d"]

    # Equity curves
    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["underlying_equity"] = (1 + df["ret_1d"]).cumprod()

    # Days in regime
    df["regime_run"] = (df["regime_pred"] != df["regime_pred"].shift()).cumsum()
    df["days_in_regime"] = df.groupby("regime_run").cumcount() + 1

    return df