"""
run_pipeline.py — REPL-Exact Version (Full GridSearchCV)

This script reproduces the REPL model exactly:

1. Load final_dataset.csv
2. Build REPL-style features
3. Label regimes (vol_state × trend_state)
4. Build REPL feature matrix (shifted by 1 day)
5. Train/test split (shuffle=False, test_size=0.2)
6. Fit baseline RandomForest
7. Run full GridSearchCV with TimeSeriesSplit (REPL grid)
8. Evaluate tuned model
9. Build overlay using predicted regimes
10. Compute transition matrix & expected durations
11. Save outputs
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from overlay import build_overlay
from regime_analysis import compute_transition_matrix, compute_expected_durations


# ---------------------------------------------------------
#  PATHS — FIXED SO DATA ALWAYS SAVES TO PROJECT ROOT
# ---------------------------------------------------------

# This file lives in <project_root>/src or <project_root> depending on your structure.
# We anchor everything to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # go up from /src to project root
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
#  REPL Feature Engineering
# ---------------------------------------------------------

def build_repl_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_20d"] = df["close"].pct_change().rolling(20).std()

    df["mom_20d"] = df["close"] / df["close"].shift(20) - 1
    df["mom_63d"] = df["close"] / df["close"].shift(63) - 1

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    df["price_vs_sma"] = df["close"] / df["sma_20"] - 1

    df["slope_20"] = df["sma_20"].diff(5)
    df["slope_50"] = df["sma_50"].diff(5)

    df["vol_of_vol"] = df["vol_20d"].rolling(10).std()

    return df


# ---------------------------------------------------------
#  REPL Regime Labeling
# ---------------------------------------------------------

def label_repl_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_state"] = df["vol_20d"] > df["vol_20d"].median()
    df["trend_state"] = df["sma_20"] > df["sma_50"]

    df["regime"] = (
        df["vol_state"].astype(int) * 2 +
        df["trend_state"].astype(int)
    )

    return df


# ---------------------------------------------------------
#  REPL Model Training (with full GridSearchCV)
# ---------------------------------------------------------

def train_repl_model_with_grid(df: pd.DataFrame):
    features = [
        "ret_1d", "ret_5d", "ret_21d",
        "vol_20d", "vol_21d", "vol_63d",
        "mom_20d", "mom_63d",
        "sma_20", "sma_50",
        "price_vs_sma",
        "slope_20", "slope_50",
        "vol_of_vol",
        "dxy",
    ]

    X = df[features].shift(1).dropna()
    y = df["regime"].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42
    )
    baseline.fit(X_train, y_train)

    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 7, 9],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2"]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=tscv,
        scoring="f1_weighted",
        n_jobs=-1
    )

    print("\nRunning full GridSearchCV (this may take several minutes)...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    return best_model, baseline, X_train, X_test, y_train, y_test, preds, grid


# ---------------------------------------------------------
#  Main Pipeline
# ---------------------------------------------------------

def main():
    print("\n=== STEP 1: Load final_dataset.csv ===")
    df = pd.read_csv(DATA_DIR / "final_dataset.csv")
    print("Dataset loaded:", df.shape)

    print("\n=== STEP 2: Build REPL-style features ===")
    df = build_repl_features(df)

    print("\n=== STEP 3: Label regimes (REPL logic) ===")
    df = label_repl_regimes(df)
    print(df["regime"].value_counts().sort_index())

    print("\n=== STEP 4: Train REPL model with full GridSearchCV ===")
    best_model, baseline, X_train, X_test, y_train, y_test, preds, grid = train_repl_model_with_grid(df)
    print("GridSearchCV complete.")
    print("\nBest Params:", grid.best_params_)

    print("\n=== STEP 5: Evaluation ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\n=== STEP 6: Regime transition analysis (TRUE regimes) ===")
    y_full = df["regime"].dropna()
    transition_matrix = compute_transition_matrix(y_full)
    expected_durations = compute_expected_durations(y_full)

    print("\nTransition Matrix:")
    print(transition_matrix)

    print("\nExpected Durations:")
    print(expected_durations)

    print("\n=== STEP 7: Build trading overlay ===")
    df = build_overlay(df, preds, X_test.index)

    print("\n=== STEP 8: Save all outputs ===")

    df.to_csv(DATA_DIR / "model_output.csv", index=False)
    df.to_csv(DATA_DIR / "overlay_output.csv", index=False)
    transition_matrix.to_csv(DATA_DIR / "transition_matrix.csv")
    expected_durations.to_csv(DATA_DIR / "expected_durations.csv")

    print("\nAll outputs saved to /data/")
    print("Pipeline complete.\n")


if __name__ == "__main__":
    main()