CLM26-Regime-Classification-Lab

---

Reproducible pipeline for classifying volatility-trend regimes in crude oil futures (CL2) and evaluating in a simple regime-aware trading overlay.

Of note, this project is part of an ongoing research series exploring regime-based modeling and market structure.

OVERVIEW
This project builds a 4-state market regime model using CL2 continuous futures and DXY.  
Regimes are defined by volatility (high/low) and trend (up/down), producing interpretable states:

- **0:** Low-vol, downtrend  
- **1:** Low-vol, uptrend  
- **2:** High-vol, downtrend  
- **3:** High-vol, uptrend  

The model uses rolling returns, volatility measures, momentum, moving averages, slope features, vol-of-vol, and DXY. All features are shifted by one day to prevent lookahead bias.

A RandomForest classifier is tuned via full GridSearchCV with TimeSeriesSplit, achieving **>80% weighted accuracy** on a strict out-of-sample test set — exceeding typical academic benchmarks for 4-class financial regime models.

---

METHOD SUMMARY
- **Feature engineering:** rolling returns, vol windows, momentum, SMA spreads, slopes, vol-of-vol, macro overlay.  
- **Regime labeling:** volatility state × trend state (binary × binary).  
- **Modeling:** RandomForest with full hyperparameter grid search (time-series aware).  
- **Evaluation:** confusion matrix, classification report, transition matrix, expected durations.  
- **Overlay:** simple long/short mapping based on predicted regimes, with cumulative equity and days-in-regime tracking.

---

REPRODUCIBILITY
The entire workflow is deterministic and leakage-free.  
Run the full pipeline with:
python build_final_dataset.py 
python run_pipeline.py


Outputs (model predictions, transition matrix, expected durations) are saved to `data/`.

---

LICENSE  
MIT License.
