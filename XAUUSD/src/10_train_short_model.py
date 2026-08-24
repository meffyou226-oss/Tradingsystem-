"""
Short-Modell trainiert.

Target für SHORT:
- Entry = close[i]
- TP = entry - tp_points * point (SHORT Gewinn)
- SL = entry + sl_points * point (SHORT Verlust)

Trainiert XGBoost für SHORT-Signale.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR, POINT)
from data_preparation import load_combined


def compute_short_target(close_arr, high_arr, low_arr, horizon, tp_points, sl_points, point=POINT):
    """Berechnet SHORT-Target: 1 wenn TP vor SL erreicht (bei SHORT-Trade)."""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    entry = close_arr
    tp_level = entry - tp_points * point  # SHORT TP (unter Entry)
    sl_level = entry + sl_points * point  # SL (ber Entry)

    horizon_actual = min(horizon, n - 1)

    for offset in range(1, horizon_actual + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)

        undetermined = targets[current_idx] == 0

        # TP hit: future low <= tp_level
        tp_hits = low_arr[future_idx] <= tp_level[current_idx]
        # SL hit: future high >= sl_level
        sl_hits = high_arr[future_idx] >= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        targets[current_idx[new_tp]] = 1

        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def main():
    print("Trainiere SHORT-Modell...")

    df = load_combined()

    # Features sind bereits berechnet (Lookahead-frei)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # SHORT-Target berechnen
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    # Target-Parameter (gleiche wie LONG)
    tp_points = 50
    sl_points = 25
    horizon = 5

    targets = compute_short_target(close_arr, high_arr, low_arr, horizon, tp_points, sl_points)
    df["target_short"] = targets

    print(f"  SHORT Target Balance: {targets.mean()*100:.1f}% positiv")

    # Splits
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target_short"].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["target_short"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target_short"].values.astype(np.float32)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Train Target: {y_train.mean()*100:.1f}% pos | Test Target: {y_test.mean()*100:.1f}% pos")

    # Training
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluation
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, y_pred_val)
    test_auc = roc_auc_score(y_test, y_pred_test)

    print(f"\n  SHORT Modell Ergebnisse:")
    print(f"    Val AUC: {val_auc:.4f}")
    print(f"    Test AUC: {test_auc:.4f}")

    # Feature Importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 10 Features (SHORT):")
    for i, row in fi.head(10).iterrows():
        print(f"    {row['feature']:25s} ({row['importance']:.4f})")

    # Speichern
    model_path = os.path.join(MODELS_DIR, "xgboost_short.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modell gespeichert: {model_path}")

    return model


if __name__ == "__main__":
    main()
