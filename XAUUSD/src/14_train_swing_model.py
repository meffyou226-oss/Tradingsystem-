"""
Swing-Modell Training für längere Horizonte (30-60 Minuten).

Ziel: TP=200-300, SL=60-100 für 10+ USD Gewinn pro Trade (bei 0.05-0.1 Lots).

Unterschied zum Scalping-Modell:
- Horizont: 30-60 Minuten (statt 5)
- TP/SL: 200-300 / 60-100 (statt 45/15)
- Ziel: Swing-Trading mit größeren Bewegungen
"""

import os
import sys
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                     REPORTS_DIR, POINT)
from data_preparation import load_combined


def compute_swing_target(close_arr, high_arr, low_arr, horizon, tp_points, sl_points):
    """Berechnet Swing-Target mit längerem Horizont."""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    entry = close_arr
    tp_level = entry + tp_points * POINT
    sl_level = entry - sl_points * POINT

    for offset in range(1, horizon + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)

        undetermined = targets[current_idx] == 0

        tp_hits = high_arr[future_idx] >= tp_level[current_idx]
        sl_hits = low_arr[future_idx] <= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        targets[current_idx[new_tp]] = 1

        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def compute_short_swing_target(close_arr, high_arr, low_arr, horizon, tp_points, sl_points):
    """Berechnet SHORT Swing-Target."""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    entry = close_arr
    tp_level = entry - tp_points * POINT
    sl_level = entry + sl_points * POINT

    for offset in range(1, horizon + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)

        undetermined = targets[current_idx] == 0

        tp_hits = low_arr[future_idx] <= tp_level[current_idx]
        sl_hits = high_arr[future_idx] >= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        targets[current_idx[new_tp]] = 1

        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def find_best_swing_params(df, feature_cols):
    """Findet beste TP/SL/Horizont für Swing."""
    print("\n--- Swing-Parameter optimieren ---")

    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    # Teste verschiedene Swing-Konfigurationen
    configs = [
        # (horizon_min, tp_points, sl_points)
        (30, 200, 60), (30, 200, 80), (30, 250, 80),
        (45, 250, 80), (45, 300, 100), (45, 300, 120),
        (60, 300, 100), (60, 300, 150), (60, 400, 150),
    ]

    results = []
    for horizon, tp, sl in configs:
        targets = compute_swing_target(close_arr, high_arr, low_arr, horizon, tp, sl)
        pos_rate = targets.mean()

        # Prüfe Balance (sollte nahe 50% sein)
        if 0.3 <= pos_rate <= 0.7:
            results.append({
                "horizon": horizon,
                "tp": tp,
                "sl": sl,
                "pos_rate": pos_rate,
                "balance": abs(pos_rate - 0.5),
            })
            print(f"  H={horizon}, TP={tp}, SL={sl}: Pos={pos_rate*100:.1f}%")

    # Wähle die beste Balance (am nächsten an 50%)
    results_df = pd.DataFrame(results).sort_values("balance")
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n  Beste Swing-Config: H={int(best['horizon'])}, TP={int(best['tp'])}, SL={int(best['sl'])}")
        return int(best["horizon"]), int(best["tp"]), int(best["sl"])

    # Fallback
    return 45, 300, 100


def train_swing_model(force_config=None):
    """Trainiert das Swing-Modell."""
    print("=" * 70)
    print("SWING-MODELL TRAINING")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Parameter finden
    if force_config:
        horizon, tp, sl = force_config
    else:
        horizon, tp, sl = find_best_swing_params(df, feature_cols)

    print(f"\n  Swing-Parameter: Horizont={horizon}min, TP={tp}, SL={sl}")

    # Targets berechnen
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    df["target_swing"] = compute_swing_target(close_arr, high_arr, low_arr, horizon, tp, sl)
    df["target_swing_short"] = compute_short_swing_target(close_arr, high_arr, low_arr, horizon, tp, sl)

    print(f"  LONG Swing Target: {df['target_swing'].mean()*100:.1f}% positiv")
    print(f"  SHORT Swing Target: {df['target_swing_short'].mean()*100:.1f}% positiv")

    # Splits
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # --- LONG Swing Modell ---
    print(f"\n--- LONG Swing Modell ---")
    y_train_long = train_df["target_swing"].values.astype(np.float32)
    y_val_long = val_df["target_swing"].values.astype(np.float32)
    y_test_long = test_df["target_swing"].values.astype(np.float32)

    from xgboost import XGBClassifier

    model_long = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model_long.fit(X_train, y_train_long, eval_set=[(X_val, y_val_long)], verbose=False)

    val_pred_long = model_long.predict_proba(X_val)[:, 1]
    test_pred_long = model_long.predict_proba(X_test)[:, 1]

    val_auc_long = roc_auc_score(y_val_long, val_pred_long)
    test_auc_long = roc_auc_score(y_test_long, test_pred_long)

    print(f"  Val AUC: {val_auc_long:.4f}")
    print(f"  Test AUC: {test_auc_long:.4f}")

    # --- SHORT Swing Modell ---
    print(f"\n--- SHORT Swing Modell ---")
    y_train_short = train_df["target_swing_short"].values.astype(np.float32)
    y_val_short = val_df["target_swing_short"].values.astype(np.float32)
    y_test_short = test_df["target_swing_short"].values.astype(np.float32)

    model_short = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model_short.fit(X_train, y_train_short, eval_set=[(X_val, y_val_short)], verbose=False)

    val_pred_short = model_short.predict_proba(X_val)[:, 1]
    test_pred_short = model_short.predict_proba(X_test)[:, 1]

    val_auc_short = roc_auc_score(y_val_short, val_pred_short)
    test_auc_short = roc_auc_score(y_test_short, test_pred_short)

    print(f"  Val AUC: {val_auc_short:.4f}")
    print(f"  Test AUC: {test_auc_short:.4f}")

    # Speichern
    long_path = os.path.join(MODELS_DIR, "xgboost_swing.pkl")
    short_path = os.path.join(MODELS_DIR, "xgboost_swing_short.pkl")

    with open(long_path, "wb") as f:
        pickle.dump(model_long, f)
    with open(short_path, "wb") as f:
        pickle.dump(model_short, f)

    print(f"\n  Modelle gespeichert:")
    print(f"    {long_path}")
    print(f"    {short_path}")

    # Config speichern
    swing_config = {
        "horizon": horizon,
        "tp_points": tp,
        "sl_points": sl,
        "long_auc": test_auc_long,
        "short_auc": test_auc_short,
    }
    config_path = os.path.join(MODELS_DIR, "swing_config.json")
    with open(config_path, "w") as f:
        json.dump(swing_config, f, indent=2)

    return swing_config


def train_swing_model(force_config=None):
    print("=" * 70)
    print("SWING-MODELL TRAINING")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Parameter finden
    if force_config:
        horizon, tp, sl = force_config
    else:
        horizon, tp, sl = find_best_swing_params(df, feature_cols)

    print(f"\n  Swing-Parameter: Horizont={horizon}min, TP={tp}, SL={sl}")

    # Targets berechnen
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    df["target_swing"] = compute_swing_target(close_arr, high_arr, low_arr, horizon, tp, sl)
    df["target_swing_short"] = compute_short_swing_target(close_arr, high_arr, low_arr, horizon, tp, sl)

    print(f"  LONG Swing Target: {df['target_swing'].mean()*100:.1f}% positiv")
    print(f"  SHORT Swing Target: {df['target_swing_short'].mean()*100:.1f}% positiv")

    # Splits
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # --- LONG Swing Modell ---
    print(f"\n--- LONG Swing Modell ---")
    y_train_long = train_df["target_swing"].values.astype(np.float32)
    y_val_long = val_df["target_swing"].values.astype(np.float32)
    y_test_long = test_df["target_swing"].values.astype(np.float32)

    from xgboost import XGBClassifier

    model_long = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model_long.fit(X_train, y_train_long, eval_set=[(X_val, y_val_long)], verbose=False)

    val_pred_long = model_long.predict_proba(X_val)[:, 1]
    test_pred_long = model_long.predict_proba(X_test)[:, 1]

    val_auc_long = roc_auc_score(y_val_long, val_pred_long)
    test_auc_long = roc_auc_score(y_test_long, test_pred_long)

    print(f"  Val AUC: {val_auc_long:.4f}")
    print(f"  Test AUC: {test_auc_long:.4f}")

    # --- SHORT Swing Modell ---
    print(f"\n--- SHORT Swing Modell ---")
    y_train_short = train_df["target_swing_short"].values.astype(np.float32)
    y_val_short = val_df["target_swing_short"].values.astype(np.float32)
    y_test_short = test_df["target_swing_short"].values.astype(np.float32)

    model_short = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model_short.fit(X_train, y_train_short, eval_set=[(X_val, y_val_short)], verbose=False)

    val_pred_short = model_short.predict_proba(X_val)[:, 1]
    test_pred_short = model_short.predict_proba(X_test)[:, 1]

    val_auc_short = roc_auc_score(y_val_short, val_pred_short)
    test_auc_short = roc_auc_score(y_test_short, test_pred_short)

    print(f"  Val AUC: {val_auc_short:.4f}")
    print(f"  Test AUC: {test_auc_short:.4f}")

    # Speichern
    long_path = os.path.join(MODELS_DIR, "xgboost_swing.pkl")
    short_path = os.path.join(MODELS_DIR, "xgboost_swing_short.pkl")

    with open(long_path, "wb") as f:
        pickle.dump(model_long, f)
    with open(short_path, "wb") as f:
        pickle.dump(model_short, f)

    print(f"\n  Modelle gespeichert:")
    print(f"    {long_path}")
    print(f"    {short_path}")

    # Config speichern
    swing_config = {
        "horizon": horizon,
        "tp_points": tp,
        "sl_points": sl,
        "long_auc": test_auc_long,
        "short_auc": test_auc_short,
    }
    config_path = os.path.join(MODELS_DIR, "swing_config.json")
    with open(config_path, "w") as f:
        json.dump(swing_config, f, indent=2)

    return swing_config


if __name__ == "__main__":
    # Best config from optimization: H=45, TP=80, SL=30
    # But for 10+ USD profit at 0.05 lots, we need TP=200+
    # Train both: small TP for high winrate, large TP for swing
    print("\n=== SMALL SWING (TP=80, SL=30) ===")
    config_small = train_swing_model(force_config=(45, 80, 30))
    print(f"\n  Small Swing Config: {config_small}")

    print("\n\n=== LARGE SWING (TP=200, SL=60) ===")
    config_large = train_swing_model(force_config=(60, 200, 60))
    print(f"\n  Large Swing Config: {config_large}")
