"""
Swing-Modell optimieren: Finde beste Balance aus TP/SL/Horizont.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, POINT)
from data_preparation import load_combined


def compute_target(close_arr, high_arr, low_arr, horizon, tp_points, sl_points, direction="long"):
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)

    entry = close_arr
    if direction == "long":
        tp_level = entry + tp_points * POINT
        sl_level = entry - sl_points * POINT
    else:
        tp_level = entry - tp_points * POINT
        sl_level = entry + sl_points * POINT

    for offset in range(1, horizon + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)

        undetermined = targets[current_idx] == 0

        if direction == "long":
            tp_hits = high_arr[future_idx] >= tp_level[current_idx]
            sl_hits = low_arr[future_idx] <= sl_level[current_idx]
        else:
            tp_hits = low_arr[future_idx] <= tp_level[current_idx]
            sl_hits = high_arr[future_idx] >= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        targets[current_idx[new_tp]] = 1

        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def main():
    print("=" * 70)
    print("SWING-MODELL OPTIMIERUNG")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    # Teste verschiedene Swing-Konfigurationen
    configs = []
    for horizon in [15, 20, 30, 45, 60]:
        for tp in [80, 100, 120, 150, 200, 250, 300]:
            for sl in [30, 40, 50, 60, 80, 100]:
                if tp / sl < 1.5:  # Mindestens 1.5:1 R:R
                    continue
                if tp / sl > 4:  # Max 4:1
                    continue

                # Schnelle Balance-Prüfung
                targets = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "long")
                pos_rate = targets.mean()

                # Filter: 25-55% positiv (gut balanciert für ML)
                if 0.25 <= pos_rate <= 0.55:
                    configs.append({
                        "horizon": horizon,
                        "tp": tp,
                        "sl": sl,
                        "rr": tp / sl,
                        "pos_rate": pos_rate,
                        "balance_score": 1 - abs(pos_rate - 0.4),  # 40% ist ideal
                    })

    configs_df = pd.DataFrame(configs).sort_values("balance_score", ascending=False)

    print(f"\n  Top 20 Swing-Konfigurationen (nach Balance):")
    print(f"  {'H':>4s} | {'TP':>5s} | {'SL':>5s} | {'R:R':>5s} | {'Pos%':>6s} | {'Score':>6s}")
    print(f"  {'-'*45}")
    for _, row in configs_df.head(20).iterrows():
        print(f"  {int(row['horizon']):4d} | {int(row['tp']):5d} | {int(row['sl']):5d} | "
              f"{row['rr']:5.1f} | {row['pos_rate']*100:5.1f}% | {row['balance_score']:6.3f}")

    # Top-5 Konfigurationen trainieren
    print(f"\n--- Top-5 trainieren ---")

    from xgboost import XGBClassifier

    results = []
    for _, config in configs_df.head(5).iterrows():
        horizon = int(config["horizon"])
        tp = int(config["tp"])
        sl = int(config["sl"])

        # Targets
        target_long = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "long")
        target_short = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "short")

        df["target_swing_long"] = target_long
        df["target_swing_short"] = target_short

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

        # LONG
        y_train = train_df["target_swing_long"].values.astype(np.float32)
        y_val = val_df["target_swing_long"].values.astype(np.float32)
        y_test = test_df["target_swing_long"].values.astype(np.float32)

        model_long = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss"
        )
        model_long.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        test_pred = model_long.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)

        # SHORT
        y_train_s = train_df["target_swing_short"].values.astype(np.float32)
        y_val_s = val_df["target_swing_short"].values.astype(np.float32)
        y_test_s = test_df["target_swing_short"].values.astype(np.float32)

        model_short = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss"
        )
        model_short.fit(X_train, y_train_s, eval_set=[(X_val, y_val_s)], verbose=False)

        test_pred_s = model_short.predict_proba(X_test)[:, 1]
        test_auc_s = roc_auc_score(y_test_s, test_pred_s)

        results.append({
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "rr": config["rr"],
            "pos_rate": config["pos_rate"],
            "long_auc": test_auc,
            "short_auc": test_auc_s,
            "avg_auc": (test_auc + test_auc_s) / 2,
        })

        print(f"  H={horizon:2d}, TP={tp:3d}, SL={sl:2d}: LONG_AUC={test_auc:.4f}, SHORT_AUC={test_auc_s:.4f}")

    results_df = pd.DataFrame(results).sort_values("avg_auc", ascending=False)
    print(f"\n  BESTE SWING-KONFIGURATION:")
    best = results_df.iloc[0]
    print(f"    H={int(best['horizon'])}, TP={int(best['tp'])}, SL={int(best['sl'])}")
    print(f"    LONG AUC: {best['long_auc']:.4f}")
    print(f"    SHORT AUC: {best['short_auc']:.4f}")

    return best


if __name__ == "__main__":
    main()
