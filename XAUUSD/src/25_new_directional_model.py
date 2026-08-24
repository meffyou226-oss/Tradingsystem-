"""
Neues LONG-Modell trainieren mit Directional-Target.

Target: 1 wenn Preis in X Minuten hoeher ist als jetzt.
Das balanciert besser und funktioniert fuer alle TP/SL.
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
                     TEST_START, TEST_END, MODELS_DIR, REPORTS_DIR)
from data_preparation import load_combined


def compute_directional_target(close_arr, horizon_minutes):
    """Target: 1 wenn close[horizon] > close[0]"""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    targets[:n-horizon_minutes] = (close_arr[horizon_minutes:] > close_arr[:n-horizon_minutes]).astype(np.float32)
    return targets


def main():
    print("=" * 70)
    print("NEUES LONG-MODELL: Directional Target")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    close_arr = df["close"].values.astype(np.float64)

    # Verschiedene Horizonte testen
    horizons = [15, 30, 45, 60, 90, 120]

    results = []
    for horizon in horizons:
        target = compute_directional_target(close_arr, horizon)
        df["target_dir"] = target

        pos_rate = target.mean()
        print(f"\n  H={horizon}min: Pos Rate = {pos_rate*100:.1f}%")

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

        y_train = train_df["target_dir"].values.astype(np.float32)
        y_val = val_df["target_dir"].values.astype(np.float32)
        y_test = test_df["target_dir"].values.astype(np.float32)

        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss",
            early_stopping_rounds=50
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)

        # Pruefe Predictions
        pred_mean = test_pred.mean()
        pred_high = (test_pred > 0.5).mean()

        print(f"    Test AUC: {test_auc:.4f}")
        print(f"    Pred Mean: {pred_mean:.3f}, Pred > 0.5: {pred_high*100:.1f}%")

        results.append({
            "horizon": horizon,
            "pos_rate": pos_rate,
            "auc": test_auc,
            "pred_mean": pred_mean,
            "pred_high": pred_high,
        })

        # Speichern wenn gut
        if test_auc > 0.55 and pred_high > 0.2:
            model_path = os.path.join(MODELS_DIR, f"xgboost_directional_h{horizon}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"    -> Gespeichert: {model_path}")

    # Zusammenfassung
    results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    print(f"\n\n  BESTE KONFIGURATION:")
    best = results_df.iloc[0]
    print(f"    H={int(best['horizon'])}min, AUC={best['auc']:.4f}, Pred>0.5={best['pred_high']*100:.1f}%")


if __name__ == "__main__":
    main()
