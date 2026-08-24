"""
Neue Ansätze fuer große TP/SL (250+ Punkte).

Ansatz 1: Richtungs-Modell (Directional)
- Predict ob Preis in 2-4 Stunden hoeher/tiefer ist
- TP/SL danach festlegen

Ansatz 2: Max Excursion Modell
- Predict ob Preis +250 erreicht bevor -100
- Andere Horizonte testen

Ansatz 3: Trailing Stop Modell
- Entry basierend auf Modell
- Trailing Stop statt fixem SL
"""

import os
import sys
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def compute_directional_target(close_arr, horizon):
    """Target: 1 wenn close[horizon] > close[0]"""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    targets[:n-horizon] = (close_arr[horizon:] > close_arr[:n-horizon]).astype(np.float32)
    return targets


def compute_max_excursion_target(high_arr, low_arr, close_arr, horizon, tp_points, sl_points, direction="long"):
    """Target: 1 wenn TP vor SL erreicht ueber den gesamten Horizont."""
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    entry = close_arr

    if direction == "long":
        tp_level = entry + tp_points * POINT
        sl_level = entry - sl_points * POINT
    else:
        tp_level = entry - tp_points * POINT
        sl_level = entry + sl_points * POINT

    for i in range(n - horizon):
        future_high = high_arr[i+1:i+horizon+1]
        future_low = low_arr[i+1:i+horizon+1]

        if direction == "long":
            tp_hit = np.any(future_high >= tp_level[i])
            sl_hit = np.any(future_low <= sl_level[i])
        else:
            tp_hit = np.any(future_low <= tp_level[i])
            sl_hit = np.any(future_high >= sl_level[i])

        if tp_hit and not sl_hit:
            targets[i] = 1
        elif tp_hit and sl_hit:
            # Beide erreicht - welches zuerst?
            tp_idx = np.where(future_high >= tp_level[i])[0][0] if direction == "long" else np.where(future_low <= tp_level[i])[0][0]
            sl_idx = np.where(future_low <= sl_level[i])[0][0] if direction == "long" else np.where(future_high >= sl_level[i])[0][0]
            targets[i] = 1 if tp_idx < sl_idx else 0

    return targets


def generate_signals(df, model, feature_cols, conf_threshold=0.60, max_daily=10):
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]
    hour = df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(len(df)) * 0.001
    has_vol = (atr_norm > 0.0002) & (atr_norm < 0.005)
    signals = np.zeros(len(df), dtype=bool)
    high_conf = predictions >= conf_threshold
    eligible = is_active & has_vol & high_conf
    current_date = None
    daily_trades = 0
    for i in range(len(df)):
        if not eligible[i]:
            continue
        date = df["timestamp"].iloc[i].date()
        if date != current_date:
            current_date = date
            daily_trades = 0
        if daily_trades < max_daily:
            signals[i] = True
            daily_trades += 1
    return signals


def train_and_test(name, df, feature_cols, target_col, tp, sl, horizon, conf_threshold=0.60):
    """Trainiert und testet ein Modell."""
    from xgboost import XGBClassifier

    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df[target_col].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

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

    # OOS Test
    signals = generate_signals(test_df, model, feature_cols, conf_threshold, 10)
    engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=horizon)
    trades_df, stats = engine.run(test_df, signals)

    n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days

    result = {
        "name": name,
        "auc": test_auc,
        "tp": tp, "sl": sl, "horizon": horizon,
        "pos_rate": y_test.mean(),
    }

    if stats and stats["n_trades"] > 10:
        result.update({
            "n_trades": stats["n_trades"],
            "trades_per_day": stats["n_trades"] / n_days,
            "win_rate": stats["win_rate"],
            "pf": stats["profit_factor"],
            "profit": stats["total_profit"],
            "dd": stats["max_drawdown"],
            "avg_trade_005": stats["total_profit"] / stats["n_trades"] * 0.05,
        })
    else:
        result.update({
            "n_trades": 0, "trades_per_day": 0, "win_rate": 0,
            "pf": 0, "profit": 0, "dd": 0, "avg_trade_005": 0,
        })

    return model, result


def main():
    print("=" * 70)
    print:("NEUE ANSAETZE FUER GROSSE TP/SL (250+)")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    results = []

    # === Ansatz 1: Directional Modell ===
    print("\n--- Ansatz 1: Directional Modell ---")
    for horizon in [60, 120, 180, 240]:
        target = compute_directional_target(close_arr, horizon)
        df[f"target_dir_{horizon}"] = target

        # TP/SL basierend auf Horizont
        tp = int(horizon * 5)  # ca. 300-1200 Punkte
        sl = int(horizon * 2)  # ca. 120-480 Punkte

        model, result = train_and_test(
            f"Dir_H{horizon}", df, feature_cols,
            f"target_dir_{horizon}", tp, sl, horizon, 0.55
        )
        results.append(result)
        print(f"  H={horizon}, TP={tp}, SL={sl}: AUC={result['auc']:.4f}, "
              f"Win={result.get('win_rate', 0)*100:.1f}%, PF={result.get('pf', 0):.2f}, "
              f"Trades/D={result.get('trades_per_day', 0):.1f}")

    # === Ansatz 2: Max Excursion mit verschiedenen Parametern ===
    print("\n--- Ansatz 2: Max Excursion ---")
    for horizon in [60, 120, 180, 240]:
        for tp, sl in [(250, 100), (300, 100), (300, 120), (400, 150)]:
            target = compute_max_excursion_target(high_arr, low_arr, close_arr, horizon, tp, sl, "long")
            col_name = f"target_me_{horizon}_{tp}_{sl}"
            df[col_name] = target

            model, result = train_and_test(
                f"ME_H{horizon}_TP{tp}_SL{sl}", df, feature_cols,
                col_name, tp, sl, horizon, 0.55
            )
            results.append(result)
            print(f"  H={horizon}, TP={tp}, SL={sl}: AUC={result['auc']:.4f}, "
                  f"Win={result.get('win_rate', 0)*100:.1f}%, PF={result.get('pf', 0):.2f}")

    # === Ergebnisse ===
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("auc", ascending=False)

    print(f"\n\n{'='*70}")
    print("ERGEBNISSE: NEUE ANSAETZE")
    print(f"{'='*70}")
    print(f"  {'Name':30s} | {'AUC':>6s} | {'Win%':>6s} | {'PF':>5s} | {'Tr/D':>5s} | {'Avg$':>6s}")
    print(f"  {'-'*70}")
    for _, row in results_df.head(15).iterrows():
        print(f"  {row['name']:30s} | {row['auc']:6.4f} | "
              f"{row.get('win_rate', 0)*100:5.1f}% | {row.get('pf', 0):5.2f} | "
              f"{row.get('trades_per_day', 0):5.1f} | {row.get('avg_trade_005', 0):6.2f}")

    # Beste speichern
    best = results_df.iloc[0]
    print(f"\n  BESTER ANSATZ:")
    print(f"    {best['name']}")
    print(f"    AUC: {best['auc']:.4f}")
    print(f"    TP={int(best['tp'])}, SL={int(best['sl'])}, H={int(best['horizon'])}")

    return results_df


if __name__ == "__main__":
    main()
