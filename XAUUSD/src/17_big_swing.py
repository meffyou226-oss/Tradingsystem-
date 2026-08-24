"""
OOS Test: Große TP/SL (250+ Points) für Live-Trading.

Ziel: Mindestens 250 Points TP für 12.5+ USD bei 0.05 Lots.
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
                     TEST_START, TEST_END, MODELS_DIR, REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
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


def main():
    print("=" * 70)
    print("GROSSER SWING: TP=250+ (12.5+ USD bei 0.05 Lots)")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    # Teste verschiedene große TP/SL-Konfigurationen
    configs = []
    for horizon in [60, 90, 120, 180]:
        for tp in [200, 250, 300, 400]:
            for sl in [80, 100, 120, 150]:
                if tp / sl < 1.5 or tp / sl > 4:
                    continue
                targets = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "long")
                pos_rate = targets.mean()
                if 0.15 <= pos_rate <= 0.45:
                    configs.append({
                        "horizon": horizon, "tp": tp, "sl": sl,
                        "rr": tp / sl, "pos_rate": pos_rate,
                    })

    configs_df = pd.DataFrame(configs)
    print(f"  {len(configs_df)} Konfigurationen gefunden")

    # Top-5 nach Balance
    configs_df["score"] = 1 - abs(configs_df["pos_rate"] - 0.30)
    top_configs = configs_df.sort_values("score", ascending=False).head(5)

    print(f"\n  Top-5 Konfigurationen:")
    for _, row in top_configs.iterrows():
        print(f"    H={int(row['horizon']):3d}, TP={int(row['tp']):3d}, SL={int(row['sl']):3d} | "
              f"R:R={row['rr']:.1f}:1 | Pos={row['pos_rate']*100:.1f}%")

    # Trainiere Top-3
    from xgboost import XGBClassifier

    results = []
    for idx, config in top_configs.head(3).iterrows():
        horizon = int(config["horizon"])
        tp = int(config["tp"])
        sl = int(config["sl"])

        print(f"\n--- Training: H={horizon}, TP={tp}, SL={sl} ---")

        # Targets
        df["target_long"] = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "long")
        df["target_short"] = compute_target(close_arr, high_arr, low_arr, horizon, tp, sl, "short")

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
        y_train = train_df["target_long"].values.astype(np.float32)
        y_val = val_df["target_long"].values.astype(np.float32)
        y_test = test_df["target_long"].values.astype(np.float32)

        model_long = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss",
            early_stopping_rounds=50
        )
        model_long.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        test_pred = model_long.predict_proba(X_test)[:, 1]
        auc_long = roc_auc_score(y_test, test_pred)

        # SHORT
        y_train_s = train_df["target_short"].values.astype(np.float32)
        y_val_s = val_df["target_short"].values.astype(np.float32)
        y_test_s = test_df["target_short"].values.astype(np.float32)

        model_short = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss",
            early_stopping_rounds=50
        )
        model_short.fit(X_train, y_train_s, eval_set=[(X_val, y_val_s)], verbose=False)
        test_pred_s = model_short.predict_proba(X_test)[:, 1]
        auc_short = roc_auc_score(y_test_s, test_pred_s)

        print(f"  LONG AUC: {auc_long:.4f}")
        print(f"  SHORT AUC: {auc_short:.4f}")

        # Speichern
        long_path = os.path.join(MODELS_DIR, f"xgboost_bigswing_h{horizon}_tp{tp}_sl{sl}.pkl")
        short_path = os.path.join(MODELS_DIR, f"xgboost_bigswing_short_h{horizon}_tp{tp}_sl{sl}.pkl")
        with open(long_path, "wb") as f:
            pickle.dump(model_long, f)
        with open(short_path, "wb") as f:
            pickle.dump(model_short, f)

        # OOS Test mit niedrigerem Threshold
        signals_long = generate_signals(test_df, model_long, feature_cols, 0.60, 10)
        signals_short = generate_signals(test_df, model_short, feature_cols, 0.60, 10)

        engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=horizon)
        trades_long, stats_long = engine.run(test_df, signals_long)
        trades_short, stats_short = engine.run(test_df, signals_short)

        if stats_long and stats_short:
            n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days
            total_trades = stats_long["n_trades"] + stats_short["n_trades"]

            results.append({
                "horizon": horizon, "tp": tp, "sl": sl,
                "rr": f"{tp/sl:.1f}:1",
                "long_auc": auc_long, "short_auc": auc_short,
                "avg_auc": (auc_long + auc_short) / 2,
                "total_trades": total_trades,
                "trades_per_day": total_trades / n_days,
                "long_winrate": stats_long["win_rate"],
                "short_winrate": stats_short["win_rate"],
                "long_pf": stats_long["profit_factor"],
                "short_pf": stats_short["profit_factor"],
                "total_profit": stats_long["total_profit"] + stats_short["total_profit"],
                "max_dd": min(stats_long["max_drawdown"], stats_short["max_drawdown"]),
                "avg_trade_005": (stats_long["total_profit"] + stats_short["total_profit"]) / total_trades * 0.05 if total_trades > 0 else 0,
            })
        else:
            print(f"  Keine Trades generiert!")

    # Ergebnisse
    if results:
        results_df = pd.DataFrame(results).sort_values("avg_auc", ascending=False)
        print(f"\n\n{'='*70}")
        print("ERGEBNISSE: GROSSER SWING (TP=250+)")
        print(f"{'='*70}")
        print(f"  {'H':>4s} | {'TP':>5s} | {'SL':>5s} | {'Tr/D':>5s} | {'L-Win':>6s} | {'S-Win':>6s} | {'L-PF':>5s} | {'S-PF':>5s} | {'AUC':>6s} | {'Avg$':>6s}")
        print(f"  {'-'*85}")
        for _, row in results_df.iterrows():
            print(f"  {int(row['horizon']):4d} | {int(row['tp']):5d} | {int(row['sl']):5d} | "
                  f"{row['trades_per_day']:5.1f} | {row['long_winrate']*100:5.1f}% | "
                  f"{row['short_winrate']*100:5.1f}% | {row['long_pf']:5.2f} | "
                  f"{row['short_pf']:5.2f} | {row['avg_auc']:6.4f} | {row['avg_trade_005']:6.2f}")

        if len(results_df) > 0:
            best = results_df.iloc[0]
            print(f"\n  BESTE KONFIGURATION:")
            print(f"    H={int(best['horizon'])}, TP={int(best['tp'])}, SL={int(best['sl'])}")
            print(f"    Trades/Tag: {best['trades_per_day']:.1f}")
            print(f"    LONG: Win={best['long_winrate']*100:.1f}%, PF={best['long_pf']:.2f}")
            print(f"    SHORT: Win={best['short_winrate']*100:.1f}%, PF={best['short_pf']:.2f}")
            print(f"    Avg AUC: {best['avg_auc']:.4f}")
            print(f"    Avg Trade (0.05L): {best['avg_trade_005']:.2f} USD")
    else:
        print(f"\n  Keine erfolgreichen Trades generiert!")
        print(f"  Problem: TP=250+ erfordert längeren Horizont und andere Features.")


if __name__ == "__main__":
    main()
