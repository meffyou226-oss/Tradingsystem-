"""
OOS Test: Große TP/SL für Live-Trading mit realistischen Lot-Größen.

Ziel: Mindestens 10 EUR/USD Gewinn pro Trade.

Bei 0.05 Lots = 0.05 USD/Punkt:
  TP=200 → 10 USD Gewinn
  TP=300 → 15 USD Gewinn

Bei 0.1 Lots = 0.10 USD/Punkt:
  TP=100 → 10 USD Gewinn
  TP=150 → 15 USD Gewinn
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN,
                     TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                     REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def generate_signals(df, model, feature_cols, conf_threshold=0.75, max_daily=8):
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]

    hour = df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(len(df)) * 0.001
    has_vol = (atr_norm > 0.0003) & (atr_norm < 0.005)

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


def test_large_tp_sl():
    print("=" * 70)
    print("OOS TEST: GROSSE TP/SL (10+ USD Gewinn pro Trade)")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    signals = generate_signals(test_df, model, feature_cols)
    n_signals = signals.sum()
    print(f"  Signale: {n_signals}")

    # Große TP/SL-Kombinationen mit angepasstem Horizont
    # Für große TP/SL brauchen wir längeren Horizont (Swing)
    combinations = [
        # TP/SL mit Horizont = TP/10 (mindestens 10 Minuten)
        (100, 30, 15), (100, 40, 15), (100, 50, 15),
        (150, 50, 20), (150, 60, 20),
        (200, 60, 25), (200, 80, 25),
        (250, 80, 30), (300, 100, 30),
        (400, 100, 45), (500, 150, 60),
    ]

    results = []
    for tp, sl, horizon in combinations:
        engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=horizon)
        trades_df, stats = engine.run(test_df, signals)

        if stats and stats["n_trades"] > 10:
            # Berechne Gewinn/Verlust in USD für verschiedene Lot-Größen
            profit_005 = stats["total_profit"] * 0.05  # 0.05 Lots
            profit_01 = stats["total_profit"] * 0.10   # 0.1 Lots
            avg_trade_005 = stats["total_profit"] / stats["n_trades"] * 0.05
            avg_trade_01 = stats["total_profit"] / stats["n_trades"] * 0.10

            results.append({
                "tp": tp,
                "sl": sl,
                "horizon": horizon,
                "rr": f"{tp/sl:.1f}:1",
                "n_trades": stats["n_trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "total_profit_points": stats["total_profit"],
                "profit_005_lots": profit_005,
                "profit_01_lots": profit_01,
                "avg_trade_005": avg_trade_005,
                "avg_trade_01": avg_trade_01,
                "max_drawdown": stats["max_drawdown"],
                "tp_rate": stats["tp_hit_rate"],
                "sl_rate": stats["sl_hit_rate"],
                "expiry_rate": stats["expiry_rate"],
            })

    results_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)

    print(f"\n  Ergebnisse (sortiert nach PF):")
    print(f"  {'TP':>5s} | {'SL':>5s} | {'H':>3s} | {'R:R':>5s} | {'Tr':>5s} | {'Win%':>6s} | {'PF':>5s} | {'P(0.05)':>8s} | {'P(0.1)':>8s} | {'Avg005':>7s} | {'DD':>5s}")
    print(f"  {'-'*105}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['tp']):5d} | {int(row['sl']):5d} | {int(row['horizon']):3d} | "
              f"{row['rr']:>5s} | {int(row['n_trades']):5d} | {row['win_rate']*100:5.1f}% | "
              f"{row['profit_factor']:5.2f} | {row['profit_005_lots']:8.1f} | "
              f"{row['profit_01_lots']:8.1f} | {row['avg_trade_005']:7.2f} | "
              f"{row['max_drawdown']:5.0f}")

    # Beste für 0.05 Lots (mindestens 10 USD Avg Trade)
    viable_005 = results_df[results_df["avg_trade_005"] >= 5]  # Mindestens 5 USD pro Trade
    if len(viable_005) > 0:
        best_005 = viable_005.iloc[0]
        print(f"\n  BESTE KONFIG FUER 0.05 LOTS:")
        print(f"    TP={int(best_005['tp'])}, SL={int(best_005['sl'])}, H={int(best_005['horizon'])}")
        print(f"    Win={best_005['win_rate']*100:.1f}% | PF={best_005['profit_factor']:.2f}")
        print(f"    Avg Trade: {best_005['avg_trade_005']:.2f} USD | Total: {best_005['profit_005_lots']:.0f} USD")
        print(f"    Max DD: {best_005['max_drawdown']:.0f} Punkte ({best_005['max_drawdown']*0.05:.1f} USD)")

    # Beste für 0.1 Lots
    viable_01 = results_df[results_df["avg_trade_01"] >= 8]
    if len(viable_01) > 0:
        best_01 = viable_01.iloc[0]
        print(f"\n  BESTE KONFIG FUER 0.1 LOTS:")
        print(f"    TP={int(best_01['tp'])}, SL={int(best_01['sl'])}, H={int(best_01['horizon'])}")
        print(f"    Win={best_01['win_rate']*100:.1f}% | PF={best_01['profit_factor']:.2f}")
        print(f"    Avg Trade: {best_01['avg_trade_01']:.2f} USD | Total: {best_01['profit_01_lots']:.0f} USD")
        print(f"    Max DD: {best_01['max_drawdown']:.0f} Punkte ({best_01['max_drawdown']*0.10:.1f} USD)")

    # Speichern
    results_path = os.path.join(REPORTS_DIR, "tp_sl_large_oos_test.json")
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\n  Gespeichert: {results_path}")

    return results_df


if __name__ == "__main__":
    test_large_tp_sl()
