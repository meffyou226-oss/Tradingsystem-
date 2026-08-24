"""
M1 Modell mit grossen TP/SL testen.

Ziel: 2-5 Signale/Tag mit TP=250+, hohe Winrate.
Ansatz: M1-Modell (AUC=0.644) mit TP=250/SL=100.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS,
                     TEST_START, TEST_END, MODELS_DIR, REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def generate_signals(df, model, feature_cols, conf_threshold=0.65, max_daily=10):
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]

    hour = df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(len(df)) * 0.001
    has_vol = atr_norm > 0.0001

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
    print("M1 MODELL MIT GROSSEM TP/SL (250/100)")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    print(f"  Test: {len(test_df)} M1 Kerzen")

    # M1 Modelle laden
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model_long = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "xgboost_short.pkl"), "rb") as f:
        model_short = pickle.load(f)

    # Verschiedene TP/SL testen mit LANGEM Horizont
    configs = [
        (250, 100, 120),
        (250, 100, 180),
        (300, 100, 180),
        (300, 120, 240),
        (400, 150, 240),
        (500, 150, 360),
    ]

    results = []
    for tp, sl, horizon in configs:
        print(f"\n--- TP={tp}, SL={sl}, H={horizon}min ---")

        # Signale generieren
        signals_long = generate_signals(test_df, model_long, feature_cols, 0.65, 10)
        signals_short = generate_signals(test_df, model_short, feature_cols, 0.65, 10)

        engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=horizon)
        trades_long, stats_long = engine.run(test_df, signals_long)
        trades_short, stats_short = engine.run(test_df, signals_short)

        n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days

        if stats_long and stats_short:
            total_trades = stats_long["n_trades"] + stats_short["n_trades"]
            total_profit = stats_long["total_profit"] + stats_short["total_profit"]
            avg_win = (stats_long["win_rate"] + stats_short["win_rate"]) / 2

            results.append({
                "tp": tp, "sl": sl, "horizon": horizon,
                "total_trades": total_trades,
                "trades_per_day": total_trades / n_days,
                "win_rate": avg_win,
                "long_pf": stats_long["profit_factor"],
                "short_pf": stats_short["profit_factor"],
                "total_profit": total_profit,
                "profit_005": total_profit * 0.05,
                "max_dd": min(stats_long["max_drawdown"], stats_short["max_drawdown"]),
            })

            print(f"  Trades: {total_trades} ({total_trades/n_days:.1f}/Tag)")
            print(f"  Winrate: {avg_win*100:.1f}%")
            print(f"  Profit: {total_profit:.0f} Punkte ({total_profit*0.05:.1f} USD bei 0.05L)")
            print(f"  PF: L={stats_long['profit_factor']:.1f}/S={stats_short['profit_factor']:.1f}")

    # Ergebnisse
    results_df = pd.DataFrame(results).sort_values("trades_per_day", ascending=False)

    print(f"\n\n{'='*70}")
    print("ERGEBNISSE: M1 MIT GROSSEM TP/SL")
    print(f"{'='*70}")
    print(f"  {'TP':>5s} | {'SL':>5s} | {'H':>4s} | {'Tr/D':>5s} | {'Win%':>6s} | {'PF-L':>5s} | {'PF-S':>5s} | {'Profit':>8s} | {'USD':>8s}")
    print(f"  {'-'*75}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['tp']):5d} | {int(row['sl']):5d} | {int(row['horizon']):4d} | "
              f"{row['trades_per_day']:5.1f} | {row['win_rate']*100:5.1f}% | "
              f"{row['long_pf']:5.1f} | {row['short_pf']:5.1f} | "
              f"{int(row['total_profit']):8d} | {row['profit_005']:8.1f}")

    # Beste Konfiguration (2-5 Trades/Tag)
    viable = results_df[(results_df["trades_per_day"] >= 2) & (results_df["trades_per_day"] <= 5)]
    if len(viable) > 0:
        best = viable.iloc[0]
        print(f"\n  BESTE KONFIG (2-5 Trades/Tag):")
        print(f"    TP={int(best['tp'])}, SL={int(best['sl'])}, H={int(best['horizon'])}min")
        print(f"    Trades/Tag: {best['trades_per_day']:.1f}")
        print(f"    Winrate: {best['win_rate']*100:.1f}%")
        print(f"    Profit: {best['profit_005']:.1f} USD/Monat (0.05L)")
    else:
        print(f"\n  Keine Konfiguration mit 2-5 Trades/Tag gefunden!")
        print(f"  Beste: TP={int(results_df.iloc[0]['tp'])}, SL={int(results_df.iloc[0]['sl'])} "
              f"mit {results_df.iloc[0]['trades_per_day']:.1f} Trades/Tag")


if __name__ == "__main__":
    main()
