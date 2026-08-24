"""
Debug: Warum werden keine Trades generiert?
Teste M1-Modell mit verschiedenen TP/SL und prüfe Signale.
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
                     TEST_START, TEST_END, MODELS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def generate_signals(df, model, feature_cols, conf_threshold=0.60, max_daily=10):
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
    print("DEBUG: M1 Modell Signale analysieren")
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

    # Signale mit niedrigem Threshold
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        signals_long = generate_signals(test_df, model_long, feature_cols, threshold, 10)
        signals_short = generate_signals(test_df, model_short, feature_cols, threshold, 10)

        n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days

        print(f"\n  Threshold={threshold}:")
        print(f"    LONG Signale: {signals_long.sum()} ({signals_long.sum()/n_days:.1f}/Tag)")
        print(f"    SHORT Signale: {signals_short.sum()} ({signals_short.sum()/n_days:.1f}/Tag)")

    # Teste mit TP=250/SL=100 und verschiedenen Horizonten
    print(f"\n\n  Backtest mit TP=250, SL=100:")
    signals_long = generate_signals(test_df, model_long, feature_cols, 0.60, 10)
    signals_short = generate_signals(test_df, model_short, feature_cols, 0.60, 10)

    for horizon in [30, 60, 120, 180, 240]:
        engine = BacktestEngine(tp_points=250, sl_points=100, horizon=horizon)
        trades_long, stats_long = engine.run(test_df, signals_long)
        trades_short, stats_short = engine.run(test_df, signals_short)

        n_trades = 0
        if stats_long:
            n_trades += stats_long["n_trades"]
        if stats_short:
            n_trades += stats_short["n_trades"]

        print(f"    H={horizon}min: {n_trades} Trades")
        if stats_long:
            print(f"      LONG: {stats_long['n_trades']} Trades, Win={stats_long['win_rate']*100:.0f}%")
        if stats_short:
            print(f"      SHORT: {stats_short['n_trades']} Trades, Win={stats_short['win_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
