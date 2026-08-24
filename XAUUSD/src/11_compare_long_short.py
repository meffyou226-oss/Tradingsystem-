"""
Vergleich LONG vs SHORT Modell Performance.
Gleiche Testmethodik für fairen Vergleich.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TARGET_PARAMS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
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

    return signals, predictions


def compare_models():
    print("=" * 70)
    print("LONG vs SHORT MODELL VERGLEICH")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    # Modelle laden
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model_long = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "xgboost_short.pkl"), "rb") as f:
        model_short = pickle.load(f)

    # Signale generieren
    signals_long, preds_long = generate_signals(test_df, model_long, feature_cols)
    signals_short, preds_short = generate_signals(test_df, model_short, feature_cols)

    engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)

    # Backtests
    trades_long, stats_long = engine.run(test_df, signals_long)
    trades_short, stats_short = engine.run(test_df, signals_short)

    n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days

    print(f"\n{'Metrik':25s} | {'LONG':>12s} | {'SHORT':>12s} | {'Besser':>8s}")
    print("-" * 65)

    metrics = [
        ("Trades", stats_long["n_trades"], stats_short["n_trades"]),
        ("Trades/Tag", f"{stats_long['n_trades']/n_days:.1f}", f"{stats_short['n_trades']/n_days:.1f}"),
        ("Win Rate", f"{stats_long['win_rate']*100:.1f}%", f"{stats_short['win_rate']*100:.1f}%"),
        ("Profit Factor", f"{stats_long['profit_factor']:.2f}", f"{stats_short['profit_factor']:.2f}"),
        ("Total Profit", f"{stats_long['total_profit']:.0f}", f"{stats_short['total_profit']:.0f}"),
        ("Max Drawdown", f"{stats_long['max_drawdown']:.0f}", f"{stats_short['max_drawdown']:.0f}"),
        ("Sharpe", f"{stats_long['sharpe_ratio']:.1f}", f"{stats_short['sharpe_ratio']:.1f}"),
        ("TP-Hit", f"{stats_long['tp_hit_rate']*100:.1f}%", f"{stats_short['tp_hit_rate']*100:.1f}%"),
        ("SL-Hit", f"{stats_long['sl_hit_rate']*100:.1f}%", f"{stats_short['sl_hit_rate']*100:.1f}%"),
        ("Avg Trade", f"{stats_long['total_profit']/stats_long['n_trades']:.3f}", f"{stats_short['total_profit']/stats_short['n_trades']:.3f}"),
        ("Max Cons. Losses", stats_long["max_consec_losses"], stats_short["max_consec_losses"]),
        ("Max Cons. Wins", stats_long["max_consec_wins"], stats_short["max_consec_wins"]),
    ]

    for name, long_val, short_val in metrics:
        try:
            l = float(str(long_val).replace("%", ""))
            s = float(str(short_val).replace("%", ""))
            if name in ["Max Drawdown", "Max Cons. Losses"]:
                besser = "LONG" if l >= s else "SHORT"
            else:
                besser = "LONG" if l > s else "SHORT" if s > l else "GLICH"
        except:
            besser = "-"
        print(f"{name:25s} | {str(long_val):>12s} | {str(short_val):>12s} | {besser:>8s}")

    # Monatlicher Vergleich
    print(f"\n{'Monat':>8s} | {'LONG Profit':>12s} | {'SHORT Profit':>12s} | {'Combined':>10s}")
    print("-" * 55)

    trades_long["month"] = trades_long["entry_time"].dt.to_period("M")
    trades_short["month"] = trades_short["entry_time"].dt.to_period("M")

    monthly_long = trades_long.groupby("month")["pnl"].sum()
    monthly_short = trades_short.groupby("month")["pnl"].sum()

    all_months = sorted(set(monthly_long.index) | set(monthly_short.index))
    for month in all_months:
        l_profit = monthly_long.get(month, 0)
        s_profit = monthly_short.get(month, 0)
        combined = l_profit + s_profit
        print(f"{str(month):>8s} | {l_profit:12.1f} | {s_profit:12.1f} | {combined:10.1f}")

    total_long = sum(monthly_long)
    total_short = sum(monthly_short)
    print("-" * 55)
    print(f"{'TOTAL':>8s} | {total_long:12.1f} | {total_short:12.1f} | {total_long+total_short:10.1f}")

    # Session-Analyse
    print(f"\n{'Session':12s} | {'LONG Win%':>10s} | {'SHORT Win%':>10s}")
    print("-" * 40)

    for session_name, hour_range in [("London", (8, 13)), ("LON/NY", (13, 16)), ("NY", (16, 20))]:
        l_mask = (trades_long["entry_time"].dt.hour >= hour_range[0]) & (trades_long["entry_time"].dt.hour < hour_range[1])
        s_mask = (trades_short["entry_time"].dt.hour >= hour_range[0]) & (trades_short["entry_time"].dt.hour < hour_range[1])

        l_wr = (trades_long.loc[l_mask, "pnl"] > 0).mean() if l_mask.sum() > 0 else 0
        s_wr = (trades_short.loc[s_mask, "pnl"] > 0).mean() if s_mask.sum() > 0 else 0

        print(f"{session_name:12s} | {l_wr*100:9.1f}% | {s_wr*100:9.1f}%")

    # Zusammenfassung
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")

    long_pf = stats_long["profit_factor"]
    short_pf = stats_short["profit_factor"]

    if short_pf > 8:
        short_bewertung = "SEHR GUT"
    elif short_pf > 5:
        short_bewertung = "GUT"
    elif short_pf > 3:
        short_bewertung = "OK"
    else:
        short_bewertung = "SCHWACH"

    print(f"  LONG  Modell: PF={long_pf:.2f} | Win={stats_long['win_rate']*100:.1f}% | {stats_long['n_trades']} Trades")
    print(f"  SHORT Modell: PF={short_pf:.2f} | Win={stats_short['win_rate']*100:.1f}% | {stats_short['n_trades']} Trades")
    print(f"  SHORT Bewertung: {short_bewertung}")

    if short_pf > long_pf:
        print(f"  -> SHORT ist BESSER als LONG!")
    elif short_pf > long_pf * 0.8:
        print(f"  -> SHORT ist VERGLEICHBAR mit LONG")
    else:
        print(f"  -> SCHWACH als LONG - evt. nur LONG handeln")

    print(f"\n  Kombiniert: {stats_long['n_trades'] + stats_short['n_trades']} Trades | "
          f"Profit: {stats_long['total_profit'] + stats_short['total_profit']:.0f} Punkte")


if __name__ == "__main__":
    compare_models()
