"""
Realistische Trading-Simulation mit praktischen Filtern.

Filter:
1. Nur London + NY Session (höchste Liquidität)
2. Max 3 Trades pro Tag
3. Min. ATR-Schwelle (keine Low-Vol-Perioden)
4. Keine Trades in den ersten 30 Min nach Session-Start
5. Keine Trades wenn Spread > 5 Punkte (simuliert)
6. Confidence >= 0.75 (hohe Qualität)

Vergleich: Ungefiltert vs. Realistisch gefiltert
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TARGET_PARAMS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                     REPORTS_DIR, POINT, SPREAD_POINTS, SLIPPAGE_POINTS)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def apply_realistic_filters(df, predictions):
    """
    Wendet realistische Trading-Filter an.

    Returns:
        filtered_signals: Boolean-Array (True = Trade)
        filter_stats: Dict mit Filter-Statistiken
    """
    n = len(df)
    signals = np.zeros(n, dtype=bool)

    # Filter 1: Nur London (8-16 UTC) + NY (13-20 UTC) Session
    hour = df["timestamp"].dt.hour.values
    is_london = (hour >= 8) & (hour < 16)
    is_ny = (hour >= 13) & (hour < 20)
    is_active_session = is_london | is_ny

    # Filter 2: ATR-Schwelle (min. Volatilität für sinnvolle Bewegung)
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(n) * 0.001
    min_atr_threshold = 0.0003  # Mindestens 0.03% ATR (filtert dead markets)
    has_volatility = atr_norm > min_atr_threshold

    # Filter 3: Confidence >= 0.75
    high_confidence = predictions >= 0.75

    # Filter 4: Keine ersten 30 Min nach Session-Start
    minute = df["timestamp"].dt.minute.values
    session_start_buffer = ~((hour == 8) & (minute < 30)) & ~((hour == 13) & (minute < 30))

    # Filter 5: Keine Trades wenn Volatilität extrem hoch (News-Spikes)
    max_atr_threshold = 0.005  # Max 0.5% ATR
    not_extreme_vol = atr_norm < max_atr_threshold

    # Kombinierte Filter
    eligible = (
        is_active_session &
        has_volatility &
        session_start_buffer &
        not_extreme_vol &
        high_confidence
    )

    # Filter 6: Max 3 Trades pro Tag (nur die besten 3 nach Confidence)
    trade_count = 0
    current_date = None
    daily_trades = 0

    for i in range(n):
        if not eligible[i]:
            continue

        date = df["timestamp"].iloc[i].date()
        if date != current_date:
            current_date = date
            daily_trades = 0

        if daily_trades < 3:
            signals[i] = True
            daily_trades += 1
            trade_count += 1

    filter_stats = {
        "total_candles": n,
        "active_session": int(is_active_session.sum()),
        "high_confidence": int(high_confidence.sum()),
        "eligible_after_filters": int(eligible.sum()),
        "final_trades": trade_count,
        "filter_rate": f"{(1 - trade_count / max(high_confidence.sum(), 1)) * 100:.1f}%",
    }

    return signals, filter_stats


def run_comparison():
    """Vergleicht ungefilterte vs. realistisch gefilterte Trades."""
    print("=" * 70)
    print("REALISTISCHE TRADING-SIMULATION")
    print("=" * 70)

    df = load_combined()
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Load model
    import pickle
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    predictions = model.predict_proba(X_test)[:, 1]

    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )

    # === Szenario 1: Ungefiltert (Threshold 0.50) ===
    print("\n--- Szenario 1: Ungefiltert (Threshold=0.50) ---")
    signals_unfiltered = predictions >= 0.50
    trades_df1, stats1 = engine.run(test_df, signals_unfiltered)
    n_days1 = (test_df["timestamp"].max() - test_df["timestamp"].min()).days
    print(f"  Trades: {stats1['n_trades']}")
    print(f"  Trades/Tag: {stats1['n_trades'] / n_days1:.0f}")
    print(f"  Win Rate: {stats1['win_rate']*100:.1f}%")
    print(f"  PF: {stats1['profit_factor']:.2f}")
    print(f"  Profit: {stats1['total_profit']:.0f}")
    print(f"  Max DD: {stats1['max_drawdown']:.0f}")

    # === Szenario 2: Nur Confidence-Filter (0.70) ===
    print("\n--- Szenario 2: Confidence >= 0.70 ---")
    signals_conf = predictions >= 0.70
    trades_df2, stats2 = engine.run(test_df, signals_conf)
    print(f"  Trades: {stats2['n_trades']}")
    print(f"  Trades/Tag: {stats2['n_trades'] / n_days1:.0f}")
    print(f"  Win Rate: {stats2['win_rate']*100:.1f}%")
    print(f"  PF: {stats2['profit_factor']:.2f}")
    print(f"  Profit: {stats2['total_profit']:.0f}")
    print(f"  Max DD: {stats2['max_drawdown']:.0f}")

    # === Szenario 3: Realistisch gefiltert ===
    print("\n--- Szenario 3: Realistisch gefiltert ---")
    print("  Filter: London+NY Session, max 3/Tag, ATR>0.03%, Confidence>=0.75")
    signals_filtered, filter_stats = apply_realistic_filters(test_df, predictions)
    trades_df3, stats3 = engine.run(test_df, signals_filtered)

    print(f"\n  Filter-Statistiken:")
    for k, v in filter_stats.items():
        print(f"    {k}: {v}")

    if stats3:
        n_days3 = n_days1
        print(f"\n  Trading-Ergebnisse:")
        print(f"    Trades: {stats3['n_trades']}")
        print(f"    Trades/Tag: {stats3['n_trades'] / n_days3:.1f}")
        print(f"    Win Rate: {stats3['win_rate']*100:.1f}%")
        print(f"    PF: {stats3['profit_factor']:.2f}")
        print(f"    Profit: {stats3['total_profit']:.0f}")
        print(f"    Max DD: {stats3['max_drawdown']:.0f}")
        print(f"    TP: {stats3['tp_hit_rate']*100:.1f}% SL: {stats3['sl_hit_rate']*100:.1f}%")

        # Monthly breakdown
        trades_df3 = trades_df3.sort_values("entry_idx").reset_index(drop=True)
        trades_df3["month"] = trades_df3["entry_time"].dt.to_period("M")
        monthly = trades_df3.groupby("month").agg(
            n_trades=("pnl", "count"),
            profit=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        ).reset_index()

        print(f"\n  Monatliche Performance:")
        for _, row in monthly.iterrows():
            print(f"    {row['month']}: {row['n_trades']:3d} Trades | "
                  f"Profit: {row['profit']:7.1f} | "
                  f"Win: {row['win_rate']*100:.0f}%")

    # === Szenario 4: Sehr konservativ (max 1/Tag, Confidence>=0.80) ===
    print("\n--- Szenario 4: Sehr konservativ (max 1/Tag, Confidence>=0.80) ---")
    signals_cons = np.zeros(len(test_df), dtype=bool)
    hour = test_df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    high_conf = predictions >= 0.80

    current_date = None
    daily_trades = 0
    for i in range(len(test_df)):
        if not (is_active[i] and high_conf[i]):
            continue
        date = test_df["timestamp"].iloc[i].date()
        if date != current_date:
            current_date = date
            daily_trades = 0
        if daily_trades < 1:
            signals_cons[i] = True
            daily_trades += 1

    trades_df4, stats4 = engine.run(test_df, signals_cons)
    if stats4:
        print(f"    Trades: {stats4['n_trades']}")
        print(f"    Trades/Tag: {stats4['n_trades'] / n_days1:.2f}")
        print(f"    Win Rate: {stats4['win_rate']*100:.1f}%")
        print(f"    PF: {stats4['profit_factor']:.2f}")
        print(f"    Profit: {stats4['total_profit']:.0f}")
        print(f"    Max DD: {stats4['max_drawdown']:.0f}")

    # === Zusammenfassung ===
    print("\n" + "=" * 70)
    print("VERGLEICH ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"{'Szenario':35s} | {'Trades':>7s} | {'/Tag':>5s} | {'Win%':>6s} | {'PF':>5s} | {'Profit':>8s}")
    print("-" * 75)

    scenarios = [
        ("1. Ungefiltert (0.50)", stats1),
        ("2. Confidence>=0.70", stats2),
        ("3. Realistisch (3/Tag)", stats3 if stats3 else None),
        ("4. Konservativ (1/Tag)", stats4 if stats4 else None),
    ]

    for name, stats in scenarios:
        if stats:
            print(f"{name:35s} | {stats['n_trades']:7d} | "
                  f"{stats['n_trades']/n_days1:5.1f} | "
                  f"{stats['win_rate']*100:5.1f}% | "
                  f"{stats['profit_factor']:5.2f} | "
                  f"{stats['total_profit']:8.0f}")

    # Save results
    results = {
        "n_days": n_days1,
        "scenarios": {}
    }
    for name, stats in scenarios:
        if stats:
            results["scenarios"][name] = {
                "n_trades": stats["n_trades"],
                "trades_per_day": stats["n_trades"] / n_days1,
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "total_profit": stats["total_profit"],
                "max_drawdown": stats["max_drawdown"],
            }

    results_path = os.path.join(REPORTS_DIR, "realistic_trading_simulation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nErgebnisse gespeichert: {results_path}")

    return results


if __name__ == "__main__":
    run_comparison()
