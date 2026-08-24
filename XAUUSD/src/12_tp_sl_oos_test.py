"""
OOS Test: TP/SL-Größen optimieren und validiertesten.

Testet verschiedene TP/SL-Kombinationen:
- Eng (20/10, 30/15)
- Mittel (45/15, 50/25)  
- Weit (60/20, 75/25, 100/25)
- Verschiedene R:R-Verhältnisse (2:1, 3:1, 4:1)

Ziel: Finde optimale TP/SL für Live-Trading.
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

    return signals


def test_tp_sl_oos():
    print("=" * 70)
    print("OOS TEST: TP/SL-GROESSEN")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    # Signale generieren (unabhängig von TP/SL)
    signals = generate_signals(test_df, model, feature_cols)
    n_signals = signals.sum()
    print(f"  Generierte Signale: {n_signals}")

    # TP/SL-Kombinationen testen (realistisch für Live-Trading)
    # Min SL=15 (Spread 3 + Slippage 2 + Puffer 10)
    # Min TP=30 (realistischer Gewinn nach Kosten)
    combinations = [
        # Konservativ (enger, sicherer)
        (30, 15), (35, 15), (40, 15), (40, 20),
        # Standard
        (45, 15), (50, 20), (50, 25), (60, 20),
        # Weit (mehr Raum für Slippage)
        (60, 30), (75, 25), (75, 30), (80, 30),
        # Sehr weit (Swing)
        (100, 30), (100, 40), (100, 50),
    ]

    results = []
    for tp, sl in combinations:
        engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=5)
        trades_df, stats = engine.run(test_df, signals)

        if stats and stats["n_trades"] > 10:
            results.append({
                "tp": tp,
                "sl": sl,
                "rr": f"{tp/sl:.1f}:1",
                "n_trades": stats["n_trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "total_profit": stats["total_profit"],
                "max_drawdown": stats["max_drawdown"],
                "sharpe": stats["sharpe_ratio"],
                "tp_rate": stats["tp_hit_rate"],
                "sl_rate": stats["sl_hit_rate"],
                "expiry_rate": stats["expiry_rate"],
                "avg_trade": stats["total_profit"] / stats["n_trades"],
            })

    results_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)

    print(f"\n  Ergebnisse (sortiert nach Profit Factor):")
    print(f"  {'TP':>5s} | {'SL':>5s} | {'R:R':>5s} | {'Trades':>7s} | {'Win%':>6s} | {'PF':>5s} | {'Profit':>7s} | {'DD':>5s} | {'TP%':>5s} | {'SL%':>5s} | {'Exp%':>5s}")
    print(f"  {'-'*95}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['tp']):5d} | {int(row['sl']):5d} | {row['rr']:>5s} | "
              f"{int(row['n_trades']):7d} | {row['win_rate']*100:5.1f}% | "
              f"{row['profit_factor']:5.2f} | {row['total_profit']:7.0f} | "
              f"{row['max_drawdown']:5.0f} | {row['tp_rate']*100:4.1f}% | "
              f"{row['sl_rate']*100:4.1f}% | {row['expiry_rate']*100:4.1f}%")

    # Beste nach verschiedenen Kriterien
    print(f"\n  BESTE KONFIGURATIONEN:")
    best_pf = results_df.iloc[0]
    print(f"    Best PF:   TP={int(best_pf['tp'])}, SL={int(best_pf['sl'])} (PF={best_pf['profit_factor']:.2f}, Win={best_pf['win_rate']*100:.1f}%)")

    best_profit = results_df.sort_values("total_profit", ascending=False).iloc[0]
    print(f"    Best Profit: TP={int(best_profit['tp'])}, SL={int(best_profit['sl'])} (Profit={best_profit['total_profit']:.0f})")

    best_dd = results_df.sort_values("max_drawdown", ascending=False).iloc[0]
    print(f"    Best DD:   TP={int(best_dd['tp'])}, SL={int(best_dd['sl'])} (DD={best_dd['max_drawdown']:.0f})")

    # Empfehlung
    # Trade-off: Hoher PF + akzeptable Winrate + geringes DD
    results_df["score"] = (
        results_df["profit_factor"] * 0.4 +
        results_df["win_rate"] * 10 * 0.3 -
        results_df["max_drawdown"].abs() * 0.3
    )
    recommended = results_df.sort_values("score", ascending=False).iloc[0]
    print(f"\n  EMPFEHLUNG:")
    print(f"    TP={int(recommended['tp'])}, SL={int(recommended['sl'])} (R:R={recommended['rr']})")
    print(f"    Win={recommended['win_rate']*100:.1f}% | PF={recommended['profit_factor']:.2f} | Profit={recommended['total_profit']:.0f} | DD={recommended['max_drawdown']:.0f}")

    # Speichern
    results_path = os.path.join(REPORTS_DIR, "tp_sl_oos_test.json")
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\n  Gespeichert: {results_path}")

    return results_df


def test_tp_sl_by_session():
    """Testet TP/SL nach Session."""
    print("\n" + "=" * 70)
    print("OOS TEST: TP/SL NACH SESSION")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    signals = generate_signals(test_df, model, feature_cols)

    sessions = {
        "London": (8, 13),
        "LON/NY": (13, 16),
        "NY": (16, 20),
    }

    best_per_session = {}
    for session_name, (start_h, end_h) in sessions.items():
        session_mask = (test_df["timestamp"].dt.hour >= start_h) & (test_df["timestamp"].dt.hour < end_h)
        session_signals = signals & session_mask

        if session_signals.sum() < 20:
            continue

        best_pf = 0
        best_config = None

        for tp, sl in [(30, 15), (45, 15), (50, 25), (60, 20), (75, 25)]:
            engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=5)
            _, stats = engine.run(test_df, session_signals)
            if stats and stats["profit_factor"] > best_pf:
                best_pf = stats["profit_factor"]
                best_config = (tp, sl, stats)

        if best_config:
            tp, sl, stats = best_config
            best_per_session[session_name] = {
                "tp": tp, "sl": sl,
                "win_rate": stats["win_rate"],
                "pf": stats["profit_factor"],
                "profit": stats["total_profit"],
                "n_trades": stats["n_trades"],
            }
            print(f"  {session_name:10s}: TP={tp:3d}, SL={sl:2d} | "
                  f"Win={stats['win_rate']*100:.1f}% | PF={stats['profit_factor']:.2f} | "
                  f"Profit={stats['total_profit']:.0f} | {stats['n_trades']} Trades")

    return best_per_session


def main():
    results = test_tp_sl_oos()
    session_results = test_tp_sl_by_session()

    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    print("  TP/SL-Test abgeschlossen. Ergebnisse in reports/tp_sl_oos_test.json")


if __name__ == "__main__":
    main()
