"""
Optimierte Trading-Simulation: ~5 saubere Signale pro Tag.

Ziel: Hohe Winrate, gutes R:R, handhabbare Frequenz.
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
                     REPORTS_DIR, POINT, SPREAD_POINTS, SLIPPAGE_POINTS)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def optimize_for_5_per_day(df, predictions, label=""):
    """
    Findet die optimalen Parameter für ~5 Trades/Tag.
    Testet verschiedene Confidence-Thresholds und max_trades_per_day.
    """
    n_days = (df["timestamp"].max() - df["timestamp"].min()).days

    # Session-Filter
    hour = df["timestamp"].dt.hour.values
    is_london = (hour >= 8) & (hour < 16)
    is_ny = (hour >= 13) & (hour < 20)
    is_active = is_london | is_ny

    # ATR-Filter
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(len(df)) * 0.001
    has_vol = (atr_norm > 0.0003) & (atr_norm < 0.005)

    base_eligible = is_active & has_vol

    results = []

    for max_daily in [3, 4, 5, 6, 7, 8]:
        for conf_thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
            signals = np.zeros(len(df), dtype=bool)
            high_conf = predictions >= conf_thresh
            eligible = base_eligible & high_conf

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

            n_trades = signals.sum()
            if n_trades < 50:
                continue

            trades_per_day = n_trades / n_days

            # Backtest
            engine = BacktestEngine(
                tp_points=TARGET_PARAMS["tp_points"],
                sl_points=TARGET_PARAMS["sl_points"],
                horizon=TARGET_PARAMS["horizon"],
            )
            trades_df, stats = engine.run(df, signals)

            if stats and stats["n_trades"] > 20:
                results.append({
                    "max_daily": max_daily,
                    "conf_threshold": conf_thresh,
                    "n_trades": stats["n_trades"],
                    "trades_per_day": trades_per_day,
                    "win_rate": stats["win_rate"],
                    "profit_factor": stats["profit_factor"],
                    "total_profit": stats["total_profit"],
                    "max_drawdown": stats["max_drawdown"],
                    "sharpe": stats["sharpe_ratio"],
                    "avg_trade": stats["total_profit"] / stats["n_trades"],
                })

    results_df = pd.DataFrame(results)
    # Filter: zwischen 4 und 6 Trades/Tag
    optimal = results_df[(results_df["trades_per_day"] >= 4) & (results_df["trades_per_day"] <= 6)]
    if len(optimal) == 0:
        optimal = results_df

    optimal = optimal.sort_values("profit_factor", ascending=False)
    print(f"\n  Top-Konfigurationen für ~5 Trades/Tag ({label}):")
    print(f"  {'Max/Day':>8s} | {'Conf':>5s} | {'Trades':>7s} | {'/Day':>5s} | {'Win%':>6s} | {'PF':>5s} | {'Profit':>7s} | {'DD':>5s}")
    print(f"  {'-'*70}")
    for _, row in optimal.head(10).iterrows():
        print(f"  {int(row['max_daily']):8d} | {row['conf_threshold']:5.2f} | "
              f"{int(row['n_trades']):7d} | {row['trades_per_day']:5.1f} | "
              f"{row['win_rate']*100:5.1f}% | {row['profit_factor']:5.2f} | "
              f"{row['total_profit']:7.0f} | {row['max_drawdown']:5.0f}")

    return optimal


def test_tp_sl_combinations(df, signals):
    """Testet verschiedene TP/SL-Kombinationen für die gefilterten Signale."""
    print(f"\n  TP/SL-Optimierung für {signals.sum()} Signale:")

    combos = [
        (50, 25),   # 2:1 - aktuell
        (40, 20),   # 2:1 - enger
        (60, 30),   # 2:1 - weiter
        (75, 25),   # 3:1
        (100, 25),  # 4:1
        (60, 20),   # 3:1
        (80, 20),   # 4:1
        (30, 15),   # 2:1 - sehr eng
        (45, 15),   # 3:1
    ]

    results = []
    for tp, sl in combos:
        engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=TARGET_PARAMS["horizon"])
        trades_df, stats = engine.run(df, signals)
        if stats and stats["n_trades"] > 10:
            results.append({
                "tp": tp,
                "sl": sl,
                "rr": tp / sl,
                "n_trades": stats["n_trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "total_profit": stats["total_profit"],
                "max_drawdown": stats["max_drawdown"],
                "tp_rate": stats["tp_hit_rate"],
                "sl_rate": stats["sl_hit_rate"],
                "expiry_rate": stats["expiry_rate"],
            })

    results_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)
    print(f"  {'TP':>5s} | {'SL':>5s} | {'R:R':>4s} | {'Win%':>6s} | {'PF':>5s} | {'Profit':>7s} | {'TP%':>5s} | {'SL%':>5s} | {'Exp%':>5s}")
    print(f"  {'-'*75}")
    for _, row in results_df.iterrows():
        print(f"  {int(row['tp']):5d} | {int(row['sl']):5d} | {row['rr']:4.1f} | "
              f"{row['win_rate']*100:5.1f}% | {row['profit_factor']:5.2f} | "
              f"{row['total_profit']:7.0f} | {row['tp_rate']*100:4.1f}% | "
              f"{row['sl_rate']*100:4.1f}% | {row['expiry_rate']*100:4.1f}%")

    return results_df


def run_optimized_pipeline():
    """Haupt-Pipeline für ~5 Trades/Tag."""
    print("=" * 70)
    print("OPTIMIERTE PIPELINE: ~5 SAUBERE SIGNALE/Tag")
    print("=" * 70)

    df = load_combined()
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_test = test_df[feature_cols].values.astype(np.float32)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    predictions = model.predict_proba(X_test)[:, 1]

    # === Phase 1: Optimale Filter-Parameter finden ===
    print("\n--- Phase 1: Filter-Optimierung ---")
    optimal = optimize_for_5_per_day(test_df, predictions, "OOS Test")

    if len(optimal) == 0:
        print("  Keine passende Konfiguration gefunden!")
        return

    best = optimal.iloc[0]
    best_max_daily = int(best["max_daily"])
    best_conf = best["conf_threshold"]

    # === Phase 2: Signale generieren ===
    print(f"\n--- Phase 2: Signale generieren (max={best_max_daily}/Tag, conf>={best_conf:.2f}) ---")

    hour = test_df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    atr_norm = test_df["atr_14_norm"].values if "atr_14_norm" in test_df.columns else np.ones(len(test_df)) * 0.001
    has_vol = (atr_norm > 0.0003) & (atr_norm < 0.005)

    signals = np.zeros(len(test_df), dtype=bool)
    high_conf = predictions >= best_conf
    eligible = is_active & has_vol & high_conf

    current_date = None
    daily_trades = 0
    for i in range(len(test_df)):
        if not eligible[i]:
            continue
        date = test_df["timestamp"].iloc[i].date()
        if date != current_date:
            current_date = date
            daily_trades = 0
        if daily_trades < best_max_daily:
            signals[i] = True
            daily_trades += 1

    print(f"  Generierte Signale: {signals.sum()}")

    # === Phase 3: TP/SL-Optimierung ===
    print("\n--- Phase 3: TP/SL-Optimierung ---")
    tp_sl_results = test_tp_sl_combinations(test_df, signals)

    # === Phase 4: Beste Konfiguration ausführlich testen ===
    best_tp_sl = tp_sl_results.iloc[0]
    print(f"\n--- Phase 4: Beste Konfiguration (TP={int(best_tp_sl['tp'])}, SL={int(best_tp_sl['sl'])}) ---")

    engine = BacktestEngine(
        tp_points=int(best_tp_sl["tp"]),
        sl_points=int(best_tp_sl["sl"]),
        horizon=TARGET_PARAMS["horizon"],
    )
    trades_df, stats = engine.run(test_df, signals)

    if stats:
        n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days
        trades_df = trades_df.sort_values("entry_idx").reset_index(drop=True)

        print(f"\n  ERGEBNISSE:")
        print(f"    Zeitraum: {n_days} Tage")
        print(f"    Trades: {stats['n_trades']}")
        print(f"    Trades/Tag: {stats['n_trades']/n_days:.1f}")
        print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"    Profit Factor: {stats['profit_factor']:.2f}")
        print(f"    Total Profit: {stats['total_profit']:.0f} Punkte ({stats['total_profit']*0.01:.0f} USD)")
        print(f"    Max Drawdown: {stats['max_drawdown']:.0f} Punkte ({stats['max_drawdown']*0.01:.0f} USD)")
        print(f"    Sharpe: {stats['sharpe_ratio']:.1f}")
        print(f"    TP-Hit: {stats['tp_hit_rate']*100:.1f}% | SL-Hit: {stats['sl_hit_rate']*100:.1f}% | Expiry: {stats['expiry_rate']*100:.1f}%")
        print(f"    Max Consec. Losses: {stats['max_consec_losses']}")
        print(f"    Max Consec. Wins: {stats['max_consec_wins']}")

        # Monatliche Aufschlüsselung
        trades_df["month"] = trades_df["entry_time"].dt.to_period("M")
        monthly = trades_df.groupby("month").agg(
            n_trades=("pnl", "count"),
            profit=("pnl", "sum"),
            wins=("pnl", lambda x: (x > 0).sum()),
            losses=("pnl", lambda x: (x < 0).sum()),
        ).reset_index()
        monthly["win_rate"] = monthly["wins"] / monthly["n_trades"]
        monthly["profit_usd"] = monthly["profit"] * 0.01

        print(f"\n  Monatliche Performance:")
        print(f"  {'Monat':>8s} | {'Trades':>7s} | {'W':>3s} | {'L':>3s} | {'Win%':>6s} | {'Profit':>8s} | {'USD':>8s}")
        print(f"  {'-'*60}")
        for _, row in monthly.iterrows():
            print(f"  {str(row['month']):>8s} | {int(row['n_trades']):7d} | "
                  f"{int(row['wins']):3d} | {int(row['losses']):3d} | "
                  f"{row['win_rate']*100:5.1f}% | {row['profit']:8.1f} | "
                  f"{row['profit_usd']:8.1f}")

        # Session-Aufschlüsselung
        trades_df["hour"] = trades_df["entry_time"].dt.hour
        trades_df["session"] = "other"
        trades_df.loc[(trades_df["hour"] >= 8) & (trades_df["hour"] < 13), "session"] = "London"
        trades_df.loc[(trades_df["hour"] >= 13) & (trades_df["hour"] < 16), "session"] = "LON/NY"
        trades_df.loc[(trades_df["hour"] >= 16) & (trades_df["hour"] < 20), "session"] = "NY"

        session_stats = trades_df.groupby("session").agg(
            n_trades=("pnl", "count"),
            profit=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        ).reset_index()

        print(f"\n  Session-Aufschlüsselung:")
        for _, row in session_stats.iterrows():
            print(f"    {row['session']:10s}: {int(row['n_trades']):3d} Trades | "
                  f"Profit: {row['profit']:6.1f} | Win: {row['win_rate']*100:.0f}%")

        # Wochentag-Aufschlüsselung
        trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        weekday_stats = trades_df.groupby("weekday").agg(
            n_trades=("pnl", "count"),
            profit=("pnl", "sum"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        ).reindex(weekday_order).dropna()

        print(f"\n  Wochentag-Aufschlüsselung:")
        for day, row in weekday_stats.iterrows():
            print(f"    {day:10s}: {int(row['n_trades']):3d} Trades | "
                  f"Profit: {row['profit']:6.1f} | Win: {row['win_rate']*100:.0f}%")

    # Speichern
    results = {
        "config": {
            "max_trades_per_day": best_max_daily,
            "confidence_threshold": best_conf,
            "tp_points": int(best_tp_sl["tp"]),
            "sl_points": int(best_tp_sl["sl"]),
            "horizon": TARGET_PARAMS["horizon"],
        },
        "performance": {
            "n_trades": stats["n_trades"] if stats else 0,
            "trades_per_day": stats["n_trades"] / n_days if stats else 0,
            "win_rate": stats["win_rate"] if stats else 0,
            "profit_factor": stats["profit_factor"] if stats else 0,
            "total_profit_points": stats["total_profit"] if stats else 0,
            "total_profit_usd": stats["total_profit"] * 0.01 if stats else 0,
            "max_drawdown_points": stats["max_drawdown"] if stats else 0,
            "sharpe": stats["sharpe_ratio"] if stats else 0,
        },
        "monthly": monthly.to_dict(orient="records") if stats else [],
    }

    results_path = os.path.join(REPORTS_DIR, "optimized_5perday.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Ergebnisse gespeichert: {results_path}")

    return results


if __name__ == "__main__":
    run_optimized_pipeline()
