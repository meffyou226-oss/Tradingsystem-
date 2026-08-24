"""
Target-Definition für XAUUSD M1-Handel.

Erstellt TP/SL-basierte Targets:
- Für jede Kerze: Entry = close[i], schaue X Minuten vorwärts
- TP/SL-Level basierend auf Punkten (1 Punkt = 0.01 USD)
- Target = 1 wenn TP vor SL erreicht wird (gewinnbringt)
- Target = 0 wenn SL vor TP erreicht wird (verlierend)
- Target = 0 wenn keiner erreicht wird (konservativ)

Testet mehrere Horizonte (5, 10, 15, 20 Min) und TP/SL-Verhältnisse.
Keine Lookahead-Bias: Features nutzen nur vergangene Daten (shift(1)).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import product

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_features.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "targets")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "target_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# XAUUSD Punktgröße: 1 Punkt = 0.01 USD
POINT = 0.01

# Parameter-Kombinationen für Target-Tests
HORIZONS = [5, 10, 15, 20]  # Minuten
SL_POINTS = [25, 50, 75]    # Stop-Loss in Punkten
RR_RATIOS = [1.0, 2.0, 3.0] # Risk/Reward Verhältnisse (TP = SL * Ratio)


def load_features(path=FEATURES_PATH):
    print("Lade Feature-Daten...")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
    return df


def compute_target_vectorized(close_arr, high_arr, low_arr, horizon, tp_points, sl_points, point=POINT):
    """
    Vektorisierte Target-Berechnung.
    
    Für jede Zeile i:
    - Entry = close[i]
    - TP = entry + tp_points * point
    - SL = entry - sl_points * point
    - Schaue horizon Minuten in die Zukunft
    - Target = 1 wenn TP vor SL erreicht (oder TP erreicht und SL nicht)
    - Target = 0 sonst (SL erreicht oder nichts erreicht)
    
    Nutzt numpy für Geschwindigkeit.
    """
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    
    # Entry = close[i], look forward horizon bars
    entry = close_arr
    tp_level = entry + tp_points * point
    sl_level = entry - sl_points * point
    
    # For each position i, check future horizon bars [i+1, i+horizon]
    # Use rolling approach with numpy
    horizon_actual = min(horizon, n - 1)
    
    for offset in range(1, horizon_actual + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)
        
        # For rows where target hasn't been determined yet
        undetermined = targets[current_idx] == 0
        
        # Check TP hit: future high >= tp_level
        tp_hits = high_arr[future_idx] >= tp_level[current_idx]
        # Check SL hit: future low <= sl_level
        sl_hits = low_arr[future_idx] <= sl_level[current_idx]
        
        # TP hit and not yet determined -> set to 1
        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)  # 2 = SL already hit
        targets[current_idx[new_tp]] = 1
        
        # SL hit and not yet determined -> mark as 2 (will be mapped to 0 later)
        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2  # SL hit
    
    # Map: 1 -> 1 (TP win), 2 -> 0 (SL loss), 0 -> 0 (no hit)
    targets[targets == 2] = 0
    
    return targets


def analyze_target_distribution(targets, name):
    n_total = len(targets)
    n_positive = (targets == 1).sum()
    n_negative = (targets == 0).sum()
    
    stats = {
        "name": name,
        "total": n_total,
        "positive": n_positive,
        "negative": n_negative,
        "pos_ratio": n_positive / n_total,
        "neg_ratio": n_negative / n_total,
    }
    
    print(f"  {name:45s} | P={n_positive/n_total*100:5.1f}% N={n_negative/n_total*100:5.1f}% | Balance: {abs(n_positive-n_negative)/n_total*100:.1f}%")
    
    return stats


def main():
    df = load_features()
    
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    
    print(f"\nBerechne Targets für {len(HORIZONS)} Horizonte x {len(SL_POINTS)} SL-Werte x {len(RR_RATIOS)} R:R-Verhältnisse = {len(HORIZONS)*len(SL_POINTS)*len(RR_RATIOS)} Kombinationen")
    print("-" * 80)
    
    all_stats = []
    best_combo = None
    best_balance = 1.0  # closest to 50/50
    
    for horizon, sl_points, rr in product(HORIZONS, SL_POINTS, RR_RATIOS):
        tp_points = int(sl_points * rr)
        name = f"h{horizon}_sl{sl_points}_rr{rr}"
        
        start = time.time()
        targets = compute_target_vectorized(close_arr, high_arr, low_arr, horizon, tp_points, sl_points)
        elapsed = time.time() - start
        
        stats = analyze_target_distribution(targets, name)
        stats["horizon"] = horizon
        stats["tp_points"] = tp_points
        stats["sl_points"] = sl_points
        stats["rr"] = rr
        stats["time_sec"] = elapsed
        all_stats.append(stats)
        
        # Track best balance (closest to 50/50)
        balance = abs(stats["pos_ratio"] - 0.5)
        if balance < best_balance:
            best_balance = balance
            best_combo = (horizon, tp_points, sl_points, rr, name)
        
        # Save target column for this combo
        df[f"target_{name}"] = targets
    
    print("-" * 80)
    print(f"\nBester balancierter Target: {best_combo[4]} (h={best_combo[0]}, tp={best_combo[1]}, sl={best_combo[2]}, rr={best_combo[3]})")
    print(f"  Balance: {100*(1-best_balance*2):.1f}% (100% = perfekt balanciert)")
    
    # Save targets DataFrame (only timestamp + target columns)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    targets_df = df[["timestamp"] + target_cols]
    
    targets_path = os.path.join(OUTPUT_DIR, "xauusd_m1_targets.csv")
    targets_df.to_csv(targets_path, index=False)
    print(f"\nTargets gespeichert: {targets_path} ({os.path.getsize(targets_path)/1024/1024:.1f} MB)")
    
    # Save statistics report
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(REPORT_DIR, "target_analysis.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistik gespeichert: {stats_path}")
    
    # Sort by balance and show top 10
    stats_df_sorted = stats_df.sort_values("pos_ratio")
    print("\nTop 10 best balancierte Targets (nah an 50/50):")
    print(stats_df_sorted[["name", "pos_ratio", "horizon", "tp_points", "sl_points", "rr"]].head(10).to_string(index=False))
    
    # Show recommended target parameters
    best_name = best_combo[4]
    best_stats = stats_df[stats_df["name"] == best_name].iloc[0]
    print(f"\nEmpfohlene Target-Parameter (für Hauptmodell):")
    print(f"  horizon: {best_stats['horizon']} Minuten")
    print(f"  TP: {int(best_stats['tp_points'])} Punkte ({best_stats['tp_points']*POINT:.2f} USD)")
    print(f"  SL: {int(best_stats['sl_points'])} Punkte ({best_stats['sl_points']*POINT:.2f} USD)")
    print(f"  R:R: {best_stats['rr']}:1")
    print(f"  Label-Balance: {best_stats['pos_ratio']*100:.1f}% positiv / {best_stats['neg_ratio']*100:.1f}% negativ")
    
    # Save a config file with best parameters for use in other modules
    config_path = os.path.join(OUTPUT_DIR, "target_config.csv")
    best_stats.to_frame().T.to_csv(config_path, index=False)
    
    return df


if __name__ == "__main__":
    main()
