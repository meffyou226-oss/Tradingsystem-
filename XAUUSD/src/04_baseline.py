"""
<<<<<<< ours
<<<<<<< ours
Baseline-Trading-Strategien (ohne ML).

Implementiert einfache, bewährte Strategien für Vergleich mit ML-Modellen:
1. Momentum: Long wenn 5-Minuten-RoC positiv
2. EMA Crossover: Long wenn EMA(20) > EMA(50)
3. RSI Mean-Reversion: Long wenn RSI(14) < 30
4. Breakout: Long wenn Close über 5-Balken-Hoch

Alle Strategien werden mit derselben TP/SL-Struktur wie das ML-Modell getestet.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add config path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURES_PATH, TARGETS_PATH, BACKTESTS_DIR, RESULTS_DIR,
                    REPORTS_DIR, TARGET_PARAMS, SPREAD_POINTS, SLIPPAGE_POINTS)
from backtest_engine import BacktestEngine, run_backtest, compute_stats


def load_features(path=FEATURES_PATH):
    """Lade Feature-Daten."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_signals_momentum(df, period=5):
    """
    Momentum-Strategie: Long wenn RoC(period) > 0.
    Einfach und effektiv als Baseline.
    """
    roc = df["close"].pct_change(period)
    return roc > 0


def generate_signals_ema_crossover(df):
    """
    EMA Crossover: Long wenn EMA(20) > EMA(50).
    Klassische Trendfolge-Strategie.
    """
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    return ema_fast > ema_slow


def generate_signals_rsi_meanrev(df, oversold=30, overbought=70):
    """
    RSI Mean-Reversion: Long wenn RSI(14) < oversold.
    """
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Long signal
    long = rsi < oversold
    return long.fillna(False)


def generate_signals_breakout(df, period=5):
    """
    Breakout-Strategie: Long wenn Close über max(Hoch der letzten period Balken).
    """
    recent_high = df["high"].shift(1).rolling(period).max()
    return df["close"] > recent_high


def generate_signals_ema_momentum(df):
    """
    Kombinierte Strategie: EMA-Trendfilter + Momentum-Entry.
    Long wenn EMA(20) > EMA(50) UND RoC(5) > 0.
    """
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    trend_ok = ema_fast > ema_slow
    momentum_ok = df["close"].pct_change(5) > 0
    return trend_ok & momentum_ok


BASELINE_STRATEGIES = {
    "momentum": generate_signals_momentum,
    "ema_crossover": generate_signals_ema_crossover,
    "rsi_meanrev": generate_signals_rsi_meanrev,
    "breakout": generate_signals_breakout,
    "ema_momentum": generate_signals_ema_momentum,
}


def run_baseline_backtests():
    """Führt alle Baseline-Strategien aus und speichert Ergebnisse."""
    print("=" * 70)
    print("Baseline-Strategien Backtest")
    print("=" * 70)
    
    df = load_features()
    print(f"Daten geladen: {len(df)} Zeilen ({df['timestamp'].iloc[0]} bis {df['timestamp'].iloc[-1]})")
    
    tp = TARGET_PARAMS["tp_points"]
    sl = TARGET_PARAMS["sl_points"]
    horizon = TARGET_PARAMS["horizon"]
    
    engine = BacktestEngine(tp_points=tp, sl_points=sl, horizon=horizon)
    
    all_results = {}
    
    for name, signal_fn in BASELINE_STRATEGIES.items():
        print(f"\n--- Strategie: {name} ---")
        
        signals = signal_fn(df)
        n_signals = signals.sum()
        print(f"  Signale generiert: {n_signals}")
        
        if n_signals == 0:
            print("  Keine Signale!")
            continue
        
        trades_df, stats = engine.run(df, signals)
        
        if stats:
            print(f"  Trades: {stats['n_trades']}")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
            print(f"  Total Profit: {stats['total_profit']:.2f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.2f}")
            print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
            print(f"  Avg Duration: {stats['avg_trade_duration_min']:.1f} min")
            print(f"  TP hit: {stats['tp_hit_rate']*100:.1f}% | SL hit: {stats['sl_hit_rate']*100:.1f}% | Expiry: {stats['expiry_rate']*100:.1f}%")
            
            all_results[name] = {
                "stats": stats,
                "n_trades": len(trades_df),
                "n_signals": int(n_signals),
            }
            
            # Save trades
            trades_path = os.path.join(BACKTESTS_DIR, f"baseline_{name}_trades.csv")
            trades_df.to_csv(trades_path, index=False)
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame([
            {
                "strategy": k,
                "n_signals": v["n_signals"],
                "n_trades": v["n_trades"],
                "win_rate": v["stats"]["win_rate"],
                "profit_factor": v["stats"]["profit_factor"],
                "total_profit": v["stats"]["total_profit"],
                "max_drawdown": v["stats"]["max_drawdown"],
                "sharpe": v["stats"]["sharpe_ratio"],
                "expectancy": v["stats"]["expectancy"],
            }
            for k, v in all_results.items()
        ])
        summary_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}")
        print("BASELINE-ZUSAMMENFASSUNG")
        print(f"{'=' * 70}")
        print(results_df.to_string(index=False))
        print(f"\nErgebnisse gespeichert: {summary_path}")
        
        # Save JSON summary
        json_path = os.path.join(REPORTS_DIR, "baseline_summary.json")
        with open(json_path, "w") as f:
            json.dump({k: {kk: str(vv) for kk, vv in v["stats"].items()} | 
                       {"n_trades": v["n_trades"], "n_signals": v["n_signals"]}
                       for k, v in all_results.items()}, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    results = run_baseline_backtests()
=======
=======
>>>>>>> theirs
Baseline-Strategien für XAUUSD M1.

Einfache regelbasierte Strategien als Benchmark:
1. Random Baseline
2. Buy & Hold
3. EMA Crossover
4. RSI Mean Reversion
5. Bollinger Bands Mean Reversion
6. Session-based Momentum

Jede Strategie generiert Signale (-1, 0, +1) und wird
mit der gleichen TP/SL-Metrik evaluiert wie das ML-Modell.
"""

import os
import numpy as np
import pandas as pd
from itertools import product

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_features.csv")
TARGETS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "targets", "xauusd_m1_targets.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

POINT = 0.01
HORIZONS = [5, 10, 15, 20]
SL_POINTS = [25, 50, 75]
RR_RATIOS = [1.0, 2.0, 3.0]


def load_data():
    print("Lade Daten...")
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    targets = pd.read_csv(TARGETS_PATH)
    targets["timestamp"] = pd.to_datetime(targets["timestamp"])
    df = df.merge(targets, on="timestamp", how="inner")
    print(f"  Geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
    return df


def compute_tp_sl_outcome(close_arr, high_arr, low_arr, signals, horizon, tp_points, sl_points):
    """
    Berechnet TP/SL Outcomes für gegebene Signale.

    Returns:
        outcomes: Array mit 1 (TP hit), 0 (SL hit / no hit), NaN (no signal)
        trade_info: Dict mit Statistiken
    """
    n = len(close_arr)
    outcomes = np.full(n, np.nan)
    entry = close_arr
    tp_level_long = entry + tp_points * POINT
    sl_level_long = entry - tp_points * POINT
    tp_level_short = entry - tp_points * POINT
    sl_level_short = entry + tp_points * POINT

    long_mask = signals == 1
    short_mask = signals == -1

    # Vectorized: für alle Long-Signals prüfen
    long_idx = np.where(long_mask)[0]
    short_idx = np.where(short_mask)[0]

    for idx_arr, is_long in [(long_idx, True), (short_idx, False)]:
        for i in idx_arr:
            if i + 1 >= n:
                continue
            end_idx = min(i + horizon + 1, n)
            future_high = high_arr[i+1:end_idx]
            future_low = low_arr[i+1:end_idx]

            if is_long:
                tp_hit = np.where(future_high >= tp_level_long[i])[0]
                sl_hit = np.where(future_low <= sl_level_long[i])[0]
            else:
                tp_hit = np.where(future_low <= tp_level_short[i])[0]
                sl_hit = np.where(future_high >= sl_level_short[i])[0]

            if len(tp_hit) == 0 and len(sl_hit) == 0:
                outcomes[i] = 0
            elif len(tp_hit) == 0:
                outcomes[i] = 0
            elif len(sl_hit) == 0:
                outcomes[i] = 1
            elif tp_hit[0] < sl_hit[0]:
                outcomes[i] = 1
            else:
                outcomes[i] = 0

    return outcomes


def evaluate_strategy(name, signals, df, horizon, tp_points, sl_points):
    """Evaluiert eine Strategie mit gegebenen TP/SL-Parametern."""
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    outcomes = compute_tp_sl_outcome(close_arr, high_arr, low_arr, signals, horizon, tp_points, sl_points)

    valid = ~np.isnan(outcomes)
    n_trades = valid.sum()
    if n_trades == 0:
        return None

    n_wins = (outcomes[valid] == 1).sum()
    n_losses = (outcomes[valid] == 0).sum()
    win_rate = n_wins / n_trades

    # Expectancy: (Win% * TP) - (Loss% * SL)
    expectancy = (win_rate * tp_points) - ((1 - win_rate) * sl_points)

    return {
        "strategy": name,
        "horizon": horizon,
        "tp_points": tp_points,
        "sl_points": sl_points,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "expectancy_points": expectancy,
    }


def random_baseline(n, seed=42):
    """Random Signale: -1, 0, +1 mit gleicher Wahrscheinlichkeit."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])


def buy_and_hold(n):
    """Immer LONG."""
    return np.ones(n, dtype=int)


def ema_crossover(df):
    """EMA 20/50 Crossover."""
    signals = np.zeros(len(df))
    dist_ema_20 = df["dist_ema_20"].values
    dist_ema_50 = df["dist_ema_50"].values
    # Use slope for crossover detection
    ema_20 = df["ema_20"].values
    ema_50 = df["ema_50"].values

    above = ema_20 > ema_50
    # Crossover: previous below, current above -> LONG
    cross_above = np.zeros(len(df), dtype=bool)
    cross_below = np.zeros(len(df), dtype=bool)
    cross_above[1:] = ~above[:-1] & above[1:]
    cross_below[1:] = above[:-1] & ~above[1:]

    signals[cross_above] = 1
    signals[cross_below] = -1
    return signals


def rsi_mean_reversion(df):
    """RSI < 30 -> LONG, RSI > 70 -> SHORT."""
    signals = np.zeros(len(df))
    rsi = df["rsi_14"].values
    signals[rsi < 30] = 1
    signals[rsi > 70] = -1
    return signals


def bollinger_mean_reversion(df):
    """Bollinger Position < 0.05 -> LONG, > 0.95 -> SHORT."""
    signals = np.zeros(len(df))
    bb_pos = df["bb_position"].values
    signals[bb_pos < 0.05] = 1
    signals[bb_pos > 0.95] = -1
    return signals


def session_momentum(df):
    """Momentum während London/NY Session."""
    signals = np.zeros(len(df))
    is_london = df["is_london_session"].values
    is_ny = df["is_ny_session"].values
    roc_5 = df["roc_5"].values

    london_momentum = is_london.astype(bool) & (roc_5 > 0.001)
    ny_momentum = is_ny.astype(bool) & (roc_5 > 0.001)
    signals[london_momentum | ny_momentum] = 1
    return signals


def combined_trend(df):
    """Trendstärke + EMA-Slope."""
    signals = np.zeros(len(df))
    adx = df["adx_14"].values
    ema_slope = df["ema_slope_20"].values
    dist_ema = df["dist_ema_20"].values

    strong_trend = adx > 25
    long_cond = strong_trend & (ema_slope > 0) & (dist_ema > 0)
    short_cond = strong_trend & (ema_slope < 0) & (dist_ema < 0)

    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def main():
    df = load_data()
    n = len(df)

    strategies = {
        "Random": random_baseline(n),
        "BuyAndHold": buy_and_hold(n),
        "EMA_Crossover": ema_crossover(df),
        "RSI_MeanReversion": rsi_mean_reversion(df),
        "Bollinger_MeanReversion": bollinger_mean_reversion(df),
        "Session_Momentum": session_momentum(df),
        "Combined_Trend": combined_trend(df),
    }

    print(f"\nEvaluiere {len(strategies)} Baseline-Strategien...")
    print(f"  mit {len(HORIZONS)} Horizonten x {len(SL_POINTS)} SL x {len(RR_RATIOS)} R:R = {len(HORIZONS)*len(SL_POINTS)*len(RR_RATIOS)} Kombinationen")
    print("-" * 90)

    all_results = []

    for strategy_name, signals in strategies.items():
        n_signals = np.abs(signals).sum()
        print(f"\n{strategy_name} ({int(n_signals)} aktive Signale):")

        for horizon, sl_points, rr in product(HORIZONS, SL_POINTS, RR_RATIOS):
            tp_points = int(sl_points * rr)
            result = evaluate_strategy(strategy_name, signals, df, horizon, tp_points, sl_points)
            if result is not None:
                all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Find best strategy per horizon
    print("\n" + "=" * 90)
    print("ERGEBNISSE: Beste Baseline-Strategien")
    print("=" * 90)

    for horizon in HORIZONS:
        subset = results_df[results_df["horizon"] == horizon].sort_values("expectancy_points", ascending=False)
        if len(subset) > 0:
            best = subset.iloc[0]
            print(f"\n  Horizon {horizon}min:")
            print(f"    Beste Strategie: {best['strategy']}")
            print(f"      TP={int(best['tp_points'])}, SL={int(best['sl_points'])} | "
                  f"WinRate={best['win_rate']*100:.1f}% | "
                  f"Trades={int(best['n_trades'])} | "
                  f"Expectancy={best['expectancy_points']:.2f} Punkte")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n\nErgebnisse gespeichert: {results_path}")

    # Summary: best overall
    best_overall = results_df.sort_values("expectancy_points", ascending=False).head(10)
    print("\nTop 10 Strategien (nach Expectancy):")
    print(best_overall[["strategy", "horizon", "tp_points", "sl_points", "n_trades", "win_rate", "expectancy_points"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
