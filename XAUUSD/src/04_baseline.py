"""
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
