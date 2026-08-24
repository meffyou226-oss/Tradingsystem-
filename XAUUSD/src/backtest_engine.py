"""
Backtest-Engine für TP/SL-basierte Trading-Simulation.

Simuliert realistisch:
- Spread und Slippage beim Entry
- TP/SL-Levels basierend auf Entry-Preis
- "First hit" Logik ("which der Level zuerst erreicht wird")
- Maximale Haltedauer (Horizont)
- Equity Curve, Drawdown, Profit Factor, Sharpe Ratio

Wird von Baseline- und ML-Strategien geteilt.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import POINT, SPREAD_POINTS, SLIPPAGE_POINTS


def simulate_trade_vectorized(close_arr, high_arr, low_arr, entry_idx,
                              tp_points, sl_points, horizon, point=POINT,
                              spread_points=SPREAD_POINTS, slippage_points=SLIPPAGE_POINTS):
    """
    Simuliert einen einzelnen Trade.
    
    Entry: close[entry_idx] + entry_cost (spread)
    TP: entry_price + tp_points * point
    SL: entry_price - sl_points * point
    
    Returns: (pnl, exit_idx, exit_reason, exit_price)
    oder None wenn nicht genug Daten vorhanden.
    """
    n = len(close_arr)
    if entry_idx + horizon >= n:
        return None

    entry_price = close_arr[entry_idx] + (spread_points + slippage_points) * point
    tp_level = entry_price + tp_points * point
    sl_level = entry_price - sl_points * point

    # Look forward horizon bars
    end_idx = min(entry_idx + horizon, n - 1)
    future_high = high_arr[entry_idx + 1: end_idx + 1]
    future_low = low_arr[entry_idx + 1: end_idx + 1]

    if len(future_high) == 0:
        return None

    # Find first TP hit and first SL hit
    tp_hit_mask = future_high >= tp_level
    sl_hit_mask = future_low <= sl_level

    tp_hit_idx = np.argmax(tp_hit_mask) if np.any(tp_hit_mask) else None
    sl_hit_idx = np.argmax(sl_hit_mask) if np.any(sl_hit_mask) else None

    # Determine which came first
    tp_actual_pos = tp_hit_idx if tp_hit_idx is not None else len(future_high)
    sl_actual_pos = sl_hit_idx if sl_hit_idx is not None else len(future_high)

    if tp_actual_pos <= sl_actual_pos and tp_hit_idx is not None:
        # TP hit first (or simultaneously)
        exit_price = tp_level
        pnl = tp_points * point  # Gross profit
        exit_reason = "tp"
    elif sl_hit_idx is not None:
        # SL hit first
        exit_price = sl_level
        pnl = -(sl_points * point)  # Gross loss
        exit_reason = "sl"
    else:
        # Neither hit: exit at horizon end
        exit_idx = entry_idx + horizon
        exit_price = close_arr[exit_idx] - spread_points * point
        pnl = exit_price - entry_price
        exit_reason = "expiry"

    if tp_hit_idx is not None and sl_hit_idx is not None and tp_actual_pos == sl_actual_pos:
        exit_reason = "tp"  # If simultaneous, TP wins (conservative for long)

    exit_idx = entry_idx + min(tp_actual_pos if tp_hit_idx is not None else float('inf'),
                                sl_actual_pos if sl_hit_idx is not None else float('inf'),
                                horizon)

    return {
        "entry_idx": entry_idx,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_idx": int(exit_idx),
        "pnl": pnl,
        "exit_reason": exit_reason,
        "tp_points": tp_points,
        "sl_points": sl_points,
        "horizon": horizon,
    }


def run_backtest(df, signals, tp_points, sl_points, horizon,
                 spread_points=SPREAD_POINTS, slippage_points=SLIPPAGE_POINTS,
                 point=POINT):
    """
    Führt Backtest durch.
    
    Args:
        df: DataFrame mit Spalten ['open','high','low','close','timestamp']
        signals: Boolean-Array, True = Long-Signal an diesem Index
        tp_points: Take Profit in Punkten
        sl_points: Stop Loss in Punkten
        horizon: Max Holding Time in Minuten (M1 = 1 Bar pro Minute)
    
    Returns:
        trades_df: DataFrame mit allen Trades
        stats: Dictionary mit Performance-Metriken
    """
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    trades = []
    n = len(df)

    for i in range(n):
        if signals[i]:
            result = simulate_trade_vectorized(
                close_arr, high_arr, low_arr, i,
                tp_points, sl_points, horizon, point,
                spread_points, slippage_points
            )
            if result is not None:
                trades.append(result)

    if len(trades) == 0:
        print("  Keine Trades generiert!")
        return pd.DataFrame(), {}

    trades_df = pd.DataFrame(trades)
    trades_df["entry_time"] = df["timestamp"].iloc[trades_df["entry_idx"]].values
    trades_df["exit_time"] = df["timestamp"].iloc[trades_df["exit_idx"]].values
    trades_df["r_multiple"] = trades_df["pnl"] / (sl_points * point)

    stats = compute_stats(trades_df, spread_points, slippage_points, tp_points, sl_points, horizon)
    return trades_df, stats


def compute_stats(trades_df, spread_points, slippage_points, tp_points, sl_points, horizon):
    """Berechnet Performance-Metriken."""
    n_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    breakevens = trades_df[trades_df["pnl"] == 0]

    total_profit = trades_df["pnl"].sum()
    gross_wins = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Max Drawdown (equity curve)
    trades_df_sorted = trades_df.sort_values("entry_idx").reset_index(drop=True)
    equity = trades_df_sorted["pnl"].cumsum()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max)
    max_drawdown = drawdown.min()

    # Average trade duration
    durations = trades_df["exit_idx"] - trades_df["entry_idx"]
    avg_duration = durations.mean()

    # TP/SL hit rates
    tp_hits = (trades_df["exit_reason"] == "tp").sum()
    sl_hits = (trades_df["exit_reason"] == "sl").sum()
    expiry_hits = (trades_df["exit_reason"] == "expiry").sum()

    # Sharpe Ratio (assumes risk-free rate = 0)
    pnl_series = trades_df_sorted["pnl"]
    sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(252 * 24 * 60) if pnl_series.std() > 0 else 0

    # Max consecutive losses
    signs = (trades_df_sorted["pnl"] > 0).astype(int).values
    max_consec_losses = 0
    max_consec_wins = 0
    current_streak = 0
    current_is_win = True
    for s in signs:
        if s == 1:
            if current_is_win:
                current_streak += 1
            else:
                current_streak = 1
                current_is_win = True
            max_consec_wins = max(max_consec_wins, current_streak)
        else:
            if not current_is_win:
                current_streak += 1
            else:
                current_streak = 1
                current_is_win = False
            max_consec_losses = max(max_consec_losses, current_streak)

    return {
        "n_trades": n_trades,
        "total_profit": total_profit,
        "gross_wins": gross_wins,
        "gross_losses": gross_losses,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "avg_trade_duration_min": avg_duration,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "expiry_hits": expiry_hits,
        "tp_hit_rate": tp_hits / n_trades,
        "sl_hit_rate": sl_hits / n_trades,
        "expiry_rate": expiry_hits / n_trades,
        "max_consec_losses": max_consec_losses,
        "max_consec_wins": max_consec_wins,
        "sharpe_ratio": sharpe,
        "spread_points": spread_points,
        "slippage_points": slippage_points,
        "tp_points": tp_points,
        "sl_points": sl_points,
        "horizon": horizon,
        "rr_ratio": tp_points / sl_points,
    }


class BacktestEngine:
    """Klasse-basierte Backtest-Engine für wiederverwendbare Simulationen."""

    def __init__(self, tp_points, sl_points, horizon,
                 spread_points=SPREAD_POINTS, slippage_points=SLIPPAGE_POINTS,
                 point=POINT):
        self.tp_points = tp_points
        self.sl_points = sl_points
        self.horizon = horizon
        self.spread_points = spread_points
        self.slippage_points = slippage_points
        self.point = point

    def run(self, df, signals):
        return run_backtest(df, signals, self.tp_points, self.sl_points, self.horizon,
                           self.spread_points, self.slippage_points, self.point)

    def run_from_predictions(self, df, predictions, threshold=0.5):
        """Konvertiert ML-Prognosen in Trading-Signale und führt Backtest durch."""
        signals = predictions >= threshold
        return self.run(df, signals)
