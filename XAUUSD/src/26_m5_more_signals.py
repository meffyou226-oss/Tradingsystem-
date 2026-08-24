"""
M5 Swing: Mehr Signale durch niedrigeren Threshold.

Ziel: 2-5 Trades/Tag mit TP=250/SL=100.
Ansatz: Threshold senken, weniger restriktive Filter.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, POINT
from backtest_engine import BacktestEngine


def load_m5_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m5_raw.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def add_features(df):
    df = df.copy()
    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    rng_safe = rng.where(rng > 0, np.nan)

    df["candle_return"] = body / df["open"]
    df["candle_range"] = rng / df["open"]
    df["body_size"] = np.abs(body) / df["open"]
    df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
    df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
    df["body_to_range"] = np.abs(body) / rng_safe

    for p in [5, 10, 20, 50, 100, 200]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"dist_ema_{p}"] = (df["close"] - ema) / ema

    for p in [20, 50, 100]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema_slope_{p}"] = (ema - ema.shift(10)) / df["close"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["atr_14_norm"] = atr_14 / df["close"]

    log_ret = np.log(df["close"] / df["close"].shift(1))
    for p in [5, 10, 20, 50]:
        df[f"vol_{p}"] = log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60) * 100

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    for p in [1, 3, 5, 10, 15, 20, 50]:
        df[f"roc_{p}"] = (df["close"] - df["close"].shift(p)) / df["close"].shift(p)

    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    bb_upper = sma + 2.0 * std
    bb_lower = sma - 2.0 * std
    df["bb_width"] = (bb_upper - bb_lower) / sma
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    for p in [5, 10, 20, 50]:
        df[f"high_{p}"] = df["high"].rolling(p).max()
        df[f"low_{p}"] = df["low"].rolling(p).min()
        df[f"dist_high_{p}"] = (df["close"] - df[f"high_{p}"]) / df["close"]
        df[f"dist_low_{p}"] = (df[f"low_{p}"] - df["close"]) / df["close"]

    for p in [10, 20, 50]:
        recent_high = df["high"].rolling(p).max()
        recent_low = df["low"].rolling(p).min()
        df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int)
        df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int)

    high_diff = df["high"].diff()
    low_diff = df["low"].diff()
    tr_abs = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
    minus_dm = (-low_diff).where((low_diff < 0) & (-low_diff > high_diff), 0)
    plus_di = (plus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
    minus_di = (minus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
    df["adx_14"] = dx.rolling(14).mean()
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_london"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
    df["is_ny"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    vol_20 = log_ret.rolling(20).std()
    vol_50 = log_ret.rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int)
    df["vol_ratio"] = vol_20 / vol_50

    return df


def compute_target(df, horizon, tp, sl, direction="long"):
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    entry = close_arr

    if direction == "long":
        tp_level = entry + tp * POINT
        sl_level = entry - sl * POINT
    else:
        tp_level = entry - tp * POINT
        sl_level = entry + sl * POINT

    for offset in range(1, horizon + 1):
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)

        undetermined = targets[current_idx] == 0

        if direction == "long":
            tp_hits = high_arr[future_idx] >= tp_level[current_idx]
            sl_hits = low_arr[future_idx] <= sl_level[current_idx]
        else:
            tp_hits = low_arr[future_idx] <= tp_level[current_idx]
            sl_hits = high_arr[future_idx] >= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        targets[current_idx[new_tp]] = 1

        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def generate_signals(df, model, feature_cols, conf_threshold=0.50, max_daily=10):
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]

    hour = df["timestamp"].dt.hour.values
    # Alle Handelszeiten (kein Session-Filter)
    is_active = hour >= 0

    signals = np.zeros(len(df), dtype=bool)
    high_conf = predictions >= conf_threshold

    current_date = None
    daily_trades = 0
    for i in range(len(df)):
        if not high_conf[i]:
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
    print("M5 SWING: Mehr Signale")
    print("=" * 70)

    df = load_m5_data()
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "candle_return", "candle_range", "body_size", "upper_wick", "lower_wick",
        "body_to_range",
        "dist_ema_5", "dist_ema_10", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
        "ema_slope_20", "ema_slope_50", "ema_slope_100",
        "atr_14_norm", "vol_5", "vol_10", "vol_20", "vol_50",
        "rsi_14",
        "roc_1", "roc_3", "roc_5", "roc_10", "roc_15", "roc_20", "roc_50",
        "bb_width", "bb_position",
        "dist_high_5", "dist_low_5", "dist_high_10", "dist_low_10",
        "dist_high_20", "dist_low_20", "dist_high_50", "dist_low_50",
        "breakout_high_10", "breakout_low_10", "breakout_high_20", "breakout_low_20",
        "breakout_high_50", "breakout_low_50",
        "adx_14", "plus_di_14", "minus_di_14",
        "hour", "day_of_week", "is_london", "is_ny",
        "hour_sin", "hour_cos",
        "vol_regime", "vol_ratio",
    ]

    test_mask = (df["timestamp"] >= "2025-09-01") & (df["timestamp"] <= "2026-08-24")
    test_df = df[test_mask].copy()

    print(f"  Test: {len(test_df)} M5 Kerzen")

    # M5 Modelle laden
    with open(os.path.join(MODELS_DIR, "xgboost_m5_swing_h6_tp200_sl80.pkl"), "rb") as f:
        model_long = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "xgboost_m5_swing_short_h6_tp200_sl80.pkl"), "rb") as f:
        model_short = pickle.load(f)

    # Teste verschiedene Thresholds
    for threshold in [0.50, 0.55, 0.60]:
        signals_long = generate_signals(test_df, model_long, feature_cols, threshold, 10)
        signals_short = generate_signals(test_df, model_short, feature_cols, threshold, 10)

        n_days = (test_df["timestamp"].max() - test_df["timestamp"].min()).days

        print(f"\n  Threshold={threshold}:")
        print(f"    LONG: {signals_long.sum()} Signale ({signals_long.sum()/n_days:.1f}/Tag)")
        print(f"    SHORT: {signals_short.sum()} Signale ({signals_short.sum()/n_days:.1f}/Tag)")

        if signals_long.sum() > 0:
            engine = BacktestEngine(tp_points=200, sl_points=80, horizon=30)
            trades_long, stats_long = engine.run(test_df, signals_long)
            if stats_long:
                print(f"    LONG Result: {stats_long['n_trades']} Trades, Win={stats_long['win_rate']*100:.0f}%, PF={stats_long['profit_factor']:.1f}")

        if signals_short.sum() > 0:
            engine = BacktestEngine(tp_points=200, sl_points=80, horizon=30)
            trades_short, stats_short = engine.run(test_df, signals_short)
            if stats_short:
                print(f"    SHORT Result: {stats_short['n_trades']} Trades, Win={stats_short['win_rate']*100:.0f}%, PF={stats_short['profit_factor']:.1f}")


if __name__ == "__main__":
    main()
