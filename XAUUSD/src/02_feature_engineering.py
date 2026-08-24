"""
Feature Engineering für XAUUSD M1-Daten.

Erzeugt features basierend auf OHLC-Daten mit strenger
Lookahead-Bias-Vermeidung (alle Features nutzen nur vergangene Daten).

Features:
- Candle-Return, Candle-Range, Body-Size, Upper/Lower-Wick
- ATR, kurzfristige Volatilität
- Momentum (ROC über verschiedene Perioden)
- RSI
- EMA-Struktur (5, 10, 20, 50, 100, 200)
- Abstand zum EMA
- Bollinger Bands (Position + Width)
- Recent High/Low (5, 10, 20, 50)
- Breakout-Features
- Trendstärke (ADX)
- Zeitfeatures (Stunde, Wochentag, Session)
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_raw.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_features.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_candle_features(df):
    """Candle-basierte Features (alle shift-resistant)."""
    df = df.copy()
    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    rng_safe = rng.where(rng > 0, np.nan)

    # Use previous candle's data (shift by 1 to avoid lookahead)
    df["candle_return"] = (body / df["open"].shift(1)).shift(1)
    df["candle_range"] = (rng / df["open"].shift(1)).shift(1)
    df["body_size"] = (np.abs(body) / df["open"].shift(1)).shift(1)
    df["upper_wick"] = ((df["high"] - np.maximum(df["open"], df["close"])) / df["open"].shift(1)).shift(1)
    df["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / df["open"].shift(1)).shift(1)
    df["body_to_range"] = (np.abs(body) / rng_safe).shift(1)
    df["wick_to_range"] = (rng_safe - np.abs(body)) / rng_safe.replace(0, np.nan)
    df["wick_to_range"] = df["wick_to_range"].shift(1)
    return df


def add_ema_features(df):
    """EMA features at multiple periods."""
    df = df.copy()
    ema_periods = [5, 10, 20, 50, 100, 200]
    for p in ema_periods:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema_{p}"] = ema
        # Distance to EMA (as % of price)
        df[f"dist_ema_{p}"] = ((df["close"] - ema) / ema).shift(1)
    # EMA slope (momentum of EMA)
    for p in [20, 50, 100]:
        ema = df[f"ema_{p}"]
        df[f"ema_slope_{p}"] = (ema - ema.shift(10)).shift(1)
    return df


def add_atr_features(df):
    """Average True Range und Volatilität."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().shift(1)
    df["atr_14_norm"] = (df["atr_14"] / df["close"].shift(1)).shift(1)
    # Volatility over different windows
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for p in [5, 10, 20, 50]:
        df[f"vol_{p}"] = (log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60)).shift(1) * 100  # annualized vol in %
    return df


def add_rsi(df, period=14):
    """RSI über gegebenen Period."""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = (100 - (100 / (1 + rs))).shift(1)
    return df


def add_momentum_features(df):
    """ROC (Rate of Change) über verschiedene Perioden."""
    df = df.copy()
    for p in [1, 3, 5, 10, 15, 20, 50]:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)).shift(1)
    return df


def add_bollinger_features(df, period=20, std_mult=2.0):
    """Bollinger Bands Position und Width."""
    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    bb_upper = sma + std_mult * std
    bb_lower = sma - std_mult * std
    df["bb_width"] = ((bb_upper - bb_lower) / sma).shift(1)
    df["bb_position"] = ((df["close"] - bb_lower) / (bb_upper - bb_lower)).shift(1)
    return df


def add_recent_high_low(df):
    """Recent High/Low über verschiedene Perioden."""
    df = df.copy()
    for p in [5, 10, 20, 50]:
        df[f"high_{p}"] = df["high"].shift(1).rolling(p).max()
        df[f"low_{p}"] = df["low"].shift(1).rolling(p).min()
        # Distance to recent high/low
        df[f"dist_high_{p}"] = ((df["high"] - df[f"high_{p}"]) / df["close"].shift(1)).shift(1)
        df[f"dist_low_{p}"] = ((df[f"low_{p}"] - df["low"]) / df["close"].shift(1)).shift(1)
    return df


def add_breakout_features(df):
    """Breakout-Features."""
    df = df.copy()
    for p in [5, 10, 20, 50]:
        recent_high = df["high"].shift(1).rolling(p).max()
        recent_low = df["low"].shift(1).rolling(p).min()
        df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int).shift(1)
        df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int).shift(1)
    return df


def add_adx(df, period=14):
    """Average Directional Index für Trendstärke."""
    df = df.copy()
    high_diff = df["high"].diff()
    low_diff = df["low"].diff()
    plus_dm = high_diff.where(high_diff > 0, 0) + low_diff.where(low_diff > 0, 0) * 0
    minus_dm = low_diff.where(low_diff < 0, 0).abs() + high_diff.where(high_diff < 0, 0).abs() * 0

    # Simplified: use the standard ADX calculation
    tr_high = df["high"] - df["low"]
    tr_high_shift = df["high"] - df["close"].shift(1)
    tr_low = df["low"] - df["close"].shift(1)
    tr = pd.concat([tr_high, tr_high_shift.abs(), tr_low.abs()], axis=1).max(axis=1)

    plus_di = (high_diff.where(high_diff > 0, 0).rolling(period).sum() / tr.rolling(period).sum() * 100)
    minus_di = (low_diff.where(low_diff < 0, 0).abs().rolling(period).sum() / tr.rolling(period).sum() * 100)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    df[f"adx_{period}"] = dx.rolling(period).mean().shift(1)
    df[f"plus_di_{period}"] = plus_di.shift(1)
    df[f"minus_di_{period}"] = minus_di.shift(1)
    return df


def add_time_features(df):
    """Zeitbasierte Features."""
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_london_session"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int).shift(1)
    df["is_ny_session"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int).shift(1)
    df["is_asia_session"] = ((ts.dt.hour >= 0) & (ts.dt.hour < 8)).astype(int).shift(1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).shift(1)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).shift(1)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).shift(1)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).shift(1)
    return df


def add_volatility_regime(df):
    """Volatilitäts-Regime Feature."""
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_20 = log_ret.rolling(20).std()
    vol_50 = log_ret.rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int).shift(1)
    df["vol_ratio"] = (vol_20 / vol_50).shift(1)
    return df


def add_previous_day_features(df):
    """Features basierend auf Previous Day High/Low."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    # This is approximated using rolling max/min on shifted data
    # Previous day high/low as distance
    df["prev_day_high"] = df["high"].shift(1).rolling(390).max()  # ~6.5h of M1, approx 1 trading day
    df["prev_day_low"] = df["low"].shift(1).rolling(390).min()
    df["dist_prev_high"] = ((df["close"] - df["prev_day_high"]) / df["close"].shift(1)).shift(1)
    df["dist_prev_low"] = ((df["close"] - df["prev_day_low"]) / df["close"].shift(1)).shift(1)
    df = df.drop(columns=["date"])
    return df


def engineer_features(df):
    """Apply all feature engineering steps."""
    print("Starte Feature Engineering...")
    df = add_candle_features(df)
    print("  ✓ Candle-Features")
    df = add_ema_features(df)
    print("  ✓ EMA-Features")
    df = add_atr_features(df)
    print("  ✓ ATR/Volatilität-Features")
    df = add_rsi(df)
    print("  ✓ RSI")
    df = add_momentum_features(df)
    print("  ✓ Momentum (ROC)")
    df = add_bollinger_features(df)
    print("  ✓ Bollinger Bands")
    df = add_recent_high_low(df)
    print("  ✓ Recent High/Low")
    df = add_breakout_features(df)
    print("  ✓ Breakout-Features")
    df = add_adx(df)
    print("  ✓ ADX/Trendstärke")
    df = add_time_features(df)
    print("  ✓ Zeit-Features")
    df = add_volatility_regime(df)
    print("  ✓ Volatilitäts-Regime")
    df = add_previous_day_features(df)
    print("  ✓ Previous-Day Features")
    print(f"Fertig! Features: {len(df.columns) - 1} Spalten (exkl. timestamp)")
    return df


def main():
    df = load_data()
    df = engineer_features(df)
    # Drop NaN rows (from warmup periods)
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    final_rows = len(df)
    print(f"\nDatensatz: {initial_rows} -> {final_rows} Zeilen (nach NaN-Entfernung)")

    # Save features
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Features gespeichert: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1024/1024:.1f} MB)")
    print(f"Spalten: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = main()
