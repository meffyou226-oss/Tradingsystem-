"""
Feature-Extraktion für Live-Daten (MT5).

Berechnet exakt dieselben Features wie im Training (src/02_feature_engineering.py),
um Modell-Kompatibilität sicherzustellen.

Wichtig: Alle Features nutzen nur vergangene Daten (Lookahead-Bias-frei).
"""

import numpy as np
import pandas as pd
import logging

from config import (FEATURE_COLUMNS, EMA_PERIODS, ATR_PERIOD, RSI_PERIOD,
                    VOL_PERIODS, ROC_PERIODS, BB_PERIOD, BB_STD,
                    RECENT_PERIODS, PREV_DAY_BARS, TRADE_AMOUNT, TP_POINTS,
                    SL_POINTS, POINT)

logger = logging.getLogger(__name__)


def compute_features(df):
    """
    Berechnet alle 63 Features aus OHLC-Daten.
    df muss Spalten: timestamp, open, high, low, close enthalten.
    Returniert DataFrame mit Feature-Spalten (letzte Zeile = aktuelle Signale).
    """
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # === Candle-Features (alle shift(1) für Lookahead-Immunität) ===
    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    rng_safe = rng.where(rng > 0, np.nan)

    df["candle_return"] = (body / df["open"].shift(1)).shift(1)
    df["candle_range"] = (rng / df["open"].shift(1)).shift(1)
    df["body_size"] = (np.abs(body) / df["open"].shift(1)).shift(1)
    df["upper_wick"] = ((df["high"] - np.maximum(df["open"], df["close"])) / df["open"].shift(1)).shift(1)
    df["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / df["open"].shift(1)).shift(1)
    df["body_to_range"] = (np.abs(body) / rng_safe).shift(1)
    df["wick_to_range"] = ((rng_safe - np.abs(body)) / rng_safe.replace(0, np.nan)).shift(1)

    # === EMA-Features ===
    for p in EMA_PERIODS:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema_{p}"] = ema
        df[f"dist_ema_{p}"] = ((df["close"] - ema) / ema).shift(1)
    for p in [20, 50, 100]:
        ema = df[f"ema_{p}"]
        df[f"ema_slope_{p}"] = (ema - ema.shift(10)).shift(1)

    # === ATR & Volatilität ===
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(ATR_PERIOD).mean().shift(1)
    df["atr_14_norm"] = (df["atr_14"] / df["close"].shift(1)).shift(1)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    for p in VOL_PERIODS:
        df[f"vol_{p}"] = (log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60)).shift(1) * 100

    # === RSI ===
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).shift(1)

    # === Momentum (ROC) ===
    for p in ROC_PERIODS:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)).shift(1)

    # === Bollinger Bands ===
    sma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    bb_upper = sma + BB_STD * std
    bb_lower = sma - BB_STD * std
    df["bb_width"] = ((bb_upper - bb_lower) / sma).shift(1)
    df["bb_position"] = ((df["close"] - bb_lower) / (bb_upper - bb_lower)).shift(1)

    # === Recent High/Low ===
    for p in RECENT_PERIODS:
        df[f"high_{p}"] = df["high"].shift(1).rolling(p).max()
        df[f"low_{p}"] = df["low"].shift(1).rolling(p).min()
        df[f"dist_high_{p}"] = ((df["high"] - df[f"high_{p}"]) / df["close"].shift(1)).shift(1)
        df[f"dist_low_{p}"] = ((df[f"low_{p}"] - df["low"]) / df["close"].shift(1)).shift(1)

    # === Breakout-Features ===
    for p in RECENT_PERIODS:
        recent_high = df["high"].shift(1).rolling(p).max()
        recent_low = df["low"].shift(1).rolling(p).min()
        df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int).shift(1)
        df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int).shift(1)

    # === ADX ===
    high_diff = df["high"].diff()
    low_diff = df["low"].diff()
    tr_combined = pd.concat([df["high"] - df["low"],
                             (df["high"] - df["close"].shift(1)).abs(),
                             (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    plus_di = (high_diff.where(high_diff > 0, 0).rolling(ATR_PERIOD).sum() /
               tr_combined.rolling(ATR_PERIOD).sum() * 100)
    minus_di = (low_diff.where(low_diff < 0, 0).abs().rolling(ATR_PERIOD).sum() /
                tr_combined.rolling(ATR_PERIOD).sum() * 100)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    df["adx_14"] = dx.rolling(ATR_PERIOD).mean().shift(1)
    df["plus_di_14"] = plus_di.shift(1)
    df["minus_di_14"] = minus_di.shift(1)

    # === Time-Features ===
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

    # === Volatility Regime ===
    vol_20 = log_ret.rolling(20).std()
    vol_50 = log_ret.rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int).shift(1)
    df["vol_ratio"] = (vol_20 / vol_50).shift(1)

    # === Previous Day Features ===
    df["prev_day_high"] = df["high"].shift(1).rolling(PREV_DAY_BARS).max()
    df["prev_day_low"] = df["low"].shift(1).rolling(PREV_DAY_BARS).min()
    df["dist_prev_high"] = ((df["close"] - df["prev_day_high"]) / df["close"].shift(1)).shift(1)
    df["dist_prev_low"] = ((df["close"] - df["prev_day_low"]) / df["close"].shift(1)).shift(1)

    return df


def get_latest_features(df):
    """
    Berechnet Features und gibt die letzte Zeile als Feature-Vektor zurück.
    
    Args:
        df: DataFrame mit mindestens 200 Zeilen für Warmup
    
    Returns:
        dict: Feature-Name -> Wert
    """
    df_features = compute_features(df)
    
    # Drop NaN rows (Warmup-Perioden)
    df_features_clean = df_features.dropna()
    
    if len(df_features_clean) == 0:
        return None
    
    latest = df_features_clean.iloc[-1]
    
    # Build feature dict matching FEATURE_COLUMNS
    features = {}
    for col in FEATURE_COLUMNS:
        if col in latest.index:
            features[col] = float(latest[col])
        else:
            features[col] = 0.0
    
    return features


def features_to_array(features_dict, feature_cols):
    """Konvertiert Feature-Dict zu numpy Array in richtiger Reihenfolge."""
    arr = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        arr[i] = features_dict.get(col, 0.0)
    return arr
