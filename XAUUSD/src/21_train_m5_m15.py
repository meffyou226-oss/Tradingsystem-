"""
M5/M15 Feature Engineering und Training.

Trainiert Modelle fuer M5 und M15 mit großen TP/SL (250+/100+).
Ziel: 5-10 Trades/Tag, hohe Winrate, 12.5+ USD pro Trade.
"""

import os
import sys
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

POINT = 0.01


def load_data(timeframe="M5"):
    """Lädt Daten für den gegebenen Timeframe."""
    path = os.path.join(DATA_DIR, f"xauusd_{timeframe.lower()}_raw.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {timeframe}: {len(df)} Kerzen geladen")
    return df


def add_features(df):
    """Fügt Features hinzu (anpassbar für M5/M15)."""
    df = df.copy()

    # Candle Features
    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    rng_safe = rng.where(rng > 0, np.nan)

    df["candle_return"] = body / df["open"]
    df["candle_range"] = rng / df["open"]
    df["body_size"] = np.abs(body) / df["open"]
    df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
    df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
    df["body_to_range"] = np.abs(body) / rng_safe

    # EMAs
    for p in [5, 10, 20, 50, 100, 200]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"dist_ema_{p}"] = (df["close"] - ema) / ema

    # EMA Slopes
    for p in [20, 50, 100]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema_slope_{p}"] = (ema - ema.shift(10)) / df["close"]

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_14_norm"] = df["atr_14"] / df["close"]

    # Volatility
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for p in [5, 10, 20, 50]:
        df[f"vol_{p}"] = log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60) * 100

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ROC
    for p in [1, 3, 5, 10, 15, 20, 50]:
        df[f"roc_{p}"] = (df["close"] - df["close"].shift(p)) / df["close"].shift(p)

    # Bollinger Bands
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    bb_upper = sma + 2.0 * std
    bb_lower = sma - 2.0 * std
    df["bb_width"] = (bb_upper - bb_lower) / sma
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # Recent High/Low
    for p in [5, 10, 20, 50]:
        df[f"high_{p}"] = df["high"].rolling(p).max()
        df[f"low_{p}"] = df["low"].rolling(p).min()
        df[f"dist_high_{p}"] = (df["close"] - df[f"high_{p}"]) / df["close"]
        df[f"dist_low_{p}"] = (df[f"low_{p}"] - df["close"]) / df["close"]

    # Breakout
    for p in [10, 20, 50]:
        recent_high = df["high"].rolling(p).max()
        recent_low = df["low"].rolling(p).min()
        df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int)
        df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int)

    # ADX
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

    # Time Features
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_london"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
    df["is_ny"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Volatility Regime
    vol_20 = log_ret.rolling(20).std()
    vol_50 = log_ret.rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int)
    df["vol_ratio"] = vol_20 / vol_50

    return df


def compute_target(df, horizon, tp_points, sl_points, direction="long"):
    """Berechnet Target für Swing."""
    close_arr = df["close"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)

    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    entry = close_arr

    if direction == "long":
        tp_level = entry + tp_points * POINT
        sl_level = entry - sl_points * POINT
    else:
        tp_level = entry - tp_points * POINT
        sl_level = entry + sl_points * POINT

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


def train_model(df, feature_cols, target_col, model_name):
    """Trainiert ein XGBoost Modell."""
    from xgboost import XGBClassifier

    # Splits
    train_mask = (df["timestamp"] >= "2024-01-01") & (df["timestamp"] <= "2025-04-30")
    val_mask = (df["timestamp"] >= "2025-05-01") & (df["timestamp"] <= "2025-08-31")
    test_mask = (df["timestamp"] >= "2025-09-01") & (df["timestamp"] <= "2026-08-24")

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df[target_col].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"    Target Balance: Train={y_train.mean()*100:.1f}%, Test={y_test.mean()*100:.1f}%")

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)

    print(f"    Test AUC: {test_auc:.4f}")

    # Speichern
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, test_auc


def main():
    print("=" * 70)
    print("M5/M15 SWING TRAINING")
    print("=" * 70)

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

    results = {}

    for timeframe in ["M5", "M15"]:
        print(f"\n{'='*50}")
        print(f"  {timeframe}")
        print(f"{'='*50}")

        df = load_data(timeframe)
        df = add_features(df)
        df = df.dropna().reset_index(drop=True)

        # Verschiedene Swing-Parameter testen
        swing_configs = [
            # (horizon_bars, tp, sl) - bars in timeframe units
            (6, 200, 80),   # M5: 30min, M15: 90min
            (9, 250, 100),  # M5: 45min, M15: 135min
            (12, 300, 100), # M5: 60min, M15: 3h
            (12, 300, 120),
            (18, 400, 150), # M5: 90min, M15: 4.5h
        ]

        for horizon_bars, tp, sl in swing_configs:
            print(f"\n  Config: H={horizon_bars} bars, TP={tp}, SL={sl}")

            # Targets
            target_long = compute_target(df, horizon_bars, tp, sl, "long")
            target_short = compute_target(df, horizon_bars, tp, sl, "short")

            df["target_long"] = target_long
            df["target_short"] = target_short

            pos_rate = target_long.mean()
            if not (0.20 <= pos_rate <= 0.55):
                print(f"    Skip: Pos rate {pos_rate*100:.1f}% nicht balanciert")
                continue

            # LONG trainieren
            print(f"    Training LONG...")
            model_long, auc_long = train_model(
                df, feature_cols, "target_long",
                f"xgboost_{timeframe.lower()}_swing_h{horizon_bars}_tp{tp}_sl{sl}"
            )

            # SHORT trainieren
            print(f"    Training SHORT...")
            model_short, auc_short = train_model(
                df, feature_cols, "target_short",
                f"xgboost_{timeframe.lower()}_swing_short_h{horizon_bars}_tp{tp}_sl{sl}"
            )

            results[f"{timeframe}_H{horizon_bars}_TP{tp}_SL{sl}"] = {
                "timeframe": timeframe,
                "horizon_bars": horizon_bars,
                "tp": tp,
                "sl": sl,
                "pos_rate": pos_rate,
                "long_auc": auc_long,
                "short_auc": auc_short,
                "avg_auc": (auc_long + auc_short) / 2,
            }

    # Zusammenfassung
    print(f"\n\n{'='*70}")
    print("ERGEBNISSE: M5/M15 SWING MODELLE")
    print(f"{'='*70}")

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df = results_df.sort_values("avg_auc", ascending=False)

    print(f"\n  {'Konfiguration':30s} | {'AUC':>6s} | {'Pos%':>6s} | {'TP/SL':>10s}")
    print(f"  {'-'*60}")
    for idx, row in results_df.iterrows():
        print(f"  {idx:30s} | {row['avg_auc']:6.4f} | {row['pos_rate']*100:5.1f}% | "
              f"{int(row['tp'])}/{int(row['sl'])}")

    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n  BESTES MODELL:")
        print(f"    {best.name}")
        print(f"    AUC: {best['avg_auc']:.4f}")
        print(f"    TP={int(best['tp'])}, SL={int(best['sl'])}")

    return results_df


if __name__ == "__main__":
    main()
