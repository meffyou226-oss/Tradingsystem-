"""
Test-Skript für den Trading Bot (LONG + SHORT, ohne MT5).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from live_bot import FeatureCalculator, CONFIG


def generate_test_data(n=500):
    np.random.seed(42)
    timestamps = []
    base_time = datetime(2026, 8, 20, 8, 0, 0)
    for i in range(n):
        t = base_time + timedelta(minutes=i)
        while t.weekday() >= 5 or t.hour < 8 or t.hour >= 20:
            t += timedelta(minutes=1)
        timestamps.append(t)

    price = 3350.0
    data = []
    for t in timestamps:
        ret = np.random.normal(0, 0.0005)
        price *= (1 + ret)
        high = price * (1 + abs(np.random.normal(0, 0.0003)))
        low = price * (1 - abs(np.random.normal(0, 0.0003)))
        open_price = price * (1 + np.random.normal(0, 0.0001))
        volume = int(np.random.uniform(50, 500))
        data.append({
            "timestamp": t, "open": open_price,
            "high": max(high, open_price, price),
            "low": min(low, open_price, price),
            "close": price, "tick_volume": volume, "spread": 30, "real_volume": volume * 100,
        })
    return pd.DataFrame(data)


def test_models():
    print("=" * 60)
    print("XAUUSD Bot Test (LONG + SHORT)")
    print("=" * 60)

    # Modelle laden
    long_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    short_path = os.path.join(MODELS_DIR, "xgboost_short.pkl")

    with open(long_path, "rb") as f:
        model_long = pickle.load(f)
    with open(short_path, "rb") as f:
        model_short = pickle.load(f)
    print(f"[OK] LONG + SHORT Modell geladen")

    # Test-Daten
    df = generate_test_data(500)
    calc = FeatureCalculator()
    df = calc.calculate(df)

    feature_cols = [c for c in calc.feature_columns if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)

    # Vorhersagen
    preds_long = model_long.predict_proba(X)[:, 1]
    preds_short = model_short.predict_proba(X)[:, 1]

    # Statistiken
    long_signals = (preds_long >= CONFIG["long_confidence_threshold"]).sum()
    short_signals = (preds_short >= CONFIG["short_confidence_threshold"]).sum()

    print(f"\n[STATS] {len(X)} Kerzen analysiert:")
    print(f"  LONG  Pred: Mean={preds_long.mean():.3f} | Std={preds_long.std():.3f} | "
          f"Signals (>={CONFIG['long_confidence_threshold']}): {long_signals} ({long_signals/len(X)*100:.1f}%)")
    print(f"  SHORT Pred: Mean={preds_short.mean():.3f} | Std={preds_short.std():.3f} | "
          f"Signals (>={CONFIG['short_confidence_threshold']}): {short_signals} ({short_signals/len(X)*100:.1f}%)")

    # Letzte Kerzen
    print(f"\n[OK] Letzte 10 Kerzen:")
    print(f"  {'Zeit':20s} | {'Close':>8s} | {'LONG':>6s} | {'SHORT':>6s} | {'Signal':>8s}")
    print(f"  {'-'*60}")

    for i in range(-10, 0):
        idx = len(X) + i
        t = df["timestamp"].iloc[idx]
        c = df["close"].iloc[idx]
        pl = preds_long[idx]
        ps = preds_short[idx]

        signal = ""
        if pl >= CONFIG["long_confidence_threshold"]:
            signal = "LONG"
        elif ps >= CONFIG["short_confidence_threshold"]:
            signal = "SHORT"

        print(f"  {t.strftime('%Y-%m-%d %H:%M'):20s} | {c:8.3f} | {pl:6.3f} | {ps:6.3f} | {signal:>8s}")

    print("\n[Test abgeschlossen]")


if __name__ == "__main__":
    test_models()
