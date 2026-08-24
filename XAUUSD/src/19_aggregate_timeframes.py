"""
M5/M15 Daten aus M1 aggregieren und speichern.

Vorteile von M5/M15:
- Natürlichereres Trading (5-10 Trades/Tag)
- Größere TP/SL möglich (250+ Punkte)
- Weniger Noise, bessere Signale
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_raw.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_m1_data():
    print("Lade M1 Daten...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  M1: {len(df)} Kerzen ({df['timestamp'].iloc[0]} bis {df['timestamp'].iloc[-1]})")
    return df


def aggregate_to_m5(df):
    """Aggregiere M1 zu M5."""
    df = df.copy()
    df["time_5min"] = df["timestamp"].dt.floor("5min")

    agg = df.groupby("time_5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()

    agg = agg.rename(columns={"time_5min": "timestamp"})
    agg = agg[["timestamp", "open", "high", "low", "close"]]
    return agg


def aggregate_to_m15(df):
    """Aggregiere M1 zu M15."""
    df = df.copy()
    df["time_15min"] = df["timestamp"].dt.floor("15min")

    agg = df.groupby("time_15min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()

    agg = agg.rename(columns={"time_15min": "timestamp"})
    agg = agg[["timestamp", "open", "high", "low", "close"]]
    return agg


def aggregate_to_h1(df):
    """Aggregiere M1 zu H1."""
    df = df.copy()
    df["time_1h"] = df["timestamp"].dt.floor("1h")

    agg = df.groupby("time_1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()

    agg = agg.rename(columns={"time_1h": "timestamp"})
    agg = agg[["timestamp", "open", "high", "low", "close"]]
    return agg


def main():
    df_m1 = load_m1_data()

    # M5 aggregieren
    print("\nAggregiere zu M5...")
    df_m5 = aggregate_to_m5(df_m1)
    m5_path = os.path.join(OUTPUT_DIR, "xauusd_m5_raw.csv")
    df_m5.to_csv(m5_path, index=False)
    print(f"  M5: {len(df_m5)} Kerzen gespeichert: {m5_path}")

    # M15 aggregieren
    print("\nAggregiere zu M15...")
    df_m15 = aggregate_to_m15(df_m1)
    m15_path = os.path.join(OUTPUT_DIR, "xauusd_m15_raw.csv")
    df_m15.to_csv(m15_path, index=False)
    print(f"  M15: {len(df_m15)} Kerzen gespeichert: {m15_path}")

    # H1 aggregieren
    print("\nAggregiere zu H1...")
    df_h1 = aggregate_to_h1(df_m1)
    h1_path = os.path.join(OUTPUT_DIR, "xauusd_h1_raw.csv")
    df_h1.to_csv(h1_path, index=False)
    print(f"  H1: {len(df_h1)} Kerzen gespeichert: {h1_path}")

    # Zusammenfassung
    print(f"\nZusammenfassung:")
    print(f"  M1:  {len(df_m1):>8,} Kerzen (1 Min)")
    print(f"  M5:  {len(df_m5):>8,} Kerzen (5 Min)")
    print(f"  M15: {len(df_m15):>8,} Kerzen (15 Min)")
    print(f"  H1:  {len(df_h1):>8,} Kerzen (60 Min)")


if __name__ == "__main__":
    main()
