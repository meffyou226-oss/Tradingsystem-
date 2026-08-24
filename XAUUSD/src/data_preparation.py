"""
Datenvorbereitung: Konvertiert CSV zu Parquet für schnelles Laden.

Lädt die großen CSV-Dateien (Features und Targets),
konvertiert sie zu Parquet-Format mit komprimierter float32-Datentypen
und speichert sie für wiederverwendete Nutzung in der ML-Pipeline.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURES_PATH, TARGETS_PATH, FEATURE_COLUMNS,
                    TARGET_COLUMN, TRAIN_START, TRAIN_END, VAL_START,
                    VAL_END, TEST_START, TEST_END, MODELS_DIR)

PARQUET_FEATURES = os.path.join(os.path.dirname(FEATURES_PATH), "xauusd_m1_features.parquet")
PARQUET_TARGETS = os.path.join(os.path.dirname(TARGETS_PATH), "xauusd_m1_targets.parquet")
COMBINED_PATH = os.path.join(os.path.dirname(FEATURES_PATH), "xauusd_m1_combined.parquet")

USECOLS_FEATURES = ["timestamp", "open", "high", "low", "close"] + FEATURE_COLUMNS


def convert_to_parquet():
    """Konvertiere CSV-Dateien zu Parquet."""
    t0 = time.time()

    # Load features (only needed columns)
    print("Lade Features CSV (kann 1-2 Minuten dauern)...")
    t1 = time.time()
    df_features = pd.read_csv(FEATURES_PATH, usecols=USECOLS_FEATURES)
    df_features["timestamp"] = pd.to_datetime(df_features["timestamp"])
    df_features = df_features.sort_values("timestamp").reset_index(drop=True)

    # Optimize dtypes
    for col in FEATURE_COLUMNS + ["open", "high", "low", "close"]:
        df_features[col] = df_features[col].astype(np.float32)

    print(f"  Features geladen: {len(df_features)} Zeilen, {len(df_features.columns)} Spalten ({time.time()-t1:.1f}s)")

    # Load targets
    print("Lade Targets CSV...")
    t1 = time.time()
    df_targets = pd.read_csv(TARGETS_PATH, usecols=["timestamp", TARGET_COLUMN])
    df_targets["timestamp"] = pd.to_datetime(df_targets["timestamp"])
    df_targets = df_targets.sort_values("timestamp").reset_index(drop=True)
    df_targets[TARGET_COLUMN] = df_targets[TARGET_COLUMN].astype(np.float32)
    print(f"  Targets geladen: {len(df_targets)} Zeilen ({time.time()-t1:.1f}s)")

    # Merge features and target
    t1 = time.time()
    df = df_features.merge(df_targets, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Zusammengeführt: {len(df)} Zeilen ({time.time()-t1:.1f}s)")

    # Save parquet
    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    t1 = time.time()
    df.to_parquet(COMBINED_PATH, compression="snappy", index=False)
    size_mb = os.path.getsize(COMBINED_PATH) / 1024 / 1024
    print(f"  Parquet gespeichert: {size_mb:.1f} MB ({time.time()-t1:.1f}s)")

    # Print split sizes
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    print(f"\nZeitbasierte Splits:")
    print(f"  Training:   {df[train_mask]['timestamp'].min()} - {df[train_mask]['timestamp'].max()} ({train_mask.sum()} Zeilen)")
    print(f"  Validation: {df[val_mask]['timestamp'].min()} - {df[val_mask]['timestamp'].max()} ({val_mask.sum()} Zeilen)")
    print(f"  Test:       {df[test_mask]['timestamp'].min()} - {df[test_mask]['timestamp'].max()} ({test_mask.sum()} Zeilen)")

    print(f"\nGesamtzeit: {time.time()-t0:.1f}s")
    return df


def load_combined():
    """Lädt die kombinierte Parquet-Datei (cache-fähig)."""
    if not os.path.exists(COMBINED_PATH):
        print("Parquet nicht gefunden, kontertiere CSV...")
        convert_to_parquet()
    return pd.read_parquet(COMBINED_PATH)


def load_splits():
    """Lädt Daten und gibt Train/Val/Test Splits zurück."""
    df = load_combined()

    train = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)].copy()
    val = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)].copy()
    test = df[(df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)].copy()

    print(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")
    return train, val, test, df


if __name__ == "__main__":
    df = convert_to_parquet()
