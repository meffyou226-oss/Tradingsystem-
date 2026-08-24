"""
Datenanalyse-Modul für XAUUSD M1-Daten.

Prüft:
- Zeitraum, Anzahl Kerzen, Spalten
- OHLC-Struktur
- Volumen (falls vorhanden)
- fehlende Kerzen
- Duplikate
- ungewöhnliche Preisbewegungen
- Datenqualität und Lücken
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_raw.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "data_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path=DATA_PATH, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def analyze_data(path=DATA_PATH):
    print("=" * 70)
    print("XAUUSD M1 Datenanalyse")
    print("=" * 70)

    # Load full dataset
    df = load_data(path)
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"\n1. DATENBEREICH & GRÖSSE")
    log(f"   Datei: {path}")
    log(f"   Dateigröße: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    log(f"   Anzahl Kerzen: {len(df)}")
    log(f"   Start: {df['timestamp'].iloc[0]}")
    log(f"   Ende:   {df['timestamp'].iloc[-1]}")
    log(f"   Dauer:  {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} Tage")

    log(f"\n2. SPALTEN & DATENTYPEN")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nan_count = df[col].isna().sum()
        log(f"   {col:15s} | dtype={dtype:10s} | NaNs={nan_count}")

    log(f"\n3. OHLC-STATISTIK")
    if "open" in df.columns and "close" in df.columns:
        log(f"   Open  - min={df['open'].min():.3f}  max={df['open'].max():.3f}  mean={df['open'].mean():.3f}")
        log(f"   High  - min={df['high'].min():.3f}  max={df['high'].max():.3f}  mean={df['high'].mean():.3f}")
        log(f"   Low   - min={df['low'].min():.3f}  max={df['low'].max():.3f}  mean={df['low'].mean():.3f}")
        log(f"   Close - min={df['close'].min():.3f}  max={df['close'].max():.3f}  mean={df['close'].mean():.3f}")

    if "volume" in df.columns:
        log(f"\n4. VOLUMEN-STATISTIK")
        log(f"   Volume - min={df['volume'].min():.0f}  max={df['volume'].max():.0f}  mean={df['volume'].mean():.0f}")
        zero_vol = (df["volume"] == 0).sum()
        log(f"   Zero-Volume-Balken: {zero_vol} ({zero_vol/len(df)*100:.1f}%)")

    log(f"\n5. DATENQUALITÄT")
    # Duplicates
    dup_ts = df["timestamp"].duplicated().sum()
    log(f"   Duplikat-Timestamps: {dup_ts}")

    full_row_dups = df.duplicated().sum()
    log(f"   Vollständige Duplikate: {full_row_dups}")

    # Check for sorted timestamps
    is_sorted = df["timestamp"].is_monotonic_increasing
    log(f"   Timestamps sortiert: {is_sorted}")

    # Check for missing 1-minute intervals
    expected_minutes = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60 + 1
    actual_minutes = len(df)
    missing_minutes = expected_minutes - actual_minutes
    log(f"   Erwartete Minuten: {expected_minutes:.0f}")
    log(f"   Tatsächliche Minuten: {actual_minutes}")
    log(f"   Fehlende Minuten: {missing_minutes:.0f} ({missing_minutes/expected_minutes*100:.1f}%)")

    # Find gaps > 2 minutes
    time_diffs = df["timestamp"].diff().dropna()
    gaps = time_diffs[time_diffs > timedelta(minutes=2)]
    log(f"   Lücken > 2 Minuten: {len(gaps)}")
    if len(gaps) > 0:
        log(f"   Größte Lücke: {gaps.max()}")
        # Categorize gaps
        weekend_gaps = gaps[gaps >= timedelta(hours=48)]
        log(f"   Wochenend-Lücken (~48h+): {len(weekend_gaps)}")
        short_gaps = gaps[gaps < timedelta(hours=48)]
        log(f"   Kurze Lücken (<48h): {len(short_gaps)}")

    log(f"\n6. UNGEWÖHNLICHE PREISBEWEGUNGEN")
    # Candle range
    df["candle_range"] = df["high"] - df["low"]
    log(f"   Kerben-Range (pip):")
    log(f"     min={df['candle_range'].min():.3f}  median={df['candle_range'].median():.3f}  mean={df['candle_range'].mean():.3f}")
    log(f"     max={df['candle_range'].max():.3f}")

    # Body size (as % of range)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["body_pct"] = df["body_size"] / df["candle_range"].replace(0, np.nan)
    log(f"   Body/Range Verhältnis:")
    log(f"     min={df['body_pct'].min():.3f}  median={df['body_pct'].median():.3f}  max={df['body_pct'].max():.3f}")

    # Extreme price changes (minute-over-minute)
    df["price_change"] = df["close"].diff().abs()
    pct_95 = df["price_change"].quantile(0.95)
    pct_99 = df["price_change"].quantile(0.99)
    log(f"   Minuten-Preisänderung (Change):")
    log(f"     95th pct={pct_95:.3f}  99th pct={pct_99:.3f}  max={df['price_change'].max():.3f}")

    # Count extreme jumps (> 0.5% in 1 minute)
    extreme = df[df["price_change"] / df["close"].shift(1) > 0.005]
    log(f"   Extreme Sprünge (>0.5% in 1min): {len(extreme)} ({len(extreme)/len(df)*100:.2f}%)")

    log(f"\n7. ZUSAMMENFASSUNG")
    log(f"   Zeitraum: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} Tage")
    log(f"   Kerben: {len(df):,}")
    log(f"   Datenqualität: GUT" if dup_ts == 0 and full_row_dups == 0 and missing_minutes/expected_minutes < 5 else "   Datenqualität: PRÜFEN")
    log(f"   Keine Duplikate: {'JA' if dup_ts == 0 else 'NEIN'}")
    log(f"   Preisbereich: {df['low'].min():.1f} - {df['high'].max():.1f}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nBericht gespeichert: {report_path}")

    return df


if __name__ == "__main__":
    df = analyze_data()
