"""
XAUUSD Daten von Dukascopy herunterladen.

Laedt M1-Daten fuer 2 Jahre herunter und speichert sie.
Dukascopy bietet kostenlose historische Tick-Daten.
"""

import os
import sys
import time
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_from_dukascopy():
    """Download XAUUSD data from Dukascopy using duka package."""
    from duka.app import App
    from duka.core.utils import TimeFrame

    print("Starte Download von Dukascopy...")

    app = App()

    # Download M1 data for 2 years
    start = datetime(2024, 1, 1)
    end = datetime(2026, 8, 24)

    print(f"  Zeitraum: {start.strftime('%Y-%m-%d')} bis {end.strftime('%Y-%m-%d')}")

    try:
        # Download M1 data
        print("\nLade M1 Daten...")
        app.download("XAUUSD", TimeFrame.M1, start, end)
        print("  M1 Download abgeschlossen")

        # Download M5 data
        print("\nLade M5 Daten...")
        app.download("XAUUSD", TimeFrame.M5, start, end)
        print("  M5 Download abgeschlossen")

        # Download M15 data
        print("\nLade M15 Daten...")
        app.download("XAUUSD", TimeFrame.M15, start, end)
        print("  M15 Download abgeschlossen")

    except Exception as e:
        print(f"  Fehler beim Download: {e}")
        print("  Versuche alternativen Download...")

    app.stop()


def download_ticks_and_aggregate():
    """Alternative: Ticks herunterladen und aggregieren."""
    import requests
    import gzip
    import struct
    from io import BytesIO

    print("\nAlternativer Download: Ticks von Dukascopy...")

    base_url = "https://data.dukascopy.com/datafeed/XAUUSD"

    start = datetime(2024, 1, 1)
    end = datetime(2026, 8, 24)

    all_ticks = []
    current = start

    while current <= end:
        # Dukascopy URL format: /YYYY/MM/DD/hh_ticks.bi5
        url = f"{base_url}/{current.year}/{current.month - 1:02d}/{current.day:02d}/{current.hour:02d}_ticks.bi5"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Decompress .bi5 file
                data = gzip.decompress(response.content)
                # Parse binary data (each record is 20 bytes)
                # Format: timestamp (4), ask (4), bid (4), ask_volume (4), bid_volume (4)
                offset = 0
                while offset + 20 <= len(data):
                    timestamp, ask, bid, ask_vol, bid_vol = struct.unpack_from(">IIIII", data, offset)
                    # Timestamp is in milliseconds since start of hour
                    tick_time = current + timedelta(milliseconds=timestamp)
                    all_ticks.append({
                        "timestamp": tick_time,
                        "ask": ask / 100000.0,  # Dukascopy uses 100000 divisor
                        "bid": bid / 100000.0,
                    })
                    offset += 20
        except Exception as e:
            pass

        current += timedelta(hours=1)

        # Progress
        if current.day == 1 and current.hour == 0:
            print(f"  Fortschritt: {current.strftime('%Y-%m-%d')}")

        # Rate limiting
        time.sleep(0.1)

    print(f"  {len(all_ticks)} Ticks geladen")

    # Aggregate to M5 and M15
    if all_ticks:
        df = pd.DataFrame(all_ticks)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Aggregate to M5
        df["time_5"] = df["timestamp"].dt.floor("5min")
        m5 = df.groupby("time_5").agg(
            open=("bid", "first"),
            high=("bid", "max"),
            low=("bid", "min"),
            close=("bid", "last"),
        ).reset_index()
        m5 = m5.rename(columns={"time_5": "timestamp"})
        m5_path = os.path.join(OUTPUT_DIR, "xauusd_m5_raw.csv")
        m5.to_csv(m5_path, index=False)
        print(f"  M5: {len(m5)} Kerzen gespeichert")

        # Aggregate to M15
        df["time_15"] = df["timestamp"].dt.floor("15min")
        m15 = df.groupby("time_15").agg(
            open=("bid", "first"),
            high=("bid", "max"),
            low=("bid", "min"),
            close=("bid", "last"),
        ).reset_index()
        m15 = m15.rename(columns={"time_15": "timestamp"})
        m15_path = os.path.join(OUTPUT_DIR, "xauusd_m15_raw.csv")
        m15.to_csv(m15_path, index=False)
        print(f"  M15: {len(m15)} Kerzen gespeichert")


def main():
    print("=" * 70)
    print("DUKASCOPY DOWNLOAD: XAUUSD M1/M5/M15")
    print("=" * 70)

    # Try duka first
    try:
        download_from_dukascopy()
    except Exception as e:
        print(f"  duka Download fehlgeschlagen: {e}")
        print("  Fallback: Ticks herunterladen...")
        download_ticks_and_aggregate()


if __name__ == "__main__":
    main()
