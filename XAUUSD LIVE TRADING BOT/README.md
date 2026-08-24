# XAUUSD Live Trading Bot

Automatischer Trading Bot für XAUUSD mit MetaTrader 5 Anbindung.

## Strategien

| Strategie | TP/SL | Horizont | AUC | Avg Trade (0.05L) |
|-----------|-------|----------|-----|-------------------|
| **M5_SWING** | 200/80 | 30min | **0.651** | **10 USD** |
| M1_SCALPING | 45/15 | 5min | 0.644 | 2.25 USD |
| M15_SWING | 250/100 | 135min | 0.643 | 12.5 USD |

## Voraussetzungen

- Windows 10/11
- Python 3.10+
- MetaTrader 5
- Broker-Konto mit XAUUSD

## Installation

1. Zip entpacken
2. `start_bot.bat` doppelklicken
3. MT5 öffnen und einloggen
4. XAUUSD zum Market Watch hinzufügen

## Starten

```
start_bot.bat
```

## Konfiguration

Datei `src/live_bot.py`:
```python
STRATEGY = "M5_SWING"  # M1_SCALPING | M5_SWING | M15_SWING
CONFIG["lot_size"] = 0.05  # Lot-Größe
```

## Bot-Verhalten

- LONG + SHORT Signale
- Max 8 Trades/Tag (5 pro Richtung)
- Nur London + NY Session (8-20 UTC)
- Spread-Filter (max 50 Punkte)
- Confidence >= 0.70

## Modelle

- `xgboost.pkl` / `xgboost_short.pkl` - M1 Scalping
- `xgboost_m5_swing_h6_tp200_sl80.pkl` - M5 Swing (empfohlen)
- `xgboost_m15_swing_h9_tp250_sl100.pkl` - M15 Swing

## Sicherheitshinweise

⚠️ Erst auf DEMO-Konto testen!
- Max 8 Trades/Tag begrenzt Risiko
- Bot handelt LONG + SHORT
- Immer überwachen!

## Disclaimer

Trading birgt hohe Risiken. Bot ist nur für Forschung.
