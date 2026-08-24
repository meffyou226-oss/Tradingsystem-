# XAUUSD Live Trading Bot

Automatischer Trading Bot für XAUUSD mit MetaTrader 5 Anbindung.

## Voraussetzungen

- **Windows 10/11** (MT5 läuft nur auf Windows)
- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **MetaTrader 5** ([Download](https://www.metatrader5.com/))
- Broker-Konto mit XAUUSD

## Installation

1. **Zip entpacken** in einen Ordner deiner Wahl
2. **`start_bot.bat` doppelklicken** – das installiert alles automatisch
3. **MetaTrader 5 öffnen** und einloggen
4. **XAUUSD zum Market Watch** hinzufügen (Rechtsklick → Symbols → XAUUSD)

## Starten

```
start_bot.bat
```

Oder manuell:
```bash
pip install -r requirements.txt
python src/live_bot.py
```

## Konfiguration

Datei `config/config.py`:

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| symbol | XAUUSD | Trading-Symbol |
| tp_points | 45 | Take Profit (0.45 USD) |
| sl_points | 15 | Stop Loss (0.15 USD) |
| confidence_threshold | 0.75 | Mindest-Wahrscheinlichkeit |
| max_trades_per_day | 5 | Max. Trades pro Tag |
| lot_size | 0.01 | Micro-Lot |
| spread_max | 50 | Max. Spread (5.0 USD) |
| trading_hours_start | 8 | UTC (London) |
| trading_hours_end | 20 | UTC (NY Ende) |

## Ablauf

1. Bot verbindet sich mit MT5
2. Holt M1-Daten (letzte 500 Kerzen)
3. Berechnet 63 Features
4. XGBoost Modell gibt Vorhersage (0-1)
5. Bei Confidence ≥ 0.75: BUY Order mit TP/SL
6. Max 5 Trades pro Tag, nur London+NY Session

## Ordnerstruktur

```
XAUUSD LIVE TRADING BOT/
├── src/
│   └── live_bot.py          # Haupt-Bot mit MT5-Anbindung
├── config/
│   └── config.py            # Konfiguration
├── models/
│   └── xgboost.pkl          # Trainiertes Modell
├── logs/
│   └── bot_YYYYMMDD.log     # Trading-Log
├── requirements.txt         # Python-Bibliotheken
└── start_bot.bat            # Starter (installiert + startet)
```

## Logs

Tägliche Logs werden im `logs/` Ordner gespeichert:
- Preis, Spread, Vorhersage
- Platzierte Orders
- Fehler

## Sicherheitshinweise

⚠️ **WICHTIG:**
- Erst auf **DEMO-Konto** testen!
- Max 5 Trades/Tag begrenzt das Risiko
- Spread-Filter vermeidet schlechte Conditions
- Bot handelt nur LONG (keine Shorts)
- Immer überwachen – kein vollautomatisches System!

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| "MT5 Initialisierung fehlgeschlagen" | MT5 öffnen und einloggen |
| "Symbol XAUUSD nicht gefunden" | Symbol zum Market Watch hinzufügen |
| "Keine Daten verfügbar" | Prüfe Internet-Verbindung |
| Bot startet nicht | Python-Version prüfen (3.10+) |

## Disclaimer

Trading birgt hohe Risiken. Dieser Bot ist nur zu Zwecken der Forschung.
Verluste sind möglich. Handel auf eigene Verantwortung.
