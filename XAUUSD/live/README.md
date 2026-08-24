# XAUUSD M1 Live-Trading Bot

Live-Trading-Implementierung des XGBoost-Modells für XAUUSD M1-Daten mit MT5-Anbindung.

## Voraussetzungen

### System
- Windows 10/11
- MetaTrader 5 (geöffnet und mit Konto verbunden)
- Python 3.10+
- Internetzugang

### Kontovorbereitung
1. **MetaTrader 5 öffnen** und mit Ihrem Konto verbinden
2. **Symbol hinzufügen**: XAUUSD im Market Watch sichtbar machen
3. **API aktivieren**: In MT5 → Extras → Experts → Enable DLL imports & Enable ActiveX

## Schnelleinstieg

### Option A: Automatisch (empfohlen)
```cmd
run_live_trading.bat
```
Diese Datei installiert alle Bibliotheken, trainiert das Modell und startet den Bot.

### Option B: Manuell
```cmd
pip install -r requirements.txt
python live_trader.py --train    # Mit Training
python live_trader.py            # Ohne Training (nutzt xgboost_model.pkl)
```

## Dateien

| Datei | Beschreibung |
|-------|-------------|
| `live_trader.py` | Haupt-Trading-Bot (MT5-Anbindung, Signalgenerierung, Risiko-Management) |
| `mt5_connector.py` | MT5 API Wrapper (Verbindung, Order-Ausführung, Position-Management) |
| `feature_extractor.py` | Live-Feature-Berechnung (84 Features, Lookahead-Bias-frei) |
| `config.py` | Trading-Konfiguration (Parameter, Pfade, Feature-Liste) |
| `xgboost_model.pkl` | Trainiertes XGBoost-Modell (2 MB) |
| `run_live_trading.bat` | All-in-One Setup & Run Script |
| `requirements.txt` | Python-Abhängigkeiten |
| `live_trading.log` | Trading-Logdatei (wird automatisch erstellt) |

## Trading-Parameter

| Parameter | Wert |
|-----------|------|
| Symbol | XAUUSD |
| Timeframe | M1 (1 Minute) |
| TP | 50 Punkte (0.50 USD) |
| SL | 25 Punkte (0.25 USD) |
| R:R | 2:1 |
| Threshold | 0.70 (konservativ) |
| Lot-Größe | 0.01 (Micro Lot) |
| Spread | ~3 Punkte (realistisch) |

## Risiko-Management

- **Max Trades/Tag**: 1.200 (50 Std * 24h)
- **Tägliches Verlustlimit**: -1.000 USD (Stop-Trading)
- **Max Drawdown**: 500 USD
- **Position-Größe**: 0.01 Lot (1 Mikro-Lot)
- **Nur Long-Positionen** (standardmäßig)

## Modell-Leistung (OOS)

| Metrik | Wert |
|--------|------|
| AUC | 0.644 |
| Win Rate | 70.4% |
| Profit Factor | 4.76 |
| Max Drawdown | -3 Punkte |

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| MT5-Verbindung fehlgeschlagen | MT5 öffnen, Konto verbinden, erneut versuchen |
| `pip: Befehl nicht gefunden` | Python neu installieren mit "Add to PATH" |
| `xgboost_model.pkl nicht gefunden` | `python live_trader.py --train` ausführen |
| Symbol nicht sichtbar | In MT5 Market Watch → Rechtsklick → XAUUSD auswählen |
| Keine Trades | Prüfe ob MT5 mit Zählungen aktiv ist (nicht Demo-Pause) |
