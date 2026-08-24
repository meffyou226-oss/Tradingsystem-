# XAUUSD M1 Intraday Live-Trading Bot

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

### Automatisch (empfohlen)
```cmd
run_live_trading.bat
```

Diese Datei:
1. Installiert alle Bibliotheken (numpy, pandas, xgboost, MetaTrader5, ...)
2. Trainiert das Modell (falls noch nicht vorhanden)
3. Startet den Live-Trading-Bot

### Manuell
```cmd
python -m pip install -r requirements.txt
python live_trader.py
```

## Trading-Parameter (Intraday)

| Parameter | Wert |
|-----------|------|
| Symbol | XAUUSD |
| Timeframe | M1 (1 Minute) |
| Horizon | 10 Minuten |
| TP | 500 Punkte (5.00 USD, **25 USD bei 0.05 Lot**) |
| SL | 250 Punkte (2.50 USD, 12.50 USD bei 0.05 Lot) |
| R:R | 2:1 |
| Lots | 0.05 (5 oz Gold) |
| Threshold | 0.35 (optimiert für ~6 Trades/Tag) |
| Max Trades/Tag | 15 |
| Trading Hours | 06:00-24:00 UTC |

## Modell-Leistung (OOS)

| Metrik | Wert |
|--------|------|
| AUC | 0.674 |
| Win Rate | 43.5% |
| Profit Factor | 1.52 |
| Trades/Tag | ~6 (bei Threshold 0.35) |
| EV/Trade | 0.73 USD (3.65 USD bei 0.05 Lot) |

## Dateien

| Datei | Funktion |
|-------|----------|
| `live_trader.py` | Trading-Bot: MT5-Anbindung, Signalgenerierung, Risk-Management |
| `mt5_connector.py` | MT5 API Wrapper (Verbindung, Orders, Positionen) |
| `feature_extractor.py` | Live-Feature-Berechnung (63 Features, Lookahead-Bias-frei) |
| `config.py` | Trading-Konfiguration (Parameter, Pfade, Feature-Liste) |
| `xgboost_model.pkl` | Trainiertes XGBoost-Modell |
| `run_live_trading.bat` | All-in-One Setup & Run Script |
| `requirements.txt` | Python-Abhängigkeiten |
| `live_trading.log` | Log-Datei (wird automatisch erstellt) |

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `.bat` schließt sich sofort | Verwende `python -m pip` (bereits im Skript integriert) |
| MT5-Verbindung fehlgeschlagen | MT5 öffnen, Konto verbinden, erneut versuchen |
| `MetaTrader5` nicht gefunden | `python -m pip install MetaTrader5` manuell ausführen |
| Keine Signale generiert | Threshold zu hoch - reduziere auf 0.30-0.35 |
| Modell fehlt | `run_live_trading.bat` trainiert automatisch |
