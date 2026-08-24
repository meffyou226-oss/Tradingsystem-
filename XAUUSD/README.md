XAUUSD M1 ML Research Pipeline
==============================

## Datenbasis
- Zeitraum: 2024-01-01 bis 2026-08-23 (965 Tage)
- 1.159.667 M1 Kerzen
- Instrument: XAUUSD (Gold vs USD)
- Preis-Typ: Bid
- Timezone: UTC
- Keine Duplikate, sauber sortiert
- ~16.6% fehlende Minuten (Wochenenden/Feiertage)

## Pipeline-Schritte
1. [x] Datenanalyse (`src/01_data_analysis.py`)
2. [ ] Feature Engineering (`src/02_feature_engineering.py`)
3. [ ] Target-Definition (`src/03_target_definition.py`)
4. [ ] Baseline-Strategie (`src/04_baseline.py`)
5. [ ] ML-Pipeline (`src/05_ml_pipeline.py`)
6. [ ] Backtesting (`src/06_backtest.py`)
7. [ ] Robustheits-Tests (`src/07_robustness.py`)
8. [ ] Haupt-Pipeline (`src/main.py`)

## Verzeichnisstruktur
- `data/`    - Rohdaten (symlink)
- `src/`     - Python-Quellcode
- `models/`  - Trainierte Modelle
- `backtests/` - Backtest-Ergebnisse
- `results/`   - CSV-Ergebnisse
- `reports/`   - Berichte und Grafiken
