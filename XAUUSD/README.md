XAUUSD M1 ML Research Pipeline
==============================

## Status: ✅ XGBoost OOS-validiert

### Datenbasis
- **Zeitraum:** 2024-01-01 bis 2026-08-23 (965 Tage)
- **Kerben:** 1.159.667 M1 Bars (Bid)
- **Instrument:** XAUUSD (Gold vs USD)
- **Timezone:** UTC

### Pipeline-Schritte
1. [x] Datenanalyse - `src/01_data_analysis.py`
2. [x] Feature Engineering - `src/02_feature_engineering.py` (84 Features)
3. [x] Target-Definition - `src/03_target_definition.py` (36 TP/SL Kombinationen)
4. [x] Baseline-Strategien - `src/04_baseline.py` (5 Strategien)
5. [x] XGBoost ML-Pipeline - `src/05_ml_pipeline.py`, `src/05_ml_pipeline_part2.py`
6. [x] OOS Backtest - `src/06_backtest_report.py`, `src/05_xgboost_oos_simple.py`
7. [ ] Robustheits-Tests (optional)

### Struktur
```
XAUUSD/
├── data/          # Rohdaten (symlink)
├── src/           # Python-Quellcode
├── models/        # Trainierte Modelle
├── backtests/     # Backtest-Trade-Dateien
├── results/       # CSV-Ergebnisse
├── reports/       # Berichte und Grafiken
└── README.md
```

### XGBoost OOS-Ergebnisse (Test: Sep 2025 - Aug 2026)
- **AUC:** 0.644 (signifikant über 0.5)
- **Accuracy:** 62.0%
- **Win Rate:** 70.4% (vs Baseline 47-52%)
- **Profit Factor:** 4.76 (vs Baseline 1.72-2.16)
- **Total Profit:** 56.224 Punkte
- **Max Drawdown:** -3 Punkte

### Zeitbasierte Splits
| Periode | Zeitraum | Zeilen |
|---------|----------|--------|
| Training | 2024-01-01 bis 2025-04-30 | 465.475 |
| Validation | 2025-05-01 bis 2025-08-31 | 115.512 |
| Test (OOS) | 2025-09-01 bis 2026-08-23 | 341.451 |

### Top Features (XGBoost)
1. `atr_14_norm` (26.6%) - Volatilität
2. `vol_50` (5.6%) - 50-Balken Volatilität
3. `candle_range` (2.2%) - Kerzenbreite
4. `is_asia_session` (1.6%) - Session-Filter
5. `dist_ema_10` (1.4%) - EMA-Abstand

### Trading-Parameter
- Horizont: 5 Minuten
- TP: 50 Punkte (0.50 USD)
- SL: 25 Punkte (0.25 USD)
- R:R: 2:1
- Spread: 3 Punkte
- Slippage: 1 Punkt
