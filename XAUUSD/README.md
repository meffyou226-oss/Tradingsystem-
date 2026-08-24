# XAUUSD M1 ML Research Pipeline

## Übersicht

Maschinelles Lernen für kurzfristigen XAUUSD-Handel auf M1-Daten.
Ziel: Robuste Trading-Edges finden, nicht künstlich perfekte Backtests.

## Datenbasis

- **Zeitraum:** 2024-01-01 bis 2026-08-23 (965 Tage)
- **Kerzen:** 1,159,667 M1-Bid-Kerzen
- **Qualität:** Keine Duplikate, sauber sortiert, ~16.6% fehlende Minuten (Wochenenden/Feiertage)
- **Preisbereich:** 1,984 - 5,597 USD

## Ergebnisse

### Target-Definition
- **Horizont:** 5 Minuten
- **TP:** 50 Punkte (0.50 USD)
- **SL:** 25 Punkte (0.25 USD)
- **R:R:** 2:1
- **Label-Balance:** 49.8% / 50.2% (perfekt balanciert)

### XGBoost OOS-Ergebnisse (Threshold=0.70)

| Metrik | Wert |
|--------|------|
| AUC | 0.644 |
| Win Rate | 80.4% |
| Profit Factor | 8.18 |
| Total Profit | 17,512 Punkte |
| Max Drawdown | -2 Punkte |
| Sharpe Ratio | 713 |

### Walk-Forward-Analyse (4 Folds)

| Metrik | Ø | ± |
|--------|---|---|
| AUC | 0.638 | 0.018 |
| Profit Factor | 6.93 | 0.82 |

### Baseline-Vergleich

| Strategie | Win Rate | PF | Profit |
|-----------|----------|-----|--------|
| Momentum | 47.2% | 1.77 | 46,880 |
| EMA Crossover | 46.9% | 1.74 | 47,806 |
| RSI Mean Reversion | 52.1% | 2.16 | 13,432 |
| Breakout | 47.1% | 1.76 | 12,449 |
| EMA Momentum | 46.5% | 1.72 | 25,842 |
| **XGBoost (OOS)** | **80.4%** | **8.18** | **17,512** |

**XGBoost schlägt beste Baseline (PF 2.16) um Faktor 3.8**

### Top 10 Features

1. `atr_14_norm` - ATR-normalisierte Volatilität
2. `vol_50` - 50-Perioden-Volatilität
3. `candle_range` - Kerzen-Range
4. `is_asia_session` = Asien-Session
5. `dist_ema_10` - Abstand zu EMA-10
6. `hour_cos` - Stunde (kodierte)
7. `is_london_session` - London-Session
8. `vol_20` - 20-Perioden-Volatilität
9. `breakout_high_5` - 5-Perioden-Breakout
10. `hour` - Stunde

## Pipeline-Struktur

```
XAUUSD/
├── data/
│   ├── xauusd_m1_raw.csv -> symlink zu download
│   ├── xauusd_m1_features.csv (84 Features)
│   ├── xauusd_m1_combined.parquet (199 MB, optimiert)
│   └── targets/
│       └── xauusd_m1_targets.csv (36 TP/SL-Kombinationen)
├── src/
│   ├── main.py (Haupt-Pipeline)
│   ├── config.py (Konfiguration)
│   ├── data_preparation.py (Parquet-Konvertierung)
│   ├── backtest_engine.py (TP/SL-Simulation)
│   ├── 01_data_analysis.py
│   ├── 02_feature_engineering.py
│   ├── 03_target_definition.py
│   ├── 04_baseline.py
│   ├── 05_ml_pipeline.py
│   ├── 05_xgboost_oos.py
│   ├── 05_xgboost_oos_simple.py
│   ├── 05_ml_pipeline_part2.py
│   └── 06_backtest_report.py
├── models/
│   └── xgboost.pkl (trainiertes Modell)
├── results/
│   ├── baseline_results.csv
│   ├── feature_importance_xgboost.csv
│   ├── predictions_xgboost.csv
│   └── xgboost_oos/
│       ├── predictions_oos.csv
│       ├── threshold_optimization.csv
│       └── walkforward_xgboost.csv
├── reports/
│   ├── final_report.json
│   ├── xgboost_oos_summary.json
│   ├── xgboost_backtest_report.png
│   ├── xgboost_diagnostics.png
│   └── data_analysis/
└── README.md
```

## Verwendung

```bash
# Komplette Pipeline ausführen
python src/main.py

# Nur Report generieren (Daten bereits vorhanden)
python src/main.py --skip-data-prep --skip-ml --skip-backtest
```

## Methoden

- **Zeitbasierte Splits:** Train (2024-01 bis 2025-04), Val (2025-05 bis 2025-08), Test (2025-09 bis 2026-08)
- **Kein Lookahead-Bias:** Alle Features verwenden nur vergangene Daten
- **Realistische Backtests:** Spread (3 Punkte) + Slippage (1 Punkt) berücksichtigt
- **Walk-Forward-Analyse:** 4 Folds für Robustheitsprüfung
- **Threshold-Optimierung:** Auf Validation-Set optimiert (0.70)

## Wichtige Erkenntnisse

1. **XGBoost findet einen robusten Edge** mit OOS-PF von 8.18 und Win Rate von 80.4%
2. **Walk-Forward bestätigt Robustheit:** Ø PF=6.93 über 4 Folds
3. **Baseline-Strategien schlecht:** Beste Baseline (RSI Mean Reversion) erreicht nur PF=2.16
4. **Volatilität ist wichtigste Feature-Gruppe:** ATR, Volatilität und Kerzen-Range dominieren
5. **Session-Features relevant:** London/Asien-Session unter Top-10 Features
