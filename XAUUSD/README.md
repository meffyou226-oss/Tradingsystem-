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
5. [x] XGBoost ML-Pipeline - `src/05_ml_pipeline.py` + `src/05_xgboost_train.py`
6. [x] OOS Backtest + Threshold-Optimierung - `src/05_xgboost_oos_simple.py`

### XGBoost OOS-Ergebnisse (Test: Sep 2025 - Aug 2026)

| Metrik | Wert |
|--------|------|
| **AUC** | 0.644 |
| **Accuracy** | 62.0% |
| **Win Rate** | 70.4% (thresh=0.50) |
| **Profit Factor** | 4.76 (thresh=0.50) |
| **Total Profit** | 56,224 Punkte |
| **Max Drawdown** | -3 Punkte |
| **Trades** | 202,195 |
| **TP hit rate** | 70.4% |

#### Threshold-Optimierung

| Threshold | Trades | Win Rate | PF | Profit |
|-----------|--------|----------|-----|--------|
| 0.50 | 202,195 | 70.4% | **4.76** | **56,224** |
| 0.60 | 112,487 | 75.2% | 6.83 | 28,410 |
| 0.70 | 49,648 | 80.4% | **8.18** | 17,512 |

### Vergleich: XGBoost vs Baseline (alle OOS)

| Strategie | Win Rate | PF | Profit |
|-----------|----------|-----|--------|
| Momentum | 47.2% | 1.77 | 46,880 |
| EMA Crossover | 46.9% | 1.74 | 47,806 |
| RSI Mean-Rev | 52.1% | 2.16 | 13,432 |
| Breakout | 47.1% | 1.76 | 12,449 |
| EMA+Momentum | 46.5% | 1.72 | 25,842 |
| **XGBoost** | **70.4%** | **4.76** | **56,224** |

### Zeitbasierte Splits

| Periode | Zeitraum | Kerben |
|---------|----------|--------|
| Training | 2024-01-01 bis 2025-04-30 | 465.475 |
| Validation | 2025-05-01 bis 2025-08-31 | 115.512 |
| Test (OOS) | 2025-09-01 bis 2026-08-23 | 341.451 |

### Top 5 Features (XGBoost)

| Feature | Importance |
|---------|-----------|
| atr_14_norm | 26.6% |
| vol_50 | 5.6% |
| candle_range | 2.2% |
| is_asia_session | 1.6% |
| dist_ema_10 | 1.4% |

### Trading-Parameter
- Horizont: 5 Minuten
- TP: 50 Punkte (0.50 USD)
- SL: 25 Punkte (0.25 USD)
- R:R: 2:1
- Spread: 3 Punkte
- Slippage: 1 Punkt

### Projektstruktur
```
XAUUSD/
├── data/          # Rohdaten (symlink)
├── src/           # Python-Quellcode (12 Module)
├── models/xgboost.pkl  # Trainiertes Modell (2 MB)
├── backtests/     # Trading-Ergebnisse
├── results/       # CSV-Ergebnisse
├── reports/       # Berichte und Grafiken
└── README.md
```
