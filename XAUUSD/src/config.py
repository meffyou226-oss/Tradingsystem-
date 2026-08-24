"""
Zentrale Konfiguration für die XAUUSD M1 ML-Research-Pipeline.

Enthält:
- Dateipfade
- Standard-Trading-Parameter (TP/SL, Spread, Horizont)
- Feature-Liste
- Modell-Konfigurationen
"""

import os

# === Pfade ===
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "xauusd_m1_raw.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "xauusd_m1_features.csv")
TARGETS_PATH = os.path.join(BASE_DIR, "data", "targets", "xauusd_m1_targets.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BACKTESTS_DIR = os.path.join(BASE_DIR, "backtests")

for d in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, BACKTESTS_DIR]:
    os.makedirs(d, exist_ok=True)

# === Trading-Parameter ===
POINT = 0.01  # 1 Punkt = 0.01 USD für XAUUSD

# Standard Target: 5-Minuten-Horizont, TP=50pts, SL=25pts, R:R=2:1
TARGET_PARAMS = {
    "horizon": 5,       # Minuten
    "tp_points": 50,    # Take Profit in Punkten
    "sl_points": 25,    # Stop Loss in Punkten
    "rr_ratio": 2.0,    # Risk/Reward
}

# Trading-Kosten
SPREAD_POINTS = 3.0     # Typischer XAUUSD Spread (3 Punkte = 0.03 USD)
SLIPPAGE_POINTS = 1.0   # Slippage (1 Punkt = 0.01 USD)

# === Zeitliche Splits (für Walk-Forward) ===
# Daten: Jan 2024 - Aug 2026
TRAIN_START = "2024-01-01"
TRAIN_END = "2025-04-30"
VAL_START = "2025-05-01"
VAL_END = "2025-08-31"
TEST_START = "2025-09-01"
TEST_END = "2026-08-24"

# Walk-Forward Perioden (monatlich)
WF_PERIOD_DAYS = 90  # Rollierender Zeitfenster

# === Feature-Spaltennamen ===
FEATURE_COLUMNS = [
    "candle_return", "candle_range", "body_size", "upper_wick", "lower_wick",
    "body_to_range", "wick_to_range",
    "dist_ema_5", "dist_ema_10", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
    "ema_slope_20", "ema_slope_50", "ema_slope_100",
    "atr_14_norm", "vol_5", "vol_10", "vol_20", "vol_50",
    "rsi_14",
    "roc_1", "roc_3", "roc_5", "roc_10", "roc_15", "roc_20", "roc_50",
    "bb_width", "bb_position",
    "dist_high_5", "dist_low_5", "dist_high_10", "dist_low_10",
    "dist_high_20", "dist_low_20", "dist_high_50", "dist_low_50",
    "breakout_high_5", "breakout_low_5", "breakout_high_10", "breakout_low_10",
    "breakout_high_20", "breakout_low_20", "breakout_high_50", "breakout_low_50",
    "adx_14", "plus_di_14", "minus_di_14",
    "hour", "day_of_week", "is_london_session", "is_ny_session", "is_asia_session",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "vol_regime", "vol_ratio",
    "dist_prev_high", "dist_prev_low",
]

# === Modell-Konfigurationen ===
MODELS = {
    "xgboost": {
        "class": "XGBClassifier",
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    },
    "lightgbm": {
        "class": "LGBMClassifier",
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 50,
            "random_state": 42,
            "n_jobs": -1,
        }
    },
}

# === Hauptmodell ===
PRIMARY_MODEL = "xgboost"
TARGET_COLUMN = "target_h5_sl25_rr2.0"
