"""
Live-Trading-Konfiguration für XAUUSD M1 mit MT5-Anbindung.

Konfiguriert:
- MT5 Verbindungsparameter
- Symbol und Timeframe
- Trading-Parameter (TP/SL, Spread, Horizont)
- Risiko-Management (Positionsgröße, Max Trades)
- Feature-Liste (muss mit Training übereinstimmen)
"""

import os

# === MT5-Konfiguration ===
MT5_LOGIN = None       # Setzen Sie Ihre Kontonummer hier (oder None für aktuelles Konto)
MT5_PASSWORD = None    # Setzen Sie Ihr Passwort hier
MT5_SERVER = None      # Setzen Sie Ihren Server hier (z.B. "MetaQuotes-Demo")

# === Trading-Parameter ===
SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
POINT = 0.01  # 1 Punkt = 0.01 USD für XAUUSD

# TP/SL (muss mit trainiertem Modell übereinstimmen)
TP_POINTS = 50    # Take Profit: 0.50 USD
SL_POINTS = 25    # Stop Loss: 0.25 USD
HORIZON_MINUTES = 5

# Risiko-Management
TRADE_AMOUNT = 0.01   # Lot-Größe (Standard: 0.01 Lot = 1 Micro Lot)
MAX_TRADES_PER_HOUR = 50
MAX_DRAWDOWN_USD = 500.0
STOP_TRADING_LOSS = -1000.0  # Stop-Trading bei täglichem Verlust

# ML-Parameter
THRESHOLD = 0.70  # Classification-Threshold für Signalgenerierung
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.pkl")

# === Feature-Spalten (muss mit Training übereinstimmen) ===
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

# Rolling-Perioden (müssen mit Training übereinstimmen)
EMA_PERIODS = [5, 10, 20, 50, 100, 200]
ATR_PERIOD = 14
RSI_PERIOD = 14
VOL_PERIODS = [5, 10, 20, 50]
ROC_PERIODS = [1, 3, 5, 10, 15, 20, 50]
BB_PERIOD = 20
BB_STD = 2.0
RECENT_PERIODS = [5, 10, 20, 50]
PREV_DAY_BARS = 390  # ~6.5 Stunden M1

# Logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_trading.log")
LOG_LEVEL = "INFO"

# Trading-Hours (UTC) - nur aktiv während Markt
TRADING_START_HOUR = 0   # 24/7 für XAUUSD
TRADING_END_HOUR = 24
ENABLE_SHORT_TRADING = False  # Nur Long-Positionen
