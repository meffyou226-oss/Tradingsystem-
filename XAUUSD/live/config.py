"""
Live-Trading-Konfiguration (Intraday): 5-Minuten-Horizont.

Optimiert für:
- ~5 Trades/Tag (5-10)
- TP: 500 Punkte (25 USD Gewinn bei 0.05 Lot)
- SL: 200 Punkte (10 USD Verlust)
- R:R: 2.5:1
- Lots: 0.05 (5 oz Gold)

Verbessert: 5 Minuten Horizon mit starker Regularisierung.
AUC=0.71, PF=2.27, Win=47.6%, 4.7 Trades/Tag
"""

import os

# === MT5-Konfiguration ===
MT5_LOGIN = None       # Setzen Sie Ihre Kontonummer hier
MT5_PASSWORD = None    # Setzen Sie Ihr Passwort hier
MT5_SERVER = None      # Setzen Sie Ihren Server hier

# === Intraday Trading-Parameter ===
SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
POINT = 0.01  # 1 Punkt = 0.01 USD

# TP/SL (optimiert für 5-10 Trades/Tag, 5-min Horizon, AUC=0.71)
HORIZON_MINUTES = 5
TP_POINTS = 500    # Take Profit: 5.00 USD/oz (25 USD bei 0.05 Lot)
SL_POINTS = 200    # Stop Loss: 2.00 USD/oz (10 USD bei 0.05 Lot)
RR_RATIO = 2.5

# Position Sizing
LOT_SIZE = 0.05         # 0.05 Lot = 5 oz Gold
TRADE_OZ = LOT_SIZE * 100  # 5 oz
PROFIT_PER_TP = TP_POINTS * POINT * TRADE_OZ  # 25.00 USD
LOSS_PER_SL = SL_POINTS * POINT * TRADE_OZ    # 10.00 USD

# ML-Parameter
THRESHOLD = 0.25  # Optimiert für ~5 Trades/Tag (AUC=0.71, PF=2.27)
TARGET_PARAMS = {"tp_points": TP_POINTS, "sl_points": SL_POINTS, "horizon": HORIZON_MINUTES, "rr": RR_RATIO}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.pkl")

# Risiko-Management
TRADE_AMOUNT = 0.05          # Lot-Größe für Order-Execution
MAX_TRADES_PER_DAY = 15
MAX_TRADES_PER_HOUR = 3
MAX_DRAWDOWN_USD = 500.0
DAILY_LOSS_LIMIT = -1000.0
STOP_TRADING_LOSS = -2000.0

# Feature-Spalten (identisch mit Training)
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
PREV_DAY_BARS = 390

# Trading-Hours (UTC) - XAUUSD 24/7, aber nur aktiv in Hauptsessions
TRADING_START_HOUR = 6   # London Session beginnt
TRADING_END_HOUR = 24

# Logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_trading.log")
