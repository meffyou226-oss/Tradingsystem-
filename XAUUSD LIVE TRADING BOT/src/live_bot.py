"""
XAUUSD Live Trading Bot - MT5 Anbindung (LONG + SHORT, M5 Swing).

Optimierte Strategie: M5 Swing (TP=200, SL=80, H=30min).
OOS: 6.3 Trades/Tag, 79% Winrate, PF=9, 157 USD/Monat (0.05 Lots).
"""

import os
import sys
import time
import json
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Konfiguration ===
CONFIG = {
    "symbol": "XAUUSD",
    "tp_points": 200,
    "sl_points": 80,
    "horizon_minutes": 30,
    "mt5_timeframe": "M5",
    "long_model": "xgboost_m5_swing_h6_tp200_sl80.pkl",
    "short_model": "xgboost_m5_swing_short_h6_tp200_sl80.pkl",
    "long_confidence_threshold": 0.55,
    "short_confidence_threshold": 0.55,
    "max_trades_per_day": 10,
    "max_trades_per_direction": 6,
    "lot_size": 0.05,
    "magic_number": 123456,
    "spread_max": 50,
    "trading_hours_start": 8,
    "trading_hours_end": 20,
    "check_interval_seconds": 60,
}


class MT5Connection:
    def __init__(self):
        self.connected = False
        self.symbol = CONFIG["symbol"]

    def connect(self):
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 nicht verfuegbar!")
            return False
        if not mt5.initialize():
            logger.error(f"MT5 Init fehlgeschlagen: {mt5.last_error()}")
            return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            for alt in ["XAUUSD.", "XAUUSDm", "XAUUSD.raw", "XAUUSD.ecn"]:
                symbol_info = mt5.symbol_info(alt)
                if symbol_info:
                    self.symbol = alt
                    break

        if not mt5.symbol_select(self.symbol, True):
            return False

        self.connected = True
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Verbunden: {account_info.server} | Balance: {account_info.balance:.2f}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def get_current_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "spread": (tick.ask - tick.bid) / 0.01}

    def get_rates(self, timeframe="M5", count=500):
        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "timestamp", "tick_volume": "tick_volume",
                                "spread": "spread", "real_volume": "real_volume"})
        return df

    def place_order(self, order_type, lot_size, sl_points, tp_points):
        price_data = self.get_current_price()
        if price_data is None:
            return None

        symbol_info = mt5.symbol_info(self.symbol)
        point = symbol_info.point

        if order_type == "BUY":
            price = price_data["ask"]
            sl = price - sl_points * point
            tp = price + tp_points * point
        else:
            price = price_data["bid"]
            sl = price + sl_points * point
            tp = price - tp_points * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": CONFIG["magic_number"],
            "comment": f"XAUUSD_M5_SWING_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order fehlgeschlagen: {result.retcode}")
            return None

        logger.info(f"Order: {order_type} {lot_size} @ {price:.3f} | SL={sl:.3f} TP={tp:.3f}")
        return result


class FeatureCalculator:
    def __init__(self):
        self.feature_columns = [
            "candle_return", "candle_range", "body_size", "upper_wick", "lower_wick",
            "body_to_range",
            "dist_ema_5", "dist_ema_10", "dist_ema_20", "dist_ema_50", "dist_ema_100", "dist_ema_200",
            "ema_slope_20", "ema_slope_50", "ema_slope_100",
            "atr_14_norm", "vol_5", "vol_10", "vol_20", "vol_50",
            "rsi_14",
            "roc_1", "roc_3", "roc_5", "roc_10", "roc_15", "roc_20", "roc_50",
            "bb_width", "bb_position",
            "dist_high_5", "dist_low_5", "dist_high_10", "dist_low_10",
            "dist_high_20", "dist_low_20", "dist_high_50", "dist_low_50",
            "breakout_high_10", "breakout_low_10", "breakout_high_20", "breakout_low_20",
            "breakout_high_50", "breakout_low_50",
            "adx_14", "plus_di_14", "minus_di_14",
            "hour", "day_of_week", "is_london", "is_ny",
            "hour_sin", "hour_cos",
            "vol_regime", "vol_ratio",
        ]

    def calculate(self, df):
        df = df.copy()
        body = df["close"] - df["open"]
        rng = df["high"] - df["low"]
        rng_safe = rng.where(rng > 0, np.nan)

        df["candle_return"] = body / df["open"]
        df["candle_range"] = rng / df["open"]
        df["body_size"] = np.abs(body) / df["open"]
        df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
        df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
        df["body_to_range"] = np.abs(body) / rng_safe

        for p in [5, 10, 20, 50, 100, 200]:
            ema = df["close"].ewm(span=p, adjust=False).mean()
            df[f"dist_ema_{p}"] = (df["close"] - ema) / ema

        for p in [20, 50, 100]:
            ema = df["close"].ewm(span=p, adjust=False).mean()
            df[f"ema_slope_{p}"] = (ema - ema.shift(10)) / df["close"]

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        df["atr_14_norm"] = atr_14 / df["close"]

        log_ret = np.log(df["close"] / df["close"].shift(1))
        for p in [5, 10, 20, 50]:
            df[f"vol_{p}"] = log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60) * 100

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        for p in [1, 3, 5, 10, 15, 20, 50]:
            df[f"roc_{p}"] = (df["close"] - df["close"].shift(p)) / df["close"].shift(p)

        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        bb_upper = sma + 2.0 * std
        bb_lower = sma - 2.0 * std
        df["bb_width"] = (bb_upper - bb_lower) / sma
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        for p in [5, 10, 20, 50]:
            df[f"high_{p}"] = df["high"].rolling(p).max()
            df[f"low_{p}"] = df["low"].rolling(p).min()
            df[f"dist_high_{p}"] = (df["close"] - df[f"high_{p}"]) / df["close"]
            df[f"dist_low_{p}"] = (df[f"low_{p}"] - df["close"]) / df["close"]

        for p in [10, 20, 50]:
            recent_high = df["high"].rolling(p).max()
            recent_low = df["low"].rolling(p).min()
            df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int)
            df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int)

        high_diff = df["high"].diff()
        low_diff = df["low"].diff()
        tr_abs = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        minus_dm = (-low_diff).where((low_diff < 0) & (-low_diff > high_diff), 0)
        plus_di = (plus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
        minus_di = (minus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
        df["adx_14"] = dx.rolling(14).mean()
        df["plus_di_14"] = plus_di
        df["minus_di_14"] = minus_di

        ts = df["timestamp"]
        df["hour"] = ts.dt.hour.astype(float)
        df["day_of_week"] = ts.dt.dayofweek.astype(float)
        df["is_london"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
        df["is_ny"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        vol_20 = log_ret.rolling(20).std()
        vol_50 = log_ret.rolling(50).std()
        df["vol_regime"] = (vol_20 > vol_50).astype(int)
        df["vol_ratio"] = vol_20 / vol_50

        return df


class TradingBot:
    def __init__(self):
        self.mt5 = MT5Connection()
        self.feature_calc = FeatureCalculator()
        self.model_long = None
        self.model_short = None
        self.running = False
        self.daily_long_trades = 0
        self.daily_short_trades = 0
        self.last_trade_date = None

    def load_models(self):
        long_path = os.path.join(MODELS_DIR, CONFIG["long_model"])
        short_path = os.path.join(MODELS_DIR, CONFIG["short_model"])

        if not os.path.exists(long_path) or not os.path.exists(short_path):
            logger.error(f"Modelle nicht gefunden!")
            return False

        with open(long_path, "rb") as f:
            self.model_long = pickle.load(f)
        with open(short_path, "rb") as f:
            self.model_short = pickle.load(f)

        logger.info(f"Modelle geladen: M5 Swing")
        return True

    def predict(self, df):
        if self.model_long is None or self.model_short is None:
            return None, None

        feature_cols = [c for c in self.feature_calc.feature_columns if c in df.columns]
        X = df[feature_cols].values.astype(np.float32)

        if len(X) == 0 or np.any(np.isnan(X[-1:])):
            return None, None

        pred_long = self.model_long.predict_proba(X[-1:])[0, 1]
        pred_short = self.model_short.predict_proba(X[-1:])[0, 1]
        return pred_long, pred_short

    def should_trade(self, prediction, direction, current_hour, spread):
        if prediction is None:
            return False

        threshold = (CONFIG["long_confidence_threshold"] if direction == "LONG"
                     else CONFIG["short_confidence_threshold"])

        if prediction < threshold:
            return False
        if current_hour < CONFIG["trading_hours_start"] or current_hour >= CONFIG["trading_hours_end"]:
            return False
        if spread > CONFIG["spread_max"]:
            return False

        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_long_trades = 0
            self.daily_short_trades = 0
            self.last_trade_date = today

        if direction == "LONG" and self.daily_long_trades >= CONFIG["max_trades_per_direction"]:
            return False
        if direction == "SHORT" and self.daily_short_trades >= CONFIG["max_trades_per_direction"]:
            return False
        if self.daily_long_trades + self.daily_short_trades >= CONFIG["max_trades_per_day"]:
            return False

        return True

    def run(self):
        logger.info("=" * 60)
        logger.info(f"XAUUSD Bot: M5 Swing (TP=200, SL=80)")
        logger.info(f"  Lot: {CONFIG['lot_size']} | Max Trades/Tag: {CONFIG['max_trades_per_day']}")
        logger.info(f"  Threshold: {CONFIG['long_confidence_threshold']}")
        logger.info("=" * 60)

        if not self.mt5.connect():
            return
        if not self.load_models():
            return

        self.running = True
        logger.info("Bot laeuft... Ctrl+C zum Stoppen.")

        try:
            while self.running:
                df = self.mt5.get_rates(CONFIG["mt5_timeframe"], 500)
                if df is None or len(df) < 200:
                    time.sleep(10)
                    continue

                df = self.feature_calc.calculate(df)
                pred_long, pred_short = self.predict(df)

                price_data = self.mt5.get_current_price()
                if price_data is None:
                    time.sleep(5)
                    continue

                current_hour = datetime.now().hour
                spread = price_data["spread"]

                logger.info(
                    f"Preis: {price_data['bid']:.3f} | Spread: {spread:.1f} | "
                    f"LONG: {pred_long:.3f if pred_long else 'N/A'} | "
                    f"SHORT: {pred_short:.3f if pred_short else 'N/A'} | "
                    f"L:{self.daily_long_trades}/S:{self.daily_short_trades}"
                )

                if self.should_trade(pred_long, "LONG", current_hour, spread):
                    logger.info(f"LONG SIGNAL: {pred_long:.3f}")
                    result = self.mt5.place_order("BUY", CONFIG["lot_size"],
                                                   CONFIG["sl_points"], CONFIG["tp_points"])
                    if result:
                        self.daily_long_trades += 1

                if self.should_trade(pred_short, "SHORT", current_hour, spread):
                    logger.info(f"SHORT SIGNAL: {pred_short:.3f}")
                    result = self.mt5.place_order("SELL", CONFIG["lot_size"],
                                                   CONFIG["sl_points"], CONFIG["tp_points"])
                    if result:
                        self.daily_short_trades += 1

                time.sleep(CONFIG["check_interval_seconds"])

        except KeyboardInterrupt:
            logger.info("Bot gestoppt")
        except Exception as e:
            logger.error(f"Fehler: {e}", exc_info=True)
        finally:
            self.running = False
            self.mt5.disconnect()


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
