"""
XAUUSD Live Trading Bot - MT5 Anbindung (LONG + SHORT).

Verbindet sich direkt mit MetaTrader 5, holt Echtzeitdaten,
berechnet Features und gibt LONG und SHORT Trading-Signale.
"""

import os
import sys
import time
import json
import pickle
import logging
from datetime import datetime, timedelta
from threading import Thread, Event

import numpy as np
import pandas as pd

# MT5 Import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNUNG: MetaTrader5 Paket nicht installiert!")

# === Pfade ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)  # Gehe ein Verzeichnis hoch
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

os.makedirs(LOGS_DIR, exist_ok=True)

# Logging
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
    "tp_points": 45,
    "sl_points": 15,
    "horizon_minutes": 5,
    "long_confidence_threshold": 0.75,
    "short_confidence_threshold": 0.75,
    "max_trades_per_day": 5,
    "max_trades_per_direction": 3,  # Max 3 LONG und 3 SHORT pro Tag
    "lot_size": 0.01,
    "magic_number": 123456,
    "spread_max": 50,
    "trading_hours_start": 8,
    "trading_hours_end": 20,
    "check_interval_seconds": 60,
}


class MT5Connection:
    """MetaTrader 5 Verbindungsklasse."""

    def __init__(self):
        self.connected = False
        self.symbol = CONFIG["symbol"]

    def connect(self):
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 Paket nicht verfuegbar!")
            return False

        if not mt5.initialize():
            logger.error(f"MT5 Initialisierung fehlgeschlagen: {mt5.last_error()}")
            return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            for alt in ["XAUUSD.", "XAUUSDm", "XAUUSD.raw", "XAUUSD.ecn"]:
                symbol_info = mt5.symbol_info(alt)
                if symbol_info:
                    self.symbol = alt
                    logger.info(f"Alternative Symbol gefunden: {alt}")
                    break

        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"Symbol konnte nicht ausgewaehlt werden!")
            return False

        self.connected = True
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Verbunden: {account_info.server} | "
                       f"Account: {account_info.login} | "
                       f"Balance: {account_info.balance:.2f} {account_info.currency}")
        return True

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 Verbindung getrennt.")

    def get_current_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": (tick.ask - tick.bid) / 0.01,
            "time": datetime.fromtimestamp(tick.time, tz=None),
        }

    def get_rates(self, count=500):
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "time": "timestamp", "tick_volume": "tick_volume",
            "spread": "spread", "real_volume": "real_volume",
        })
        return df

    def place_order(self, order_type, lot_size, sl_points, tp_points):
        price_data = self.get_current_price()
        if price_data is None:
            return None

        symbol_info = mt5.symbol_info(self.symbol)
        point = symbol_info.point

        if order_type == "BUY":
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = price_data["ask"]
            sl = price - sl_points * point
            tp = price + tp_points * point
        else:
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = price_data["bid"]
            sl = price + sl_points * point
            tp = price - tp_points * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": CONFIG["magic_number"],
            "comment": f"XAUUSD_ML_{order_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order fehlgeschlagen: {result.retcode} - {result.comment}")
            return None

        logger.info(f"Order: {order_type} {lot_size} @ {price:.3f} | SL={sl:.3f} TP={tp:.3f}")
        return result

    def get_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return positions if positions else []


class FeatureCalculator:
    """Berechnet Features aus M1-Daten."""

    def __init__(self):
        self.feature_columns = [
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

    def calculate(self, df):
        df = df.copy()
        body = df["close"] - df["open"]
        rng = df["high"] - df["low"]
        rng_safe = rng.where(rng > 0, np.nan)

        df["candle_return"] = (body / df["open"].shift(1)).shift(1)
        df["candle_range"] = (rng / df["open"].shift(1)).shift(1)
        df["body_size"] = (np.abs(body) / df["open"].shift(1)).shift(1)
        df["upper_wick"] = ((df["high"] - np.maximum(df["open"], df["close"])) / df["open"].shift(1)).shift(1)
        df["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / df["open"].shift(1)).shift(1)
        df["body_to_range"] = (np.abs(body) / rng_safe).shift(1)
        df["wick_to_range"] = ((rng_safe - np.abs(body)) / rng_safe.replace(0, np.nan)).shift(1)

        for p in [5, 10, 20, 50, 100, 200]:
            ema = df["close"].ewm(span=p, adjust=False).mean()
            df[f"dist_ema_{p}"] = ((df["close"] - ema) / ema).shift(1)

        for p in [20, 50, 100]:
            ema = df["close"].ewm(span=p, adjust=False).mean()
            df[f"ema_slope_{p}"] = ((ema - ema.shift(10)) / df["close"].shift(1)).shift(1)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        df["atr_14_norm"] = (atr_14 / df["close"].shift(1)).shift(1)

        log_ret = np.log(df["close"] / df["close"].shift(1))
        for p in [5, 10, 20, 50]:
            df[f"vol_{p}"] = (log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60)).shift(1) * 100

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - (100 / (1 + rs))).shift(1)

        for p in [1, 3, 5, 10, 15, 20, 50]:
            df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)).shift(1)

        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        bb_upper = sma + 2.0 * std
        bb_lower = sma - 2.0 * std
        df["bb_width"] = ((bb_upper - bb_lower) / sma).shift(1)
        df["bb_position"] = ((df["close"] - bb_lower) / (bb_upper - bb_lower)).shift(1)

        for p in [5, 10, 20, 50]:
            df[f"high_{p}"] = df["high"].shift(1).rolling(p).max()
            df[f"low_{p}"] = df["low"].shift(1).rolling(p).min()
            df[f"dist_high_{p}"] = ((df["close"] - df[f"high_{p}"]) / df["close"].shift(1)).shift(1)
            df[f"dist_low_{p}"] = ((df[f"low_{p}"] - df["close"]) / df["close"].shift(1)).shift(1)

        for p in [5, 10, 20, 50]:
            recent_high = df["high"].shift(1).rolling(p).max()
            recent_low = df["low"].shift(1).rolling(p).min()
            df[f"breakout_high_{p}"] = (df["close"] > recent_high).astype(int).shift(1)
            df[f"breakout_low_{p}"] = (df["close"] < recent_low).astype(int).shift(1)

        high_diff = df["high"].diff()
        low_diff = df["low"].diff()
        tr_abs = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        minus_dm = (-low_diff).where((low_diff < 0) & (-low_diff > high_diff), 0)
        plus_di = (plus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
        minus_di = (minus_dm.rolling(14).sum() / tr_abs.rolling(14).sum() * 100)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
        df["adx_14"] = dx.rolling(14).mean().shift(1)
        df["plus_di_14"] = plus_di.shift(1)
        df["minus_di_14"] = minus_di.shift(1)

        ts = df["timestamp"]
        df["hour"] = ts.dt.hour.astype(float)
        df["day_of_week"] = ts.dt.dayofweek.astype(float)
        df["is_london_session"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int)
        df["is_ny_session"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int)
        df["is_asia_session"] = ((ts.dt.hour >= 0) & (ts.dt.hour < 8)).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        vol_20 = log_ret.rolling(20).std()
        vol_50 = log_ret.rolling(50).std()
        df["vol_regime"] = (vol_20 > vol_50).astype(int).shift(1)
        df["vol_ratio"] = (vol_20 / vol_50).shift(1)

        df["prev_day_high"] = df["high"].shift(1).rolling(390).max()
        df["prev_day_low"] = df["low"].shift(1).rolling(390).min()
        df["dist_prev_high"] = ((df["close"] - df["prev_day_high"]) / df["close"].shift(1)).shift(1)
        df["dist_prev_low"] = ((df["close"] - df["prev_day_low"]) / df["close"].shift(1)).shift(1)

        return df


class TradingBot:
    """Haupt-Trading-Bot mit LONG + SHORT."""

    def __init__(self):
        self.mt5 = MT5Connection()
        self.feature_calc = FeatureCalculator()
        self.model_long = None
        self.model_short = None
        self.running = False
        self.daily_long_trades = 0
        self.daily_short_trades = 0
        self.last_trade_date = None
        self.stop_event = Event()

    def load_models(self):
        """Lädt LONG und SHORT Modell."""
        long_path = os.path.join(MODELS_DIR, "xgboost.pkl")
        short_path = os.path.join(MODELS_DIR, "xgboost_short.pkl")

        if not os.path.exists(long_path):
            logger.error(f"LONG Modell nicht gefunden: {long_path}")
            return False
        if not os.path.exists(short_path):
            logger.error(f"SHORT Modell nicht gefunden: {short_path}")
            return False

        with open(long_path, "rb") as f:
            self.model_long = pickle.load(f)
        with open(short_path, "rb") as f:
            self.model_short = pickle.load(f)

        logger.info("Modelle geladen: xgboost.pkl (LONG) + xgboost_short.pkl (SHORT)")
        return True

    def predict(self, df):
        """Gibt LONG und SHORT Vorhersagen zurück."""
        if self.model_long is None or self.model_short is None:
            return None, None

        feature_cols = [c for c in self.feature_calc.feature_columns if c in df.columns]
        X = df[feature_cols].values.astype(np.float32)

        if len(X) == 0:
            return None, None

        X_last = X[-1:]
        if np.any(np.isnan(X_last)):
            return None, None

        pred_long = self.model_long.predict_proba(X_last)[0, 1]
        pred_short = self.model_short.predict_proba(X_last)[0, 1]
        return pred_long, pred_short

    def should_trade(self, prediction, direction, current_hour, spread):
        """Prüft ob Trade erlaubt ist."""
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

        # Tähler resetten
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_long_trades = 0
            self.daily_short_trades = 0
            self.last_trade_date = today

        # Max Trades pro Richtung
        if direction == "LONG" and self.daily_long_trades >= CONFIG["max_trades_per_direction"]:
            return False
        if direction == "SHORT" and self.daily_short_trades >= CONFIG["max_trades_per_direction"]:
            return False

        # Max Trades total
        if self.daily_long_trades + self.daily_short_trades >= CONFIG["max_trades_per_day"]:
            return False

        return True

    def run(self):
        """Hauptloop."""
        logger.info("=" * 60)
        logger.info("XAUUSD Live Trading Bot (LONG + SHORT)")
        logger.info("=" * 60)

        if not self.mt5.connect():
            logger.error("Keine MT5-Verbindung!")
            return

        if not self.load_models():
            logger.error("Modelle konnten nicht geladen werden!")
            return

        self.running = True
        logger.info("Bot laeuft... Ctrl+C zum Stoppen.")

        try:
            while not self.stop_event.is_set():
                df = self.mt5.get_rates(count=500)
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

                # LONG Signal
                if self.should_trade(pred_long, "LONG", current_hour, spread):
                    logger.info(f"LONG SIGNAL: Confidence={pred_long:.3f}")
                    result = self.mt5.place_order("BUY", CONFIG["lot_size"],
                                                   CONFIG["sl_points"], CONFIG["tp_points"])
                    if result:
                        self.daily_long_trades += 1
                        logger.info(f"BUY platziert! L:{self.daily_long_trades}")

                # SHORT Signal
                if self.should_trade(pred_short, "SHORT", current_hour, spread):
                    logger.info(f"SHORT SIGNAL: Confidence={pred_short:.3f}")
                    result = self.mt5.place_order("SELL", CONFIG["lot_size"],
                                                   CONFIG["sl_points"], CONFIG["tp_points"])
                    if result:
                        self.daily_short_trades += 1
                        logger.info(f"SELL platziert! S:{self.daily_short_trades}")

                time.sleep(CONFIG["check_interval_seconds"])

        except KeyboardInterrupt:
            logger.info("Bot gestoppt (Ctrl+C)")
        except Exception as e:
            logger.error(f"Fehler: {e}", exc_info=True)
        finally:
            self.running = False
            self.mt5.disconnect()
            logger.info("Bot beendet.")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
