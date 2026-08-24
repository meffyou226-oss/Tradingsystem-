"""
Live-Trading-Bot für XAUUSD M1 mit XGBoost.

Workflow:
1. Verbindung zu lokalem MT5-Terminal herstellen
2. Kontoverbindung und Symbol-Info prüfen
3. In einer Schleife:
   a. Letzte 500 M1-Bars von MT5 abrufen
   b. Features berechnen (Lookahead-Bias-frei)
   c. XGBoost-Modell Vorhersage machen
   d. Signal generieren (Threshold)
   e. Trade ausführen (Entry + TP/SL)
   f. Offene Positionen überwachen
   g. Risiko-Management prüfen

Verwendung:
    python live_trader.py                    # Live-Trading mit vorhandenem Modell
"""

import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

# Setup logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "live_trading.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger()

# Add to path
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))

import numpy as np

def load_model():
    """Lädt das trainierte XGBoost-Modell."""
    import pickle

    MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Modell nicht gefunden: {MODEL_PATH}")
        logger.info("Bitte zuerst mit run_live_trading.bat oder 'python ..\\src\\05_xgboost_train.py' trainieren.")
        return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Modell geladen: {MODEL_PATH}")
    return model


def get_feature_columns():
    """Lädt die Feature-Spalten aus config.py (src/config.py)."""
    try:
        from config import FEATURE_COLUMNS, TARGET_PARAMS, TP_POINTS, SL_POINTS, HORIZON_MINUTES, THRESHOLD, TRADE_AMOUNT, POINT, SYMBOL, MAX_TRADES_PER_HOUR, STOP_TRADING_LOSS
        return FEATURE_COLUMNS, TARGET_PARAMS, TP_POINTS, SL_POINTS, HORIZON_MINUTES, THRESHOLD, TRADE_AMOUNT, POINT, SYMBOL, MAX_TRADES_PER_HOUR, STOP_TRADING_LOSS
    except ImportError:
        # Fallback: hardcodierte Werte
        return None


def main():
    from mt5_connector import (connect_mt5, get_symbol_info, fetch_m1_data,
                                send_order, get_open_positions, close_position, shutdown_mt5)
    from feature_extractor import get_latest_features, features_to_array

    logger.info("=" * 60)
    logger.info("XAUUSD M1 Live-Trading Bot (XGBoost)")
    logger.info("=" * 60)

    # Load configuration from src/config.py
    try:
        from config import (FEATURE_COLUMNS, TP_POINTS, SL_POINTS,
                            TARGET_PARAMS, THRESHOLD, TRADE_AMOUNT, SYMBOL,
                            POINT, HORIZON_MINUTES, MAX_TRADES_PER_HOUR,
                            STOP_TRADING_LOSS, TRADING_START_HOUR, TRADING_END_HOUR)
        logger.info(f"Konfiguration geladen aus src/config.py")
    except ImportError:
        logger.error("Kann src/config.py nicht laden. Bitte von XAUUSD/ ausführen.")
        return

    # 1. Load model
    model = load_model()
    if model is None:
        sys.exit(1)

    logger.info(f"Trading Parameter: TP={TP_POINTS}pts, SL={SL_POINTS}pts, H={HORIZON_MINUTES}min")
    logger.info(f"Threshold: {THRESHOLD}, Lot: {TRADE_AMOUNT}, Symbol: {SYMBOL}")

    # 2. Connect to MT5
    logger.info("Verbinde mit MT5-Terminal...")
    mt5 = connect_mt5()
    if mt5 is None:
        logger.error("MT5-Verbindung fehlgeschlagen. Bitte MT5 öffnen und erneut versuchen.")
        sys.exit(1)

    # 3. Verify symbol
    info = get_symbol_info(mt5, SYMBOL)
    if info is None:
        logger.error(f"Kann Symbol {SYMBOL} nicht abrufen. Bitte im MT5 Market Watch hinzufügen.")
        shutdown_mt5(mt5)
        sys.exit(1)

    logger.info(f"Symbol: {SYMBOL} | Bid={info.bid:.3f} Ask={info.ask:.3f} | "
                f"Spread={info.spread} | Digits={info.digits}")

    # 4. Trading loop
    class State:
        def __init__(self):
            self.active = False
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.start_day = datetime.utcnow().day

    state = State()

    logger.info("Trading gestartet. Strg+C zum Stoppen.")

    try:
        while True:
            now = datetime.utcnow()

            # Reset daily counter
            if now.day != state.start_day:
                state.trades_today = 0
                state.daily_pnl = 0.0
                state.start_day = now.day
                logger.info("Tageszähler zurückgesetzt.")

            # Check trading hours
            if not (TRADING_START_HOUR <= now.hour < TRADING_END_HOUR):
                time.sleep(60)
                continue

            # Risk management
            if state.trades_today > MAX_TRADES_PER_HOUR * 24:
                logger.error("Max Trades Limit erreicht!")
                break
            if state.daily_pnl < STOP_TRADING_LOSS:
                logger.error(f"Täglicher Verlust Limit erreicht: {state.daily_pnl:.2f}")
                break

            if state.active:
                # Check if position still open
                positions = get_open_positions(mt5, SYMBOL)
                if len(positions) == 0:
                    logger.info("Position wurde geschlossen (TP/SL erreicht)")
                    state.active = False
                else:
                    pos = positions[0]
                    state.daily_pnl = pos.pnl
                    logger.info(f"Position offen: P&L={pos.pnl:.2f}USD")
            else:
                # Generate signal
                df = fetch_m1_data(mt5, SYMBOL, n_bars=500)
                features = get_latest_features(df)

                if features is not None:
                    feature_arr = features_to_array(features, FEATURE_COLUMNS)
                    proba = model.predict_proba(feature_arr.reshape(1, -))[:, 1][0]

                    if proba >= THRESHOLD:
                        logger.info(f"Signal: {proba:.4f} >= {THRESHOLD} -> LONG")
                        entry_price = info.ask
                        tp = entry_price + TP_POINTS * POINT
                        sl = entry_price - SL_POINTS * POINT
                        result = send_order(mt5, SYMBOL, "BUY", TRADE_AMOUNT,
                                            entry_price, SL_POINTS, TP_POINTS, POINT)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            state.active = True
                            state.trades_today += 1
                            logger.info(f"Trade eröffnet: Ticket#{result.order} {TRADE_AMOUNT}lot")
                    else:
                        logger.info(f"Kein Signal: {proba:.4f} < {THRESHOLD}")

            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        logger.info("Durch Benutzer gestoppt (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close positions
        if state.active:
            for pos in get_open_positions(mt5, SYMBOL):
                close_position(mt5, pos)
        shutdown_mt5(mt5)
        logger.info("Trading Bot gestoppt.")


if __name__ == "__main__":
    main()
