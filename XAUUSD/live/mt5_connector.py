"""
MT5-Verbindungs-Manager für Live-Trading.

Verbindet sich mit dem lokal laufenden MetaTrader 5 Terminal.
Stellt Funktionen für:
- Verbindungsaufbau/abbau
- Symbol-Informationen
- Preisdaten (M1 Candles)
- Order-Ausführung (Entry, TP, SL)
- Position-Management
"""

import time
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def connect_mt5(login=None, password=None, server=None, retry=3, retry_delay=5):
    """
    Verbindet mit MT5-Terminal.
    
    Args:
        login: Kontonummer (None = verwende aktuelles Konto)
        password: Passwort
        server: Servername
        retry: Anzahl der Verbindungsversuche
        retry_delay: Wartezeit zwischen Versuchen (Sekunden)
    
    Returns:
        MT5-Modul oder None bei Fehler
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5-Paket nicht installiert. Bitte 'pip install MetaTrader5' ausführen.")
        return None

    for attempt in range(retry):
        try:
            authorized = mt5.initialize(
                login=login,
                password=password,
                server=server,
                pythonic=True
            ) if login and password and server else mt5.initialize(pythonic=True)

            if authorized:
                logger.info(f"MT5-Verbindung hergestellt (Versuch {attempt+1}/{retry})")
                logger.info(f"  MT5 Version: {mt5.version()}")
                logger.info(f"  Terminal: {mt5.terminal_info()}")
                return mt5
            else:
                err = mt5.last_error()
                logger.warning(f"MT5-Verbindung fehlgeschlagen (Versuch {attempt+1}): {err}")
                mt5.shutdown()
        except Exception as e:
            logger.error(f"MT5-Verbindungsfehler (Versuch {attempt+1}): {e}")

        if attempt < retry - 1:
            time.sleep(retry_delay)

    return None


def get_symbol_info(mt5, symbol):
    """Ruft Symbol-Informationen ab."""
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        # Symbol nicht sichtbar -> zum Markt hinzufügen
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Kann Symbol {symbol} nicht selektieren: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
    return info


def fetch_m1_data(mt5, symbol, n_bars=500):
    """
    Holt letzte n M1-Candles für das angegebene Symbol.
    
    Returns:
        DataFrame mit Spalten: timestamp, open, high, low, close
    """
    import pandas as pd

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise Exception(f"Keine Daten für {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df.columns = ["timestamp", "open", "high", "low", "close"]
    return df


def fetch_m1_data_by_time(mt5, symbol, from_time, to_time):
    """
    Holt M1-Daten für einen angegebenen Zeitraum.
    """
    import pandas as pd

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_time, to_time)
    if rates is None or len(rates) == 0:
        raise Exception(f"Keine Daten für {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df.columns = ["timestamp", "open", "high", "low", "close"]
    return df


def send_order(mt5, symbol, order_type, volume, price, sl_points, tp_points,
               point_value, dev_distance=10):
    """
    Sendet eine Market-Order an MT5.
    
    Args:
        mt5: MT5-Modul
        symbol: Trading-Symbol
        order_type: "BUY" oder "SELL"
        volume: Lot-Größe
        price: Entry-Preis (wird ignoriert für Market-Orders)
        sl_points: Stop-Loss in Punkten
        tp_points: Take-Profit in Punkten
        point_value: Wert eines Punkts
        dev_distance: max Deviation in Punkten
    
    Returns:
        Order-Result-Dict
    """
    info = get_symbol_info(mt5, symbol)

    if order_type == "BUY":
        order_type_val = mt5.ORDER_TYPE_BUY
        sl_price = price - sl_points * point_value
        tp_price = price + tp_points * point_value
    else:
        order_type_val = mt5.ORDER_TYPE_SELL
        sl_price = price + sl_points * point_value
        tp_price = price - tp_points * point_value

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type_val,
        "price": price,
        "sl": round(sl_price, 3),
        "tp": round(tp_price, 3),
        "deviation": dev_distance,
        "magic": 20240824,
        "comment": "XGBoost_AutoTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error() if result is None else f"retcode={result.retcode}"
        logger.error(f"Order fehlgeschlagen: {err}")
        if result:
            logger.error(f"  {result.comment}")
    else:
        logger.info(f"Order ausgeführt: {order_type} {volume} {symbol} "
                     f"Entry={result.price:.3f} SL={sl_price:.3f} TP={tp_price:.3f}")

    return result


def close_position(mt5, position, dev_distance=10):
    """Schließt eine offene Position."""
    symbol = position.symbol
    volume = position.volume
    order_type = position.type  # POSITION_TYPE_BUY or POSITION_TYPE_SELL

    info = get_symbol_info(mt5, symbol)
    if order_type == 1:  # BUY
        close_type = mt5.ORDER_TYPE_SELL
        price = info.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = info.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": symbol,
        "volume": float(volume),
        "type": close_type,
        "price": price,
        "deviation": dev_distance,
        "magic": 20240824,
        "comment": "XGBoost_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Close fehlgeschlagen: {result.comment if result else mt5.last_error()}")
    else:
        logger.info(f"Position geschlossen: {symbol} Profit={result.profit:.2f}")

    return result


def get_open_positions(mt5, symbol=None):
    """Ruft offene Positionen ab."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return positions if positions else []


def shutdown_mt5(mt5):
    """Trennt die MT5-Verbindung."""
    if mt5:
        mt5.shutdown()
        logger.info("MT5-Verbindung getrennt.")
