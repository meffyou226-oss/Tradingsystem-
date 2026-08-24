"""
Konfiguration für den Live Trading Bot (LONG + SHORT).
"""

TRADING_CONFIG = {
    # Symbol (kann je nach Broker variieren: XAUUSD, XAUUSDm, XAUUSD.raw, etc.)
    "symbol": "XAUUSD",

    # Trading-Parameter (aus dem trainierten Modell)
    "tp_points": 45,       # Take Profit in Punkten (0.45 USD)
    "sl_points": 15,       # Stop Loss in Punkten (0.15 USD)
    "horizon_minutes": 5,  # Maximale Haltedauer

    # Bot-Verhalten
    "long_confidence_threshold": 0.75,   # Mindest-Wahrscheinlichkeit LONG
    "short_confidence_threshold": 0.75,  # Mindest-Wahrscheinlichkeit SHORT
    "max_trades_per_day": 5,              # Max. Trades pro Tag (total)
    "max_trades_per_direction": 3,       # Max. LONG oder SHORT pro Tag
    "lot_size": 0.01,                    # Lot-Größe (Micro-Lot)
    "magic_number": 123456,              # Magic Number für Bot-Orders

    # Filter
    "spread_max": 50,              # Max. Spread in Punkten (5.0 USD)
    "trading_hours_start": 8,      # UTC Start (London Session)
    "trading_hours_end": 20,       # UTC Ende (NY Session)

    # Intervall
    "check_interval_seconds": 60,  # Prüfe jede 60 Sekunden
}

# Risiko-Einstellungen
RISK_CONFIG = {
    "max_daily_loss": 100,      # Max. Verlust pro Tag in USD
    "max_open_trades": 3,       # Max. gleichzeitig offene Trades
    "risk_per_trade_pct": 1.0,  # Risiko pro Trade in % des Kontos
}
