"""
Konfiguration für den Live Trading Bot (LONG + SHORT).
TP/SL optimiert durch OOS-Test mit realistischen Bedingungen.
Empfehlung: TP=45, SL=15 (PF=13.64, Win=82%, DD=-1)
SL=15 beruecksichtigt Spread (3) + Slippage (2-5) + Puffer
"""

TRADING_CONFIG = {
    "symbol": "XAUUSD",
    "tp_points": 45,        # OOS-optimiert: PF=13.64
    "sl_points": 15,        # Realistisch: Spread 3 + Slippage 5 + Puffer 7
    "horizon_minutes": 5,
    "long_confidence_threshold": 0.75,
    "short_confidence_threshold": 0.75,
    "max_trades_per_day": 5,
    "max_trades_per_direction": 3,
    "lot_size": 0.01,
    "magic_number": 123456,
    "spread_max": 50,
    "trading_hours_start": 8,
    "trading_hours_end": 20,
    "check_interval_seconds": 60,
}

RISK_CONFIG = {
    "max_daily_loss": 100,
    "max_open_trades": 3,
    "risk_per_trade_pct": 1.0,
}
