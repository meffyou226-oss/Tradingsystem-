"""
Konfiguration für den Live Trading Bot (LONG + SHORT).
Optimierte M5 Swing Strategie: TP=200, SL=80, Threshold=0.55.
OOS: 6.3 Trades/Tag, 79% Winrate, PF=9
"""

TRADING_CONFIG = {
    "symbol": "XAUUSD",
    "tp_points": 200,        # OOS-optimiert (6.3 Trades/Tag, 79% Win)
    "sl_points": 80,         # Realistisch: Spread+Slippage+Puffer
    "horizon_minutes": 30,   # 6 M5-Bars
    "long_confidence_threshold": 0.55,  # Optimum aus OOS
    "short_confidence_threshold": 0.55,
    "max_trades_per_day": 10,
    "max_trades_per_direction": 6,
    "lot_size": 0.05,        # 10 USD/Trade bei TP=200
    "magic_number": 123456,
    "spread_max": 50,
    "trading_hours_start": 8,
    "trading_hours_end": 20,
    "check_interval_seconds": 60,
}

RISK_CONFIG = {
    "max_daily_loss": 100,
    "max_open_trades": 5,
    "risk_per_trade_pct": 1.0,
}
