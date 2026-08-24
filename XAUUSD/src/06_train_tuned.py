"""
Verbessertes XGBoost-Training mit Hyperparameter-Tuning.

Optimiert für:
- 10-Minuten-Horizon (Intraday, kein Scalping)
- 5-10 Trades/Tag
- Höhere AUC durch Grid Search über Hyperparameter
"""

import os, sys, time, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

# === Pfade ===
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "download", "xauusd-m1-bid-2024-01-01-2026-08-24T11-58.csv")
MODEL_DIR = "XAUUSD/models"
REPORTS_DIR = "XAUUSD/reports"
RESULTS_DIR = "XAUUSD/results"
LIVE_DIR = "XAUUSD/live"

for d in [MODEL_DIR, REPORTS_DIR, RESULTS_DIR, LIVE_DIR, "XAUUSD/backtests", "XAUUSD/data/targets"]:
    os.makedirs(d, exist_ok=True)

# === Parameter ===
POINT = 0.01
HORIZON = 10
TP = 500
SL = 250
RR = 2.0
LOTS = 0.05
TRADE_OZ = 5  # 0.05 * 100
SPREAD = 3.0
SLIPPAGE = 1.0

TRAIN_START = "2024-01-01"
TRAIN_END = "2025-06-30"
VAL_START = "2025-07-01"
VAL_END = "2025-09-30"
TEST_START = "2025-10-01"
TEST_END = "2026-08-24"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_all_features(df):
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    rng_safe = rng.where(rng > 0, np.nan)
    sv = df["open"].shift(1)

    df["candle_return"] = (body / sv).shift(1)
    df["candle_range"] = (rng / sv).shift(1)
    df["body_size"] = (np.abs(body) / sv).shift(1)
    df["upper_wick"] = ((df["high"] - np.maximum(df["open"], df["close"])) / sv).shift(1)
    df["lower_wick"] = ((np.minimum(df["open"], df["close"]) - df["low"]) / sv).shift(1)
    df["body_to_range"] = (np.abs(body) / rng_safe).shift(1)
    df["wick_to_range"] = ((rng_safe - np.abs(body)) / rng_safe.replace(0, np.nan)).shift(1)

    for p in [5, 10, 20, 50, 100, 200]:
        ema = df["close"].ewm(span=p, adjust=False).mean()
        df[f"dist_ema_{p}"] = ((df["close"] - ema) / ema).shift(1)
    for p in [20, 50, 100]:
        er = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema_slope_{p}"] = (er - er.shift(10)).shift(1)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14_norm"] = (tr.rolling(14).mean() / df["close"].shift(1)).shift(1)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    for p in [5, 10, 20, 50]:
        df[f"vol_{p}"] = (log_ret.rolling(p).std() * np.sqrt(365 * 24 * 60)).shift(1) * 100

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).shift(1)

    for p in [1, 3, 5, 10, 15, 20, 50]:
        df[f"roc_{p}"] = ((df["close"] - df["close"].shift(p)) / df["close"].shift(p)).shift(1)

    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    df["bb_width"] = ((bb_upper - bb_lower) / sma).shift(1)
    df["bb_position"] = ((df["close"] - bb_lower) / (bb_upper - bb_lower)).shift(1)

    for p in [5, 10, 20, 50]:
        rh = df["high"].shift(1).rolling(p).max()
        rl = df["low"].shift(1).rolling(p).min()
        df[f"dist_high_{p}"] = ((df["high"] - rh) / df["close"].shift(1)).shift(1)
        df[f"dist_low_{p}"] = ((rl - df["low"]) / df["close"].shift(1)).shift(1)
        df[f"breakout_high_{p}"] = (df["close"] > rh).astype(int).shift(1)
        df[f"breakout_low_{p}"] = (df["close"] < rl).astype(int).shift(1)

    hd = df["high"].diff()
    ld = df["low"].diff()
    tr_combined = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    plus_di = hd.where(hd > 0, 0).rolling(14).sum() / tr_combined.rolling(14).sum() * 100
    minus_di = ld.where(ld < 0, 0).abs().rolling(14).sum() / tr_combined.rolling(14).sum() * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di.replace(0, np.nan))).replace([np.inf, -np.inf], 0) * 100
    df["adx_14"] = dx.rolling(14).mean().shift(1)
    df["plus_di_14"] = plus_di.shift(1)
    df["minus_di_14"] = minus_di.shift(1)

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype(float)
    df["day_of_week"] = ts.dt.dayofweek.astype(float)
    df["is_london_session"] = ((ts.dt.hour >= 8) & (ts.dt.hour < 16)).astype(int).shift(1)
    df["is_ny_session"] = ((ts.dt.hour >= 13) & (ts.dt.hour < 20)).astype(int).shift(1)
    df["is_asia_session"] = ((ts.dt.hour >= 0) & (ts.dt.hour < 8)).astype(int).shift(1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).shift(1)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).shift(1)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).shift(1)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).shift(1)

    vol_20 = log_ret.rolling(20).std()
    vol_50 = log_ret.rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int).shift(1)
    df["vol_ratio"] = (vol_20 / vol_50).shift(1)

    df["prev_day_high"] = df["high"].shift(1).rolling(390).max()
    df["prev_day_low"] = df["low"].shift(1).rolling(390).min()
    df["dist_prev_high"] = ((df["close"] - df["prev_day_high"]) / df["close"].shift(1)).shift(1)
    df["dist_prev_low"] = ((df["close"] - df["prev_day_low"]) / df["close"].shift(1)).shift(1)
    return df


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


def compute_target(close, high, low, horizon, tp_pts, sl_pts, point=POINT):
    n = len(close)
    targets = np.zeros(n, dtype=np.float32)
    tp_level = close + tp_pts * point
    sl_level = close - sl_pts * point

    for offset in range(1, horizon + 1):
        if offset >= n:
            break
        future_idx = np.arange(offset, n)
        current_idx = np.arange(0, n - offset)
        undetermined = targets[current_idx] == 0

        tp_hits = high[future_idx] >= tp_level[current_idx]
        sl_hits = low[future_idx] <= sl_level[current_idx]

        new_tp = tp_hits & undetermined & (targets[current_idx] != 2)
        new_sl = sl_hits & undetermined & (targets[current_idx] != 1)

        targets[current_idx[new_tp]] = 1
        targets[current_idx[new_sl]] = 2

    targets[targets == 2] = 0
    return targets


def run_backtest(df, signals, tp_pts, sl_pts, horizon):
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(df)
    trades = []

    for i in range(n):
        if signals[i]:
            if i + horizon >= n:
                continue
            entry = close[i] + (SPREAD + SLIPPAGE) * POINT
            tp_lvl = entry + tp_pts * POINT
            sl_lvl = entry - sl_pts * POINT
            fh = high[i+1:min(i+horizon+1, n)]
            fl = low[i+1:min(i+horizon+1, n)]
            if len(fh) == 0:
                continue
            tp_idx = np.argmax(fh >= tp_lvl) if np.any(fh >= tp_lvl) else None
            sl_idx = np.argmax(fl <= sl_lvl) if np.any(fl <= sl_lvl) else None
            tp_pos = tp_idx if tp_idx is not None else len(fh)
            sl_pos = sl_idx if sl_idx is not None else len(fh)
            if tp_pos <= sl_pos and tp_idx is not None:
                pnl = tp_pts * POINT
                reason = "tp"
            elif sl_idx is not None:
                pnl = -(sl_pts * POINT)
                reason = "sl"
            else:
                pnl = close[min(i+horizon, n-1)] - SPREAD * POINT - entry
                reason = "expiry"
            trades.append({"entry_time": df["timestamp"].iloc[i], "pnl": pnl, "exit_reason": reason})

    if not trades:
        return pd.DataFrame(), {}

    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    wins = tdf[tdf["pnl"] > 0]

    total = tdf["pnl"].sum() * TRADE_OZ
    gw = wins["pnl"].sum() * TRADE_OZ if len(wins) > 0 else 0
    gl = abs(tdf[tdf["pnl"] < 0]["pnl"].sum()) * TRADE_OZ if len(tdf[tdf["pnl"] < 0]) > 0 else 0
    pf = gw / gl if gl > 0 else float('inf')

    equity = tdf["pnl"].cumsum() * TRADE_OZ
    max_dd = (equity - equity.expanding().max()).min()

    return tdf, {
        "n_trades": n_trades,
        "win_rate": len(wins) / n_trades,
        "profit_factor": pf,
        "total_profit": total,
        "max_drawdown": max_dd,
        "tp_hits": (tdf["exit_reason"] == "tp").sum(),
        "sl_hits": (tdf["exit_reason"] == "sl").sum(),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("Verbessertes XGBoost-Training (Hyperparameter-Tuning)")
    print(f"  Horizon: {HORIZON} Min | TP={TP}pts | SL={SL}pts | Lots={LOTS}")
    print("=" * 70)

    df = load_data()
    print(f"Daten: {len(df)} Kerzen ({df['timestamp'].min()} - {df['timestamp'].max()})")

    print("\nBerechne Features...")
    df = compute_all_features(df)

    print(f"Berechne Target: target_h{HORIZON}_sl{SL}_rr{RR}")
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    df["target"] = compute_target(close, high, low, HORIZON, TP, SL)
    print(f"  Balance: {df['target'].mean():.2%}")

    df = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)

    train_m = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_m = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_m = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    X_train = df.loc[train_m, FEATURE_COLUMNS].values.astype(np.float32)
    y_train = df.loc[train_m, "target"].values.astype(np.float32)
    X_val = df.loc[val_m, FEATURE_COLUMNS].values.astype(np.float32)
    y_val = df.loc[val_m, "target"].values.astype(np.float32)
    X_test = df.loc[test_m, FEATURE_COLUMNS].values.astype(np.float32)
    y_test = df.loc[test_m, "target"].values.astype(np.float32)

    val_df = df[val_m].reset_index(drop=True)
    test_df = df[test_m].reset_index(drop=True)
    n_test_days = (pd.to_datetime(TEST_END) - pd.to_datetime(TEST_START)).days + 1

    print(f"\nSplit: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} ({n_test_days}d)")

    # Hyperparameter grid
    param_grid = [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 1000},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1000},
        {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 2000},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 1000},
        {"max_depth": 7, "learning_rate": 0.03, "n_estimators": 1000},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 500},
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500},
    ]

    base_params = dict(
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=50
    )

    best_model = None
    best_auc = 0
    best_threshold = 0.30
    best_stats = None
    best_config = None

    print("\n--- Hyperparameter-Tuning ---")
    for params in param_grid:
        cfg = {**base_params, **params}
        t1 = time.time()
        model = XGBClassifier(**cfg)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Find best threshold for 5-10 trades/day
        best_thresh = 0.30
        best_pf = 0
        for thresh in np.arange(0.25, min(y_val_proba.max(), 0.6), 0.02):
            signals = y_val_proba >= thresh
            _, stats = run_backtest(val_df, signals, TP, SL, HORIZON)
            if stats and stats["n_trades"] > 0:
                tday = stats["n_trades"] / (len(val_df) / (24 * 50))
                if 2 <= tday <= 10 and stats["profit_factor"] > best_pf:
                    best_pf = stats["profit_factor"]
                    best_thresh = thresh

        # Evaluate on test
        signals = y_test_proba >= best_thresh
        _, test_stats = run_backtest(test_df, signals, TP, SL, HORIZON)
        tday = test_stats["n_trades"] / n_test_days if test_stats else 0

        print(f"  d={params['max_depth']} lr={params['learning_rate']} ne={params['n_estimators']} | "
              f"val_AUC={val_auc:.4f} test_AUC={test_auc:.4f} "
              f"thresh={best_thresh:.2f} trades={tday:.0f}/d PF={test_stats.get('profit_factor',0):.2f}"
              f" ({time.time()-t1:.1f}s)")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_threshold = best_thresh
            best_stats = test_stats
            best_config = cfg

    print(f"\n  Bestes Setup: depth={best_config['max_depth']} lr={best_config['learning_rate']}")
    print(f"  Val AUC: {best_auc:.4f}")

    # Save best model
    with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    import shutil
    shutil.copy(os.path.join(MODEL_DIR, "xgboost.pkl"), os.path.join(LIVE_DIR, "xgboost_model.pkl"))

    # Final OOS
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n=== ENDGÜLTIGE OOS-ERGEBNISSE ===")
    print(f"  AUC: {test_auc:.4f} (Val: {best_auc:.4f})")
    print(f"  Threshold: {best_threshold:.2f}")
    if best_stats:
        print(f"  Trades: {best_stats['n_trades']} ({best_stats['n_trades']/n_test_days:.0f}/Tag)")
        print(f"  Win Rate: {best_stats['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {best_stats['profit_factor']:.2f}")
        print(f"  Total Profit: {best_stats['total_profit']:.0f} USD")
        print(f"  Max Drawdown: {best_stats['max_drawdown']:.0f} USD")
        print(f"  EV/Trade: {best_stats['total_profit']/best_stats['n_trades']:.2f} USD")
        print(f"  TP hits: {best_stats['tp_hits']} ({best_stats['tp_hits']/best_stats['n_trades']*100:.1f}%)")

    # Summary
    summary = {
        "model": "XGBoost_Intraday_Tuned",
        "horizon_minutes": HORIZON,
        "tp_points": TP, "sl_points": SL,
        "rr_ratio": RR, "lot_size": LOTS, "trade_oz": TRADE_OZ,
        "profit_per_tp": TP * POINT * TRADE_OZ,
        "loss_per_sl": SL * POINT * TRADE_OZ,
        "spread": SPREAD, "slippage": SLIPPAGE,
        "best_iteration": best_model.best_iteration + 1,
        "best_params": {k: v for k, v in best_config.items() if k in ["max_depth", "learning_rate", "n_estimators"]},
        "threshold": best_threshold,
        "val_auc": best_auc,
        "test_auc": test_auc,
        "n_features": len(FEATURE_COLUMNS),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "best_stats": best_stats if best_stats else {},
    }

    with open(os.path.join(REPORTS_DIR, "xgboost_intraday_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Regenerate ZIP
    import zipfile
    live_files = []
    for root, _, files in os.walk("XAUUSD/live"):
        for fn in files:
            fp = os.path.join(root, fn)
            arcname = os.path.relpath(fp, "XAUUSD")
            live_files.append((fp, arcname))

    zip_path = "XAUUSD/live_trading_xauusd.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp, arcname in live_files:
            zf.write(fp, arcname)

    print(f"\n{'='*70}")
    print(f"FERTIG in {time.time()-t0:.1f}s")
    print(f"  AUC={test_auc:.4f} | PF={summary.get('best_stats',{}).get('profit_factor',0):.2f}")
    print(f"  ZIP: {zip_path} ({os.path.getsize(zip_path)/1024:.1f} KB)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
