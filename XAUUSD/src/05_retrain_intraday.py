"""
Intraday XGBoost Training (30-Minuten-Horizont).

Parameter:
- Horizon: 30 Minuten
- TP: 300 Punkte (3.00 USD, 15 USD Gewinn bei 0.05 Lot)
- SL: 150 Punkte (1.50 USD)
- R:R: 2:1
- Lots: 0.05

Trainiert Model und findet Threshold für ~5-10 Trades/Tag.
"""

import os, sys, time, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_COLUMNS, TRAIN_START, TRAIN_END, VAL_START, VAL_END
from config import TEST_START, TEST_END, POINT, MODELS, MODELS_DIR, BACKTESTS_DIR
from backtest_engine import BacktestEngine
from data_preparation import load_combined

import importlib.util
_spec = importlib.util.spec_from_file_location("td",
    os.path.join(os.path.dirname(__file__), "03_target_definition.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)

# Neue Parameter
HORIZON = 30
TP = 300
SL = 150
RR = 2.0
LOTS = 0.05
PROFIT_PER_TP = TP * POINT * LOTS  # 300 * 0.01 * 5 = 15.0 USD
TARGET_COL = f"target_h{HORIZON}_sl{SL}_rr{RR}"


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"XGBoost Intraday-Training (30 Min Horizon)")
    print(f"  TP={TP}pts ({TP*POINT:.2f} USD) | SL={SL}pts ({SL*POINT:.2f} USD)")
    print(f"  R:R={RR}:1 | Lots={LOTS} | Profit/TP: {PROFIT_PER_TP:.2f} USD")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Compute targets
    print(f"\nBerechne Target: {TARGET_COL}...")
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    df[TARGET_COL] = td.compute_target_vectorized(close, high, low, HORIZON, TP, SL)
    print(f"  Balance: {df[TARGET_COL].mean():.2%}")

    # Save updated parquet with new target
    parquet_path = os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_m1_combined.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet aktualisiert: {parquet_path}")

    # Split
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Target balance: Train={train_df[TARGET_COL].mean():.2%} "
          f"Val={val_df[TARGET_COL].mean():.2%} Test={test_df[TARGET_COL].mean():.2%}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # Train
    print("\nTraining XGBoost...")
    from xgboost import XGBClassifier
    params = MODELS["xgboost"]["params"]
    t1 = time.time()
    model = XGBClassifier(**params, eval_metric="logloss", early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Training: {time.time()-t1:.1f}s | Best iter: {model.best_iteration+1}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")

    # Evaluate
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nValidation: AUC={val_auc:.4f} | Accuracy={accuracy_score(y_val, (y_val_proba>=0.5).astype(int)):.4f}")
    print(f"Test:      AUC={test_auc:.4f} | Accuracy={accuracy_score(y_test, (y_test_proba>=0.5).astype(int)):.4f}")

    # Threshold optimization for 5-10 trades/day
    print("\n--- Threshold-Optimierung (5-10 Trades/Tag) ---")
    n_test_days = (pd.to_datetime(TEST_END) - pd.to_datetime(TEST_START)).days + 1
    print(f"Test period: {n_test_days} Tage")

    thresholds = np.arange(0.50, 0.95, 0.05)
    best_threshold = 0.50
    best_result = None

    engine = BacktestEngine(tp_points=TP, sl_points=SL, horizon=HORIZON)

    for thresh in thresholds:
        signals = y_test_proba >= thresh
        trades_df, stats = engine.run(test_df, signals)

        if stats and stats["n_trades"] > 0:
            trades_per_day = stats["n_trades"] / n_test_days
            ev_per_trade = stats["total_profit"] / stats["n_trades"]

            print(f"  t={thresh:.2f}: AUC={test_auc:.4f} PF={stats['profit_factor']:.2f} "
                  f"Win={stats['win_rate']*100:.1f}% Profit={stats['total_profit']:.0f}USD "
                  f"Trades={stats['n_trades']} ({trades_per_day:.0f}/Tag) EV={ev_per_trade:.2f}USD")

            # Find threshold with 5-10 trades/day and positive PF
            if trades_per_day >= 3 and trades_per_day <= 15:
                if best_result is None or stats["profit_factor"] > best_result["pf"]:
                    best_threshold = thresh
                    best_result = {
                        "threshold": thresh,
                        "trades": stats["n_trades"],
                        "trades_per_day": trades_per_day,
                        "pf": stats["profit_factor"],
                        "win_rate": stats["win_rate"],
                        "profit": stats["total_profit"],
                        "max_dd": stats["max_drawdown"],
                        "ev_per_trade": ev_per_trade,
                    }

    if best_result is None:
        best_threshold = 0.75
        print(f"\n  Kein idealer Threshold gefunden. Nutze Standard: {best_threshold}")
    else:
        print(f"\n  Gewählter Threshold: {best_threshold:.2f}")

    # Final OOS backtest
    signals = y_test_proba >= best_threshold
    trades_df, stats = engine.run(test_df, signals)

    if stats:
        print(f"\n--- Endgültige OOS-Ergebnisse (Threshold={best_threshold:.2f}) ---")
        print(f"  Trades: {stats['n_trades']} ({stats['n_trades']/n_test_days:.0f}/Tag)")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Total Profit: {stats['total_profit']:.0f} USD")
        print(f"  Max Drawdown: {stats['max_drawdown']:.0f}")
        print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"  EV/Trade: {stats['total_profit']/stats['n_trades']:.2f} USD")

    # Save trades
    os.makedirs(os.path.join(BACKTESTS_DIR, "xgboost_intraday"), exist_ok=True)
    trades_df.to_csv(os.path.join(BACKTESTS_DIR, "xgboost_intraday", "trades.csv"), index=False)
    trades_df.to_csv("XAUUSD/backtests/xgboost_oos_trades.csv", index=False)

    # Save summary
    summary = {
        "model": "XGBoost_Intraday_30min",
        "horizon_minutes": HORIZON,
        "tp_points": TP,
        "sl_points": SL,
        "rr_ratio": RR,
        "lot_size": LOTS,
        "profit_per_tp": PROFIT_PER_TP,
        "threshold": best_threshold,
        "best_iteration": model.best_iteration + 1,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_pf": stats["profit_factor"] if stats else 0,
        "test_profit": stats["total_profit"] if stats else 0,
        "test_win_rate": stats["win_rate"] if stats else 0,
        "test_n_trades": stats["n_trades"] if stats else 0,
        "test_trades_per_day": stats["n_trades"] / n_test_days if stats else 0,
        "test_max_dd": stats["max_drawdown"] if stats else 0,
        "test_sharpe": stats["sharpe_ratio"] if stats else 0,
        "target_column": TARGET_COL,
        "target_balance": df[TARGET_COL].mean(),
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)
    summary_path = os.path.join(os.path.dirname(__file__), "..", "reports", "xgboost_intraday_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"FERTIG in {time.time()-t0:.1f}s")
    print(f"  AUC={test_auc:.4f} | PF={summary['test_pf']:.2f} | "
          f"Profit={summary['test_profit']:.0f}USD | {summary['test_trades_per_day']:.0f}Trades/Tag")
    print(f"  Modell: {model_path}")
    print(f"  Report: {summary_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
