"""Einfacher XGBoost OOS-Test."""
import sys, os, pickle, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TARGET_PARAMS, TEST_START,
                    TEST_END, TRAIN_START, TRAIN_END, MODELS_DIR,
                    REPORTS_DIR, POINT, SPREAD_POINTS, SLIPPAGE_POINTS)
from backtest_engine import BacktestEngine

def main():
    from data_preparation import load_combined
    df = load_combined()
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    print(f"Train: {len(train_df)} rows | Test(OOS): {len(test_df)} rows")
    print(f"Train period: {train_df['timestamp'].iloc[0]} bis {train_df['timestamp'].iloc[-1]}")
    print(f"Test period:  {test_df['timestamp'].iloc[0]} bis {test_df['timestamp'].iloc[-1]}")

    y_test = test_df[TARGET_COLUMN].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOOS Classification: AUC={auc:.4f} Accuracy={acc:.4f}")

    # Backtest
    signals = y_proba >= 0.5
    engine = BacktestEngine(tp_points=TARGET_PARAMS["tp_points"],
                            sl_points=TARGET_PARAMS["sl_points"],
                            horizon=TARGET_PARAMS["horizon"])
    trades_df, stats = engine.run(test_df, signals)

    print(f"\nOOS Trading Backtest:")
    print(f"  Trades: {stats['n_trades']} | Win: {stats['win_rate']*100:.1f}%")
    print(f"  PF: {stats['profit_factor']:.2f} | Profit: {stats['total_profit']:.0f}")
    print(f"  Max DD: {stats['max_drawdown']:.0f} | Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"  TP:{stats['tp_hit_rate']*100:.1f}% SL:{stats['sl_hit_rate']*100:.1f}% Exp:{stats['expiry_rate']*100:.1f}%")

    # Baseline comparison
    bl = pd.read_csv(os.path.join(RESULTS_DIR := os.path.join(os.path.dirname(__file__), "..", "results"), "baseline_results.csv"))
    print(f"\nBaseline Comparison:")
    print(f"{'Strategy':20s} | {'Win%':>6s} | {'PF':>6s} | {'Profit':>10s}")
    print("-" * 50)
    for _, r in bl.iterrows():
        print(f"{r['strategy']:20s} | {r['win_rate']*100:5.1f}% | {r['profit_factor']:6.2f} | {r['total_profit']:10.0f}")
    print(f"{'XGBoost(OOS)':20s} | {stats['win_rate']*100:5.1f}% | {stats['profit_factor']:6.2f} | {stats['total_profit']:10.0f}")

    summary = {"oos_auc": float(auc), "oos_accuracy": float(acc),
               "oos_pf": float(stats['profit_factor']), "oos_profit": float(stats['total_profit']),
               "oos_winrate": float(stats['win_rate']), "oos_trades": int(stats['n_trades']),
               "max_dd": float(stats['max_drawdown']), "sharpe": float(stats['sharpe_ratio'])}
    with open(os.path.join(REPORTS_DIR, "xgboost_oos_simple.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
