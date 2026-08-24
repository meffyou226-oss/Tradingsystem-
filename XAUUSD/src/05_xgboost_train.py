"""
Fokussierte XGBoost-Training und OOS-Test.

Schnelle Pipeline:
1. Lade Parquet (20s statt 20s für CSV)
2. Trainiere XGBoost mit Early Stopping
3. OOS-Test auf Sep 2025 - Aug 2026
4. Threshold-Optimierung auf Validation
5. Vergleich mit Baseline
6. Speichere Modell + Ergebnisse
"""

import os, sys, time, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_START, TRAIN_END,
                    VAL_START, VAL_END, TEST_START, TEST_END,
                    TARGET_PARAMS, MODELS, PRIMARY_MODEL, MODELS_DIR,
                    RESULTS_DIR, REPORTS_DIR, BACKTESTS_DIR)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def main():
    t0 = time.time()
    print("=" * 70)
    print("XGBoost Training + OOS-Test (fokussiert)")
    print("=" * 70)

    # Load
    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train balance: {train_df[TARGET_COLUMN].mean():.2%} | Val: {val_df[TARGET_COLUMN].mean():.2%} | Test: {test_df[TARGET_COLUMN].mean():.2%}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[TARGET_COLUMN].values.astype(np.float32)

    # Train
    print("\nTraining XGBoost...")
    from xgboost import XGBClassifier
    params = MODELS["xgboost"]["params"]
    t1 = time.time()
    model = XGBClassifier(**params, eval_metric="logloss", early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Training time: {time.time()-t1:.1f}s | Best iter: {model.best_iteration+1}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")

    # Validate
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_acc = accuracy_score(y_val, (y_val_proba >= 0.5).astype(int))
    print(f"\nValidation: AUC={val_auc:.4f} Acc={val_acc:.4f}")

    # Test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_acc = accuracy_score(y_test, (y_test_proba >= 0.5).astype(int))
    print(f"Test:      AUC={test_auc:.4f} Acc={test_acc:.4f}")

    # Threshold optimization
    print("\n--- Threshold-Optimierung ---")
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_pf = 0
    best_threshold = 0.5
    engine = BacktestEngine(tp_points=TARGET_PARAMS["tp_points"],
                            sl_points=TARGET_PARAMS["sl_points"],
                            horizon=TARGET_PARAMS["horizon"])

    for thresh in thresholds:
        signals = y_val_proba >= thresh
        trades, stats = engine.run(val_df, signals)
        if stats and stats["n_trades"] > 1000:
            print(f"  thresh={thresh:.2f}: AUC={val_auc:.4f} PF={stats['profit_factor']:.2f} "
                  f"Win={stats['win_rate']*100:.1f}% Profit={stats['total_profit']:.0f}")
            if stats["profit_factor"] > best_pf:
                best_pf = stats["profit_factor"]
                best_threshold = thresh

    print(f"\n  Bester Threshold: {best_threshold:.2f} (Val PF={best_pf:.2f})")

    # OOS Backtest with best threshold
    print("\n--- OOS Backtest ---")
    signals = y_test_proba >= best_threshold
    trades_df, stats = engine.run(test_df, signals)

    print(f"  Trades: {stats['n_trades']}")
    print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Total Profit: {stats['total_profit']:.0f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.0f}")
    print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"  Avg Trade: {stats['total_profit']/stats['n_trades']:.2f}")
    print(f"  TP:{stats['tp_hit_rate']*100:.1f}% SL:{stats['sl_hit_rate']*100:.1f}% Exp:{stats['expiry_rate']*100:.1f}%")

    # Save results
    trades_path = os.path.join(BACKTESTS_DIR, "xgboost_oos_trades.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"\n  Trades gespeichert: {trades_path}")

    pred_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "prediction_proba": y_test_proba,
        "prediction": y_test_proba >= best_threshold,
        "actual": y_test,
    })
    pred_path = os.path.join(RESULTS_DIR, "predictions_xgboost.csv")
    pred_df.to_csv(pred_path, index=False)

    # Feature importance
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi_path = os.path.join(RESULTS_DIR, "feature_importance_xgboost.csv")
    fi.to_csv(fi_path, index=False)
    print(f"  Feature Importance: {fi_path}")

    # Threshold optimization results
    thresh_df = pd.DataFrame([{
        "best_threshold": best_threshold,
        "best_val_pf": best_pf,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "test_pf": stats["profit_factor"],
        "test_profit": stats["total_profit"],
        "test_win_rate": stats["win_rate"],
        "test_n_trades": stats["n_trades"],
        "test_max_dd": stats["max_drawdown"],
    }])
    thresh_df.to_csv(os.path.join(RESULTS_DIR, "xgboost_oos_summary.csv"), index=False)

    # Summary JSON
    summary = {
        "model": "XGBoost",
        "best_iteration": model.best_iteration + 1,
        "best_threshold": best_threshold,
        "val_auc": val_auc, "val_acc": val_acc,
        "test_auc": test_auc, "test_acc": test_acc,
        "test_pf": stats["profit_factor"],
        "test_profit": stats["total_profit"],
        "test_win_rate": stats["win_rate"],
        "test_n_trades": stats["n_trades"],
        "test_max_dd": stats["max_drawdown"],
        "test_sharpe": stats["sharpe_ratio"],
        "tp_hit_rate": stats["tp_hit_rate"],
        "sl_hit_rate": stats["sl_hit_rate"],
        "target_params": TARGET_PARAMS,
        "train_time_sec": time.time() - t0,
    }
    json_path = os.path.join(REPORTS_DIR, "xgboost_oos_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"FERTIG in {time.time()-t0:.1f}s")
    print(f"  AUC={test_auc:.4f} | PF={stats['profit_factor']:.2f} | Profit={stats['total_profit']:.0f}")
    print(f"  Model: {model_path}")
    print(f"  Report: {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
