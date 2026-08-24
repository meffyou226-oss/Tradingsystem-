"""
ML-Pipeline für XAUUSD M1 Trading-Signal-Vorhersage.

Verwendet zeitbasierte Splits (keine zufällige Aufteilung):
- Training:   2024-01-01 bis 2025-04-30
- Validation: 2025-05-01 bis 2025-08-31
- Test:       2025-09-01 bis 2026-08-23 (Out-of-Sample, unangetastet)

Trainiert und vergleicht:
- XGBoost (primär)
- LightGBM
- Random Forest

Bewertung:
- Klassifikations-Metriken (AUC, F1, Accuracy, Precision, Recall)
- Trading-Metriken (Profit Factor, Win Rate, Max Drawdown, Sharpe)
- Feature-Importance
"""

import os
import sys
import time
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, MODELS, PRIMARY_MODEL,
                    TARGET_PARAMS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                    TEST_START, TEST_END, SPREAD_POINTS, SLIPPAGE_POINTS,
                    MODELS_DIR, RESULTS_DIR, REPORTS_DIR, BACKTESTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(BACKTESTS_DIR, exist_ok=True)


def prepare_data(df):
    """Bereitet Features und Targets für Training vor."""
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)
    return X, y, feature_cols


def train_model(model_name, params, X_train, y_train, X_val, y_val):
    """Trainiert ein einzelnes Modell mit Early Stopping."""
    print(f"\n  Training: {model_name}")
    t0 = time.time()

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, eval_metric="logloss", early_stopping_rounds=50)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        best_iter = model.best_iteration + 1 if hasattr(model, 'best_iteration') else params["n_estimators"]
        print(f"    Best iteration: {best_iter}")

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[])
        best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ > 0 else params["n_estimators"]
        # Retrain with best iteration
        params["n_estimators"] = best_iter
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        print(f"    Best iteration: {best_iter}")

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        best_iter = params["n_estimators"]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_time = time.time() - t0
    print(f"    Train time: {train_time:.1f}s")
    return model


def evaluate_model(model, X, y):
    """Bewertet das Modell mit Klassifikations-Metriken."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        "confusion": confusion_matrix(y, y_pred).tolist(),
    }


def run_ml_backtest(df, model, feature_cols):
    """Führt Backtest mit ML-Prognosen durch."""
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]
    signals = predictions >= 0.5

    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )

    trades_df, stats = engine.run(df, signals)
    return trades_df, stats, predictions


def run_walk_forward(df, model_name, params, n_splits=5):
    """Walk-Forward-Analyse für Robustheit."""
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Split by time into n_splits sequential folds
    n = len(df)
    fold_size = n // (n_splits + 2)  # +2 for initial train/val
    wf_results = []

    print(f"\n  Walk-Forward ({n_splits} Folds)...")
    for fold in range(n_splits):
        train_end_idx = fold * fold_size + fold_size * 3  # 3x fold_size for train+val
        test_end_idx = min(train_end_idx + fold_size, n)

        if test_end_idx <= train_end_idx or train_end_idx >= n:
            break

        train_val = df.iloc[:train_end_idx]
        test = df.iloc[train_end_idx:test_end_idx]

        X_train, y_train, _ = prepare_data(train_val)
        X_test, y_test, _ = prepare_data(test)

        # Train
        if model_name == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(**params, eval_metric="logloss")
            model.fit(X_train, y_train, verbose=False)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # Trading backtest
        trades_df, stats, _ = run_ml_backtest(test, model, feature_cols)
        wf_results.append({
            "fold": fold + 1,
            "train_end": str(train_val["timestamp"].iloc[-1]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "auc": auc,
            "accuracy": acc,
            "pf": stats["profit_factor"] if stats else 0,
            "n_trades": stats["n_trades"] if stats else 0,
            "total_profit": stats["total_profit"] if stats else 0,
            "win_rate": stats["win_rate"] if stats else 0,
            "max_dd": stats["max_drawdown"] if stats else 0,
        })
        print(f"    Fold {fold+1}: AUC={auc:.4f} PF={stats['profit_factor']:.2f} Trades={stats['n_trades']} Profit={stats['total_profit']:.0f}")

    return pd.DataFrame(wf_results)


def get_feature_importance(model, feature_cols, model_name):
    """Extrahiert Feature-Importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    return pd.DataFrame({"feature": feature_cols, "importance": [0] * len(feature_cols)})


def main():
    print("=" * 70)
    print("XAUUSD M1 ML-Pipeline")
    print("=" * 70)

    # Load data
    print("\nLade kombinierte Daten ...")
    df = load_combined()
    print(f"  Gesamt: {len(df)} Zeilen ({df['timestamp'].iloc[0]} bis {df['timestamp'].iloc[-1]})")

    # Time-based splits
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Prepare data
    X_train, y_train, feature_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    print(f"\nFeature-Spalten: {len(feature_cols)}")
    print(f"Train target balance: {y_train.mean():.2%} positive")
    print(f"Val target balance:   {y_val.mean():.2%} positive")
    print(f"Test target balance:  {y_test.mean():.2%} positive")

    # Train models
    results = {}
    models_trained = {}

    for model_name in MODELS:
        params = MODELS[model_name]["params"]
        model = train_model(model_name, params, X_train, y_train, X_val, y_val)

        # Evaluate on validation
        val_metrics = evaluate_model(model, X_val, y_val)
        print(f"    Val Metrics: AUC={val_metrics['auc']:.4f} Acc={val_metrics['accuracy']:.4f} F1={val_metrics['f1']:.4f}")

        # Evaluate on test
        test_metrics = evaluate_model(model, X_test, y_test)
        print(f"    Test Metrics: AUC={test_metrics['auc']:.4f} Acc={test_metrics['accuracy']:.4f} F1={test_metrics['f1']:.4f}")

        # Trading backtest on test set
        trades_df, bt_stats, predictions = run_ml_backtest(test_df, model, feature_cols)
        print(f"    Test Backtest: PF={bt_stats['profit_factor']:.2f} Win={bt_stats['win_rate']*100:.1f}% "
              f"Profit={bt_stats['total_profit']:.0f} Trades={bt_stats['n_trades']}")

        # Trading backtest on validation
        val_trades, val_bt, _ = run_ml_backtest(val_df, model, feature_cols)
        print(f"    Val Backtest: PF={val_bt['profit_factor']:.2f} Win={val_bt['win_rate']*100:.1f}% "
              f"Profit={val_bt['total_profit']:.0f} Trades={val_bt['n_trades']}")

        results[model_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_trading": val_bt,
            "test_trading": bt_stats,
        }
        models_trained[model_name] = model

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"    Modell gespeichert: {model_path}")

        # Save feature importance
        fi = get_feature_importance(model, feature_cols, model_name)
        fi_path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name}.csv")
        fi.to_csv(fi_path, index=False)

        # Save test trades
        if len(trades_df) > 0:
            trades_path = os.path.join(BACKTESTS_DIR, f"ml_{model_name}_test_trades.csv")
            trades_df.to_csv(trades_path, index=False)

        # Save test predictions
        pred_df = pd.DataFrame({
            "timestamp": test_df["timestamp"],
            "prediction": predictions,
            "actual": y_test,
        })
        pred_path = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
        pred_df.to_csv(pred_path, index=False)

    # Walk-forward analysis for primary model
    print(f"\n{'=' * 70}")
    print(f"Walk-Forward-Analyse: {PRIMARY_MODEL}")
    print(f"{'=' * 70}")
    wf_results = run_walk_forward(df, PRIMARY_MODEL, MODELS[PRIMARY_MODEL]["params"])
    wf_path = os.path.join(RESULTS_DIR, f"walkforward_{PRIMARY_MODEL}.csv")
    wf_results.to_csv(wf_path, index=False)
    print(f"  Walk-Forward-Ergebnisse: {len(wf_results)} Folds")
    print(f"  Ø AUC: {wf_results['auc'].mean():.4f} ± {wf_results['auc'].std():.4f}")
    print(f"  Ø Profit Factor: {wf_results['pf'].mean():.2f} ± {wf_results['pf'].std():.2f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("ML-PIPELINE ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            "model": name,
            "test_auc": res["test_metrics"]["auc"],
            "test_accuracy": res["test_metrics"]["accuracy"],
            "test_f1": res["test_metrics"]["f1"],
            "test_pf": res["test_trading"]["profit_factor"],
            "test_winrate": res["test_trading"]["win_rate"],
            "test_profit": res["test_trading"]["total_profit"],
            "test_max_dd": res["test_trading"]["max_drawdown"],
            "test_sharpe": res["test_trading"]["sharpe_ratio"],
            "test_n_trades": res["test_trading"]["n_trades"],
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "ml_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nErgebnisse gespeichert: {summary_path}")

    # Save full results as JSON
    json_path = os.path.join(REPORTS_DIR, "ml_pipeline_results.json")
    with open(json_path, "w") as f:
        json.dump({k: {kk: {kkk: str(vvv) for kkk, vvv in vv.items()}
                      for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)
    print(f"Report gespeichert: {json_path}")

    return results, models_trained


if __name__ == "__main__":
    results, models = main()
