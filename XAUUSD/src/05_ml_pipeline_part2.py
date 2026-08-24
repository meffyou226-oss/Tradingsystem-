"""
Schritt 2: Vervollständige ML-Pipeline.

Führt aus, was fehlt:
- Random Forest Training
- Walk-Forward-Analyse (5-Folds)
- Kompletter Vergleich: Baseline vs ML
- Equity Curve und Drawdown-Vergleich
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
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, MODELS, PRIMARY_MODEL,
                    TARGET_PARAMS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                    TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                    REPORTS_DIR, BACKTESTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def load_trained_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_data(df):
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)
    return X, y, feature_cols


def train_rf(X_train, y_train, X_val, y_val, feature_cols):
    """Trainiert Random Forest."""
    print("Training: random_forest")
    t0 = time.time()
    from sklearn.ensemble import RandomForestClassifier
    
    params = MODELS["random_forest"]["params"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print(f"    Train time: {time.time()-t0:.1f}s")
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_proba)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"    Val: AUC={val_auc:.4f} Acc={val_acc:.4f}")
    
    # Save model
    path = os.path.join(MODELS_DIR, "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved: {path}")
    
    return model


def run_walk_forward(df):
    """Walk-Forward-Analyse für XGBoost (primäres Modell)."""
    print(f"\nWalk-Forward-Analyse: {PRIMARY_MODEL}")
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 2)
    wf_results = []
    
    params = dict(MODELS[PRIMARY_MODEL]["params"])
    
    for fold in range(n_splits):
        train_end_idx = fold * fold_size + fold_size * 3
        test_end_idx = min(train_end_idx + fold_size, n)
        
        if test_end_idx <= train_end_idx or train_end_idx >= n:
            break
        
        train_val = df.iloc[:train_end_idx]
        test = df.iloc[train_end_idx:test_end_idx]
        
        X_train, y_train, _ = prepare_data(train_val)
        X_test, y_test, _ = prepare_data(test)
        
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, eval_metric="logloss", early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        # Trading backtest
        signals = y_proba >= 0.5
        engine = BacktestEngine(
            tp_points=TARGET_PARAMS["tp_points"],
            sl_points=TARGET_PARAMS["sl_points"],
            horizon=TARGET_PARAMS["horizon"],
        )
        trades_df, bt_stats = engine.run(test, signals)
        
        if bt_stats:
            wf_results.append({
                "fold": fold + 1,
                "train_end": str(train_val["timestamp"].iloc[-1]),
                "test_end": str(test["timestamp"].iloc[-1]),
                "auc": auc,
                "accuracy": acc,
                "pf": bt_stats["profit_factor"],
                "n_trades": bt_stats["n_trades"],
                "total_profit": bt_stats["total_profit"],
                "win_rate": bt_stats["win_rate"],
                "max_dd": bt_stats["max_drawdown"],
                "sharpe": bt_stats["sharpe_ratio"],
            })
            print(f"  Fold {fold+1}: AUC={auc:.4f} PF={bt_stats['profit_factor']:.2f} "
                  f"Profit={bt_stats['total_profit']:.0f} Trades={bt_stats['n_trades']}")
    
    wf_df = pd.DataFrame(wf_results)
    wf_path = os.path.join(RESULTS_DIR, f"walkforward_{PRIMARY_MODEL}.csv")
    wf_df.to_csv(wf_path, index=False)
    print(f"\n  Ø AUC: {wf_df['auc'].mean():.4f} ± {wf_df['auc'].std():.4f}")
    print(f"  Ø PF: {wf_df['pf'].mean():.2f} ± {wf_df['pf'].std():.2f}")
    return wf_df


def generate_comparison_report():
    """Generiert umfassenden Vergleich: Baseline vs ML."""
    print(f"\n{'=' * 70}")
    print("COMPLIATIVE VERGLEICH: BASELINE vs ML")
    print(f"{'=' * 70}")
    
    df = load_combined()
    
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()
    
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_test, y_test, _ = prepare_data(test_df)
    
    results = []
    
    # Load baseline results
    bl_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(bl_path):
        bl = pd.read_csv(bl_path)
        for _, row in bl.iterrows():
            results.append({
                "model": f"Baseline_{row['strategy']}",
                "test_auc": 0.5,  # Not applicable
                "test_accuracy": row["win_rate"],
                "test_pf": row["profit_factor"],
                "test_total_profit": row["total_profit"],
                "test_win_rate": row["win_rate"],
                "test_max_dd": row["max_drawdown"],
                "test_sharpe": row["sharpe"],
                "test_n_trades": row["n_trades"],
            })
    
    # ML models
    for model_name in ["xgboost", "lightgbm"]:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            continue
        
        model = load_trained_model(model_name)
        X = test_df[feature_cols].values.astype(np.float32)
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        # Trading backtest
        signals = y_proba >= 0.5
        engine = BacktestEngine(
            tp_points=TARGET_PARAMS["tp_points"],
            sl_points=TARGET_PARAMS["sl_points"],
            horizon=TARGET_PARAMS["horizon"],
        )
        trades_df, bt_stats = engine.run(test_df, signals)
        
        if bt_stats:
            results.append({
                "model": model_name.upper(),
                "test_auc": auc,
                "test_accuracy": acc,
                "test_pf": bt_stats["profit_factor"],
                "test_total_profit": bt_stats["total_profit"],
                "test_win_rate": bt_stats["win_rate"],
                "test_max_dd": bt_stats["max_drawdown"],
                "test_sharpe": bt_stats["sharpe_ratio"],
                "test_n_trades": bt_stats["n_trades"],
            })
    
    results_df = pd.DataFrame(results)
    
    # Save comparison
    results_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nGespeichert: {results_path}")
    
    return results_df


def main():
    print("=" * 70)
    print("ML Pipeline - Teil 2: Vervollständigung")
    print("=" * 70)
    
    df = load_combined()
    
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # 1. Train Random Forest
    print("\n--- Random Forest Training ---")
    X_train, y_train, feature_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)
    
    rf_model = train_rf(X_train, y_train, X_val, y_val, feature_cols)
    
    # Full RF evaluation
    from sklearn.metrics import f1_score
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    print(f"  Test: AUC={test_auc:.4f} Acc={test_acc:.4f} F1={test_f1:.4f}")
    
    # RF Backtest
    signals = y_proba >= 0.5
    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )
    trades_df, bt_stats = engine.run(test_df, signals)
    if bt_stats:
        print(f"  Test Backtest: PF={bt_stats['profit_factor']:.2f} Win={bt_stats['win_rate']*100:.1f}% "
              f"Profit={bt_stats['total_profit']:.0f} Trades={bt_stats['n_trades']}")
    
    # Feature importance RF
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(RESULTS_DIR, "feature_importance_random_forest.csv")
    fi.to_csv(fi_path, index=False)
    
    # Save RF trades
    if len(trades_df) > 0:
        trades_path = os.path.join(BACKTESTS_DIR, "ml_random_forest_test_trades.csv")
        trades_df.to_csv(trades_path, index=False)
    
    # 2. Walk-Forward Analysis
    wf_df = run_walk_forward(df)
    
    # 3. Comparison Report
    comparison = generate_comparison_report()
    
    # 4. Save summary JSON
    summary = {
        "primary_model": PRIMARY_MODEL,
        "target_params": TARGET_PARAMS,
        "data_splits": {
            "train": f"{TRAIN_START} to {TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
        "walk_forward": {
            "mean_auc": float(wf_df["auc"].mean()),
            "std_auc": float(wf_df["auc"].std()),
            "mean_pf": float(wf_df["pf"].mean()),
            "std_pf": float(wf_df["pf"].std()),
            "n_folds": len(wf_df),
        },
        "model_comparison": comparison.to_dict(orient="records"),
    }
    
    summary_path = os.path.join(REPORTS_DIR, "ml_pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    print(f"Walk-Forward: Ø AUC={wf_df['auc'].mean():.4f} ± {wf_df['auc'].std():.4f}")
    print(f"Walk-Forward: Ø PF={wf_df['pf'].mean():.2f} ± {wf_df['pf'].std():.2f}")
    print(f"Modellvergleich gespeichert: {os.path.join(REPORTS_DIR, 'model_comparison.csv')}")
    print(f"Zusammenfassung: {summary_path}")


if __name__ == "__main__":
    main()
