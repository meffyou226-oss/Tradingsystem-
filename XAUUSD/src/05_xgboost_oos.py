"""
XGBoost OOS-Verbesserung & Walk-Forward.

Fokus:
1. XGBoost Modell und Predictions laden
2. Threshold-Optimierung (0.3 - 0.7) auf Validation
3. Walk-Forward-Analyse (5-Folds)
4. OOS-Test mit optimiertem Threshold
5. Vergleich mit Baseline
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
                             precision_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, PRIMARY_MODEL,
                    TARGET_PARAMS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                    TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                    REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined

MODEL_PATH = os.path.join(MODELS_DIR, f"{PRIMARY_MODEL}.pkl")
RESULTS_DIR_LOCAL = os.path.join(os.path.dirname(__file__), "..", "results", "xgboost_oos")
os.makedirs(RESULTS_DIR_LOCAL, exist_ok=True)


def load_data_and_model():
    """Lade Combined-Daten und XGBoost-Modell."""
    print("Lade Daten und Modell...")
    df = load_combined()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    # Split
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return df, model, feature_cols, train_df, val_df, test_df


def find_optimal_threshold(model, val_df, feature_cols):
    """Findet optimalen Classification-Threshold auf Validation-Set."""
    print("\n--- Threshold-Optimierung ---")
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float32)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.3, 0.71, 0.05)
    best_pf = 0
    best_threshold = 0.5
    best_stats = None
    
    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )
    
    results = []
    for thresh in thresholds:
        signals = y_proba >= thresh
        trades_df, stats = engine.run(val_df, signals)
        if stats:
            results.append({
                "threshold": thresh,
                "pf": stats["profit_factor"],
                "win_rate": stats["win_rate"],
                "profit": stats["total_profit"],
                "n_trades": stats["n_trades"],
                "accuracy": accuracy_score(y_val, signals.astype(int)),
                "auc": roc_auc_score(y_val, y_proba),
            })
            print(f"  thresh={thresh:.2f}: AUC={results[-1]['auc']:.4f} "
                  f"Acc={results[-1]['accuracy']:.4f} PF={stats['profit_factor']:.2f} "
                  f"Win={stats['win_rate']*100:.1f}% Profit={stats['total_profit']:.0f} "
                  f"Trades={stats['n_trades']}")
            
            if stats["profit_factor"] > best_pf and stats["n_trades"] > 1000:
                best_pf = stats["profit_factor"]
                best_threshold = thresh
                best_stats = stats
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR_LOCAL, "threshold_optimization.csv"), index=False)
    
    print(f"\n  Bester Threshold: {best_threshold:.2f} (PF={best_pf:.2f})")
    return best_threshold, best_stats, y_proba


def walk_forward_analysis(df, feature_cols, best_threshold):
    """Walk-Forward-Analyse mit XGBoost."""
    print(f"\n--- Walk-Forward-Analyse (5 Folds) ---")
    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 2)
    wf_results = []
    
    from xgboost import XGBClassifier
    
    for fold in range(n_splits):
        train_end_idx = fold * fold_size + fold_size * 3
        test_end_idx = min(train_end_idx + fold_size, n)
        
        if test_end_idx <= train_end_idx:
            break
        
        train_val = df.iloc[:train_end_idx]
        wf_test = df.iloc[train_end_idx:test_end_idx]
        
        X_train = train_val[feature_cols].values.astype(np.float32)
        y_train = train_val[TARGET_COLUMN].values.astype(np.float32)
        X_test = wf_test[feature_cols].values.astype(np.float32)
        y_test = wf_test[TARGET_COLUMN].values.astype(np.float32)
        
        model = XGBClassifier(
            n_estimators=370, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss"
        )
        model.fit(X_train, y_train, verbose=False)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        signals = y_proba >= best_threshold
        engine = BacktestEngine(
            tp_points=TARGET_PARAMS["tp_points"],
            sl_points=TARGET_PARAMS["sl_points"],
            horizon=TARGET_PARAMS["horizon"],
        )
        trades_df, bt_stats = engine.run(wf_test, signals)
        
        if bt_stats:
            wf_results.append({
                "fold": fold + 1,
                "train_end": str(train_val["timestamp"].iloc[-1]),
                "test_end": str(wf_test["timestamp"].iloc[-1]),
                "auc": auc,
                "pf": bt_stats["profit_factor"],
                "total_profit": bt_stats["total_profit"],
                "win_rate": bt_stats["win_rate"],
                "n_trades": bt_stats["n_trades"],
                "max_dd": bt_stats["max_drawdown"],
                "sharpe": bt_stats["sharpe_ratio"],
            })
            print(f"  Fold {fold+1}: AUC={auc:.4f} PF={bt_stats['profit_factor']:.2f} "
                  f"Profit={bt_stats['total_profit']:.0f} Trades={bt_stats['n_trades']}")
    
    wf_df = pd.DataFrame(wf_results)
    wf_df.to_csv(os.path.join(RESULTS_DIR_LOCAL, "walkforward_xgboost.csv"), index=False)
    
    print(f"\n  Ø AUC: {wf_df['auc'].mean():.4f} ± {wf_df['auc'].std():.4f}")
    print(f"  Ø PF: {wf_df['pf'].mean():.2f} ± {wf_df['pf'].std():.2f}")
    print(f"  Ø Profit: {wf_df['total_profit'].mean():.0f} ± {wf_df['total_profit'].std():.0f}")
    
    return wf_df


def oos_test_evaluation(model, test_df, feature_cols, best_threshold):
    """Detaillierte OOS-Test-Bewertung."""
    print(f"\n--- OOS Test-Bewertung ---")
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[TARGET_COLUMN].values.astype(np.float32)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  Threshold: {best_threshold:.2f}")
    
    # Trading backtest
    signals = y_proba >= best_threshold
    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )
    trades_df, stats = engine.run(test_df, signals)
    
    if stats:
        print(f"\n  Trading Backtest (OOS):")
        print(f"    Trades:      {stats['n_trades']}")
        print(f"    Win Rate:    {stats['win_rate']*100:.1f}%")
        print(f"    Profit Factor: {stats['profit_factor']:.2f}")
        print(f"    Total Profit: {stats['total_profit']:.0f}")
        print(f"    Max Drawdown: {stats['max_drawdown']:.0f}")
        print(f"    Sharpe:       {stats['sharpe_ratio']:.2f}")
        print(f"    Avg Trade:    {stats['total_profit']/stats['n_trades']:.2f}")
        print(f"    TP hit: {stats['tp_hit_rate']*100:.1f}% | SL hit: {stats['sl_hit_rate']*100:.1f}% | Expiry: {stats['expiry_rate']*100:.1f}%")
        print(f"    Max consec losses: {stats['max_consec_losses']}")
        print(f"    Max consec wins: {stats['max_consec_wins']}")
    
    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(BACKTESTS_DIR := os.path.join(os.path.dirname(__file__), "..", "backtests"),
                                      f"xgboost_oos_trades.csv"), index=False)
    
    # Save predictions
    pred_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "prediction_proba": y_proba,
        "prediction": y_pred,
        "actual": y_test.astype(int),
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR_LOCAL, "predictions_oos.csv"), index=False)
    
    return stats, auc


def compare_with_baseline(oos_stats):
    """Vergleicht XGBoost OOS mit Baseline-Strategien."""
    print(f"\n--- Vergleich: XGBoost vs Baseline ---")
    
    bl_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    bl = pd.read_csv(bl_path)
    
    print(f"{'Strategie':20s} | {'Win Rate':>8s} | {'ProfPF':>8s} | {'Profit':>10s} | {'Trades':>8s}")
    print("-" * 65)
    
    for _, row in bl.iterrows():
        print(f"{row['strategy']:20s} | {row['win_rate']*100:7.1f}% | {row['profit_factor']:8.2f} | {row['total_profit']:10.0f} | {row['n_trades']:8}")
    
    print(f"\n{'XGBoost (OOS)':20s} | {oos_stats['win_rate']*100:7.1f}% | {oos_stats['profit_factor']:8.2f} | {oos_stats['total_profit']:10.0f} | {oos_stats['n_trades']:8}")
    
    # Determine if ML beats baseline
    best_baseline_pf = bl["profit_factor"].max()
    best_baseline_profit = bl["total_profit"].max()
    
    if oos_stats["profit_factor"] > best_baseline_pf:
        print(f"\n  ✅ XGBoost SCHLÄGT Best-Baseline (PF={best_baseline_pf:.2f} vs {oos_stats['profit_factor']:.2f})")
    else:
        print(f"\n  ⚠️  XGBoost untertrifft Best-Baseline")
    
    if oos_stats["total_profit"] > best_baseline_profit:
        print(f"  ✅ XGBoost SCHLÄGT Best-Baseline (Profit={best_baseline_profit:.0f} vs {oos_stats['total_profit']:.0f})")
    
    comparison = {
        "xgboost": {
            "model": "XGBoost_OOS",
            "win_rate": oos_stats["win_rate"],
            "profit_factor": oos_stats["profit_factor"],
            "total_profit": oos_stats["total_profit"],
            "n_trades": oos_stats["n_trades"],
            "max_drawdown": oos_stats["max_drawdown"],
            "sharpe": oos_stats["sharpe_ratio"],
        },
        "baselines": bl.to_dict(orient="records"),
    }
    
    comp_path = os.path.join(REPORTS_DIR, "xgboost_vs_baseline.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    return comparison


def plot_predictions_distribution(model, val_df, feature_cols, best_threshold):
    """Erstellt Visualisierung der Prediction-Verteilung."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float32)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prediction distribution by actual class
    pos = y_proba[y_val == 1]
    neg = y_proba[y_val == 0]
    axes[0, 0].hist(pos, bins=50, alpha=0.6, label=f"TP Hit (n={len(pos)})", color="green")
    axes[0, 0].hist(neg, bins=50, alpha=0.6, label=f"SL/Nein (n={len(neg)})", color="red")
    axes[0, 0].axvline(best_threshold, color="black", linestyle="--", label=f"Threshold={best_threshold:.2f}")
    axes[0, 0].set_xlabel("Prediction Probability")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("XGBoost Prediction Distribution (Validation)")
    axes[0, 0].legend()
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_val, y_proba):.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve (Validation)")
    axes[0, 1].legend()
    
    # 3. Calibration
    from sklearn.calibration import calibration_curve
    if len(np.unique(y_val)) > 1:
        frac_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)
        axes[1, 0].plot(mean_pred, frac_pos, "s-", label="XGBoost")
        axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].set_title("Calibration Curve (Validation)")
        axes[1, 0].legend()
    
    # 4. Feature importance
    fi = pd.read_csv(os.path.join(RESULTS_DIR, "feature_importance_xgboost.csv"))
    fi.head(15).plot.barh(x="feature", y="importance", ax=axes[1, 1], legend=False)
    axes[1, 1].set_xlabel("Importance")
    axes[1, 1].set_title("Top 15 Feature Importance (XGBoost)")
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "xgboost_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Diagramm gespeichert: {plot_path}")
    
    return plot_path


def main():
    print("=" * 70)
    print(f"XGBoost OOS-Analyse & Verbesserung")
    print(f"{'=' * 70}")
    
    # Load data and model
    df, model, feature_cols, train_df, val_df, test_df = load_data_and_model()
    
    # 1. Threshold optimization
    best_threshold, val_stats, _ = find_optimal_threshold(model, val_df, feature_cols)
    
    # 2. OOS Test
    oos_stats, oos_auc = oos_test_evaluation(model, test_df, feature_cols, best_threshold)
    
    # 3. Walk-Forward
    wf_df = walk_forward_analysis(df, feature_cols, best_threshold)
    
    # 4. Comparison with baseline
    comparison = compare_with_baseline(oos_stats)
    
    # 5. Plots
    plot_predictions_distribution(model, val_df, feature_cols, best_threshold)
    
    # 6. Summary
    print(f"\n{'=' * 70}")
    print("XGBoost OOS-FAZIT")
    print(f"{'=' * 70}")
    print(f"  Bester Threshold: {best_threshold:.2f}")
    print(f"  OOS AUC: {oos_auc:.4f}")
    print(f"  OOS Profit Factor: {oos_stats['profit_factor']:.2f}")
    print(f"  OOS Total Profit: {oos_stats['total_profit']:.0f}")
    print(f"  OOS Win Rate: {oos_stats['win_rate']*100:.1f}%")
    print(f"  Walk-Forward Ø AUC: {wf_df['auc'].mean():.4f} ± {wf_df['auc'].std():.4f}")
    print(f"  Walk-Forward Ø PF: {wf_df['pf'].mean():.2f} ± {wf_df['pf'].std():.2f}")
    
    summary = {
        "best_threshold": best_threshold,
        "oos_auc": oos_auc,
        "oos_pf": oos_stats["profit_factor"],
        "oos_profit": oos_stats["total_profit"],
        "oos_winrate": oos_stats["win_rate"],
        "wf_mean_auc": float(wf_df["auc"].mean()),
        "wf_std_auc": float(wf_df["auc"].std()),
        "wf_mean_pf": float(wf_df["pf"].mean()),
        "wf_std_pf": float(wf_df["pf"].std()),
    }
    summary_path = os.path.join(REPORTS_DIR, "xgboost_oos_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Zusammenfassung: {summary_path}")


if __name__ == "__main__":
    main()
