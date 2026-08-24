"""
Erstellt Visualisierungen für XGBoost OOS-Ergebnisse.

Diagramme:
1. Equity Curve (Training vs Validation vs OOS Test)
2. Drawdown-Kurve
3. Monatliche Performance (Walk-Forward-Stil)
4. Trade-Dauerverteilung
5. Feature-Importance-Bar-Chart
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TARGET_PARAMS,
                    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                    TEST_START, TEST_END, MODELS_DIR, REPORTS_DIR,
                    SPREAD_POINTS, SLIPPAGE_POINTS, POINT)
from backtest_engine import BacktestEngine, run_backtest
from data_preparation import load_combined

os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    X = df[feature_cols].values.astype(np.float32)
    y_proba = model.predict_proba(X)[:, 1]
    signals = y_proba >= 0.5

    print("Führe Full Backtest durch (alle Daten)...")
    engine = BacktestEngine(
        tp_points=TARGET_PARAMS["tp_points"],
        sl_points=TARGET_PARAMS["sl_points"],
        horizon=TARGET_PARAMS["horizon"],
    )
    
    trades_df, stats = engine.run(df, signals)
    print(f"  Trades: {stats['n_trades']} | PF: {stats['profit_factor']:.2f} | Profit: {stats['total_profit']:.0f}")

    # Mark periods
    train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)

    # Assign period to trades
    trades_df["period"] = "unknown"
    trades_df.loc[trades_df["entry_idx"].map(lambda i: train_mask.iloc[i] if i < len(train_mask) else False), "period"] = "train"
    trades_df.loc[trades_df["entry_idx"].map(lambda i: val_mask.iloc[i] if i < len(val_mask) else False), "period"] = "val"
    trades_df.loc[trades_df["entry_idx"].map(lambda i: test_mask.iloc[i] if i < len(test_mask) else False), "period"] = "test"

    # Sort and compute equity curve per period
    trades_sorted = trades_df.sort_values("entry_idx").reset_index(drop=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. Equity Curve (gesamt)
    equity = trades_sorted["pnl"].cumsum()
    timestamps = trades_sorted["entry_time"].values
    
    ax1 = axes[0, 0]
    ax1.plot(timestamps, equity.values, linewidth=0.5, color="blue")
    
    # Highlight test period
    test_start_ts = pd.Timestamp(TEST_START)
    ax1.axvline(test_start_ts, color="red", linestyle="--", alpha=0.7, label="OOS Test Start")
    ax1.axvline(pd.Timestamp(TRAIN_END), color="orange", linestyle="--", alpha=0.5, label="Validation Start")
    
    ax1.set_title("XGBoost Equity Curve (alle Trades)")
    ax1.set_ylabel("Kumulative P&L (Punkte)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    running_max = equity.expanding().max()
    drawdown = equity - running_max
    
    ax2 = axes[0, 1]
    ax2.fill_between(timestamps, drawdown.values, 0, color="red", alpha=0.3)
    ax2.plot(timestamps, drawdown.values, color="red", linewidth=0.5)
    ax2.axvline(test_start_ts, color="black", linestyle="--", alpha=0.5)
    ax2.set_title("Drawdown-Kurve")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)

    # 3. Monatliche Performance
    trades_sorted["month"] = trades_sorted["entry_time"].dt.to_period("M").astype(str)
    monthly = trades_sorted.groupby("month").agg(
        profit=("pnl", "sum"),
        n_trades=("pnl", "count"),
    ).reset_index()
    # Calc PF separately
    pf_monthly = trades_sorted.groupby("month").apply(
        lambda x: x[x["pnl"] > 0]["pnl"].sum() / abs(x[x["pnl"] < 0]["pnl"].sum())
        if x[x["pnl"] < 0]["pnl"].sum() != 0 else 0
    )
    monthly["pf"] = monthly["month"].map(pf_monthly)
    
    ax3 = axes[1, 0]
    colors = ["green" if p >= 0 else "red" for p in monthly["profit"]]
    ax3.bar(range(len(monthly)), monthly["profit"], color=colors, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.axvline(x=sum(monthly["month"] < VAL_START), color="orange", linestyle="--", alpha=0.5)
    ax3.axvline(x=sum(monthly["month"] < TEST_START), color="red", linestyle="--", alpha=0.5)
    ax3.set_xticks(range(0, len(monthly), 3))
    ax3.set_xticklabels([monthly["month"].iloc[i] for i in range(0, len(monthly), 3)], rotation=45)
    ax3.set_title("Monatliche Performance")
    ax3.set_ylabel("Profit (Punkte)")
    ax3.grid(True, alpha=0.3)

    # 4. Period Performance Comparison
    period_stats = trades_sorted.groupby("period").agg(
        n_trades=("pnl", "count"),
        total_profit=("pnl", "sum"),
    )
    period_stats["win_rate"] = trades_sorted.groupby("period").apply(
        lambda x: (x["pnl"] > 0).mean()
    )
    period_stats["pf"] = trades_sorted.groupby("period").apply(
        lambda x: x[x["pnl"] > 0]["pnl"].sum() / abs(x[x["pnl"] < 0]["pnl"].sum())
        if x[x["pnl"] < 0]["pnl"].sum() != 0 else 0
    )
    
    ax4 = axes[1, 1]
    periods = ["train", "val", "test"]
    profs = [period_stats.loc[p, "pf"] if p in period_stats.index else 0 for p in periods]
    bars = ax4.bar(periods, profs, color=["steelblue", "orange", "red"], alpha=0.7)
    ax4.set_title("Profit Factor nach Periode")
    ax4.set_ylabel("Profit Factor")
    ax4.grid(True, alpha=0.3)
    for bar, val in zip(bars, profs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center")

    # 5. TP/SL Exit Distribution
    exit_reasons = trades_sorted["exit_reason"].value_counts()
    
    ax5 = axes[2, 0]
    ax5.pie(exit_reasons.values, labels=exit_reasons.index, autopct="%1.1f%%", startangle=90)
    ax5.set_title("Exit-Verteilung")

    # 6. Feature Importance
    fi = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "feature_importance_xgboost.csv"))
    fi_top = fi.head(15).iloc[::-1]  # Reverse for horizontal bar
    
    ax6 = axes[2, 1]
    ax6.barh(range(len(fi_top)), fi_top["importance"], color="steelblue")
    ax6.set_yticks(range(len(fi_top)))
    ax6.set_yticklabels(fi_top["feature"], fontsize=8)
    ax6.set_title("Top 15 Feature Importance")
    ax6.set_xlabel("Importance")
    ax6.grid(True, alpha=0.3)

    plt.suptitle("XAUUSD XGBoost M1 - OOS Backtest Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = os.path.join(REPORTS_DIR, "xgboost_backtest_report.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Report gespeichert: {plot_path}")

    # Print period comparison
    print(f"\nPeriod Performance:")
    for p in periods:
        if p in period_stats.index:
            print(f"  {p}: Trades={period_stats.loc[p, 'n_trades']} "
                  f"Profit={period_stats.loc[p, 'total_profit']:.0f} "
                  f"WinRate={period_stats.loc[p, 'win_rate']*100:.1f}% "
                  f"PF={period_stats.loc[p, 'pf']:.2f}")

    # Save full stats
    import json
    with open(os.path.join(REPORTS_DIR, "xgboost_backtest_stats.json"), "w") as f:
        json.dump({k: str(v) for k, v in stats.items()}, f, indent=2)

    return trades_df, stats


if __name__ == "__main__":
    main()
