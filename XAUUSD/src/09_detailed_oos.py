"""
Detaillierter Out-of-Sample Test für XAUUSD M1.

1. Rolling Window OOS (6 Monate Rollen)
2. Regime-basierte Analyse (Trend/Range/HighVol/LowVol)
3. Statistische Signifikanz (t-test, Sharpe-Test)
4. Monte Carlo Simulation
5. Drawdown-Analyse
6. Stabilitäts-Test (Parameter-Sensitivität)
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FEATURE_COLUMNS, TARGET_COLUMN, TARGET_PARAMS,
                     TRAIN_START, TRAIN_END, VAL_START, VAL_END,
                     TEST_START, TEST_END, MODELS_DIR, RESULTS_DIR,
                     REPORTS_DIR, POINT)
from backtest_engine import BacktestEngine
from data_preparation import load_combined


def generate_signals(df, model, feature_cols, conf_threshold=0.75, max_daily=8):
    X = df[feature_cols].values.astype(np.float32)
    predictions = model.predict_proba(X)[:, 1]

    hour = df["timestamp"].dt.hour.values
    is_active = ((hour >= 8) & (hour < 16)) | ((hour >= 13) & (hour < 20))
    atr_norm = df["atr_14_norm"].values if "atr_14_norm" in df.columns else np.ones(len(df)) * 0.001
    has_vol = (atr_norm > 0.0003) & (atr_norm < 0.005)

    signals = np.zeros(len(df), dtype=bool)
    high_conf = predictions >= conf_threshold
    eligible = is_active & has_vol & high_conf

    current_date = None
    daily_trades = 0
    for i in range(len(df)):
        if not eligible[i]:
            continue
        date = df["timestamp"].iloc[i].date()
        if date != current_date:
            current_date = date
            daily_trades = 0
        if daily_trades < max_daily:
            signals[i] = True
            daily_trades += 1

    return signals, predictions


def rolling_window_oos(df, feature_cols, n_windows=6):
    print("\n" + "=" * 70)
    print("1. ROLLING WINDOW OOS-TEST")
    print("=" * 70)

    results = []
    start_date = pd.Timestamp(TRAIN_START)
    end_date = pd.Timestamp(TEST_END)

    for window in range(n_windows):
        train_start = start_date + pd.DateOffset(months=window * 3)
        train_end = train_start + pd.DateOffset(months=12)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=3)

        if test_end > end_date:
            break

        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 10000 or len(test_df) < 5000:
            continue

        from xgboost import XGBClassifier
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[TARGET_COLUMN].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[TARGET_COLUMN].values.astype(np.float32)

        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, tree_method="hist", eval_metric="logloss"
        )
        model.fit(X_train, y_train, verbose=False)

        signals, predictions = generate_signals(test_df, model, feature_cols)

        engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)
        trades_df, bt_stats = engine.run(test_df, signals)

        auc = 0.5
        if len(np.unique(y_test)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, predictions)

        if bt_stats:
            results.append({
                "window": window + 1,
                "train_period": f"{train_start.strftime('%Y-%m')} bis {train_end.strftime('%Y-%m')}",
                "test_period": f"{test_start.strftime('%Y-%m')} bis {test_end.strftime('%Y-%m')}",
                "n_trades": bt_stats["n_trades"],
                "win_rate": bt_stats["win_rate"],
                "profit_factor": bt_stats["profit_factor"],
                "total_profit": bt_stats["total_profit"],
                "max_drawdown": bt_stats["max_drawdown"],
                "auc": auc,
            })
            print(f"  Window {window+1}: Train={results[-1]['train_period']} | "
                  f"Test={results[-1]['test_period']}")
            print(f"    Trades={bt_stats['n_trades']} | Win={bt_stats['win_rate']*100:.1f}% | "
                  f"PF={bt_stats['profit_factor']:.2f} | Profit={bt_stats['total_profit']:.0f} | "
                  f"AUC={auc:.4f}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print(f"\n  Zusammenfassung ({len(results_df)} Windows):")
        print(f"    OE Win Rate: {results_df['win_rate'].mean()*100:.1f}% +/- {results_df['win_rate'].std()*100:.1f}%")
        print(f"    OE Profit Factor: {results_df['profit_factor'].mean():.2f} +/- {results_df['profit_factor'].std():.2f}")
        print(f"    OE AUC: {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")
        print(f"    Profitable Windows: {(results_df['total_profit'] > 0).sum()}/{len(results_df)}")

    return results_df


def regime_analysis(df, feature_cols):
    print("\n" + "=" * 70)
    print("2. REGIME-BASIERTE ANALYSE")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    signals, predictions = generate_signals(test_df, model, feature_cols)

    atr_norm = test_df["atr_14_norm"].values if "atr_14_norm" in test_df.columns else np.ones(len(test_df)) * 0.001
    adx = test_df["adx_14"].values if "adx_14" in test_df.columns else np.ones(len(test_df)) * 20

    atr_median = np.nanmedian(atr_norm)
    adx_median = np.nanmedian(adx)

    regimes = {
        "High_Trend": (adx > adx_median) & (atr_norm > atr_median),
        "Low_Trend": (adx > adx_median) & (atr_norm <= atr_median),
        "High_Range": (adx <= adx_median) & (atr_norm > atr_median),
        "Low_Range": (adx <= adx_median) & (atr_norm <= atr_median),
    }

    engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)

    regime_results = []
    for regime_name, regime_mask in regimes.items():
        regime_signals = signals & regime_mask
        n_regime_candles = regime_mask.sum()
        n_signals = regime_signals.sum()

        if n_signals < 10:
            continue

        trades_df, bt_stats = engine.run(test_df, regime_signals)

        if bt_stats:
            regime_results.append({
                "regime": regime_name,
                "n_candles": int(n_regime_candles),
                "n_trades": bt_stats["n_trades"],
                "win_rate": bt_stats["win_rate"],
                "profit_factor": bt_stats["profit_factor"],
                "total_profit": bt_stats["total_profit"],
                "max_drawdown": bt_stats["max_drawdown"],
            })
            print(f"  {regime_name:15s}: {n_regime_candles:6d} Kerzen | "
                  f"{bt_stats['n_trades']:4d} Trades | "
                  f"Win={bt_stats['win_rate']*100:.1f}% | "
                  f"PF={bt_stats['profit_factor']:.2f} | "
                  f"Profit={bt_stats['total_profit']:.0f}")

    return pd.DataFrame(regime_results)


def statistical_significance(df, feature_cols):
    print("\n" + "=" * 70)
    print("3. STATISTISCHE SIGNIFIKANZ")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    signals, predictions = generate_signals(test_df, model, feature_cols)

    engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)
    trades_df, bt_stats = engine.run(test_df, signals)

    if not bt_stats:
        return {}

    pnl = trades_df["pnl"].values

    # t-Test
    t_stat, p_value = stats.ttest_1samp(pnl, 0)
    p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Sharpe Ratio Test
    sharpe = bt_stats["sharpe_ratio"]
    n = len(pnl)
    sharpe_se = np.sqrt((1 + sharpe**2 / 2) / n)
    sharpe_z = sharpe / sharpe_se
    sharpe_p = 1 - stats.norm.cdf(sharpe_z)

    # Profit Factor Bootstrap
    n_bootstrap = 10000
    pf_bootstrap = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(pnl, size=n, replace=True)
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        pf = wins / losses if losses > 0 else float("inf")
        pf_bootstrap.append(pf)
    pf_ci = np.percentile(pf_bootstrap, [2.5, 97.5])

    # Random-Strategie Vergleich
    random_pnls = []
    for _ in range(1000):
        p_signal = signals.sum() / len(test_df)
        random_signals = np.random.choice([True, False], size=len(test_df), p=[p_signal, 1-p_signal])
        if random_signals.sum() > 10:
            _, random_bt = engine.run(test_df, random_signals)
            if random_bt:
                random_pnls.append(random_bt["total_profit"])

    random_mean = np.mean(random_pnls) if random_pnls else 0
    random_std = np.std(random_pnls) if random_pnls else 1
    outperformance = (bt_stats["total_profit"] - random_mean) / random_std if random_std > 0 else 0

    print(f"  t-Test (P&L > 0):")
    print(f"    t-Stat: {t_stat:.3f}")
    print(f"    p-Wert (one-tailed): {p_value_one_tailed:.6f}")
    print(f"    Signifikant (p<0.05): {'JA' if p_value_one_tailed < 0.05 else 'NEIN'}")

    print(f"\n  Sharpe Ratio Test:")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Z-Score: {sharpe_z:.2f}")
    print(f"    p-Wert: {sharpe_p:.6f}")
    print(f"    Signifikant (p<0.05): {'JA' if sharpe_p < 0.05 else 'NEIN'}")

    print(f"\n  Profit Factor Bootstrap (95% CI):")
    print(f"    PF: {bt_stats['profit_factor']:.2f}")
    print(f"    95% CI: [{pf_ci[0]:.2f}, {pf_ci[1]:.2f}]")

    print(f"\n  Random-Strategie Vergleich:")
    print(f"    XGBoost Profit: {bt_stats['total_profit']:.0f}")
    print(f"    Random OE Profit: {random_mean:.0f} +/- {random_std:.0f}")
    print(f"    Outperformance (Z): {outperformance:.2f} Std-Devs")
    print(f"    Signifikant besser als Random: {'JA' if outperformance > 2 else 'NEIN'}")

    return {
        "t_test": {"t_stat": float(t_stat), "p_value": float(p_value_one_tailed)},
        "sharpe_test": {"sharpe": float(sharpe), "z_score": float(sharpe_z), "p_value": float(sharpe_p)},
        "pf_ci": [float(x) for x in pf_ci],
        "random_comparison": {"z_score": float(outperformance)},
    }


def monte_carlo_simulation(df, feature_cols, n_simulations=1000):
    print("\n" + "=" * 70)
    print("4. MONTE CARLO SIMULATION")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    signals, _ = generate_signals(test_df, model, feature_cols)

    engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)
    trades_df, bt_stats = engine.run(test_df, signals)

    if not bt_stats:
        return {}

    pnl = trades_df["pnl"].values
    n_trades = len(pnl)

    final_equity = []
    max_drawdowns = []

    for _ in range(n_simulations):
        shuffled_pnl = np.random.permutation(pnl)
        equity = shuffled_pnl.cumsum()
        final_equity.append(equity[-1])

        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_drawdowns.append(drawdown.min())

    final_equity = np.array(final_equity)
    max_drawdowns = np.array(max_drawdowns)

    print(f"  {n_simulations} Simulationen ({n_trades} Trades):")
    print(f"    OE Final Equity: {final_equity.mean():.0f} +/- {final_equity.std():.0f}")
    print(f"    5% VaR: {np.percentile(final_equity, 5):.0f}")
    print(f"    1% VaR: {np.percentile(final_equity, 1):.0f}")
    print(f"    OE Max Drawdown: {max_drawdowns.mean():.0f} +/- {max_drawdowns.std():.0f}")
    print(f"    Worst Case DD: {np.percentile(max_drawdowns, 1):.0f}")
    print(f"    P(Profit > 0): {(final_equity > 0).mean()*100:.1f}%")

    return {
        "equity_mean": float(final_equity.mean()),
        "equity_std": float(final_equity.std()),
        "var_5": float(np.percentile(final_equity, 5)),
        "var_1": float(np.percentile(final_equity, 1)),
        "dd_mean": float(max_drawdowns.mean()),
        "dd_std": float(max_drawdowns.std()),
        "prob_profit": float((final_equity > 0).mean()),
    }


def drawdown_analysis(df, feature_cols):
    print("\n" + "=" * 70)
    print("5. DRAWDOWN-ANALYSE")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    signals, _ = generate_signals(test_df, model, feature_cols)

    engine = BacktestEngine(tp_points=45, sl_points=15, horizon=5)
    trades_df, bt_stats = engine.run(test_df, signals)

    if not bt_stats:
        return {}

    trades_df = trades_df.sort_values("entry_idx").reset_index(drop=True)
    equity = trades_df["pnl"].cumsum()
    running_max = equity.expanding().max()
    drawdown = equity - running_max

    # Find drawdown periods
    in_dd = drawdown < 0
    dd_periods = []
    dd_start = None

    for i in range(len(drawdown)):
        if in_dd[i] and dd_start is None:
            dd_start = i
        elif not in_dd[i] and dd_start is not None:
            dd_periods.append({
                "start_idx": dd_start,
                "end_idx": i - 1,
                "duration": i - dd_start,
                "max_dd": float(drawdown[dd_start:i].min()),
            })
            dd_start = None

    if dd_start is not None:
        dd_periods.append({
            "start_idx": dd_start,
            "end_idx": len(drawdown) - 1,
            "duration": len(drawdown) - dd_start,
            "max_dd": float(drawdown[dd_start:].min()),
        })

    print(f"  Gesamt Drawdown-Perioden: {len(dd_periods)}")
    print(f"  OE Drawdown: {drawdown.min():.0f} Punkte")
    print(f"  OE Drawdown-Dauer: {max([p['duration'] for p in dd_periods]) if dd_periods else 0} Trades")

    if dd_periods:
        dd_df = pd.DataFrame(dd_periods).sort_values("max_dd")
        print(f"\n  Top 5 Drawdowns:")
        for i, row in dd_df.head(5).iterrows():
            start_idx = int(row["start_idx"])
            end_idx = int(row["end_idx"])
            start_time = trades_df.iloc[start_idx]["entry_time"]
            end_time = trades_df.iloc[end_idx]["entry_time"]
            print(f"    DD={row['max_dd']:.0f} | Dauer={int(row['duration'])} Trades | "
                  f"{start_time.strftime('%Y-%m-%d')} bis {end_time.strftime('%Y-%m-%d')}")

    return {
        "n_dd_periods": len(dd_periods),
        "max_dd": float(drawdown.min()),
        "avg_dd_duration": float(np.mean([p["duration"] for p in dd_periods])) if dd_periods else 0,
    }


def parameter_stability(df, feature_cols):
    print("\n" + "=" * 70)
    print("6. PARAMETER-STABILITAT")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    test_mask = (df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)
    test_df = df[test_mask].copy()

    # Teste verschiedene Kombinationen
    configs = [
        {"conf": 0.70, "max_daily": 5, "tp": 45, "sl": 15},
        {"conf": 0.75, "max_daily": 5, "tp": 45, "sl": 15},
        {"conf": 0.80, "max_daily": 5, "tp": 45, "sl": 15},
        {"conf": 0.75, "max_daily": 3, "tp": 45, "sl": 15},
        {"conf": 0.75, "max_daily": 8, "tp": 45, "sl": 15},
        {"conf": 0.75, "max_daily": 5, "tp": 50, "sl": 25},
        {"conf": 0.75, "max_daily": 5, "tp": 30, "sl": 15},
        {"conf": 0.75, "max_daily": 5, "tp": 60, "sl": 20},
    ]

    results = []
    for cfg in configs:
        signals, _ = generate_signals(test_df, model, feature_cols,
                                       conf_threshold=cfg["conf"],
                                       max_daily=cfg["max_daily"])
        engine = BacktestEngine(tp_points=cfg["tp"], sl_points=cfg["sl"], horizon=5)
        _, bt_stats = engine.run(test_df, signals)

        if bt_stats:
            results.append({
                "conf": cfg["conf"],
                "max_daily": cfg["max_daily"],
                "tp": cfg["tp"],
                "sl": cfg["sl"],
                "n_trades": bt_stats["n_trades"],
                "win_rate": bt_stats["win_rate"],
                "profit_factor": bt_stats["profit_factor"],
                "total_profit": bt_stats["total_profit"],
                "max_drawdown": bt_stats["max_drawdown"],
            })

    results_df = pd.DataFrame(results)
    print(f"  {'Conf':>5s} | {'MaxD':>4s} | {'TP':>4s} | {'SL':>4s} | {'Trades':>7s} | {'Win%':>6s} | {'PF':>5s} | {'Profit':>7s}")
    print("  " + "-" * 65)
    for _, row in results_df.iterrows():
        print(f"  {row['conf']:5.2f} | {int(row['max_daily']):4d} | "
              f"{int(row['tp']):4d} | {int(row['sl']):4d} | "
              f"{int(row['n_trades']):7d} | {row['win_rate']*100:5.1f}% | "
              f"{row['profit_factor']:5.2f} | {row['total_profit']:7.0f}")

    # Stability metrics
    pf_std = results_df["profit_factor"].std()
    pf_mean = results_df["profit_factor"].mean()
    print(f"\n  Stabilitaet:")
    print(f"    PF Mittelwert: {pf_mean:.2f}")
    print(f"    PF Std-Dev: {pf_std:.2f}")
    print(f"    PF CV (Std/Mean): {pf_std/pf_mean:.2f}")
    print(f"    Alle Konfigurationen profitabel: {'JA' if (results_df['total_profit'] > 0).all() else 'NEIN'}")

    return results_df


def main():
    print("=" * 70)
    print("DETAILLIERTER OUT-OF-SAMPLE TEST")
    print("=" * 70)

    df = load_combined()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # 1. Rolling Window
    rolling_results = rolling_window_oos(df, feature_cols)

    # 2. Regime-Analyse
    regime_results = regime_analysis(df, feature_cols)

    # 3. Statistische Signifikanz
    stat_results = statistical_significance(df, feature_cols)

    # 4. Monte Carlo
    mc_results = monte_carlo_simulation(df, feature_cols)

    # 5. Drawdown-Analyse
    dd_results = drawdown_analysis(df, feature_cols)

    # 6. Parameter-Stabilitaet
    stability_results = parameter_stability(df, feature_cols)

    # Speichern
    all_results = {
        "rolling_window": rolling_results.to_dict(orient="records") if len(rolling_results) > 0 else [],
        "regime_analysis": regime_results.to_dict(orient="records") if len(regime_results) > 0 else [],
        "statistical_tests": stat_results,
        "monte_carlo": mc_results,
        "drawdown": dd_results,
        "parameter_stability": stability_results.to_dict(orient="records") if len(stability_results) > 0 else [],
    }

    results_path = os.path.join(REPORTS_DIR, "detailed_oos_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAlle Ergebnisse gespeichert: {results_path}")


if __name__ == "__main__":
    main()
