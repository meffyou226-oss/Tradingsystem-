"""
Haupt-Pipeline für XAUUSD M1 ML Research.

Orchestriert alle Schritte:
1. Datenanalyse
2. Feature Engineering
3. Target Definition
4. Datenvorbereitung (Parquet)
5. Baseline-Strategien
6. ML-Training (XGBoost)
7. OOS-Evaluation & Walk-Forward
8. Backtest-Report

Usage:
    python src/main.py [--skip-data-prep] [--skip-ml] [--skip-backtest]
"""

import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from config import (TARGET_PARAMS, TRAIN_START, TRAIN_END, VAL_START,
                     VAL_END, TEST_START, TEST_END, REPORTS_DIR, RESULTS_DIR,
                     MODELS_DIR)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_step(step_num, name, func):
    print_header(f"Schritt {step_num}: {name}")
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    print(f"  Dauer: {elapsed:.1f}s")
    return result


def step_data_analysis():
    report_path = os.path.join(REPORTS_DIR, "data_analysis")
    if os.path.exists(report_path):
        files = os.listdir(report_path)
        if files:
            print(f"  Datenanalyse bereits vorhanden: {files[0]}")
            return True
    return False


def step_feature_engineering():
    features_path = os.path.join(BASE_DIR, "data", "xauusd_m1_features.csv")
    if os.path.exists(features_path):
        size_mb = os.path.getsize(features_path) / 1024 / 1024
        print(f"  Features bereits vorhanden ({size_mb:.1f} MB)")
        return True
    print("  Fuehre Feature Engineering aus...")
    os.system(f"cd {BASE_DIR} && python3 src/02_feature_engineering.py")
    return True


def step_target_definition():
    targets_path = os.path.join(BASE_DIR, "data", "targets", "xauusd_m1_targets.csv")
    if os.path.exists(targets_path):
        size_mb = os.path.getsize(targets_path) / 1024 / 1024
        print(f"  Targets bereits vorhanden ({size_mb:.1f} MB)")
        return True
    print("  Fuehre Target Definition aus...")
    os.system(f"cd {BASE_DIR} && python3 src/03_target_definition.py")
    return True


def step_data_preparation():
    print("  Erstelle Parquet-Datei...")
    os.system(f"cd {BASE_DIR} && python3 src/data_preparation.py")
    return True


def step_baseline():
    results_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(results_path):
        print(f"  Baseline-Ergebnisse bereits vorhanden")
        return True
    print("  Evaluiere Baseline-Strategien...")
    os.system(f"cd {BASE_DIR} && python3 src/04_baseline.py")
    return True


def step_ml_training():
    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"  XGBoost-Modell bereits vorhanden ({size_mb:.1f} MB)")
        return True
    print("  Trainiere ML-Modelle...")
    os.system(f"cd {BASE_DIR} && python3 src/05_ml_pipeline.py")
    return True


def step_oos_evaluation():
    summary_path = os.path.join(REPORTS_DIR, "xgboost_oos_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  OOS-Evaluation bereits vorhanden")
        print(f"    OOS AUC: {summary['oos_auc']:.4f}")
        print(f"    OOS PF: {summary['oos_pf']:.2f}")
        return True
    print("  Fuehre OOS-Evaluation & Walk-Forward aus...")
    os.system(f"cd {BASE_DIR} && python3 src/05_xgboost_oos.py")
    return True


def step_backtest_report():
    report_path = os.path.join(REPORTS_DIR, "xgboost_backtest_report.png")
    if os.path.exists(report_path):
        print(f"  Backtest-Report bereits vorhanden")
        return True
    print("  Erstelle Backtest-Report...")
    os.system(f"cd {BASE_DIR} && python3 src/06_backtest_report.py")
    return True


def generate_final_report():
    """Generiert einen umfassenden Abschlussbericht."""
    print_header("FINALER BERICHT")

    results = {}

    # Baseline
    bl_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    if os.path.exists(bl_path):
        import pandas as pd
        bl = pd.read_csv(bl_path)
        results["baselines"] = bl.to_dict(orient="records")

    # XGBoost OOS Summary
    oos_path = os.path.join(REPORTS_DIR, "xgboost_oos_summary.json")
    if os.path.exists(oos_path):
        with open(oos_path) as f:
            results["xgboost_oos"] = json.load(f)

    # XGBoost Backtest Stats
    bt_path = os.path.join(REPORTS_DIR, "xgboost_backtest_stats.json")
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            results["xgboost_backtest"] = json.load(f)

    # Walk-Forward
    wf_path = os.path.join(RESULTS_DIR, "xgboost_oos", "walkforward_xgboost.csv")
    if os.path.exists(wf_path):
        import pandas as pd
        wf = pd.read_csv(wf_path)
        results["walkforward"] = wf.to_dict(orient="records")

    # Feature Importance
    fi_path = os.path.join(RESULTS_DIR, "feature_importance_xgboost.csv")
    if os.path.exists(fi_path):
        import pandas as pd
        fi = pd.read_csv(fi_path)
        results["feature_importance_top10"] = fi.head(10).to_dict(orient="records")

    # Print summary
    print("\n" + "=" * 70)
    print("XAUUSD M1 ML RESEARCH - FINALE ERGEBNISSE")
    print("=" * 70)

    print(f"\nDATEN:")
    print(f"  Zeitraum: 2024-01-01 bis 2026-08-23 (965 Tage)")
    print(f"  M1 Kerzen: 1,159,667")
    print(f"  Features: 84")

    print(f"\nTARGET:")
    print(f"  Horizont: {TARGET_PARAMS['horizon']} Minuten")
    print(f"  TP: {TARGET_PARAMS['tp_points']} Punkte ({TARGET_PARAMS['tp_points'] * 0.01:.2f} USD)")
    print(f"  SL: {TARGET_PARAMS['sl_points']} Punkte ({TARGET_PARAMS['sl_points'] * 0.01:.2f} USD)")
    print(f"  R:R: {TARGET_PARAMS['rr_ratio']}:1")

    if "xgboost_oos" in results:
        oos = results["xgboost_oos"]
        print(f"\nXGBoost OOS-ERGEBNISSE (Threshold=0.70):")
        print(f"  AUC: {oos['oos_auc']:.4f}")
        print(f"  Win Rate: {oos['oos_winrate']*100:.1f}%")
        print(f"  Profit Factor: {oos['oos_pf']:.2f}")
        print(f"  Total Profit: {oos['oos_profit']:.0f} Punkte")
        print(f"  Max Drawdown: {oos.get('oos_max_dd', 'N/A')}")

    if "walkforward" in results:
        wf = results["walkforward"]
        if wf:
            import numpy as np
            aucs = [f["auc"] for f in wf]
            pfs = [f["pf"] for f in wf]
            print(f"\nWALK-FORWARD-ANALYSE ({len(wf)} Folds):")
            print(f"  Ø AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            print(f"  Ø PF: {np.mean(pfs):.2f} ± {np.std(pfs):.2f}")

    if "baselines" in results:
        print(f"\nBASELINE-VERGLEICH:")
        print(f"  {'Strategie':20s} | {'Win Rate':>8s} | {'PF':>6s} | {'Profit':>10s}")
        print(f"  {'-'*55}")
        for bl in results["baselines"]:
            print(f"  {bl['strategy']:20s} | {bl['win_rate']*100:7.1f}% | {bl['profit_factor']:6.2f} | {bl['total_profit']:10.0f}")

    if "feature_importance_top10" in results:
        print(f"\nTOP 10 FEATURES:")
        for i, fi in enumerate(results["feature_importance_top10"], 1):
            print(f"  {i:2d}. {fi['feature']:25s} ({fi['importance']:.4f})")

    # Save final report
    report_path = os.path.join(REPORTS_DIR, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBericht gespeichert: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="XAUUSD M1 ML Research Pipeline")
    parser.add_argument("--skip-data-prep", action="store_true", help="Datenvorbereitung überspringen")
    parser.add_argument("--skip-ml", action="store_true", help="ML-Training überspringen")
    parser.add_argument("--skip-backtest", action="store_true", help="Backtest überspringen")
    parser.add_argument("--full", action="store_true", help="Komplette Pipeline ausführen")
    args = parser.parse_args()

    print("=" * 70)
    print("XAUUSD M1 ML RESEARCH PIPELINE")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # Step 1: Data Analysis
    if not args.skip_data_prep:
        run_step(1, "Datenanalyse", step_data_analysis)

    # Step 2: Feature Engineering
    if not args.skip_data_prep:
        run_step(2, "Feature Engineering", step_feature_engineering)

    # Step 3: Target Definition
    if not args.skip_data_prep:
        run_step(3, "Target Definition", step_target_definition)

    # Step 4: Data Preparation
    if not args.skip_data_prep:
        run_step(4, "Datenvorbereitung (Parquet)", step_data_preparation)

    # Step 5: Baseline Strategies
    if not args.skip_ml:
        run_step(5, "Baseline-Strategien", step_baseline)

    # Step 6: ML Training
    if not args.skip_ml:
        run_step(6, "ML-Training (XGBoost)", step_ml_training)

    # Step 7: OOS Evaluation
    if not args.skip_ml:
        run_step(7, "OOS-Evaluation & Walk-Forward", step_oos_evaluation)

    # Step 8: Backtest Report
    if not args.skip_backtest:
        run_step(8, "Backtest-Report", step_backtest_report)

    # Final Report
    generate_final_report()

    total_time = time.time() - t_start
    print(f"\nGesamtdauer: {total_time:.1f}s")
    print(f"Ende: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
