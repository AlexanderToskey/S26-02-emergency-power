"""
train_eval.py - Oracle Evaluation for the Two-Stage Outage Scope Model
Tests the "Theoretical Maximum" of the specialists by using perfect routing.
"""

import numpy as np
from pathlib import Path

from src.data_loader import (
    load_eagle_outages,
    load_noaa_weather,
    merge_weather_outages,
    load_ghcnd_weather,
    merge_ghcnd_weather,
)
from src.preprocessor import run_full_pipeline
from src.two_stage_model import TwoStageOutageModel
from src.evaluator import evaluateModel, printEvaluationReport


def main():
    data_dir = Path("data")

    # --- 1. Data Ingestion ---
    print("\nLoading data and merging...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)
    weather = load_noaa_weather(data_dir / "noaa_storm_events_va_2014_2022.csv")
    merged = merge_weather_outages(outages, weather)
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")
    merged = merge_ghcnd_weather(merged, ghcnd)

    # --- 2. Preprocessing ---
    print("\nRunning preprocessing pipeline...")
    X_full, y = run_full_pipeline(merged)

    # --- 3. Temporal Split Logic ---
    TEST_YEAR = 2022
    print(f"\n[!] TEMPORAL SPLIT: Training on 2014-{TEST_YEAR-1}, Testing on {TEST_YEAR}")
    
    train_mask = X_full['year'] < TEST_YEAR
    test_mask = X_full['year'] == TEST_YEAR

    INCLUDE_DYNAMIC_FEATURES = False 
    cols_to_drop = ['year']
    if not INCLUDE_DYNAMIC_FEATURES:
        print("[!] GENERALIZATION MODE: Removing growth features.")
        cols_to_drop += ["initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m"]

    X = X_full.drop(columns=cols_to_drop)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples (Year {TEST_YEAR}): {len(X_test):,}")

    THRESHOLD = 500.0

    # --- 4. Training ---
    print("\n" + "="*50)
    print("TRAINING SPECIALIZED SCOPE REGRESSORS")
    print("="*50)
    
    # Initialize and train the Two-Stage architecture
    model = TwoStageOutageModel(thresholdMin=THRESHOLD)
    model.train(X_train, y_train, validationSplit=0.2)

    # --- 5. ORACLE EVALUATION (Bypassing Stage 1) ---
    print("\n" + "="*50)
    print("ORACLE EVALUATION: PERFECT ROUTING")
    print("="*50)

    # Define the Oracle Masks based on ACTUAL values in the test set
    test_large_mask = y_test >= THRESHOLD
    test_small_mask = y_test < THRESHOLD

    # Initialize final prediction array
    final_preds = np.empty_like(y_test, dtype=float)

    # 5a. Small Specialist Oracle Run
    if test_small_mask.sum() > 0:
        print(f"Routing {test_small_mask.sum():,} actual small events to Small Specialist...")
        # We access the sub-regressor directly to skip the internal classifier
        # Note: Regressor predicts log1p, so we must expm1 it.
        pred_log_small = model.shortRegressor.predict(X_test[test_small_mask])
        final_preds[test_small_mask] = np.expm1(np.clip(pred_log_small, -20, 20))

    # 5b. Large Specialist Oracle Run
    if test_large_mask.sum() > 0:
        print(f"Routing {test_large_mask.sum():,} actual large events to Large Specialist...")
        pred_log_large = model.longRegressor.predict(X_test[test_large_mask])
        final_preds[test_large_mask] = np.expm1(np.clip(pred_log_large, -20, 20))

    # --- 6. Final Oracle Report ---
    print("\n" + "="*50)
    print("THEORETICAL MAXIMUM PIPELINE PERFORMANCE")
    print("="*50)
    
    metrics = evaluateModel(y_test.values, final_preds)
    printEvaluationReport(metrics)

    print("\nTop Features for Large Specialist (Oracle Context):")
    l_imp = model.getLongRegressorImportances()
    for k, v in sorted(l_imp.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {k:30s} {v:.4f}")

    # --- 7. Explainer (Optional) ---
    """
    print("\nGenerating SHAP Explanations...")
    try:
        # Note: Explainer currently only supports single-stage XGB models. 
        # You would need to pass model.classifier or model.longRegressor directly.
        explainer = OutageExplainer(model.classifier, X_train)
        explainer.plotSummary(X_test, savePath="shap_summary.png")
        print("SHAP summary plot saved to 'shap_summary.png'.")
    except Exception as e:
        print(f"Warning: Could not generate SHAP plots. Error: {e}")
    """

if __name__ == "__main__":
    main()
