"""
train_eval.py - Train and evaluate the Outage Scope Model
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_eagle_outages,
    load_noaa_weather,
    merge_weather_outages,
    load_ghcnd_weather,
    merge_ghcnd_weather,
)
from src.preprocessor import run_full_pipeline
from src.model import OutageScopeModel
from src.evaluator import evaluateModel, printEvaluationReport
from src.explainer import OutageExplainer


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
    
    # Create masks based on the 'year' column we just added to the pipeline
    train_mask = X_full['year'] < TEST_YEAR
    test_mask = X_full['year'] == TEST_YEAR

    # Define features to use for training (preventing memorization)
    INCLUDE_DYNAMIC_FEATURES = False 
    
    # We use 'year' for the split, but we MUST drop it before training 
    # so the model doesn't "memorize" specific years.
    cols_to_drop = ['year']
    if not INCLUDE_DYNAMIC_FEATURES:
        print("[!] GENERALIZATION MODE: Removing growth features.")
        cols_to_drop += ["initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m"]

    X = X_full.drop(columns=cols_to_drop)

    # Apply the split manually
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples (Year {TEST_YEAR}): {len(X_test):,}")

    # --- 4. Training ---
    # NOTE: Do NOT call train_test_split here, or it will overwrite the temporal split.
    print(f"\nTraining OutageScopeModel on {X.shape[1]} features...")
    model = OutageScopeModel()
    model.train(X_train, y_train, largeWeight=4.0)

    # --- 5. Evaluation ---
    print("\nEvaluating results on 2022 data...")
    preds = model.predict(X_test)
    metrics = evaluateModel(y_test.values, preds)
    printEvaluationReport(metrics)

    print("\nTop 15 Feature Importances:")
    importances = model.getFeatureImportances()
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {k:30s} {v:.4f}")

    # --- 6. Explainer (Optional) ---
    # Uncomment the block below to generate the SHAP summary plot
    """
    print("\nGenerating SHAP Explanations...")
    try:
        explainer = OutageExplainer(model, X_train)
        explainer.plotSummary(X_test, savePath="shap_summary.png")
        print("SHAP summary plot saved to 'shap_summary.png'.")
    except Exception as e:
        print(f"Warning: Could not generate SHAP plots. Error: {e}")
    """

if __name__ == "__main__":
    main()