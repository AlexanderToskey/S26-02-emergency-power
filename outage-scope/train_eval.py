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
    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    print("Loading weather data...")
    weather_path = data_dir / "noaa_storm_events_va_2014_2022.csv"
    weather = load_noaa_weather(weather_path)

    print("Merging with NOAA Storm Events...")
    merged = merge_weather_outages(outages, weather)

    print("Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")

    print("Merging with GHCN-Daily daily weather...")
    merged = merge_ghcnd_weather(merged, ghcnd)

    # --- 2. Preprocessing & Leakage Control ---
    print("\nRunning preprocessing pipeline...")
    # Get all potential features from the preprocessor
    X_full, y = run_full_pipeline(merged)

    # TOGGLE: Set to False to test if the model actually generalizes weather patterns.
    # Set to True if you want to use initial outage growth as a predictor.
    INCLUDE_DYNAMIC_FEATURES = False 

    if not INCLUDE_DYNAMIC_FEATURES:
        print("[!] GENERALIZATION MODE: Removing initial 15m growth features to prevent leakage.")
        static_features = [
            "fips_code", "month", "hour", "dayofweek",
            "has_weather_event", "max_magnitude", "magnitude_missing",
            "prcp_mm", "snow_mm", "snwd_mm", "tmax_c", "tmin_c", "awnd_ms",
            "wt_fog", "wt_thunder", "wt_ice", "wt_blowing_snow", "wt_freezing_rain", "wt_snow"
        ]
        # Keep static features + any one-hot encoded 'event_' columns
        event_cols = [c for c in X_full.columns if c.startswith("event_")]
        X = X_full[static_features + event_cols]
    else:
        print("[!] LEAKAGE MODE: Including initial 15-minute growth features.")
        X = X_full

    # --- 3. Training ---
    print("\nSplitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training OutageScopeModel on {X.shape[1]} features...")
    model = OutageScopeModel()

    # Use largeWeight to prioritize learning from high-impact events
    model.train(X_train, y_train, largeWeight=4.0)

    # --- 4. Evaluation ---
    print("\nEvaluating results...")
    preds = model.predict(X_test)

    metrics = evaluateModel(y_test.values, preds)
    printEvaluationReport(metrics)

    print("\nTop 15 Feature Importances (XGBoost Native):")
    importances = model.getFeatureImportances()
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {k:30s} {v:.4f}")

    # --- 5. Explainer (Optional) ---
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