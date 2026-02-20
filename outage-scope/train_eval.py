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

    print("Running preprocessing...")
    # y is peak_customers_affected
    X, y = run_full_pipeline(merged)  

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model (log-transform handled internally)...")
    model = OutageScopeModel()

    # Train using raw customer counts; model handles log internally
    # largeWeight=2.0 upweights large outages (>= 500 customers) 
    model.train(X_train, y_train, largeWeight=2.0)

    print("Evaluating...")
    preds = model.predict(X_test)  # returns customer counts

    metrics = evaluateModel(y_test.values, preds)
    printEvaluationReport(metrics)

    print("\nTop feature importances (XGBoost Native):")
    importances = model.getFeatureImportances()
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{k:30s} {v:.4f}")

    # Hooking up the explainer
    # print("\nGenerating SHAP Explanations...")
    # try:
    #     explainer = OutageExplainer(model, X_train)
        
    #     # We'll use the test set for the summary plot to see how features 
    #     # drove the predictions on unseen data.
    #     explainer.plotSummary(X_test, savePath="shap_summary.png")
    #     print("SHAP summary plot saved successfully to 'shap_summary.png'.")
        
    # except Exception as e:
    #     print(f"Warning: Could not generate SHAP plots. Error: {e}")


if __name__ == "__main__":
    main()