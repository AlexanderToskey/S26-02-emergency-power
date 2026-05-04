#This file trains a simple Isolation Forest anomaly detector on the merged occurrence/weather dataset

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#Helper functions for loading and preprocessing the data
from outage_occurrence.data_loader_occurrence import (
    load_eagle_outages,
    load_ghcnd_weather,
    build_occurrence_labels,
    merge_occurrence_with_weather,
    summarize_class_balance,
)
from outage_occurrence.preprocessor_occurrence import run_full_pipeline
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


def main():
    # Get the base directory and get the data and model directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / "models"

    #Load outage data
    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    # Fix column naming for standalone training
    # Some older CSVs might have 'customers_out', but build_occurrence_labels expects 'customers_affected'
    if 'customers_out' in outages.columns and 'customers_affected' not in outages.columns:
        outages = outages.rename(columns={'customers_out': 'customers_affected'})

    #Builds county-day occurrence labels
    print("Building county-day occurrence labels...")
    occurrence = build_occurrence_labels(outages)

    #Loads daily weather
    print("Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")

    #Merge labels with weather
    print("Merging occurrence labels with weather...")
    merged = merge_occurrence_with_weather(occurrence, ghcnd)

    summarize_class_balance(merged)

    #Preprocessing pipeline
    print("Running preprocessing...")
    X, y = run_full_pipeline(merged)  #y = outage_occurred (0/1)

    #Remove year column to avoid SHAP/feature mismatch
    if "year" in X.columns:
        print("[preprocessor] Dropping 'year' from features to avoid single-use leakage...")
        X = X.drop(columns=["year"])

    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)

    #Save trained model and threshold for later use in inference
    joblib.dump(model, model_dir / "isolation_forest.joblib")
    print("Tier 2 Anomaly Detector saved")


if __name__ == "__main__":
    main()
