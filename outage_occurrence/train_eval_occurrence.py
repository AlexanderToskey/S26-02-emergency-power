#Training and evaluation script for the outage occurrence model
#Builds county-day labels from EAGLE-I, merges GHCND weather, trains the stacking ensemble

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader_occurrence import (
    load_eagle_outages,
    load_ghcnd_weather,
    build_occurrence_labels,
    merge_occurrence_with_weather,
    summarize_class_balance,
)
from preprocessor_occurrence import run_full_pipeline
from occurrence_model import OutageOccurrenceModel
from evaluator_occurrence import evaluateModel, printEvaluationReport
from occurrence_explainer_model import OutageOccurrenceExplainer


def main():
    #Get the base directory and get the data and model directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / "models"

    #Loads outage data
    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    #Older EAGLE-I exports use 'customers_out'; the occurrence pipeline expects 'customers_affected'
    if 'customers_out' in outages.columns and 'customers_affected' not in outages.columns:
        outages = outages.rename(columns={'customers_out': 'customers_affected'})

    #Build county-day occurrence labels
    print("Building county-day occurrence labels...")
    #Min_customers=100 filters routine 1-5 customer line faults
    occurrence = build_occurrence_labels(outages, min_customers=100)

    #Loads daily weather
    print("Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")

    #Merges labels with weather
    print("Merging occurrence labels with weather...")
    merged = merge_occurrence_with_weather(occurrence, ghcnd)

    #Merge county historical stats so the model can use them instead of fips_code
    print("Merging county historical stats...")
    county_stats = pd.read_csv(data_dir / "county_stats.csv")
    county_stats["fips_code"] = county_stats["fips_code"].astype(str).str.zfill(5)
    merged["fips_code"] = merged["fips_code"].astype(str).str.zfill(5)
    merged = merged.merge(county_stats, on="fips_code", how="left")

    summarize_class_balance(merged)

    #Preprocessing
    print("Running preprocessing...")
    X, y = run_full_pipeline(merged)  #y = outage_occurred

    #Remove year column to avoid SHAP/feature mismatch
    #Year would let the model memorize annual trends instead of learning weather patterns
    if "year" in X.columns:
        print("[preprocessor] Dropping 'year' from features to avoid single-use leakage...")
        X = X.drop(columns=["year"])


    print("\nExporting a small sample of the preprocessed data...")
    
    #Recombine features (X) and target (y) so they are in one file
    sample_export = X.copy()
    sample_export['target_occurrence'] = y
    
    #Grab a random sample of 100 rows
    small_sample = sample_export.sample(n=100, random_state=42)
    
    #Save it to the data directory
    sample_path = data_dir / "preprocessed_sample_occurrence.csv"
    small_sample.to_csv(sample_path, index=False)
    print(f"Saved 100 sample rows to {sample_path}")

    #Train/Test split
    print("Splitting train/test...")
    #Stratify ensures both splits have the same ~10% positive rate, not just by chance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    

    #Train models
    print("Training outage occurrence model...")
    model = OutageOccurrenceModel()

    #Occurrence models usually support class_weight or scale_pos_weight
    model.train(X_train, y_train)

    #Tune after training to find the F2-optimal cutoff on held-out data
    model.tune_threshold(X_test, y_test)

    #Evaluate
    print("Evaluating...")
    preds, y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluateModel(y_test.values, y_pred, y_prob)
    printEvaluationReport(metrics)

    #Save Model to Central Directory
    print("\nSaving occurrence model...")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "occurrence_ensemble.joblib")    #renamed to reflect stacking ensemble, not a single model
    print(f"Saved to {model_dir / 'occurrence_model.joblib'}")


if __name__ == "__main__":
    main()