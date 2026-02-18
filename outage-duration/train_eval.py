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
from src.model import OutageDurationModel
from src.evaluator import evaluateModel, printEvaluationReport


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
    X, y = run_full_pipeline(merged)  # y is duration in minutes

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model (log-duration handled internally)...")
    model = OutageDurationModel()

    # Train using raw minutes; model handles log internally
    # longWeight=2.0 upweights outages >= 4hr to improve long outage accuracy
    model.train(X_train, y_train, longWeight=2.0)

    print("Evaluating...")
    preds = model.predict(X_test)  # already returns minutes

    metrics = evaluateModel(y_test.values, preds)
    printEvaluationReport(metrics)

    print("\nTop feature importances:")
    importances = model.getFeatureImportances()
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{k:30s} {v:.4f}")


if __name__ == "__main__":
    main()
