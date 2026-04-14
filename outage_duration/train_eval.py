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
from src.two_stage_model import TwoStageOutageModel
from src.evaluator import (
    evaluateModel,
    evaluateClassifier,
    printEvaluationReport,
    printTwoStageReport,
)

# set True to run the two-stage model (+ single-stage baseline for comparison)
# set False to run only the single-stage model
USE_TWO_STAGE = True


def main():
    data_dir = Path("../data")

    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    print("Loading weather data...")
    weather_path = data_dir / "noaa_storm_events_va_2014_2022.csv"
    weather = load_noaa_weather(weather_path)

    print("Merging with NOAA Storm Events...")
    merged = merge_weather_outages(outages, weather)

    print("Loading GHCN-Daily weather data...")
    # ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")
    ghcnd = load_ghcnd_weather(data_dir / "openmeteo_va_historical.csv")


    print("\n[DEBUG] Verifying GHCN-Daily Feature Presence...")
    # Check the raw weather file before merging
    weather_cols = ghcnd.columns.tolist()
    print(f"Columns found in CSV: {weather_cols}")

    # Check for actual data (not just NaNs) in the new wind features
    for col in ['wsf2_ms', 'wsf5_ms', 'awnd_ms', 'wsfg_ms']:
        if col in ghcnd.columns:
            non_zero = (ghcnd[col] > 0).sum()
            print(f"  {col}: {non_zero:,} non-zero records")
        else:
            print(f"  {col}: NOT FOUND IN CSV")

    print("Merging with GHCN-Daily daily weather...")
    merged = merge_ghcnd_weather(merged, ghcnd)

    print("Running preprocessing...")
    X, y = run_full_pipeline(merged)  # y is duration in minutes

    print("\n[DEBUG] Final Feature Check (Training Set):")
    for col in ['max_wind_speed', 'awnd_ms', 'prcp_mm', 'wsfg_ms']:
        if col in X.columns:
            non_zero_pct = (X[col] > 0).mean() * 100
            print(f"  {col}: {non_zero_pct:.2f}% of samples have non-zero data")


    # print("\nExporting a small sample of the preprocessed data...")
    
    # # Recombine features (X) and target (y) so they are in one file
    # sample_export = X.copy()
    # sample_export['TARGET_duration'] = y
    
    # # Grab a random sample of 100 rows
    # small_sample = sample_export.sample(n=100, random_state=42)
    
    # # Save it to the data directory
    # sample_path = data_dir / "preprocessed_sample_duration.csv"
    # small_sample.to_csv(sample_path, index=False)
    # print(f"Saved 100 sample rows to {sample_path}")
    # # ---------------------------------------

    # print("Splitting train/test...")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    
    # --- 3. Temporal Split Logic ---
    TEST_YEAR = 2022
    print(f"\n[!] TEMPORAL SPLIT: Training on 2014-{TEST_YEAR-1}, Testing on {TEST_YEAR}")
    
    train_mask = X['year'] < TEST_YEAR
    test_mask = X['year'] == TEST_YEAR

                # "event_Avalanche":         0,
                # "event_Coastal Flood":     0,
                # "event_Cold/Wind Chill":   int(tmin < -10),
                # "event_Debris Flow":       0,
                # "event_Drought":           0,
                # "event_Excessive Heat":    int(row.get("tmax_c", 0) > 38),
                # "event_Extreme Cold/Wind Chill": int(tmin < -20),
                # "event_Funnel Cloud":      0,
                # "event_Heat":              int(row.get("tmax_c", 0) > 32),
                # "event_Rip Current":       0,
                # "event_Tornado":           0,
                # "event_Tropical Storm":    int(gust > 33 and prcp > 25),
                # "event_Wildfire":          0,

    INCLUDE_DYNAMIC_FEATURES = True
    # cols_to_drop = ['year', 'county_max_customers', 'initial_impact_density']
    cols_to_drop = ['year']
    cols_to_drop += ['event_Avalanche','event_Coastal Flood','event_Debris Flow','event_Drought',
                     'event_Funnel Cloud','event_Rip Current','event_Tornado','event_Wildfire']
    cols_to_drop += ['event_None','county_median_duration','county_long_rate']

    if not INCLUDE_DYNAMIC_FEATURES:
        print("[!] GENERALIZATION MODE: Removing growth features.")
        cols_to_drop += ["initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m", "initial_impact_density"]

    # cols_to_drop = []
    # cols_to_drop = X_full.columns

    # cols_to_keep = ['county_median_scope', 'has_weather_event', 'max_magnitude', 'event_High Wind', 'event_Thunderstorm Wind']
    # cols_to_keep = ['max_magnitude']

    # cols_to_drop = X_full.columns.difference(cols_to_keep)


    X = X.drop(columns=cols_to_drop)

    # print("\nExporting a small sample of the preprocessed data...")
    
    # # Recombine features (X) and target (y) so they are in one file
    # sample_export = X.copy()
    # sample_export['TARGET_peak_customers_affected'] = y
    
    # # Grab a random sample of 100 rows
    # small_sample = sample_export.sample(n=100, random_state=42)
    
    # # Save it to the data directory
    # sample_path = data_dir / "preprocessed_sample_scope.csv"
    # small_sample.to_csv(sample_path, index=False)
    # print(f"Saved 100 sample rows to {sample_path}")
    # ---------------------------------------

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples (Year {TEST_YEAR}): {len(X_test):,}")

    if USE_TWO_STAGE:
        # ── Two-stage model ────────────────────────────────────────────────────
        print("\nTraining two-stage model (classifier + two regressors)...")
        twoStage = TwoStageOutageModel()
        twoStage.train(X_train, y_train)

        # Raise threshold to prioritize short-outage accuracy.
        # At 0.70 the classifier needs high confidence before routing to the
        # long regressor, so fewer short outages get misrouted.
        # Long accuracy will drop -- that is acceptable for this configuration.
        twoStage.classifierThreshold = 0.70
        print(f"[train_eval] Classifier threshold overridden to {twoStage.classifierThreshold:.2f} "
              f"(short-accuracy focus)")

        print("\nEvaluating two-stage model...")
        hardPreds, routing = twoStage.predictWithRouting(X_test)
        classMetrics = evaluateClassifier(y_test.values, routing)
        twoStageMetrics = evaluateModel(y_test.values, hardPreds)

        # ── Single-stage baseline for comparison ───────────────────────────────
        print("\nTraining single-stage baseline (for comparison)...")
        singleModel = OutageDurationModel()
        singleModel.train(X_train, y_train, longWeight=2.0)
        singlePreds = singleModel.predict(X_test)
        singleMetrics = evaluateModel(y_test.values, singlePreds)

        printTwoStageReport(classMetrics, twoStageMetrics, singleMetrics)

        print("\nTop classifier importances:")
        for k, v in sorted(
            twoStage.getClassifierImportances().items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {k:30s} {v:.4f}")

        print("\nTop short regressor importances:")
        for k, v in sorted(
            twoStage.getShortRegressorImportances().items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {k:30s} {v:.4f}")

        print("\nTop long regressor importances:")
        for k, v in sorted(
            twoStage.getLongRegressorImportances().items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {k:30s} {v:.4f}")

        # print("\nSaving two-stage duration model...")
        # model_dir = Path("../models")
        # model_dir.mkdir(parents=True, exist_ok=True)
        
        # # Save the two-stage model
        # twoStage.save(model_dir / "duration_model.joblib")

    else:
        # ── Single-stage model only ────────────────────────────────────────────
        print("\nTraining model (log-duration handled internally)...")
        model = OutageDurationModel()
        model.train(X_train, y_train, longWeight=2.0)

        print("Evaluating...")
        preds = model.predict(X_test)
        metrics = evaluateModel(y_test.values, preds)
        printEvaluationReport(metrics)

        print("\nTop feature importances:")
        for k, v in sorted(
            model.getFeatureImportances().items(), key=lambda x: x[1], reverse=True
        )[:15]:
            print(f"{k:30s} {v:.4f}")


if __name__ == "__main__":
    main()
