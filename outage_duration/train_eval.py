#Training and evaluation script for the outage duration model
#Supports both a two-stage model (classifier + two specialists) and a
#single-stage baseline so you can compare them side by side.

from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

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

#Flip this to False if you just want to run the single-stage baseline on its own for testing
USE_TWO_STAGE = True


def main():
    """Load outage + weather data, run preprocessing, train/evaluate the duration model, and save to disk."""
    data_dir = Path("../data")

    #Load all available EAGLE-I outage files and merge with storm event records
    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    print("Loading weather data...")
    weather_path = data_dir / "noaa_storm_events_va_2014_2022.csv"
    weather = load_noaa_weather(weather_path)

    print("Merging with NOAA Storm Events...")
    merged = merge_weather_outages(outages, weather)

    #Using Open-Meteo historical data in place of GHCND
    print("Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "openmeteo_va_historical.csv")

    #Quick check to make sure the wind columns actually have data before training
    print("\n[DEBUG] Verifying GHCN-Daily Feature Presence...")
    weather_cols = ghcnd.columns.tolist()
    print(f"Columns found in CSV: {weather_cols}")

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

    #If these come back as all zeros something went wrong in the merge
    print("\n[DEBUG] Final Feature Check (Training Set):")
    for col in ['max_wind_speed', 'awnd_ms', 'prcp_mm', 'wsfg_ms']:
        if col in X.columns:
            non_zero_pct = (X[col] > 0).mean() * 100
            print(f"  {col}: {non_zero_pct:.2f}% of samples have non-zero data")

    #Temporal split training on everything before 2022, test on 2022 only
    #This avoids leaking future data into training, which would inflate results
    TEST_YEAR = 2022
    print(f"\n[!] TEMPORAL SPLIT: Training on 2014-{TEST_YEAR-1}, Testing on {TEST_YEAR}")

    train_mask = X['year'] < TEST_YEAR
    test_mask = X['year'] == TEST_YEAR

    #Drop year and rare event types that don't add signal for Virginia outages
    INCLUDE_DYNAMIC_FEATURES = True
    cols_to_drop = ['year']
    cols_to_drop += ['event_Avalanche','event_Coastal Flood','event_Debris Flow','event_Drought',
                     'event_Funnel Cloud','event_Rip Current','event_Tornado','event_Wildfire']

    #Dynamic features are only available during nowcasting
    if not INCLUDE_DYNAMIC_FEATURES:
        print("[!] GENERALIZATION MODE: Removing growth features.")
        cols_to_drop += ["initial_customers_affected", "delta_customers_affected_15m", "pct_growth_15m"]

    X = X.drop(columns=cols_to_drop)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples (Year {TEST_YEAR}): {len(X_test):,}")

    if USE_TWO_STAGE:
        #Train the classifier with two specialist regressors together
        print("\nTraining two-stage model (classifier + two regressors)...")
        twoStage = TwoStageOutageModel()
        twoStage.train(X_train, y_train)

        print(f"[train_eval] Using auto-tuned classifier threshold: {twoStage.classifierThreshold:.2f}")

        #Get predictions with routing labels so we can evaluate the classifier separately
        print("\nEvaluating two-stage model...")
        hardPreds, routing = twoStage.predictWithRouting(X_test)
        classMetrics = evaluateClassifier(y_test.values, routing)
        twoStageMetrics = evaluateModel(y_test.values, hardPreds)

        #Train a single XGBoost as a baseline to show what the two-stage architecture gains
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

        #---Oracle evaluation---
        #Bypass the classifier and route using true labels
        #This shows the theoretical best the specialists could achieve with perfect routing
        THRESHOLD = 240.0  # 4 hours in minutes

        print("\n" + "="*50)
        print("ORACLE EVALUATION: PERFECT ROUTING")
        print("="*50)

        test_large_mask = y_test >= THRESHOLD
        test_small_mask = y_test < THRESHOLD
        final_preds = np.empty_like(y_test, dtype=float)

        if test_small_mask.sum() > 0:
            print(f"Routing {test_small_mask.sum():,} actual small events to Small Specialist...")
            #Regressors predict duration so we need to undo that with expm1
            pred_log_small = twoStage.shortRegressor.predict(X_test[test_small_mask])
            final_preds[test_small_mask] = np.expm1(np.clip(pred_log_small, -20, 20))

        if test_large_mask.sum() > 0:
            print(f"Routing {test_large_mask.sum():,} actual large events to Large Specialist...")
            pred_log_large = twoStage.longRegressor.predict(X_test[test_large_mask])
            final_preds[test_large_mask] = np.expm1(np.clip(pred_log_large, -20, 20))

        print("\n" + "="*50)
        print("THEORETICAL MAXIMUM PIPELINE PERFORMANCE")
        print("="*50)

        metrics = evaluateModel(y_test.values, final_preds)
        printEvaluationReport(metrics)

        print("\nTop Features for Short Specialist (Oracle Context):")
        s_imp = twoStage.getShortRegressorImportances()
        for k, v in sorted(s_imp.items(), key=lambda x: x[1], reverse=True)[:50]:
            print(f"  {k:30s} {v:.4f}")

        print("\nTop Features for Large Specialist (Oracle Context):")
        l_imp = twoStage.getLongRegressorImportances()
        for k, v in sorted(l_imp.items(), key=lambda x: x[1], reverse=True)[:50]:
            print(f"  {k:30s} {v:.4f}")

        print("\nSaving two-stage duration model...")
        model_dir = Path("../models")
        model_dir.mkdir(parents=True, exist_ok=True)
        twoStage.save(model_dir / "duration_model.joblib")

    else:
        #Single-stage fallback
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
