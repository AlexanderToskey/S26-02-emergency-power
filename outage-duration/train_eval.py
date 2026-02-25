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
