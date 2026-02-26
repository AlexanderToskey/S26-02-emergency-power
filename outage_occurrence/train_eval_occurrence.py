from pathlib import Path
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
    data_dir = Path("data")

    # ------------------------------------------------------------------
    # STEP 1: Load outage data
    # ------------------------------------------------------------------
    print("\nLoading outage data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)

    # ------------------------------------------------------------------
    # STEP 2: Build county-day occurrence labels
    # ------------------------------------------------------------------
    print("Building county-day occurrence labels...")
    occurrence = build_occurrence_labels(outages)

    # ------------------------------------------------------------------
    # STEP 3: Load daily weather
    # ------------------------------------------------------------------
    print("Loading GHCN-Daily weather data...")
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")

    # ------------------------------------------------------------------
    # STEP 4: Merge labels with weather
    # ------------------------------------------------------------------
    print("Merging occurrence labels with weather...")
    merged = merge_occurrence_with_weather(occurrence, ghcnd)

    summarize_class_balance(merged)

    # ------------------------------------------------------------------
    # STEP 5: Preprocessing
    # ------------------------------------------------------------------
    print("Running preprocessing...")
    X, y = run_full_pipeline(merged)  # y = outage_occurred (0/1)

    # ------------------------------------------------------------------
    # STEP 6: Train/Test split
    # ------------------------------------------------------------------
    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # important for imbalanced data
    )

    # ------------------------------------------------------------------
    # STEP 7: Train model
    # ------------------------------------------------------------------
    print("Training outage occurrence model...")
    model = OutageOccurrenceModel()

    # Occurrence models usually support class_weight or scale_pos_weight
    model.train(X_train, y_train)

    # ------------------------------------------------------------------
    # STEP 8: Evaluate
    # ------------------------------------------------------------------
    print("Evaluating...")
    #y_prob = model.predict(X_test)[:, 1]
    preds, y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluateModel(y_test.values, y_pred, y_prob)
    printEvaluationReport(metrics)

    # ------------------------------------------------------------------
    # STEP 8.5: Save Model to Central Directory
    # ------------------------------------------------------------------
    print("\nSaving occurrence model...")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Assuming OutageOccurrenceModel has a save() method
    model.save(model_dir / "occurrence_model.joblib")

    # ------------------------------------------------------------------
    # STEP 9: Feature importance
    # ------------------------------------------------------------------
    #print("\nTop feature importances:")
    #importances = model.getFeatureImportances()

    #for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]:
    #    print(f"{k:30s} {v:.4f}")

    # ------------------------------------------------------------------
    # STEP 10: SHAP Explainer
    # ------------------------------------------------------------------
    # Uncomment if explainer supports classification
    print("\nGenerating SHAP Explanations...")
    try:
        explainer = OutageOccurrenceExplainer(model, X_train)
        explainer.plotSummary(X_test, path="shap_summary_occurrence.png")
        print("SHAP summary plot saved successfully.")

        shap_values = explainer.computeShapValues(X.head(10))
        print("SHAP values shape:", shap_values.shape)
    
        explainer.plotFeatureImportance(X.head(200), path="shap_importance.png")
        #explainer.plotFeatureImportance(X.head(200))
        print("Feature importance plot saved as shap_importance.png")
        
        importance_dict = explainer.getFeatureImportanceDict(X.head(500))
        print("\nTop Feature Importances:")
        for feature, score in list(importance_dict.items())[:5]:
            print(f"{feature}: {score:.4f}")

    except Exception as e:
        print(f"Warning: Could not generate SHAP plots. Error: {e}")


if __name__ == "__main__":
    main()