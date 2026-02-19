
# Imports
import numpy as np
import pandas as pd

from occurrence_model import OutageOccurenceModel
from occurrence_explainer_model import OutageOccurrenceExplainer

# TODO IMPORTANT: Test data synthetically generated, replace with actual data

def generate_synthetic_weather_data(n_samples: int = 1000):

    rng = np.random.default_rng(42)

    # Generate random values for each feature
    wind_speed = np.random.normal(15, 10, n_samples).clip(0)
    precipitation = np.random.exponential(2, n_samples)
    temperature = np.random.normal(60, 20, n_samples)
    humidity = np.random.uniform(20, 100, n_samples)
    lightning_strikes = np.random.poisson(1.5, n_samples)

    # Synthetic outage rule:
    # High wind + high precipitation + lightning increases outage risk
    outage_probability = (
        0.03 * wind_speed +
        0.15 * precipitation +
        0.4 * lightning_strikes +
        0.01 * humidity -
        0.01 * temperature
    )

    # Generate an outage probability
    outage_probability = 1 / (1 + np.exp(-0.1 * outage_probability))
    y = (outage_probability > 0.5).astype(int)

    # Merge each feature into a synthetic dataset
    X = pd.DataFrame({
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "temperature": temperature,
        "humidity": humidity,
        "lightning_strikes": lightning_strikes
    })

    return X, pd.Series(y)

def test_occurrence_model():
    print("\n===== Testing OutageOccurenceModel =====")

    X, y = generate_synthetic_weather_data(2000)

    model = OutageOccurenceModel()
    model.train(X, y)

    predictions, probabilities = model.predict(X.head(5))

    print("Sample Predictions:", predictions)
    print("Sample Probabilities:", probabilities)

    model.save("test_outage_model.pkl")
    print("Model saved successfully.")

    loaded_model = OutageOccurenceModel.load("test_outage_model.pkl")
    print("Model loaded successfully.")

    preds_loaded, _ = loaded_model.predict(X.head(5))
    print("Loaded Model Predictions:", preds_loaded)

def test_explainer():
    print("\n===== Testing OutageOccurrenceExplainer =====")

    X, y = generate_synthetic_weather_data(1500)

    model = OutageOccurenceModel()
    model.train(X, y)

    explainer = OutageOccurrenceExplainer(model, X)

    # Compute SHAP values
    shap_values = explainer.computeShapValues(X.head(10))
    print("SHAP values shape:", shap_values.shape)

    # Plot summary
    #explainer.plotSummary(X.head(200), path="shap_summary.png")
    explainer.plotSummary(X.head(200))
    print("Summary plot saved as shap_summary.png")

    # Plot feature importance
    #explainer.plotFeatureImportance(X.head(200), path="shap_importance.png")
    explainer.plotFeatureImportance(X.head(200))
    print("Feature importance plot saved as shap_importance.png")

    # Explain single prediction
    #explainer.explainSinglePrediction(X.head(10), idx=0, path="shap_single.png")
    explainer.explainSinglePrediction(X.head(10), idx=0)
    print("Single prediction explanation saved as shap_single.png")

    # Print importance dictionary
    importance_dict = explainer.getFeatureImportanceDict(X.head(500))
    print("\nTop Feature Importances:")
    for feature, score in list(importance_dict.items())[:5]:
        print(f"{feature}: {score:.4f}")

# Main function
if __name__ == "__main__":
    test_occurrence_model()
    test_explainer()