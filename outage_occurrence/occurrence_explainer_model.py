
# Imports
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

class OutageOccurrenceExplainer:

    def __init__(self, model, X: pd.DataFrame):
        
        # If the XGBoost model isn't trained, raise an error
        if not model.is_trained:
            raise RuntimeError("Warning during __init__(): Model isn't trained, call train() first")
        
        # Copy the XGBoost model
        self.model_wrapper = model
        self.model = model.model
        self.feature_names = model.feature_columns

        # Limit the number of background data to 1000, can be changed
        if len(X) > 1000:
            X_background = X.sample(1000, random_state=42)
        else:
            X_background = X.copy()

        # Create the SHAP explainer model
        self.explainer = shap.TreeExplainer(self.model)
        self.background_data = X_background

    def computeShapValues(self, X: pd.DataFrame) -> np.ndarray:
        
        # Extract the necessary features from the data frame
        X = X[self.feature_names]

        # Extract the SHAP values from the explainer
        shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values can return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    def plotSummary(self, X: pd.DataFrame=None, path: str=None):

        if X is None:
            X = self.background_data

        # Compute the SHAP values
        shap_values = self.computeShapValues(X)

        # Plot the SHAP values
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)

        # Save the plot to the specified path
        # Otherwise display the plot
        if path:
            plt.savefig(path, bbox_inches="tight")
            print(f"Summary plot saved to {path}")
        else:
            plt.show()

        plt.close()

    def plotFeatureImportance(self, X: pd.DataFrame=None, path: str=None):

        if X is None:
            X = self.background_data

        shap_values = self.computeShapValues(X)

        # Compute the mean importance of the SHAP values
        mean_importance = np.abs(shap_values).mean(axis=0)

        # Compute feature importance, sort by most important
        feature_importance = pd.Series(mean_importance, index=self.feature_names)
        feature_importance = feature_importance.sort_values(ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(8, 6))
        feature_importance.head(20).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("SHAP Feature Importance")

        # Save the plot to the specified path
        # Otherwise display the plot
        if path:
            plt.savefig(path, bbox_inches="tight")
            print(f"Feature importance plot saved to {path}")
        else:
            plt.show()

        plt.close()

    def explainSinglePrediction(self, X: pd.DataFrame=None, idx: int=0, path: str=None):

        # Compute the SHAP values
        X = X[self.feature_names]
        shap_values = self.computeShapValues(X)

        # Get the expected value from the SHAP explainer
        expected_value = self.explainer.expected_value

        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        # Get the SHAP value at the specified index
        single_shap = shap_values[idx]
        single_instance = X.iloc[idx]

        # Plot the prediction SHAP value as a waterfall plot
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=single_shap,
                base_values=expected_value,
                data=single_instance,
                feature_names=self.feature_names
            ),
            show=False
        )

        # Save the plot to the specified path
        # Otherwise display the plot
        if path:
            plt.savefig(path, bbox_inches="tight")
            print("Single prediction plot saved to {path}")
        else:
            plt.show()

        plt.close()

    def getFeatureImportanceDict(self, X: pd.DataFrame=None) -> Dict[str, float]:
        
        if X is None:
            X = self.background_data

        # Compute the SHAP values
        shap_values = self.computeShapValues(X)

        # Compute the mean importance of the SHAP values
        mean_importance = np.abs(shap_values).mean(axis=0)

        # Create a dictionary with key-pair values: feature, importance
        importance_dict = {
            feature: float(importance)
            for feature, importance in zip(self.feature_names, mean_importance)
        }

        # Sort dictionary by most importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        return importance_dict