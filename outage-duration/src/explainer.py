"""
explainer.py - SHAP-based model explainability

generates explanations for model predictions using SHAP values.
helps understand what features are driving the predictions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


class OutageExplainer:
    """
    SHAP explainer wrapper for our outage duration model

    handles computing shap values and generating various
    explanation visualizations
    """

    def __init__(self, model, X: pd.DataFrame):
        """
        initialize the explainer with a trained model

        Args:
            model: trained OutageDurationModel instance (or just the xgb model)
            X: sample of training data for the explainer background
        """
        # handle both our wrapper class and raw xgb model
        if hasattr(model, 'model'):
            self.xgbModel = model.model
        else:
            self.xgbModel = model

        # use a sample for background if dataset is large
        if len(X) > 1000:
            backgroundData = X.sample(n=1000, random_state=42)
        else:
            backgroundData = X

        self.explainer = shap.TreeExplainer(self.xgbModel)
        self.featureNames = list(X.columns)
        self.shapValues = None
        self.X = None

    def computeShapValues(self, X: pd.DataFrame) -> np.ndarray:
        """
        computes SHAP values for the given data

        Args:
            X: dataframe of features to explain

        Returns:
            array of SHAP values
        """
        self.X = X
        self.shapValues = self.explainer.shap_values(X)
        return self.shapValues

    def plotSummary(self, X: pd.DataFrame = None, savePath: str = None):
        """
        creates a SHAP summary plot showing feature importance

        Args:
            X: data to explain (uses previously computed if None)
            savePath: optional path to save the figure
        """
        if X is not None:
            self.computeShapValues(X)

        if self.shapValues is None:
            raise ValueError("no shap values computed yet, pass X or call computeShapValues first")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shapValues, self.X, feature_names=self.featureNames, show=False)

        if savePath:
            plt.savefig(savePath, bbox_inches='tight', dpi=150)
            print(f"summary plot saved to {savePath}")

        plt.show()

    def plotFeatureImportance(self, X: pd.DataFrame = None, savePath: str = None):
        """
        bar chart of mean absolute SHAP values (feature importance)

        Args:
            X: data to explain
            savePath: optional path to save the figure
        """
        if X is not None:
            self.computeShapValues(X)

        if self.shapValues is None:
            raise ValueError("no shap values computed yet")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shapValues, self.X, feature_names=self.featureNames,
                         plot_type="bar", show=False)

        if savePath:
            plt.savefig(savePath, bbox_inches='tight', dpi=150)
            print(f"importance plot saved to {savePath}")

        plt.show()

    def explainSinglePrediction(self, X: pd.DataFrame, idx: int = 0,
                                 savePath: str = None):
        """
        explains a single prediction with a waterfall plot

        Args:
            X: feature dataframe
            idx: index of the sample to explain
            savePath: optional path to save the figure
        """
        shapValues = self.explainer.shap_values(X.iloc[[idx]])

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=shapValues[0],
                           base_values=self.explainer.expected_value,
                           data=X.iloc[idx].values,
                           feature_names=self.featureNames),
            show=False
        )

        if savePath:
            plt.savefig(savePath, bbox_inches='tight', dpi=150)

        plt.show()

    def getFeatureImportanceDict(self, X: pd.DataFrame = None) -> Dict[str, float]:
        """
        returns feature importances as a dict

        Args:
            X: data to compute SHAP values for

        Returns:
            dict mapping feature names to mean |SHAP| values
        """
        if X is not None:
            self.computeShapValues(X)

        if self.shapValues is None:
            raise ValueError("no shap values computed yet")

        meanAbsShap = np.abs(self.shapValues).mean(axis=0)

        return dict(zip(self.featureNames, meanAbsShap))
