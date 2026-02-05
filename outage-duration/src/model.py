"""
model.py - XGBoost model for outage duration prediction

handles model creation, training, saving/loading, and prediction.
kept it pretty simple since xgboost does most of the heavy lifting
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split


class OutageDurationModel:
    """
    XGBoost regression model for predicting outage duration

    wraps the xgboost regressor with some convenience methods
    for our specific use case
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        initialize the model with optional custom parameters

        Args:
            params: xgboost parameters dict, uses defaults if not provided
        """
        self.defaultParams = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }

        self.params = {**self.defaultParams, **(params or {})}
        self.model = xgb.XGBRegressor(**self.params)
        self.isTrained = False

    def train(self, X: pd.DataFrame, y: pd.Series,
              validationSplit: float = 0.2) -> Dict[str, Any]:
        """
        trains the model on the provided data

        Args:
            X: feature dataframe
            y: target series (duration in minutes)
            validationSplit: fraction of data to hold out for validation

        Returns:
            dict with training info (val scores, feature importances, etc)
        """
        # split for validation
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=validationSplit, random_state=42
        )

        # fit with early stopping
        self.model.fit(
            xTrain, yTrain,
            eval_set=[(xVal, yVal)],
            verbose=False
        )

        self.isTrained = True
        self.featureNames = list(X.columns)

        # get validation predictions for reporting
        valPreds = self.model.predict(xVal)

        trainInfo = {
            'train_samples': len(xTrain),
            'val_samples': len(xVal),
            'feature_names': self.featureNames,
            'feature_importances': dict(zip(self.featureNames,
                                           self.model.feature_importances_))
        }

        return trainInfo

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        makes predictions on new data

        Args:
            X: feature dataframe (same format as training data)

        Returns:
            array of predicted durations in minutes
        """
        if not self.isTrained:
            raise ValueError("model hasnt been trained yet, call train() first")

        return self.model.predict(X)

    def save(self, path: str):
        """
        saves the trained model to disk

        Args:
            path: where to save the model file
        """
        if not self.isTrained:
            raise ValueError("cant save untrained model")

        modelData = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.featureNames
        }

        joblib.dump(modelData, path)
        print(f"model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'OutageDurationModel':
        """
        loads a previously saved model

        Args:
            path: path to the saved model file

        Returns:
            OutageDurationModel instance ready for predictions
        """
        modelData = joblib.load(path)

        instance = cls(params=modelData['params'])
        instance.model = modelData['model']
        instance.featureNames = modelData['feature_names']
        instance.isTrained = True

        print(f"model loaded from {path}")
        return instance

    def getFeatureImportances(self) -> Dict[str, float]:
        """
        returns feature importances from the trained model

        Returns:
            dict mapping feature names to importance scores
        """
        if not self.isTrained:
            raise ValueError("model hasnt been trained yet")

        return dict(zip(self.featureNames, self.model.feature_importances_))
