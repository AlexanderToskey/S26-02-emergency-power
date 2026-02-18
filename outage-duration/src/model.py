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
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1
        }

        self.params = {**self.defaultParams, **(params or {})}
        self.model = xgb.XGBRegressor(**self.params)
        self.isTrained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validationSplit: float = 0.2,
        longThresholdMin: float = 240.0,
        longWeight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Train the model.

        IMPORTANT CONTRACT:
        - y must be RAW outage duration in minutes (not log).
        - We compute weights using raw minutes.
        - We train XGBoost on log1p(minutes).
        - predict() returns minutes.
        """
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=validationSplit, random_state=42
        )

        # Ensure raw minutes (no log!)
        yTrainMin = pd.to_numeric(yTrain, errors="coerce").astype(float).values
        yValMin = pd.to_numeric(yVal, errors="coerce").astype(float).values

        # Drop any invalid rows (just in case)
        train_ok = np.isfinite(yTrainMin) & (yTrainMin >= 0)
        val_ok = np.isfinite(yValMin) & (yValMin >= 0)

        xTrain = xTrain.loc[xTrain.index[train_ok]]
        yTrainMin = yTrainMin[train_ok]

        xVal = xVal.loc[xVal.index[val_ok]]
        yValMin = yValMin[val_ok]

        # Sample weights based on RAW minutes
        long_mask_train = (yTrainMin >= longThresholdMin)
        long_mask_val = (yValMin >= longThresholdMin)

        wTrain = np.where(long_mask_train, float(longWeight), 1.0).astype(float)
        wVal = np.where(long_mask_val, float(longWeight), 1.0).astype(float)

        print(f"[model] USING WEIGHTED TRAIN: longWeight={longWeight}, longThresholdMin={longThresholdMin}")
        print(f"[model] wTrain unique={np.unique(wTrain)}  long_frac={long_mask_train.mean():.3f}")

        # Train on log1p(minutes)
        yTrainLog = np.log1p(yTrainMin)
        yValLog = np.log1p(yValMin)

        self.model.fit(
            xTrain,
            yTrainLog,
            sample_weight=wTrain,
            eval_set=[(xVal, yValLog)],
            sample_weight_eval_set=[wVal],
            verbose=False,
        )

        self.isTrained = True
        self.featureNames = list(X.columns)

        return {
            "train_samples": len(xTrain),
            "val_samples": len(xVal),
            "feature_names": self.featureNames,
            "feature_importances": dict(zip(self.featureNames, self.model.feature_importances_)),
        }



    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.isTrained:
            raise ValueError("model hasnt been trained yet, call train() first")

        predLog = self.model.predict(X)

        # Safety clip: prevents exp overflow if model outputs something insane
        predLog = np.clip(predLog, -20, 20)

        return np.expm1(predLog)

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
