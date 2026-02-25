"""
model.py - XGBoost model for outage scope prediction

handles model creation, training, saving/loading, and prediction.
kept it pretty simple since xgboost does most of the heavy lifting.
Target variable is peak_customers_affected.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split


class OutageScopeModel:
    """
    XGBoost regression model for predicting outage scope (customers affected)

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
            'max_depth': 5,
            'learning_rate': 0.05,
            'reg_lambda': 10,
            'n_estimators': 1000,
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
        largeThresholdCustomers: float = 500.0,
        largeWeight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Train the model.

        IMPORTANT CONTRACT:
        - y must be RAW outage scope in customer counts (not log).
        - We compute weights using raw customer counts.
        - We train XGBoost on log1p(customers).
        - predict() returns customer counts.
        """
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=validationSplit, random_state=42
        )

        # Ensure raw customer counts (no log!)
        yTrainCust = pd.to_numeric(yTrain, errors="coerce").astype(float).values
        yValCust = pd.to_numeric(yVal, errors="coerce").astype(float).values

        # Drop any invalid rows (just in case)
        train_ok = np.isfinite(yTrainCust) & (yTrainCust >= 0)
        val_ok = np.isfinite(yValCust) & (yValCust >= 0)

        xTrain = xTrain.loc[xTrain.index[train_ok]]
        yTrainCust = yTrainCust[train_ok]

        xVal = xVal.loc[xVal.index[val_ok]]
        yValCust = yValCust[val_ok]

        # Sample weights based on RAW customer counts
        large_mask_train = (yTrainCust >= largeThresholdCustomers)
        large_mask_val = (yValCust >= largeThresholdCustomers)

        wTrain = np.where(large_mask_train, float(largeWeight), 1.0).astype(float)
        wVal = np.where(large_mask_val, float(largeWeight), 1.0).astype(float)

        print(f"[model] USING WEIGHTED TRAIN: largeWeight={largeWeight}, largeThresholdCustomers={largeThresholdCustomers}")
        print(f"[model] wTrain unique={np.unique(wTrain)}  large_frac={large_mask_train.mean():.3f}")

        # Train on log1p(customers) to handle the right-skewed distribution of outages
        yTrainLog = np.log1p(yTrainCust)
        yValLog = np.log1p(yValCust)

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
    def load(cls, path: str) -> 'OutageScopeModel':
        """
        loads a previously saved model

        Args:
            path: path to the saved model file

        Returns:
            OutageScopeModel instance ready for predictions
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