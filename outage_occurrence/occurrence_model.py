
# Imports
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

class OutageOccurenceModel:

    def __init__(self, model_params: Optional[Dict[str, Any]]=None):

        # Default parameters for the XGBoost model in case custom
        # parameters aren't set
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False
        }

        self.model = xgb.XGBClassifier(**default_params)    # XGBoost model
        self.feature_columns = None                         # Features used for training the model
        self.is_trained = False                             # Boolean to set when model is trained

    def train(self, X: pd.DataFrame, y: pd.Series, validationSplit: float=0.2):

        # Convert the columns in the dataset to a list
        self.feature_columns = X.columns.tolist()

        # Partition the dataset into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validationSplit,
            random_state=42,
            stratify=y
        )

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Set the model as trained
        self.is_trained = True

    def predict(self, X: pd.DataFrame):

        # If the model isn't trained, raise an error
        if not self.is_trained:
            raise RuntimeError("Warning during predict(): Model isn't trained, call train() first")
        
        # Extract the features from the dataset which were used for training
        X = X[self.feature_columns]

        # Extract the probabilities and predictions from the model
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        return predictions, probabilities

    def save(self, path: str):

        # If the model isn't trained, raise an error
        if not self.is_trained:
            raise RuntimeError("Warning during save(): Model isn't trained, call train() first")

        # Convert the path string to a Path object
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the model to the specified path location
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained
        }, path)

    @classmethod
    def load(cls, path: str) -> 'OutageOccurenceModel':
        
        # Convert the path string to a Path object
        path = Path(path)

        # If the path doesn't exist, throw an error
        if not path.exists():
            raise FileNotFoundError(f"Warning during load(): Model file not found at {path}")

        # Extract the model data
        saved_data = joblib.load(path)

        # Create a new instance of the model class
        instance = cls()

        # Load the saved data into the new model class
        instance.model = saved_data["model"]
        instance.feature_columns = saved_data["feature_columns"]
        instance.is_trained = saved_data["is_trained"]

        # Return the new instance
        return instance
    