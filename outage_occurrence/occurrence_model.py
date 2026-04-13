
# Imports
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


class OutageOccurrenceModel:

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
            # ~8:1 negative:positive ratio at >=100 customer threshold
            "scale_pos_weight": 10,
        }

        # XGBoost model
        xgb_model = xgb.XGBClassifier(**default_params)   

        # Random forrest model
        rf = RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )

        # Gradient boosting model
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3
        )

        # Logistic regression model
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )

        # Stacking model
        self.model = StackingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            final_estimator = LogisticRegression(
                class_weight="balanced",
                max_iter=3000
            ),
            stack_method='predict_proba',
            passthrough=False
        )

        self.threshold = 0.09         # Stacking threshold
        self.feature_columns = None   # Features used for training the model
        self.is_trained = False       # Boolean to set when model is trained

    def tune_threshold(self, X_val, y_val):
        probs = self.model.predict_proba(X_val)[:, 1]

        best_t = 0.5
        best_f2 = 0

        for t in np.linspace(0.05, 0.5, 20):  # focus low thresholds
            preds = (probs >= t).astype(int)

            tp = ((preds == 1) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)

            # F2 emphasizes recall more than precision
            beta = 2
            f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)

            if f2 > best_f2:
                best_f2 = f2
                best_t = t

        self.threshold = best_t
        print(f"[model] Best threshold (F2): {self.threshold:.3f}")

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
        self.model.fit(X_train, y_train)

        # Tune the threshold
        self.tune_threshold(X_val, y_val)

        # Set the model as trained
        self.is_trained = True

    def predict(self, X: pd.DataFrame):

        # If the model isn't trained, raise an error
        if not self.is_trained:
            raise RuntimeError("Warning during predict(): Model isn't trained, call train() first")
        
        # Extract the features from the dataset which were used for training
        # Protect against missing columns
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Extract the probabilities and predictions from the model
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

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
            "is_trained": self.is_trained,
            "threshold": self.threshold
        }, path)

    @classmethod
    def load(cls, path: str) -> 'OutageOccurrenceModel':
        
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
        instance.threshold = saved_data.get("threshold", 0.5)

        # Return the new instance
        return instance
    