import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        self.scaler.fit(X)
        self.is_fitted = True

    def transform(self, X):
        if not self.is_fitted:
            raise Exception("Scaler not fitted yet")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        joblib.dump(self.scaler, path)

    def load(self, path):
        self.scaler = joblib.load(path)
        self.is_fitted = True


class IsolationForestModel:
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X):
        self.model.fit(X)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model not trained yet")
        return self.model.predict(X)  # -1 = anomaly, 1 = normal

    def anomaly_score(self, X):
        if not self.is_trained:
            raise Exception("Model not trained yet")
        return self.model.decision_function(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_trained = True


class ModelTrainer:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def train(self, X):
        X_scaled = self.preprocessor.fit_transform(X)
        self.model.train(X_scaled)

    def save_pipeline(self, scaler_path, model_path):
        self.preprocessor.save(scaler_path)
        self.model.save(model_path)


class RealTimeAnomalyDetector:
    def __init__(self, scaler_path, model_path):
        self.preprocessor = DataPreprocessor()
        self.model = IsolationForestModel()

        self.preprocessor.load(scaler_path)
        self.model.load(model_path)

    def detect(self, X_new):
        X_scaled = self.preprocessor.transform(X_new)
        predictions = self.model.predict(X_scaled)
        scores = self.model.anomaly_score(X_scaled)

        return {
            "prediction": predictions,
            "score": scores
        }