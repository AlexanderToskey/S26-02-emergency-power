"""
two_stage_model.py - Two-stage outage scope prediction model

Stage 1: XGBClassifier predicts small (< 500) vs large (>= 500) outage.
Stage 2: Two separate XGBRegressors, one for each class.

Regressors are trained on TRUE-label subsets (no selection bias from
classifier errors bleeding into regressor training).

At inference, Stage 1 routes each sample to the appropriate regressor,
then Stage 2 returns a customer count.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Dict, Any, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


CUSTOMER_THRESHOLD = 500.0


class TwoStageScopeModel:
    """
    Two-stage model for outage scope prediction.

    Stage 1: binary XGBClassifier (0=small, 1=large)
    Stage 2a: XGBRegressor trained on true-small samples
    Stage 2b: XGBRegressor trained on true-large samples

    At inference, Stage 1 routes, Stage 2 predicts scope in customers.
    """

    def __init__(
        self,
        classifierParams: Dict[str, Any] = None,
        shortRegressorParams: Dict[str, Any] = None,
        longRegressorParams: Dict[str, Any] = None,
        thresholdMin: float = CUSTOMER_THRESHOLD,
    ):
        self.thresholdMin = thresholdMin

        # ── Stage 1: classifier ───────────────────────────────────────────────
        defaultClassifierParams = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'scale_pos_weight': 20.0,  # compensates for extreme small/large imbalance
            'random_state': 42,
            'n_jobs': -1,
        }
        self.classifierParams = {**defaultClassifierParams, **(classifierParams or {})}
        self.classifier = xgb.XGBClassifier(**self.classifierParams)

        # ── Stage 2a: small outage regressor ──────────────────────────────────
        defaultShortParams = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'min_child_weight': 10,  # more conservative
            'random_state': 42,
            'n_jobs': -1,
        }
        self.shortRegressorParams = {**defaultShortParams, **(shortRegressorParams or {})}
        self.shortRegressor = xgb.XGBRegressor(**self.shortRegressorParams)

        # ── Stage 2b: large outage regressor ──────────────────────────────────
        defaultLongParams = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'min_child_weight': 5,
            'random_state': 42,
            'n_jobs': -1,
        }
        self.longRegressorParams = {**defaultLongParams, **(longRegressorParams or {})}
        self.longRegressor = xgb.XGBRegressor(**self.longRegressorParams)

        # filled in during train()
        self.classifierThreshold = 0.5
        self.isTrained = False
        self.featureNames = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validationSplit: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train all three sub-models on a single shared train/val split.

        y must be raw outage scope in customers (not log).
        Regressors train on log1p(customers) and predict() returns expm1.
        Classifier threshold is swept on the validation set to maximize
        balanced accuracy for the large-outage class.
        """
        # shared split so all three sub-models see the same val set
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=validationSplit, random_state=42
        )

        yTrainCust = pd.to_numeric(yTrain, errors="coerce").astype(float).values
        yValCust = pd.to_numeric(yVal, errors="coerce").astype(float).values

        # drop invalid rows
        trainOk = np.isfinite(yTrainCust) & (yTrainCust >= 0)
        valOk = np.isfinite(yValCust) & (yValCust >= 0)

        xTrain = xTrain.loc[xTrain.index[trainOk]]
        yTrainCust = yTrainCust[trainOk]

        xVal = xVal.loc[xVal.index[valOk]]
        yValCust = yValCust[valOk]

        # binary labels: 0 = small, 1 = large
        yTrainLabel = (yTrainCust >= self.thresholdMin).astype(int)
        yValLabel = (yValCust >= self.thresholdMin).astype(int)

        # ── Stage 1: train classifier ─────────────────────────────────────────
        longFracTrain = yTrainLabel.mean()
        longFracVal = yValLabel.mean()
        print(
            f"\n[two_stage] Stage 1 classifier: "
            f"large frac train={longFracTrain:.3f}, val={longFracVal:.3f}"
        )

        self.classifier.fit(
            xTrain,
            yTrainLabel,
            eval_set=[(xVal, yValLabel)],
            verbose=False,
        )

        # tune threshold on val set: maximize balanced accuracy
        valProbs = self.classifier.predict_proba(xVal)[:, 1]
        shortMaskVal_bool = yValLabel == 0
        longMaskVal_bool = yValLabel == 1

        bestThreshold, bestBalAcc = 0.5, -1.0
        for thresh in np.arange(0.05, 0.95, 0.02):
            preds = (valProbs >= thresh).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            recallShort = np.mean(preds[shortMaskVal_bool] == 0) if shortMaskVal_bool.sum() > 0 else 0.0
            recallLong = np.mean(preds[longMaskVal_bool] == 1) if longMaskVal_bool.sum() > 0 else 0.0
            balAcc = (recallShort + recallLong) / 2.0
            if balAcc > bestBalAcc:
                bestBalAcc, bestThreshold = balAcc, float(thresh)

        self.classifierThreshold = bestThreshold
        valClassPreds = (valProbs >= bestThreshold).astype(int)
        routingAcc = np.mean(valClassPreds == yValLabel) * 100
        f1Long = f1_score(yValLabel, valClassPreds, pos_label=1)

        print(
            f"[two_stage] Threshold tuned: {bestThreshold:.2f} "
            f"(val balanced_acc={bestBalAcc:.3f}, routing acc={routingAcc:.1f}%, F1 large={f1Long:.3f})"
        )

        # ── Stage 2a: small regressor (true-small subset only) ───────────────
        shortMaskTrain = yTrainCust < self.thresholdMin
        shortMaskVal = yValCust < self.thresholdMin

        xTrainShort = xTrain.loc[xTrain.index[shortMaskTrain]]
        yTrainShortLog = np.log1p(yTrainCust[shortMaskTrain])
        xValShort = xVal.loc[xVal.index[shortMaskVal]]
        yValShortLog = np.log1p(yValCust[shortMaskVal])

        print(
            f"\n[two_stage] Stage 2a small regressor: "
            f"{shortMaskTrain.sum():,} train, {shortMaskVal.sum():,} val"
        )
        self.shortRegressor.fit(
            xTrainShort,
            yTrainShortLog,
            eval_set=[(xValShort, yValShortLog)],
            verbose=False,
        )

        # ── Stage 2b: large regressor (true-large subset only) ─────────────────
        longMaskTrain = yTrainCust >= self.thresholdMin
        longMaskVal = yValCust >= self.thresholdMin

        xTrainLong = xTrain.loc[xTrain.index[longMaskTrain]]
        yTrainLongLog = np.log1p(yTrainCust[longMaskTrain])
        xValLong = xVal.loc[xVal.index[longMaskVal]]
        yValLongLog = np.log1p(yValCust[longMaskVal])

        print(
            f"[two_stage] Stage 2b large regressor: "
            f"{longMaskTrain.sum():,} train, {longMaskVal.sum():,} val"
        )
        self.longRegressor.fit(
            xTrainLong,
            yTrainLongLog,
            eval_set=[(xValLong, yValLongLog)],
            verbose=False,
        )

        self.isTrained = True
        self.featureNames = list(X.columns)

        return {
            "classifier_threshold": self.classifierThreshold,
            "classifier_val_f1_long": f1Long,
            "classifier_routing_acc": routingAcc,
            "short_train_samples": int(shortMaskTrain.sum()),
            "long_train_samples": int(longMaskTrain.sum()),
            "feature_names": self.featureNames,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict outage scope in customer counts.
        Uses hard routing — each sample goes to exactly one
        regressor based on whether p_long >= classifierThreshold.
        """
        if not self.isTrained:
            raise ValueError("model hasn't been trained yet, call train() first")

        pLong = self.classifier.predict_proba(X)[:, 1]

        # hard routing path
        isLong = pLong >= self.classifierThreshold
        preds = np.zeros(len(X))

        shortMask = ~isLong
        if shortMask.sum() > 0:
            predLogShort = self.shortRegressor.predict(X[shortMask])
            predLogShort = np.clip(predLogShort, -20, 20)
            preds[shortMask] = np.expm1(predLogShort)

        if isLong.sum() > 0:
            predLogLong = self.longRegressor.predict(X[isLong])
            predLogLong = np.clip(predLogLong, -20, 20)
            preds[isLong] = np.expm1(predLogLong)

        return preds

    def predictWithRouting(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions_in_customers, routing_labels).

        routing_labels: 0 = routed to small regressor, 1 = routed to large regressor.
        Useful for evaluating classifier accuracy separately from regression accuracy.
        """
        if not self.isTrained:
            raise ValueError("model hasn't been trained yet, call train() first")

        probs = self.classifier.predict_proba(X)[:, 1]
        isLong = probs >= self.classifierThreshold

        preds = np.zeros(len(X))

        shortMask = ~isLong
        if shortMask.sum() > 0:
            predLogShort = self.shortRegressor.predict(X[shortMask])
            predLogShort = np.clip(predLogShort, -20, 20)
            preds[shortMask] = np.expm1(predLogShort)

        if isLong.sum() > 0:
            predLogLong = self.longRegressor.predict(X[isLong])
            predLogLong = np.clip(predLogLong, -20, 20)
            preds[isLong] = np.expm1(predLogLong)

        return preds, isLong.astype(int)

    def getClassifierImportances(self) -> Dict[str, float]:
        if not self.isTrained:
            raise ValueError("model hasn't been trained yet")
        return dict(zip(self.featureNames, self.classifier.feature_importances_))

    def getShortRegressorImportances(self) -> Dict[str, float]:
        if not self.isTrained:
            raise ValueError("model hasn't been trained yet")
        return dict(zip(self.featureNames, self.shortRegressor.feature_importances_))

    def getLongRegressorImportances(self) -> Dict[str, float]:
        if not self.isTrained:
            raise ValueError("model hasn't been trained yet")
        return dict(zip(self.featureNames, self.longRegressor.feature_importances_))

    def save(self, path: str):
        if not self.isTrained:
            raise ValueError("can't save untrained model")
        joblib.dump({
            'classifier': self.classifier,
            'shortRegressor': self.shortRegressor,
            'longRegressor': self.longRegressor,
            'classifierThreshold': self.classifierThreshold,
            'thresholdMin': self.thresholdMin,
            'featureNames': self.featureNames,
        }, path)
        print(f"two-stage model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TwoStageOutageModel':
        data = joblib.load(path)
        instance = cls(thresholdMin=data['thresholdMin'])
        instance.classifier = data['classifier']
        instance.shortRegressor = data['shortRegressor']
        instance.longRegressor = data['longRegressor']
        instance.classifierThreshold = data['classifierThreshold']
        instance.featureNames = data['featureNames']
        instance.isTrained = True
        print(f"two-stage model loaded from {path}")
        return instance