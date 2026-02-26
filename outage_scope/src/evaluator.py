"""
evaluator.py - model evaluation metrics

calculates MAE, RMSE, MAPE and the custom accuracy metrics
defined in the PRD (within tolerance for short vs long outages)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculateMAE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    mean absolute error

    Args:
        yTrue: actual values
        yPred: predicted values

    Returns:
        MAE in same units as input (minutes)
    """
    return np.mean(np.abs(yTrue - yPred))


def calculateRMSE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    root mean squared error

    Args:
        yTrue: actual values
        yPred: predicted values

    Returns:
        RMSE in same units as input (minutes)
    """
    return np.sqrt(np.mean((yTrue - yPred) ** 2))


def calculateMAPE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    mean absolute percentage error

    note: filters out any zeros in yTrue to avoid division issues

    Args:
        yTrue: actual values
        yPred: predicted values

    Returns:
        MAPE as a percentage (0-100 scale)
    """
    # avoid division by zero
    mask = yTrue != 0
    yTrue = yTrue[mask]
    yPred = yPred[mask]

    return np.mean(np.abs((yTrue - yPred) / yTrue)) * 100


def calculateToleranceAccuracy(yTrue: np.ndarray, yPred: np.ndarray,
                               shortThreshold: float = 500, # 500 Customers
                               shortTolerance: float = 50,  # +/- 50 customers
                               longTolerancePct: float = 0.20) -> Dict[str, float]: # +/- 20%
    """
    calculates accuracy within tolerance as defined in the PRD for Scope

    for outages < 500 customers: accuracy within +/- 50 customers
    for outages >= 500 customers: accuracy within +/- 20%

    Args:
        yTrue: actual customer counts
        yPred: predicted customer counts
        shortThreshold: cutoff between small and large (default 500)
        shortTolerance: tolerance for small outages (default 50 customers)
        longTolerancePct: percentage tolerance for large outages (default 0.20)
    """
    errors = np.abs(yTrue - yPred)

    # split into small and large outages
    shortMask = yTrue < shortThreshold
    longMask = yTrue >= shortThreshold

    # calculate accuracy for each group
    shortWithinTol = errors[shortMask] <= shortTolerance
    # For large outages, tolerance is relative to actual count
    longWithinTol = errors[longMask] <= (yTrue[longMask] * longTolerancePct)

    shortAcc = np.mean(shortWithinTol) * 100 if shortMask.sum() > 0 else 0
    longAcc = np.mean(longWithinTol) * 100 if longMask.sum() > 0 else 0

    # overall accuracy
    withinTol = np.where(yTrue < shortThreshold,
                         errors <= shortTolerance,
                         errors <= (yTrue * longTolerancePct))
    overallAcc = np.mean(withinTol) * 100

    return {
        'short_outage_accuracy': shortAcc,
        'long_outage_accuracy': longAcc,
        'overall_accuracy': overallAcc,
        'short_outage_count': int(shortMask.sum()),
        'long_outage_count': int(longMask.sum())
    }


def evaluateModel(yTrue: np.ndarray, yPred: np.ndarray) -> Dict[str, float]:
    """
    runs all evaluation metrics and returns a summary

    this is the main function to call for full model evaluation.
    includes all the metrics specified in the PRD

    Args:
        yTrue: actual durations in minutes
        yPred: predicted durations in minutes

    Returns:
        dict with all metrics
    """
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)

    metrics = {
        'mae': calculateMAE(yTrue, yPred),
        'rmse': calculateRMSE(yTrue, yPred),
        'mape': calculateMAPE(yTrue, yPred)
    }

    # add tolerance-based accuracy
    tolMetrics = calculateToleranceAccuracy(yTrue, yPred)
    metrics.update(tolMetrics)

    return metrics


def evaluateClassifier(
    yTrue: np.ndarray,
    yPredLabel: np.ndarray,
    threshold: float = 500.0,
) -> Dict[str, float]:
    """
    Evaluate Stage 1 classifier routing performance.

    Args:
        yTrue: true outage durations in minutes
        yPredLabel: predicted routing labels (0=short, 1=long)
        threshold: short/long cutoff in minutes (default 240)

    Returns:
        dict with classifier and routing metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    yTrue = np.array(yTrue)
    yPredLabel = np.array(yPredLabel)
    trueLabel = (yTrue >= threshold).astype(int)

    precision = precision_score(trueLabel, yPredLabel, zero_division=0)
    recall = recall_score(trueLabel, yPredLabel, zero_division=0)
    f1 = f1_score(trueLabel, yPredLabel, zero_division=0)
    acc = accuracy_score(trueLabel, yPredLabel)

    shortMask = trueLabel == 0
    longMask = trueLabel == 1

    longRecall = np.mean(yPredLabel[longMask] == 1) * 100 if longMask.sum() > 0 else 0.0
    shortRecall = np.mean(yPredLabel[shortMask] == 0) * 100 if shortMask.sum() > 0 else 0.0

    return {
        'classifier_accuracy': acc * 100,
        'classifier_precision_long': precision * 100,
        'classifier_recall_long': recall * 100,
        'classifier_f1_long': f1,
        'routing_accuracy': acc * 100,
        'long_correctly_routed_pct': longRecall,
        'short_correctly_routed_pct': shortRecall,
        'true_short_count': int(shortMask.sum()),
        'true_long_count': int(longMask.sum()),
        'routed_short_count': int((yPredLabel == 0).sum()),
        'routed_long_count': int((yPredLabel == 1).sum()),
    }


def printTwoStageReport(
    classifierMetrics: Dict[str, float],
    regressionMetrics: Dict[str, float],
    singleModelMetrics: Optional[Dict[str, float]] = None,
):
    """
    Print a formatted two-stage model evaluation report.

    Args:
        classifierMetrics: from evaluateClassifier()
        regressionMetrics: from evaluateModel()
        singleModelMetrics: optional baseline single-model metrics for comparison
    """
    print("\n" + "=" * 55)
    print("TWO-STAGE MODEL EVALUATION REPORT")
    print("=" * 55)

    print(f"\nStage 1 Classifier:")
    print(f"  Routing accuracy:          {classifierMetrics['routing_accuracy']:.1f}%")
    print(f"  Long outages → long reg:   {classifierMetrics['long_correctly_routed_pct']:.1f}% "
          f"(n={classifierMetrics['true_long_count']})")
    print(f"  Short outages → short reg: {classifierMetrics['short_correctly_routed_pct']:.1f}% "
          f"(n={classifierMetrics['true_short_count']})")
    print(f"  Precision (long):          {classifierMetrics['classifier_precision_long']:.1f}%")
    print(f"  Recall (long):             {classifierMetrics['classifier_recall_long']:.1f}%")
    print(f"  F1 (long):                 {classifierMetrics['classifier_f1_long']:.3f}")
    print(f"  Routed to short reg:       {classifierMetrics['routed_short_count']:,}")
    print(f"  Routed to long reg:        {classifierMetrics['routed_long_count']:,}")

    print(f"\nStage 2 End-to-End Regression:")
    print(f"  MAE:  {regressionMetrics['mae']:.2f} customers")
    print(f"  RMSE: {regressionMetrics['rmse']:.2f} customers")
    print(f"  MAPE: {regressionMetrics['mape']:.2f}%")

    print(f"\nTolerance Accuracy:")
    print(f"  Small outages (<500 cust): {regressionMetrics['short_outage_accuracy']:.1f}% "
          f"(n={regressionMetrics['short_outage_count']})")
    print(f"  Large outages (>=500 cust): {regressionMetrics['long_outage_accuracy']:.1f}% "
          f"(n={regressionMetrics['long_outage_count']})")
    print(f"  Overall: {regressionMetrics['overall_accuracy']:.1f}%")

    if singleModelMetrics is not None:
        delta = regressionMetrics['overall_accuracy'] - singleModelMetrics['overall_accuracy']
        deltaShort = regressionMetrics['short_outage_accuracy'] - singleModelMetrics['short_outage_accuracy']
        deltaLong = regressionMetrics['long_outage_accuracy'] - singleModelMetrics['long_outage_accuracy']
        print(f"\nVs single-stage baseline:")
        print(f"  Overall accuracy: {'+' if delta >= 0 else ''}{delta:.1f}pp "
              f"({singleModelMetrics['overall_accuracy']:.1f}% → {regressionMetrics['overall_accuracy']:.1f}%)")
        print(f"  Short accuracy:   {'+' if deltaShort >= 0 else ''}{deltaShort:.1f}pp "
              f"({singleModelMetrics['short_outage_accuracy']:.1f}% → {regressionMetrics['short_outage_accuracy']:.1f}%)")
        print(f"  Long accuracy:    {'+' if deltaLong >= 0 else ''}{deltaLong:.1f}pp "
              f"({singleModelMetrics['long_outage_accuracy']:.1f}% → {regressionMetrics['long_outage_accuracy']:.1f}%)")

    print(f"\nPRD Target Check:")
    mapeOk = "PASS" if regressionMetrics['mape'] < 15 else ("MARGINAL" if regressionMetrics['mape'] < 25 else "FAIL")
    accOk = "PASS" if regressionMetrics['overall_accuracy'] >= 90 else ("MARGINAL" if regressionMetrics['overall_accuracy'] >= 70 else "FAIL")
    print(f"  MAPE < 15%:      {mapeOk}")
    print(f"  Accuracy >= 90%: {accOk}")
    print("=" * 55 + "\n")


def printEvaluationReport(metrics: Dict[str, float]):
    """
    prints a formatted evaluation report

    Args:
        metrics: dict from evaluateModel()
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)

    print(f"\nError Metrics:")
    print(f"  MAE:  {metrics['mae']:.2f} customers")
    print(f"  RMSE: {metrics['rmse']:.2f} customers")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\nAccuracy Within Tolerance:")
    print(f"  Short outages (<500 Customers): {metrics['short_outage_accuracy']:.1f}% "
          f"(n={metrics['short_outage_count']})")
    print(f"  Long outages (>=Customers): {metrics['long_outage_accuracy']:.1f}% "
          f"(n={metrics['long_outage_count']})")
    print(f"  Overall: {metrics['overall_accuracy']:.1f}%")

    # check against PRD targets
    print(f"\nPRD Target Check:")
    mapeOk = "PASS" if metrics['mape'] < 15 else ("MARGINAL" if metrics['mape'] < 25 else "FAIL")
    accOk = "PASS" if metrics['overall_accuracy'] >= 90 else ("MARGINAL" if metrics['overall_accuracy'] >= 70 else "FAIL")
    print(f"  MAPE < 15%: {mapeOk}")
    print(f"  Accuracy >= 90%: {accOk}")

    print("="*50 + "\n")
