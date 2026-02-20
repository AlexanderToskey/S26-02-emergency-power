"""
evaluator.py - model evaluation metrics for outage scope

calculates MAE, RMSE, MAPE and the custom accuracy metrics
defined in the PRD (within tolerance for small vs large outages)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculateMAE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    mean absolute error

    Args:
        yTrue: actual values
        yPred: predicted values

    Returns:
        MAE in same units as input (customers)
    """
    return np.mean(np.abs(yTrue - yPred))


def calculateRMSE(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    root mean squared error

    Args:
        yTrue: actual values
        yPred: predicted values

    Returns:
        RMSE in same units as input (customers)
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
                               smallThreshold: float = 500,
                               smallToleranceAbs: float = 50,
                               largeTolerancePct: float = 0.20) -> Dict[str, float]:
    """
    calculates accuracy within tolerance adapted for scope

    for outages < 500 customers: accuracy within +/- 50 customers
    for outages >= 500 customers: accuracy within +/- 20% of actual

    Args:
        yTrue: actual scope in customers
        yPred: predicted scope in customers
        smallThreshold: cutoff between small and large outages (default 500 customers)
        smallToleranceAbs: absolute tolerance for small outages (default 50 customers)
        largeTolerancePct: percentage tolerance for large outages (default 20%)

    Returns:
        dict with accuracy metrics for small, large, and overall
    """
    errorsAbs = np.abs(yTrue - yPred)
    
    # Avoid division by zero for percentage errors
    with np.errstate(divide='ignore', invalid='ignore'):
        errorsPct = np.where(yTrue > 0, errorsAbs / yTrue, 0)

    # split into small and large outages
    smallMask = yTrue < smallThreshold
    largeMask = yTrue >= smallThreshold

    # calculate accuracy for each group
    smallWithinTol = errorsAbs[smallMask] <= smallToleranceAbs
    largeWithinTol = errorsPct[largeMask] <= largeTolerancePct

    smallAcc = np.mean(smallWithinTol) * 100 if smallMask.sum() > 0 else 0
    largeAcc = np.mean(largeWithinTol) * 100 if largeMask.sum() > 0 else 0

    # overall accuracy (each prediction judged by appropriate tolerance)
    withinTol = np.where(yTrue < smallThreshold,
                         errorsAbs <= smallToleranceAbs,
                         errorsPct <= largeTolerancePct)
    overallAcc = np.mean(withinTol) * 100

    return {
        'small_outage_accuracy': smallAcc,
        'large_outage_accuracy': largeAcc,
        'overall_accuracy': overallAcc,
        'small_outage_count': int(smallMask.sum()),
        'large_outage_count': int(largeMask.sum())
    }


def evaluateModel(yTrue: np.ndarray, yPred: np.ndarray) -> Dict[str, float]:
    """
    runs all evaluation metrics and returns a summary

    this is the main function to call for full model evaluation.

    Args:
        yTrue: actual scope in customers
        yPred: predicted scope in customers

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


def printEvaluationReport(metrics: Dict[str, float]):
    """
    prints a formatted evaluation report

    Args:
        metrics: dict from evaluateModel()
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT (SCOPE)")
    print("="*50)

    print(f"\nError Metrics:")
    print(f"  MAE:  {metrics['mae']:.2f} customers")
    print(f"  RMSE: {metrics['rmse']:.2f} customers")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\nAccuracy Within Tolerance:")
    print(f"  Small outages (<500 cust): {metrics['small_outage_accuracy']:.1f}% "
          f"(n={metrics['small_outage_count']})")
    print(f"  Large outages (>=500 cust): {metrics['large_outage_accuracy']:.1f}% "
          f"(n={metrics['large_outage_count']})")
    print(f"  Overall: {metrics['overall_accuracy']:.1f}%")

    # check against PRD targets
    print(f"\nPRD Target Check:")
    mapeOk = "PASS" if metrics['mape'] < 15 else ("MARGINAL" if metrics['mape'] < 25 else "FAIL")
    accOk = "PASS" if metrics['overall_accuracy'] >= 90 else ("MARGINAL" if metrics['overall_accuracy'] >= 70 else "FAIL")
    print(f"  MAPE < 15%: {mapeOk}")
    print(f"  Accuracy >= 90%: {accOk}")

    print("="*50 + "\n")