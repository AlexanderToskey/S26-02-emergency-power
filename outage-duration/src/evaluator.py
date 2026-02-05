"""
evaluator.py - model evaluation metrics

calculates MAE, RMSE, MAPE and the custom accuracy metrics
defined in the PRD (within tolerance for short vs long outages)
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
                               shortThreshold: float = 240,
                               shortTolerance: float = 30,
                               longTolerance: float = 120) -> Dict[str, float]:
    """
    calculates accuracy within tolerance as defined in the PRD

    for outages < 4hr (240 min): accuracy within +/- 30 min
    for outages >= 4hr: accuracy within +/- 2hr (120 min)

    Args:
        yTrue: actual durations in minutes
        yPred: predicted durations in minutes
        shortThreshold: cutoff between short and long outages (default 240 min = 4hr)
        shortTolerance: tolerance for short outages (default 30 min)
        longTolerance: tolerance for long outages (default 120 min)

    Returns:
        dict with accuracy metrics for short, long, and overall
    """
    errors = np.abs(yTrue - yPred)

    # split into short and long outages
    shortMask = yTrue < shortThreshold
    longMask = yTrue >= shortThreshold

    # calculate accuracy for each group
    shortWithinTol = errors[shortMask] <= shortTolerance
    longWithinTol = errors[longMask] <= longTolerance

    shortAcc = np.mean(shortWithinTol) * 100 if shortMask.sum() > 0 else 0
    longAcc = np.mean(longWithinTol) * 100 if longMask.sum() > 0 else 0

    # overall accuracy (each prediction judged by appropriate tolerance)
    withinTol = np.where(yTrue < shortThreshold,
                         errors <= shortTolerance,
                         errors <= longTolerance)
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
    print(f"  MAE:  {metrics['mae']:.2f} minutes")
    print(f"  RMSE: {metrics['rmse']:.2f} minutes")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\nAccuracy Within Tolerance:")
    print(f"  Short outages (<4hr): {metrics['short_outage_accuracy']:.1f}% "
          f"(n={metrics['short_outage_count']})")
    print(f"  Long outages (>=4hr): {metrics['long_outage_accuracy']:.1f}% "
          f"(n={metrics['long_outage_count']})")
    print(f"  Overall: {metrics['overall_accuracy']:.1f}%")

    # check against PRD targets
    print(f"\nPRD Target Check:")
    mapeOk = "PASS" if metrics['mape'] < 15 else ("MARGINAL" if metrics['mape'] < 25 else "FAIL")
    accOk = "PASS" if metrics['overall_accuracy'] >= 90 else ("MARGINAL" if metrics['overall_accuracy'] >= 70 else "FAIL")
    print(f"  MAPE < 15%: {mapeOk}")
    print(f"  Accuracy >= 90%: {accOk}")

    print("="*50 + "\n")
