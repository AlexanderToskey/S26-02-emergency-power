import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def calculateAccuracy(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    return accuracy_score(yTrue, yPred)


def calculatePrecision(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    return precision_score(yTrue, yPred, zero_division=0)


def calculateRecall(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    return recall_score(yTrue, yPred, zero_division=0)


def calculateF1(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    return f1_score(yTrue, yPred, zero_division=0)


def calculateROCAUC(yTrue: np.ndarray, yProb: np.ndarray) -> float:
    """
    ROC-AUC requires probability scores.
    """
    if yProb is None:
        return float("nan")
    return roc_auc_score(yTrue, yProb)


def calculatePRAUC(yTrue: np.ndarray, yProb: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) is more informative for imbalanced datasets.
    """
    if yProb is None:
        return float("nan")
    return average_precision_score(yTrue, yProb)


def evaluateModel(
    yTrue: np.ndarray,
    yPred: np.ndarray,
    yProb: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Runs full classification evaluation.

    Args:
        yTrue: actual binary labels (0/1)
        yPred: predicted binary labels (0/1)
        yProb: predicted probabilities for class 1 (optional but recommended)

    Returns:
        dict of evaluation metrics
    """
    yTrue = np.array(yTrue)
    yPred = np.array(yPred)

    metrics = {
        "accuracy": calculateAccuracy(yTrue, yPred),
        "precision": calculatePrecision(yTrue, yPred),
        "recall": calculateRecall(yTrue, yPred),
        "f1_score": calculateF1(yTrue, yPred),
        "roc_auc": calculateROCAUC(yTrue, yProb) if yProb is not None else float("nan"),
        "pr_auc": calculatePRAUC(yTrue, yProb) if yProb is not None else float("nan"),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(yTrue, yPred).ravel()

    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "positive_rate": float(np.mean(yTrue))
    })

    return metrics


def printEvaluationReport(metrics: Dict[str, float]):
    """
    Prints formatted evaluation report for outage occurrence model.
    """

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT (OUTAGE OCCURRENCE)")
    print("=" * 60)

    print("\nClassification Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']}")
    print(f"  FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}")
    print(f"  TP: {metrics['true_positives']}")

    print("\nDataset Info:")
    print(f"  Positive rate: {metrics['positive_rate']:.4f}")

    # Recommended thresholds for imbalanced outage prediction
    print("\nModel Quality Check (Occurrence Context):")

    if metrics["recall"] >= 0.70:
        recall_status = "PASS"
    elif metrics["recall"] >= 0.50:
        recall_status = "MARGINAL"
    else:
        recall_status = "FAIL"

    if metrics["pr_auc"] >= 0.30:
        pr_status = "PASS"
    elif metrics["pr_auc"] >= 0.15:
        pr_status = "MARGINAL"
    else:
        pr_status = "FAIL"

    print(f"  Recall >= 0.70 (storm detection): {recall_status}")
    print(f"  PR-AUC >= 0.30 (imbalance robustness): {pr_status}")

    print("=" * 60 + "\n")