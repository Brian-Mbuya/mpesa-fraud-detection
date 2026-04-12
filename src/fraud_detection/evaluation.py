"""Evaluation helpers for anomaly predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    predictions: np.ndarray
    confusion_matrix: np.ndarray
    metrics: dict[str, float]


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> EvaluationResult:
    """Evaluate thresholded anomaly scores against labels."""

    preds = (scores >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }
    return EvaluationResult(
        predictions=preds,
        confusion_matrix=confusion_matrix(y_true, preds),
        metrics=metrics,
    )
