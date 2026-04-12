"""Model training and scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score


def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.03,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
) -> IsolationForest:
    """Train an Isolation Forest model."""

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X)
    return model


def score_dataset(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return a fraud score where larger values are more suspicious."""

    return -model.decision_function(X)


def calibrate_threshold(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    strategy: str = "best_f1",
) -> float:
    """Choose a score threshold for converting anomaly scores to fraud flags."""

    y = np.asarray(y_true)
    if strategy == "quantile":
        contamination = max(0.001, float(y.mean()))
        return float(np.quantile(scores, 1.0 - contamination))
    if strategy != "best_f1":
        raise ValueError("strategy must be 'best_f1' or 'quantile'")

    quantiles = np.linspace(0.8, 0.995, 80)
    thresholds = np.quantile(scores, quantiles)
    best_threshold = thresholds[0]
    best_score = -1.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold
