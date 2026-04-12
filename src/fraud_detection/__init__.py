"""Fraud detection package."""

from .data import generate_synthetic_data
from .evaluation import EvaluationResult, evaluate_predictions
from .model import calibrate_threshold, score_dataset, train_isolation_forest
from .preprocess import build_feature_frame, prepare_dataset

__all__ = [
    "EvaluationResult",
    "build_feature_frame",
    "calibrate_threshold",
    "evaluate_predictions",
    "generate_synthetic_data",
    "prepare_dataset",
    "score_dataset",
    "train_isolation_forest",
]
