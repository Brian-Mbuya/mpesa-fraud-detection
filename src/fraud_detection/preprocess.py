"""Feature engineering and preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from the raw synthetic transactions."""

    feature_frame = df.copy()
    feature_frame["log_amount"] = np.log1p(feature_frame["amount"])
    feature_frame["amount_to_balance_ratio"] = feature_frame["amount"] / np.maximum(
        feature_frame["balance_before"],
        1.0,
    )
    feature_frame["hour_sin"] = np.sin(2 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["hour_cos"] = np.cos(2 * np.pi * feature_frame["hour"] / 24.0)
    feature_frame["night_transaction"] = feature_frame["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    feature_frame["young_account"] = (feature_frame["account_age_days"] < 45).astype(int)
    feature_frame["high_velocity"] = (feature_frame["transaction_velocity_1h"] >= 5).astype(int)
    return feature_frame


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, ColumnTransformer, pd.DataFrame]:
    """Return transformed features, labels, the preprocessor, and engineered frame."""

    feature_frame = build_feature_frame(df)
    y = feature_frame["is_fraud"].astype(int)

    numeric_features = [
        "amount",
        "log_amount",
        "hour",
        "hour_sin",
        "hour_cos",
        "account_age_days",
        "failed_pin_attempts",
        "location_risk_score",
        "transaction_velocity_1h",
        "balance_before",
        "balance_after",
        "amount_to_balance_ratio",
        "recipient_is_new",
        "is_weekend",
        "night_transaction",
        "young_account",
        "high_velocity",
    ]
    categorical_features = ["transaction_type", "channel"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    X = preprocessor.fit_transform(feature_frame[numeric_features + categorical_features])
    return X, y, preprocessor, feature_frame
