import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X, n_estimators=200, contamination=0.03, random_state=42):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_is_fraud(model, X):
    """Return binary predictions (1=fraud, 0=normal)."""
    pred = model.predict(X)
    # convert: normal=1 anomaly=-1 → 0/1
    return np.where(pred == -1, 1, 0)
