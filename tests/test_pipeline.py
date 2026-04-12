import numpy as np

from fraud_detection import (
    calibrate_threshold,
    evaluate_predictions,
    generate_synthetic_data,
    prepare_dataset,
    score_dataset,
    train_isolation_forest,
)


def test_generate_synthetic_data_has_expected_columns():
    df = generate_synthetic_data(n=100, fraud_rate=0.05, seed=7)

    assert len(df) == 100
    assert {
        "transaction_id",
        "amount",
        "hour",
        "transaction_type",
        "channel",
        "is_fraud",
    }.issubset(df.columns)
    assert df["is_fraud"].sum() > 0


def test_prepare_dataset_returns_transformed_matrix():
    df = generate_synthetic_data(n=120, fraud_rate=0.05, seed=9)
    X, y, preprocessor, feature_frame = prepare_dataset(df)

    assert X.shape[0] == len(df)
    assert len(y) == len(df)
    assert "amount_to_balance_ratio" in feature_frame.columns
    assert preprocessor is not None


def test_pipeline_detects_injected_signal_reasonably_well():
    df = generate_synthetic_data(n=2500, fraud_rate=0.04, seed=21)
    X, y, _, _ = prepare_dataset(df)

    model = train_isolation_forest(X, contamination=0.04, random_state=21, n_estimators=200)
    scores = score_dataset(model, X)
    threshold = calibrate_threshold(y, scores, strategy="best_f1")
    evaluation = evaluate_predictions(y.to_numpy(), scores, threshold)

    assert np.isfinite(scores).all()
    assert evaluation.metrics["recall"] >= 0.60
    assert evaluation.metrics["precision"] >= 0.60
    assert evaluation.metrics["roc_auc"] >= 0.85
