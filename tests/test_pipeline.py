def test_pipeline_runs():
    # smoke test: import library and run small pipeline
    from src.fraud_detection import generate_synthetic_data, prepare_features, train_isolation_forest, predict_is_fraud

    df = generate_synthetic_data(n=200, fraud_count=5)
    X, y = prepare_features(df)
    model = train_isolation_forest(X, n_estimators=10, contamination=0.05)
    preds = predict_is_fraud(model, X)
    assert len(preds) == len(df)
