"""Run the fraud detection demo end to end."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_detection import (
    calibrate_threshold,
    evaluate_predictions,
    generate_synthetic_data,
    prepare_dataset,
    score_dataset,
    train_isolation_forest,
)


def main() -> None:
    df = generate_synthetic_data(n=5000, fraud_rate=0.03, seed=42)
    X, y, preprocessor, feature_frame = prepare_dataset(df)

    model = train_isolation_forest(X, contamination=0.03, random_state=42)
    scores = score_dataset(model, X)
    threshold = calibrate_threshold(y, scores, strategy="best_f1")
    evaluation = evaluate_predictions(y, scores, threshold)

    print("Fraud Detection Demo")
    print("=" * 60)
    print(f"Transactions: {len(df)}")
    print(f"Injected fraud rate: {y.mean():.2%}")
    print(f"Chosen threshold: {threshold:.4f}")
    print()
    print("Metrics")
    print("-" * 60)
    for name, value in evaluation.metrics.items():
        print(f"{name:>12}: {value:.4f}")

    print()
    print("Confusion Matrix")
    print("-" * 60)
    print(evaluation.confusion_matrix)

    ranked = feature_frame.copy()
    ranked["fraud_score"] = scores
    ranked["predicted_fraud"] = evaluation.predictions
    ranked["is_fraud"] = y.to_numpy()
    columns = [
        "amount",
        "transaction_type",
        "channel",
        "transaction_velocity_1h",
        "failed_pin_attempts",
        "location_risk_score",
        "fraud_score",
        "predicted_fraud",
        "is_fraud",
    ]

    print()
    print("Top flagged transactions")
    print("-" * 60)
    print(ranked.sort_values("fraud_score", ascending=False)[columns].head(10).to_string(index=False))

    _ = preprocessor


if __name__ == "__main__":
    main()
