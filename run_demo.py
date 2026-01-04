"""Run a short pipeline: generate data, prepare features, train, evaluate."""
from src.fraud_detection import (
    generate_synthetic_data,
    prepare_features,
    train_isolation_forest,
    predict_is_fraud,
)
from sklearn.metrics import classification_report, confusion_matrix


def main():
    df = generate_synthetic_data()
    X, y = prepare_features(df)

    model = train_isolation_forest(X)
    pred = predict_is_fraud(model, X)

    print("Confusion Matrix:")
    print(confusion_matrix(y, pred))
    print("\nClassification Report:")
    print(classification_report(y, pred))

if __name__ == '__main__':
    main()
