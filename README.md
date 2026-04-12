# M-Pesa Fraud Detection

A stronger, presentation-ready fraud detection project built around synthetic
mobile-money transactions and an Isolation Forest anomaly detector.

This version goes beyond a minimal demo by adding:

- richer synthetic M-Pesa-like transaction signals
- reusable feature engineering and preprocessing
- anomaly score threshold calibration
- evaluation metrics for fraud screening
- simple visualizations and top-risk transaction review
- tests that validate the pipeline behavior

## What The Project Demonstrates

- realistic synthetic fraud pattern injection
- unsupervised anomaly detection with Isolation Forest
- feature engineering for transaction risk signals
- threshold selection based on validation labels
- portfolio-friendly Python package structure

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python run_demo.py
pytest
```

## Project Layout

```text
src/fraud_detection/
  __init__.py
  data.py
  preprocess.py
  model.py
  evaluation.py
  visualize.py
run_demo.py
tests/test_pipeline.py
requirements.txt
```

## Modeling Notes

The project uses an unsupervised model because real fraud labels are often
scarce or delayed in production systems. To make the demo more credible, the
synthetic data injects several fraud behaviors instead of only one:

- unusually large transfers
- bursts of rapid repeat transactions
- late-night activity
- new recipient interactions
- high-risk location patterns
- repeated failed PIN attempts

The model is still intentionally simple enough to explain in an attachment or
internship interview.

## Possible Next Extensions

- compare Isolation Forest against Local Outlier Factor or One-Class SVM
- add SHAP or feature-attribution style explanations
- expose the scorer behind a small API or Streamlit app
- simulate drift and monitor alert rates over time
