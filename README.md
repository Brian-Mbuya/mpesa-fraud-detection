# Fraud Detection — AI/ML Attachment Project

A compact fraud-detection example using synthetic transaction data and Isolation Forest. This repository is organized to be presentation-ready for an AI/ML attachment:

- Clear package structure under `src/fraud_detection`
- Reproducible data generation and training API
- Small demo script and a minimal test

Quickstart

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
python run_demo.py
```

Repository layout

- `src/fraud_detection/` — library modules (`data.py`, `preprocess.py`, `model.py`, `visualize.py`)
- `notebooks/` — original notebook copy
- `run_demo.py` — runnable pipeline to generate data, train and evaluate
- `requirements.txt` — Python dependencies
- `tests/` — minimal smoke tests

Git / GitHub

To push to GitHub (after creating a repo on GitHub):

```bash
git init
git add .
git commit -m "Initial project import — fraud detection demo"
# Add remote and push (replace URL)
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

License

This project is released under the MIT License. See `LICENSE`.

Contact

If you want me to prepare a polished README with expanded methodology, visuals, or a small report, tell me which parts to expand.