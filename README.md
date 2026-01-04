# 🚨 Fraud Detection – AI/ML Attachment Project

This is a practical **fraud-detection mini-project** built for AI/ML attachment applications.  
It uses **synthetic mobile-money style transactions** and an **Isolation Forest model** to detect suspicious or unusual behaviour.

Small, clean, easy to run, and presentation-ready.

---

## 🎯 What this project demonstrates

- synthetic transaction data generation  
- anomaly / fraud detection using ML  
- unsupervised learning (Isolation Forest)  
- evaluation metrics and simple visualizations  
- clear Python project structure  

A good fit for **attachment, internship applications, and portfolio use**.

---

## ⚙️ Quickstart

### 1️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows**

```bash
.\.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the demo script

```bash
python run_demo.py
```

---

## 🗂️ Repository layout

```text
src/fraud_detection/
 ├── data.py          # generate synthetic transactions
 ├── preprocess.py    # feature encoding / preparation
 ├── model.py         # train Isolation Forest
 └── visualize.py     # plotting helpers
notebooks/            # original exploration notebook
run_demo.py           # end-to-end demo pipeline
requirements.txt      # dependencies
tests/                # light smoke tests
```

---

## 🧠 What the model actually does

- creates fake mobile-money style transactions  
- injects “fraud-like” behaviour such as:
  - unusually large amounts  
  - suspicious late-night activity  
- trains an **unsupervised anomaly detector**  
- flags suspicious transactions without needing real labels  

> No real transaction data is used.

---

## 🧾 License

Released under the **MIT License**.

---

## 🙋 Contact

If you’re reviewing this for **attachment placement** and would like:

- a short report  
- more experiments  
- deployment as a small app  

I’m happy to extend the project.
