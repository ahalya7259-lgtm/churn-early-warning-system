# Churn Early-Warning System

Predict which customers are likely to churn, understand *why*, and act early.

This project is an end-to-end machine learning system that includes:
- Data cleaning & feature engineering
- Model training (Logistic Regression / Random Forest)
- Churn probability scoring
- Feature importance / risk drivers
- **FastAPI** backend for predictions
- **Streamlit** dashboard for business users

> Portfolio project by **Ahalya** — Aspiring Data Analyst

---

## Features

| Feature | Description |
|---------|-------------|
| EDA & Training Notebook | Explore data and train models |
| Saved Model | Best model persisted with joblib |
| Prediction Script | Score new customers |
| FastAPI | `POST /predict` endpoint |
| Streamlit Dashboard | Interactive risk view + single customer scoring |
| Risk Drivers | Shows top factors contributing to churn |

---

## Project Structure

```
churn-early-warning-system/
├── data/
│   ├── raw/                 # place original dataset here
│   └── processed/           # cleaned data (generated)
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│   ├── predict.py
│   └── explain.py
├── api/
│   └── main.py              # FastAPI app
├── dashboard/
│   └── app.py               # Streamlit app
├── models/                  # saved .joblib files
├── requirements.txt
└── README.md
```

---

## Dataset

Recommended starter dataset (free on Kaggle):

**Telco Customer Churn**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in `data/raw/`.

You can also use any similar churn dataset (Bank, SaaS, etc.).

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ahalya7259-lgtm/churn-early-warning-system.git
cd churn-early-warning-system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Dataset

Place the CSV in `data/raw/telco_churn.csv` (or update the path in the scripts).

### 3. Train the Model

```bash
python src/train_model.py
```

This will:
- Clean the data
- Train Logistic Regression + Random Forest
- Save the best model to `models/churn_model.joblib`
- Print evaluation metrics

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

API docs: http://localhost:8000/docs

### 5. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

---

## Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 850.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

---

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn
- Joblib (model persistence)
- FastAPI + Uvicorn
- Streamlit
- Matplotlib / Seaborn / Plotly

---

## Author

**Ahalya** — BCA Graduate | Aspiring Data Analyst  
GitHub: [ahalya7259-lgtm](https://github.com/ahalya7259-lgtm)  
LinkedIn: [linkedin.com/in/ahalya-k-096a6a30a](https://linkedin.com/in/ahalya-k-096a6a30a)

## License

MIT
