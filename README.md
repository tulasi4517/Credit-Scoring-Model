# 💳 Full-Stack Credit Scoring & Risk Assessment System (MongoDB Atlas Integration)

A production-ready **Full-Stack Machine Learning Application** built with **Flask**, **MongoDB Atlas Cloud Database**, **scikit-learn**, and a modern **Glassmorphic Frontend Dashboard**.

This system calculates real-time credit scores (300–850 FICO scale), determines automated loan decisions (**APPROVED**, **MANUAL REVIEW**, **DECLINED**), and automatically persists applicant records and risk evaluation analytics to **MongoDB Atlas**.

---

## 🏗️ Full-Stack Architecture

- **Frontend**: HTML5, CSS3 Glassmorphism, JavaScript, live MongoDB connection diagnostics, tabbed assessment dashboard, and cloud history viewer.
- **Backend**: Flask REST API providing prediction endpoints, model inference, and MongoDB database management.
- **Database**: **MongoDB Atlas Cloud Database** (via `pymongo` and `dnspython`) with automated fallback to local/in-memory store.
- **Machine Learning**: Logistic Regression, Random Forest, and Gradient Boosting models trained on domain features (Revolving Credit Utilization, DTI, Income/Debt ratios).

---

## 📁 Repository Structure

```text
├── .env.example                 # Environment configuration template for MongoDB Atlas
├── .env                         # Local environment variables (MongoDB Atlas URI, Port)
├── requirements.txt             # Python dependencies (Flask, PyMongo, dnspython, etc.)
├── database.py                  # MongoDB Atlas database client, CRUD helper functions & status check
├── app.py                       # Full-stack Flask REST API server
├── templates/
│   └── index.html               # Glassmorphic Frontend Dashboard (Assessment Engine & Atlas History)
├── 01_dataset_loader.py         # Synthetic dataset generator
├── 02_preprocessing.py          # Data imputation & feature engineering
├── 03_model_training.py          # ML model training pipeline
├── 04_model_evaluation.py        # ROC-AUC, Gini, KS-Statistic evaluation
├── 05_model_explainability.py    # Log-Odds & feature importance analysis
├── data/                        # CSV datasets
├── models/                      # Trained model artifacts (`.joblib`)
└── reports/                     # Model evaluation reports & charts
```

---

## 🍃 MongoDB Atlas Setup & Configuration

1. Create a free **MongoDB Atlas** cluster at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).
2. Obtain your **MongoDB Atlas Connection String** (e.g. `mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/credit_scoring_db?retryWrites=true&w=majority`).
3. Open or create `.env` in the root directory and set your `MONGO_URI`:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/credit_scoring_db?retryWrites=true&w=majority
DB_NAME=credit_scoring_db
COLLECTION_NAME=credit_assessments
PORT=5000
```

> **Note**: If `MONGO_URI` is not set or network connection is unavailable, the application automatically uses safe fallback mode so the application runs seamlessly in all environments while clearly displaying connectivity status on the UI.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full-Stack Server

```bash
python app.py
```

### 3. Open Dashboard

Navigate to **`http://127.0.0.1:5000`** in your browser.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Evaluates credit risk input, computes FICO score, and saves record to MongoDB Atlas. |
| `GET` | `/api/assessments` | Retrieves stored credit assessments from MongoDB Atlas (with optional `risk_band` query filter). |
| `DELETE` | `/api/assessments/<id>` | Deletes a specific assessment record from MongoDB Atlas. |
| `GET` | `/api/db-status` | Returns MongoDB Atlas connectivity status, database name, and record statistics. |
| `POST` | `/api/db-reconnect` | Triggers immediate reconnection test to MongoDB Atlas. |

---

## 📊 ML Model Metrics Summary

| Model | Accuracy | Precision | Recall | F1-Score | **ROC-AUC** | **Gini Coeff** | **KS Stat** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 65.7% | 25.89% | **66.67%** | **0.3729** | **0.7218** | **0.4437** | **0.3352** |
| **Gradient Boosting** | 68.2% | 23.81% | 49.02% | 0.3205 | 0.6507 | 0.3014 | 0.2292 |
| **Random Forest** | 80.1% | 29.09% | 20.92% | 0.2433 | 0.6376 | 0.2752 | 0.2324 |
