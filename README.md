# 💳 Credit Scoring Model - Machine Learning System

An end-to-end Machine Learning Credit Risk Assessment Pipeline built with Python, scikit-learn, and Flask. This project generates synthetic credit applicant data, performs feature engineering, trains interpretable and ensemble classification models, evaluates credit domain metrics (ROC-AUC, Gini Coefficient, KS Statistic), and provides an interactive web dashboard for real-time credit score generation (300-850 FICO scale).

---

## 🌟 Key Features

- **Synthetic Data Engine**: Generates 5,000 realistic credit applicant profiles with ~5% missing values (MCAR) for preprocessing benchmarking.
- **Domain Feature Engineering**: Computes Revolving Credit Utilization, Debt-to-Income (DTI) ratio, Debt-to-Limit ratio, and Income per Employment Year.
- **Imbalance-Aware Model Suite**:
  - **Logistic Regression**: Interpretable banking industry baseline.
  - **Random Forest Classifier**: Non-linear ensemble model.
  - **Gradient Boosting**: High-performance decision tree booster.
- **Credit Domain Evaluation**:
  - Standard Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrices.
  - Banking Metrics: **ROC-AUC (0.7218)**, **Gini Coefficient (0.4437)**, **KS Statistic (0.3352)**.
- **Interactive Web App**: Modern Flask web interface allowing users to input financial parameters and receive instant credit risk scoring and automated loan decisioning (**Approved**, **Manual Review**, **Declined**).

---

## 📁 Repository Structure

```text
├── 01_dataset_loader.py         # Synthetic data generation engine
├── 02_preprocessing.py          # Imputation, feature engineering & scaling
├── 03_model_training.py          # Model training with class imbalance handling
├── 04_model_evaluation.py        # ROC-AUC, Gini, KS-Statistic & plot generation
├── 05_model_explainability.py    # Log-Odds coefficients & feature importance ranking
├── app.py                       # Flask web application backend
├── templates/
│   └── index.html               # Web application UI dashboard
├── data/                        # Raw & preprocessed CSV datasets
├── models/                      # Trained model artifacts & preprocessor pipeline
└── reports/                     # Evaluation metrics summary & high-res plots
```

---

## 📊 Model Performance Summary

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | **ROC-AUC** | **Gini Coefficient** | **KS Statistic** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 65.7% | 25.89% | **66.67%** | **0.3729** | **0.7218** | **0.4437** | **0.3352** |
| **Gradient Boosting** | 68.2% | 23.81% | 49.02% | 0.3205 | 0.6507 | 0.3014 | 0.2292 |
| **Random Forest** | 80.1% | 29.09% | 20.92% | 0.2433 | 0.6376 | 0.2752 | 0.2324 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required Packages:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn flask joblib
  ```

### Running the Web Application

To launch the interactive Credit Scoring Dashboard UI:

```bash
python app.py
```

Then navigate to **`http://127.0.0.1:5000`** in your browser.

---

## 💻 Running the ML Pipeline Step-by-Step

If you wish to re-run the pipeline steps from scratch:

```bash
python 01_dataset_loader.py
python 02_preprocessing.py
python 03_model_training.py
python 04_model_evaluation.py
python 05_model_explainability.py
python app.py
```
