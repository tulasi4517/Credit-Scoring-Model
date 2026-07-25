import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load preprocessor artifact
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.joblib")
if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
else:
    preprocessor = None

# Load available trained models
models = {}
for model_key in ['logistic_regression', 'random_forest', 'gradient_boosting']:
    path = os.path.join("models", f"{model_key}.joblib")
    if os.path.exists(path):
        models[model_key] = joblib.load(path)

def compute_ratios(df):
    """Computes engineered ratios for inference."""
    df_feat = df.copy()
    eps = 1e-5
    df_feat['revol_utilization'] = np.round(df_feat['current_balance'] / (df_feat['credit_limit'] + eps), 4)
    df_feat['dti_ratio'] = np.round(df_feat['total_debt'] / (df_feat['annual_income'] + eps), 4)
    df_feat['debt_to_limit_ratio'] = np.round(df_feat['total_debt'] / (df_feat['credit_limit'] + eps), 4)
    df_feat['income_per_emp_year'] = np.round(df_feat['annual_income'] / (df_feat['employment_years'] + 1.0), 2)
    return df_feat

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        selected_model_name = data.get('model_name', 'logistic_regression')
        
        # Build DataFrame from input JSON
        raw_input = pd.DataFrame([{
            'age': float(data['age']),
            'annual_income': float(data['annual_income']),
            'total_debt': float(data['total_debt']),
            'credit_limit': float(data['credit_limit']),
            'current_balance': float(data['current_balance']),
            'num_late_payments_12m': float(data['num_late_payments_12m']),
            'employment_years': float(data['employment_years'])
        }])
        
        # Preprocessing & Imputation
        if preprocessor is not None:
            raw_imputed = pd.DataFrame(
                preprocessor['imputer'].transform(raw_input),
                columns=preprocessor['raw_feature_names']
            )
        else:
            raw_imputed = raw_input
            
        # Feature Engineering
        engineered = compute_ratios(raw_imputed)
        
        # Feature Scaling
        if preprocessor is not None:
            scaled_input = pd.DataFrame(
                preprocessor['scaler'].transform(engineered),
                columns=preprocessor['feature_names']
            )
        else:
            scaled_input = engineered
            
        # Model Selection
        model = models.get(selected_model_name, list(models.values())[0])
        
        # Predict Default Probability
        prob_default = float(model.predict_proba(scaled_input)[0, 1])
        
        # FICO-style Credit Score Conversion (300 to 850 scale)
        credit_score = int(round(300 + 550 * (1.0 - prob_default)))
        credit_score = max(300, min(850, credit_score))
        
        # Risk Band & Decision Thresholds
        if credit_score >= 720:
            risk_band = "Low Risk"
            decision = "APPROVED"
        elif credit_score >= 620:
            risk_band = "Medium Risk"
            decision = "MANUAL REVIEW"
        else:
            risk_band = "High Risk"
            decision = "DECLINED"
            
        revol_util = float(engineered['revol_utilization'].iloc[0])
        dti_ratio = float(engineered['dti_ratio'].iloc[0])
        
        return jsonify({
            'credit_score': credit_score,
            'probability_default': round(prob_default, 4),
            'risk_band': risk_band,
            'decision': decision,
            'revol_utilization': round(revol_util, 4),
            'dti_ratio': round(dti_ratio, 4),
            'model_used': selected_model_name.replace("_", " ").title()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 65)
    print("  Credit Scoring Application Server Running on http://127.0.0.1:5000")
    print("=" * 65)
    # Disable debug mode and reloader to prevent Windows subprocess crashes
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
