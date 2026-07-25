import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """Loads raw credit dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run 01_dataset_loader.py first.")
    return pd.read_csv(file_path)

def engineer_features(df):
    """
    Engineers financial ratios and credit risk domain features.
    
    Features engineered:
    - revol_utilization: current_balance / credit_limit
    - dti_ratio: total_debt / annual_income (Debt-to-Income)
    - debt_to_limit_ratio: total_debt / credit_limit
    - income_per_emp_year: annual_income / (employment_years + 1)
    """
    df_feat = df.copy()
    
    # Avoid division by zero
    eps = 1e-5
    
    df_feat['revol_utilization'] = np.round(df_feat['current_balance'] / (df_feat['credit_limit'] + eps), 4)
    df_feat['dti_ratio'] = np.round(df_feat['total_debt'] / (df_feat['annual_income'] + eps), 4)
    df_feat['debt_to_limit_ratio'] = np.round(df_feat['total_debt'] / (df_feat['credit_limit'] + eps), 4)
    df_feat['income_per_emp_year'] = np.round(df_feat['annual_income'] / (df_feat['employment_years'] + 1.0), 2)
    
    return df_feat

def run_preprocessing_pipeline():
    print("=" * 65)
    print("  Credit Scoring Model - Preprocessing & Feature Engineering")
    print("=" * 65)
    
    raw_path = os.path.join("data", "credit_data_raw.csv")
    print(f"\n[+] Loading raw dataset from '{raw_path}'...")
    df_raw = load_data(raw_path)
    print(f"[OK] Raw data loaded ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns).")
    
    # Separate features and target
    X = df_raw.drop(columns=['credit_risk'])
    y = df_raw['credit_risk']
    
    # 1. Train-Test Split (80% train, 20% test, stratified on target)
    print("\n[+] Splitting data into Train (80%) and Test (20%) sets (Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[OK] Train set: {X_train_raw.shape[0]} samples | Test set: {X_test_raw.shape[0]} samples.")
    
    # 2. Impute missing values using Median Strategy (fitted ONLY on train set)
    print("\n[+] Imputing missing values with Median Strategy...")
    imputer = SimpleImputer(strategy='median')
    
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )
    
    # 3. Domain Feature Engineering
    print("[+] Engineering credit domain features (Util, DTI, Ratios)...")
    X_train_eng = engineer_features(X_train_imp)
    X_test_eng = engineer_features(X_test_imp)
    
    feature_names = list(X_train_eng.columns)
    print(f"[OK] Total features after engineering: {len(feature_names)}")
    print(f"     Features: {feature_names}")
    
    # 4. Feature Scaling (StandardScaler fitted on train set)
    print("[+] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_eng),
        columns=feature_names,
        index=X_train_eng.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_eng),
        columns=feature_names,
        index=X_test_eng.index
    )
    
    # 5. Save Processed Datasets
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\n[+] Saving cleaned train/test data to 'data/'...")
    X_train_scaled.to_csv(os.path.join("data", "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join("data", "X_test.csv"), index=False)
    y_train.to_csv(os.path.join("data", "y_train.csv"), index=False)
    y_test.to_csv(os.path.join("data", "y_test.csv"), index=False)
    
    # Save preprocessor pipeline object for inference in Web App
    preprocessor_artifact = {
        'imputer': imputer,
        'scaler': scaler,
        'feature_names': feature_names,
        'raw_feature_names': list(X.columns)
    }
    joblib.dump(preprocessor_artifact, os.path.join("models", "preprocessor.joblib"))
    print("[OK] Saved preprocessor pipeline to 'models/preprocessor.joblib'.")
    
    print("\n" + "=" * 65)
    print("  Preprocessing Completed Successfully!")
    print("=" * 65)

if __name__ == "__main__":
    run_preprocessing_pipeline()
