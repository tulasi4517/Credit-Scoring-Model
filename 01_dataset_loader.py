import os
import numpy as np
import pandas as pd

def generate_synthetic_credit_data(num_samples=5000, random_seed=42):
    """
    Generates a realistic synthetic credit scoring dataset with target 'credit_risk'.
    
    Parameters:
        num_samples (int): Number of rows to generate.
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Raw credit dataset with missing values and realistic features.
    """
    np.random.seed(random_seed)
    
    # 1. Demographics & Employment
    age = np.random.randint(18, 70, size=num_samples)
    
    # Employment years bounded by (age - 18)
    max_emp = np.maximum(0, age - 18)
    employment_years = np.round(np.random.uniform(0, 1, size=num_samples) * max_emp, 1)
    
    # 2. Financial Metrics
    # Income: Log-normal distribution centered around ~$55,000
    log_income = np.random.normal(loc=10.9, scale=0.6, size=num_samples)
    annual_income = np.round(np.exp(log_income), 2)
    annual_income = np.clip(annual_income, 15000, 300000)
    
    # Credit Limit: Dependent on income & age factor
    base_limit = annual_income * np.random.uniform(0.15, 0.45, size=num_samples)
    credit_limit = np.round(np.clip(base_limit, 1000, 50000), -2)  # Round to nearest 100
    
    # Current Balance (Revolving balance): Function of credit limit and utilization
    utilization_rate = np.random.beta(a=2, b=5, size=num_samples) # Skewed towards lower utilization
    current_balance = np.round(credit_limit * utilization_rate, 2)
    
    # Total Debt: Revolving balance + other installment loans (auto, student, mortgage)
    installment_debt = annual_income * np.random.uniform(0.0, 2.5, size=num_samples)
    total_debt = np.round(current_balance + installment_debt, 2)
    
    # 3. Behavioral Features
    # Late payments in last 12 months: Poisson distributed
    num_late_payments_12m = np.random.poisson(lam=0.6, size=num_samples)
    num_late_payments_12m = np.clip(num_late_payments_12m, 0, 10)
    
    # 4. Target Generation (Credit Risk: 0 = Low Risk / Good, 1 = High Risk / Bad)
    # Risk Score calculation (Logistic regression-style log-odds)
    dti_ratio = total_debt / (annual_income + 1e-5)
    revol_util = current_balance / (credit_limit + 1e-5)
    
    # Log-odds of default / high risk
    log_odds = (
        -2.2
        + 1.8 * revol_util
        + 0.8 * num_late_payments_12m
        + 0.6 * np.log1p(dti_ratio)
        - 0.03 * employment_years
        - 0.015 * (age - 18)
        - 0.3 * np.log(annual_income / 10000)
        + np.random.normal(0, 0.5, size=num_samples) # Stochastic noise
    )
    
    prob_default = 1 / (1 + np.exp(-log_odds))
    credit_risk = (np.random.uniform(0, 1, size=num_samples) < prob_default).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'total_debt': total_debt,
        'credit_limit': credit_limit,
        'current_balance': current_balance,
        'num_late_payments_12m': num_late_payments_12m,
        'employment_years': employment_years,
        'credit_risk': credit_risk
    })
    
    # 5. Inject ~5% random missing values (MCAR) in feature columns to test preprocessing pipeline
    feature_cols = [c for c in df.columns if c != 'credit_risk']
    for col in feature_cols:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan
        
    return df

def main():
    print("=" * 65)
    print("  Credit Scoring Model - Dataset Loader & Synthetic Generator")
    print("=" * 65)
    
    raw_dir = "data"
    raw_path = os.path.join(raw_dir, "credit_data_raw.csv")
    
    # Ensure data directory exists
    os.makedirs(raw_dir, exist_ok=True)
    
    print("\n[+] Generating synthetic credit dataset (5,000 samples)...")
    df = generate_synthetic_credit_data(num_samples=5000, random_seed=42)
    
    print(f"[+] Saving raw dataset to '{raw_path}'...")
    df.to_csv(raw_path, index=False)
    print(f"[OK] File successfully saved ({df.shape[0]} rows, {df.shape[1]} columns).")
    
    print("\n" + "=" * 65)
    print("  Dataset Overview & Summary Statistics")
    print("=" * 65)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Missing Values Count per Feature (~5% Expected) ---")
    missing_summary = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage (%)': np.round((df.isnull().sum() / len(df)) * 100, 2)
    })
    print(missing_summary)
    
    print("\n--- Summary Statistics (df.describe()) ---")
    desc = df.describe().T
    print(desc.to_string())
    
    print("\n--- Class Distribution (Target: credit_risk) ---")
    counts = df['credit_risk'].value_counts()
    percentages = df['credit_risk'].value_counts(normalize=True) * 100
    class_dist = pd.DataFrame({
        'Label': ['0 (Low Risk / Good Credit)', '1 (High Risk / Bad Credit)'],
        'Count': counts.values,
        'Percentage (%)': percentages.values
    })
    print(class_dist.to_string(index=False))
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
