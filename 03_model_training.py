import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

def load_processed_data():
    """Loads preprocessed training and testing data."""
    X_train = pd.read_csv(os.path.join("data", "X_train.csv"))
    X_test = pd.read_csv(os.path.join("data", "X_test.csv"))
    y_train = pd.read_csv(os.path.join("data", "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join("data", "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models():
    print("=" * 65)
    print("  Credit Scoring Model - Model Training & Imbalance Handling")
    print("=" * 65)
    
    print("\n[+] Loading preprocessed datasets...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"[OK] Training features: {X_train.shape[1]} | Samples: {X_train.shape[0]}")
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = num_neg / (num_pos + 1e-5)
    print(f"[+] Class Balance: {num_neg} Good (0), {num_pos} Bad (1) [Imbalance Ratio: ~{pos_weight:.2f}:1]")
    
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            C=0.5,
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        ),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
    }
    
    os.makedirs("models", exist_ok=True)
    
    for name, model in models.items():
        print(f"\n[+] Fitting {name}...")
        model.fit(X_train, y_train)
        
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        model_path = os.path.join("models", model_filename)
        joblib.dump(model, model_path)
        
        print(f"[OK] {name} trained & saved to '{model_path}'.")
        
    print("\n" + "=" * 65)
    print("  Model Training Phase Completed!")
    print("=" * 65)

if __name__ == "__main__":
    train_and_evaluate_models()
