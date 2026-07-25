import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_explainability_pipeline():
    print("=" * 65)
    print("  Credit Scoring Model - Feature Importance & Risk Explainability")
    print("=" * 65)
    
    os.makedirs("reports", exist_ok=True)
    
    X_train = pd.read_csv(os.path.join("data", "X_train.csv"))
    feature_names = list(X_train.columns)
    
    # Load pre-trained models
    lr_model = joblib.load(os.path.join("models", "logistic_regression.joblib"))
    rf_model = joblib.load(os.path.join("models", "random_forest.joblib"))
    
    # 1. Logistic Regression Coefficients (Signed Impact)
    lr_coefs = lr_model.coef_[0]
    df_lr = pd.DataFrame({
        'Feature': feature_names,
        'Importance': lr_coefs,
        'Abs_Importance': np.abs(lr_coefs)
    }).sort_values(by='Abs_Importance', ascending=False)
    
    # 2. Random Forest Gini Importance
    rf_importances = rf_model.feature_importances_
    df_rf = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Logistic Regression Feature Coefficients (Log-Odds Impact) ---")
    print(df_lr[['Feature', 'Importance']].to_string(index=False))
    
    print("\n--- Random Forest Top Feature Importances ---")
    print(df_rf.to_string(index=False))
    
    # Plot Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # LR Plot
    colors_lr = ['crimson' if val > 0 else 'teal' for val in df_lr['Importance']]
    sns.barplot(x='Importance', y='Feature', data=df_lr, palette=colors_lr, ax=axes[0])
    axes[0].set_title('Logistic Regression Feature Coefficients\n(Red = Increases Risk, Teal = Decreases Risk)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value (Log-Odds Impact)')
    
    # RF Plot
    sns.barplot(x='Importance', y='Feature', data=df_rf, palette='crest', ax=axes[1])
    axes[1].set_title('Random Forest Feature Importance (Gini)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Feature Importance')
    
    plt.tight_layout()
    plot_path = os.path.join("reports", "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n[OK] Feature importance plot saved to '{plot_path}'.")
    print("=" * 65)

if __name__ == "__main__":
    run_explainability_pipeline()
