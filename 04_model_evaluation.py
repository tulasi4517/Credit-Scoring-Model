import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

def calculate_ks_statistic(y_true, y_probs):
    """Calculates the Kolmogorov-Smirnov (KS) statistic and Gini coefficient."""
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_probs})
    df = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
    
    total_goods = np.sum(df['y_true'] == 0)
    total_bads = np.sum(df['y_true'] == 1)
    
    df['cum_bads'] = (df['y_true'] == 1).cumsum() / (total_bads + 1e-5)
    df['cum_goods'] = (df['y_true'] == 0).cumsum() / (total_goods + 1e-5)
    
    ks_stat = np.max(np.abs(df['cum_bads'] - df['cum_goods']))
    return ks_stat

def run_evaluation_pipeline():
    print("=" * 65)
    print("  Credit Scoring Model - Comprehensive Evaluation & Risk Metrics")
    print("=" * 65)
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Load Test Data
    X_test = pd.read_csv(os.path.join("data", "X_test.csv"))
    y_test = pd.read_csv(os.path.join("data", "y_test.csv")).values.ravel()
    
    # Find all trained model artifacts
    model_files = glob.glob(os.path.join("models", "*.joblib"))
    model_files = [f for f in model_files if 'preprocessor' not in f]
    
    if not model_files:
        raise FileNotFoundError("No trained model artifacts found in 'models/'. Run 03_model_training.py first.")
        
    metrics_list = []
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    fig_cm, axes_cm = plt.subplots(1, len(model_files), figsize=(5 * len(model_files), 4))
    if len(model_files) == 1:
        axes_cm = [axes_cm]
        
    for idx, model_path in enumerate(model_files):
        model_name = os.path.basename(model_path).replace(".joblib", "").replace("_", " ").title()
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        gini = 2 * roc_auc - 1
        ks_stat = calculate_ks_statistic(y_test, y_prob)
        
        metrics_list.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall (Sensitivity)': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
            'Gini Coefficient': round(gini, 4),
            'KS Statistic': round(ks_stat, 4)
        })
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f}, Gini = {gini:.3f})', lw=2)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], cbar=False)
        axes_cm[idx].set_title(f'{model_name} Confusion Matrix')
        axes_cm[idx].set_xlabel('Predicted Risk (0=Good, 1=Bad)')
        axes_cm[idx].set_ylabel('Actual Risk (0=Good, 1=Bad)')
        
    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    roc_plot_path = os.path.join("reports", "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    
    # Finalize Confusion Matrix plot
    fig_cm.tight_layout()
    cm_plot_path = os.path.join("reports", "confusion_matrices.png")
    fig_cm.savefig(cm_plot_path, dpi=300)
    plt.close(fig_cm)
    
    # Create Summary DataFrame
    df_metrics = pd.DataFrame(metrics_list)
    summary_csv_path = os.path.join("reports", "metrics_summary.csv")
    df_metrics.to_csv(summary_csv_path, index=False)
    
    print("\n" + "=" * 65)
    print("  Model Evaluation Performance Summary")
    print("=" * 65)
    print(df_metrics.to_string(index=False))
    print("\n[+] Visualizations saved to 'reports/':")
    print(f"    - ROC Curves: {roc_plot_path}")
    print(f"    - Confusion Matrices: {cm_plot_path}")
    print(f"    - Summary CSV: {summary_csv_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation_pipeline()
