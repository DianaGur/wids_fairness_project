import os
import numpy as np
import pandas as pd

from src.data_loader import load_wids_dataset, get_target_and_features
from src.preprocessing import prepare_and_split_data, apply_imputation
from src.models import train_xgboost_model, predict_model
from src.evaluation import evaluate_performance_and_fairness

RESULTS_DIR = "results"

# Experimental steps corresponding to the presentation
EXPERIMENTS = [
    ("00. Native XGBoost (Pre-Imputation Fairness Audit)", "none"),
    ("01. Traditional Value Imputation (Global Median)", "global_median"),
    ("02. Mean/Median Stratified by Subgroup", "stratified_median"),
    ("03. K-Nearest Neighbour (K=5)", "knn"),
    ("04. MICE (Multivariate Chained Equations)", "mice"),
    ("05. MissForest (Iterative Random Forest)", "missforest"),
]

def run_experiment(label, method_code, X_train, X_test, y_train, y_test, X_test_raw):
    print(f"\n==================================================")
    print(f" EXPERIMENT: {label}")
    print(f"==================================================")

    X_tr_imp, X_te_imp = apply_imputation(X_train, X_test, method=method_code)
    # X_tr_imp may be a row subsample of X_train for sample-bounded methods
    # (knn/mice/missforest), so realign y_train to whichever rows survived.
    y_train_aligned = y_train.loc[X_tr_imp.index]
    model = train_xgboost_model(X_tr_imp, y_train_aligned)
    y_probs, y_preds = predict_model(model, X_te_imp)
    results = evaluate_performance_and_fairness(X_test_raw, y_test, y_probs, y_preds)

    return {
        'Method': label,
        'Global AUC': round(results['Global AUC'], 4),
        'Global FNR': round(results['Global FNR'], 4),
        'Gender Delta FNR': round(results.get('Delta_FNR_Gender', np.nan), 4),
        'Age Delta FNR': round(results.get('Delta_FNR_Age', np.nan), 4),
    }

def main():
    # 1. Load and Split Data
    df = load_wids_dataset()
    X, y = get_target_and_features(df)
    X_train, X_test, y_train, y_test = prepare_and_split_data(X, y)

    X_test_raw = X.loc[X_test.index]

    all_summary_results = [
        run_experiment(label, method_code, X_train, X_test, y_train, y_test, X_test_raw)
        for label, method_code in EXPERIMENTS
    ]

    # Display final comparative audit table
    summary_df = pd.DataFrame(all_summary_results)
    print("\n\n==================================================")
    print(" FINAL FAIRNESS & PERFORMANCE COMPARISON TABLE")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "fairness_comparison.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()