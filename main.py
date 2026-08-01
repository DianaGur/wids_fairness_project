from src.data_loader import load_wids_dataset, get_target_and_features
from src.preprocessing import prepare_and_split_data, apply_imputation
from src.models import train_xgboost_model, predict_model
from src.evaluation import evaluate_performance_and_fairness
import pandas as pd

def main():
    # 1. Load and Split Data
    df = load_wids_dataset()
    X, y, df_clean = get_target_and_features(df)
    X_train, X_test, y_train, y_test = prepare_and_split_data(X, y)
    
    X_test_raw = X.loc[X_test.index]

    # List of experimental steps corresponding to presentation
    experiments = [
        ("00. Native XGBoost (Pre-Imputation Fairness Audit)", "none"),
        ("01. Traditional Value Imputation (Global Median)", "global_median"),
        ("02. Mean/Median Stratified by Subgroup", "stratified_median"),
        ("03. K-Nearest Neighbour (K=5)", "knn"),
        ("04. MICE (Multivariate Chained Equations)", "mice"),
        ("05. MissForest (Iterative Random Forest)", "missforest"),
    ]

    all_summary_results = []

    for label, method_code in experiments:
        print(f"\n==================================================")
        print(f" EXPERIMENT: {label}")
        print(f"==================================================")
        
        # 1. Apply Imputation
        X_tr_imp, X_te_imp = apply_imputation(X_train, X_test, method=method_code)
        
        # 2. Train Model
        model = train_xgboost_model(X_tr_imp, y_train)
        
        # 3. Predict & Evaluate
        y_probs, y_preds = predict_model(model, X_te_imp)
        results = evaluate_performance_and_fairness(X_test_raw, y_test, y_probs, y_preds)
        
        # Collect summary row
        all_summary_results.append({
            'Method': label,
            'Global AUC': round(results['Global AUC'], 4),
            'Global FNR': round(results['Global FNR'], 4),
            'Gender ΔFNR': round(results.get('Delta_FNR_Gender', 0.0), 4),
            'Age ΔFNR': round(results.get('Delta_FNR_Age', 0.0), 4)
        })

    # Display final comparative audit table
    summary_df = pd.DataFrame(all_summary_results)
    print("\n\n==================================================")
    print(" FINAL FAIRNESS & PERFORMANCE COMPARISON TABLE")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    main()