import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder

def prepare_and_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    if cat_cols:
        X_encoded[cat_cols] = encoder.fit_transform(X_encoded[cat_cols])
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
def apply_imputation(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "none", n_neighbors: int = 5, max_samples: int = 20000):
    """
        Supported methods matching presentation:
        00. 'none'              : Raw data (XGBoost Native handling)
        01. 'global_median'     : Traditional Value Imputation
        02. 'stratified_median' : Mean/Median Stratified by Subgroup (Gender/Age)
        03. 'knn'               : K-Nearest Neighbour (K=5)
        04. 'mice'              : Multivariate Imputation by Chained Equations
        05. 'missforest'        : Iterative Random Forest / ExtraTrees Architecture
    """

    print(f"Running Imputation Method: [{method.upper()}]...")
    
    if method == "none":
        return X_train.copy(), X_test.copy()

    elif method == "global_median":
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_train)
        X_te_imp = imputer.transform(X_test)

    elif method == "stratified_median":
        X_tr_imp, X_te_imp = X_train.copy(), X_test.copy()
        group_col = 'gender' if 'gender' in X_train.columns else None
        
        if group_col:
            num_cols = [c for c in X_train.columns if c != group_col]
            for group in X_train[group_col].dropna().unique():
                tr_mask = (X_train[group_col] == group)
                te_mask = (X_test[group_col] == group)
                
                group_medians = X_train.loc[tr_mask, num_cols].median()
                X_tr_imp.loc[tr_mask, num_cols] = X_tr_imp.loc[tr_mask, num_cols].fillna(group_medians)
                X_te_imp.loc[te_mask, num_cols] = X_te_imp.loc[te_mask, num_cols].fillna(group_medians)
        
        overall_imputer = SimpleImputer(strategy="median")
        X_tr_imp = overall_imputer.fit_transform(X_tr_imp)
        X_te_imp = overall_imputer.transform(X_te_imp)

    elif method in ["knn", "mice", "missforest"]:
        # דגימת סאמפל מהיר מתוך ה-Train כדי למנוע תקיעה של ה-CPU והזיכרון
        sample_size = min(len(X_train), max_samples)
        train_sample = X_train.sample(n=sample_size, random_state=42)
        
        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif method == "mice":
            imputer = IterativeImputer(max_iter=5, random_state=42)
        elif method == "missforest":
            rf_estimator = ExtraTreesRegressor(n_estimators=10, max_depth=6, random_state=42, n_jobs=-1)
            imputer = IterativeImputer(estimator=rf_estimator, max_iter=3, random_state=42)

        # Fit on sampled subset, Transform full train and test
        imputer.fit(train_sample)
        X_tr_imp = imputer.transform(X_train)
        X_te_imp = imputer.transform(X_test)

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    X_train_imp = pd.DataFrame(X_tr_imp, columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(X_te_imp, columns=X_test.columns, index=X_test.index)
    print(f"Imputation [{method.upper()}] completed successfully.")
    return X_train_imp, X_test_imp
