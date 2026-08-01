import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder

def prepare_and_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit the encoder on the training split only, so test-set category
    # vocabulary never leaks into the encoding used for training.
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    return X_train, X_test, y_train, y_test
AGE_BUCKET_WIDTH = 10

def apply_imputation(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "none",
                      n_neighbors: int = 5, max_samples: int = 10000, random_state: int = 42):
    """
        Supported methods matching presentation:
        00. 'none'              : Raw data (XGBoost Native handling)
        01. 'global_median'     : Traditional Value Imputation
        02. 'stratified_median' : Mean/Median Stratified by Subgroup (Gender/Age)
        03. 'knn'               : K-Nearest Neighbour (K=5)
        04. 'mice'              : Multivariate Imputation by Chained Equations
        05. 'missforest'        : Iterative Random Forest / ExtraTrees Architecture

        For 'knn'/'mice'/'missforest', both fitting AND transforming the training
        set are bounded to at most max_samples rows (KNN transform cost scales as
        rows_transformed x reference_rows x features, so transforming the full
        training set against even a small reference is what actually causes
        multi-hour runs on a CPU-only machine, not just the fit step). The
        returned X_train_imp is therefore a subsample when one of these methods
        is used and callers must re-align y_train to its index. The test set is
        always transformed in full so subgroup fairness metrics keep their
        statistical power.
    """

    print(f"Running Imputation Method: [{method.upper()}]...")

    if method == "none":
        return X_train.copy(), X_test.copy()

    elif method == "global_median":
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr_imp = imputer.fit_transform(X_train)
        X_te_imp = imputer.transform(X_test)

    elif method == "stratified_median":
        X_tr_imp, X_te_imp = X_train.copy(), X_test.copy()
        group_cols = [c for c in ['gender', 'age'] if c in X_train.columns]

        if group_cols:
            num_cols = [c for c in X_train.columns if c not in group_cols]

            # Bucket age into decades so it acts as a grouping key rather than
            # a continuous column (grouping by exact age would create too
            # many near-empty strata to compute a meaningful median from).
            train_keys = X_train[group_cols].copy()
            test_keys = X_test[group_cols].copy()
            if 'age' in group_cols:
                train_keys['age'] = (train_keys['age'] // AGE_BUCKET_WIDTH) * AGE_BUCKET_WIDTH
                test_keys['age'] = (test_keys['age'] // AGE_BUCKET_WIDTH) * AGE_BUCKET_WIDTH

            for group_values, tr_group in train_keys.dropna().groupby(group_cols):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)

                tr_mask = X_train.index.isin(tr_group.index)
                te_mask = pd.Series(True, index=X_test.index)
                for col, val in zip(group_cols, group_values):
                    te_mask &= (test_keys[col] == val)

                group_medians = X_train.loc[tr_mask, num_cols].median()
                X_tr_imp.loc[tr_mask, num_cols] = X_tr_imp.loc[tr_mask, num_cols].fillna(group_medians)
                X_te_imp.loc[te_mask, num_cols] = X_te_imp.loc[te_mask, num_cols].fillna(group_medians)

        overall_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr_imp = overall_imputer.fit_transform(X_tr_imp)
        X_te_imp = overall_imputer.transform(X_te_imp)

    elif method in ["knn", "mice", "missforest"]:
        # Bound both fit AND transform of the training set to max_samples rows.
        # Transforming the full training set is the actual bottleneck for KNN
        # (cost scales with rows_transformed x reference_rows), so sampling
        # only for the fit step isn't enough to keep runtime manageable.
        sample_size = min(len(X_train), max_samples)
        if sample_size < len(X_train):
            print(f"Bounding training set to {sample_size} rows (of {len(X_train)}) for [{method.upper()}] "
                  f"to keep runtime manageable on CPU; test set is still evaluated in full.")
        X_train = X_train.sample(n=sample_size, random_state=random_state)

        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        elif method == "mice":
            imputer = IterativeImputer(max_iter=5, random_state=random_state, keep_empty_features=True)
        elif method == "missforest":
            rf_estimator = ExtraTreesRegressor(n_estimators=10, max_depth=6, random_state=random_state, n_jobs=-1)
            imputer = IterativeImputer(estimator=rf_estimator, max_iter=3, random_state=random_state, keep_empty_features=True)

        imputer.fit(X_train)
        X_tr_imp = imputer.transform(X_train)
        X_te_imp = imputer.transform(X_test)

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    X_train_imp = pd.DataFrame(X_tr_imp, columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(X_te_imp, columns=X_test.columns, index=X_test.index)
    print(f"Imputation [{method.upper()}] completed successfully.")
    return X_train_imp, X_test_imp
