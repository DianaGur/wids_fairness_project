import xgboost as xgb
import pandas as pd

DEFAULT_RANDOM_STATE = 42

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float = None,
                         random_state: int = DEFAULT_RANDOM_STATE) -> xgb.XGBClassifier:
    """
    Trains an XGBoost Classifier on the preprocessed training set.
    """
    print("Training XGBoost model...")

    # Handling Class Imbalance automatically if scale_pos_weight is not provided
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1  # Uses all CPU cores
    )

    model.fit(X_train, y_train)
    print("XGBoost model training complete.")
    return model

def predict_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, threshold: float = 0.5):
    """
    Returns probability predictions and binary classes based on a decision threshold.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    return y_probs, y_preds