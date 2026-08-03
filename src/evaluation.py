import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

ELDERLY_AGE_THRESHOLD = 65

# ====================================================================
# PURPOSE: Calculates the False Negative Rate (FNR) for a set of
#          predictions.
# ----------------------------------------------------------------------
# PARAMS:  y_true - ground-truth binary labels
#          y_pred - predicted binary labels
#
# RETURNS: float - FNR = FN / (FN + TP)
#
# NOTES:   Returns NaN (instead of 0.0) when FNR is undefined, i.e. no
#          ground-truth positives or an empty/degenerate subgroup, so
#          an unmeasurable subgroup is never mistaken for a perfectly
#          fair one.
# ====================================================================
def calculate_fnr(y_true, y_pred) -> float:
    """
    Calculates False Negative Rate (FNR = FN / (FN + TP)).
    Returns NaN when FNR is undefined (no ground-truth positives, or an
    empty/degenerate subgroup) rather than 0.0, so an unmeasurable subgroup
    is never mistaken for a perfectly fair one.
    """
    if len(y_true) == 0:
        return np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan

# ====================================================================
# PURPOSE: Evaluates global model performance and audits fairness
#          (FNR disparity) across gender and age subgroups.
# ----------------------------------------------------------------------
# PARAMS:  df_test_raw - raw (pre-encoding) test features, used to read
#                         the 'gender' and 'age' sensitive attributes
#          y_true       - ground-truth test labels
#          y_probs      - predicted positive-class probabilities
#          y_preds      - thresholded binary predictions
#
# RETURNS: dict with keys 'Global AUC', 'Global FNR', 'Subgroup FNR'
#          (per-subgroup FNR values), and, when the relevant column is
#          present, 'Delta_FNR_Gender' / 'Delta_FNR_Age' (absolute FNR
#          gap between subgroups)
#
# NOTES:   Gender is split into Female/Male via value matching on
#          {0,'F','female'} / {1,'M','male'}. Age is split into
#          elderly (>= ELDERLY_AGE_THRESHOLD) vs young cohorts. Either
#          audit is skipped if its column is absent or entirely NaN.
# ====================================================================
def evaluate_performance_and_fairness(df_test_raw: pd.DataFrame, y_true: pd.Series, y_probs: np.ndarray, y_preds: np.ndarray):
    """
    Evaluates global AUC, FNR, and demographic disparity across gender and age cohorts.
    """
    # 1. Global Metrics
    global_auc = roc_auc_score(y_true, y_probs)
    global_fnr = calculate_fnr(y_true, y_preds)
    
    # Reconstruct evaluation table with sensitive demographic attributes
    eval_df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_preds,
        'gender': df_test_raw['gender'].values if 'gender' in df_test_raw.columns else None,
        'age': df_test_raw['age'].values if 'age' in df_test_raw.columns else None
    }, index=y_true.index)
    
    results = {
        'Global AUC': global_auc,
        'Global FNR': global_fnr,
        'Subgroup FNR': {}
    }
    
    # 2. Gender Audit
    if 'gender' in eval_df.columns and eval_df['gender'].notna().any():
        # Assuming encoding: 0 = Female, 1 = Male (or string 'F'/'M')
        female_mask = eval_df['gender'].isin([0, 'F', 'female'])
        male_mask = eval_df['gender'].isin([1, 'M', 'male'])
        
        fnr_female = calculate_fnr(eval_df.loc[female_mask, 'y_true'], eval_df.loc[female_mask, 'y_pred'])
        fnr_male = calculate_fnr(eval_df.loc[male_mask, 'y_true'], eval_df.loc[male_mask, 'y_pred'])
        
        results['Subgroup FNR']['Female'] = fnr_female
        results['Subgroup FNR']['Male'] = fnr_male
        results['Delta_FNR_Gender'] = abs(fnr_female - fnr_male)

    # 3. Age Audit (Elderly >= threshold vs Young < threshold)
    if 'age' in eval_df.columns and eval_df['age'].notna().any():
        elderly_mask = eval_df['age'] >= ELDERLY_AGE_THRESHOLD
        young_mask = eval_df['age'] < ELDERLY_AGE_THRESHOLD

        fnr_elderly = calculate_fnr(eval_df.loc[elderly_mask, 'y_true'], eval_df.loc[elderly_mask, 'y_pred'])
        fnr_young = calculate_fnr(eval_df.loc[young_mask, 'y_true'], eval_df.loc[young_mask, 'y_pred'])

        results['Subgroup FNR'][f'Elderly (>={ELDERLY_AGE_THRESHOLD})'] = fnr_elderly
        results['Subgroup FNR'][f'Young (<{ELDERLY_AGE_THRESHOLD})'] = fnr_young
        results['Delta_FNR_Age'] = abs(fnr_elderly - fnr_young)

    return results