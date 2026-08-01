import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

def calculate_fnr(y_true, y_pred) -> float:
    """
    Calculates False Negative Rate (FNR = FN / (FN + TP))
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fnr

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

    # 3. Age Audit (Elderly >= 65 vs Young < 65)
    if 'age' in eval_df.columns and eval_df['age'].notna().any():
        elderly_mask = eval_df['age'] >= 65
        young_mask = eval_df['age'] < 65
        
        fnr_elderly = calculate_fnr(eval_df.loc[elderly_mask, 'y_true'], eval_df.loc[elderly_mask, 'y_pred'])
        fnr_young = calculate_fnr(eval_df.loc[young_mask, 'y_true'], eval_df.loc[young_mask, 'y_pred'])
        
        results['Subgroup FNR']['Elderly (>=65)'] = fnr_elderly
        results['Subgroup FNR']['Young (<65)'] = fnr_young
        results['Delta_FNR_Age'] = abs(fnr_elderly - fnr_young)
        
    return results