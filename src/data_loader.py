import os
import pandas as pd
import kagglehub

def load_wids_dataset(local_raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads the WiDS ICU dataset from local 'data/raw' folder or via Kagglehub.
    Ensures that the large main dataset CSV is selected (not dictionaries or small templates).
    """
    csv_file_path = None
    
    # 1. Search in local data/raw first
    if os.path.exists(local_raw_dir):
        csv_files = [os.path.join(local_raw_dir, f) for f in os.listdir(local_raw_dir) if f.endswith(".csv")]
        if csv_files:
            # Pick the largest CSV file in the directory
            csv_file_path = max(csv_files, key=os.path.getsize)
            print(f"Found local dataset at: {csv_file_path}")

    # 2. Download via Kagglehub if no local file exists
    if not csv_file_path:
        print("Local file not found. Downloading main dataset via Kagglehub...")
        path = kagglehub.dataset_download("behordeun/wids2020")
        
        all_csvs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    all_csvs.append(os.path.join(root, file))
        
        if all_csvs:
            # Pick the largest CSV file from downloaded folder
            csv_file_path = max(all_csvs, key=os.path.getsize)

    if not csv_file_path or not os.path.exists(csv_file_path):
        raise FileNotFoundError("Could not locate any valid CSV dataset file.")

    print(f"Loading dataset: {os.path.basename(csv_file_path)}...")
    df = pd.read_csv(csv_file_path)
    print(f"Dataset successfully loaded! Shape: {df.shape}")
    return df

def get_target_and_features(df: pd.DataFrame, target_col: str = "diabetes_mellitus"):
    """
    Splits DataFrame into features (X) and target variable (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    # Drop rows where the target label itself is missing
    df_clean = df.dropna(subset=[target_col]).copy()
    
    y = df_clean[target_col].astype(int)
    X = df_clean.drop(columns=[target_col])
    
    # Drop identifier columns that shouldn't be used for prediction
    cols_to_drop = [col for col in ['encounter_id', 'patient_id', 'hospital_id', 'icu_id'] if col in X.columns]
    X = X.drop(columns=cols_to_drop)
    
    return X, y, df_clean