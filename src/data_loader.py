import os
import shutil
import pandas as pd
import kagglehub

KAGGLE_DATASET_SLUG = "behordeun/wids2020"

# ====================================================================
# PURPOSE: Loads the WiDS ICU dataset, preferring a local CSV over a
#          fresh download.
# ----------------------------------------------------------------------
# PARAMS:  local_raw_dir - path to the folder checked first for an
#                           existing dataset CSV (default "data/raw")
#
# RETURNS: pd.DataFrame - the loaded dataset
#
# NOTES:   If no local CSV is found, downloads the dataset via Kagglehub
#          and copies it into local_raw_dir so later runs skip the
#          download. Always picks the largest CSV in a directory to
#          avoid selecting dictionaries/templates instead of the main
#          dataset file.
# ====================================================================
def load_wids_dataset(local_raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads the WiDS ICU dataset from local 'data/raw' folder or via Kagglehub.
    Ensures that the large main dataset CSV is selected (not dictionaries or small templates).
    A dataset downloaded via Kagglehub is copied into local_raw_dir so later runs
    don't need to re-download it.
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
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)

        all_csvs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    all_csvs.append(os.path.join(root, file))

        if all_csvs:
            # Pick the largest CSV file from downloaded folder
            downloaded_csv_path = max(all_csvs, key=os.path.getsize)

            # Persist a copy into local_raw_dir so subsequent runs skip the download
            os.makedirs(local_raw_dir, exist_ok=True)
            csv_file_path = os.path.join(local_raw_dir, os.path.basename(downloaded_csv_path))
            shutil.copy2(downloaded_csv_path, csv_file_path)
            print(f"Saved dataset copy to: {csv_file_path}")

    if not csv_file_path or not os.path.exists(csv_file_path):
        raise FileNotFoundError("Could not locate any valid CSV dataset file.")

    print(f"Loading dataset: {os.path.basename(csv_file_path)}...")
    df = pd.read_csv(csv_file_path)
    print(f"Dataset successfully loaded! Shape: {df.shape}")
    return df

# ====================================================================
# PURPOSE: Splits a DataFrame into features (X) and target (y), and
#          drops identifier columns that shouldn't be used for
#          prediction.
# ----------------------------------------------------------------------
# PARAMS:  df         - full dataset including the target column
#          target_col - name of the target column (default
#                        "diabetes_mellitus")
#
# RETURNS: (X, y) - X is the feature DataFrame (identifier columns
#           removed), y is the target Series cast to int
#
# NOTES:   Rows where the target itself is missing are dropped before
#          the split.
# ====================================================================
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

    return X, y