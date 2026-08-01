# wids_fairness_project

Audits how different missing-data imputation strategies affect model
performance (AUC) and fairness (False Negative Rate disparity across gender
and age subgroups) when predicting `diabetes_mellitus` on the WiDS 2020 ICU
dataset (`behordeun/wids2020` on Kaggle).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Downloading the dataset requires Kaggle API credentials. Place your
`kaggle.json` in `~/.kaggle/kaggle.json`, or set the `KAGGLE_USERNAME` and
`KAGGLE_KEY` environment variables. See the
[Kaggle API docs](https://www.kaggle.com/docs/api) for details.

## Running

```bash
python main.py
```

On first run, the dataset is downloaded via `kagglehub` and a copy is saved
to `data/raw/` so subsequent runs load it locally instead of re-downloading.
Alternatively, place a CSV directly in `data/raw/` before running.

The script runs six experiments (no imputation, global median, subgroup-
stratified median, KNN, MICE, and MissForest), printing a comparison table of
global AUC/FNR and per-subgroup FNR disparity for each. The final table is
also saved to `results/fairness_comparison.csv`.
