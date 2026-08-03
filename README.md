# wids_fairness_project

## Abstract

Electronic Health Record (EHR) datasets from Intensive Care Units (ICUs) suffer from severe missing data patterns resulting from selective diagnostic ordering. In acute clinical settings, misclassifying a diabetic patient as non-diabetic poses severe risks to patient safety. Furthermore, improper missing data handling can induce systematic performance disparities across protected demographic cohorts.

To systematically evaluate the impact of traditional versus complex data imputation methodologies on prediction accuracy and demographic fairness (biological gender and age) in clinical predictive pipelines, we utilized 130,157 patient encounters from the WiDS/eICU database and established a standardized predictive pipeline targeting inpatient diabetes mellitus (`diabetes_mellitus`) using an XGBoost Classifier. We audited six missing data strategies: Native XGBoost sparsity handling, Global Median Imputation, Subgroup-Stratified Median Imputation, K-Nearest Neighbors (KNN, K=5), Multivariate Imputation by Chained Equations (MICE), and Iterative Random Forest (MissForest). Predictive performance was evaluated using AUC-ROC, while clinical safety and fairness were quantified via the False Negative Rate (FNR) and subgroup disparity (∆FNR).

This framework provides an empirical foundation for deploying safe, equitable, and clinically reliable AI systems in high-acuity healthcare environments.

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
