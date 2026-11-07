# Credit Risk Modeling: LendingClub (now Happen Bank)

An end-to-end probability-of-default (PD) model built on LendingClub's accepted-loans data (2007–2018Q4), developed in Databricks with PySpark, scikit-learn, LightGBM and MLflow.

> LendingClub rebranded as **Happen Bank** in 2026. The dataset and the notebooks use the LendingClub name because the loans were issued under that brand.

The pipeline goes from a raw 2.26M-row file to a validated model. It's built to avoid leakage: only application-time features are used, the split is out-of-time (train 2012–2015, test 2016–2018), and every model run is tracked in MLflow.

## Model performance

All metrics are **out-of-time test performance** on the 2016–2018 vintages.

| Model variant | Test AUC | Test log-loss |
| --- | --- | --- |
| Logistic regression (52 features) | 0.707 | 0.603 |
| **LightGBM fixed challenger (52 features), champion** | **0.714** | **0.592** |
| LightGBM tuned (random-search winner) | 0.713 | 0.602 |
| LightGBM without LC grading (`grade`, `sub_grade`) | 0.713 | 0.605 |
| LightGBM without grading + interest rate | 0.710 | 0.603 |
| LightGBM fully independent (no LC grading/pricing) | 0.702 | 0.601 |

**Modeling findings**

- **The signal is concentrated and nearly linear.** Most of it comes from LendingClub's own grade (`sub_grade` 34% and `grade` 19% of split gain). That's why trees add little over logistic regression (0.714 vs 0.707 AUC).
- **Borrower attributes alone keep most of the ranking power.** Without any LC grading or pricing, AUC is 0.702 of 0.714. `term`, FICO and DTI carry it.
- **Re-expressing columns doesn't add information.** Engineered affordability ratios don't move LightGBM (0.714 → 0.714), because trees already build ratios from the raw columns.
- **The plateau is real, not a tuning artifact.** Tuning halves the overfit gap (train−test AUC 0.021 vs 0.044) but doesn't improve out-of-time ranking.

## Pipeline

| Notebook | What it does | Output |
| --- | --- | --- |
| [`01_data_profiling`](notebooks/01_data_profiling.ipynb) | Profiles the raw file (2,260,701 loans × 151 columns). Keeps the 38 application-time columns and excludes post-origination fields to prevent leakage. Checks bureau-column coverage by vintage. | `workspace.default.lending_club` |
| [`02_data_cleaning`](notebooks/02_data_cleaning.ipynb) | Type coercion, documented row drops (2.37%), ordinal encodings, missing-indicator flags plus median fills, and the target (1 = Charged Off/Default, 0 = Fully Paid; unresolved loans excluded). | `workspace.default.lending_club_cleaned`: 1,296,898 loans, ~20% default rate, zero nulls |
| [`03_feature_engineering_modeling`](notebooks/03_feature_engineering_modeling.ipynb) | Fixed-mapping encodings, temporal split, LR baseline vs LightGBM challenger, feature-ablation experiments, temporally validated hyperparameter search, final model. All runs are logged to MLflow. | Champion model in MLflow |

## Reusable modules (`src/`)

The notebook logic is factored into importable, pure-pandas modules. They don't mutate their inputs, so each step can be tested and reused outside the notebooks.

| Module | Contents |
| --- | --- |
| [`config.py`](src/config.py) | Single source of truth for table names, the temporal cutoff, feature lists, and LR/LightGBM hyperparameters (fixed and tuned) |
| [`cleaning.py`](src/cleaning.py) | Numeric coercion, unsalvageable-row mask, `emp_length` ordinal encoding, history and bureau missing flags with median fills, target construction |
| [`features.py`](src/features.py) | Ordinal encodings, credit-history length, rare-category collapsing, out-of-time split, train-only one-hot model matrices |
| [`evaluation.py`](src/evaluation.py) | Ranking metrics (ROC-AUC, Gini, PR-AUC, KS, Brier, log-loss) and decile/lift table |

```python
from src.features import temporal_split, build_model_matrices
from src.evaluation import rank_metrics

train_df, test_df = temporal_split(df)
X_train, X_test, y_train, y_test = build_model_matrices(train_df, test_df)
```

## Methodology highlights

- **No outcome leakage.** Post-origination columns (`total_pymnt`, `recoveries`, `last_fico_*`, …) are dropped at the start, and all encodings are fixed mappings rather than learned from the data.
- **Out-of-time evaluation.** The model trains on 2012–2015 and is tested on 2016–2018, which mirrors scoring future applicants in production.
- **Untouched test set during tuning.** The search fits on 2012–2014 and early-stops on a 2015 validation tail carved from the training window.
- **Controlled ablations.** Feature families are removed one at a time and retrained with identical settings, to find where the signal actually lives.
- **Missingness as signal.** A null in a history column (e.g. `mths_since_last_delinq`) means no such history exists, so it becomes an explicit flag.
- **Experiment tracking.** Every variant is an MLflow run, so each reported number traces back to a logged artifact.

## Data

The raw file is `accepted_2007_to_2018Q4.csv.gz` from the [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) dataset on Kaggle. It isn't stored in this repo. To reproduce, download it and update the path in the first cell of `01_data_profiling`.

## Running it

The notebooks were developed on **Databricks serverless compute** and use Unity Catalog tables (`workspace.default.*`) as hand-offs between stages. To run them:

1. Import the notebooks into a Databricks workspace (or attach this repo as a Git folder).
2. Upload the raw CSV and set its path in `01_data_profiling`.
3. Set the MLflow experiment path in `03_feature_engineering_modeling` and `src/config.py` (currently `/Users/<your_username>/credit_risk_lending_club`).
4. Run notebooks 01 → 03 in order.

Outside Databricks, install `requirements.txt` and replace the `spark.read.table(...)` hand-offs with local files.

## Tech stack

PySpark · pandas · scikit-learn · LightGBM · MLflow · Databricks (Unity Catalog, serverless)
