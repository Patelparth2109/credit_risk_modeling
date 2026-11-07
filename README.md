# Credit Risk Modeling — Lending Club

An end-to-end probability-of-default (PD) model built on Lending Club's accepted-loans data (2007–2018Q4), developed in Databricks with PySpark, scikit-learn, LightGBM and MLflow.

The pipeline is built to avoid leakage. Only application-time features are used, the split is out-of-time (train 2012–2015, test 2016–2018), and every model run is tracked in MLflow. The project goes beyond AUC: calibrated PDs are turned into loan-level economics to choose the model and approval policy that make the most money.

## Business impact

Simulated on the untouched 2016–2018 out-of-time window: **517,958 accepted loans, $7.49B funded, 22.4% actual default rate.**

| Policy (LightGBM, calibrated PDs) | Approval rate | Realized net profit | Incremental vs approve-all | Expected credit loss avoided |
| --- | --- | --- | --- | --- |
| Approve-all (Lending Club's actual book) | 100% | $475M (6.3% ROI) | — | — |
| PD cutoff at 90% approval | 90% | $283–298M | **−$177M or worse** | — |
| **Expected-value rule (recommended)** | **88.8%** | **$496M (7.4% ROI)** | **+$20.6M (+4.3%)** | **$32.3M** |

**Key findings**

- **Screening on risk score alone destroys value.** Lending Club's pricing already charges for risk. The riskiest decile defaulted at ≈49% but still returned ≈18% ROI on 20–31% APRs. A PD cutoff declines the most profitable loans first and never beats approve-all at any approval rate.
- **The value is in pricing-aware screening.** The expected-value rule approves a loan iff `(1−PD) × profit_good + PD × profit_bad > 0`. It declines the ≈11% of thin-margin, near-prime loans whose risk isn't paid for, and keeps well-priced risk.
- **The result is robust.** Across LGD 60–90% and funding costs 4–8%, the EV rule stays profitable ($215M–$892M) and tightens automatically from 100% to 54% approval as the economics get worse.
- **Calibration comes first.** Isotonic calibration on out-of-fold train predictions cuts Brier from ≈0.205 to 0.159 before any money math. All policies are selected on the train window and evaluated once on test.

**Recommended strategy:** score applicants with the fixed LightGBM model plus isotonic calibration, approve with the EV rule instead of a fixed PD cutoff, and monitor realized default rate and calibration drift against the 22.4% out-of-time benchmark.

**Caveats:** only accepted loans are observed, so the effect of screening beyond Lending Club's own acceptance isn't claimed. Prepayment is ignored, which affects every policy equally. The results come from one out-of-time window.

## Model performance

All metrics are **out-of-time test performance** on the 2016–2018 vintages.

| Model variant | Test AUC | Test log-loss |
| --- | --- | --- |
| Logistic regression (52 features) | 0.707 | 0.603 |
| **LightGBM fixed challenger (52 features), recommended** | **0.714** | **0.592** |
| LightGBM tuned (random-search winner) | 0.713 | 0.602 |
| LightGBM without LC grading (`grade`, `sub_grade`) | 0.713 | 0.605 |
| LightGBM without grading + interest rate | 0.710 | 0.603 |
| LightGBM fully independent (no LC grading/pricing) | 0.702 | 0.601 |

**Modeling findings**

- Most of the signal comes from Lending Club's own grading and pricing (`sub_grade` 34% and `grade` 19% of split gain). That signal is nearly linear, so trees add little over logistic regression (0.714 vs 0.707 AUC).
- Raw borrower attributes alone keep most of the ranking power (AUC 0.702 of 0.714). `term`, FICO and DTI carry it.
- Engineered affordability ratios add nothing for LightGBM, since trees already build ratios from the raw columns. New features have to add *new information*.
- Tuning halves the overfit gap (train−test AUC 0.021 vs 0.044) but doesn't improve out-of-time ranking. The signal plateau is real, not a tuning artifact.

The fixed LightGBM challenger also has the highest financial impact on every money metric (see [Business impact](#business-impact)), so it's the recommended PD scorer.

## Pipeline

| Notebook | What it does | Output |
| --- | --- | --- |
| [`01_data_profiling`](notebooks/01_data_profiling.ipynb) | Profiles the raw file (2,260,701 loans × 151 columns). Keeps the 38 application-time columns and excludes post-origination fields to prevent leakage. Checks bureau-column coverage by vintage. | `workspace.default.lending_club` |
| [`02_data_cleaning`](notebooks/02_data_cleaning.ipynb) | Type coercion, documented row drops (2.37%), ordinal encodings, missing-indicator flags plus median fills, and the target (1 = Charged Off/Default, 0 = Fully Paid; unresolved loans excluded). | `workspace.default.lending_club_cleaned`: 1,296,898 loans, ~20% default rate, zero nulls |
| [`03_feature_engineering_modeling`](notebooks/03_feature_engineering_modeling.ipynb) | Fixed business-rule encodings, temporal split, LR baseline vs LightGBM challenger, feature-ablation experiments, temporally validated hyperparameter search, final model. All runs are logged to MLflow. | Model runs in MLflow |
| [`04_business_impact`](notebooks/04_business_impact.ipynb) | Reloads the three champion models from MLflow without retraining. Isotonic calibration on out-of-fold train scores, loan unit economics (amortized EAD, LGD, cost of funds), portfolio simulation at equal approval rates, EV-rule and risk-cap policies selected on train, sensitivity analysis. | Recommended model and lending policy |

## Reusable modules (`src/`)

The notebook logic is factored into importable, pure-pandas modules. They don't mutate their inputs, so each step can be tested and reused outside the notebooks.

| Module | Contents |
| --- | --- |
| [`config.py`](src/config.py) | Single source of truth for table names, the temporal cutoff, feature lists, LR/LightGBM hyperparameters (fixed and tuned) and economic assumptions |
| [`cleaning.py`](src/cleaning.py) | Numeric coercion, unsalvageable-row mask, `emp_length` ordinal encoding, history and bureau missing flags with median fills, target construction |
| [`features.py`](src/features.py) | Business-rule ordinal encodings, credit-history length, rare-category collapsing, out-of-time split, train-only one-hot model matrices |
| [`evaluation.py`](src/evaluation.py) | Ranking metrics (ROC-AUC, Gini, PR-AUC, KS, Brier, log-loss) and decile/lift table |
| [`economics.py`](src/economics.py) | Loan unit economics (amortized EAD, LGD, cost of funds) and the expected-value approval rule: approve iff expected profit > 0 |

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
- **Missingness as signal.** A null in a history column (e.g. `mths_since_last_delinq`) means no such history exists, so it becomes an explicit flag.
- **Experiment tracking.** Every variant is an MLflow run, so each reported number traces back to a logged artifact.

## Data

The raw file is `accepted_2007_to_2018Q4.csv.gz` from the [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) dataset on Kaggle. It isn't stored in this repo. To reproduce, download it and update the path in the first cell of `01_data_profiling`.

## Running it

The notebooks were developed on **Databricks serverless compute** and use Unity Catalog tables (`workspace.default.*`) as hand-offs between stages. To run them:

1. Import the notebooks into a Databricks workspace (or attach this repo as a Git folder).
2. Upload the raw CSV and set its path in `01_data_profiling`.
3. Set the MLflow experiment path in `03_feature_engineering_modeling` and `src/config.py` (currently `/Users/<your_username>/credit_risk_lending_club`).
4. Run notebooks 01 → 04 in order. Notebook 04 needs `%pip install lightgbm` on serverless compute to load the LightGBM artifacts.

Outside Databricks, install `requirements.txt` and replace the `spark.read.table(...)` hand-offs with local files.

## Tech stack

PySpark · pandas · scikit-learn · LightGBM · MLflow · Databricks (Unity Catalog, serverless)
