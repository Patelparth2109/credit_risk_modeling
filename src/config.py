"""Central configuration for the Lending Club credit-risk pipeline.

All paths, seeds, and model constants live here so every notebook and test references
one definition. Values mirror the executed notebooks exactly — changing them invalidates
comparability with the logged MLflow runs.
"""

# --- reproducibility ---
RANDOM_STATE = 42

# --- data locations (Databricks workspace / Unity Catalog) ---
RAW_SELECTED_TABLE = 'workspace.default.lending_club'        # 38 application-time columns
CLEANED_TABLE = 'workspace.default.lending_club_cleaned'    # cleaned + target, 1,296,898 rows
MLFLOW_EXPERIMENT = '/Users/<your_username>/credit_risk_lending_club'

# --- temporal split ---
# Train on 2012-2015 vintages, test (out-of-time) on 2016-2018. The cutoff also
# guarantees every test loan at least two years of seasoning, limiting survivorship
# bias in the most recent vintages.
TEMPORAL_CUTOFF = '2016-01-01'

# --- feature engineering ---
ONEHOT_COLUMNS = ['home_ownership', 'verification_status', 'purpose']
NON_FEATURE_COLUMNS = [
    'loan_status',       # source of the target -> keeping it would be direct leakage
    'funded_amnt',        # duplicate of loan_amnt, no added signal
    'funded_amnt_inv',    # duplicate of loan_amnt, no added signal
    'addr_state',         # 51 levels, marginal signal for a baseline model
    'earliest_cr_line',   # replaced by credit_history_months
]
CREDIT_HISTORY_CAP_MONTHS = 720  # caps junk 999-month entries from bad earliest_cr_line years

# --- model hyperparameters (exactly as used by the champion runs) ---
LR_PARAMS = {'max_iter': 1000, 'class_weight': 'balanced'}

# Fixed challenger: used for every ablation comparison and the final recommendation.
LGBM_FIXED_PARAMS = {
    'objective': 'binary',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 100,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
}

# Random-search winner (25-config search; validated on a 2015 train-tail split,
# early stopping at 223 trees). See notebooks/03_feature_engineering_modeling.
LGBM_TUNED_PARAMS = {
    'objective': 'binary',
    'n_estimators': 223,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'min_child_samples': 200,
    'colsample_bytree': 0.8,
    'subsample': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
}
