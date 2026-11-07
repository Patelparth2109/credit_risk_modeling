"""Feature engineering: encodings, derived features, temporal split, model matrices.

Mirrors notebooks/03_feature_engineering_modeling. All encodings use fixed,
business-defined mappings — nothing is learned from the data, so there is no
leakage risk. One-hot encoding is fit on the train window only.
"""

from __future__ import annotations

import pandas as pd

from .config import ONEHOT_COLUMNS, TEMPORAL_CUTOFF, NON_FEATURE_COLUMNS

GRADE_MAP = {g: i for i, g in enumerate('ABCDEFG')}  # A best ... G worst


def sub_grade_to_ordinal(sub_grade: str) -> int:
    """'A1' (best) -> 0 ... 'G5' (worst) -> 34."""
    return (ord(sub_grade[0]) - 65) * 5 + int(sub_grade[1]) - 1


def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the simple categoricals with known, business-defined mappings:

    * term                  -> loan length in months (36 / 60); chosen at application
    * initial_list_status   -> 'w' whole / 'f' fractional -> 1 / 0
    * application_type      -> Individual -> 0, Joint App -> 1
    * grade / sub_grade     -> LC's risk grades, ordinal by risk
    """
    out = df.copy()
    out['term'] = out['term'].str.strip().str.replace(' months', '', regex=False).astype(int)
    out['initial_list_status'] = out['initial_list_status'].map({'w': 1, 'f': 0})
    out['application_type'] = out['application_type'].map({'Individual': 0, 'Joint App': 1})
    out['grade'] = out['grade'].map(GRADE_MAP)
    out['sub_grade'] = [sub_grade_to_ordinal(s) for s in out['sub_grade']]
    return out


def derive_credit_history_months(df: pd.DataFrame) -> pd.DataFrame:
    """Months from the borrower's earliest credit line to the loan issue date.

    Both dates are known at application. Records with a credit line dated after the
    issue date (data errors) are floored at 0.
    """
    out = df.copy()
    cr_line = pd.to_datetime(out['earliest_cr_line'], format='%b-%Y')
    issue = pd.to_datetime(out['issue_d'], format='%b-%Y')
    months = (issue.dt.year - cr_line.dt.year) * 12 + (issue.dt.month - cr_line.dt.month)
    out['credit_history_months'] = months.clip(lower=0)
    return out


def collapse_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rare levels so one-hot encoding stays clean:

    * home_ownership: ANY + NONE -> OTHER
    * purpose: 'educational' -> 'other'
    """
    out = df.copy()
    out['home_ownership'] = out['home_ownership'].replace({'ANY': 'OTHER', 'NONE': 'OTHER'})
    out['purpose'] = out['purpose'].replace({'educational': 'other'})
    return out


def drop_non_feature_columns(df: pd.DataFrame, columns=NON_FEATURE_COLUMNS) -> pd.DataFrame:
    """Drop columns that must not be model features (target source, duplicates, ids)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def temporal_split(df: pd.DataFrame, cutoff: str = TEMPORAL_CUTOFF):
    """Out-of-time split on issue_d: train < cutoff, test >= cutoff.

    No shuffling, no stratification — the split is purely temporal, mirroring
    production (learn from the past, score the future). issue_d is dropped from
    both frames after defining the split.
    """
    issue = pd.to_datetime(df['issue_d'], format='%b-%Y')
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[issue < cutoff_ts].drop(columns='issue_d').reset_index(drop=True)
    test = df[issue >= cutoff_ts].drop(columns='issue_d').reset_index(drop=True)
    return train, test


def build_model_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    onehot_columns=ONEHOT_COLUMNS,
    target: str = 'loan_default',
):
    """One-hot encode FIT ON TRAIN ONLY; reindex the test matrix onto the train columns.

    drop_first=True avoids collinearity for logistic regression. Test-only
    categories never create columns; missing categories fill with 0.
    """
    train_encoded = pd.get_dummies(train_df, columns=onehot_columns, drop_first=True, dtype=int)
    test_encoded = pd.get_dummies(test_df, columns=onehot_columns, drop_first=True, dtype=int).reindex(
        columns=train_encoded.columns, fill_value=0
    )
    y_train = train_encoded.pop(target)
    y_test = test_encoded.pop(target)
    return train_encoded, test_encoded, y_train, y_test
