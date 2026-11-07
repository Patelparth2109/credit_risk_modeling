"""Data-cleaning transformations for the Lending Club dataset.

Each function mirrors one documented decision in notebooks/02_data_cleaning. All
functions are pure pandas: they take a DataFrame and return a new one without
mutating the input. No outcome information is used anywhere in this module.
"""

from __future__ import annotations

import pandas as pd

EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10,
}

# The source CSV stored many numbers as strings; coerce them (junk -> NaN).
NUMERIC_COLUMNS = [
    'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'acc_now_delinq',
    'tot_cur_bal', 'tot_coll_amt', 'collections_12_mths_ex_med',
]

# Nulls here mean the value was never reported -> row cannot be salvaged.
NULL_DROP_COLUMNS = ['revol_util', 'dti', 'pub_rec_bankruptcies', 'tax_liens', 'mort_acc']
# Partial-null core application fields (a handful of corrupted rows each).
PARTIAL_NULL_COLUMNS = [
    'annual_inc', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
    'initial_list_status', 'application_type', 'acc_now_delinq', 'purpose', 'addr_state',
]

# Terminal bad outcomes -> 1; repaid in full -> 0; anything else (Current, Late,
# In Grace Period) is unresolved and must be excluded from modeling.
TARGET_MAP = {'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}


def coerce_numeric_columns(df: pd.DataFrame, columns=NUMERIC_COLUMNS) -> pd.DataFrame:
    """Convert CSV string columns to real numerics; junk values become NaN."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def unsalvageable_row_mask(
    df: pd.DataFrame,
    null_drop_columns=NULL_DROP_COLUMNS,
    partial_null_columns=PARTIAL_NULL_COLUMNS,
) -> pd.Series:
    """Boolean mask of rows that cannot be kept:

    1. loan_amnt null            -> completely empty rows
    2. nulls in unreported fields (revol_util, dti, bankruptcies, tax_liens, mort_acc)
    3. nulls in core application fields (fico, open_acc, ...) -> partial rows
    4. emp_length not a valid category and not null -> corrupted values
    """
    null_drop = [c for c in null_drop_columns if c in df.columns]
    partial = [c for c in partial_null_columns if c in df.columns]
    valid_emp = set(EMP_LENGTH_MAP)
    return (
        df['loan_amnt'].isnull()
        | df[null_drop].isnull().any(axis=1)
        | df[partial].isnull().any(axis=1)
        | (~df['emp_length'].isin(valid_emp) & df['emp_length'].notna())
    )


def encode_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal 0-10 ('< 1 year' -> 0 ... '10+ years' -> 10).

    Null (never reported) -> -1, kept DISTINCT from 0 so 'unknown' is not conflated
    with 'employed less than a year'; a companion emp_length_missing flag lets
    models learn 'unknown' as its own effect.
    """
    out = df.copy()
    out['emp_length_missing'] = out['emp_length'].isnull().astype(int)
    out['emp_length'] = out['emp_length'].map(EMP_LENGTH_MAP).fillna(-1).astype(int)
    return out


def add_history_flags_and_median_fill(
    df: pd.DataFrame,
    columns=(('mths_since_last_delinq', 'has_delinq_history'),
             ('mths_since_last_record', 'has_public_record')),
) -> pd.DataFrame:
    """Null in a history column means the history does not exist.

    Capture the absence as a flag, then median-fill the numeric column so models
    that reject NaN (logistic regression) accept the matrix.
    """
    out = df.copy()
    for col, flag in columns:
        if col in out.columns:
            out[flag] = out[col].notna().astype(int)
            out[col] = out[col].fillna(out[col].median())
    return out


def add_bureau_missing_flags_and_fill(
    df: pd.DataFrame,
    columns=(('tot_cur_bal', 'tot_cur_bal_missing'),
             ('tot_coll_amt', 'tot_coll_amt_missing')),
) -> pd.DataFrame:
    """Same pattern for the late-reported bureau balance columns.

    collections_12_mths_ex_med has no nulls in the resolved vintages; any residual
    null is filled with 0 (its overwhelmingly common value) as a safeguard.
    """
    out = df.copy()
    for col, flag in columns:
        if col in out.columns:
            out[flag] = out[col].isna().astype(int)
            out[col] = out[col].fillna(out[col].median())
    if 'collections_12_mths_ex_med' in out.columns:
        out['collections_12_mths_ex_med'] = out['collections_12_mths_ex_med'].fillna(0)
    return out


def build_target(df: pd.DataFrame, status_column: str = 'loan_status') -> pd.Series:
    """Binary default flag. Unresolved outcomes map to NaN — exclude those rows."""
    return df[status_column].map(TARGET_MAP)
