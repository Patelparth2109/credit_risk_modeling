"""Ranking-quality metrics for default-risk models.

Used identically for every model and variant so comparisons are like-for-like.
Note: models trained with class_weight='balanced' inflate raw probabilities —
treat scores as rankings (AUC / KS / deciles) and calibrate before using them
as probabilities (see notebooks/04_business_impact).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve,
)


def rank_metrics(y_true, prob) -> dict:
    """Core ranking metrics: ROC-AUC, Gini, PR-AUC, KS, Brier, log-loss."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    auc = roc_auc_score(y_true, prob)
    fpr, tpr, _ = roc_curve(y_true, prob)
    return {
        'roc_auc': auc,
        'gini': 2 * auc - 1,
        'pr_auc': average_precision_score(y_true, prob),
        'ks': float(np.max(tpr - fpr)),
        'brier': brier_score_loss(y_true, prob),
        'log_loss': log_loss(y_true, prob),
    }


def decile_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Rank loans by predicted risk, show actual default rate and lift per decile.

    A good model concentrates defaults in the top deciles (high lift in D10).
    """
    df = pd.DataFrame({'prob': np.asarray(prob), 'actual': np.asarray(y_true)})
    df['decile'] = pd.qcut(df['prob'].rank(method='first'), n_bins,
                           labels=[f'D{i}' for i in range(1, n_bins + 1)])
    table = df.groupby('decile', observed=True).agg(
        n=('actual', 'count'),
        avg_predicted=('prob', 'mean'),
        actual_rate=('actual', 'mean'),
    )
    table['lift'] = table['actual_rate'] / df['actual'].mean()
    return table
