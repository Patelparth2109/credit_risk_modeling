"""Loan unit economics and value-based approval rules.

Assumptions (documented in notebooks/04_business_impact and the README):

* Performing loan: cash in = installment x term (full-term repayment, prepayment
  ignored); cost = principal + funding & servicing at COST_OF_FUNDS per year on
  the average outstanding balance (~ principal / 2).
* Defaulted loan: default assumed at mid-term (month term // 2). EAD = the exact
  remaining amortized balance at that month. Loss = LGD x EAD, net of ALL
  post-default recoveries — recoveries are counted exactly once, nowhere else.
* Profit is the investor/lender view; platform fee revenue is not modeled.
"""

from __future__ import annotations

import numpy as np

from .config import COST_OF_FUNDS, LGD


def amortized_balance(P, n, r, t):
    """Remaining balance of a fully amortizing loan after t of n monthly payments.

    P: principal, n: term in months, r: monthly interest rate, t: months elapsed.
    """
    growth = (1 + r) ** n
    return P * (growth - (1 + r) ** t) / (growth - 1)


def unit_economics(X, lgd: float = LGD, cof: float = COST_OF_FUNDS) -> dict:
    """Per-loan profit if it performs vs if it defaults (vectorized over a DataFrame).

    X must contain the application-time columns loan_amnt, term, int_rate, installment.
    Returns a dict of numpy arrays: P, n, t_d (default month), ead,
    profit_good, profit_bad.
    """
    P = X['loan_amnt'].to_numpy(float)
    n = X['term'].to_numpy(int)
    r = X['int_rate'].to_numpy(float) / 100 / 12
    a = X['installment'].to_numpy(float)
    t_d = n // 2  # default assumed at mid-term
    ead = amortized_balance(P, n, r, t_d)
    funding_good = P * 0.5 * (n / 12) * cof
    funding_bad = ((P + ead) / 2) * (t_d / 12) * cof
    return {
        'P': P, 'n': n, 't_d': t_d, 'ead': ead,
        'profit_good': a * n - P - funding_good,
        'profit_bad': a * t_d + (1 - lgd) * ead - P - funding_bad,
    }


def expected_value(pd_arr, ue: dict) -> np.ndarray:
    """Expected profit per loan under calibrated PDs: (1-PD) x profit_good + PD x profit_bad."""
    return (1 - np.asarray(pd_arr)) * ue['profit_good'] + np.asarray(pd_arr) * ue['profit_bad']


def ev_rule_approved(pd_arr, ue: dict) -> np.ndarray:
    """The value-based approval policy: approve a loan iff its expected profit is positive.

    Unlike a fixed PD cutoff, this keeps risk that is paid for by the loan's own
    pricing and declines only thin-margin loans (see notebooks/04_business_impact).
    """
    return expected_value(pd_arr, ue) > 0
