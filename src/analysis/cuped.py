"""CUPED variance reduction.

CUPED (Controlled-experiment Using Pre-Experiment Data) removes the part of the
outcome that was already predictable from a pre-experiment covariate X:

    Y_cuped = Y - theta * (X - mean(X)),   theta = cov(Y, X) / var(X)

theta is estimated pooled across both arms. Because X is measured *before*
treatment, subtracting it cannot bias the treatment effect -- it only strips
out pre-existing variance. The estimate stays the same in expectation while its
standard error (and thus the CI) shrinks. The variance reduction equals rho^2,
the squared correlation between Y and X.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.analysis.ab_test import ABResult, analyze_ab


@dataclass
class CUPEDResult:
    theta: float
    variance_reduction: float       # fraction, e.g. 0.42 == 42% lower variance
    rho: float                      # correlation between outcome and covariate
    naive: ABResult                 # A/B on the raw outcome
    adjusted: ABResult              # A/B on the CUPED-adjusted outcome
    ci_width_naive: float
    ci_width_adjusted: float


def apply_cuped(
    df: pd.DataFrame,
    outcome_col: str = "metric",
    covariate_col: str = "pre_metric",
) -> tuple[pd.Series, float, float]:
    """Return (adjusted_outcome, theta, variance_reduction_fraction)."""
    y = df[outcome_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)

    var_x = np.var(x, ddof=1)
    if var_x == 0:
        # Degenerate covariate: nothing to subtract.
        return pd.Series(y, index=df.index), 0.0, 0.0

    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    y_cuped = y - theta * (x - x.mean())

    variance_reduction = 1.0 - np.var(y_cuped, ddof=1) / np.var(y, ddof=1)
    return pd.Series(y_cuped, index=df.index), float(theta), float(variance_reduction)


def analyze_with_cuped(
    df: pd.DataFrame,
    outcome_col: str = "metric",
    covariate_col: str = "pre_metric",
    treatment_col: str = "treatment",
    alpha: float = CONFIG.alpha,
) -> CUPEDResult:
    """Run the naive A/B and the CUPED-adjusted A/B side by side."""
    naive = analyze_ab(
        df, outcome_col=outcome_col, treatment_col=treatment_col,
        outcome_type="continuous", alpha=alpha,
    )

    y_cuped, theta, var_red = apply_cuped(df, outcome_col, covariate_col)
    adj_df = df.copy()
    adj_df["_cuped"] = y_cuped
    adjusted = analyze_ab(
        adj_df, outcome_col="_cuped", treatment_col=treatment_col,
        outcome_type="continuous", alpha=alpha,
    )

    rho = float(np.corrcoef(df[outcome_col], df[covariate_col])[0, 1])

    return CUPEDResult(
        theta=theta,
        variance_reduction=var_red,
        rho=rho,
        naive=naive,
        adjusted=adjusted,
        ci_width_naive=naive.ci_high - naive.ci_low,
        ci_width_adjusted=adjusted.ci_high - adjusted.ci_low,
    )
