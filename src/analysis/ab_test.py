"""Frequentist A/B analysis.

Continuous outcomes  -> Welch's two-sample t-test (unequal variances).
Binary outcomes      -> two-proportion z-test.

Every result reports the absolute lift, the relative lift, a confidence
interval on the absolute lift, the p-value, and a single boolean "significant?"
so the verdict engine and the UI never have to recompute or reinterpret.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

from src.config import CONFIG


@dataclass
class ABResult:
    outcome_type: str
    mean_control: float
    mean_treatment: float
    n_control: int
    n_treatment: int
    absolute_lift: float
    relative_lift: float           # absolute_lift / mean_control
    ci_low: float                  # CI on the ABSOLUTE lift
    ci_high: float
    p_value: float
    significant: bool
    alpha: float
    test_name: str

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


def analyze_ab(
    df: pd.DataFrame,
    outcome_col: str = "metric",
    treatment_col: str = "treatment",
    outcome_type: str = "continuous",
    alpha: float = CONFIG.alpha,
) -> ABResult:
    """Run the appropriate two-sample test and return a tidy result."""
    control = df.loc[df[treatment_col] == 0, outcome_col].to_numpy()
    treat = df.loc[df[treatment_col] == 1, outcome_col].to_numpy()
    n0, n1 = len(control), len(treat)
    if n0 == 0 or n1 == 0:
        raise ValueError("Both treatment arms must be non-empty.")

    m0, m1 = float(np.mean(control)), float(np.mean(treat))
    absolute_lift = m1 - m0

    if outcome_type == "continuous":
        # Welch's t-test: robust to unequal variances between arms.
        tstat, p = stats.ttest_ind(treat, control, equal_var=False)
        v0, v1 = np.var(control, ddof=1), np.var(treat, ddof=1)
        se = np.sqrt(v1 / n1 + v0 / n0)
        # Welch-Satterthwaite degrees of freedom
        dfree = (v1 / n1 + v0 / n0) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)
        )
        tcrit = stats.t.ppf(1 - alpha / 2, dfree)
        ci_low = absolute_lift - tcrit * se
        ci_high = absolute_lift + tcrit * se
        test_name = "Welch's two-sample t-test"
    elif outcome_type == "binary":
        successes = np.array([treat.sum(), control.sum()])
        nobs = np.array([n1, n0])
        zstat, p = proportions_ztest(count=successes, nobs=nobs, alternative="two-sided")
        # CI on the difference in proportions (treatment - control)
        ci_low, ci_high = confint_proportions_2indep(
            count1=int(treat.sum()), nobs1=n1,
            count2=int(control.sum()), nobs2=n0,
            method="wald", alpha=alpha,
        )
        test_name = "Two-proportion z-test"
    else:
        raise ValueError(f"unknown outcome_type: {outcome_type!r}")

    return ABResult(
        outcome_type=outcome_type,
        mean_control=m0,
        mean_treatment=m1,
        n_control=n0,
        n_treatment=n1,
        absolute_lift=float(absolute_lift),
        relative_lift=float(absolute_lift / m0) if m0 != 0 else float("nan"),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
        significant=bool(p < alpha),
        alpha=alpha,
        test_name=test_name,
    )
