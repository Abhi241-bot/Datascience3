"""Difference-in-differences (DiD).

For a before/after, treatment/control panel where the two groups differ in
levels but would have moved in parallel absent treatment. DiD removes both the
fixed group gap and the common time shock, leaving the causal effect:

    outcome ~ group + period + (group x period)

The coefficient on the ``group x period`` interaction is the DiD estimate. We
fit it with OLS and read the CI / p-value straight off the regression, which is
the standard, interview-defensible way to get inference for DiD.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from src.config import CONFIG


@dataclass
class DiDResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    # the four group/period cell means, for a transparent 2x2 table
    pre_control: float
    post_control: float
    pre_treat: float
    post_treat: float
    naive_post_diff: float          # post-period treat-minus-control (the biased number)


def estimate_did(
    df: pd.DataFrame,
    outcome_col: str = "outcome",
    group_col: str = "group",       # 1 = treatment group, 0 = control group
    period_col: str = "period",     # 1 = post, 0 = pre
    alpha: float = CONFIG.alpha,
) -> DiDResult:
    """Estimate the DiD effect via an OLS interaction model."""
    data = df.rename(columns={outcome_col: "y", group_col: "g", period_col: "t"})
    model = smf.ols("y ~ g * t", data=data).fit()

    coef_name = "g:t"
    estimate = float(model.params[coef_name])
    ci = model.conf_int(alpha=alpha).loc[coef_name]
    ci_low, ci_high = float(ci[0]), float(ci[1])
    p = float(model.pvalues[coef_name])

    cell = data.groupby(["g", "t"])["y"].mean()
    pre_c = float(cell.get((0, 0), float("nan")))
    post_c = float(cell.get((0, 1), float("nan")))
    pre_t = float(cell.get((1, 0), float("nan")))
    post_t = float(cell.get((1, 1), float("nan")))

    return DiDResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p,
        significant=bool(p < alpha),
        pre_control=pre_c,
        post_control=post_c,
        pre_treat=pre_t,
        post_treat=post_t,
        naive_post_diff=post_t - post_c,
    )
