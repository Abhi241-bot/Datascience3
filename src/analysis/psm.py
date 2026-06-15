"""Propensity score matching via DoWhy.

The headline contrast of the whole platform. On the observational scenario,
treatment was self-selected: confounders (pre_metric, age) drive both who got
treated and the outcome, so the naive treatment-minus-control difference is
biased. DoWhy adjusts for those confounders by the backdoor criterion and
propensity-score matching, recovering an estimate much closer to the truth.

We deliberately report BOTH numbers (naive and adjusted) so the UI/README can
show the bias being removed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.config import CONFIG

# DoWhy is chatty; quiet it so logs don't drown the app.
logging.getLogger("dowhy").setLevel(logging.WARNING)


@dataclass
class PSMResult:
    naive_estimate: float           # biased: raw treat - control mean difference
    adjusted_estimate: float        # PSM/backdoor estimate after adjusting for confounders
    confounders: List[str] = field(default_factory=list)
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_used: int = 0
    method: str = "backdoor.propensity_score_matching"


def estimate_psm(
    df: pd.DataFrame,
    outcome_col: str = "metric",
    treatment_col: str = "treatment",
    confounders: List[str] | None = None,
    max_n: int = 4000,
    with_ci: bool = False,
    seed: int = CONFIG.random_seed,
) -> PSMResult:
    """Estimate the treatment effect adjusting for confounders via DoWhy PSM.

    Parameters
    ----------
    confounders : columns to adjust for (the backdoor set). Defaults to
        ``["pre_metric", "age"]`` which matches the simulator's confounders.
    max_n : matching is O(n^2)-ish; subsample above this for tractability.
    with_ci : if True, ask DoWhy for a bootstrap CI (slower).
    """
    from dowhy import CausalModel

    if confounders is None:
        confounders = ["pre_metric", "age"]

    cols = [outcome_col, treatment_col, *confounders]
    data = df[cols].dropna()
    if len(data) > max_n:
        data = data.sample(max_n, random_state=seed).reset_index(drop=True)

    # Naive (biased) difference for the contrast.
    c = data.loc[data[treatment_col] == 0, outcome_col]
    t = data.loc[data[treatment_col] == 1, outcome_col]
    naive = float(t.mean() - c.mean())

    model = CausalModel(
        data=data,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=confounders,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_matching",
        target_units="ate",
        confidence_intervals=with_ci,
    )
    adjusted = float(estimate.value)

    ci_low = ci_high = float("nan")
    if with_ci:
        try:
            lo, hi = estimate.get_confidence_intervals()
            ci_low, ci_high = float(np.ravel(lo)[0]), float(np.ravel(hi)[0])
        except Exception:
            pass

    return PSMResult(
        naive_estimate=naive,
        adjusted_estimate=adjusted,
        confounders=list(confounders),
        ci_low=ci_low,
        ci_high=ci_high,
        n_used=len(data),
    )
