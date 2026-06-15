"""Synthetic experiment data with KNOWN ground-truth effects.

This is the test oracle for the entire platform: every downstream method is
trusted only because it recovers the effect we inject here. Two scenarios:

* ``randomized``    -- treatment is a coin flip, independent of everything.
                       The naive A/B difference is unbiased for the true effect.
* ``observational`` -- treatment assignment depends on confounders that also
                       drive the outcome (power users self-select into treatment).
                       The naive A/B difference is BIASED; causal adjustment
                       (PSM / DoWhy) is needed to recover the truth. This is the
                       whole pitch.

A single pre-period covariate (``pre_metric``) does double duty:
  - it predicts the outcome  -> used by CUPED for variance reduction,
  - it drives both assignment and outcome in the observational scenario
    -> it is the confounder that PSM must adjust for.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroundTruth:
    """The injected truth, kept alongside the data so tests can assert recovery."""
    scenario: str
    outcome_type: str
    true_ate: float                 # the population average treatment effect actually realized
    base_effect: float              # effect for the non-high-uplift segment
    uplift_bonus: float             # extra effect added for the high-uplift segment
    high_uplift_segment: int        # which segment value (0/1) responds most
    heterogeneous: bool
    base_rate: float
    n: int
    seed: int
    effect_by_segment: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_experiment(
    n: int = 20_000,
    base_effect: float = 2.0,
    base_rate: float = 50.0,
    scenario: str = "randomized",
    outcome_type: str = "continuous",
    heterogeneous: bool = False,
    uplift_bonus: float = 4.0,
    high_uplift_segment: int = 1,
    confounding_strength: float = 1.5,
    pre_period_strength: float = 0.7,
    intended_split: float = 0.5,
    noise_sd: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Generate a cross-sectional experiment table with a known average effect.

    Parameters
    ----------
    n : total number of users.
    base_effect : true treatment effect on the metric for ordinary users
        (absolute lift for continuous outcomes; additive probability lift for binary).
    base_rate : baseline mean of the metric in control.
    scenario : ``"randomized"`` or ``"observational"``.
    outcome_type : ``"continuous"`` or ``"binary"``.
    heterogeneous : if True, the ``high_uplift_segment`` gets ``base_effect + uplift_bonus``.
    confounding_strength : how strongly confounders push assignment (observational only).
    pre_period_strength : how strongly ``pre_metric`` predicts the outcome (drives CUPED gains).
    intended_split : intended fraction in treatment (used by randomized assignment + SRM checks).
    noise_sd : outcome noise standard deviation (continuous only).
    seed : RNG seed for reproducibility.

    Returns
    -------
    (df, ground_truth)
    """
    if scenario not in {"randomized", "observational"}:
        raise ValueError(f"unknown scenario: {scenario!r}")
    if outcome_type not in {"continuous", "binary"}:
        raise ValueError(f"unknown outcome_type: {outcome_type!r}")

    rng = np.random.default_rng(seed)

    # --- Covariates ---------------------------------------------------------
    # pre_metric: a standardized pre-experiment measurement of the same metric.
    pre_metric = rng.normal(0.0, 1.0, size=n)
    # age: a second confounder so PSM has a real adjustment set, not a single knob.
    age = rng.normal(0.0, 1.0, size=n)
    # segment: an observable subgroup; the source of treatment-effect heterogeneity.
    segment = rng.integers(0, 2, size=n)

    # --- Treatment assignment ----------------------------------------------
    if scenario == "randomized":
        treatment = (rng.random(n) < intended_split).astype(int)
    else:
        # Confounded assignment: users with higher pre_metric / age self-select in.
        logit = confounding_strength * pre_metric + 0.8 * confounding_strength * age
        p_treat = 1.0 / (1.0 + np.exp(-logit))
        treatment = (rng.random(n) < p_treat).astype(int)

    # --- Per-user treatment effect (heterogeneity) -------------------------
    if heterogeneous:
        tau = base_effect + uplift_bonus * (segment == high_uplift_segment).astype(float)
    else:
        tau = np.full(n, float(base_effect))

    # --- Outcome ------------------------------------------------------------
    if outcome_type == "continuous":
        signal = (
            base_rate
            + pre_period_strength * 10.0 * pre_metric  # strong pre->outcome link for CUPED
            + 3.0 * age
            + tau * treatment
        )
        metric = signal + rng.normal(0.0, noise_sd, size=n)
        df_metric = {"metric": metric}
    else:
        base_logit = np.log(base_rate / (1.0 - base_rate))
        latent = base_logit + 0.6 * pre_metric + 0.4 * age
        p0 = 1.0 / (1.0 + np.exp(-latent))             # untreated conversion probability
        p_treated = np.clip(p0 + tau, 0.0, 1.0)        # additive probability lift
        p_final = np.where(treatment == 1, p_treated, p0)
        metric = (rng.random(n) < p_final).astype(int)
        df_metric = {"converted": metric}

    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "pre_metric": pre_metric,
            "age": age,
            "segment": segment,
            "treatment": treatment,
            **df_metric,
        }
    )

    # Record the realized population ATE (exact for this draw of segments).
    true_ate = float(np.mean(tau))
    effect_by_segment = {
        int(s): float(np.mean(tau[segment == s])) for s in np.unique(segment)
    }

    gt = GroundTruth(
        scenario=scenario,
        outcome_type=outcome_type,
        true_ate=true_ate,
        base_effect=float(base_effect),
        uplift_bonus=float(uplift_bonus if heterogeneous else 0.0),
        high_uplift_segment=int(high_uplift_segment),
        heterogeneous=heterogeneous,
        base_rate=float(base_rate),
        n=int(n),
        seed=int(seed),
        effect_by_segment=effect_by_segment,
    )
    return df, gt


def simulate_did_panel(
    n_units: int = 4_000,
    true_did_effect: float = 5.0,
    base_level: float = 50.0,
    time_trend: float = 8.0,
    group_gap: float = 12.0,
    noise_sd: float = 6.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Generate a 2x2 difference-in-differences panel (group x period).

    Parallel trends hold by construction: both groups share ``time_trend`` from
    pre to post; only the treated group in the post period gets ``true_did_effect``
    added on top. A constant ``group_gap`` makes the groups non-comparable in
    levels (so a naive post-period comparison is biased) -- DiD differences it out.

    Returns a long-format panel: one row per (unit, period).
    """
    rng = np.random.default_rng(seed)

    unit = np.arange(n_units)
    group = (rng.random(n_units) < 0.5).astype(int)          # 1 = treated group
    unit_fe = rng.normal(0.0, 4.0, size=n_units)             # unit fixed effects

    rows = []
    for period in (0, 1):  # 0 = pre, 1 = post
        post = period
        treated_now = group * post
        outcome = (
            base_level
            + unit_fe
            + group_gap * group           # time-invariant group difference (bias source)
            + time_trend * post           # common shock both groups share
            + true_did_effect * treated_now
            + rng.normal(0.0, noise_sd, size=n_units)
        )
        rows.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "group": group,        # 1 = treatment group, 0 = control group
                    "period": post,        # 0 = pre, 1 = post
                    "treated": treated_now,
                    "outcome": outcome,
                }
            )
        )

    df = pd.concat(rows, ignore_index=True)
    gt = GroundTruth(
        scenario="did_panel",
        outcome_type="continuous",
        true_ate=float(true_did_effect),
        base_effect=float(true_did_effect),
        uplift_bonus=0.0,
        high_uplift_segment=1,
        heterogeneous=False,
        base_rate=float(base_level),
        n=int(n_units),
        seed=int(seed),
        effect_by_segment={},
    )
    return df, gt


if __name__ == "__main__":  # quick smoke check
    for sc in ("randomized", "observational"):
        d, g = simulate_experiment(scenario=sc, heterogeneous=True, seed=1)
        naive = d.loc[d.treatment == 1, "metric"].mean() - d.loc[d.treatment == 0, "metric"].mean()
        print(f"{sc:14s} true_ate={g.true_ate:6.3f}  naive_diff={naive:6.3f}")
