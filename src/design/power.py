"""Power analysis & sample-size planning.

Two directions an experimenter needs before launch:
  * "How many users per arm do I need to detect an X% lift?"   -> required_sample_size
  * "Given the N I can realistically get, what lift can I see?" -> detectable_effect

Both binary (proportions) and continuous (means) outcomes are supported, using
statsmodels' standardized-effect-size machinery so the numbers match the
textbook / statsmodels reference an interviewer would check against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.power import (
    NormalIndPower,
    TTestIndPower,
)
from statsmodels.stats.proportion import proportion_effectsize

from src.config import CONFIG


@dataclass
class SampleSizeResult:
    per_arm: int
    total: int
    baseline: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    effect_size: float
    outcome_type: str


def required_sample_size(
    baseline: float,
    mde_relative: float | None = None,
    mde_absolute: float | None = None,
    outcome_type: str = "binary",
    sd: float | None = None,
    alpha: float = CONFIG.alpha,
    power: float = CONFIG.power,
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Required sample size per arm to detect a given minimum detectable effect.

    Parameters
    ----------
    baseline : control mean (binary: baseline conversion rate; continuous: mean).
    mde_relative : minimum detectable effect as a fraction of baseline (e.g. 0.05).
    mde_absolute : MDE in absolute units (alternative to ``mde_relative``).
    outcome_type : ``"binary"`` or ``"continuous"``.
    sd : outcome standard deviation (required for continuous).
    ratio : n_treatment / n_control.
    """
    if mde_relative is None and mde_absolute is None:
        mde_relative = CONFIG.mde_relative
    if mde_absolute is None:
        mde_absolute = baseline * mde_relative
    if mde_relative is None:
        mde_relative = mde_absolute / baseline if baseline else float("nan")

    if outcome_type == "binary":
        p1, p2 = baseline, baseline + mde_absolute
        if not (0 < p2 < 1):
            raise ValueError(f"baseline+MDE={p2:.4f} out of (0,1) for a proportion.")
        effect_size = abs(proportion_effectsize(p2, p1))
        analysis = NormalIndPower()
    elif outcome_type == "continuous":
        if sd is None or sd <= 0:
            raise ValueError("continuous outcome requires a positive sd.")
        effect_size = abs(mde_absolute) / sd          # Cohen's d
        analysis = TTestIndPower()
    else:
        raise ValueError(f"unknown outcome_type: {outcome_type!r}")

    n_per_arm = analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=ratio, alternative="two-sided"
    )
    per_arm = int(np.ceil(n_per_arm))
    return SampleSizeResult(
        per_arm=per_arm,
        total=int(np.ceil(per_arm * (1 + ratio))),
        baseline=baseline,
        mde_absolute=float(mde_absolute),
        mde_relative=float(mde_relative),
        alpha=alpha,
        power=power,
        effect_size=float(effect_size),
        outcome_type=outcome_type,
    )


def detectable_effect(
    baseline: float,
    n_per_arm: int,
    outcome_type: str = "binary",
    sd: float | None = None,
    alpha: float = CONFIG.alpha,
    power: float = CONFIG.power,
) -> dict:
    """Reverse problem: smallest effect detectable at given n, alpha, power."""
    if outcome_type == "binary":
        analysis = NormalIndPower()
        es = analysis.solve_power(
            nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
        )
        # invert proportion_effectsize (Cohen's h) for p2 given p1
        h = es
        phi1 = 2 * np.arcsin(np.sqrt(baseline))
        phi2 = phi1 + h
        p2 = np.sin(phi2 / 2) ** 2
        mde_absolute = p2 - baseline
    elif outcome_type == "continuous":
        if sd is None or sd <= 0:
            raise ValueError("continuous outcome requires a positive sd.")
        analysis = TTestIndPower()
        es = analysis.solve_power(
            nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
        )
        mde_absolute = es * sd
    else:
        raise ValueError(f"unknown outcome_type: {outcome_type!r}")

    return {
        "mde_absolute": float(mde_absolute),
        "mde_relative": float(mde_absolute / baseline) if baseline else float("nan"),
        "effect_size": float(es),
        "n_per_arm": int(n_per_arm),
        "alpha": alpha,
        "power": power,
    }
