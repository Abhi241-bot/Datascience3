"""Sample Ratio Mismatch (SRM) check.

A simple but high-value guardrail: if you intended a 50/50 split but the data
arrives 52/48 on a million users, the randomization or logging is broken and
EVERY downstream number is suspect. We test the observed arm counts against the
intended split with a chi-square goodness-of-fit test. A *significant* result is
BAD news (the split is off), so we flag when p < srm_alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from src.config import CONFIG


@dataclass
class SRMResult:
    passed: bool                 # True = split looks fine
    p_value: float
    chi2: float
    observed: list[int]
    expected: list[float]
    intended_split: list[float]
    alpha: float
    message: str


def srm_check(
    counts: Sequence[int],
    intended_split: Sequence[float] | None = None,
    alpha: float = CONFIG.srm_alpha,
) -> SRMResult:
    """Chi-square goodness-of-fit of observed arm counts vs intended split.

    Parameters
    ----------
    counts : observed unit counts per arm, e.g. ``[n_control, n_treatment]``.
    intended_split : intended proportions, e.g. ``[0.5, 0.5]`` (defaults to equal).
    alpha : significance threshold; p below this raises the SRM flag.
    """
    observed = np.asarray(counts, dtype=float)
    k = len(observed)
    if k < 2:
        raise ValueError("SRM check needs at least two arms.")
    if intended_split is None:
        intended_split = [1.0 / k] * k
    split = np.asarray(intended_split, dtype=float)
    split = split / split.sum()

    total = observed.sum()
    expected = split * total

    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
    passed = bool(p >= alpha)

    if passed:
        msg = f"No SRM detected (p={p:.4f} >= {alpha}). Split looks healthy."
    else:
        obs_frac = observed / total
        msg = (
            f"SRM DETECTED (p={p:.2e} < {alpha}). Observed split "
            f"{np.round(obs_frac, 4).tolist()} differs from intended "
            f"{np.round(split, 4).tolist()} -- randomization/logging may be broken; "
            f"treat downstream results with caution."
        )

    return SRMResult(
        passed=passed,
        p_value=float(p),
        chi2=float(chi2),
        observed=observed.astype(int).tolist(),
        expected=expected.tolist(),
        intended_split=split.tolist(),
        alpha=alpha,
        message=msg,
    )
