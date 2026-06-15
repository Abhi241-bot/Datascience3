"""Single source of truth for statistical thresholds.

Every alpha / power / MDE / decision threshold used anywhere in the platform
lives here. Do not hard-code these numbers elsewhere -- import from this module
so the whole app stays internally consistent and defensible in an interview.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Hypothesis testing ---
    alpha: float = 0.05          # two-sided significance level
    power: float = 0.80          # desired statistical power (1 - beta)

    # --- Experiment design defaults ---
    mde_relative: float = 0.05   # minimum detectable effect, as a relative lift (5%)

    # --- Verdict thresholds ---
    # A result must be both statistically significant AND clear the practical
    # floor below to earn a "ship". This separates statistical from practical
    # significance -- a senior-DS distinction.
    practical_floor_relative: float = 0.01   # >=1% relative lift to matter commercially
    srm_alpha: float = 0.001     # SRM is a guardrail; use a strict alpha to avoid false alarms

    # --- Reproducibility ---
    random_seed: int = 42


CONFIG = Config()
