"""Verdict engine: statistics -> a business decision.

Turns an A/B result (plus optional SRM, power, and causal context) into a
plain-English SHIP / DON'T SHIP / INCONCLUSIVE call with the reasoning spelled
out. The logic mirrors how a careful DS would actually argue it:

  * SRM failed            -> INCONCLUSIVE (the data itself is untrustworthy).
  * significant + clears
    the practical floor    -> SHIP.
  * significant but below
    the practical floor     -> DON'T SHIP (real but too small to matter).
  * not significant but
    underpowered            -> INCONCLUSIVE (collect N more, here's how many).
  * not significant and
    adequately powered      -> DON'T SHIP (we'd have seen it if it were there).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.config import CONFIG
from src.analysis.ab_test import ABResult
from src.design.power import required_sample_size


SHIP = "SHIP"
DONT_SHIP = "DON'T SHIP"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class Verdict:
    decision: str                    # SHIP | DON'T SHIP | INCONCLUSIVE
    headline: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_additional_n: Optional[int] = None

    def as_markdown(self) -> str:
        emoji = {SHIP: "[SHIP]", DONT_SHIP: "[DON'T SHIP]", INCONCLUSIVE: "[INCONCLUSIVE]"}[self.decision]
        lines = [f"### {emoji} {self.decision}", "", f"**{self.headline}**", ""]
        for r in self.reasons:
            lines.append(f"- {r}")
        for w in self.warnings:
            lines.append(f"- WARNING: {w}")
        if self.recommended_additional_n:
            lines.append(f"- Collect ~**{self.recommended_additional_n:,} more users per arm** to reach adequate power.")
        return "\n".join(lines)


def decide(
    ab: ABResult,
    srm_passed: bool = True,
    srm_message: str = "",
    practical_floor_relative: float = CONFIG.practical_floor_relative,
    outcome_sd: Optional[float] = None,
    causal_note: Optional[str] = None,
) -> Verdict:
    """Produce a verdict from an A/B result and optional context.

    Parameters
    ----------
    ab : the primary A/B result (raw or CUPED-adjusted).
    srm_passed : False if the sample-ratio-mismatch guardrail fired.
    practical_floor_relative : minimum relative lift that is worth shipping.
    outcome_sd : outcome SD; enables the "need N more" sample-size suggestion
        when a continuous result is non-significant and possibly underpowered.
    causal_note : optional one-liner about confounding / causal adjustment to
        surface in the reasoning (e.g. "naive estimate was biased; PSM used").
    """
    reasons: List[str] = []
    warnings: List[str] = []

    rel = ab.relative_lift
    abs_lift = ab.absolute_lift
    floor = practical_floor_relative

    # 1) Guardrail: a broken split poisons everything else.
    if not srm_passed:
        v = Verdict(
            decision=INCONCLUSIVE,
            headline="Sample ratio mismatch detected -- the experiment data can't be trusted yet.",
            reasons=[
                srm_message or "Observed arm split differs from the intended split.",
                "Fix randomization/logging and re-run before reading the effect.",
            ],
        )
        if causal_note:
            v.reasons.append(causal_note)
        return v

    if causal_note:
        reasons.append(causal_note)

    direction = "increase" if abs_lift >= 0 else "decrease"
    effect_str = (
        f"Estimated {direction} of {abs_lift:+.4g} "
        f"({rel:+.2%} relative), 95% CI [{ab.ci_low:.4g}, {ab.ci_high:.4g}], p={ab.p_value:.2g}."
    )

    # 2) Significant result.
    if ab.significant:
        reasons.append(effect_str)
        if abs(rel) >= floor and abs_lift > 0:
            reasons.append(
                f"Effect is statistically significant AND clears the practical floor "
                f"of {floor:.0%} relative lift."
            )
            return Verdict(
                decision=SHIP,
                headline="Significant, positive, and large enough to matter.",
                reasons=reasons,
                warnings=warnings,
            )
        if abs_lift <= 0:
            reasons.append("Effect is significant but in the WRONG direction (a regression).")
            return Verdict(
                decision=DONT_SHIP,
                headline="Significant but harmful -- do not ship.",
                reasons=reasons,
                warnings=warnings,
            )
        reasons.append(
            f"Effect is significant but the {rel:.2%} relative lift is below the "
            f"{floor:.0%} practical floor -- statistically real, commercially negligible."
        )
        return Verdict(
            decision=DONT_SHIP,
            headline="Real but too small to be worth the change.",
            reasons=reasons,
            warnings=warnings,
        )

    # 3) Not significant -- is it a true null or just underpowered?
    reasons.append(effect_str)
    additional_n = None
    underpowered = False
    if ab.outcome_type == "continuous" and outcome_sd and ab.mean_control != 0:
        target_abs = abs(ab.mean_control) * floor
        try:
            need = required_sample_size(
                baseline=ab.mean_control, mde_absolute=target_abs,
                outcome_type="continuous", sd=outcome_sd,
            ).per_arm
            current = min(ab.n_control, ab.n_treatment)
            if need > current:
                underpowered = True
                additional_n = int(need - current)
        except Exception:
            pass
    elif ab.outcome_type == "binary" and ab.mean_control not in (0, 1):
        target_abs = abs(ab.mean_control) * floor
        try:
            need = required_sample_size(
                baseline=ab.mean_control, mde_absolute=target_abs, outcome_type="binary",
            ).per_arm
            current = min(ab.n_control, ab.n_treatment)
            if need > current:
                underpowered = True
                additional_n = int(need - current)
        except Exception:
            pass

    if underpowered:
        return Verdict(
            decision=INCONCLUSIVE,
            headline="No significant effect yet, but the test is underpowered for the floor that matters.",
            reasons=reasons + [
                "Can't rule out a practically meaningful effect at the current sample size."
            ],
            warnings=warnings,
            recommended_additional_n=additional_n,
        )

    return Verdict(
        decision=DONT_SHIP,
        headline="No effect, and the test had enough power to find one.",
        reasons=reasons + [
            "Adequately powered to detect the practical floor, yet nothing significant turned up."
        ],
        warnings=warnings,
    )
