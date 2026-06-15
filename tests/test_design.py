"""Phase 2: power calc matches statsmodels; SRM flags a skewed split."""
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.design.power import required_sample_size, detectable_effect
from src.design.srm import srm_check


def test_sample_size_matches_statsmodels_reference():
    r = required_sample_size(baseline=0.10, mde_absolute=0.01, outcome_type="binary")
    es = proportion_effectsize(0.11, 0.10)
    ref = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.80,
                                       ratio=1.0, alternative="two-sided")
    assert abs(r.per_arm - ref) <= 1  # within rounding


def test_detectable_effect_roundtrips():
    r = required_sample_size(baseline=0.10, mde_absolute=0.01, outcome_type="binary")
    d = detectable_effect(baseline=0.10, n_per_arm=r.per_arm, outcome_type="binary")
    assert d["mde_absolute"] == pytest.approx(0.01, abs=1e-4)


def test_srm_passes_healthy_and_flags_skew():
    assert srm_check([10000, 10050]).passed is True
    bad = srm_check([10000, 11200])
    assert bad.passed is False
    assert "SRM DETECTED" in bad.message
