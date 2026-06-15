"""Phase 4: PSM removes confounding bias; the causal forest finds the segment.

These are the differentiator's guarantees, asserted against known truth.
Marked slow because they fit DoWhy / EconML models.
"""
import pytest

from src.data.simulate import simulate_experiment
from src.analysis.psm import estimate_psm
from src.analysis.uplift import estimate_uplift


@pytest.mark.slow
def test_psm_recovers_truth_where_naive_is_biased():
    df, gt = simulate_experiment(scenario="observational", n=12000, base_effect=2.0, seed=41)
    res = estimate_psm(df, outcome_col="metric")
    bias_naive = abs(res.naive_estimate - gt.true_ate)
    bias_psm = abs(res.adjusted_estimate - gt.true_ate)
    # the naive number is badly biased ...
    assert bias_naive > 3.0
    # ... and PSM gets substantially closer to the truth.
    assert bias_psm < bias_naive / 2


@pytest.mark.slow
def test_causal_forest_identifies_high_uplift_segment():
    df, gt = simulate_experiment(scenario="randomized", n=12000, base_effect=2.0,
                                 heterogeneous=True, uplift_bonus=6.0,
                                 high_uplift_segment=1, seed=42)
    res = estimate_uplift(df, outcome_col="metric")
    assert res.top_segment == gt.high_uplift_segment
    # the recovered uplift ordering should separate the two segments clearly
    table = res.segment_table.set_index("segment")
    assert table.loc[1, "uplift"] > table.loc[0, "uplift"] + 2.0
