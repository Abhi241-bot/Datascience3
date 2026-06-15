"""Phase 4: difference-in-differences must recover the injected effect."""
from src.data.simulate import simulate_did_panel
from src.analysis.did import estimate_did


def test_did_recovers_truth():
    panel, gt = simulate_did_panel(n_units=5000, true_did_effect=5.0, group_gap=12.0, seed=31)
    res = estimate_did(panel)
    # CI covers the truth ...
    assert res.ci_low <= gt.true_ate <= res.ci_high
    # ... and the naive post-period diff is biased by roughly the group gap.
    assert res.naive_post_diff > gt.true_ate + 5.0
    assert res.significant
