"""Phase 1 oracle: the simulator must report a retrievable, correct truth."""
import numpy as np

from src.data.simulate import simulate_experiment, simulate_did_panel


def test_randomized_naive_recovers_truth():
    df, gt = simulate_experiment(scenario="randomized", n=30000, base_effect=2.0, seed=1)
    naive = df.loc[df.treatment == 1, "metric"].mean() - df.loc[df.treatment == 0, "metric"].mean()
    # Under randomization the naive diff is unbiased for the ATE.
    assert abs(naive - gt.true_ate) < 0.5


def test_observational_naive_is_biased():
    df, gt = simulate_experiment(scenario="observational", n=30000, base_effect=2.0, seed=1)
    naive = df.loc[df.treatment == 1, "metric"].mean() - df.loc[df.treatment == 0, "metric"].mean()
    # Confounding inflates the naive estimate well beyond the truth.
    assert naive > gt.true_ate + 2.0


def test_ground_truth_retrievable():
    df, gt = simulate_experiment(heterogeneous=True, uplift_bonus=4.0, seed=2)
    assert gt.true_ate == gt.true_ate  # not NaN
    assert set(gt.effect_by_segment.keys()) == {0, 1}
    assert len(df) == gt.n


def test_did_panel_has_known_effect():
    panel, gt = simulate_did_panel(n_units=2000, true_did_effect=5.0, seed=3)
    assert gt.true_ate == 5.0
    assert set(panel["period"].unique()) == {0, 1}
    assert set(panel["group"].unique()) == {0, 1}
