"""Phase 3: A/B test must recover the known simulated effect within its CI."""
from src.data.simulate import simulate_experiment
from src.analysis.ab_test import analyze_ab


def test_continuous_ci_covers_truth():
    df, gt = simulate_experiment(scenario="randomized", n=30000, base_effect=2.0, seed=11)
    res = analyze_ab(df, outcome_col="metric", outcome_type="continuous")
    assert res.ci_low <= gt.true_ate <= res.ci_high
    assert res.significant
    assert res.test_name.startswith("Welch")


def test_binary_ci_covers_truth():
    df, gt = simulate_experiment(scenario="randomized", n=50000, base_effect=0.03,
                                 base_rate=0.20, outcome_type="binary", seed=12)
    res = analyze_ab(df, outcome_col="converted", outcome_type="binary")
    assert res.ci_low <= gt.true_ate <= res.ci_high


def test_null_effect_not_significant():
    df, _ = simulate_experiment(scenario="randomized", n=20000, base_effect=0.0, seed=13)
    res = analyze_ab(df, outcome_col="metric", outcome_type="continuous")
    assert not res.significant
