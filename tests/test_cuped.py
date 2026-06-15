"""Phase 3: CUPED must shrink variance without biasing the estimate."""
from src.data.simulate import simulate_experiment
from src.analysis.cuped import analyze_with_cuped, apply_cuped


def test_cuped_shrinks_ci_and_keeps_truth():
    df, gt = simulate_experiment(scenario="randomized", n=30000, base_effect=2.0, seed=21)
    c = analyze_with_cuped(df, outcome_col="metric", covariate_col="pre_metric")
    # variance reduction is positive when the covariate predicts the outcome
    assert c.variance_reduction > 0.05
    # CUPED CI is strictly tighter than naive
    assert c.ci_width_adjusted < c.ci_width_naive
    # estimate stays valid: CI still covers the truth
    assert c.adjusted.ci_low <= gt.true_ate <= c.adjusted.ci_high


def test_apply_cuped_preserves_mean():
    df, _ = simulate_experiment(scenario="randomized", n=10000, seed=22)
    y_adj, theta, var_red = apply_cuped(df, "metric", "pre_metric")
    # subtracting a mean-centred term leaves the overall mean ~unchanged
    assert abs(y_adj.mean() - df["metric"].mean()) < 1e-6
    assert 0.0 <= var_red < 1.0
