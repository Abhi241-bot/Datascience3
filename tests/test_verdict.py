"""Phase 5: verdict maps stats to the right business decision."""
from src.data.simulate import simulate_experiment
from src.analysis.ab_test import analyze_ab
from src.report.verdict import decide, SHIP, DONT_SHIP, INCONCLUSIVE


def _ab(df):
    return analyze_ab(df, outcome_col="metric", outcome_type="continuous")


def test_clear_positive_ships():
    df, _ = simulate_experiment(scenario="randomized", n=40000, base_effect=4.0, seed=51)
    v = decide(_ab(df), srm_passed=True, outcome_sd=float(df["metric"].std()))
    assert v.decision == SHIP


def test_null_well_powered_dont_ship():
    df, _ = simulate_experiment(scenario="randomized", n=40000, base_effect=0.0, seed=52)
    v = decide(_ab(df), srm_passed=True, outcome_sd=float(df["metric"].std()))
    assert v.decision == DONT_SHIP


def test_underpowered_inconclusive():
    df, _ = simulate_experiment(scenario="randomized", n=300, base_effect=0.5, seed=53)
    v = decide(_ab(df), srm_passed=True, outcome_sd=float(df["metric"].std()))
    assert v.decision == INCONCLUSIVE
    assert v.recommended_additional_n and v.recommended_additional_n > 0


def test_srm_failure_forces_inconclusive():
    df, _ = simulate_experiment(scenario="randomized", n=40000, base_effect=4.0, seed=54)
    v = decide(_ab(df), srm_passed=False, srm_message="skewed split")
    assert v.decision == INCONCLUSIVE
