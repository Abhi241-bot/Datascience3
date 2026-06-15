"""Causal Experimentation Platform -- Streamlit entrypoint.

Run with:  streamlit run app.py

Flow (left to right tabs): pick data (upload or simulate) -> design check
(power + SRM) -> A/B + CUPED -> causal analysis (naive vs adjusted) -> verdict.
The thin UI here deliberately just orchestrates the tested methods in src/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CONFIG
from src.data.simulate import simulate_experiment, simulate_did_panel
from src.data.validate import validate_experiment_df
from src.design.power import required_sample_size, detectable_effect
from src.design.srm import srm_check
from src.analysis.ab_test import analyze_ab
from src.analysis.cuped import analyze_with_cuped
from src.analysis.did import estimate_did
from src.report.verdict import decide

st.set_page_config(page_title="Causal Experimentation Platform", page_icon="[stats]", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy computations (causal methods are slow; key on a data signature) #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_psm(df: pd.DataFrame, outcome_col: str):
    from src.analysis.psm import estimate_psm
    return estimate_psm(df, outcome_col=outcome_col, with_ci=False)


@st.cache_data(show_spinner=False)
def _cached_uplift(df: pd.DataFrame, outcome_col: str):
    from src.analysis.uplift import estimate_uplift
    return estimate_uplift(df, outcome_col=outcome_col)


def _ci_plot(labels, estimates, lows, highs, truth=None, title="", xlab="Estimated effect"):
    """Horizontal point-estimate + CI plot; optional ground-truth reference line."""
    fig = go.Figure()
    for lab, est, lo, hi in zip(labels, estimates, lows, highs):
        fig.add_trace(go.Scatter(
            x=[est], y=[lab], mode="markers", marker=dict(size=12),
            error_x=dict(type="data", symmetric=False, array=[hi - est], arrayminus=[est - lo]),
            name=lab, showlegend=False,
        ))
    if truth is not None:
        fig.add_vline(x=truth, line_dash="dash", line_color="green",
                      annotation_text=f"true = {truth:.3g}", annotation_position="top")
    fig.update_layout(title=title, xaxis_title=xlab, height=120 + 60 * len(labels),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: data source                                                        #
# --------------------------------------------------------------------------- #
st.sidebar.title("Data")
source = st.sidebar.radio("Source", ["Simulate", "Upload CSV"], index=0)

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.gt = None
    st.session_state.outcome_col = "metric"

if source == "Simulate":
    st.sidebar.subheader("Scenario")
    scenario = st.sidebar.selectbox(
        "Assignment", ["randomized", "observational"],
        help="randomized = clean A/B; observational = confounded self-selection (naive estimate is biased).",
    )
    n = st.sidebar.slider("Users (total)", 2000, 50000, 20000, step=2000)
    base_rate = st.sidebar.number_input("Baseline metric (control mean)", value=50.0)
    base_effect = st.sidebar.number_input("True treatment effect", value=2.0, step=0.5)
    heterogeneous = st.sidebar.checkbox("Heterogeneous effect (inject a high-uplift segment)", value=True)
    uplift_bonus = st.sidebar.number_input("Extra effect for high-uplift segment", value=6.0,
                                           step=1.0, disabled=not heterogeneous)
    intended_split = st.sidebar.slider("Intended treatment split", 0.1, 0.9, 0.5, step=0.05)
    seed = st.sidebar.number_input("Seed", value=CONFIG.random_seed, step=1)
    if st.sidebar.button("Simulate data", type="primary"):
        df, gt = simulate_experiment(
            n=int(n), base_effect=base_effect, base_rate=base_rate, scenario=scenario,
            heterogeneous=heterogeneous, uplift_bonus=uplift_bonus,
            intended_split=intended_split, seed=int(seed),
        )
        st.session_state.df = df
        st.session_state.gt = gt
        st.session_state.outcome_col = "metric"
else:
    up = st.sidebar.file_uploader("Experiment CSV", type=["csv"])
    treatment_col = st.sidebar.text_input("Treatment column", value="treatment")
    outcome_col = st.sidebar.text_input("Outcome column", value="metric")
    if up is not None:
        df = pd.read_csv(up)
        res = validate_experiment_df(df, treatment_col=treatment_col, outcome_col=outcome_col)
        if not res.ok:
            st.sidebar.error(" ; ".join(res.errors))
        else:
            for w in res.warnings:
                st.sidebar.warning(w)
            st.session_state.df = df
            st.session_state.gt = None
            st.session_state.outcome_col = outcome_col


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
st.title("Causal Experimentation Platform")
st.caption("Self-serve experiment analysis that flags broken randomization and recovers "
           "true effects with causal methods.")

df = st.session_state.df
gt = st.session_state.gt
outcome_col = st.session_state.outcome_col

if df is None:
    st.info("Pick a data source in the sidebar. Hit **Simulate data** to walk the full flow in under a minute.")
    st.stop()

truth = gt.true_ate if gt is not None else None

tab_data, tab_design, tab_ab, tab_causal, tab_verdict = st.tabs(
    ["1. Data", "2. Design check", "3. A/B + CUPED", "4. Causal analysis", "5. Verdict"]
)

# --- Tab 1: Data ----------------------------------------------------------- #
with tab_data:
    c1, c2, c3 = st.columns(3)
    n0 = int((df["treatment"] == 0).sum())
    n1 = int((df["treatment"] == 1).sum())
    c1.metric("Total users", f"{len(df):,}")
    c2.metric("Control / Treatment", f"{n0:,} / {n1:,}")
    if gt is not None:
        c3.metric("True ATE (injected)", f"{gt.true_ate:.3f}")
    st.dataframe(df.head(20), width="stretch")
    if gt is not None:
        st.caption(f"Scenario: **{gt.scenario}** | heterogeneous: **{gt.heterogeneous}** | "
                   f"effect by segment: {{ {', '.join(f'{k}: {v:.2f}' for k, v in gt.effect_by_segment.items())} }}")

# --- Tab 2: Design check --------------------------------------------------- #
with tab_design:
    st.subheader("Sample Ratio Mismatch (SRM)")
    srm = srm_check([n0, n1])
    (st.success if srm.passed else st.error)(srm.message)

    st.subheader("Power / sample-size planning")
    colp1, colp2 = st.columns(2)
    with colp1:
        base = float(df[outcome_col].mean()) if df[outcome_col].mean() != 0 else 1.0
        sd = float(df[outcome_col].std())
        mde_rel = st.slider("Target relative MDE", 0.01, 0.20, 0.05, step=0.01)
        ss = required_sample_size(baseline=base, mde_relative=mde_rel,
                                  outcome_type="continuous", sd=sd)
        st.metric("Required N per arm", f"{ss.per_arm:,}",
                  help=f"to detect a {mde_rel:.0%} relative lift at alpha={CONFIG.alpha}, power={CONFIG.power}")
    with colp2:
        de = detectable_effect(baseline=base, n_per_arm=min(n0, n1),
                               outcome_type="continuous", sd=sd)
        st.metric("Detectable relative MDE at current N", f"{de['mde_relative']:.2%}",
                  help="smallest relative lift this sample size can detect")

# --- Tab 3: A/B + CUPED ---------------------------------------------------- #
with tab_ab:
    has_cov = "pre_metric" in df.columns
    if has_cov:
        cu = analyze_with_cuped(df, outcome_col=outcome_col, covariate_col="pre_metric")
        ab = cu.naive
    else:
        ab = analyze_ab(df, outcome_col=outcome_col, outcome_type="continuous")
        cu = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control mean", f"{ab.mean_control:.3f}")
    c2.metric("Treatment mean", f"{ab.mean_treatment:.3f}")
    c3.metric("Absolute lift", f"{ab.absolute_lift:+.3f}")
    c4.metric("Relative lift", f"{ab.relative_lift:+.2%}")
    st.write(f"**{ab.test_name}** -> p = `{ab.p_value:.3g}`, "
             f"95% CI on lift = `[{ab.ci_low:.3f}, {ab.ci_high:.3f}]`, "
             f"significant: **{ab.significant}**")

    if cu is not None:
        st.subheader("CUPED variance reduction")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Variance reduction", f"{cu.variance_reduction:.1%}")
        cc2.metric("Naive CI width", f"{cu.ci_width_naive:.3f}")
        cc3.metric("CUPED CI width", f"{cu.ci_width_adjusted:.3f}",
                   delta=f"{cu.ci_width_adjusted - cu.ci_width_naive:.3f}")
        fig = _ci_plot(
            ["Naive A/B", "CUPED-adjusted"],
            [cu.naive.absolute_lift, cu.adjusted.absolute_lift],
            [cu.naive.ci_low, cu.adjusted.ci_low],
            [cu.naive.ci_high, cu.adjusted.ci_high],
            truth=truth, title="CUPED tightens the confidence interval",
        )
        st.plotly_chart(fig, width="stretch")

# --- Tab 4: Causal analysis ------------------------------------------------ #
with tab_causal:
    st.subheader("The money shot: naive A/B vs causal adjustment")
    has_conf = all(c in df.columns for c in ["pre_metric", "age"])
    if not has_conf:
        st.info("Causal adjustment needs confounder columns (`pre_metric`, `age`). "
                "Use the simulator's observational scenario to see the bias get removed.")
    else:
        try:
            with st.spinner("Running DoWhy propensity-score matching..."):
                psm = _cached_psm(df, outcome_col)
        except Exception as exc:  # missing/incompatible DoWhy build on the host
            st.warning(f"Propensity-score matching unavailable in this environment "
                       f"(DoWhy/cvxpy import failed: {exc}). Run locally for the full causal stack.")
            psm = None
        if psm is not None:
            labels = ["Naive A/B (unadjusted)", "PSM / DoWhy (adjusted)"]
            ests = [psm.naive_estimate, psm.adjusted_estimate]
            fig = _ci_plot(labels, ests, ests, ests, truth=truth,
                           title="Propensity-score matching removes confounding bias")
            st.plotly_chart(fig, width="stretch")
            cc1, cc2 = st.columns(2)
            cc1.metric("Naive estimate (biased)", f"{psm.naive_estimate:.3f}")
            cc2.metric("Adjusted estimate (PSM)", f"{psm.adjusted_estimate:.3f}")
            if truth is not None:
                bias_naive = abs(psm.naive_estimate - truth)
                bias_psm = abs(psm.adjusted_estimate - truth)
                st.success(f"PSM cut the bias from **{bias_naive:.3f}** (naive) to **{bias_psm:.3f}** "
                           f"vs the known true effect of {truth:.3f}.")

    st.divider()
    st.subheader("Heterogeneous effects (causal forest)")
    if "segment" in df.columns:
        try:
            with st.spinner("Fitting EconML causal forest..."):
                up = _cached_uplift(df, outcome_col)
        except Exception as exc:  # missing/incompatible EconML build on the host
            st.warning(f"Causal forest unavailable in this environment "
                       f"(EconML import failed: {exc}). Run locally for the full causal stack.")
            up = None
        if up is not None:
            st.dataframe(up.segment_table, width="stretch")
            fig2 = _ci_plot(
                [f"segment {int(r.segment)}" for r in up.segment_table.itertuples()],
                up.segment_table["uplift"].tolist(),
                up.segment_table["ci_low"].tolist(),
                up.segment_table["ci_high"].tolist(),
                title="Per-segment uplift", xlab="Estimated uplift (CATE)",
            )
            st.plotly_chart(fig2, width="stretch")
            st.info(f"Highest-uplift segment: **{up.top_segment}**"
                    + (f" (injected high segment was {gt.high_uplift_segment})" if gt is not None else ""))
    else:
        st.info("No `segment` column to analyse heterogeneity.")

    st.divider()
    st.subheader("Difference-in-differences (separate before/after panel)")
    st.caption("DiD needs a panel (group x period). Generate one to see DiD remove a "
               "fixed group gap a naive post-period comparison would mistake for an effect.")
    did_effect = st.number_input("True DiD effect for demo panel", value=5.0, step=1.0)
    if st.button("Simulate DiD panel & estimate"):
        panel, gtp = simulate_did_panel(true_did_effect=did_effect, seed=int(CONFIG.random_seed))
        did = estimate_did(panel)
        d1, d2, d3 = st.columns(3)
        d1.metric("True DiD", f"{gtp.true_ate:.3f}")
        d2.metric("Naive post diff (biased)", f"{did.naive_post_diff:.3f}")
        d3.metric("DiD estimate", f"{did.estimate:.3f}",
                  help=f"95% CI [{did.ci_low:.3f}, {did.ci_high:.3f}]")
        st.success(f"DiD recovered {did.estimate:.3f} (CI [{did.ci_low:.3f}, {did.ci_high:.3f}]) "
                   f"vs a naive post-period diff of {did.naive_post_diff:.3f}.")

# --- Tab 5: Verdict -------------------------------------------------------- #
with tab_verdict:
    has_cov = "pre_metric" in df.columns
    if has_cov:
        cu = analyze_with_cuped(df, outcome_col=outcome_col, covariate_col="pre_metric")
        primary = cu.adjusted     # use the tighter CUPED estimate when available
    else:
        primary = analyze_ab(df, outcome_col=outcome_col, outcome_type="continuous")

    srm = srm_check([n0, n1])
    causal_note = None
    if gt is not None and gt.scenario == "observational":
        causal_note = ("Assignment is confounded (observational): the naive A/B number is biased "
                       "upward -- trust the PSM/DoWhy adjusted estimate over the raw difference.")

    v = decide(primary, srm_passed=srm.passed, srm_message=srm.message,
               outcome_sd=float(df[outcome_col].std()), causal_note=causal_note)
    st.markdown(v.as_markdown())
    if truth is not None:
        st.caption(f"(Known true effect for this simulation: {truth:.3f})")
