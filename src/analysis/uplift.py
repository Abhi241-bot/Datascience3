"""Heterogeneous treatment effects via an EconML causal forest.

Average effects hide who actually responds. A CausalForestDML estimates a
per-user conditional effect (CATE) as a function of observable features, letting
us rank segments by uplift and surface the high-responder group. On simulated
data this should recover the segment we deliberately injected with extra lift.

Double machine learning (the "DML" in CausalForestDML) first partials out the
confounders W from both treatment and outcome with flexible nuisance models,
then fits a forest on the residuals -- so it is valid even when assignment was
confounded, as long as the confounders are included in W.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.config import CONFIG


@dataclass
class UpliftResult:
    segment_table: pd.DataFrame      # per-segment mean CATE + CI
    top_segment: int                 # segment value with the highest estimated uplift
    ate: float                       # overall average treatment effect from the forest
    feature_cols: List[str] = field(default_factory=list)


def estimate_uplift(
    df: pd.DataFrame,
    outcome_col: str = "metric",
    treatment_col: str = "treatment",
    segment_col: str = "segment",
    confounder_cols: List[str] | None = None,
    max_n: int = 8000,
    seed: int = CONFIG.random_seed,
) -> UpliftResult:
    """Fit a causal forest and return a segment-level uplift table.

    Features ``X`` are the effect-modifier(s) we want heterogeneity over
    (the segment); confounders ``W`` are adjusted for but not used to split the
    reported segments.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    if confounder_cols is None:
        confounder_cols = ["pre_metric", "age"]

    data = df.dropna(subset=[outcome_col, treatment_col, segment_col, *confounder_cols])
    if len(data) > max_n:
        data = data.sample(max_n, random_state=seed).reset_index(drop=True)

    y = data[outcome_col].to_numpy(dtype=float)
    tr = data[treatment_col].to_numpy(dtype=int)
    X = data[[segment_col]].to_numpy(dtype=float)
    W = data[confounder_cols].to_numpy(dtype=float)

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=seed),
        discrete_treatment=True,
        n_estimators=300,
        min_samples_leaf=20,
        random_state=seed,
    )
    est.fit(y, tr, X=X, W=W)

    cate = est.effect(X)             # per-row estimated treatment effect
    data = data.assign(_cate=cate)

    rows = []
    for seg, grp in data.groupby(segment_col):
        idx = grp.index.to_numpy()
        seg_X = data.loc[idx, [segment_col]].to_numpy(dtype=float)
        lo, hi = est.effect_interval(seg_X, alpha=CONFIG.alpha)
        rows.append(
            {
                "segment": int(seg),
                "n": int(len(grp)),
                "uplift": float(grp["_cate"].mean()),
                "ci_low": float(np.mean(lo)),
                "ci_high": float(np.mean(hi)),
            }
        )
    seg_table = pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)
    top_segment = int(seg_table.iloc[0]["segment"])

    return UpliftResult(
        segment_table=seg_table,
        top_segment=top_segment,
        ate=float(np.mean(cate)),
        feature_cols=[segment_col],
    )
