"""Generate the README 'money shot': naive vs causal estimates on confounded data.

Runs the observational (confounded) scenario where assignment is self-selected,
then shows that the naive A/B difference is badly biased while DoWhy's
propensity-score matching recovers an estimate close to the known truth.

Usage:  python scripts/make_figures.py
Writes: assets/bias_contrast.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.simulate import simulate_experiment
from src.analysis.psm import estimate_psm


def main() -> None:
    os.makedirs("assets", exist_ok=True)

    df, gt = simulate_experiment(scenario="observational", n=15000, base_effect=2.0, seed=7)
    psm = estimate_psm(df, outcome_col="metric", max_n=5000)

    labels = ["Naive A/B\n(unadjusted)", "Causal (PSM / DoWhy)\n(adjusted)"]
    values = [psm.naive_estimate, psm.adjusted_estimate]
    colors = ["#dc2626", "#2563eb"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(gt.true_ate, color="#16a34a", linestyle="--", linewidth=2,
               label=f"True effect = {gt.true_ate:.2f}")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Estimated treatment effect")
    ax.set_title("Confounded data: naive A/B is biased, causal adjustment recovers the truth")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join("assets", "bias_contrast.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  true={gt.true_ate:.3f}  naive={psm.naive_estimate:.3f}  psm={psm.adjusted_estimate:.3f}")


if __name__ == "__main__":
    main()
