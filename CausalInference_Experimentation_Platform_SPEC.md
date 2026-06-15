# Causal Inference & A/B Experimentation Platform — Build Spec

> **For the AI agent (Antigravity):** This is a build specification, not a tutorial. Build phase by phase. After each phase, stop and confirm the acceptance criteria before moving on. Prefer simple, correct statistics over flashy UI. Do NOT over-engineer: no database server required, no auth, no cloud account. The whole thing must run with `streamlit run app.py`.

---

## 0. Project Goal (read first)

Build a **self-serve experimentation analysis platform**: a user uploads (or simulates) experiment data, and the app (1) runs a proper A/B test analysis, and (2) — the differentiator — applies **causal inference methods** when randomization is broken or impossible, then outputs a plain-English **"ship / don't ship"** recommendation with effect sizes and confidence intervals.

**Why this is the point:** Causal inference is the single fastest-growing Data Science skill, and it's what senior DS interviews at Netflix/Microsoft/Booking weight most — yet almost no student portfolio shows it. This project signals statistical maturity, not just "I ran a t-test." The recruiter hook is a live app where they upload a CSV and instantly get a rigorous verdict.

### Scope discipline
This is a **statistics** project with a thin UI, NOT a UI project. Spend effort on correctness of the methods, not on styling. Every number on screen must be defensible in an interview.

---

## 1. Tech Stack (use exactly these)

| Layer | Tool | Why |
|---|---|---|
| Core stats | **statsmodels**, **scipy** | t-tests, regression, CIs, power |
| A/B design | custom + statsmodels | power analysis, sample sizing, SRM check |
| Causal methods | **DoWhy** + **EconML** | DiD, PSM, IV, causal forests |
| Variance reduction | custom | CUPED |
| Data | pandas, numpy | — |
| App | **Streamlit** | fastest path to a clickable demo |
| (Optional) analytics layer | **dbt** + **DuckDB** | shows analytics-engineering depth |
| Plotting | plotly or matplotlib | effect plots, CI bands |
| Language | Python 3.11 | — |

---

## 2. Repository Structure

```
causal-experimentation-platform/
├── README.md                      # recruiter-grade, written LAST
├── requirements.txt
├── app.py                         # Streamlit entrypoint
├── src/
│   ├── config.py                  # thresholds (alpha, MDE, power) in ONE place
│   ├── data/
│   │   ├── simulate.py            # generates synthetic experiment data with KNOWN ground-truth effect
│   │   └── validate.py            # schema checks on uploaded CSV
│   ├── design/
│   │   ├── power.py               # power analysis + sample size calculator
│   │   └── srm.py                 # sample ratio mismatch check (chi-square)
│   ├── analysis/
│   │   ├── ab_test.py             # frequentist A/B: t-test/z-test, CIs, p-values
│   │   ├── cuped.py               # CUPED variance reduction using pre-period covariate
│   │   ├── did.py                 # difference-in-differences
│   │   ├── psm.py                 # propensity score matching (DoWhy)
│   │   └── uplift.py              # heterogeneous treatment effects (EconML causal forest)
│   └── report/
│       └── verdict.py             # turns stats into a plain-English ship/don't-ship call
├── tests/
│   ├── test_ab_test.py            # validates against simulated KNOWN effect
│   ├── test_cuped.py
│   └── test_did.py
├── (optional) dbt_project/        # DuckDB-backed models that produce the analysis tables
└── notebooks/
    └── 01_method_walkthrough.ipynb
```

---

## 3. Build Phases (do in order; confirm acceptance criteria each time)

### Phase 1 — Synthetic data with known ground truth (do this FIRST)
- `simulate.py`: generate experiment data where you **control the true treatment effect**, sample sizes, base rate, pre-period covariate (for CUPED), and optional confounding (for the causal methods). Support both a clean randomized scenario and a "broken randomization / observational" scenario.
- **Why first:** every method you build is validated by checking it recovers the known effect you injected. This is your test oracle.
- **✅ Acceptance:** `simulate.py` produces a DataFrame; the true effect is recorded and retrievable for tests.

### Phase 2 — A/B test design tools
- `power.py`: given baseline rate, minimum detectable effect (MDE), alpha, and power, compute required sample size per arm; and the reverse (given n, what MDE is detectable).
- `srm.py`: sample ratio mismatch check — chi-square test that the actual split matches the intended split (a real-world gotcha that flags broken experiments).
- **✅ Acceptance:** power calc matches a known reference value (e.g., from statsmodels) within rounding; SRM correctly flags a deliberately skewed split.

### Phase 3 — Frequentist A/B analysis + CUPED
- `ab_test.py`: two-sample test (t-test for continuous, z-test/proportions for binary), absolute & relative lift, confidence intervals, p-value, and a clear "statistically significant?" flag.
- `cuped.py`: CUPED variance reduction using a pre-experiment covariate; report the variance reduction % and the tightened CI.
- **✅ Acceptance:** On simulated data, the estimated effect's CI contains the known true effect; CUPED visibly shrinks the CI vs the naive estimate.

### Phase 4 — Causal methods (the differentiator)
- `did.py`: difference-in-differences for a before/after, treatment/control panel; report the DiD estimate + CI.
- `psm.py`: propensity score matching via DoWhy on the observational/confounded scenario; show the estimate before vs after adjusting for confounders.
- `uplift.py`: EconML causal forest for **heterogeneous treatment effects** — which segments respond most. Output a segment-level uplift table/plot.
- **✅ Acceptance:** On the confounded scenario, the naive A/B estimate is biased, but PSM/DoWhy recovers (closer to) the true effect — and the README/app shows this contrast explicitly. The causal forest identifies the high-uplift segment you injected.

### Phase 5 — Verdict engine
- `verdict.py`: combine the results into a plain-English recommendation: ship / don't ship / inconclusive — with the reasoning (effect size, significance, sample adequacy, any SRM warning, segment notes). This "connect analysis to a business decision" step is what makes it read as senior-level.
- **✅ Acceptance:** Given a clearly-positive simulated experiment → "ship"; a null one → "don't ship"; an underpowered one → "inconclusive, need N more samples."

### Phase 6 — Streamlit app
- `app.py`: tabs/sections for (1) upload CSV **or** "simulate", (2) design check (power + SRM), (3) A/B results with CUPED, (4) causal analysis comparison, (5) the verdict. Include the effect plots and the naive-vs-causal contrast.
- **✅ Acceptance:** A recruiter can hit "simulate" and walk the whole flow to a verdict in under a minute, no setup.

### Phase 7 — (Optional) dbt + DuckDB layer
- If time allows, add a small dbt project on DuckDB that transforms raw event logs into the analysis-ready tables the app consumes. This adds an analytics-engineering signal (dbt is a rising DS skill).
- **✅ Acceptance:** `dbt run` builds the models; the app can read from them.

### Phase 8 — Tests, README, deploy
- `tests/`: each method validated against the known simulated effect.
- `README.md` (write last): see Phase 9.
- **Deploy to Streamlit Community Cloud** (free) — clickable URL is mandatory.
- **✅ Acceptance:** tests green; deployed app reachable at a public URL.

---

## 4. README requirements (Phase 9 — what recruiters read)
In this order:
1. One-paragraph problem + one-line value ("Self-serve experiment analysis that flags broken randomization and recovers true effects with causal methods").
2. **Live demo link** + a short GIF walking sim → verdict.
3. **The money shot:** a side-by-side showing the naive A/B estimate is biased on observational data while the causal method recovers the truth — with a chart.
4. Method list with one line each on *when to use which*.
5. One-command run (`streamlit run app.py`) + quickstart.

---

## 5. Hard Constraints / Guardrails
- **Validate every method against the known simulated effect** — this is non-negotiable; it's how you (and the interviewer) trust the numbers.
- **No database server, no auth, no cloud.** Streamlit + (optional) DuckDB only.
- **One config file** for alpha/power/MDE/thresholds.
- **Statistics correctness > UI polish.** Do not spend time on styling at the expense of method correctness.
- If time runs short, cut the dbt layer and the uplift/causal-forest piece LAST — but A/B + CUPED + (DiD or PSM) + verdict MUST work, because the naive-vs-causal contrast is the whole pitch.

## 6. Definition of Done
- [ ] `streamlit run app.py` works from a fresh clone
- [ ] Simulate → design check → A/B + CUPED → causal analysis → verdict flows end to end
- [ ] Methods recover the known simulated effect (validated in tests)
- [ ] The naive-vs-causal bias contrast is shown clearly (chart + README)
- [ ] Plain-English ship/don't-ship verdict with reasoning
- [ ] Deployed to Streamlit Cloud with a clickable URL
- [ ] README with demo link, the bias-contrast chart, and method guidance
