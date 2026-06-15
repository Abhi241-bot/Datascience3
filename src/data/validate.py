"""Schema checks for user-uploaded experiment CSVs.

Keeps the app honest: an uploaded file must have a treatment column and an
outcome column before any test runs. Returns a structured result instead of
raising, so the Streamlit layer can show a friendly message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    treatment_col: Optional[str] = None
    outcome_col: Optional[str] = None


def validate_experiment_df(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "metric",
) -> ValidationResult:
    """Validate that ``df`` can be analysed as an experiment.

    Rules
    -----
    * treatment and outcome columns must exist,
    * treatment must be binary (exactly two distinct non-null values),
    * outcome must be numeric,
    * both arms must be non-empty.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        return ValidationResult(ok=False, errors=["Uploaded data is empty."])

    if treatment_col not in df.columns:
        errors.append(f"Missing treatment column '{treatment_col}'.")
    if outcome_col not in df.columns:
        errors.append(f"Missing outcome column '{outcome_col}'.")
    if errors:
        return ValidationResult(ok=False, errors=errors)

    treat = df[treatment_col].dropna()
    groups = sorted(treat.unique().tolist())
    if len(groups) != 2:
        errors.append(
            f"Treatment column '{treatment_col}' must have exactly 2 groups; "
            f"found {len(groups)}: {groups[:5]}."
        )
    elif set(groups) != {0, 1}:
        warnings.append(
            f"Treatment groups {groups} are not 0/1; mapping {groups[0]}->control, "
            f"{groups[1]}->treatment."
        )

    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
        errors.append(f"Outcome column '{outcome_col}' must be numeric.")

    if not errors and len(groups) == 2:
        n0 = int((df[treatment_col] == groups[0]).sum())
        n1 = int((df[treatment_col] == groups[1]).sum())
        if n0 == 0 or n1 == 0:
            errors.append("One of the treatment arms is empty.")
        elif min(n0, n1) < 30:
            warnings.append(f"Small arm size (control={n0}, treatment={n1}); results will be noisy.")

    return ValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )
