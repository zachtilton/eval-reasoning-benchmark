"""
Secondary label-substitution analyses on top of the RC3-corrected scoring.

Re-scores the corrected database against a reference label in which a
specified set of fragments carries an alternative classification, then
re-runs fragment adjudication and the three H2 statistical tests. The locked
gold standard and the *_corrected primary outputs are never modified;
results go to separate files and are reported alongside the primary results.

The first use is the external-reviewer triangulation: the independent
reviewer's call replaces the anchor's on the 35 contested fragments only
(logs/contested_fragments_review_*). Recovered from the inline script that
produced results/H2_statistical_tests_triangulated.json (2026-11-07).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.statistics import (
    architecture_z_test,
    chi_squared_model_comparison,
    mcnemars_test_prompt_effects,
)

from .accuracy import adjudicate_fragment_outcomes
from .config import (
    CONTESTED_REVIEW_RESOLVED_PATH,
    FRAGMENT_OUTCOMES_TRIANGULATED_PATH,
    H2_TRIANGULATED_RESULTS_PATH,
    SCORED_DB_CORRECTED_INPUT_PATH,
    SCORED_DB_TRIANGULATED_PATH,
)


def _json_default(obj):
    """Serialize numpy scalars (and anything else) for json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def rescore_with_label_overrides(
    database_df: pd.DataFrame,
    label_overrides: dict,
    suffix: str,
) -> pd.DataFrame:
    """
    Re-score every run against expert labels with selected fragments overridden.

    Applies the same accuracy and pass/fail rules as corrected_scoring.py
    (parsing failures and missing classifications are unscoreable and fail),
    against a reference label that equals expert_classification everywhere
    except the fragment_ids in label_overrides.

    Args:
        database_df: RC3-corrected scored database.
        label_overrides: {fragment_id: "sound" | "not_sound"} substitutions.
        suffix: Column suffix for the new label/accuracy/outcome columns.

    Returns:
        Copy of database_df with expert_classification_{suffix},
        classification_accuracy_{suffix}, and run_outcome_{suffix} added.
    """
    db = database_df.copy()
    label_col = f"expert_classification_{suffix}"
    acc_col = f"classification_accuracy_{suffix}"
    out_col = f"run_outcome_{suffix}"

    db[label_col] = db.apply(
        lambda r: label_overrides.get(r["fragment_id"], r["expert_classification"]),
        axis=1,
    )

    def _accuracy(row: pd.Series) -> str | None:
        if row.get("error_flag"):
            return None
        if pd.isna(row["classification_output"]) or pd.isna(row[label_col]):
            return None
        return "correct" if row["classification_output"] == row[label_col] else "incorrect"

    db[acc_col] = db.apply(_accuracy, axis=1)
    db[out_col] = "fail"
    db.loc[db[acc_col] == "correct", out_col] = "pass"
    return db


def run_label_substitution_analysis(
    label_overrides: dict,
    suffix: str,
    scored_db_output_path: Path,
    fragment_outcomes_output_path: Path,
    results_output_path: Path,
    scored_db_input_path: Path = SCORED_DB_CORRECTED_INPUT_PATH,
) -> dict:
    """
    Re-score, adjudicate, and re-run the H2 tests under label overrides.

    Args:
        label_overrides: {fragment_id: "sound" | "not_sound"} substitutions.
        suffix: Tag for new columns (e.g. "triangulated").
        scored_db_output_path: Where to write the re-scored run-level database.
        fragment_outcomes_output_path: Where to write fragment outcomes.
        results_output_path: Where to write the H2 test results JSON.
        scored_db_input_path: RC3-corrected scored database to start from.

    Returns:
        Dict with the three H2 test results.
    """
    db = pd.read_csv(scored_db_input_path)
    db = rescore_with_label_overrides(db, label_overrides, suffix)

    n_overridden = db.loc[db["fragment_id"].isin(label_overrides), "fragment_id"].nunique()
    print(f"Fragments with overridden labels: {n_overridden} (requested {len(label_overrides)})")

    adjudication_input = db.drop(columns=["run_outcome"]).rename(
        columns={f"run_outcome_{suffix}": "run_outcome"}
    )
    fragment_df = adjudicate_fragment_outcomes(adjudication_input, Path(fragment_outcomes_output_path))

    results = {
        "test1_mcnemar": mcnemars_test_prompt_effects(fragment_df),
        "test2_chi_squared": chi_squared_model_comparison(fragment_df),
        "test3_architecture_z": architecture_z_test(fragment_df),
    }

    Path(results_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    db.to_csv(scored_db_output_path, index=False)
    print(f"Saved {results_output_path}\nSaved {scored_db_output_path}")
    return results


def run_reviewer_triangulation(
    resolved_path: Path = CONTESTED_REVIEW_RESOLVED_PATH,
) -> dict:
    """
    External-reviewer secondary analysis on the 35 contested fragments.

    Substitutes the independent reviewer's classification for the anchor's
    on the contested fragments only; all other fragments keep the locked
    gold label.

    Args:
        resolved_path: Merged reviewer key + returned ratings.

    Returns:
        Dict with the three H2 test results.
    """
    resolved = pd.read_csv(resolved_path)
    overrides = dict(zip(resolved["fragment_id"], resolved["reviewer_classification"]))
    return run_label_substitution_analysis(
        label_overrides=overrides,
        suffix="triangulated",
        scored_db_output_path=SCORED_DB_TRIANGULATED_PATH,
        fragment_outcomes_output_path=FRAGMENT_OUTCOMES_TRIANGULATED_PATH,
        results_output_path=H2_TRIANGULATED_RESULTS_PATH,
    )


if __name__ == "__main__":
    run_reviewer_triangulation()
