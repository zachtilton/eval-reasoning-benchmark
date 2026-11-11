"""
RC2-corrected re-scoring: provisional-final results with the coherence
gate removed from Step 2 (spec Section 2.4).

Background: RC2's blind stratified human review (2026-11-05) found that
the automated coherence pipeline's "incoherent" and "ambiguous" calls had
0% agreement with human judgment in every tested sub-stratum --
incoherent_confident (64/64 overridden, including the 7 cases where Tier 2
independently agreed with Tier 1), and all three ambiguous sub-groups
(153/153 overridden, including a full census of the hardest 64 three-way
splits). Only the coherent_confident bucket held up (94/94). No surviving
sub-component of the "not coherent" determination could be trusted as a
scoring gate.

This module does NOT modify the original scored_database.csv /
fragment_outcomes.csv / performance_matrix.csv -- those remain the
as-designed, pre-RC2 historical results. It re-runs Steps 2-4 from the
already-scored database (Step 1 coherence columns are kept as a retained
diagnostic, not deleted) with one change: classification accuracy is
computed for every response with a usable classification, regardless of
coherence_final. Genuinely unusable rows (no classification_output to
compare, or a parsing error_flag) remain excluded exactly as before --
RC2 only impugns the coherence *instrument*, not those independent
exclusions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .accuracy import (
    _wilson_ci,
    adjudicate_fragment_outcomes,
    load_gold_standard,
)
from .config import (
    FRAGMENT_OUTCOMES_CORRECTED_PATH,
    PERFORMANCE_MATRIX_CORRECTED_PATH,
    PRIMARY_MODEL,
    RESPONSES_DIR,
    SCORED_DB_CORRECTED_PATH,
    SCORED_DB_PATH,
)


def compare_to_gold_standard_corrected(
    database_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare model classifications to the gold standard for every response.

    Unlike the original compare_to_gold_standard, this does not bypass
    rows where coherence_final != "coherent" -- RC2 found that bypass
    rule's input (the coherence pipeline) unreliable in the "not coherent"
    direction. A response is scored whenever it has a usable
    classification_output and did not fail to parse (error_flag).

    Args:
        database_df: Scored database (coherence columns retained but no
            longer used as a gate).
        gold_df: Gold standard DataFrame from load_gold_standard().

    Returns:
        DataFrame with classification_accuracy_corrected column added.
    """
    merged = database_df.merge(
        gold_df[["fragment_id", "expert_classification"]],
        on="fragment_id",
        how="left",
        suffixes=("", "_gold"),
    )
    # scored_database.csv already carries expert_classification from the
    # original run; prefer the freshly-merged gold copy for this step and
    # drop the duplicate to avoid a stale _gold column lingering.
    if "expert_classification_gold" in merged.columns:
        merged["expert_classification"] = merged["expert_classification_gold"]
        merged = merged.drop(columns=["expert_classification_gold"])

    def _accuracy(row: pd.Series) -> str | None:
        if row.get("error_flag"):
            return None  # parsing failure — nothing to score, unrelated to RC2
        if pd.isna(row["classification_output"]) or pd.isna(row["expert_classification"]):
            return None  # no usable classification to compare
        if row["classification_output"] == row["expert_classification"]:
            return "correct"
        return "incorrect"

    merged["classification_accuracy_corrected"] = merged.apply(_accuracy, axis=1)
    return merged


def assign_run_outcomes_corrected(database_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign corrected Pass/Fail using accuracy alone (coherence gate removed).

    Decision rule: Pass = correct classification. Fail = incorrect, or no
    usable classification/parsing failure. coherence_final is retained as
    a column for reference but no longer determines the outcome.

    failure_type_corrected is "type_1" for every failure (coherent/incoherent
    no longer distinguishes failures once the coherence gate is removed --
    every failure here is "wrong answer," the category the original design
    called eligible for qualitative coding).

    Args:
        database_df: Database with classification_accuracy_corrected populated.

    Returns:
        DataFrame with run_outcome_corrected and failure_type_corrected added.
    """
    database_df = database_df.copy()

    correct = database_df["classification_accuracy_corrected"] == "correct"
    unscoreable = database_df["classification_accuracy_corrected"].isna()

    database_df["run_outcome_corrected"] = "fail"
    database_df.loc[correct, "run_outcome_corrected"] = "pass"

    database_df["failure_type_corrected"] = None
    database_df.loc[~correct & ~unscoreable, "failure_type_corrected"] = "type_1"
    database_df.loc[unscoreable, "failure_type_corrected"] = "unscoreable"

    pass_n = (database_df["run_outcome_corrected"] == "pass").sum()
    fail_n = (database_df["run_outcome_corrected"] == "fail").sum()
    unscoreable_n = unscoreable.sum()
    total = len(database_df)

    print(f"\nRun-Level Outcomes, CORRECTED ({total:,} runs):")
    print(f"  Pass:         {pass_n:,} ({pass_n/total:.1%})")
    print(f"  Fail:         {fail_n:,} ({fail_n/total:.1%})")
    print(f"  Unscoreable:  {unscoreable_n} (parsing failure or missing classification)")

    return database_df


def run_corrected_scoring(
    scored_db_path: Path = SCORED_DB_PATH,
    output_db_path: Path = SCORED_DB_CORRECTED_PATH,
    fragment_outcomes_path: Path = FRAGMENT_OUTCOMES_CORRECTED_PATH,
    primary_model: str = PRIMARY_MODEL,
) -> dict:
    """
    Re-run Steps 2-4 from the existing scored database with the RC2-informed
    accuracy-only outcome rule, writing to separate *_corrected files.

    Does not touch scored_database.csv / fragment_outcomes.csv /
    performance_matrix.csv -- those remain the original, pre-RC2 results.

    Args:
        scored_db_path: Path to the existing (Step-1-complete) scored database.
        output_db_path: Path to write the corrected scored database.
        fragment_outcomes_path: Path to write corrected fragment outcomes.
        primary_model: Model identifier receiving full diagnostic treatment.

    Returns:
        Dict summary with before/after pass-rate comparison.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("RC2-CORRECTED RE-SCORING — START")
    print(f"  Source (original scored database): {scored_db_path}")
    print(f"{sep}\n")

    database_df = pd.read_csv(scored_db_path)
    original_run_outcome = database_df["run_outcome"].copy()

    gold_df = load_gold_standard()

    print(f"\n{'—'*50}")
    print("STEP 2 (corrected): Classification Accuracy, Coherence Gate Removed")
    print(f"{'—'*50}")
    database_df = compare_to_gold_standard_corrected(database_df, gold_df)

    print(f"\n{'—'*50}")
    print("STEP 3 (corrected): Run-Level Outcome Assignment")
    print(f"{'—'*50}")
    database_df = assign_run_outcomes_corrected(database_df)

    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    database_df.to_csv(output_db_path, index=False)
    print(f"\nCorrected scored database saved: {output_db_path}")

    print(f"\n{'—'*50}")
    print("STEP 4 (corrected): Fragment-Level Adjudication (Majority Rule)")
    print(f"{'—'*50}")
    # adjudicate_fragment_outcomes groups on run_outcome -- feed it the
    # corrected column under the name it expects, scoped to a throwaway copy.
    adjudication_input = database_df.drop(columns=["run_outcome"]).rename(
        columns={"run_outcome_corrected": "run_outcome"}
    )
    fragment_df = adjudicate_fragment_outcomes(adjudication_input, fragment_outcomes_path)

    # ------------------------------------------------------------------
    # Before/after comparison
    # ------------------------------------------------------------------
    old_pass_rate = (original_run_outcome == "pass").mean()
    new_pass_rate = (database_df["run_outcome_corrected"] == "pass").mean()

    print(f"\n{'—'*50}")
    print("BEFORE / AFTER — RUN-LEVEL PASS RATE")
    print(f"{'—'*50}")
    print(f"  Original (coherence-gated):   {old_pass_rate:.1%}")
    print(f"  Corrected (accuracy-only):    {new_pass_rate:.1%}")
    print(f"  Delta:                        {new_pass_rate - old_pass_rate:+.1%}")

    print(f"\n{'—'*50}")
    print("BEFORE / AFTER — FRAGMENT PASS RATE BY MODEL")
    print(f"{'—'*50}")
    original_fragment = pd.read_csv(
        RESPONSES_DIR / "fragment_outcomes.csv"
    ) if (RESPONSES_DIR / "fragment_outcomes.csv").exists() else None

    by_model_comparison: dict = {}
    for model in sorted(fragment_df["model_family"].unique()):
        new_rate = (
            fragment_df.loc[fragment_df["model_family"] == model, "fragment_outcome"] == "pass"
        ).mean()
        old_rate = None
        if original_fragment is not None:
            old_sub = original_fragment[original_fragment["model_family"] == model]
            if len(old_sub):
                old_rate = (old_sub["fragment_outcome"] == "pass").mean()
        by_model_comparison[model] = {"original": old_rate, "corrected": new_rate}
        old_str = f"{old_rate:.1%}" if old_rate is not None else "n/a"
        print(f"  {model:<28} original={old_str:>7}  corrected={new_rate:.1%}")

    perf_matrix = (
        fragment_df.groupby(["model_family", "prompt_condition"])
        .agg(
            fragment_pass_rate=("fragment_outcome", lambda x: (x == "pass").mean()),
            unanimous_agreement_rate=("unanimous_agreement", "mean"),
            n_fragments=("fragment_id", "count"),
        )
        .reset_index()
    )
    ci_lower, ci_upper = zip(
        *perf_matrix.apply(
            lambda row: _wilson_ci(row["fragment_pass_rate"], row["n_fragments"]),
            axis=1,
        )
    )
    perf_matrix["ci_lower_95"] = ci_lower
    perf_matrix["ci_upper_95"] = ci_upper
    perf_matrix = perf_matrix.sort_values(["model_family", "prompt_condition"])

    PERFORMANCE_MATRIX_CORRECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    perf_matrix.to_csv(PERFORMANCE_MATRIX_CORRECTED_PATH, index=False)
    print(f"\nCorrected performance matrix saved: {PERFORMANCE_MATRIX_CORRECTED_PATH}")

    summary = {
        "n_responses": len(database_df),
        "original_run_pass_rate": float(old_pass_rate),
        "corrected_run_pass_rate": float(new_pass_rate),
        "by_model_fragment_pass_rate": by_model_comparison,
        "output_files": {
            "scored_database_corrected": str(output_db_path),
            "fragment_outcomes_corrected": str(fragment_outcomes_path),
            "performance_matrix_corrected": str(PERFORMANCE_MATRIX_CORRECTED_PATH),
        },
    }

    print(f"\n{sep}")
    print("RC2-CORRECTED RE-SCORING — COMPLETE")
    print(f"{sep}\n")

    return summary
