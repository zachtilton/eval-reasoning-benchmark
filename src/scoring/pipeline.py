"""
Orchestrates the 4-step automated scoring pipeline (spec Section 2.4).

This module is the single entry point for scoring.  It loads the response
database, applies each step in order, and saves the scored database.

Pipeline steps:
  Step 1 — Internal Coherence Check (edge_cases + coherence modules)
  Step 2 — Classification Accuracy (accuracy module)
  Step 3 — Run-Level Outcome Assignment (accuracy module)
  Step 4 — Fragment-Level Adjudication (accuracy module)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .accuracy import (
    adjudicate_fragment_outcomes,
    assign_run_outcomes,
    calculate_performance_metrics,
    compare_to_gold_standard,
    export_performance_matrix,
    identify_failures_for_coding,
    load_gold_standard,
    print_performance_summary,
)
from .coherence import validate_all_responses
from .config import (
    FRAGMENT_OUTCOMES_PATH,
    PRIMARY_MODEL,
    RESPONSE_DB_PATH,
    RESPONSES_DIR,
    SCORED_DB_PATH,
)
from .edge_cases import apply_edge_case_handling


def run_scoring_pipeline(
    response_db_path: Path = RESPONSE_DB_PATH,
    scored_db_path: Path = SCORED_DB_PATH,
    fragment_outcomes_path: Path = FRAGMENT_OUTCOMES_PATH,
    primary_model: str = PRIMARY_MODEL,
) -> dict:
    """
    Execute the complete 4-step automated scoring pipeline.

    Loads the response database, applies edge case handling, coherence
    validation, classification accuracy comparison, run-level outcome
    assignment, and fragment-level adjudication in sequence.  Saves
    the scored database and fragment outcomes CSV after each major step.

    Args:
        response_db_path: Path to the response_database.csv from execution.
        scored_db_path: Path to write scored_database.csv output.
        fragment_outcomes_path: Path to write fragment_outcomes.csv output.
        primary_model: Model identifier receiving full diagnostic treatment.

    Returns:
        Dict with summary: n_responses, n_fragments_complete, performance
        metrics, and paths to output files.
    """
    pipeline_start = datetime.now()
    sep = "=" * 70

    print(f"\n{sep}")
    print("SCORING PIPELINE — START")
    print(f"  Response database: {response_db_path}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Load response database
    # ------------------------------------------------------------------
    if not response_db_path.exists():
        raise FileNotFoundError(
            f"Response database not found: {response_db_path}\n"
            "Run the API execution batch first."
        )

    database_df = pd.read_csv(response_db_path)
    print(f"Loaded {len(database_df):,} response records.")

    # ------------------------------------------------------------------
    # Pre-step: edge case detection
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("PRE-STEP: Edge Case Detection")
    print(f"{'—'*50}")
    database_df = apply_edge_case_handling(database_df)

    # ------------------------------------------------------------------
    # Step 1: Coherence validation (tiered)
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("STEP 1: Internal Coherence Validation")
    print(f"{'—'*50}")
    database_df = validate_all_responses(database_df)

    # ------------------------------------------------------------------
    # Step 2: Classification accuracy (coherent responses only)
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("STEP 2: Classification Accuracy vs. Gold Standard")
    print(f"{'—'*50}")
    gold_df = load_gold_standard()
    database_df = compare_to_gold_standard(database_df, gold_df)

    # ------------------------------------------------------------------
    # Step 3: Run-level outcome assignment
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("STEP 3: Run-Level Outcome Assignment")
    print(f"{'—'*50}")
    database_df = assign_run_outcomes(database_df)

    # Save scored database
    scored_db_path.parent.mkdir(parents=True, exist_ok=True)
    database_df.to_csv(scored_db_path, index=False)
    print(f"\nScored database saved: {scored_db_path}")

    # ------------------------------------------------------------------
    # Step 4: Fragment-level adjudication
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("STEP 4: Fragment-Level Adjudication (Majority Rule)")
    print(f"{'—'*50}")
    fragment_df = adjudicate_fragment_outcomes(database_df, fragment_outcomes_path)

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print("PERFORMANCE METRICS")
    print(f"{'—'*50}")
    metrics = calculate_performance_metrics(fragment_df, database_df)
    print_performance_summary(metrics)

    perf_matrix = export_performance_matrix(
        fragment_df,
        output_path=RESPONSES_DIR / "performance_matrix.csv",
    )

    # ------------------------------------------------------------------
    # Identify primary-model failures for coding
    # ------------------------------------------------------------------
    print(f"\n{'—'*50}")
    print(f"FAILURE IDENTIFICATION ({primary_model.upper()})")
    print(f"{'—'*50}")
    failures_df = identify_failures_for_coding(
        fragment_df,
        database_df,
        primary_model=primary_model,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    duration = (datetime.now() - pipeline_start).total_seconds()

    summary = {
        "n_responses": len(database_df),
        "n_excluded": int(database_df.get("exclude_from_analysis", pd.Series(False)).sum()),
        "n_coherent": int((database_df["coherence_final"] == "coherent").sum()),
        "n_incoherent": int((database_df["coherence_final"] == "incoherent").sum()),
        "n_ambiguous": int((database_df["coherence_final"] == "ambiguous").sum()),
        "n_fragment_combinations": len(fragment_df),
        "overall_fragment_pass_rate": metrics["overall"]["fragment_pass_rate"],
        "n_type1_failures_for_coding": len(failures_df),
        "duration_seconds": round(duration, 1),
        "output_files": {
            "scored_database": str(scored_db_path),
            "fragment_outcomes": str(fragment_outcomes_path),
            "performance_matrix": str(RESPONSES_DIR / "performance_matrix.csv"),
        },
    }

    print(f"\n{sep}")
    print("SCORING PIPELINE — COMPLETE")
    print(f"  Duration:                  {duration:.1f}s")
    print(f"  Responses processed:       {summary['n_responses']:,}")
    print(f"  Infrastructure excluded:   {summary['n_excluded']}")
    print(f"  Fragment-level pass rate:  {summary['overall_fragment_pass_rate']:.1%}")
    print(f"  Type 1 failures (coding):  {summary['n_type1_failures_for_coding']}")
    print(f"{sep}\n")

    return summary
