"""
src/scoring — Automated scoring pipeline for the evaluative reasoning benchmark.

Module layout
-------------
config.py      — Indicator lists, LLM prompt, path constants, pipeline params
coherence.py   — Tiered coherence validation (rule-based → LLM), RC2 support
accuracy.py    — Gold standard loading, classification accuracy, outcome
                 assignment, fragment adjudication, performance metrics
edge_cases.py  — Edge case detection and handling (G.6 decision rules)
pipeline.py    — Orchestrates all 4 scoring steps end-to-end

Public interface
----------------
Run the full 4-step pipeline:
    run_scoring_pipeline()

Run individual pipeline steps:
    apply_edge_case_handling(database_df)
    validate_all_responses(database_df)
    compare_to_gold_standard(database_df, gold_df)
    assign_run_outcomes(database_df)
    adjudicate_fragment_outcomes(database_df)
    calculate_performance_metrics(fragment_df, database_df)

RC2 reliability check:
    select_rc2_sample(database_df)
    calculate_rc2_reliability(database_df, manual_reviews_df)

Failure coding preparation:
    identify_failures_for_coding(fragment_df, database_df)
"""

from .pipeline import run_scoring_pipeline

from .coherence import (
    validate_all_responses,
    validate_coherence,
    select_rc2_sample,
    calculate_rc2_reliability,
)
from .accuracy import (
    load_gold_standard,
    compare_to_gold_standard,
    assign_run_outcomes,
    adjudicate_fragment_outcomes,
    calculate_performance_metrics,
    print_performance_summary,
    export_performance_matrix,
    identify_failures_for_coding,
)
from .edge_cases import (
    apply_edge_case_handling,
    detect_edge_case,
)

__all__ = [
    # Pipeline orchestration
    "run_scoring_pipeline",
    # Coherence
    "validate_all_responses",
    "validate_coherence",
    "select_rc2_sample",
    "calculate_rc2_reliability",
    # Accuracy & outcomes
    "load_gold_standard",
    "compare_to_gold_standard",
    "assign_run_outcomes",
    "adjudicate_fragment_outcomes",
    "calculate_performance_metrics",
    "print_performance_summary",
    "export_performance_matrix",
    "identify_failures_for_coding",
    # Edge cases
    "apply_edge_case_handling",
    "detect_edge_case",
]
