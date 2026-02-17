"""
Analysis package — H.1 through H.5 (spec Sections 3.2-3.6).

Public API surface:

    Performance (H.1):
        generate_performance_matrix, generate_collapsed_summaries,
        export_performance_tables, wilson_confidence_interval

    Statistics (H.2):
        mcnemars_test_prompt_effects, chi_squared_model_comparison,
        architecture_z_test, run_all_statistical_tests

    Failure patterns (H.3):
        synthesize_failure_patterns, export_failure_patterns

    Challenge cases (H.4):
        identify_challenge_cases, export_challenge_cases

    Meta-evaluation (H.5):
        conduct_meta_evaluation, export_meta_evaluation

    Runner:
        run_full_analysis
"""

from .challenge_cases import export_challenge_cases, identify_challenge_cases
from .failure_patterns import export_failure_patterns, synthesize_failure_patterns
from .meta_evaluation import conduct_meta_evaluation, export_meta_evaluation
from .performance import (
    export_performance_tables,
    generate_collapsed_summaries,
    generate_performance_matrix,
    wilson_confidence_interval,
)
from .runner import run_full_analysis
from .statistics import (
    architecture_z_test,
    chi_squared_model_comparison,
    mcnemars_test_prompt_effects,
    run_all_statistical_tests,
)

__all__ = [
    # H.1
    "generate_performance_matrix",
    "generate_collapsed_summaries",
    "export_performance_tables",
    "wilson_confidence_interval",
    # H.2
    "mcnemars_test_prompt_effects",
    "chi_squared_model_comparison",
    "architecture_z_test",
    "run_all_statistical_tests",
    # H.3
    "synthesize_failure_patterns",
    "export_failure_patterns",
    # H.4
    "identify_challenge_cases",
    "export_challenge_cases",
    # H.5
    "conduct_meta_evaluation",
    "export_meta_evaluation",
    # runner
    "run_full_analysis",
]
