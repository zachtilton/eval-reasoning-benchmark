"""
Analysis pipeline runner — orchestrates H.1 through H.5 (spec Sections 3.2-3.6).

Loads all required CSVs from the paths defined in config, runs each analysis
module in order, and exports all results to RESULTS_DIR.

Usage (from project root):
    python -m src.analysis.runner

Or programmatically:
    from src.analysis.runner import run_full_analysis
    results = run_full_analysis()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import (
    FAILURE_CODES_PATH,
    FRAGMENT_OUTCOMES_PATH,
    GOLD_STD_DIR,
    PRIMARY_MODEL,
    RESULTS_DIR,
    SCORED_DB_PATH,
)
from .challenge_cases import export_challenge_cases, identify_challenge_cases
from .failure_patterns import export_failure_patterns, synthesize_failure_patterns
from .meta_evaluation import conduct_meta_evaluation, export_meta_evaluation
from .performance import export_performance_tables
from .statistics import run_all_statistical_tests


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_fragment_outcomes() -> pd.DataFrame:
    """Load fragment-level adjudicated outcomes CSV."""
    if not FRAGMENT_OUTCOMES_PATH.exists():
        print(f"ERROR: fragment outcomes not found at {FRAGMENT_OUTCOMES_PATH}")
        sys.exit(1)
    df = pd.read_csv(FRAGMENT_OUTCOMES_PATH)
    print(f"Loaded fragment outcomes: {len(df)} rows from {FRAGMENT_OUTCOMES_PATH.name}")
    return df


def _load_failure_codes() -> pd.DataFrame:
    """Load GPT 5.2 failure codes CSV (may be empty before coding is complete)."""
    if not FAILURE_CODES_PATH.exists():
        print(f"WARNING: failure codes not found at {FAILURE_CODES_PATH} — "
              "H.3 and H.4 will be skipped.")
        return pd.DataFrame()
    df = pd.read_csv(FAILURE_CODES_PATH)
    print(f"Loaded failure codes: {len(df)} rows from {FAILURE_CODES_PATH.name}")
    return df


def _load_gold_standard() -> pd.DataFrame:
    """Load the locked gold standard CSV (date-stamped filename)."""
    candidates = sorted(GOLD_STD_DIR.glob("gold_standard_locked_*.csv"))
    if not candidates:
        print(f"WARNING: no locked gold standard found in {GOLD_STD_DIR} — "
              "H.5 defensibility check will use empty DataFrame.")
        return pd.DataFrame()
    path = candidates[-1]  # most recent by name sort
    df = pd.read_csv(path)
    print(f"Loaded gold standard: {len(df)} rows from {path.name}")
    return df


def _build_reliability_results(fragment_df: pd.DataFrame) -> dict:
    """
    Attempt to load reliability check results from the scored database.

    If the scored_database.csv exists and has the expected columns, extracts
    RC2 summary stats. RC1 and RC3 are recorded manually; this function
    returns '[PENDING]' for those unless a results JSON already exists.
    """
    reliability: dict = {}

    # Try reading pre-saved reliability JSON (written by scoring pipeline)
    rc_json_path = RESULTS_DIR / "reliability_checks.json"
    if rc_json_path.exists():
        import json
        with rc_json_path.open(encoding="utf-8") as fh:
            reliability = json.load(fh)
        print(f"Loaded reliability results from {rc_json_path.name}")
    else:
        print(f"NOTE: {rc_json_path.name} not found — reliability fields will be [PENDING]")

    return reliability


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_full_analysis(
    output_dir: Path = RESULTS_DIR,
    skip_if_no_failures: bool = False,
) -> dict:
    """
    Execute the full H.1–H.5 analysis pipeline and export all results.

    Args:
        output_dir: Root directory for exported tables and reports.
        skip_if_no_failures: If True, skip H.3/H.4 silently when the
            failure codes file is empty rather than raising.

    Returns:
        Dict with keys: h1_performance, h2_statistics, h3_failures,
        h4_challenge_cases, h5_meta_evaluation.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("FULL ANALYSIS PIPELINE  (H.1 – H.5)")
    print(f"{sep}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    fragment_df      = _load_fragment_outcomes()
    failure_codes_df = _load_failure_codes()
    gold_standard_df = _load_gold_standard()
    reliability      = _build_reliability_results(fragment_df)

    all_results: dict = {}

    # ── H.1: Performance matrix ──
    print(f"\n{sep}")
    print("H.1  PERFORMANCE MATRIX")
    print(sep)
    matrix, by_model, by_prompt, by_arch = export_performance_tables(
        fragment_df, output_dir=output_dir
    )
    all_results["h1_performance"] = {
        "matrix": matrix,
        "by_model": by_model,
        "by_prompt": by_prompt,
        "by_architecture": by_arch,
    }

    # ── H.2: Statistical tests ──
    print(f"\n{sep}")
    print("H.2  STATISTICAL TESTS")
    print(sep)
    h2 = run_all_statistical_tests(fragment_df, output_dir=output_dir)
    all_results["h2_statistics"] = h2

    # ── H.3: Failure pattern synthesis ──
    print(f"\n{sep}")
    print(f"H.3  FAILURE PATTERN SYNTHESIS  ({PRIMARY_MODEL})")
    print(sep)
    if failure_codes_df.empty and skip_if_no_failures:
        print("  Skipped — no failure codes available.")
        all_results["h3_failures"] = {}
    else:
        h3 = synthesize_failure_patterns(failure_codes_df, fragment_df=fragment_df)
        export_failure_patterns(h3, output_dir=output_dir)
        all_results["h3_failures"] = h3

    # ── H.4: Challenge case analysis ──
    print(f"\n{sep}")
    print("H.4  CHALLENGE CASE ANALYSIS")
    print(sep)
    if failure_codes_df.empty and skip_if_no_failures:
        print("  Skipped — no failure codes available.")
        h4: dict = {}
    else:
        h4 = identify_challenge_cases(fragment_df, failure_codes_df)
        export_challenge_cases(h4, output_dir=output_dir)
    all_results["h4_challenge_cases"] = h4

    # ── H.5: Meta-evaluation ──
    print(f"\n{sep}")
    print("H.5  META-EVALUATION AND VALIDITY ASSESSMENT")
    print(sep)
    h5 = conduct_meta_evaluation(
        fragment_df=fragment_df,
        failure_codes_df=failure_codes_df,
        gold_standard_df=gold_standard_df,
        reliability_results=reliability,
        challenge_results=h4,
    )
    export_meta_evaluation(h5, output_dir=output_dir)
    all_results["h5_meta_evaluation"] = h5

    print(f"\n{sep}")
    print(f"ALL ANALYSES COMPLETE — results in {output_dir}")
    print(sep)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_analysis()
