"""
Challenge case identification (spec Section 3.5; Appendix H.4).

Corrections over the appendix:
- Criterion 1 inner loop replaced with a precomputed pivot to avoid O(n_failures
  × n_rows) repeated full-DataFrame scans per failure row.
- Overlap keys changed from fragile f"{frag_id}_{prompt}" strings to
  (fragment_id, prompt_condition) tuples, immune to underscore collisions.
- DataFrame `.empty` used instead of `len(...) > 0` for empty-check consistency.
- Threshold constants imported from config rather than hard-coded inline.
- All output paths use pathlib from config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import (
    COMPARISON_MODEL_THRESHOLD,
    PRIMARY_MODEL,
    RECURRING_FAILURE_THRESHOLD,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# H.4: Criterion 1 — Systematic Disagreement
# ---------------------------------------------------------------------------

def _build_comparison_fail_lookup(
    fragment_df: pd.DataFrame,
    primary_model: str,
) -> dict[tuple, list[str]]:
    """
    Precompute, for every (fragment_id, prompt_condition) pair, the list of
    *comparison* models that produced a 'fail' outcome.

    Returns a dict mapping (fragment_id, prompt_condition) → [model, ...].
    This allows O(1) lookup per primary failure instead of O(n) DataFrame
    filters inside the loop (appendix bug).
    """
    comparison_fails = fragment_df[
        (fragment_df["model_family"] != primary_model)
        & (fragment_df["fragment_outcome"] == "fail")
    ][["fragment_id", "prompt_condition", "model_family"]]

    lookup: dict[tuple, list[str]] = {}
    for row in comparison_fails.itertuples(index=False):
        key = (row.fragment_id, row.prompt_condition)
        lookup.setdefault(key, []).append(row.model_family)

    return lookup


def _criterion_1_systematic_disagreement(
    fragment_df: pd.DataFrame,
    primary_model: str,
    comparison_model_threshold: int,
) -> dict:
    """
    Criterion 1: GPT 5.2 fails AND ≥threshold comparison models also fail
    on the same fragment-prompt combination.
    """
    # Precompute comparison failure lookup — avoids repeated full-DF scans
    fail_lookup = _build_comparison_fail_lookup(fragment_df, primary_model)

    primary_failures = fragment_df[
        (fragment_df["model_family"] == primary_model)
        & (fragment_df["fragment_outcome"] == "fail")
    ][["fragment_id", "prompt_condition"]]

    systematic_cases: list[dict] = []
    for row in primary_failures.itertuples(index=False):
        key = (row.fragment_id, row.prompt_condition)
        failing_others = fail_lookup.get(key, [])
        n_others = len(failing_others)

        if n_others >= comparison_model_threshold:
            systematic_cases.append({
                "fragment_id": row.fragment_id,
                "prompt_condition": row.prompt_condition,
                "n_comparison_failures": n_others,
                "failing_comparison_models": failing_others,
                "total_models_failing": n_others + 1,  # +1 for primary
            })

    return {
        "label": "Systematic Disagreement",
        "description": (
            f"{primary_model} fails AND ≥{comparison_model_threshold} "
            "comparison models also fail"
        ),
        "n_cases": len(systematic_cases),
        "cases": pd.DataFrame(systematic_cases) if systematic_cases else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# H.4: Criterion 2 — Recurring Failure Mode
# ---------------------------------------------------------------------------

def _criterion_2_recurring_failure_mode(
    failure_codes_df: pd.DataFrame,
    primary_model: str,
    recurring_failure_threshold: float,
) -> dict:
    """
    Criterion 2: A single failure code accounts for ≥threshold fraction
    of all primary-model Type 1 failure cases.
    """
    total_failures = len(failure_codes_df)
    if total_failures == 0:
        return {
            "label": "Recurring Failure Mode",
            "description": (
                f"Single code ≥{recurring_failure_threshold * 100:.0f}% "
                f"of {primary_model} failures"
            ),
            "n_codes": 0,
            "codes": pd.DataFrame(),
        }

    code_counts = failure_codes_df["primary_code"].value_counts()
    qualifying = code_counts[code_counts / total_failures >= recurring_failure_threshold]

    recurring_cases: list[dict] = []
    for code, count in qualifying.items():
        pct = count / total_failures * 100
        affected = (
            failure_codes_df[failure_codes_df["primary_code"] == code]["fragment_id"]
            .tolist()
        )
        recurring_cases.append({
            "failure_code": code,
            "count": int(count),
            "percentage": round(pct, 2),
            "affected_fragments": affected,
        })

    return {
        "label": "Recurring Failure Mode",
        "description": (
            f"Single code ≥{recurring_failure_threshold * 100:.0f}% "
            f"of {primary_model} failures"
        ),
        "n_codes": len(recurring_cases),
        "codes": pd.DataFrame(recurring_cases) if recurring_cases else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# H.4: Overlap analysis
# ---------------------------------------------------------------------------

def _overlap_analysis(
    c1_result: dict,
    failure_codes_df: pd.DataFrame,
) -> dict:
    """
    Compare which fragment-prompt combinations appear in both criteria.

    Uses tuple keys (fragment_id, prompt_condition) instead of fragile
    underscore-joined strings (appendix bug — underscores can appear in IDs).
    """
    # Criterion 1 keys
    if not c1_result["cases"].empty:
        c1_keys: set[tuple] = {
            (row.fragment_id, row.prompt_condition)
            for row in c1_result["cases"].itertuples(index=False)
        }
    else:
        c1_keys = set()

    # Criterion 2 keys — build from failure_codes_df matching any recurring code
    c2_keys: set[tuple] = set()
    if not c1_result["cases"].empty:
        # We need c2 codes from the parent function; pass through failure_codes_df
        # directly and check what fraction any code covers (≥ threshold)
        total = len(failure_codes_df)
        if total > 0:
            from .config import RECURRING_FAILURE_THRESHOLD as _threshold
            counts = failure_codes_df["primary_code"].value_counts()
            qualifying_codes = set(
                counts[counts / total >= _threshold].index.tolist()
            )
            sub = failure_codes_df[
                failure_codes_df["primary_code"].isin(qualifying_codes)
            ]
            for row in sub.itertuples(index=False):
                c2_keys.add((row.fragment_id, row.prompt_condition))

    overlap   = c1_keys & c2_keys
    c1_only   = c1_keys - c2_keys
    c2_only   = c2_keys - c1_keys

    high = len(overlap) > max(len(c1_only), len(c2_only), 0)

    return {
        "n_overlap": len(overlap),
        "n_criterion_1_only": len(c1_only),
        "n_criterion_2_only": len(c2_only),
        "overlap_keys": [list(k) for k in sorted(overlap)],
        "interpretation": "high_overlap" if high else "low_overlap",
        "narrative": (
            "High overlap suggests genuinely difficult cases that challenge "
            "models broadly."
            if high else
            "Low overlap suggests orthogonal difficulty dimensions: systematic "
            "disagreement and recurring weaknesses identify distinct fragment "
            "characteristics."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_challenge_cases(
    fragment_df: pd.DataFrame,
    failure_codes_df: pd.DataFrame,
    primary_model: str = PRIMARY_MODEL,
    comparison_model_threshold: int = COMPARISON_MODEL_THRESHOLD,
    recurring_failure_threshold: float = RECURRING_FAILURE_THRESHOLD,
) -> dict:
    """
    Identify challenge cases meeting either of two criteria (spec Section 3.5).

    Criterion 1 (Systematic Disagreement): primary model fails AND ≥threshold
    comparison models also fail on the same fragment-prompt combination.

    Criterion 2 (Recurring Failure Mode): a single failure code accounts for
    ≥threshold fraction of all primary-model Type 1 failure cases.

    Args:
        fragment_df: Fragment-level outcomes for all models.
            Required columns: fragment_id, prompt_condition, model_family,
            fragment_outcome.
        failure_codes_df: GPT 5.2 Type 1 failure codes.
            Required columns: fragment_id, prompt_condition, primary_code.
        primary_model: Primary diagnostic model identifier.
        comparison_model_threshold: Min comparison model failures (Criterion 1).
        recurring_failure_threshold: Min proportion for Criterion 2.

    Returns:
        Dict with keys: criterion_1, criterion_2, overlap, summary.
    """
    c1 = _criterion_1_systematic_disagreement(
        fragment_df, primary_model, comparison_model_threshold
    )
    c2 = _criterion_2_recurring_failure_mode(
        failure_codes_df, primary_model, recurring_failure_threshold
    )
    overlap = _overlap_analysis(c1, failure_codes_df)

    # Total unique challenge fragments (union of both criteria)
    c1_frags: set[str] = set()
    if not c1["cases"].empty:
        c1_frags = set(c1["cases"]["fragment_id"].tolist())
    c2_frags: set[str] = set()
    if not c2["codes"].empty:
        for affected in c2["codes"]["affected_fragments"]:
            c2_frags.update(affected)

    total_unique = len(c1_frags | c2_frags)

    print(f"\nH.4 Challenge Cases:")
    print(f"  Criterion 1 (Systematic Disagreement): {c1['n_cases']} cases")
    print(f"  Criterion 2 (Recurring Failure Mode):  {c2['n_codes']} codes")
    print(f"  Overlap: {overlap['n_overlap']} ({overlap['interpretation']})")
    print(f"  Total unique challenge fragments: {total_unique}")

    return {
        "criterion_1": c1,
        "criterion_2": c2,
        "overlap": overlap,
        "summary": {
            "n_systematic_disagreement_cases": c1["n_cases"],
            "n_recurring_failure_codes": c2["n_codes"],
            "n_overlap": overlap["n_overlap"],
            "total_unique_challenge_fragments": total_unique,
        },
    }


def export_challenge_cases(
    results: dict,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """
    Export H.4 challenge case tables and JSON to the results directory.

    Args:
        results: Output from :func:`identify_challenge_cases`.
        output_dir: Directory for output files.
    """
    if not results:
        print("No challenge case data to export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Criterion 1 cases CSV
    c1_cases = results["criterion_1"]["cases"]
    if not c1_cases.empty:
        # Serialize list column for CSV
        export_df = c1_cases.copy()
        export_df["failing_comparison_models"] = export_df[
            "failing_comparison_models"
        ].apply(lambda x: "|".join(x) if isinstance(x, list) else str(x))
        export_df.to_csv(
            output_dir / "H4_systematic_disagreement_cases.csv", index=False
        )

    # Criterion 2 codes CSV
    c2_codes = results["criterion_2"]["codes"]
    if not c2_codes.empty:
        export_codes = c2_codes.copy()
        export_codes["affected_fragments"] = export_codes[
            "affected_fragments"
        ].apply(lambda x: "|".join(str(f) for f in x) if isinstance(x, list) else str(x))
        export_codes.to_csv(
            output_dir / "H4_recurring_failure_modes.csv", index=False
        )

    # Overlap + summary JSON
    json_payload = {
        "overlap": results["overlap"],
        "summary": results["summary"],
        "criterion_1_description": results["criterion_1"]["description"],
        "criterion_2_description": results["criterion_2"]["description"],
    }
    with (output_dir / "H4_challenge_cases.json").open("w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2, default=str)

    print(f"H.4 tables exported to {output_dir}")
