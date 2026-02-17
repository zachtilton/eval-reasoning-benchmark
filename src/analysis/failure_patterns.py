"""
GPT 5.2 failure pattern synthesis (spec Section 3.4; Appendix H.3).

Corrections over the appendix:
- DOMAIN_LABELS moved to config.py; no longer hard-coded inline.
- Co-occurrence analysis extended to include tertiary_code (appendix only
  checked secondary_code, but spec allows 1-3 failure codes per case).
- Domain filtering uses explicit regex anchoring (r'^F{n}\\.') to avoid
  any false matches against malformed or emergent code strings.
- 'prompt_sensitive_domains' threshold documented as a named constant.
- All output paths use pathlib from config.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import DOMAIN_LABELS, PRIMARY_MODEL, RESULTS_DIR

# Minimum prompt-condition differential (zero_shot_count − few_shot_count)
# to classify a domain as "prompt-sensitive" — used in diagnostic profile.
_PROMPT_SENSITIVE_MIN_DIFF: int = 3
_PROMPT_RESISTANT_MAX_DIFF: int = 1
_PROMPT_RESISTANT_MIN_FAILURES: int = 3


# ---------------------------------------------------------------------------
# H.3: Within-domain analysis
# ---------------------------------------------------------------------------

def _domain_failure_count(
    failure_codes_df: pd.DataFrame,
    domain_num: int,
) -> pd.DataFrame:
    """Return rows whose primary_code belongs to the given domain number."""
    import re
    pattern = re.compile(rf"^F{domain_num}\.")
    return failure_codes_df[
        failure_codes_df["primary_code"].apply(
            lambda c: bool(pattern.match(str(c)))
        )
    ]


def _within_domain_analysis(failure_codes_df: pd.DataFrame) -> dict:
    """
    Compute per-domain failure frequency, checkpoint violations, and
    prompt-condition breakdown (spec Section 3.4 within-domain analysis).
    """
    total_failures = len(failure_codes_df)
    domain_analysis: dict = {}

    for domain_num, domain_label in DOMAIN_LABELS.items():
        domain_rows = _domain_failure_count(failure_codes_df, domain_num)
        domain_count = len(domain_rows)
        domain_pct = domain_count / total_failures * 100 if total_failures > 0 else 0.0

        checkpoint_counts = domain_rows["primary_code"].value_counts().to_dict()
        zero_count = int((domain_rows["prompt_condition"] == "zero_shot").sum())
        few_count  = int((domain_rows["prompt_condition"] == "few_shot").sum())

        domain_analysis[domain_label] = {
            "domain_number": domain_num,
            "failure_count": domain_count,
            "failure_percentage": round(domain_pct, 2),
            "checkpoint_violations": checkpoint_counts,
            "zero_shot_count": zero_count,
            "few_shot_count": few_count,
            "prompt_differential": zero_count - few_count,
        }

    return domain_analysis


# ---------------------------------------------------------------------------
# H.3: Cross-domain co-occurrence
# ---------------------------------------------------------------------------

def _cross_domain_co_occurrence(failure_codes_df: pd.DataFrame) -> dict:
    """
    Build a domain-level co-occurrence matrix from cases with ≥2 codes.

    Extends the appendix which only inspected secondary_code — per spec,
    cases can have up to 3 codes (primary + secondary + tertiary), so all
    pairwise domain combinations are counted.
    """
    co_occurrences: Counter = Counter()

    for _, row in failure_codes_df.iterrows():
        primary = str(row.get("primary_code") or "").split(".")[0]

        code_domains = {primary} if primary else set()

        secondary = str(row.get("secondary_code") or "")
        if secondary and secondary.lower() not in ("none", "nan", ""):
            code_domains.add(secondary.split(".")[0])

        tertiary = str(row.get("tertiary_code") or "")
        if tertiary and tertiary.lower() not in ("none", "nan", ""):
            code_domains.add(tertiary.split(".")[0])

        for pair in combinations_sorted(code_domains):
            co_occurrences[pair] += 1

    return {str(k): v for k, v in co_occurrences.most_common()}


def combinations_sorted(domains: set) -> list[tuple]:
    """Return all 2-element sorted tuples from a set (order-independent pairs)."""
    domain_list = sorted(domains)
    return [
        (domain_list[i], domain_list[j])
        for i in range(len(domain_list))
        for j in range(i + 1, len(domain_list))
    ]


# ---------------------------------------------------------------------------
# H.3: Emergent code analysis
# ---------------------------------------------------------------------------

def _emergent_code_analysis(failure_codes_df: pd.DataFrame) -> dict:
    """Quantify emergent (inductive) vs. deductive code usage."""
    total = len(failure_codes_df)
    emergent  = failure_codes_df[failure_codes_df["code_type"] == "inductive"]
    deductive = failure_codes_df[failure_codes_df["code_type"] == "deductive"]

    emergent_pct = len(emergent) / total * 100 if total > 0 else 0.0
    return {
        "total_emergent": len(emergent),
        "total_deductive": len(deductive),
        "total_failures": total,
        "emergent_percentage": round(emergent_pct, 2),
        "emergent_code_list": emergent["primary_code"].value_counts().to_dict(),
        # < 20% emergent → framework adequately covers failure modes
        "framework_adequacy": (
            "adequate" if emergent_pct < 20.0 else "supplemented"
        ),
    }


# ---------------------------------------------------------------------------
# H.3: Diagnostic profile
# ---------------------------------------------------------------------------

def _diagnostic_profile(
    domain_analysis: dict,
    failure_codes_df: pd.DataFrame,
) -> dict:
    """
    Summarize GPT 5.2's primary failure domains and most common checkpoints.
    """
    ranked = sorted(
        domain_analysis.items(),
        key=lambda x: x[1]["failure_count"],
        reverse=True,
    )

    return {
        "primary_failure_domains": [d[0] for d in ranked[:2]],
        "most_common_checkpoints": (
            failure_codes_df["primary_code"].value_counts().head(5).to_dict()
        ),
        "prompt_sensitive_domains": [
            d[0] for d in ranked
            if abs(d[1]["prompt_differential"]) > _PROMPT_SENSITIVE_MIN_DIFF
        ],
        "prompt_resistant_domains": [
            d[0] for d in ranked
            if (
                abs(d[1]["prompt_differential"]) <= _PROMPT_RESISTANT_MAX_DIFF
                and d[1]["failure_count"] > _PROMPT_RESISTANT_MIN_FAILURES
            )
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_failure_patterns(
    failure_codes_df: pd.DataFrame,
    fragment_df: pd.DataFrame | None = None,
) -> dict:
    """
    Synthesize GPT 5.2 failure patterns for Appendix H.3 (spec Section 3.4).

    Covers: within-domain analysis, cross-domain co-occurrence (including
    tertiary codes), emergent code analysis, and GPT 5.2 diagnostic profile.

    Args:
        failure_codes_df: Failure mode codes DataFrame.
            Required columns: fragment_id, prompt_condition, primary_code,
            secondary_code (optional), tertiary_code (optional),
            domain, code_type.
        fragment_df: Fragment-level outcomes (used for denominators if needed;
                     currently informational — pass None to skip).

    Returns:
        Dict with keys: within_domain, co_occurrence, emergent_codes,
        diagnostic_profile.
    """
    if failure_codes_df.empty:
        print("WARNING: failure_codes_df is empty — nothing to synthesize.")
        return {}

    domain_analysis = _within_domain_analysis(failure_codes_df)
    co_occ = _cross_domain_co_occurrence(failure_codes_df)
    emergent = _emergent_code_analysis(failure_codes_df)
    profile  = _diagnostic_profile(domain_analysis, failure_codes_df)

    synthesis = {
        "within_domain": domain_analysis,
        "co_occurrence": co_occ,
        "emergent_codes": emergent,
        "diagnostic_profile": profile,
    }

    # Print summary
    print(f"\nGPT 5.2 Failure Pattern Synthesis ({len(failure_codes_df)} failures):")
    print(f"  Primary failure domains: {profile['primary_failure_domains']}")
    print(f"  Top checkpoints: {list(profile['most_common_checkpoints'].keys())[:3]}")
    print(f"  Emergent codes: {emergent['total_emergent']} "
          f"({emergent['emergent_percentage']:.1f}%) — "
          f"framework {emergent['framework_adequacy']}")

    return synthesis


def export_failure_patterns(
    synthesis: dict,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """
    Export H.3 failure pattern tables and JSON to the results directory.

    Args:
        synthesis: Output from :func:`synthesize_failure_patterns`.
        output_dir: Directory for output files.
    """
    if not synthesis:
        print("No synthesis data to export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Domain failure distribution table
    domain_rows = [
        {
            "domain": name,
            "domain_number": data["domain_number"],
            "failure_count": data["failure_count"],
            "failure_pct": f"{data['failure_percentage']:.1f}%",
            "zero_shot": data["zero_shot_count"],
            "few_shot": data["few_shot_count"],
            "prompt_differential": data["prompt_differential"],
        }
        for name, data in synthesis["within_domain"].items()
    ]
    pd.DataFrame(domain_rows).sort_values("domain_number").to_csv(
        output_dir / "H3_domain_failure_distribution.csv", index=False
    )

    with (output_dir / "H3_failure_synthesis.json").open("w", encoding="utf-8") as fh:
        json.dump(synthesis, fh, indent=2, default=str)

    print(f"H.3 tables exported to {output_dir}")
