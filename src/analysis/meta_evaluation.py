"""
Meta-evaluation and validity assessment (spec Section 3.6; Appendix H.5).

Corrections over the appendix:
- Reliability key extraction uses a helper that handles both flat
  {rc1_kappa: x} and nested {rc1: {kappa: x}} dict formats, rather than
  hardcoding one layout (appendix assumed flat keys).
- c1_cases empty-check uses DataFrame.empty instead of len(...) > 0.
- Markdown export uses pathlib; output_dir default comes from config.
- boundary_case_flag coerced to int before summing (handles bool columns).
- All output paths use pathlib from config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import KAPPA_TARGET, RESULTS_DIR


# ---------------------------------------------------------------------------
# Helper: extract reliability kappas from varied dict structures
# ---------------------------------------------------------------------------

def _extract_reliability(
    reliability_results: dict,
    check: str,
) -> tuple[object, object]:
    """
    Extract (kappa, met_target) for a reliability check from a results dict.

    Supports two layouts produced by the scoring pipeline:
      - Flat:  {'rc1_kappa': 0.85, 'rc1_met': True, ...}
      - Nested: {'rc1': {'kappa': 0.85, 'met_target': True}, ...}

    Returns ('[PENDING]', '[PENDING]') when the key is absent.
    """
    # Try nested first
    sub = reliability_results.get(check)
    if isinstance(sub, dict):
        kappa = sub.get("kappa", sub.get("achieved_kappa", "[PENDING]"))
        met   = sub.get("met_target", sub.get("met", "[PENDING]"))
        return kappa, met

    # Try flat keys
    kappa = reliability_results.get(f"{check}_kappa", "[PENDING]")
    met   = reliability_results.get(f"{check}_met", "[PENDING]")
    return kappa, met


# ---------------------------------------------------------------------------
# H.5: Construct validity triangulation
# ---------------------------------------------------------------------------

def _construct_validity() -> dict:
    """
    Return the structured set of theoretical validity assessment questions.

    These are documented predictions that must be compared against H.3 / H.4
    findings after data collection; they are not computed automatically.
    """
    return {
        "assessment_questions": [
            {
                "question": (
                    "Does GPT 5.2 struggle more with Synthesis & Integration "
                    "(Domain 4) than Evidence Identification (Domain 2)?"
                ),
                "theoretical_prediction": (
                    "Yes — synthesis requires higher-order integration that "
                    "should be harder for models than pattern-matching evidence."
                ),
                "data_source": "H.3 domain failure distribution",
                "finding": "[TO BE COMPLETED AFTER DATA COLLECTION]",
            },
            {
                "question": (
                    "Does few-shot calibration improve performance on domains "
                    "emphasizing tacit judgment (Domain 6: Qualifications & "
                    "Transparency)?"
                ),
                "theoretical_prediction": (
                    "Yes — calibration examples should help models learn "
                    "implicit standards for qualification and transparency."
                ),
                "data_source": "H.3 prompt differential by domain",
                "finding": "[TO BE COMPLETED AFTER DATA COLLECTION]",
            },
            {
                "question": (
                    "Do challenge cases cluster around theoretically complex "
                    "evaluation criteria (e.g., sustainability vs. relevance)?"
                ),
                "theoretical_prediction": (
                    "Yes — criteria requiring longer causal chains and "
                    "counterfactual reasoning should produce more failures."
                ),
                "data_source": "H.4 challenge case characteristics",
                "finding": "[TO BE COMPLETED AFTER DATA COLLECTION]",
            },
        ]
    }


# ---------------------------------------------------------------------------
# H.5: Gold standard defensibility
# ---------------------------------------------------------------------------

def _gold_standard_defensibility(
    gold_standard_df: pd.DataFrame,
    challenge_results: dict,
) -> dict:
    """
    Identify fragments that warrant analytical reflection against gold standard.

    Note: the gold standard remains LOCKED. This section is analytical
    documentation, not revision.
    """
    # Fragments where multiple models systematically disagreed (Criterion 1)
    review_fragments: list = []
    c1 = challenge_results.get("criterion_1", {})
    c1_cases = c1.get("cases", pd.DataFrame())
    if isinstance(c1_cases, pd.DataFrame) and not c1_cases.empty:
        review_fragments = c1_cases["fragment_id"].unique().tolist()

    # Boundary cases in gold standard
    boundary_count: int = 0
    if "boundary_case_flag" in gold_standard_df.columns:
        boundary_count = int(gold_standard_df["boundary_case_flag"].astype(int).sum())

    return {
        "fragments_for_re_review": review_fragments,
        "n_fragments_for_re_review": len(review_fragments),
        "boundary_cases_in_gold_standard": boundary_count,
        "note": (
            "Gold standard remains LOCKED. Re-review is analytical reflection, "
            "not revision. Document any cases where model responses reveal "
            "ambiguity in original expert classifications."
        ),
        "review_template": {
            "fragment_id": "[ID]",
            "original_classification": "[sound/not_sound]",
            "n_models_disagreeing": "[count]",
            "model_rationale_summary": "[key arguments from disagreeing models]",
            "defensibility_assessment": "[defensible/ambiguous/acknowledged_limitation]",
            "notes": "[analytical reflection]",
        },
    }


# ---------------------------------------------------------------------------
# H.5: Reliability synthesis
# ---------------------------------------------------------------------------

def _reliability_synthesis(reliability_results: dict) -> dict:
    """
    Summarize RC1, RC2, RC3 reliability checks against κ ≥ 0.80 target.
    """
    checks = []
    specs = [
        ("rc1", "Expert temporal consistency"),
        ("rc2", "Automated coherence validation"),
        ("rc3", "Failure mode coding temporal consistency"),
    ]
    all_met: list[bool | str] = []

    for check_key, description in specs:
        kappa, met = _extract_reliability(reliability_results, check_key)
        if isinstance(kappa, float):
            kappa_display = round(kappa, 4)
        else:
            kappa_display = kappa

        if isinstance(met, bool):
            all_met.append(met)

        checks.append({
            "label": check_key.upper(),
            "description": description,
            "target": f"κ ≥ {KAPPA_TARGET}",
            "achieved_kappa": kappa_display,
            "met_target": met,
        })

    # Overall assessment — only derive from booleans when ALL checks returned one.
    # If any check is still pending, the overall verdict is also pending.
    if all_met and len(all_met) == len(specs):
        if all(all_met):
            overall = "All reliability targets met (κ ≥ 0.80 for RC1, RC2, RC3)."
        elif any(all_met):
            # zip against specs in order; only bool entries are counted
            failed = [
                s[0].upper() for s, m in zip(specs, all_met)
                if isinstance(m, bool) and not m
            ]
            overall = (
                f"Partial reliability: {', '.join(failed)} below target. "
                "See individual checks for details."
            )
        else:
            overall = (
                "No reliability targets met — review coherence validation "
                "pipeline and coding protocol before proceeding."
            )
    else:
        overall = "[TO BE COMPLETED: All targets met / Partial / Limitations noted]"

    return {
        "checks": checks,
        "overall_assessment": overall,
    }


# ---------------------------------------------------------------------------
# H.5: Methodological limitations
# ---------------------------------------------------------------------------

def _methodological_limitations() -> list[dict]:
    """Return the structured table of pre-specified methodological limitations."""
    return [
        {
            "threat": "Fragment selection bias",
            "description": (
                "Conclusion sections may not represent the full range of "
                "evaluative reasoning in UN evaluation reports."
            ),
            "mitigation": (
                "Random sampling from large UN corpus; ecological validity "
                "argument for conclusion sections as key reasoning sites."
            ),
            "residual_risk": (
                "Moderate — UN evaluation style may differ from other "
                "institutional or national evaluation contexts."
            ),
        },
        {
            "threat": "Corpus scope",
            "description": (
                "Generalizability limited to UN evaluation contexts and their "
                "conventions for reporting evaluative conclusions."
            ),
            "mitigation": (
                "Explicit scope limitation documented; UN corpus provides a "
                "standardized, professionally produced baseline."
            ),
            "residual_risk": (
                "Acknowledged — benchmark applies to meta-evaluation of "
                "written reports, not verbal or interactive reasoning."
            ),
        },
        {
            "threat": "Single-expert judgment",
            "description": (
                "Gold standard reflects one expert rather than an inter-rater "
                "consensus panel."
            ),
            "mitigation": (
                "Pre-commitment protocol; temporal consistency check (RC1); "
                "explicit written rationales for each classification."
            ),
            "residual_risk": (
                "Moderate — study tests replication of individual expert "
                "reasoning, not consensus, which is an acknowledged trade-off."
            ),
        },
        {
            "threat": "Prompt engineering constraints",
            "description": (
                "Single prompt formulation per condition; exactly 4 few-shot "
                "calibration examples (2 sound, 2 not sound)."
            ),
            "mitigation": (
                "Fixed prompts enable controlled comparison; prompt text fully "
                "documented in appendices for reproducibility."
            ),
            "residual_risk": (
                "Low — alternative prompt variants are a future research "
                "direction, not a threat to internal validity."
            ),
        },
        {
            "threat": "Single-model deep diagnostic",
            "description": (
                "Detailed failure mode coding (H.3) limited to GPT 5.2 "
                "(primary diagnostic model)."
            ),
            "mitigation": (
                "Comparative pass-rate metrics across all 6 models; "
                "architecture-level comparisons (H.2 Test 3) provide "
                "aggregate cross-model picture."
            ),
            "residual_risk": (
                "Acknowledged — cross-architecture failure-mode profiles "
                "remain a recommended extension."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def conduct_meta_evaluation(
    fragment_df: pd.DataFrame,
    failure_codes_df: pd.DataFrame,
    gold_standard_df: pd.DataFrame,
    reliability_results: dict,
    challenge_results: dict,
) -> dict:
    """
    Meta-evaluation and validity assessment (spec Section 3.6 / Appendix H.5).

    Produces a structured report covering construct validity triangulation,
    gold standard defensibility, reliability synthesis, and methodological
    limitations. Many assessments are documented structured prompts requiring
    qualitative judgment after data collection.

    Args:
        fragment_df: Fragment-level outcomes (informational; checked for shape).
        failure_codes_df: GPT 5.2 Type 1 failure codes.
        gold_standard_df: Gold standard DataFrame; optionally includes a
            'boundary_case_flag' column.
        reliability_results: Dict with RC1/RC2/RC3 kappa results.
            Accepts flat {'rc1_kappa': x} or nested {'rc1': {'kappa': x}}.
        challenge_results: Output from :func:`challenge_cases.identify_challenge_cases`.

    Returns:
        Dict with keys: construct_validity, gold_standard_defensibility,
        reliability_synthesis, limitations, data_summary.
    """
    meta: dict = {}

    meta["construct_validity"] = _construct_validity()
    meta["gold_standard_defensibility"] = _gold_standard_defensibility(
        gold_standard_df, challenge_results
    )
    meta["reliability_synthesis"] = _reliability_synthesis(reliability_results)
    meta["limitations"] = _methodological_limitations()

    # Descriptive data summary (not analytic — purely informational)
    meta["data_summary"] = {
        "n_fragments_total": int(
            fragment_df["fragment_id"].nunique() if not fragment_df.empty else 0
        ),
        "n_model_prompt_combinations": int(len(fragment_df)) if not fragment_df.empty else 0,
        "n_type1_failures_coded": int(len(failure_codes_df)),
        "n_gold_standard_items": int(len(gold_standard_df)),
        "n_challenge_cases_criterion_1": int(
            challenge_results.get("summary", {}).get("n_systematic_disagreement_cases", 0)
        ),
        "n_challenge_cases_criterion_2_codes": int(
            challenge_results.get("summary", {}).get("n_recurring_failure_codes", 0)
        ),
    }

    print("\nH.5 Meta-Evaluation complete.")
    print(f"  Reliability overall: {meta['reliability_synthesis']['overall_assessment']}")
    print(f"  Fragments for re-review: "
          f"{meta['gold_standard_defensibility']['n_fragments_for_re_review']}")

    return meta


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_meta_evaluation(
    meta: dict,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """
    Export H.5 meta-evaluation report as JSON and Markdown.

    Args:
        meta: Output from :func:`conduct_meta_evaluation`.
        output_dir: Directory for output files.
    """
    if not meta:
        print("No meta-evaluation data to export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with (output_dir / "H5_meta_evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # Markdown narrative
    lines: list[str] = [
        "# Appendix H.5: Meta-Evaluation and Validity Assessment\n",
    ]

    lines.append("## 1. Construct Validity Triangulation\n")
    for q in meta["construct_validity"]["assessment_questions"]:
        lines.append(f"**Q:** {q['question']}\n")
        lines.append(f"- **Prediction:** {q['theoretical_prediction']}")
        lines.append(f"- **Data source:** {q['data_source']}")
        lines.append(f"- **Finding:** {q['finding']}\n")

    lines.append("## 2. Gold Standard Defensibility\n")
    gsd = meta["gold_standard_defensibility"]
    lines.append(f"- Fragments flagged for analytical reflection: "
                 f"{gsd['n_fragments_for_re_review']}")
    lines.append(f"- Boundary cases in gold standard: "
                 f"{gsd['boundary_cases_in_gold_standard']}")
    lines.append(f"\n> {gsd['note']}\n")

    lines.append("## 3. Reliability Synthesis\n")
    lines.append("| Check | Description | Target | Achieved κ | Met? |")
    lines.append("|-------|-------------|--------|------------|------|")
    for check in meta["reliability_synthesis"]["checks"]:
        lines.append(
            f"| {check['label']} | {check['description']} | "
            f"{check['target']} | {check['achieved_kappa']} | {check['met_target']} |"
        )
    lines.append(f"\n**Overall:** {meta['reliability_synthesis']['overall_assessment']}\n")

    lines.append("## 4. Methodological Limitations\n")
    for lim in meta["limitations"]:
        lines.append(f"### {lim['threat']}\n")
        lines.append(f"{lim['description']}\n")
        lines.append(f"- **Mitigation:** {lim['mitigation']}")
        lines.append(f"- **Residual risk:** {lim['residual_risk']}\n")

    lines.append("## 5. Data Summary\n")
    for k, v in meta["data_summary"].items():
        label = k.replace("_", " ").capitalize()
        lines.append(f"- {label}: {v}")

    with (output_dir / "H5_meta_evaluation.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"H.5 report exported to {output_dir}")
