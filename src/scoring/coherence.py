"""
Tiered internal coherence validation (spec Section 2.4, Step 1).

Covers Appendix G.4.  Key corrections over the appendix code:

- validate_all_responses was missing its return statement (silent data loss).
- validate_all_responses appended 'sentiment_result'/'sentiment_polarity' fields
  that belong to a discarded sentiment-analysis approach.  These are removed.
- validate_coherence output keys now match the scored_database schema
  (coherence_final, coherence_rule_based, strength_indicator_count, etc.)
  rather than the appendix's mismatched names.
- select_rc2_sample now implements the stratified spec requirement
  (20 coherent + 20 incoherent + 10 ambiguous) instead of a random 10%
  of ambiguous cases only.
- calculate_rc2_reliability now compares coherence_final (the pipeline's
  definitive output) rather than rule_based_result.
- LLM call uses the ANTHROPIC_API_VERSION constant instead of a hardcoded
  date string.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .config import (
    ANTHROPIC_API_VERSION,
    CONFIDENCE_THRESHOLD,
    HAIKU_MODEL_ID,
    LLM_COHERENCE_PROMPT,
    RC2_AMBIGUOUS_MAX,
    RC2_COHERENT_SAMPLE,
    RC2_INCOHERENT_SAMPLE,
    RC2_KAPPA_TARGET,
    RC2_SAMPLE_PATH,
    STRENGTH_INDICATORS,
    WEAKNESS_INDICATORS,
)


# ---------------------------------------------------------------------------
# Tier 1: rule-based keyword matching
# ---------------------------------------------------------------------------

def count_indicators(rationale_text: str, indicator_list: list[str]) -> int:
    """
    Count how many indicator keywords appear in a rationale string.

    Args:
        rationale_text: Rationale text from model output.
        indicator_list: List of keyword strings to search for.

    Returns:
        Number of matched indicator keywords (each counted once).
    """
    text_lower = rationale_text.lower()
    return sum(1 for indicator in indicator_list if indicator in text_lower)


def rule_based_coherence_check(
    classification: str | None,
    rationale_text: str | None,
) -> tuple[str, int, int]:
    """
    Assess coherence using keyword indicator counts (Tier 1).

    Logic (spec Section 2.4):
    - "sound" + strength indicators dominate → coherent signal
    - "not_sound" + weakness indicators dominate → coherent signal
    - Mismatch or tie → incoherent signal

    Note: this function returns a *preliminary* judgment.  Whether to trust
    it or route to Tier 2 is determined by the confidence threshold in
    :func:`validate_coherence`.

    Args:
        classification: Normalized classification ('sound' or 'not_sound').
        rationale_text: Rationale text from model output.

    Returns:
        Tuple of (coherence_signal: str, strength_count: int, weakness_count: int).
        coherence_signal is 'coherent' or 'incoherent'.
    """
    if not classification or not rationale_text:
        return "incoherent", 0, 0

    strength_count = count_indicators(rationale_text, STRENGTH_INDICATORS)
    weakness_count = count_indicators(rationale_text, WEAKNESS_INDICATORS)

    if classification == "sound":
        signal = "coherent" if strength_count > weakness_count else "incoherent"
    elif classification == "not_sound":
        signal = "coherent" if weakness_count > strength_count else "incoherent"
    else:
        signal = "incoherent"

    return signal, strength_count, weakness_count


# ---------------------------------------------------------------------------
# Tier 2: LLM coherence screen (Claude Haiku, temperature 0)
# ---------------------------------------------------------------------------

def llm_coherence_check(
    classification: str | None,
    rationale_text: str | None,
) -> tuple[str, str]:
    """
    Second-tier coherence screen using Claude Haiku at temperature 0.

    Called only when Tier 1 is not confident (indicator difference below
    threshold).  Receives ONLY the classification and rationale — no fragment
    text, gold standard, or study metadata — to preserve independence from
    the benchmarked task (spec Section 2.4 rationale).

    Args:
        classification: Normalized classification ('sound' or 'not_sound').
        rationale_text: Rationale text from model output.

    Returns:
        Tuple of (coherence_result: str, raw_response: str).
        coherence_result is 'coherent', 'incoherent', or 'ambiguous'.
        'ambiguous' is returned when the LLM call fails or produces
        an unrecognized response — failure mode flags for manual review.
    """
    if not classification or not rationale_text:
        return "incoherent", "missing_input"

    display_class = classification.replace("_", " ")
    prompt = LLM_COHERENCE_PROMPT.format(
        classification=display_class,
        rationale=rationale_text,
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "ambiguous", "error: ANTHROPIC_API_KEY not set"

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
            json={
                "model": HAIKU_MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()

        result_text = response.json()["content"][0]["text"].strip().lower()

        # "incoherent" contains "coherent" as a substring, so check incoherent first
        if "incoherent" in result_text:
            return "incoherent", result_text
        if "coherent" in result_text:
            return "coherent", result_text
        return "ambiguous", result_text

    except Exception as exc:
        # LLM call failure → flag for manual review rather than silently assign
        print(f"  LLM coherence check failed: {exc}")
        return "ambiguous", f"error: {exc}"


# ---------------------------------------------------------------------------
# Combined tiered validation
# ---------------------------------------------------------------------------

def validate_coherence(
    classification: str | None,
    rationale_text: str | None,
    confidence_threshold: int = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Validate internal coherence using the tiered pipeline (spec Section 2.4).

    Tier 1 (rule-based): if indicator difference ≥ threshold → assign.
    Tier 2 (LLM): if not confident → screen with Haiku.
    Disagreement between tiers → 'ambiguous' + manual_review_flag.

    Output keys match the scored_database schema exactly:
    coherence_final, coherence_rule_based, strength_indicator_count,
    weakness_indicator_count, coherence_llm, manual_review_flag.

    Args:
        classification: Normalized classification ('sound' or 'not_sound').
        rationale_text: Rationale text from model output.
        confidence_threshold: Minimum |strength - weakness| for Tier 1 confidence.

    Returns:
        Dict with coherence scoring fields matching the scored_database schema.
    """
    rule_signal, strength_count, weakness_count = rule_based_coherence_check(
        classification, rationale_text
    )

    indicator_diff = abs(strength_count - weakness_count)
    tier1_confident = indicator_diff >= confidence_threshold

    if tier1_confident:
        return {
            "coherence_rule_based": rule_signal,
            "strength_indicator_count": strength_count,
            "weakness_indicator_count": weakness_count,
            "coherence_llm": "not_checked",
            "coherence_final": rule_signal,
            "manual_review_flag": False,
        }

    # Tier 1 not confident → route to LLM screen
    llm_result, llm_raw = llm_coherence_check(classification, rationale_text)

    if rule_signal == llm_result:
        # Both tiers agree — use their shared judgment
        return {
            "coherence_rule_based": rule_signal,
            "strength_indicator_count": strength_count,
            "weakness_indicator_count": weakness_count,
            "coherence_llm": llm_result,
            "coherence_final": rule_signal,
            "manual_review_flag": False,
        }

    # Tiers disagree — flag ambiguous for manual review
    return {
        "coherence_rule_based": rule_signal,
        "strength_indicator_count": strength_count,
        "weakness_indicator_count": weakness_count,
        "coherence_llm": llm_result,
        "coherence_final": "ambiguous",
        "manual_review_flag": True,
    }


# ---------------------------------------------------------------------------
# Batch application
# ---------------------------------------------------------------------------

def validate_all_responses(
    database_df: pd.DataFrame,
    confidence_threshold: int = CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply coherence validation to all responses in the database.

    Error-flagged responses (failed parsing) are assigned 'incoherent'
    immediately without calling the LLM (they already failed a prior step).

    Args:
        database_df: Response database DataFrame.
        confidence_threshold: Forwarded to :func:`validate_coherence`.

    Returns:
        DataFrame with coherence columns added (modifies a copy).
    """
    print(f"Validating coherence for {len(database_df)} responses...")

    results: list[dict] = []

    for pos, (_, row) in enumerate(database_df.iterrows(), start=1):

        if row.get("error_flag"):
            # Already failed parsing — assign incoherent without LLM call
            results.append({
                "coherence_rule_based": "incoherent",
                "strength_indicator_count": 0,
                "weakness_indicator_count": 0,
                "coherence_llm": "not_checked",
                "coherence_final": "incoherent",
                "manual_review_flag": False,
            })
        else:
            result = validate_coherence(
                row["classification_output"],
                row["rationale_text"],
                confidence_threshold=confidence_threshold,
            )
            results.append(result)

        if pos % 100 == 0:
            print(f"  Processed {pos}/{len(database_df)} responses")

    coherence_df = pd.DataFrame(results, index=database_df.index)
    result_df = pd.concat([database_df, coherence_df], axis=1)

    # Summary
    coherent_n = (result_df["coherence_final"] == "coherent").sum()
    incoherent_n = (result_df["coherence_final"] == "incoherent").sum()
    ambiguous_n = (result_df["coherence_final"] == "ambiguous").sum()
    total = len(result_df)

    tier_rule = (result_df.get("coherence_llm") == "not_checked").sum()
    tier_llm = (
        result_df["coherence_llm"].isin(["coherent", "incoherent"]) &
        (result_df["coherence_final"] != "ambiguous")
    ).sum()
    tier_disagree = (result_df["manual_review_flag"] == True).sum()  # noqa: E712

    print(f"\nCoherence Validation Summary:")
    print(f"  Coherent:              {coherent_n} ({coherent_n/total:.1%})")
    print(f"  Incoherent:            {incoherent_n} ({incoherent_n/total:.1%})")
    print(f"  Ambiguous (manual):    {ambiguous_n} ({ambiguous_n/total:.1%})")
    print(f"\nScreening Tier Usage:")
    print(f"  Rule-based only:       {tier_rule} ({tier_rule/total:.1%})")
    print(f"  LLM-confirmed:         {tier_llm} ({tier_llm/total:.1%})")
    print(f"  Disagreement (manual): {tier_disagree} ({tier_disagree/total:.1%})")

    return result_df


# ---------------------------------------------------------------------------
# RC2: coherence pipeline validation
# ---------------------------------------------------------------------------

def select_rc2_sample(
    database_df: pd.DataFrame,
    seed: int = 42,
    output_path: Path = RC2_SAMPLE_PATH,
) -> pd.DataFrame:
    """
    Select a stratified manual review sample for RC2 reliability check.

    Stratification follows spec Section 2.4 exactly:
      - 20 coherent responses (random sample)
      - 20 incoherent responses (random sample)
      - 10 ambiguous responses, or all ambiguous if fewer than 10 exist

    The appendix implemented random 10% of ambiguous cases only, which does
    not match the spec's stratified design.

    Args:
        database_df: Database with coherence_final column populated.
        seed: Random seed for reproducibility.
        output_path: Path to save the sample CSV for manual review.

    Returns:
        Stratified sample DataFrame.
    """
    rng = random.Random(seed)

    def sample_stratum(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=seed)

    coherent_pool = database_df[database_df["coherence_final"] == "coherent"]
    incoherent_pool = database_df[database_df["coherence_final"] == "incoherent"]
    ambiguous_pool = database_df[database_df["coherence_final"] == "ambiguous"]

    coherent_sample = sample_stratum(coherent_pool, RC2_COHERENT_SAMPLE)
    incoherent_sample = sample_stratum(incoherent_pool, RC2_INCOHERENT_SAMPLE)
    # "all ambiguous if < 10"
    ambiguous_n = min(len(ambiguous_pool), RC2_AMBIGUOUS_MAX)
    ambiguous_sample = (
        ambiguous_pool.sample(n=ambiguous_n, random_state=seed)
        if ambiguous_n > 0
        else pd.DataFrame(columns=database_df.columns)
    )

    sample_df = pd.concat(
        [coherent_sample, incoherent_sample, ambiguous_sample],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"\nRC2 Sample Selected:")
    print(f"  Coherent:   {len(coherent_sample)} / {len(coherent_pool)} available")
    print(f"  Incoherent: {len(incoherent_sample)} / {len(incoherent_pool)} available")
    print(f"  Ambiguous:  {len(ambiguous_sample)} / {len(ambiguous_pool)} available")
    print(f"  Total:      {len(sample_df)}")
    print(f"  Saved to:   {output_path}")

    return sample_df


def record_manual_review_decision(
    fragment_id: str,
    model: str,
    prompt_condition: str,
    run_number: int,
    manual_coherence: str,
    reviewer_notes: str = "",
) -> dict:
    """
    Build a manual coherence review record for RC2 reliability tracking.

    Args:
        fragment_id: Fragment identifier.
        model: Model identifier.
        prompt_condition: 'zero_shot' or 'few_shot'.
        run_number: Run number (1–3).
        manual_coherence: Researcher's judgment: 'coherent' or 'incoherent'.
        reviewer_notes: Optional notes explaining the judgment.

    Returns:
        Dict representing the manual review record.
    """
    return {
        "fragment_id": fragment_id,
        "model_family": model,
        "prompt_condition": prompt_condition,
        "run_number": run_number,
        "manual_coherence": manual_coherence,
        "review_timestamp": datetime.now().isoformat(),
        "reviewer_notes": reviewer_notes,
    }


def calculate_rc2_reliability(
    database_df: pd.DataFrame,
    manual_reviews_df: pd.DataFrame,
) -> dict:
    """
    Calculate Cohen's kappa for RC2 (automated coherence pipeline vs. manual).

    Uses coherence_final (the pipeline's definitive output) as the automated
    judgment, not rule_based_result as the appendix incorrectly specified.
    Ambiguous cases from the automated pipeline are included; the manual
    reviewer makes a binary coherent/incoherent judgment for each.

    Args:
        database_df: Scored database with coherence_final populated.
        manual_reviews_df: DataFrame of manual review records
                           (from :func:`record_manual_review_decision`).

    Returns:
        Dict with kappa, agreement_rate, sample_size, target_met,
        and confusion_matrix.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    merge_keys = ["fragment_id", "model_family", "prompt_condition", "run_number"]
    merged = database_df.merge(manual_reviews_df, on=merge_keys, how="inner")

    if merged.empty:
        raise ValueError("No matching rows between database and manual reviews.")

    # Use coherence_final (pipeline output) vs. manual_coherence (researcher)
    # Map 'ambiguous' → 'incoherent' for binary kappa calculation
    auto_labels = (
        merged["coherence_final"]
        .replace("ambiguous", "incoherent")
        .values
    )
    manual_labels = merged["manual_coherence"].values

    kappa = cohen_kappa_score(auto_labels, manual_labels)
    agreement = float((auto_labels == manual_labels).mean())
    cm = confusion_matrix(
        manual_labels, auto_labels, labels=["coherent", "incoherent"]
    )

    results = {
        "kappa": round(kappa, 3),
        "agreement_rate": round(agreement, 4),
        "sample_size": len(merged),
        "target_met": kappa >= RC2_KAPPA_TARGET,
        "confusion_matrix": cm,
    }

    print(f"\nRC2 Reliability Check:")
    print(f"  Cohen's kappa:   {kappa:.3f}  (target ≥ {RC2_KAPPA_TARGET})")
    print(f"  Agreement rate:  {agreement:.1%}")
    print(f"  Sample size:     {len(merged)}")
    print(f"  Target met:      {'YES' if results['target_met'] else 'NO — recalibrate'}")
    print(f"\nConfusion Matrix (rows = manual, cols = automated):")
    print(f"                  Automated")
    print(f"             Coherent  Incoherent")
    print(f"  Coherent      {cm[0, 0]:4d}       {cm[0, 1]:4d}")
    print(f"  Incoherent    {cm[1, 0]:4d}       {cm[1, 1]:4d}")

    return results
