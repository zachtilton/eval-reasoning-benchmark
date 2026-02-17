"""
Edge case detection and handling (spec Edge Case Rules; Appendix G.6).

Key corrections over the appendix code:

- apply_edge_case_handling replaced pd.concat(axis=1) with direct column
  assignment to prevent silent row misalignment on non-default indexes.
- detect_edge_case now coerces error_details to str before .lower() to
  handle NaN values that arrive from CSV reads.
- has_contradictory_classifications rewritten to avoid the fragile
  count('sound') - count('not sound') arithmetic, which produces incorrect
  results when "not sound" appears multiple times (each occurrence also
  increments count('sound')).  Now uses word-boundary regex to count
  standalone "sound" occurrences separately from "not sound".
- is_nonsensical: the evaluation-keyword check is retained but the
  threshold for repetition is documented clearly.
"""

from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Edge case detection helpers
# ---------------------------------------------------------------------------

def is_nonsensical(rationale: str) -> bool:
    """
    Detect gibberish or completely off-topic rationale content.

    Two heuristics:
    1. Low lexical diversity (< 30% unique word ratio for texts > 10 words)
       indicates repetitive/looping output.
    2. Absence of any evaluation-domain keyword suggests the rationale
       is off-topic rather than addressing the fragment.

    Args:
        rationale: Rationale text from model output.

    Returns:
        True if the content appears nonsensical.
    """
    words = rationale.lower().split()
    if len(words) > 10 and len(set(words)) / len(words) < 0.30:
        return True  # < 30% unique words → repetitive output

    evaluation_keywords = [
        "evaluat", "criterion", "evidence", "conclusion", "judgment",
        "assessment", "reasoning", "analysis", "synthesis", "standard",
    ]
    text_lower = rationale.lower()
    if not any(kw in text_lower for kw in evaluation_keywords):
        return True  # No domain vocabulary → likely off-topic

    return False


def has_clear_final_classification(
    rationale: str,
    stated_classification: str,
) -> bool:
    """
    Check whether an ambivalent rationale ends with an explicit final judgment.

    Looks for a conclusory indicator in the last non-empty sentence and
    confirms the stated classification appears there.

    Args:
        rationale: Rationale text from model output.
        stated_classification: Classification the model stated
                               ('sound' or 'not_sound').

    Returns:
        True if a clear final classification is detectable.
    """
    sentences = [s.strip() for s in rationale.split(".") if s.strip()]
    final_sentence = sentences[-1] if sentences else rationale

    conclusory_markers = [
        "therefore", "thus", "in conclusion", "overall",
        "ultimately", "final judgment", "verdict", "in summary",
    ]
    final_lower = final_sentence.lower()

    if not any(marker in final_lower for marker in conclusory_markers):
        return False

    # Normalize stated_classification for substring search
    display = stated_classification.replace("_", " ")
    return display in final_lower


def has_contradictory_classifications(rationale: str) -> bool:
    """
    Detect whether a rationale asserts both "sound" and "not sound".

    Uses word-boundary regex to count standalone occurrences of "sound"
    that are NOT part of "not sound", avoiding the appendix's fragile
    ``count('sound') - count('not sound')`` arithmetic (where each
    "not sound" occurrence also increments count('sound'), producing
    incorrect subtraction results with repeated phrases).

    Args:
        rationale: Rationale text from model output.

    Returns:
        True if both "sound" and "not sound" appear as substantive judgments.
    """
    text = rationale.lower()

    not_sound_count = len(re.findall(r"\bnot\s+sound\b", text))
    # "sound" occurrences that are not part of "not sound"
    # Remove "not sound" first, then count remaining standalone "sound"
    text_without_not_sound = re.sub(r"\bnot\s+sound\b", "", text)
    standalone_sound_count = len(re.findall(r"\bsound\b", text_without_not_sound))

    return standalone_sound_count > 0 and not_sound_count > 0


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_edge_case(response_dict: dict) -> tuple[str | None, str, str]:
    """
    Detect and categorize a response edge case for appropriate handling.

    Decision table (spec Edge Case Rules + G.6.3):
    - Substantive refusal     → fail  (reasoning failure)
    - Technical timeout       → exclude (infrastructure issue)
    - Safety policy trigger   → fail, flagged separately
    - Missing classification  → fail
    - Missing/short rationale → fail
    - Nonsensical content     → fail
    - Unresolved ambivalence  → fail
    - Contradictory claims    → fail
    - No edge case            → process normally

    Args:
        response_dict: Dict with keys classification_output, rationale_text,
                       and error_details (any may be None/NaN).

    Returns:
        Tuple of (edge_case_type: str | None, handling_action: str, notes: str).
        edge_case_type matches the scored_database schema values.
    """
    classification = response_dict.get("classification_output")
    rationale = response_dict.get("rationale_text") or ""
    # Coerce error_details to str — CSV reads produce NaN for empty cells
    error = str(response_dict.get("error_details") or "").lower()

    # --- Technical infrastructure failures ---
    if "timeout" in error:
        return "technical_timeout", "exclude", "API timeout — infrastructure issue"

    # --- Safety policy triggers ---
    if "safety" in error or "content policy" in error or "policy violation" in error:
        return "safety_trigger", "fail_flagged", "Safety policy triggered"

    # --- Substantive refusals ---
    refusal_phrases = [
        "cannot evaluate",
        "unable to assess",
        "insufficient information",
        "not enough context",
        "cannot determine",
        "impossible to judge",
    ]
    if rationale and any(phrase in rationale.lower() for phrase in refusal_phrases):
        return "substantive_refusal", "fail", "Model refused to evaluate"

    # --- Missing elements ---
    if not classification:
        return "missing_classification", "fail", "No classification provided"

    if not rationale or len(rationale.split()) < 10:
        return "missing_rationale", "fail", "Rationale missing or too brief"

    # --- Content quality checks ---
    if is_nonsensical(rationale):
        return "nonsensical_content", "fail", "Rationale is gibberish or off-topic"

    # --- Multi-part judgment checks ---
    ambivalence_phrases = [
        "could be either",
        "it depends",
        "unclear whether",
        "difficult to say",
        "both interpretations",
        "on one hand",
    ]
    if any(phrase in rationale.lower() for phrase in ambivalence_phrases):
        if not has_clear_final_classification(rationale, classification):
            return "unresolved_ambivalence", "fail", "No clear final judgment"

    if has_contradictory_classifications(rationale):
        return "contradictory_claims", "fail", "Rationale contradicts classification"

    return None, "process_normally", ""


# ---------------------------------------------------------------------------
# Batch application
# ---------------------------------------------------------------------------

def apply_edge_case_handling(database_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply edge case detection to all responses and record results.

    Uses direct column assignment rather than pd.concat(axis=1) to avoid
    silent row misalignment when the DataFrame index is non-default.

    Sets exclude_from_analysis = True for technical timeouts (infrastructure
    failures not attributable to model reasoning).  All other failures
    remain in the analysis as Fail outcomes.

    Args:
        database_df: Response database DataFrame.

    Returns:
        Copy of the DataFrame with edge_case_type and exclude_from_analysis
        columns added (matching the scored_database schema).
    """
    result_df = database_df.copy()

    edge_case_types: list[str | None] = []
    handling_actions: list[str] = []

    for _, row in result_df.iterrows():
        response_dict = {
            "classification_output": row.get("classification_output"),
            "rationale_text": row.get("rationale_text"),
            "error_details": row.get("error_details"),
        }
        edge_type, action, _ = detect_edge_case(response_dict)
        edge_case_types.append(edge_type)
        handling_actions.append(action)

    result_df["edge_case_type"] = edge_case_types
    result_df["exclude_from_analysis"] = [
        action == "exclude" for action in handling_actions
    ]

    # Summary
    type_counts = pd.Series(edge_case_types).value_counts(dropna=False)
    excluded = result_df["exclude_from_analysis"].sum()

    print("\nEdge Case Summary:")
    for edge_type, count in type_counts.items():
        label = str(edge_type) if edge_type else "(none)"
        print(f"  {label}: {count}")
    print(f"  Excluded from analysis (infrastructure failures): {excluded}")

    return result_df
