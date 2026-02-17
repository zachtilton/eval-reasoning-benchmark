"""
Unit tests for src/scoring/edge_cases.py.

Covers every row in the edge-case decision table (spec Edge Case Rules /
Appendix G.6.3): substantive refusal, technical timeout, safety trigger,
broken format, nonsensical content, unresolved ambivalence, contradictory
claims, clear final classification extracted from multi-part rationale,
and normal (no edge case) responses.

Also tests apply_edge_case_handling for correct exclude_from_analysis logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.scoring.edge_cases import (
    apply_edge_case_handling,
    detect_edge_case,
    has_clear_final_classification,
    has_contradictory_classifications,
    is_nonsensical,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _detect(classification=None, rationale=None, error_details=None):
    """Shorthand to call detect_edge_case with a dict."""
    return detect_edge_case({
        "classification_output": classification,
        "rationale_text": rationale,
        "error_details": error_details,
    })


# ---------------------------------------------------------------------------
# Class: detect_edge_case — decision table
# ---------------------------------------------------------------------------

class TestDetectEdgeCaseSubstantiveRefusal:
    """Substantive refusals → fail, reasoning failure."""

    def test_cannot_evaluate_phrase(self):
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="I cannot evaluate this fragment without more context.",
        )
        assert edge_type == "substantive_refusal"
        assert action == "fail"

    def test_unable_to_assess_phrase(self):
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="I am unable to assess the quality of this reasoning.",
        )
        assert edge_type == "substantive_refusal"
        assert action == "fail"

    def test_insufficient_information_phrase(self):
        edge_type, action, _ = _detect(
            classification="not_sound",
            rationale="There is insufficient information to make a judgment.",
        )
        assert edge_type == "substantive_refusal"
        assert action == "fail"

    def test_cannot_determine_phrase(self):
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="Based on the provided text, I cannot determine whether the reasoning is sound.",
        )
        assert edge_type == "substantive_refusal"
        assert action == "fail"

    def test_not_enough_context_phrase(self):
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="There is not enough context to judge this conclusion.",
        )
        assert edge_type == "substantive_refusal"
        assert action == "fail"


class TestDetectEdgeCaseTechnicalTimeout:
    """API timeout → exclude (infrastructure failure, not model failure)."""

    def test_timeout_in_error_details_excludes(self):
        edge_type, action, _ = _detect(
            error_details="Request timeout after 60 seconds",
        )
        assert edge_type == "technical_timeout"
        assert action == "exclude"

    def test_timeout_keyword_case_insensitive(self):
        edge_type, action, _ = _detect(error_details="TIMEOUT occurred")
        assert edge_type == "technical_timeout"
        assert action == "exclude"

    def test_nan_error_details_not_timeout(self):
        """NaN / None error_details (from CSV read) must not trigger timeout."""
        import math
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="The evaluation demonstrates logical and comprehensive reasoning.",
            error_details=float("nan"),
        )
        assert edge_type != "technical_timeout"
        assert action != "exclude"


class TestDetectEdgeCaseSafetyTrigger:
    """Safety policy trigger → fail, flagged separately."""

    def test_safety_in_error_details(self):
        edge_type, action, _ = _detect(error_details="Safety filter activated")
        assert edge_type == "safety_trigger"
        assert action == "fail_flagged"

    def test_content_policy_in_error_details(self):
        edge_type, action, _ = _detect(error_details="content policy violation detected")
        assert edge_type == "safety_trigger"
        assert action == "fail_flagged"

    def test_policy_violation_in_error_details(self):
        edge_type, action, _ = _detect(error_details="policy violation: harmful content")
        assert edge_type == "safety_trigger"
        assert action == "fail_flagged"


class TestDetectEdgeCaseMissingElements:
    """Missing classification or rationale → fail."""

    def test_missing_classification_fails(self):
        edge_type, action, _ = _detect(
            classification=None,
            rationale="The evaluation demonstrates sound reasoning throughout.",
        )
        assert edge_type == "missing_classification"
        assert action == "fail"

    def test_empty_classification_fails(self):
        edge_type, action, _ = _detect(
            classification="",
            rationale="The evaluation demonstrates sound reasoning throughout.",
        )
        assert edge_type == "missing_classification"
        assert action == "fail"

    def test_missing_rationale_fails(self):
        edge_type, action, _ = _detect(
            classification="sound",
            rationale=None,
        )
        assert edge_type == "missing_rationale"
        assert action == "fail"

    def test_short_rationale_under_10_words_fails(self):
        """Rationale with fewer than 10 words → missing_rationale."""
        edge_type, action, _ = _detect(
            classification="sound",
            rationale="Reasoning is clear.",  # 3 words
        )
        assert edge_type == "missing_rationale"
        assert action == "fail"

    def test_exactly_10_words_does_not_fail_as_missing(self):
        """10-word rationale meets the minimum; must not be flagged as missing."""
        ten_words = "The evaluation demonstrates reasonable evidence for this particular assessment."
        # Count: The(1) evaluation(2) demonstrates(3) reasonable(4) evidence(5)
        #        for(6) this(7) particular(8) assessment(9) — 9 words; add one more
        ten_words = "The evaluation demonstrates reasonable evidence for this particular judgment here."
        words = ten_words.split()
        assert len(words) == 10, f"Fixture error: got {len(words)} words"

        edge_type, _, _ = _detect(classification="sound", rationale=ten_words)
        assert edge_type != "missing_rationale"


class TestDetectEdgeCaseNonsensical:
    """Correct format + nonsensical content → fail, reasoning failure."""

    def test_repetitive_output_is_nonsensical(self):
        """Low lexical diversity (< 30% unique words for > 10 words) → fail."""
        repetitive = ("the " * 20).strip()  # all same word, 0% diversity
        edge_type, action, _ = _detect(classification="sound", rationale=repetitive)
        assert edge_type == "nonsensical_content"
        assert action == "fail"

    def test_off_topic_rationale_is_nonsensical(self):
        """No evaluation-domain keywords → off-topic → fail."""
        off_topic = (
            "The weather today is sunny with temperatures expected to rise "
            "throughout the afternoon hours into the evening time period."
        )
        edge_type, action, _ = _detect(classification="sound", rationale=off_topic)
        assert edge_type == "nonsensical_content"
        assert action == "fail"

    def test_evaluation_rationale_is_not_nonsensical(self):
        """Normal evaluation rationale with domain vocabulary → not nonsensical."""
        normal = (
            "The evaluation demonstrates strong evidence-based reasoning. "
            "The conclusion addresses the criterion with appropriate synthesis "
            "and acknowledges limitations in the assessment."
        )
        edge_type, _, _ = _detect(classification="sound", rationale=normal)
        assert edge_type != "nonsensical_content"


class TestDetectEdgeCaseAmbivalence:
    """Unresolved ambivalence → fail; clear final classification → process normally."""

    def test_ambivalence_without_final_classification_fails(self):
        """Ambivalence phrase + no conclusory marker → unresolved_ambivalence."""
        ambiguous = (
            "On one hand, the evaluation provides clear evidence. "
            "However, the synthesis remains incomplete in key areas."
        )
        edge_type, action, _ = _detect(classification="sound", rationale=ambiguous)
        assert edge_type == "unresolved_ambivalence"
        assert action == "fail"

    def test_ambivalence_with_clear_final_is_processed_normally(self):
        """
        Ambivalence phrase + explicit final classification → NOT unresolved.
        The clear final judgment resolves the ambivalence (spec Edge Case Rules).
        """
        resolved = (
            "On one hand, some evidence is missing. "
            "Overall, therefore, the reasoning is sound and defensible."
        )
        edge_type, action, _ = _detect(classification="sound", rationale=resolved)
        # Should NOT be unresolved_ambivalence; may be None or another type
        assert edge_type != "unresolved_ambivalence"
        assert action != "fail" or edge_type is None or edge_type == "contradictory_claims"

    def test_it_depends_phrase_triggers_ambivalence_check(self):
        rationale = (
            "It depends on the interpretation of the criterion. "
            "The evidence base may or may not support the conclusion here."
        )
        edge_type, action, _ = _detect(classification="sound", rationale=rationale)
        assert edge_type == "unresolved_ambivalence"
        assert action == "fail"


class TestDetectEdgeCaseContradictory:
    """Contradictory claims → fail."""

    def test_sound_and_not_sound_in_same_rationale_fails(self):
        """Rationale asserting both 'sound' and 'not sound' → contradictory_claims."""
        contradictory = (
            "The methodology demonstrates sound reasoning in some areas. "
            "The conclusions are not sound and lack adequate evidence."
        )
        edge_type, action, _ = _detect(
            classification="sound", rationale=contradictory
        )
        assert edge_type == "contradictory_claims"
        assert action == "fail"

    def test_not_sound_only_is_not_contradictory(self):
        rationale = (
            "The conclusion is not sound. It lacks evidence and appropriate synthesis "
            "throughout the evaluation report sections."
        )
        edge_type, _, _ = _detect(classification="not_sound", rationale=rationale)
        assert edge_type != "contradictory_claims"

    def test_sound_only_is_not_contradictory(self):
        rationale = (
            "The evaluation demonstrates sound reasoning with comprehensive evidence "
            "and appropriate synthesis across all domains."
        )
        edge_type, _, _ = _detect(classification="sound", rationale=rationale)
        assert edge_type != "contradictory_claims"


class TestDetectEdgeCaseNormal:
    """Clean, valid response → no edge case detected."""

    def test_normal_response_returns_none_edge_case(self):
        edge_type, action, notes = _detect(
            classification="sound",
            rationale=(
                "The evaluation demonstrates comprehensive evidence-based reasoning. "
                "It appropriately synthesizes all criteria with logical rigor and "
                "acknowledges limitations in the assessment process."
            ),
        )
        assert edge_type is None
        assert action == "process_normally"
        assert notes == ""

    def test_not_sound_normal_response(self):
        edge_type, action, _ = _detect(
            classification="not_sound",
            rationale=(
                "The evaluation lacks key evidence and fails to address central "
                "criteria. The reasoning is flawed and the synthesis is wholly "
                "inadequate for a sound judgment."
            ),
        )
        assert edge_type is None
        assert action == "process_normally"


# ---------------------------------------------------------------------------
# Class: helper functions
# ---------------------------------------------------------------------------

class TestHasClearFinalClassification:

    def test_therefore_plus_sound_returns_true(self):
        rationale = "Evidence is mixed. Therefore, the conclusion is sound overall."
        assert has_clear_final_classification(rationale, "sound") is True

    def test_overall_plus_not_sound_returns_true(self):
        rationale = "There are weaknesses. Overall, the reasoning is not sound."
        assert has_clear_final_classification(rationale, "not_sound") is True

    def test_no_conclusory_marker_returns_false(self):
        rationale = "The evidence is mixed. The conclusion addresses some criteria."
        assert has_clear_final_classification(rationale, "sound") is False

    def test_conclusory_marker_wrong_classification_returns_false(self):
        """'Therefore... not sound' with classification='sound' → False."""
        rationale = "The evidence is weak. Therefore, the reasoning is not sound."
        assert has_clear_final_classification(rationale, "sound") is False

    def test_in_summary_marker_works(self):
        rationale = "There are issues. In summary, the evaluation is sound."
        assert has_clear_final_classification(rationale, "sound") is True


class TestHasContradictoryClassifications:

    def test_sound_and_not_sound_both_present(self):
        text = "The argument is sound in methodology. The conclusions are not sound."
        assert has_contradictory_classifications(text) is True

    def test_not_sound_only_not_contradictory(self):
        text = "The conclusion is not sound. It lacks evidence and appropriate synthesis."
        assert has_contradictory_classifications(text) is False

    def test_sound_only_not_contradictory(self):
        text = "The reasoning is sound and well-supported by evidence."
        assert has_contradictory_classifications(text) is False

    def test_multiple_not_sound_with_one_standalone_sound(self):
        """
        'not sound' appearing twice should not inflate standalone 'sound' count.
        Regression test for the appendix's fragile arithmetic.
        """
        text = "It is not sound in structure. It is also not sound in evidence. But sound in intent."
        # not_sound_count = 2; after removing 'not sound', standalone sound = 1 → True
        assert has_contradictory_classifications(text) is True

    def test_not_sound_substring_does_not_give_false_positive(self):
        """
        'not sound' must not increment standalone 'sound' count.
        Regression test against count('sound') - count('not sound') subtraction.
        """
        text = "The evaluation is not sound at all."
        # If broken: count('sound')=1, count('not sound')=1 → 1-1=0 → not contradictory ✓
        # If correct: not_sound=1, standalone_sound=0 → False ✓
        assert has_contradictory_classifications(text) is False


class TestIsNonsensical:

    def test_low_lexical_diversity_is_nonsensical(self):
        repetitive = "hello " * 15  # 15 identical words, ~6.7% unique
        assert is_nonsensical(repetitive.strip()) is True

    def test_off_topic_no_evaluation_keywords(self):
        off_topic = (
            "The sunset was beautiful with orange and pink hues across the horizon "
            "as the birds flew home to their nests for the night."
        )
        assert is_nonsensical(off_topic) is True

    def test_evaluation_vocabulary_not_nonsensical(self):
        normal = (
            "The evaluation demonstrates sound reasoning with appropriate synthesis "
            "of evidence across all criteria. The assessment is comprehensive."
        )
        assert is_nonsensical(normal) is False

    def test_short_text_not_flagged_for_diversity(self):
        """Text with ≤ 10 words is not subject to diversity check."""
        short = "This is good."
        # Only 3 words → diversity check skipped; needs domain keyword check
        # No evaluation keywords → nonsensical
        assert is_nonsensical(short) is True

    def test_short_text_with_evaluation_keyword_not_flagged(self):
        short = "The evaluation is sound."
        # ≤ 10 words; has 'evaluat' → not nonsensical
        assert is_nonsensical(short) is False


# ---------------------------------------------------------------------------
# Class: apply_edge_case_handling (batch)
# ---------------------------------------------------------------------------

class TestApplyEdgeCaseHandling:

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "classification_output": "sound",
            "rationale_text": (
                "The evaluation demonstrates comprehensive evidence-based reasoning "
                "and appropriately synthesizes all relevant criteria."
            ),
            "error_details": None,
        }
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def test_timeout_sets_exclude_true(self):
        df = self._make_df([{"error_details": "Request timeout"}])
        result = apply_edge_case_handling(df)
        assert bool(result.iloc[0]["exclude_from_analysis"]) == True

    def test_substantive_refusal_sets_exclude_false(self):
        """Refusals are scored as Fail but NOT excluded from analysis."""
        df = self._make_df([{
            "rationale_text": "I cannot evaluate this without more context provided."
        }])
        result = apply_edge_case_handling(df)
        assert bool(result.iloc[0]["exclude_from_analysis"]) == False
        assert result.iloc[0]["edge_case_type"] == "substantive_refusal"

    def test_safety_trigger_sets_exclude_false(self):
        df = self._make_df([{"error_details": "safety filter triggered"}])
        result = apply_edge_case_handling(df)
        assert bool(result.iloc[0]["exclude_from_analysis"]) == False
        assert result.iloc[0]["edge_case_type"] == "safety_trigger"

    def test_normal_response_sets_exclude_false_and_type_none(self):
        df = self._make_df([{}])
        result = apply_edge_case_handling(df)
        assert bool(result.iloc[0]["exclude_from_analysis"]) == False
        assert result.iloc[0]["edge_case_type"] is None

    def test_output_is_copy_not_mutation(self):
        df = self._make_df([{}])
        original_cols = set(df.columns)
        result = apply_edge_case_handling(df)
        assert "edge_case_type" not in original_cols
        assert "edge_case_type" in result.columns

    def test_mixed_rows_correct_exclude_flags(self):
        df = self._make_df([
            {"error_details": "Request timeout"},    # exclude
            {"error_details": None},                 # keep
            {"error_details": "safety content policy violation"},  # keep (flagged fail)
        ])
        result = apply_edge_case_handling(df)
        assert bool(result.iloc[0]["exclude_from_analysis"]) == True
        assert bool(result.iloc[1]["exclude_from_analysis"]) == False
        assert bool(result.iloc[2]["exclude_from_analysis"]) == False

    def test_required_columns_added(self):
        df = self._make_df([{}])
        result = apply_edge_case_handling(df)
        assert "edge_case_type" in result.columns
        assert "exclude_from_analysis" in result.columns

    def test_non_default_index_no_row_misalignment(self):
        """
        Regression test: apply_edge_case_handling must use direct column assignment
        (not pd.concat axis=1), which would misalign rows on non-default indexes.
        """
        rows = [
            {"error_details": "Request timeout"},
            {"error_details": None},
        ]
        df = self._make_df(rows)
        df.index = [10, 20]  # non-default index

        result = apply_edge_case_handling(df)

        assert result.index.tolist() == [10, 20]
        assert bool(result.loc[10, "exclude_from_analysis"]) == True
        assert bool(result.loc[20, "exclude_from_analysis"]) == False
