"""
Unit tests for src/scoring/coherence.py.

Covers:
- Tier 1 (rule-based): all four indicator-alignment combinations, tied/balanced
  routing, missing inputs.
- Tier 2 (LLM screen): agree → assign, disagree → ambiguous, prompt content
  restricted to classification + rationale only.
- validate_coherence: confidence threshold routing, output key completeness,
  manual_review_flag semantics.
- validate_all_responses: error-flagged fast-path, output shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scoring.coherence import (
    count_indicators,
    llm_coherence_check,
    rule_based_coherence_check,
    validate_all_responses,
    validate_coherence,
)
from src.scoring.config import (
    CONFIDENCE_THRESHOLD,
    LLM_COHERENCE_PROMPT,
    STRENGTH_INDICATORS,
    WEAKNESS_INDICATORS,
)

from .conftest import (
    BALANCED_RATIONALE,
    STRONG_NOT_SOUND_MISMATCH_RATIONALE,
    STRONG_NOT_SOUND_RATIONALE,
    STRONG_SOUND_RATIONALE,
    WEAK_SOUND_MISMATCH_RATIONALE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strength_count(text: str) -> int:
    return count_indicators(text, STRENGTH_INDICATORS)


def _weakness_count(text: str) -> int:
    return count_indicators(text, WEAKNESS_INDICATORS)


# ---------------------------------------------------------------------------
# Class: Tier 1 — rule-based coherence check
# ---------------------------------------------------------------------------

class TestTier1RuleBased:
    """Tests for rule_based_coherence_check (Tier 1 logic only)."""

    # ── Happy-path alignment cases ─────────────────────────────────────────

    def test_sound_with_strength_indicators_is_coherent(self):
        """'sound' classification + strength-dominated rationale → coherent."""
        s = _strength_count(STRONG_SOUND_RATIONALE)
        w = _weakness_count(STRONG_SOUND_RATIONALE)
        assert s > w, "Fixture must have more strength than weakness indicators"

        signal, sc, wc = rule_based_coherence_check("sound", STRONG_SOUND_RATIONALE)

        assert signal == "coherent"
        assert sc == s
        assert wc == w

    def test_not_sound_with_weakness_indicators_is_coherent(self):
        """'not_sound' classification + weakness-dominated rationale → coherent."""
        s = _strength_count(STRONG_NOT_SOUND_RATIONALE)
        w = _weakness_count(STRONG_NOT_SOUND_RATIONALE)
        assert w > s, "Fixture must have more weakness than strength indicators"

        signal, sc, wc = rule_based_coherence_check("not_sound", STRONG_NOT_SOUND_RATIONALE)

        assert signal == "coherent"
        assert sc == s
        assert wc == w

    # ── Mismatched classification cases ────────────────────────────────────

    def test_sound_with_weakness_indicators_is_incoherent(self):
        """'sound' classification + weakness-dominated rationale → incoherent."""
        s = _strength_count(WEAK_SOUND_MISMATCH_RATIONALE)
        w = _weakness_count(WEAK_SOUND_MISMATCH_RATIONALE)
        assert w > s, "Fixture must have more weakness than strength indicators"

        signal, _, _ = rule_based_coherence_check("sound", WEAK_SOUND_MISMATCH_RATIONALE)

        assert signal == "incoherent"

    def test_not_sound_with_strength_indicators_is_incoherent(self):
        """'not_sound' classification + strength-dominated rationale → incoherent."""
        s = _strength_count(STRONG_NOT_SOUND_MISMATCH_RATIONALE)
        w = _weakness_count(STRONG_NOT_SOUND_MISMATCH_RATIONALE)
        assert s > w, "Fixture must have more strength than weakness indicators"

        signal, _, _ = rule_based_coherence_check(
            "not_sound", STRONG_NOT_SOUND_MISMATCH_RATIONALE
        )

        assert signal == "incoherent"

    # ── Tie / balanced case ────────────────────────────────────────────────

    def test_balanced_indicators_produces_incoherent_signal(self):
        """Equal strength and weakness counts → incoherent signal (tie goes to incoherent)."""
        s = _strength_count(BALANCED_RATIONALE)
        w = _weakness_count(BALANCED_RATIONALE)
        assert s == w, f"Fixture must be balanced; got strength={s}, weakness={w}"

        signal, _, _ = rule_based_coherence_check("sound", BALANCED_RATIONALE)

        # spec: "Confident signal (indicators substantially outweigh opposite)"
        # Tie is not confident → signal defaults to incoherent
        assert signal == "incoherent"

    # ── Missing / empty input ──────────────────────────────────────────────

    def test_none_classification_returns_incoherent(self):
        signal, sc, wc = rule_based_coherence_check(None, STRONG_SOUND_RATIONALE)
        assert signal == "incoherent"
        assert sc == 0 and wc == 0

    def test_none_rationale_returns_incoherent(self):
        signal, sc, wc = rule_based_coherence_check("sound", None)
        assert signal == "incoherent"
        assert sc == 0 and wc == 0

    def test_both_none_returns_incoherent(self):
        signal, sc, wc = rule_based_coherence_check(None, None)
        assert signal == "incoherent"
        assert sc == 0 and wc == 0

    def test_empty_string_rationale_returns_incoherent(self):
        signal, _, _ = rule_based_coherence_check("sound", "")
        assert signal == "incoherent"

    # ── Case-insensitivity ────────────────────────────────────────────────

    def test_indicator_matching_is_case_insensitive(self):
        """count_indicators must match uppercase and mixed-case text."""
        upper = count_indicators("DEMONSTRATES COMPREHENSIVE SYNTHESIS", STRENGTH_INDICATORS)
        lower = count_indicators("demonstrates comprehensive synthesis", STRENGTH_INDICATORS)
        assert upper == lower
        assert upper > 0


# ---------------------------------------------------------------------------
# Class: Tier 2 — LLM coherence screen
# ---------------------------------------------------------------------------

class TestTier2LLMScreen:
    """Tests for llm_coherence_check behavior (API mocked throughout)."""

    # ── Prompt content restriction ─────────────────────────────────────────

    def test_llm_prompt_contains_only_classification_and_rationale(self):
        """
        The LLM must receive ONLY classification + rationale — no fragment text,
        no gold standard, no study metadata (spec Section 2.4 Tier 2 rationale).
        """
        fragment_text = "UNIQUE_FRAGMENT_SENTINEL_XYZ"
        gold_standard  = "UNIQUE_GOLD_SENTINEL_ABC"

        captured_payload = {}

        def mock_post(url, headers, json, timeout):  # noqa: ARG001
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "content": [{"text": "coherent"}]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("src.scoring.coherence.requests.post", side_effect=mock_post), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm_coherence_check("sound", "The evaluation demonstrates sound reasoning.")

        prompt_sent = captured_payload["messages"][0]["content"]
        assert fragment_text not in prompt_sent, (
            "Fragment text must NOT be included in LLM coherence prompt"
        )
        assert gold_standard not in prompt_sent, (
            "Gold standard must NOT be included in LLM coherence prompt"
        )
        # The prompt must contain the classification and rationale
        assert "sound" in prompt_sent
        assert "demonstrates sound reasoning" in prompt_sent

    def test_llm_prompt_template_fields(self):
        """LLM_COHERENCE_PROMPT uses only {classification} and {rationale} fields."""
        import string
        formatter = string.Formatter()
        field_names = {
            field_name
            for _, field_name, _, _ in formatter.parse(LLM_COHERENCE_PROMPT)
            if field_name is not None
        }
        assert field_names == {"classification", "rationale"}, (
            f"LLM prompt must reference only classification and rationale, got: {field_names}"
        )

    # ── Agreement → assign ─────────────────────────────────────────────────

    def test_llm_returns_coherent_when_response_says_coherent(self):
        with patch("src.scoring.coherence.requests.post") as mock_post, \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_post.return_value.json.return_value = {
                "content": [{"text": "coherent"}]
            }
            mock_post.return_value.raise_for_status = MagicMock()

            result, raw = llm_coherence_check("sound", "The reasoning is sound.")

        assert result == "coherent"

    def test_llm_returns_incoherent_when_response_says_incoherent(self):
        with patch("src.scoring.coherence.requests.post") as mock_post, \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_post.return_value.json.return_value = {
                "content": [{"text": "incoherent"}]
            }
            mock_post.return_value.raise_for_status = MagicMock()

            result, _ = llm_coherence_check("not_sound", "The argument lacks evidence.")

        assert result == "incoherent"

    def test_incoherent_checked_before_coherent_substring(self):
        """
        'incoherent' contains 'coherent' as a substring; code must check
        'incoherent' first to avoid false-positives.
        """
        with patch("src.scoring.coherence.requests.post") as mock_post, \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_post.return_value.json.return_value = {
                "content": [{"text": "incoherent"}]
            }
            mock_post.return_value.raise_for_status = MagicMock()

            result, _ = llm_coherence_check("sound", "some rationale")

        assert result == "incoherent", (
            "Must return 'incoherent', not 'coherent', when response text is 'incoherent'"
        )

    # ── API failures → ambiguous ───────────────────────────────────────────

    def test_missing_api_key_returns_ambiguous(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result, raw = llm_coherence_check("sound", "some rationale")

        assert result == "ambiguous"
        assert "ANTHROPIC_API_KEY" in raw

    def test_api_exception_returns_ambiguous(self):
        with patch("src.scoring.coherence.requests.post", side_effect=Exception("network error")), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result, raw = llm_coherence_check("sound", "some rationale")

        assert result == "ambiguous"
        assert "error" in raw

    def test_unrecognized_llm_response_returns_ambiguous(self):
        with patch("src.scoring.coherence.requests.post") as mock_post, \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_post.return_value.json.return_value = {
                "content": [{"text": "I am not sure"}]
            }
            mock_post.return_value.raise_for_status = MagicMock()

            result, _ = llm_coherence_check("sound", "some rationale")

        assert result == "ambiguous"

    # ── Missing inputs ────────────────────────────────────────────────────

    def test_none_classification_returns_incoherent_without_api_call(self):
        with patch("src.scoring.coherence.requests.post") as mock_post:
            result, raw = llm_coherence_check(None, "some rationale")

        assert result == "incoherent"
        assert raw == "missing_input"
        mock_post.assert_not_called()

    def test_none_rationale_returns_incoherent_without_api_call(self):
        with patch("src.scoring.coherence.requests.post") as mock_post:
            result, raw = llm_coherence_check("sound", None)

        assert result == "incoherent"
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Class: validate_coherence (full tiered logic)
# ---------------------------------------------------------------------------

class TestValidateCoherence:
    """Tests for the combined tiered validate_coherence function."""

    REQUIRED_KEYS = {
        "coherence_rule_based",
        "strength_indicator_count",
        "weakness_indicator_count",
        "coherence_llm",
        "coherence_final",
        "manual_review_flag",
    }

    def test_output_contains_all_required_schema_keys(self):
        result = validate_coherence("sound", STRONG_SOUND_RATIONALE)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    # ── Tier 1 confident → no LLM call ────────────────────────────────────

    def test_high_diff_sound_assigns_without_llm(self):
        """
        When |strength - weakness| ≥ CONFIDENCE_THRESHOLD and classification is
        'sound' with strength-dominated rationale → coherent, llm='not_checked'.
        """
        with patch("src.scoring.coherence.llm_coherence_check") as mock_llm:
            result = validate_coherence("sound", STRONG_SOUND_RATIONALE)

        mock_llm.assert_not_called()
        assert result["coherence_final"] == "coherent"
        assert result["coherence_llm"] == "not_checked"
        assert result["manual_review_flag"] is False

    def test_high_diff_not_sound_assigns_without_llm(self):
        """
        'not_sound' + weakness-dominated rationale → coherent, llm='not_checked'.
        """
        with patch("src.scoring.coherence.llm_coherence_check") as mock_llm:
            result = validate_coherence("not_sound", STRONG_NOT_SOUND_RATIONALE)

        mock_llm.assert_not_called()
        assert result["coherence_final"] == "coherent"
        assert result["coherence_llm"] == "not_checked"

    def test_mismatch_high_diff_assigns_incoherent_without_llm(self):
        """
        'sound' + weakness-dominated rationale with diff ≥ threshold → incoherent,
        no LLM call.
        """
        s = _strength_count(WEAK_SOUND_MISMATCH_RATIONALE)
        w = _weakness_count(WEAK_SOUND_MISMATCH_RATIONALE)
        if abs(s - w) < CONFIDENCE_THRESHOLD:
            pytest.skip("Fixture diff below threshold; adjust fixture if needed")

        with patch("src.scoring.coherence.llm_coherence_check") as mock_llm:
            result = validate_coherence("sound", WEAK_SOUND_MISMATCH_RATIONALE)

        mock_llm.assert_not_called()
        assert result["coherence_final"] == "incoherent"

    # ── Tier 2 routing ────────────────────────────────────────────────────

    def test_balanced_rationale_routes_to_tier2(self):
        """
        When |strength - weakness| < CONFIDENCE_THRESHOLD, must call LLM.
        """
        s = _strength_count(BALANCED_RATIONALE)
        w = _weakness_count(BALANCED_RATIONALE)
        assert abs(s - w) < CONFIDENCE_THRESHOLD, "Fixture must be below threshold"

        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")) as mock_llm:
            validate_coherence("sound", BALANCED_RATIONALE)

        mock_llm.assert_called_once()

    def test_empty_rationale_routes_to_tier2(self):
        """
        Empty/None rationale → diff=0 < threshold → must attempt Tier 2 (or
        fast-exit via llm_coherence_check's own None guard).
        Verified by checking coherence_llm != 'not_checked'.
        """
        result = validate_coherence("sound", None)
        # LLM fast-exits with "incoherent" for None input, but was still called
        assert result["coherence_llm"] != "not_checked"

    # ── Tier 2 agree → assign ─────────────────────────────────────────────

    def test_tiers_agree_coherent_assigns_coherent(self):
        # Need: rule_signal == "coherent" for "sound" AND diff < threshold (→ Tier 2).
        # Condition: strength_count > weakness_count, |diff| < 3.
        # "demonstrates" + "establishes" (2 strength), "missing" (1 weakness) → diff=1.
        # For "sound": 2 > 1 → rule_signal = "coherent".  1 < 3 → routes to LLM.
        slightly_coherent_rationale = (
            "The evaluation demonstrates and establishes a reasonable case, "
            "although one element is missing."
        )
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")):
            result = validate_coherence("sound", slightly_coherent_rationale)

        assert result["coherence_final"] == "coherent"
        assert result["manual_review_flag"] is False

    def test_tiers_agree_incoherent_assigns_incoherent(self):
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("incoherent", "incoherent")):
            # sound + balanced → rule_signal = incoherent; LLM also says incoherent
            result = validate_coherence("sound", BALANCED_RATIONALE)

        assert result["coherence_final"] == "incoherent"
        assert result["manual_review_flag"] is False

    # ── Tier 2 disagree → ambiguous + manual review ────────────────────────

    def test_tier_disagreement_sets_ambiguous_and_manual_review_flag(self):
        """
        When rule-based and LLM disagree, coherence_final='ambiguous' and
        manual_review_flag=True (spec: 'flag as ambiguous → route to manual review').
        """
        # rule_based_coherence_check("sound", BALANCED_RATIONALE) → "incoherent"
        # Inject LLM saying "coherent" → disagreement
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")):
            result = validate_coherence("sound", BALANCED_RATIONALE)

        # rule says incoherent, LLM says coherent → disagree
        assert result["coherence_rule_based"] == "incoherent"
        assert result["coherence_llm"] == "coherent"
        assert result["coherence_final"] == "ambiguous"
        assert result["manual_review_flag"] is True


# ---------------------------------------------------------------------------
# Class: validate_all_responses (batch)
# ---------------------------------------------------------------------------

class TestValidateAllResponses:
    """Tests for the batch validate_all_responses function."""

    def _make_db(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "classification_output": "sound",
            "rationale_text": STRONG_SOUND_RATIONALE,
            "error_flag": False,
        }
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def test_error_flagged_row_is_incoherent_without_llm(self):
        """
        Rows with error_flag=True must be assigned coherent='incoherent' and
        coherence_llm='not_checked' — no LLM call (already failed parsing).
        """
        db = self._make_db([{"error_flag": True, "rationale_text": None}])

        with patch("src.scoring.coherence.llm_coherence_check") as mock_llm:
            result = validate_all_responses(db)

        mock_llm.assert_not_called()
        assert result.iloc[0]["coherence_final"] == "incoherent"
        assert result.iloc[0]["coherence_llm"] == "not_checked"

    def test_output_has_all_coherence_columns(self):
        db = self._make_db([{}])
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")):
            result = validate_all_responses(db)

        for col in [
            "coherence_rule_based", "strength_indicator_count",
            "weakness_indicator_count", "coherence_llm",
            "coherence_final", "manual_review_flag",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_row_count_matches_input(self):
        db = self._make_db([{}, {}, {}])
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")):
            result = validate_all_responses(db)

        assert len(result) == len(db)

    def test_non_error_row_is_processed_normally(self):
        """Non-error rows run through the tiered pipeline and get coherence_final."""
        db = self._make_db([{"error_flag": False}])
        with patch("src.scoring.coherence.llm_coherence_check",
                   return_value=("coherent", "coherent")):
            result = validate_all_responses(db)

        assert result.iloc[0]["coherence_final"] in {"coherent", "incoherent", "ambiguous"}
