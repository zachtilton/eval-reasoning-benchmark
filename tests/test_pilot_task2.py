"""
Unit tests for src/pilot/task2_pilot_batch.py.

Follows the mocked-requests.post + patch.dict(os.environ) pattern already
established in tests/test_coherence.py — this repo's only prior example of
mocking an outbound LLM API call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pilot.fragments import load_fragments
from src.pilot.task2_pilot_batch import (
    build_pilot_matrix,
    process_pilot_result,
    run_single_pilot_call,
    run_task2,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_fragments.jsonl"


def _openai_shaped_response(reasoning_tokens=20):
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": '{"classification": "sound", "rationale": "Sufficient evidence and argument shown here."}'}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


def _anthropic_shaped_response():
    resp = MagicMock()
    resp.json.return_value = {
        "content": [{"type": "text", "text": '{"classification": "not sound", "rationale": "Insufficient evidence and weak argument shown here."}'}],
        "usage": {"input_tokens": 90, "output_tokens": 40},  # no completion_tokens_details
    }
    resp.raise_for_status = MagicMock()
    return resp


def _mock_google_interaction(reasoning_tokens=15):
    """
    Gemini 3.1 Pro Preview High uses the Interactions API (google-genai SDK),
    not requests.post — this mocks the Interaction object's .output_text
    and .usage fields, per the shape confirmed via SDK source introspection
    (Interaction.model_fields, Usage.model_fields in google-genai 2.16.0).
    """
    interaction = MagicMock()
    interaction.status = "completed"
    interaction.output_text = '{"classification": "sound", "rationale": "Sufficient evidence and argument shown here."}'
    interaction.usage.total_input_tokens = 80
    interaction.usage.total_output_tokens = 30
    interaction.usage.total_thought_tokens = reasoning_tokens
    return interaction


def _openai_responses_shaped_response(reasoning_tokens=35):
    resp = MagicMock()
    resp.json.return_value = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"classification": "sound", "rationale": "Sufficient evidence and argument shown here."}'}],
            }
        ],
        "usage": {
            "input_tokens": 110,
            "output_tokens": 55,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestBuildPilotMatrix:
    def test_cross_product(self):
        matrix = build_pilot_matrix(
            ["R1", "R2"], models=["m1", "m2", "m3"], prompts=["zero_shot", "few_shot"]
        )
        assert len(matrix) == 2 * 3 * 2
        assert ("R1", "m1", "zero_shot") in matrix
        assert ("R2", "m3", "few_shot") in matrix


class TestRunSinglePilotCallDryRun:
    def test_dry_run_returns_canned_success(self):
        fragments = load_fragments(FIXTURE_PATH)
        result = run_single_pilot_call(fragments[0], "claude_opus_5", "zero_shot", dry_run=True)
        assert result["status"] == "success"
        assert result["parsed_response"]["token_count_input"] > 0


class TestRunSinglePilotCallReasoningDispatch:
    def test_openai_family_reports_reasoning_tokens(self):
        fragments = load_fragments(FIXTURE_PATH)
        with patch("src.api_client.executor.requests.post", return_value=_openai_shaped_response(20)), \
             patch.dict("os.environ", {"MOONSHOT_API_KEY": "test-key"}):
            result = run_single_pilot_call(fragments[0], "kimi_k3", "zero_shot")
        assert result["status"] == "success"
        assert result["parsed_response"]["reasoning_tokens"] == 20

    def test_anthropic_family_reports_none(self):
        fragments = load_fragments(FIXTURE_PATH)
        with patch("src.api_client.executor.requests.post", return_value=_anthropic_shaped_response()), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = run_single_pilot_call(fragments[0], "claude_opus_5", "zero_shot")
        assert result["status"] == "success"
        assert result["parsed_response"]["reasoning_tokens"] is None

    def test_gemini_interactions_api_reports_reasoning_tokens(self):
        """Gemini 3.1 Pro Preview High uses google-genai's Client, not requests.post."""
        fragments = load_fragments(FIXTURE_PATH)
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = _mock_google_interaction(15)
        with patch("google.genai.Client", return_value=mock_client) as mock_client_cls, \
             patch("src.api_client.executor.requests.post") as mock_post, \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            result = run_single_pilot_call(fragments[0], "gemini_3_pro_preview_high", "zero_shot")

        assert result["status"] == "success"
        assert result["parsed_response"]["reasoning_tokens"] == 15
        assert result["parsed_response"]["token_count_input"] == 80
        assert result["parsed_response"]["token_count_output"] == 30
        mock_post.assert_not_called()  # must not fall through to the requests-based path
        mock_client_cls.assert_called_once_with(api_key="test-key")

        # generation_config sent to the SDK: no temperature/top_p (not part of
        # this API's GenerationConfig), thinking_level from reasoning_effort_payload
        _, call_kwargs = mock_client.interactions.create.call_args
        assert call_kwargs["model"] == "models/gemini-3.1-pro-preview"
        assert fragments[0]["fragment"] in call_kwargs["input"]  # rendered prompt includes the fragment text
        gen_config = call_kwargs["generation_config"]
        assert "temperature" not in gen_config
        assert "top_p" not in gen_config
        assert gen_config["thinking_level"] == "high"
        assert gen_config["max_output_tokens"] == 4000  # STANDARD_PARAMS, raised after the pilot's live max_tokens finding

    def test_gemini_missing_output_text_raises(self):
        """
        Tests execute_google_interactions_request directly, not through
        run_single_pilot_call/make_api_call_with_retry — a real RuntimeError
        here would hit the retry wrapper's "unknown error, retry once" path
        and sleep ~10s for real before failing. Testing the unit directly
        avoids that and is more precise anyway.
        """
        from src.api_client.executor import execute_google_interactions_request
        from config.api_config import API_CONFIG

        mock_client = MagicMock()
        empty_interaction = MagicMock()
        empty_interaction.status = "completed"
        empty_interaction.output_text = None
        mock_client.interactions.create.return_value = empty_interaction
        with patch("google.genai.Client", return_value=mock_client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with pytest.raises(RuntimeError, match="no output_text"):
                execute_google_interactions_request(
                    API_CONFIG["gemini_3_pro_preview_high"], "prompt text", "gemini_3_pro_preview_high"
                )

    def test_gemini_incomplete_status_raises(self):
        """
        CONFIRMED via live diagnostic (2026-08-03): max_output_tokens=500 let
        thinking consume 477 tokens, leaving 19 for the answer and
        status="incomplete" — the truncated-output bug seen in the pilot
        run. max_tokens=4000 (STANDARD_PARAMS) avoids this in practice, but
        the guard must still catch it if a future budget is too tight again.
        """
        from src.api_client.executor import execute_google_interactions_request
        from config.api_config import API_CONFIG

        mock_client = MagicMock()
        incomplete_interaction = MagicMock()
        incomplete_interaction.status = "incomplete"
        incomplete_interaction.output_text = "The claim is"
        mock_client.interactions.create.return_value = incomplete_interaction
        with patch("google.genai.Client", return_value=mock_client), \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with pytest.raises(RuntimeError, match="incomplete"):
                execute_google_interactions_request(
                    API_CONFIG["gemini_3_pro_preview_high"], "prompt text", "gemini_3_pro_preview_high"
                )

    def test_gemini_missing_api_key_raises(self):
        from src.api_client.executor import execute_google_interactions_request
        from config.api_config import API_CONFIG

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                execute_google_interactions_request(
                    API_CONFIG["gemini_3_pro_preview_high"], "prompt text", "gemini_3_pro_preview_high"
                )

    def test_openai_responses_family_reports_reasoning_tokens(self):
        """GPT-5.6 Sol's Responses API shape: output[]/usage.output_tokens_details."""
        fragments = load_fragments(FIXTURE_PATH)
        with patch("src.api_client.executor.requests.post", return_value=_openai_responses_shaped_response(35)), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = run_single_pilot_call(fragments[0], "gpt_5_6_sol", "zero_shot")
        assert result["status"] == "success"
        assert result["parsed_response"]["reasoning_tokens"] == 35
        assert result["parsed_response"]["token_count_input"] == 110
        assert result["parsed_response"]["token_count_output"] == 55


class TestProcessPilotResult:
    def test_success_row_shape(self):
        result = {
            "status": "success",
            "parsed_response": {
                "token_count_input": 100,
                "token_count_output": 50,
                "reasoning_tokens": 20,
            },
        }
        row = process_pilot_result(result, "claude_opus_5", "zero_shot", "TEST-2020-001")
        assert row["model"] == "claude_opus_5"
        assert row["prompt_condition"] == "zero_shot"
        assert row["report_id"] == "TEST-2020-001"
        assert row["input_tokens"] == 100
        assert row["reasoning_tokens"] == 20
        assert row["cost"] is not None  # claude_opus_5 has verified rates

    def test_failure_row_has_none_fields(self):
        result = {"status": "failed", "parsed_response": None}
        row = process_pilot_result(result, "gpt_5_6_sol", "few_shot", "TEST-2020-002")
        assert row["input_tokens"] is None
        assert row["cost"] is None


class TestRunTask2:
    def test_raises_on_empty_pinned_ids(self):
        with pytest.raises(ValueError, match="No pinned report_ids"):
            run_task2(pinned_report_ids=[], dry_run=True, fragments_path=FIXTURE_PATH)

    def test_dry_run_produces_expected_row_count(self):
        df = run_task2(
            pinned_report_ids=["TEST-2020-001", "TEST-2020-002"],
            models=["claude_opus_5", "kimi_k3"],
            prompts=["zero_shot", "few_shot"],
            fragments_path=FIXTURE_PATH,
            dry_run=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2 * 2 * 2  # 2 report_ids x 2 models x 2 prompts

    def test_dry_run_writes_no_files(self, tmp_path):
        output_path = tmp_path / "pilot_usage_log.csv"
        response_dir = tmp_path / "responses"
        run_task2(
            pinned_report_ids=["TEST-2020-001"],
            models=["claude_opus_5"],
            prompts=["zero_shot"],
            output_path=output_path,
            response_dir=response_dir,
            fragments_path=FIXTURE_PATH,
            dry_run=True,
        )
        assert not output_path.exists()
        assert not response_dir.exists()

    def test_unknown_pinned_id_raises(self):
        with pytest.raises(ValueError, match="not found in fragments.jsonl"):
            run_task2(
                pinned_report_ids=["NOT-A-REAL-ID"],
                fragments_path=FIXTURE_PATH,
                dry_run=True,
            )

    def test_ambiguous_pinned_id_raises(self, tmp_path):
        """A report_id matching >1 fragment must fail loudly, not pick one."""
        ambiguous_path = tmp_path / "ambiguous_fragments.jsonl"
        ambiguous_path.write_text(
            '{"report_id": 1359, "criterion": "relevance", "paragraph_count": 1, "fragment": "Text A."}\n'
            '{"report_id": 1359, "criterion": "sustainability", "paragraph_count": 1, "fragment": "Text B."}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="match more than one fragment"):
            run_task2(
                pinned_report_ids=[1359],
                models=["claude_opus_5"],
                prompts=["zero_shot"],
                fragments_path=ambiguous_path,
                dry_run=True,
            )
