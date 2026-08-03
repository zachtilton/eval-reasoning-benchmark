"""
Unit tests for the pilot-driven config wiring in config/api_config.py and
src/api_client/executor.py: per-model unsupported_params stripping and
reasoning_effort_payload merging.
"""

from __future__ import annotations

from config.api_config import API_CONFIG
from config.model_params import STANDARD_PARAMS
from src.api_client.executor import build_request_payload, filter_params


class TestVerifiedModelValues:
    """Regression tests for the user-supplied verified model_id/endpoint/auth values."""

    def test_gpt_5_6_sol_uses_responses_endpoint(self):
        config = API_CONFIG["gpt_5_6_sol"]
        assert config["endpoint"] == "https://api.openai.com/v1/responses"
        assert config["model_id"] == "gpt-5.6-sol"

    def test_gemini_model_id(self):
        assert API_CONFIG["gemini_3_pro_preview_high"]["model_id"] == "gemini-3.1-pro-preview"

    def test_kimi_k3_endpoint_domain(self):
        assert API_CONFIG["kimi_k3"]["endpoint"].startswith("https://api.moonshot.ai/")

    def test_deepseek_model_id(self):
        assert API_CONFIG["deepseek_v4_pro"]["model_id"] == "deepseek-v4-pro"

    def test_glm_5_2_uses_fireworks(self):
        config = API_CONFIG["glm_5_2"]
        assert config["endpoint"] == "https://api.fireworks.ai/inference/v1/chat/completions"
        assert config["model_id"] == "accounts/fireworks/models/glm-5p2"
        assert config["api_key_env"] == "FIREWORKS_API_KEY"


class TestFilterParamsUnsupported:
    def test_claude_opus_5_strips_sampling_params(self):
        result = filter_params("claude_opus_5", STANDARD_PARAMS)
        assert "temperature" not in result
        assert "top_p" not in result
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        # max_tokens survives — it's not in unsupported_params
        assert "max_tokens" in result

    def test_gpt_5_6_sol_strips_sampling_params(self):
        # CONFIRMED via live diagnostic calls (2026-08-02): 400 "Unsupported
        # parameter" for both 'temperature' and 'top_p'.
        result = filter_params("gpt_5_6_sol", STANDARD_PARAMS)
        assert "temperature" not in result
        assert "top_p" not in result

    def test_other_models_keep_temperature(self):
        for model in ("kimi_k3", "deepseek_v4_pro", "glm_5_2"):
            result = filter_params(model, STANDARD_PARAMS)
            assert "temperature" in result, f"{model} should not have temperature stripped"

    def test_gemini_interactions_api_has_no_temperature_or_top_p(self):
        """
        GenerationConfig on the Interactions API has no temperature/top_p
        fields at all (confirmed via SDK source introspection) — unlike
        Claude Opus 5's unsupported_params, this isn't model-specific
        stricture, it's genuinely absent from the family's param mapping.
        """
        result = filter_params("gemini_3_pro_preview_high", STANDARD_PARAMS)
        assert "temperature" not in result
        assert "top_p" not in result
        assert "max_output_tokens" in result  # family-mapped name
        assert "maxOutputTokens" not in result  # that's the old "google" (REST) family's name


class TestBuildRequestPayloadReasoningEffort:
    def test_claude_opus_5_gets_thinking_and_effort(self):
        config = API_CONFIG["claude_opus_5"]
        payload = build_request_payload(config, "prompt text", "claude_opus_5")
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "high"}
        assert "temperature" not in payload

    def test_gpt_5_6_sol_uses_responses_api_shape(self):
        """GPT-5.6 Sol is the openai_responses family, not Chat Completions."""
        config = API_CONFIG["gpt_5_6_sol"]
        payload = build_request_payload(config, "prompt text", "gpt_5_6_sol")
        assert payload["input"] == "prompt text"
        assert "messages" not in payload
        # "mode": "pro" removed (2026-08-03) — caused multi-pass reasoning
        # that overshot max_output_tokens up to 3.1x in live pilot calls.
        assert payload["reasoning"] == {"effort": "max"}
        # max_tokens is renamed max_output_tokens in this family
        assert "max_output_tokens" in payload
        assert "max_tokens" not in payload

    def test_kimi_k3_gets_reasoning_effort(self):
        config = API_CONFIG["kimi_k3"]
        payload = build_request_payload(config, "prompt text", "kimi_k3")
        assert payload["reasoning_effort"] == "max"
        assert payload["messages"] == [{"role": "user", "content": "prompt text"}]

    def test_deepseek_v4_pro_gets_reasoning_effort_and_thinking(self):
        config = API_CONFIG["deepseek_v4_pro"]
        payload = build_request_payload(config, "prompt text", "deepseek_v4_pro")
        assert payload["reasoning_effort"] == "high"
        assert payload["thinking"] == {"type": "enabled"}

    def test_glm_5_2_gets_reasoning_effort_without_thinking(self):
        config = API_CONFIG["glm_5_2"]
        payload = build_request_payload(config, "prompt text", "glm_5_2")
        assert payload["reasoning_effort"] == "max"
        # Fireworks rejects requests combining reasoning_effort with "thinking"
        assert "thinking" not in payload
