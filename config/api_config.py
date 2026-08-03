"""
API endpoint and authentication configuration (spec Section 2.2; Appendix G.1).

This is the AUTHORITATIVE source for API configuration.
src/api_client/config.py imports from here — do not maintain parallel copies.

BEFORE RUNNING THE BENCHMARK:
1. Set all six API key environment variables listed under api_key_env.
2. Verify model_id values are still current at execution time (models are
   selected as top-3 SOTA — IDs may change between now and execution, per
   spec Section 2.2).
3. Record the API version/snapshot date in your Appendix F.1 session log.

ENVIRONMENT VARIABLES REQUIRED:
    OPENAI_API_KEY      — GPT-5.6 Sol, Max Effort (OpenAI, Responses API)
    ANTHROPIC_API_KEY   — Claude Opus 5, thinking x high effort (Anthropic)
    GOOGLE_API_KEY      — Gemini 3 Pro Preview High (Google)
    DEEPSEEK_API_KEY    — DeepSeek V4 Pro (DeepSeek)
    MOONSHOT_API_KEY    — Kimi K3 (Moonshot AI)
    FIREWORKS_API_KEY   — GLM 5.2 (served via Fireworks, not Zhipu/BigModel)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# API configuration — one entry per benchmark model family
# ---------------------------------------------------------------------------
#
# Fields:
#   endpoint      — Full base URL for the completion/response API
#   model_id      — Provider-specific model identifier string
#   auth_type     — Authentication mechanism:
#                     'bearer'         → Authorization: Bearer <key> header
#                     'x-api-key'      → x-api-key header (Anthropic)
#                     'api_key_param'  → key= URL query parameter (Google)
#   api_key_env   — Name of the environment variable holding the API key
#   unsupported_params — Universal param names this specific model rejects
#                     (400 if sent), even if its API family generally accepts
#                     them. Consulted before the family-level PARAM_MAPPING
#                     in src/api_client/executor.py:filter_params.
#   reasoning_effort_payload — Extra fields merged into the request payload
#                     for models with a configurable reasoning/thinking
#                     effort level (the pilot's 5 "effort-configured"
#                     models). Absent for models that don't take one.
#
# model_id, endpoint, and reasoning_effort_payload below were supplied by
# the user, sourced from each provider's current official API docs
# (2026-08-02) — no longer placeholders. Structural mismatches between
# these shapes and the surrounding harness code are flagged inline where
# they required changes elsewhere (executor.py, parser.py,
# model_params.py) rather than silently papered over.

API_CONFIG: dict[str, dict] = {
    # ── Closed models ──────────────────────────────────────────────────────
    "gpt_5_6_sol": {
        # STRUCTURAL FLAG: uses OpenAI's Responses API (/v1/responses), not
        # Chat Completions — the "reasoning" object below is a Responses-API
        # shape only and 400s against /v1/chat/completions. This required a
        # new "openai_responses" wire family (see MODEL_API_FAMILY below,
        # PARAM_MAPPING["openai_responses"] in config/model_params.py, the
        # openai_responses branch in executor.build_request_payload, and
        # parser.extract_response_content / parser.get_reasoning_tokens'
        # "output" / "output_tokens_details" handling). The response
        # envelope shape (output[] / usage.output_tokens_details) follows
        # OpenAI's documented Responses API conventions but was not
        # independently re-verified against live docs in this session —
        # confirm against a real response before running for real.
        "endpoint": "https://api.openai.com/v1/responses",
        "model_id": "gpt-5.6-sol",
        "auth_type": "bearer",
        "api_key_env": "OPENAI_API_KEY",
        # CONFIRMED via live diagnostic calls (2026-08-02): 400 "Unsupported
        # parameter" for both 'temperature' and 'top_p' — same pattern as
        # Claude Opus 5 (rejects sampling params outright).
        "unsupported_params": frozenset({"temperature", "top_p"}),
        # "mode": "pro" REMOVED — CONFIRMED via live diagnostic calls
        # (2026-08-03): it triggers multiple parallel/sequential reasoning
        # passes (observed 3 "reasoning" output blocks vs. 1 without it,
        # each re-injecting the full input context) that blow past
        # max_output_tokens=4000 as a soft target rather than a hard cap —
        # 5 of 12 real pilot calls overshot up to 3.1x (12,588 tokens).
        # Dropping "mode" while keeping effort:"max" (preserving the
        # spec/README "Max Effort" designation) reproduced the SAME prompt
        # at 594 output tokens, comfortably under budget, status:"completed".
        "reasoning_effort_payload": {
            "reasoning": {"effort": "max"},
        },
    },
    "claude_opus_5": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model_id": "claude-opus-5",
        "auth_type": "x-api-key",
        "api_key_env": "ANTHROPIC_API_KEY",
        # Claude Opus 5 rejects all sampling params outright (400) — grounded
        # via the claude-api skill. STANDARD_PARAMS sends temperature to every
        # model by default; this must be stripped in filter_params.
        "unsupported_params": frozenset({
            "temperature", "top_p", "frequency_penalty", "presence_penalty",
        }),
        # Grounded via the claude-api skill: Claude Opus 5 thinking is on by
        # default via {"type": "adaptive"}; effort lives in output_config.
        "reasoning_effort_payload": {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        },
    },
    "gemini_3_pro_preview_high": {
        # STRUCTURAL FLAG: uses Google's Interactions API
        # (client.interactions.create via the google-genai SDK), confirmed
        # GA (not preview/beta) via https://ai.google.dev/gemini-api/docs
        # and https://ai.google.dev/gemini-api/docs/interactions-overview —
        # a different surface than the classic generateContent REST
        # endpoint the rest of this file assumes. This required a new
        # "google_interactions_sdk" family (see MODEL_API_FAMILY below,
        # PARAM_MAPPING["google_interactions_sdk"] in config/model_params.py,
        # and the SDK-based branch in executor.execute_api_request —
        # build_request_headers/build_request_payload/build_endpoint_url
        # are bypassed entirely for this model, so "auth_type" below is
        # informational only, not consulted).
        #
        # model_id "gemini-3.1-pro-preview" and the field names below
        # (output_text, usage.total_input_tokens/total_output_tokens/
        # total_thought_tokens, GenerationConfig.thinking_level/
        # max_output_tokens) were confirmed by directly introspecting the
        # installed google-genai 2.16.0 SDK's type definitions (pydantic
        # model_fields) under Python 3.11 — not a live API call, but real
        # source, not a guess. GenerationConfig has NO temperature/top_p
        # fields at all for this API (see PARAM_MAPPING). thinking_level's
        # type allows "minimal"/"low"/"medium"/"high" — broader than what
        # was described as valid for this model; kept "high" as specified.
        #
        # google-genai>=2.3.0 (required for .interactions to exist) needs
        # Python 3.10+ — see requirements.txt.
        "model_id": "gemini-3.1-pro-preview",
        "auth_type": "api_key_param",  # informational only — see note above
        "api_key_env": "GOOGLE_API_KEY",
        "reasoning_effort_payload": {
            "thinking_level": "high",
        },
    },
    # ── Open models ────────────────────────────────────────────────────────
    "kimi_k3": {
        # Endpoint path CONFIRMED correct via live call (2026-08-02) — got a
        # structured API-level 400, not a 404, so the assumed
        # "/chat/completions" suffix was right.
        "endpoint": "https://api.moonshot.ai/v1/chat/completions",
        "model_id": "kimi-k3",
        "auth_type": "bearer",
        "api_key_env": "MOONSHOT_API_KEY",
        # Thinking is always on for K3 and cannot be disabled — reasoning_effort
        # only controls depth, not an on/off switch.
        #
        # CONFIRMED via live diagnostic calls (2026-08-02): 400 "invalid
        # temperature: only 1 is allowed" and (after fixing that) 400
        # "invalid top_p: only 0.95 is allowed" — not merely unsupported
        # (which would mean dropping them), both must be forced to exact
        # values. Placed here (not unsupported_params) because this dict
        # merges in AFTER filter_params via payload.update(), overriding
        # STANDARD_PARAMS' temperature=0/top_p=1.0 rather than dropping the keys.
        "reasoning_effort_payload": {
            "reasoning_effort": "max",
            "temperature": 1,
            "top_p": 0.95,
        },
    },
    "deepseek_v4_pro": {
        # Endpoint not supplied — kept as the prior selection, still unverified.
        "endpoint": "https://api.deepseek.com/v1/chat/completions",  # verify at execution
        "model_id": "deepseek-v4-pro",
        "auth_type": "bearer",
        "api_key_env": "DEEPSEEK_API_KEY",
        # The OpenAI SDK's extra_body={"thinking": {"type": "enabled"}} is an
        # SDK-level mechanism for injecting extra top-level JSON fields — this
        # codebase builds raw JSON payloads directly, so the wire-level
        # equivalent is just sending "thinking" as its own top-level key
        # alongside reasoning_effort. No structural mismatch: both merge in
        # via the existing openai_compatible top-level payload.update().
        "reasoning_effort_payload": {
            "reasoning_effort": "high",
            "thinking": {"type": "enabled"},
        },
    },
    "glm_5_2": {
        # STRUCTURAL FLAG: provider switched from Zhipu/BigModel to
        # Fireworks — different endpoint AND different auth env var
        # (FIREWORKS_API_KEY, not ZHIPU_API_KEY). Cascades to the module
        # docstring above, README.md's API Key Setup table, and CLAUDE.md.
        "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model_id": "accounts/fireworks/models/glm-5p2",
        "auth_type": "bearer",
        "api_key_env": "FIREWORKS_API_KEY",
        # Do not add a "thinking" key alongside reasoning_effort here —
        # Fireworks rejects requests that set both.
        "reasoning_effort_payload": {
            "reasoning_effort": "max",
        },
    },
}

# ---------------------------------------------------------------------------
# API family → wire-format mapping
# ---------------------------------------------------------------------------
# Used by executor to select the correct request payload structure.

MODEL_API_FAMILY: dict[str, str] = {
    "gpt_5_6_sol":                "openai_responses",  # NOT openai_compatible — see STRUCTURAL FLAG above
    "claude_opus_5":              "anthropic",
    "gemini_3_pro_preview_high":  "google_interactions_sdk",  # NOT "google" (generateContent REST) — see STRUCTURAL FLAG above
    "kimi_k3":                    "openai_compatible",
    "deepseek_v4_pro":            "openai_compatible",
    "glm_5_2":                    "openai_compatible",
}

# ---------------------------------------------------------------------------
# Model metadata (spec Section 2.2)
# ---------------------------------------------------------------------------

# Architecture classification (used in analysis H.2 / H.5 tests)
# Open/closed classification for the 3 open-weight models follows their
# predecessor lineages (Kimi, DeepSeek, GLM) — flagged for a one-line
# confirmation since it feeds architecture-comparison stats.
ARCHITECTURE_MAP: dict[str, str] = {
    "gpt_5_6_sol":                "closed",
    "claude_opus_5":              "closed",
    "gemini_3_pro_preview_high":  "closed",
    "kimi_k3":                    "open",
    "deepseek_v4_pro":            "open",
    "glm_5_2":                    "open",
}

# Canonical display order: closed models first, then open
MODEL_ORDER: list[str] = [
    "gpt_5_6_sol", "claude_opus_5", "gemini_3_pro_preview_high",  # closed
    "kimi_k3", "deepseek_v4_pro", "glm_5_2",                       # open
]

# Primary diagnostic model (receives full failure-mode coding in H.3)
PRIMARY_MODEL: str = "gpt_5_6_sol"

# ---------------------------------------------------------------------------
# Anthropic-specific header
# ---------------------------------------------------------------------------
# Update to the current stable version at execution time.

ANTHROPIC_API_VERSION: str = "2023-06-01"
