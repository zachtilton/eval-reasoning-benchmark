"""
API endpoint and authentication configuration (spec Section 2.2; Appendix G.1).

This is the AUTHORITATIVE source for API configuration.
src/api_client/config.py imports from here — do not maintain parallel copies.

BEFORE RUNNING THE BENCHMARK:
1. Set all six API key environment variables listed under api_key_env.
2. Verify model IDs are current (models are selected as top-3 SOTA at execution
   time — IDs may change between now and execution, per spec Section 2.2).
3. Record the API version/snapshot date in your Appendix F.1 session log.

ENVIRONMENT VARIABLES REQUIRED:
    OPENAI_API_KEY      — GPT 5.2 (OpenAI)
    ANTHROPIC_API_KEY   — Claude Opus 4.6 (Anthropic)
    GOOGLE_API_KEY      — Gemini 3 Pro (Google)
    DEEPSEEK_API_KEY    — DeepSeek V3.2 Thinking (DeepSeek)
    MOONSHOT_API_KEY    — Kimi K2 Thinking (Moonshot AI)
    ZHIPU_API_KEY       — GLM 4.7 (Zhipu AI / BigModel)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# API configuration — one entry per benchmark model family
# ---------------------------------------------------------------------------
#
# Fields:
#   endpoint      — Full base URL for the chat/completion API
#   model_id      — Provider-specific model identifier string
#   auth_type     — Authentication mechanism:
#                     'bearer'         → Authorization: Bearer <key> header
#                     'x-api-key'      → x-api-key header (Anthropic)
#                     'api_key_param'  → key= URL query parameter (Google)
#   api_key_env   — Name of the environment variable holding the API key
#
# IMPORTANT: Verify model_id values against provider release notes at
# execution time. IDs below reflect the Jan 2026 SOTA selection (spec §2.2).

API_CONFIG: dict[str, dict] = {
    # ── Closed models ──────────────────────────────────────────────────────
    "gpt_5": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-5.2",         # verify at execution
        "auth_type": "bearer",
        "api_key_env": "OPENAI_API_KEY",
    },
    # Note: appendix G.1 incorrectly used key 'claude_opus_4_5' with model ID
    # 'claude-opus-4-5-20251101'. Spec §2.2 and CLAUDE.md both specify 4.6.
    "claude_opus_4_6": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model_id": "claude-opus-4-6",  # verify at execution
        "auth_type": "x-api-key",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini_3_pro": {
        "endpoint": (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-3-pro:generateContent"
        ),
        "model_id": "gemini-3-pro",    # verify at execution
        "auth_type": "api_key_param",  # key passed as URL query parameter
        "api_key_env": "GOOGLE_API_KEY",
    },
    # ── Open models ────────────────────────────────────────────────────────
    "deepseek_v3": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model_id": "deepseek-chat",   # verify at execution (V3.2 Thinking)
        "auth_type": "bearer",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi_k2": {
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "model_id": "kimi-k2",         # verify at execution (K2 Thinking)
        "auth_type": "bearer",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "glm_4_7": {
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model_id": "glm-4-plus",      # verify at execution (GLM 4.7)
        "auth_type": "bearer",
        "api_key_env": "ZHIPU_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# API family → wire-format mapping
# ---------------------------------------------------------------------------
# Used by executor to select the correct request payload structure.

MODEL_API_FAMILY: dict[str, str] = {
    "gpt_5":          "openai_compatible",
    "claude_opus_4_6": "anthropic",
    "gemini_3_pro":   "google",
    "deepseek_v3":    "openai_compatible",
    "kimi_k2":        "openai_compatible",
    "glm_4_7":        "openai_compatible",
}

# ---------------------------------------------------------------------------
# Model metadata (spec Section 2.2)
# ---------------------------------------------------------------------------

# Architecture classification (used in analysis H.2 / H.5 tests)
ARCHITECTURE_MAP: dict[str, str] = {
    "gpt_5":          "closed",
    "claude_opus_4_6": "closed",
    "gemini_3_pro":   "closed",
    "deepseek_v3":    "open",
    "kimi_k2":        "open",
    "glm_4_7":        "open",
}

# Canonical display order: closed models first, then open
MODEL_ORDER: list[str] = [
    "gpt_5", "claude_opus_4_6", "gemini_3_pro",  # closed
    "deepseek_v3", "kimi_k2", "glm_4_7",          # open
]

# Primary diagnostic model (receives full failure-mode coding in H.3)
PRIMARY_MODEL: str = "gpt_5"

# ---------------------------------------------------------------------------
# Anthropic-specific header
# ---------------------------------------------------------------------------
# Update to the current stable version at execution time.

ANTHROPIC_API_VERSION: str = "2023-06-01"
