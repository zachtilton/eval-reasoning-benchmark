"""
API configuration, model parameters, and project path constants.

All constants used across G.1–G.3 modules are centralized here so that
config is separated from logic (per code standards in CLAUDE.md).

Note: Model IDs and endpoints should be verified and updated at execution
time per spec Section 2.2 ("API versions documented at time of execution").
Model selections are subject to change to ensure top-3 reasoning SOTA for
each architecture category.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Resolve from this file: src/api_client/config.py → src/api_client → src → root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESPONSES_DIR = DATA_DIR / "responses"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"

RESPONSE_DB_PATH = RESPONSES_DIR / "response_database.csv"
FAILED_CALLS_LOG = LOGS_DIR / "failed_calls.jsonl"
EXECUTION_ORDER_LOG = LOGS_DIR / "execution_order_log.csv"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# API_CONFIG: one entry per model family used in the benchmark.
# Closed models: Claude Opus 4.6, GPT 5.2, Gemini 3 Pro
# Open models:   DeepSeek V3.2 Thinking, Kimi K2 Thinking, GLM 4.7
API_CONFIG: dict[str, dict] = {
    "gpt_5": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-5.2",
        "auth_type": "bearer",
        "api_key_env": "OPENAI_API_KEY",
    },
    # G.1 appendix incorrectly used 'claude_opus_4_5'; spec and G.3 both
    # specify claude_opus_4_6.  Model ID updated to match CLAUDE.md.
    "claude_opus_4_6": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model_id": "claude-opus-4-6",
        "auth_type": "x-api-key",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini_3_pro": {
        "endpoint": (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-3-pro:generateContent"
        ),
        "model_id": "gemini-3-pro",
        "auth_type": "api_key_param",  # key passed as URL query parameter
        "api_key_env": "GOOGLE_API_KEY",
    },
    "deepseek_v3": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model_id": "deepseek-chat",
        "auth_type": "bearer",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi_k2": {
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "model_id": "kimi-k2",  # verify latest Kimi K2 Thinking ID at execution
        "auth_type": "bearer",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "glm_4_7": {
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model_id": "glm-4-plus",  # verify latest GLM 4.7 ID at execution
        "auth_type": "bearer",
        "api_key_env": "ZHIPU_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# Standardized API parameters (spec Section 2.2)
# ---------------------------------------------------------------------------

# Universal parameters; unsupported ones are filtered per model in executor.py.
STANDARD_PARAMS: dict[str, int | float] = {
    "temperature": 0,        # Maximum determinism
    "max_tokens": 500,       # Output limit
    "top_p": 1.0,            # No nucleus sampling restriction
    "frequency_penalty": 0,  # No repetition penalty
    "presence_penalty": 0,   # No topic diversity penalty
}

# Expected token ranges for cost estimation and pre-flight validation
EXPECTED_TOKENS: dict[str, int] = {
    "input_avg": 750,    # prompt + fragment
    "output_avg": 250,   # classification + rationale
    "input_max": 1200,   # upper validation bound
    "output_max": 500,   # API hard limit
}

# Anthropic API version string — update at execution time per spec
ANTHROPIC_API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Parameter mapping: universal names → provider-specific names
# None means the parameter is not supported and should be omitted.
# ---------------------------------------------------------------------------

PARAM_MAPPING: dict[str, dict[str, str | None]] = {
    "openai_compatible": {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
    },
    "anthropic": {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": None,   # not supported
        "presence_penalty": None,    # not supported
    },
    "google": {
        "temperature": "temperature",
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "frequency_penalty": None,   # not supported
        "presence_penalty": None,    # not supported
    },
}

# Map model family identifiers to their wire-format family
MODEL_API_FAMILY: dict[str, str] = {
    "gpt_5": "openai_compatible",
    "claude_opus_4_6": "anthropic",
    "gemini_3_pro": "google",
    "deepseek_v3": "openai_compatible",
    "kimi_k2": "openai_compatible",
    "glm_4_7": "openai_compatible",
}

# ---------------------------------------------------------------------------
# Execution parameters (spec Section 2.2–2.3)
# ---------------------------------------------------------------------------

MODELS: list[str] = list(API_CONFIG.keys())
PROMPT_CONDITIONS: list[str] = ["zero_shot", "few_shot"]
RUNS_PER_COMBO: int = 3           # independent runs per fragment-model-prompt
INTER_RUN_DELAY_SECONDS: int = 5  # delay between runs within a combo
REQUEST_TIMEOUT_SECONDS: int = 60 # HTTP request timeout

TOTAL_FRAGMENTS: int = 150
# Total expected calls: 150 × 6 × 2 × 3 = 5,400
TOTAL_EXPECTED_CALLS: int = (
    TOTAL_FRAGMENTS * len(MODELS) * len(PROMPT_CONDITIONS) * RUNS_PER_COMBO
)
