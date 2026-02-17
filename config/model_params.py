"""
Model parameters, API request configuration, and execution constants
(spec Section 2.2–2.3; Appendix G.1).

This is the AUTHORITATIVE source for all parameter and execution constants.
src/api_client/config.py imports from here — do not maintain parallel copies.

Design rationale:
- Temperature = 0 maximizes determinism; 3-run design captures residual
  variance at temperature 0 (spec Section 2.2).
- max_tokens = 500 gives headroom above expected 250-token outputs while
  keeping cost bounded.
- PARAM_MAPPING handles provider-specific parameter naming differences
  and silently drops unsupported parameters (None → omitted).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standardized API parameters (spec Section 2.2)
# ---------------------------------------------------------------------------

STANDARD_PARAMS: dict[str, int | float] = {
    "temperature": 0,        # Maximum determinism (spec §2.2)
    "max_tokens": 500,       # Output limit; expected output ~250 tokens
    "top_p": 1.0,            # No nucleus sampling restriction
    "frequency_penalty": 0,  # No repetition penalty
    "presence_penalty": 0,   # No topic diversity penalty
}

# ---------------------------------------------------------------------------
# Provider-specific parameter name mapping
# ---------------------------------------------------------------------------
# For each API family, maps universal parameter name → provider name.
# None means the parameter is not supported; executor will omit it silently.

PARAM_MAPPING: dict[str, dict[str, str | None]] = {
    "openai_compatible": {
        # GPT, DeepSeek, Kimi, GLM all use standard OpenAI field names
        "temperature":       "temperature",
        "max_tokens":        "max_tokens",
        "top_p":             "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty":  "presence_penalty",
    },
    "anthropic": {
        "temperature":       "temperature",
        "max_tokens":        "max_tokens",   # hoisted to top-level in payload
        "top_p":             "top_p",
        "frequency_penalty": None,           # not supported
        "presence_penalty":  None,           # not supported
    },
    "google": {
        # Gemini uses generationConfig sub-object with camelCase names
        "temperature":       "temperature",
        "max_tokens":        "maxOutputTokens",
        "top_p":             "topP",
        "frequency_penalty": None,           # not supported
        "presence_penalty":  None,           # not supported
    },
}

# ---------------------------------------------------------------------------
# Token expectations (spec Section 2.2)
# ---------------------------------------------------------------------------
# Used for pre-flight cost estimation and response validation.
# Actual counts recorded per-call in response_database.csv.

EXPECTED_TOKENS: dict[str, int] = {
    "input_avg":  750,   # prompt template + target fragment
    "output_avg": 250,   # classification line + 2-4 sentence rationale
    "input_max":  1200,  # upper bound; flag and review if exceeded
    "output_max": 500,   # matches max_tokens hard limit
}

# ---------------------------------------------------------------------------
# Cost estimates (spec Section 2.2)
# ---------------------------------------------------------------------------
# Per-call cost rates (USD per 1,000 tokens) used to project total spend.
# Rates are approximate and should be verified at execution time.
# Total budget: ~$500; projected spend ~$168-173 (closed ~$158, open ~$9-15).

COST_ESTIMATES: dict[str, dict[str, float]] = {
    "gpt_5": {
        "input_per_1k":  0.015,
        "output_per_1k": 0.060,
        "architecture":  "closed",
    },
    "claude_opus_4_6": {
        "input_per_1k":  0.015,
        "output_per_1k": 0.075,
        "architecture":  "closed",
    },
    "gemini_3_pro": {
        "input_per_1k":  0.007,
        "output_per_1k": 0.021,
        "architecture":  "closed",
    },
    "deepseek_v3": {
        "input_per_1k":  0.0003,
        "output_per_1k": 0.0009,
        "architecture":  "open",
    },
    "kimi_k2": {
        "input_per_1k":  0.0014,
        "output_per_1k": 0.0028,
        "architecture":  "open",
    },
    "glm_4_7": {
        "input_per_1k":  0.0014,
        "output_per_1k": 0.0014,
        "architecture":  "open",
    },
}


def estimate_total_cost(
    n_fragments: int = 150,
    n_models: int = 6,
    n_prompts: int = 2,
    n_runs: int = 3,
) -> dict[str, float]:
    """
    Estimate total API cost across all benchmark calls.

    Args:
        n_fragments: Number of benchmark fragments (default 150).
        n_models: Number of model families (default 6).
        n_prompts: Number of prompt conditions (default 2).
        n_runs: Runs per fragment-model-prompt combination (default 3).

    Returns:
        Dict with per-model and total cost estimates (USD).
    """
    n_calls_per_model = n_fragments * n_prompts * n_runs
    estimates: dict[str, float] = {}

    for model, rates in COST_ESTIMATES.items():
        input_cost  = (EXPECTED_TOKENS["input_avg"]  / 1000) * rates["input_per_1k"]
        output_cost = (EXPECTED_TOKENS["output_avg"] / 1000) * rates["output_per_1k"]
        estimates[model] = round((input_cost + output_cost) * n_calls_per_model, 2)

    estimates["total_usd"] = round(sum(
        v for k, v in estimates.items() if k != "total_usd"
    ), 2)
    return estimates


# ---------------------------------------------------------------------------
# Execution constants (spec Section 2.2–2.3)
# ---------------------------------------------------------------------------

# Prompt conditions in canonical order
PROMPT_CONDITIONS: list[str] = ["zero_shot", "few_shot"]

# Runs per fragment × model × prompt combination (for consistency analysis)
RUNS_PER_COMBO: int = 3

# Mandatory delay between runs within the same combination (spec §2.3)
INTER_RUN_DELAY_SECONDS: int = 5

# HTTP request timeout; triggers automatic retry on expiry
REQUEST_TIMEOUT_SECONDS: int = 60

# Retry schedule: 10s, 30s, 90s (spec §2.3 — up to 3 retries)
RETRY_BACKOFF_SECONDS: list[int] = [10, 30, 90]

# Maximum retries per call (length of RETRY_BACKOFF_SECONDS)
MAX_RETRIES: int = len(RETRY_BACKOFF_SECONDS)

# Benchmark scale
TOTAL_FRAGMENTS: int = 150
# 150 × 6 × 2 × 3 = 5,400
TOTAL_EXPECTED_CALLS: int = TOTAL_FRAGMENTS * 6 * len(PROMPT_CONDITIONS) * RUNS_PER_COMBO

# Session management: process in batches of this many calls (spec §2.3)
SESSION_BATCH_SIZE: int = 1000  # ~14-28 fragments per session

# Random seed for fragment execution order randomization (reproducibility)
EXECUTION_ORDER_SEED: int = 42

# Number of calibration examples in few-shot prompt (exactly 4; spec §1.3)
N_CALIBRATION_EXAMPLES: int = 4
# Balance: 2 sound, 2 not sound (spec §1.3)
N_CALIBRATION_SOUND: int = 2
N_CALIBRATION_NOT_SOUND: int = 2
