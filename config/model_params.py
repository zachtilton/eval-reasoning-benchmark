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
    "temperature": 0,        # Maximum determinism (spec §2.2) — overridden per-model where required (see api_config.py)
    # RAISED from 500 (spec §2.2's original ~250-token-output assumption) —
    # CONFIRMED via live pilot call (2026-08-02): for reasoning-enabled
    # models, thinking/reasoning tokens count against this same ceiling.
    # DeepSeek V4 Pro at max_tokens=500 consumed the entire budget on
    # reasoning (500 output_tokens == 500 reasoning_tokens) and returned
    # NO visible answer at all; at max_tokens=4000 on the same real
    # fragment it used 866 reasoning + ~73 visible tokens and completed
    # normally (finish_reason: "stop"). 4000 is a safety margin, not a
    # precisely measured minimum — the pilot's corpus token audit only
    # tested one (short, 267-token) fragment; longer fragments (corpus max
    # ~2008 tokens) or harder reasoning may need more. This directly
    # affects real-run cost projections (Task 3) — a higher ceiling doesn't
    # automatically mean higher spend (models stop early via finish_reason
    # "stop" when done), but the worst case is now ~8x the original
    # assumption. Confirm this value before the real 5,400-call run.
    "max_tokens": 4000,
    "top_p": 1.0,            # No nucleus sampling restriction — overridden per-model where required (see api_config.py)
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
        # DeepSeek, Kimi, GLM (via Fireworks) all use standard OpenAI Chat
        # Completions field names. GPT-5.6 Sol does NOT belong here — it
        # uses the separate "openai_responses" family below.
        "temperature":       "temperature",
        "max_tokens":        "max_tokens",
        "top_p":             "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty":  "presence_penalty",
    },
    "openai_responses": {
        # OpenAI's Responses API (GPT-5.6 Sol) — max_tokens is named
        # max_output_tokens here, and frequency/presence penalty are not
        # part of this API's parameter surface (Chat-Completions-specific).
        "temperature":       "temperature",
        "max_tokens":        "max_output_tokens",
        "top_p":             "top_p",
        "frequency_penalty": None,           # not supported by Responses API
        "presence_penalty":  None,           # not supported by Responses API
    },
    "anthropic": {
        "temperature":       "temperature",
        "max_tokens":        "max_tokens",   # hoisted to top-level in payload
        "top_p":             "top_p",
        "frequency_penalty": None,           # not supported
        "presence_penalty":  None,           # not supported
    },
    "google": {
        # Classic generateContent REST endpoint. No current model uses this
        # family (Gemini 3.1 Pro Preview High uses "google_interactions_sdk"
        # below instead) — left in place as generic infrastructure in case
        # a future model targets the classic REST endpoint again.
        "temperature":       "temperature",
        "max_tokens":        "maxOutputTokens",
        "top_p":             "topP",
        "frequency_penalty": None,           # not supported
        "presence_penalty":  None,           # not supported
    },
    "google_interactions_sdk": {
        # Google's Interactions API (Gemini 3.1 Pro Preview High). Verified
        # via direct introspection of google-genai 2.16.0's GenerationConfig
        # pydantic model — it has NO temperature or top_p fields at all
        # (not "unsupported for this specific model", genuinely absent from
        # the type), unlike every other family here where sampling params
        # are at least present-but-droppable.
        "temperature":       None,           # not part of this API's GenerationConfig
        "max_tokens":        "max_output_tokens",
        "top_p":             None,           # not part of this API's GenerationConfig
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
    "output_avg": 250,   # classification line + 1-2 sentence rationale; pilot Task 1/3 supersedes this once measured
    "input_max":  1200,  # upper bound; flag and review if exceeded
    "output_max": 500,   # matches max_tokens hard limit
}

# ---------------------------------------------------------------------------
# Cost estimates (spec Section 2.2)
# ---------------------------------------------------------------------------
# Per-call cost rates (USD per 1,000 tokens) used to project total spend.
# Rates are approximate and should be verified at execution time.
# Total budget: ~$500; projected spend ~$168-173 (closed ~$158, open ~$9-15).

COST_ESTIMATES: dict[str, dict[str, float | None]] = {
    "gpt_5_6_sol": {
        "input_per_1k":  None,  # verify at execution — placeholder; pilot Task 3 supersedes once measured
        "output_per_1k": None,  # verify at execution — placeholder
        "architecture":  "closed",
    },
    "claude_opus_5": {
        "input_per_1k":  0.005,  # $5/MTok, grounded via claude-api skill
        "output_per_1k": 0.025,  # $25/MTok, grounded via claude-api skill
        "architecture":  "closed",
    },
    "gemini_3_pro_preview_high": {
        "input_per_1k":  None,  # verify at execution — placeholder
        "output_per_1k": None,  # verify at execution — placeholder
        "architecture":  "closed",
    },
    "kimi_k3": {
        "input_per_1k":  None,  # verify at execution — placeholder
        "output_per_1k": None,  # verify at execution — placeholder
        "architecture":  "open",
    },
    "deepseek_v4_pro": {
        "input_per_1k":  None,  # verify at execution — placeholder
        "output_per_1k": None,  # verify at execution — placeholder
        "architecture":  "open",
    },
    "glm_5_2": {
        "input_per_1k":  None,  # verify at execution — placeholder
        "output_per_1k": None,  # verify at execution — placeholder
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
        if rates["input_per_1k"] is None or rates["output_per_1k"] is None:
            continue  # rate not yet verified for this model; excluded from the estimate
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

# HTTP request timeout; triggers automatic retry on expiry.
# RAISED from 60 — CONFIRMED via live pilot call (2026-08-03): GPT-5.6 Sol's
# few_shot call (larger prompt, max reasoning effort) timed out at 60s on
# all 3 retry attempts. First raised to 180s and confirmed working (the same
# call completed in 168.014s once retried), but that left only ~12s of
# margin — the real corpus has fragments up to ~2008 tokens vs. report_id
# 4425's smaller one, so raised further to 300s for real headroom on
# max-effort reasoning models before the full 5-fragment/5,400-call run.
REQUEST_TIMEOUT_SECONDS: int = 300

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

# Number of calibration examples in few-shot prompt (exactly 2; spec §1.3)
N_CALIBRATION_EXAMPLES: int = 2
# Balance: 1 sound, 1 not sound (spec §1.3)
N_CALIBRATION_SOUND: int = 1
N_CALIBRATION_NOT_SOUND: int = 1
