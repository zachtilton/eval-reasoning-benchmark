"""
Pilot cost/token audit configuration (pre-registered pilot spec, Aug 2026).

Pilot-specific constants only — the authoritative model/parameter config
still lives in api_config.py / model_params.py; this module imports from
those rather than duplicating anything.

Unlike src/api_client/config.py, path constants live here directly rather
than in a separate src/pilot/config.py wrapper — the pilot is a small,
one-off tool, not a persistent multi-module layer.

BEFORE RUNNING TASK 2:
    Fill in PILOT_PINNED_REPORT_IDS after running Task 1 and inspecting
    logs/corpus_token_audit.csv. Choose the shortest, ~median, longest,
    plus 1-2 mid-range fragments by report_id — pin them here as a fixed
    constant, not by re-selecting at runtime. Task 2 raises ValueError at
    startup if this list is empty.

    report_id is an int in the real corpus (data/raw/fragments.jsonl),
    e.g. 6487 — not a string. Pin ints here, not "6487"-style strings, or
    the lookup in task2_pilot_batch.run_task2 won't match.

    2 of the 150 report_ids (1359, 1383) each cover 2 fragments under
    different criteria — Task 1 prints these at the top of its output as a
    warning. Do not pin either of them; Task 2 raises ValueError rather
    than silently picking one of the two fragments.

ENVIRONMENT VARIABLES REQUIRED:
    See config/api_config.py's module docstring for the 6 required env vars.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = PROJECT_ROOT / "data"
LOGS_DIR      = PROJECT_ROOT / "logs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.api_config import API_CONFIG, MODEL_API_FAMILY  # noqa: E402
from config.model_params import COST_ESTIMATES, PROMPT_CONDITIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed pilot fragment selection
# ---------------------------------------------------------------------------
# Manually picked after inspecting logs/corpus_token_audit.csv (Task 1),
# spanning the fragment-length distribution: 267-1457 tokens. These 5
# fragments will also be re-rated in the full 5,400-call run — the pilot
# reuse is intentional, not a leak from a separate held-out set.
#
#   report_id  criterion                 tokens
#   4425       relevance                    267
#   6782       relevance                    382
#   3381       efficiency                   649
#   1122       effectiveness                880
#   177        sustainability               1457

PILOT_PINNED_REPORT_IDS: list[int] = [4425, 6782, 3381, 1122, 177]

# ---------------------------------------------------------------------------
# Pilot execution constants
# ---------------------------------------------------------------------------

# Explicit ordered list (not API_CONFIG.keys() at call time) so pilot output
# CSVs have deterministic row/column order independent of dict iteration.
PILOT_MODELS: list[str] = list(API_CONFIG.keys())

# Own name (not model_params.PROMPT_CONDITIONS re-imported elsewhere) so
# pilot code never accidentally imports the 3-run production RUNS_PER_COMBO.
PILOT_PROMPT_CONDITIONS: list[str] = list(PROMPT_CONDITIONS)

# The pilot runs exactly 1 run per fragment-model-prompt combination (not 3).
PILOT_RUN_NUMBER: int = 1

# ---------------------------------------------------------------------------
# Output paths — isolated from the real 5,400-call dataset
# ---------------------------------------------------------------------------

FRAGMENTS_PATH = DATA_DIR / "raw" / "fragments.jsonl"

CORPUS_TOKEN_AUDIT_PATH      = LOGS_DIR / "corpus_token_audit.csv"
CALIBRATION_TOKEN_AUDIT_PATH = LOGS_DIR / "calibration_token_audit.csv"

PILOT_LOGS_DIR          = LOGS_DIR / "pilot"
PILOT_USAGE_LOG_PATH    = PILOT_LOGS_DIR / "pilot_usage_log.csv"
PILOT_COST_PROJECTION_PATH = PILOT_LOGS_DIR / "cost_projection_updated.csv"

PILOT_RESPONSES_DIR = DATA_DIR / "responses" / "pilot"

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

CL100K_ENCODING_NAME: str = "cl100k_base"

# ---------------------------------------------------------------------------
# Cost projection
# ---------------------------------------------------------------------------

# Task 3 flags a model when reasoning_tokens / output_tokens exceeds this.
REASONING_FLAG_THRESHOLD: float = 3.0


def compute_call_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int | None,
) -> float | None:
    """
    Compute the USD cost of a single pilot call.

    Reasoning tokens are billed at the provider's output-token rate — no
    provider is documented as billing them separately. Returns None if the
    model's COST_ESTIMATES entry has no verified rate yet (see
    model_params.COST_ESTIMATES).

    Whether reasoning_tokens should be ADDED to output_tokens for billing
    depends on the provider's usage schema, and is NOT uniform across
    models — CONFIRMED via the real pilot data (2026-08-03): for every
    anthropic/openai_responses/openai_compatible call logged, reasoning_tokens
    was always <= output_tokens (ratios 0.33x-0.96x), the signature of
    output_tokens already being the total (reasoning + visible answer) with
    reasoning_tokens merely a breakdown within it — adding it again would
    double-bill. Only google_interactions_sdk (Gemini 3.1 Pro Preview High)
    reports two genuinely separate pools (total_output_tokens vs.
    total_thought_tokens, confirmed via SDK type introspection) — there,
    reasoning routinely EXCEEDS output (ratios up to 11.9x in pilot data),
    which is only possible if they're independent and must both be paid for.

    Args:
        model: Model identifier from API_CONFIG.
        input_tokens: Input token count for the call.
        output_tokens: Visible output token count for the call.
        reasoning_tokens: Hidden reasoning token count, or None if the
            provider doesn't report it separately.

    Returns:
        Cost in USD, or None if rates aren't yet verified for this model.
    """
    rates = COST_ESTIMATES.get(model)
    if rates is None:
        return None

    input_rate = rates.get("input_per_1k")
    output_rate = rates.get("output_per_1k")
    if input_rate is None or output_rate is None:
        return None

    reasoning_is_separate_pool = MODEL_API_FAMILY.get(model) == "google_interactions_sdk"
    billable_output_tokens = (
        output_tokens + (reasoning_tokens or 0) if reasoning_is_separate_pool else output_tokens
    )
    cost = (input_tokens / 1000) * input_rate + (billable_output_tokens / 1000) * output_rate
    return round(cost, 6)
