"""
API configuration, model parameters, and project path constants.

Imports all constants from the authoritative config package at the project
root (config/api_config.py, config/model_params.py) and re-exports them
under the same names so the rest of src/api_client/ is unaffected.

Path constants (PROMPTS_DIR, RESPONSE_DB_PATH, etc.) are defined here
rather than in config/ because they are code-layer concerns, not
user-configurable settings.

Per spec Section 2.2: "API versions documented at time of execution."
Update model_id values in config/api_config.py before running the benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths — defined here, not in config/ (code-layer concern)
# ---------------------------------------------------------------------------

# Resolve from this file: src/api_client/config.py → src/api_client → src → root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR      = PROJECT_ROOT / "data"
RESPONSES_DIR = DATA_DIR / "responses"
LOGS_DIR      = PROJECT_ROOT / "logs"
CONFIG_DIR    = PROJECT_ROOT / "config"
PROMPTS_DIR   = CONFIG_DIR / "prompts"

RESPONSE_DB_PATH      = RESPONSES_DIR / "response_database.csv"
FAILED_CALLS_LOG      = LOGS_DIR / "failed_calls.jsonl"
EXECUTION_ORDER_LOG   = LOGS_DIR / "execution_order_log.csv"

# ---------------------------------------------------------------------------
# Import from authoritative config package
# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `import config` resolves correctly
# regardless of the working directory at invocation time.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.api_config import (  # noqa: E402
    ANTHROPIC_API_VERSION,
    API_CONFIG,
    ARCHITECTURE_MAP,
    MODEL_API_FAMILY,
    MODEL_ORDER,
    PRIMARY_MODEL,
)
from config.model_params import (  # noqa: E402
    EXECUTION_ORDER_SEED,
    EXPECTED_TOKENS,
    INTER_RUN_DELAY_SECONDS,
    MAX_RETRIES,
    N_CALIBRATION_EXAMPLES,
    PARAM_MAPPING,
    PROMPT_CONDITIONS,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    RUNS_PER_COMBO,
    SESSION_BATCH_SIZE,
    STANDARD_PARAMS,
    TOTAL_EXPECTED_CALLS,
    TOTAL_FRAGMENTS,
)

# ---------------------------------------------------------------------------
# Derived constants (kept here for backward compatibility)
# ---------------------------------------------------------------------------

MODELS: list[str] = list(API_CONFIG.keys())

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESPONSES_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "PROMPTS_DIR",
    "RESPONSE_DB_PATH",
    "FAILED_CALLS_LOG",
    "EXECUTION_ORDER_LOG",
    # API config
    "API_CONFIG",
    "MODEL_API_FAMILY",
    "ARCHITECTURE_MAP",
    "MODEL_ORDER",
    "PRIMARY_MODEL",
    "ANTHROPIC_API_VERSION",
    # Model params
    "STANDARD_PARAMS",
    "PARAM_MAPPING",
    "EXPECTED_TOKENS",
    "PROMPT_CONDITIONS",
    "RUNS_PER_COMBO",
    "INTER_RUN_DELAY_SECONDS",
    "REQUEST_TIMEOUT_SECONDS",
    "RETRY_BACKOFF_SECONDS",
    "MAX_RETRIES",
    "SESSION_BATCH_SIZE",
    "EXECUTION_ORDER_SEED",
    "TOTAL_FRAGMENTS",
    "TOTAL_EXPECTED_CALLS",
    "N_CALIBRATION_EXAMPLES",
    # Derived
    "MODELS",
]
