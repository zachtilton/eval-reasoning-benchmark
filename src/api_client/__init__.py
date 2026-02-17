"""
src/api_client — API execution layer for the evaluative reasoning benchmark.

Module layout
-------------
config.py    — API_CONFIG, STANDARD_PARAMS, path constants, model mappings
parser.py    — response parsing, classification normalization, rationale validation
retry.py     — error categorization, exponential backoff, retry logic, rerun queue
executor.py  — request construction, API call execution, database persistence
batch.py     — fragment ordering, batch orchestration, session management

Public interface
----------------
Process a batch of fragments:
    process_fragments_batch(fragments, ...)

Randomize execution order before a session:
    generate_execution_order(fragment_ids, seed=42)

Plan and monitor sessions:
    calculate_session_plan()
    track_execution_progress()

Manage the failed-call rerun queue:
    load_failed_calls()
    rerun_failed_calls(failed_calls, fragment_data)
    clear_failed_calls_log()
"""

from .batch import (
    calculate_session_plan,
    generate_execution_order,
    process_fragments_batch,
    track_execution_progress,
)
from .retry import (
    clear_failed_calls_log,
    load_failed_calls,
    rerun_failed_calls,
)

__all__ = [
    # Batch orchestration
    "process_fragments_batch",
    "generate_execution_order",
    "calculate_session_plan",
    "track_execution_progress",
    # Rerun queue
    "load_failed_calls",
    "rerun_failed_calls",
    "clear_failed_calls_log",
]
