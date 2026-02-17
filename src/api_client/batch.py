"""
Fragment execution ordering, batch processing workflow, and session management.

Covers Appendix G.1 orchestration sections.

Key corrections over the appendix code:
- process_fragments_batch now calls make_api_call_with_retry (not the
  undefined make_api_call) and persists each response immediately.
- track_execution_progress reads error_flag (the actual schema column)
  rather than the non-existent 'status' column.
- generate_execution_order uses random.Random(seed) for an isolated RNG
  instance rather than global random.seed() to avoid side effects.
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import (
    EXECUTION_ORDER_LOG,
    INTER_RUN_DELAY_SECONDS,
    MODELS,
    PROMPT_CONDITIONS,
    RESPONSE_DB_PATH,
    RUNS_PER_COMBO,
    TOTAL_EXPECTED_CALLS,
    TOTAL_FRAGMENTS,
)
from .executor import process_and_store_response
from .retry import log_failed_call, make_api_call_with_retry


def generate_execution_order(
    fragment_ids: list[str],
    seed: int = 42,
    log_path: Path = EXECUTION_ORDER_LOG,
) -> list[str]:
    """
    Produce a reproducible randomized execution order for the fragment corpus.

    The shuffled order is intentionally distinct from the gold-standard
    sequence (spec Section 2.3).  The mapping is logged to CSV for audit.

    Uses ``random.Random(seed)`` (an isolated instance) rather than the
    global ``random.seed()`` to avoid side effects on other code.

    Args:
        fragment_ids: List of fragment IDs (e.g. ``['F_001', ..., 'F_150']``).
        seed: Random seed for reproducibility across sessions.
        log_path: Path to write the execution order CSV log.

    Returns:
        List of fragment IDs in randomized execution order.
    """
    rng = random.Random(seed)
    order = fragment_ids.copy()
    rng.shuffle(order)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    order_df = pd.DataFrame({
        "execution_position": range(1, len(order) + 1),
        "fragment_id": order,
    })
    order_df.to_csv(log_path, index=False)
    print(f"Execution order ({len(order)} fragments, seed={seed}) logged to {log_path}")

    return order


def calculate_session_plan(
    total_fragments: int = TOTAL_FRAGMENTS,
    n_models: int = len(MODELS),
    n_prompts: int = len(PROMPT_CONDITIONS),
    runs: int = RUNS_PER_COMBO,
    calls_per_session: int = 1000,
) -> dict:
    """
    Compute the batch execution plan across multiple sessions.

    Total calls = fragments × models × prompts × runs = 150 × 6 × 2 × 3 = 5,400.
    Spec recommends sessions of 500–1,000 calls.

    Args:
        total_fragments: Total fragments in the benchmark corpus.
        n_models: Number of model families.
        n_prompts: Number of prompt conditions.
        runs: Independent runs per fragment-model-prompt combination.
        calls_per_session: Target API calls per session.

    Returns:
        Dict with keys: ``total_api_calls``, ``calls_per_fragment``,
        ``fragments_per_session``, ``estimated_sessions``,
        ``calls_per_session``.
    """
    total_calls = total_fragments * n_models * n_prompts * runs
    calls_per_fragment = n_models * n_prompts * runs
    fragments_per_session = calls_per_session // calls_per_fragment
    # Ceiling division without math.ceil
    estimated_sessions = -(-total_fragments // fragments_per_session)

    plan = {
        "total_api_calls": total_calls,
        "calls_per_fragment": calls_per_fragment,
        "fragments_per_session": fragments_per_session,
        "estimated_sessions": estimated_sessions,
        "calls_per_session": calls_per_session,
    }

    print("Execution Plan:")
    print(f"  Total API calls:       {plan['total_api_calls']:,}")
    print(f"  Calls per fragment:    {plan['calls_per_fragment']}")
    print(f"  Fragments per session: {plan['fragments_per_session']}")
    print(f"  Estimated sessions:    {plan['estimated_sessions']}")

    return plan


def track_execution_progress(db_path: Path = RESPONSE_DB_PATH) -> dict:
    """
    Report completion status by reading the current response database.

    Reads ``error_flag`` (the actual schema column) to distinguish clean
    responses from error-flagged ones.  The appendix code incorrectly
    referenced a non-existent ``status`` column.

    Args:
        db_path: Path to the response database CSV.

    Returns:
        Dict with keys: ``completed_calls``, ``failed_calls``,
        ``total_calls``, ``completion_pct``,
        ``fragments_complete``, ``fragments_remaining``.
        Returns an empty dict if the database does not yet exist.
    """
    if not db_path.exists():
        print("No response database found yet.")
        return {}

    df = pd.read_csv(db_path)

    # error_flag == False → clean, parseable response
    clean = int((df["error_flag"] == False).sum())  # noqa: E712
    flagged = int((df["error_flag"] == True).sum())  # noqa: E712
    total = len(df)

    # A fragment is "complete" when all 36 calls (6 × 2 × 3) have been written
    calls_per_fragment = len(MODELS) * len(PROMPT_CONDITIONS) * RUNS_PER_COMBO
    fragments_complete = int(
        df.groupby("fragment_id")
        .size()
        .apply(lambda n: n == calls_per_fragment)
        .sum()
    )

    progress = {
        "completed_calls": clean,
        "failed_calls": flagged,
        "total_calls": total,
        "completion_pct": round(total / TOTAL_EXPECTED_CALLS * 100, 1),
        "fragments_complete": fragments_complete,
        "fragments_remaining": TOTAL_FRAGMENTS - fragments_complete,
    }

    print(f"\nProgress Update:")
    print(f"  Rows written:          {total:,} / {TOTAL_EXPECTED_CALLS:,} "
          f"({progress['completion_pct']}%)")
    print(f"  Clean responses:       {clean:,}")
    print(f"  Error-flagged:         {flagged:,}")
    print(f"  Fragments complete:    {fragments_complete} / {TOTAL_FRAGMENTS}")

    return progress


def process_fragments_batch(
    fragments: list[dict],
    models: list[str] = MODELS,
    prompt_conditions: list[str] = PROMPT_CONDITIONS,
    runs_per_combo: int = RUNS_PER_COMBO,
    inter_run_delay: int = INTER_RUN_DELAY_SECONDS,
    db_path: Path = RESPONSE_DB_PATH,
) -> dict:
    """
    Process a batch of fragments through all model-prompt-run combinations.

    Execution order follows spec Section 2.3:
    - Fragments are processed sequentially.
    - All 6 models evaluate each fragment under both conditions before
      moving to the next fragment.
    - 3 independent runs per model-prompt combo with 5-second delays between
      runs (delay applied between runs, not after the last run).
    - Each successful response is persisted immediately via
      :func:`executor.process_and_store_response`.
    - Failed calls are logged to the JSONL queue via
      :func:`retry.log_failed_call` for later rerun in a separate session.

    Args:
        fragments: List of dicts, each with ``'id'`` (str) and ``'text'`` (str).
        models: Ordered list of model identifiers from ``API_CONFIG``.
        prompt_conditions: List of prompt condition strings.
        runs_per_combo: Independent runs per fragment-model-prompt combination.
        inter_run_delay: Seconds to wait between runs within a combo.
        db_path: Path to the response database CSV.

    Returns:
        Dict with session summary: ``total_attempted``, ``total_succeeded``,
        ``total_failed``, ``session_duration_seconds``.
    """
    session_start = datetime.now()
    total_attempted = 0
    total_succeeded = 0
    total_failed = 0

    total_calls_this_batch = (
        len(fragments) * len(models) * len(prompt_conditions) * runs_per_combo
    )
    print(
        f"\nStarting batch: {len(fragments)} fragments × {len(models)} models × "
        f"{len(prompt_conditions)} prompts × {runs_per_combo} runs "
        f"= {total_calls_this_batch:,} calls\n"
    )

    for frag_idx, fragment in enumerate(fragments, start=1):
        print(f"\n[{frag_idx}/{len(fragments)}] Fragment {fragment['id']}")

        for model in models:
            for prompt_type in prompt_conditions:
                for run_num in range(1, runs_per_combo + 1):
                    total_attempted += 1

                    result = make_api_call_with_retry(
                        fragment=fragment,
                        model=model,
                        prompt_type=prompt_type,
                        run_number=run_num,
                    )

                    if result["status"] == "success":
                        process_and_store_response(
                            fragment=fragment,
                            model=model,
                            prompt_type=prompt_type,
                            run_number=run_num,
                            parsed_response=result["parsed_response"],
                            db_path=db_path,
                        )
                        total_succeeded += 1
                    else:
                        log_failed_call(result)
                        total_failed += 1

                    # 5-second delay between runs, not after the last run in a combo
                    if run_num < runs_per_combo:
                        time.sleep(inter_run_delay)

    duration = (datetime.now() - session_start).total_seconds()
    summary = {
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "session_duration_seconds": round(duration, 1),
    }

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"BATCH COMPLETE")
    print(f"  Attempted: {total_attempted:,}")
    print(f"  Succeeded: {total_succeeded:,}")
    print(f"  Failed:    {total_failed:,}")
    print(f"  Duration:  {duration / 60:.1f} min")
    print(f"{sep}\n")

    return summary
