"""
Resumable, session-batched driver for the full 5,400-call benchmark run.

Builds on the existing orchestration primitives rather than replacing them:
- batch.generate_execution_order / batch.calculate_session_plan for fragment
  ordering and session-sizing math.
- retry.make_api_call_with_retry for the actual HTTP call + backoff.
- executor.process_and_store_response to persist each success immediately.
- retry.log_failed_call to queue hard failures for later rerun via
  retry.rerun_failed_calls -- this driver does not retry the failed-calls
  queue itself; that remains a separate, explicit step.

Why this exists instead of just calling batch.process_fragments_batch(fragments)
directly: that function has no notion of "already done." Handing it a
fragment whose session was interrupted partway through its 36 calls would
redo the already-completed combos and double-bill them. This module adds
combo-level resume on top of the same primitives, so every session is safe
to interrupt and re-invoke without risk of duplicate spend.

Known data gap -- duplicate report_ids: the real 150-row corpus has 2
report_ids that each cover 2 fragments under different criteria (see
src.pilot.fragments.find_duplicate_report_ids). The response database keys
rows by (fragment_id, model, prompt_condition, run_number) using report_id
as fragment_id -- for those 2 ids, that key cannot distinguish the two
underlying fragments. Rather than silently merging or dropping one, this
driver sets both aside (excluded from automatic selection) and reports them
loudly every session, so they get a deliberate decision instead of silent
data loss.

Usage -- always smoke-test with --dry-run against the small test fixture
before ever touching the real corpus or spending anything:

    python -m src.api_client.run_benchmark --plan
    python -m src.api_client.run_benchmark --status
    python -m src.api_client.run_benchmark --fragments 10 --dry-run \\
        --fragments-path tests/fixtures/sample_fragments.jsonl
    python -m src.api_client.run_benchmark --fragments 10
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Loads .env into os.environ -- executor.py reads API keys via os.getenv()
# and has no .env-loading of its own, so this must happen before any real
# (non-dry-run) API call in this process.
load_dotenv()

from .batch import calculate_session_plan, generate_execution_order, track_execution_progress
from .config import (
    EXECUTION_ORDER_LOG,
    FAILED_CALLS_LOG,
    INTER_RUN_DELAY_SECONDS,
    LOGS_DIR,
    MODEL_ORDER,
    PROMPT_CONDITIONS,
    RESPONSE_DB_PATH,
    RESPONSES_DIR,
    RUNS_PER_COMBO,
)
from .executor import process_and_store_response
from .retry import log_failed_call, make_api_call_with_retry

BENCHMARK_SESSION_LOG: Path = LOGS_DIR / "benchmark_session_log.jsonl"

# Dry-run runs are fully isolated from every real-run file (mirrors the
# pilot's isolation guarantee in task2_pilot_batch.py) so a smoke test can
# never touch real data, logs, or the real execution order.
DRYRUN_RESPONSE_DB_PATH: Path = RESPONSES_DIR / "dryrun_response_database.csv"
DRYRUN_EXECUTION_ORDER_LOG: Path = LOGS_DIR / "dryrun_execution_order_log.csv"
DRYRUN_FAILED_CALLS_LOG: Path = LOGS_DIR / "dryrun_failed_calls.jsonl"
DRYRUN_SESSION_LOG: Path = LOGS_DIR / "dryrun_benchmark_session_log.jsonl"

# 10 fragments x 36 calls/fragment = 360 calls/session -- conservative
# default chunk size, smaller than the spec's 500-1,000 call recommendation,
# so early sessions are cheap to sanity-check before committing to bigger ones.
DEFAULT_FRAGMENTS_PER_SESSION = 10

CALLS_PER_FRAGMENT = len(MODEL_ORDER) * len(PROMPT_CONDITIONS) * RUNS_PER_COMBO


def load_fragment_groups(fragments_path: Path) -> tuple[dict[int, dict], dict[int, list[str]]]:
    """
    Load the fragment corpus and separate unambiguous from duplicate report_ids.

    Args:
        fragments_path: Path to the fragments JSONL corpus.

    Returns:
        Tuple of:
        - ``fragments_by_id``: report_id -> fragment record, for report_ids
          that map to exactly one fragment.
        - ``ambiguous``: report_id -> list of criterion values, for
          report_ids that map to more than one fragment (set aside, not
          included in ``fragments_by_id``).
    """
    from src.pilot.fragments import load_fragments  # noqa: PLC0415

    records = load_fragments(fragments_path)
    grouped: dict[int, list[dict]] = {}
    for r in records:
        grouped.setdefault(r["report_id"], []).append(r)

    fragments_by_id = {rid: recs[0] for rid, recs in grouped.items() if len(recs) == 1}
    ambiguous = {
        rid: [r["criterion"] for r in recs]
        for rid, recs in grouped.items()
        if len(recs) > 1
    }
    return fragments_by_id, ambiguous


def get_execution_order(
    report_ids: list[int],
    seed: int = 42,
    log_path: Path = EXECUTION_ORDER_LOG,
) -> list[int]:
    """
    Return the fragment execution order, reusing a prior session's logged
    order instead of reshuffling.

    Once an order has been generated and logged for a benchmark run, every
    later session must reuse it verbatim -- reshuffling between sessions
    would make combo-level resume harder to reason about and breaks the
    "logged for audit" guarantee in batch.generate_execution_order's
    docstring.

    Args:
        report_ids: Unambiguous report_ids to order (see
            :func:`load_fragment_groups`).
        seed: Random seed, only used if no execution order is logged yet.
        log_path: Path to the execution order CSV log.

    Returns:
        List of report_ids (ints) in execution order.
    """
    if log_path.exists():
        order_df = pd.read_csv(log_path)
        print(f"Reusing existing execution order from {log_path} ({len(order_df)} fragments).")
        return order_df.sort_values("execution_position")["fragment_id"].tolist()

    return generate_execution_order(report_ids, seed=seed, log_path=log_path)


def get_completed_combos(db_path: Path = RESPONSE_DB_PATH) -> set[tuple]:
    """
    Return the set of (fragment_id, model, prompt_condition, run_number)
    combos already present in the response database.

    Used to skip combos a prior, interrupted session already completed, so
    re-invoking a session never redoes (and double-bills) finished work.

    Args:
        db_path: Path to the response database CSV.

    Returns:
        Set of 4-tuples. Empty set if the database doesn't exist yet.
    """
    if not db_path.exists():
        return set()
    df = pd.read_csv(db_path)
    if df.empty:
        return set()
    return set(
        zip(
            df["fragment_id"],
            df["model_family"],
            df["prompt_condition"],
            df["run_number"],
        )
    )


def get_exhausted_combos(failed_log_path: Path = FAILED_CALLS_LOG) -> set[tuple]:
    """
    Return the set of (fragment_id, model, prompt_condition, run_number)
    combos that have already exhausted all retry attempts and been logged
    to the failed-calls queue.

    A combo here is NOT "done" -- it genuinely failed. But without this,
    every new session recomputes "what's left" purely from the response
    database, which has no row for an exhausted failure (it only ever
    records successes) -- so a persistently-failing combo (e.g. a provider
    outage) gets re-attempted, and re-burns its full retry budget (up to
    ~16 minutes at the 300s/attempt, 3-attempt, 10/30/90s-backoff schedule),
    at the start of every single future session until someone deliberately
    replays the failed-calls queue via retry.rerun_failed_calls. Treating
    exhausted combos as "already accounted for" here keeps a flaky provider
    from stalling progress on every resume, while retry.rerun_failed_calls
    remains the one explicit, deliberate path for actually retrying them.

    Args:
        failed_log_path: Path to the failed-calls JSONL queue.

    Returns:
        Set of 4-tuples. Empty set if the log doesn't exist yet.
    """
    if not failed_log_path.exists():
        return set()
    combos: set[tuple] = set()
    with failed_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            combos.add(
                (record["fragment_id"], record["model"], record["prompt_type"], record["run_number"])
            )
    return combos


def select_next_fragments(
    execution_order: list[int],
    fragments_by_id: dict[int, dict],
    skip_combos: set[tuple],
    n_fragments: int,
    models: list[str] = MODEL_ORDER,
    prompt_conditions: list[str] = PROMPT_CONDITIONS,
    runs_per_combo: int = RUNS_PER_COMBO,
) -> list[dict]:
    """
    Select the next ``n_fragments`` not-yet-fully-handled fragments, in
    execution order, as executor-shaped records ({"id", "text"}).

    A fragment counts as handled only when every one of its
    models x prompt_conditions x runs_per_combo combos is already in
    ``skip_combos`` -- a fragment with even one missing combo is selected
    again so run_benchmark_session's own combo-level skip can fill in just
    what's missing, rather than redoing the whole fragment.

    Args:
        execution_order: Unambiguous report_id order for this benchmark run.
        fragments_by_id: report_id -> fragment record, from
            :func:`load_fragment_groups`.
        skip_combos: Combos to treat as already handled -- the union of
            succeeded combos (:func:`get_completed_combos`) and combos that
            already exhausted their retries (:func:`get_exhausted_combos`).
            A fragment is not re-selected just because one of its combos
            genuinely failed; see :func:`get_exhausted_combos` for why.
        n_fragments: Max number of not-yet-handled fragments to return.
        models: Canonical model order.
        prompt_conditions: Canonical prompt-condition order.
        runs_per_combo: Runs per fragment-model-prompt combo.

    Returns:
        List of executor-shaped fragment dicts ({"id", "text"}), in
        execution order.
    """
    from src.pilot.fragments import to_executor_fragment  # noqa: PLC0415

    calls_per_fragment = len(models) * len(prompt_conditions) * runs_per_combo
    selected: list[dict] = []
    for report_id in execution_order:
        if report_id not in fragments_by_id:
            continue  # ambiguous report_id, set aside -- see load_fragment_groups
        done = sum(
            1
            for model in models
            for prompt in prompt_conditions
            for run in range(1, runs_per_combo + 1)
            if (report_id, model, prompt, run) in skip_combos
        )
        if done >= calls_per_fragment:
            continue
        selected.append(to_executor_fragment(fragments_by_id[report_id]))
        if len(selected) >= n_fragments:
            break
    return selected


def _dry_run_result(fragment: dict, model: str, prompt_type: str, run_num: int) -> dict:
    """Canned success result for --dry-run, shaped like make_api_call_with_retry's return value."""
    return {
        "fragment_id": fragment["id"],
        "model": model,
        "prompt_type": prompt_type,
        "run_number": run_num,
        "status": "success",
        "parsed_response": {
            "classification": "sound",
            "rationale": "Dry-run rationale for smoke-testing run_benchmark.py.",
            "token_count_input": 0,
            "token_count_output": 0,
            "reasoning_tokens": None,
            "parse_method": "dry-run",
            "latency_seconds": 0.0,
            "api_version": "dry-run",
        },
        "attempts": 1,
        "timestamp": datetime.now().isoformat(),
        "error": None,
    }


def run_benchmark_session(
    fragments: list[dict],
    skip_combos: set[tuple],
    models: list[str] = MODEL_ORDER,
    prompt_conditions: list[str] = PROMPT_CONDITIONS,
    runs_per_combo: int = RUNS_PER_COMBO,
    inter_run_delay: int = INTER_RUN_DELAY_SECONDS,
    db_path: Path = RESPONSE_DB_PATH,
    failed_log_path: Path = FAILED_CALLS_LOG,
    dry_run: bool = False,
) -> dict:
    """
    Run one resumable session: every combo for ``fragments`` not already in
    ``skip_combos``.

    Mirrors batch.process_fragments_batch's loop (same fragment -> model ->
    prompt -> run nesting, same 5-second inter-run delay, same immediate
    per-call persistence) but skips any combo already accounted for, so
    this is always safe to re-invoke after an interruption.

    Args:
        fragments: Executor-shaped fragment dicts ({"id", "text"}) to
            process in this session, from :func:`select_next_fragments`.
        skip_combos: Combos to treat as already handled and not re-attempt
            -- the union of succeeded combos and combos that already
            exhausted their retries (see :func:`get_exhausted_combos` for
            why exhausted failures are skipped here rather than retried
            automatically).
        models: Canonical model order.
        prompt_conditions: Canonical prompt-condition order.
        runs_per_combo: Runs per fragment-model-prompt combo.
        inter_run_delay: Seconds between runs within a combo.
        db_path: Path to the response database CSV.
        failed_log_path: Path to the failed-calls JSONL queue.
        dry_run: If True, use a canned response instead of a real HTTP call.

    Returns:
        Dict with ``attempted``, ``succeeded``, ``failed``, ``skipped``,
        ``session_duration_seconds``.
    """
    session_start = datetime.now()
    attempted = succeeded = failed = skipped = 0

    for frag_idx, fragment in enumerate(fragments, start=1):
        print(f"\n[{frag_idx}/{len(fragments)}] Fragment {fragment['id']}")

        for model in models:
            for prompt_type in prompt_conditions:
                for run_num in range(1, runs_per_combo + 1):
                    combo = (fragment["id"], model, prompt_type, run_num)
                    if combo in skip_combos:
                        skipped += 1
                        continue

                    attempted += 1

                    if dry_run:
                        result = _dry_run_result(fragment, model, prompt_type, run_num)
                    else:
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
                        succeeded += 1
                    else:
                        log_failed_call(result, log_path=failed_log_path)
                        failed += 1

                    if run_num < runs_per_combo and not dry_run:
                        time.sleep(inter_run_delay)

    duration = (datetime.now() - session_start).total_seconds()
    summary = {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "session_duration_seconds": round(duration, 1),
    }

    sep = "=" * 60
    print(f"\n{sep}")
    print("SESSION COMPLETE")
    print(f"  Attempted:              {attempted:,}")
    print(f"  Succeeded:              {succeeded:,}")
    print(f"  Failed:                 {failed:,}")
    print(f"  Skipped (already done): {skipped:,}")
    print(f"  Duration:               {duration / 60:.1f} min")
    print(f"{sep}\n")

    return summary


def _append_session_log(summary: dict, log_path: Path) -> None:
    """Append one session's summary (plus a timestamp) to the session log JSONL."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {**summary, "timestamp": datetime.now().isoformat()}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def print_status(
    db_path: Path = RESPONSE_DB_PATH,
    failed_log_path: Path = FAILED_CALLS_LOG,
) -> dict:
    """
    Print and return the current benchmark status without running anything.

    Combines batch.track_execution_progress (response database) with a
    count of calls currently sitting in the failed-calls retry queue.

    Args:
        db_path: Path to the response database CSV.
        failed_log_path: Path to the failed-calls JSONL queue.

    Returns:
        Progress dict from track_execution_progress, plus
        ``failed_calls_queued``.
    """
    progress = track_execution_progress(db_path)
    n_failed_queued = 0
    if failed_log_path.exists():
        with failed_log_path.open("r", encoding="utf-8") as fh:
            n_failed_queued = sum(1 for line in fh if line.strip())
    print(
        f"  Failed calls queued for rerun: {n_failed_queued:,} "
        f"(see retry.rerun_failed_calls to replay them)"
    )
    progress["failed_calls_queued"] = n_failed_queued
    return progress


def run_next_session(
    n_fragments: int = DEFAULT_FRAGMENTS_PER_SESSION,
    fragments_path: Path | None = None,
    dry_run: bool = False,
    models: list[str] = MODEL_ORDER,
) -> dict | None:
    """
    Select and run the next not-yet-complete session of up to
    ``n_fragments`` fragments, then persist a session summary.

    This is the single entry point meant to be invoked once per session to
    work through the corpus in bounded, resumable chunks -- re-invoking it
    after an interruption, or in a fresh session entirely, picks up exactly
    where the response database left off.

    Args:
        n_fragments: Number of not-yet-complete fragments to process this
            session (each fragment is CALLS_PER_FRAGMENT calls).
        fragments_path: Override for the fragment corpus path. Defaults to
            the real 150-row corpus; pass a small fixture for --dry-run
            smoke tests.
        dry_run: If True, use canned responses and fully isolated
            dry-run-only files (never the real database, execution order,
            failed-calls log, or session log).
        models: Which models to run this session. Defaults to all of
            MODEL_ORDER. Pass a subset to deliberately leave a model
            unattempted for now (e.g. a provider that's currently down) --
            this is the honest way to skip one: the skipped combos stay
            genuinely untouched, neither fabricated as "done" nor logged as
            a failure that was never actually attempted.

    Returns:
        Session summary dict, or None if no not-yet-complete fragments
        remain (either the benchmark is done, or only ambiguous report_ids
        are left -- see the printed warning in that case).
    """
    from config.pilot_config import FRAGMENTS_PATH  # noqa: PLC0415

    path = fragments_path or FRAGMENTS_PATH

    db_path = DRYRUN_RESPONSE_DB_PATH if dry_run else RESPONSE_DB_PATH
    order_log = DRYRUN_EXECUTION_ORDER_LOG if dry_run else EXECUTION_ORDER_LOG
    session_log = DRYRUN_SESSION_LOG if dry_run else BENCHMARK_SESSION_LOG
    failed_log = DRYRUN_FAILED_CALLS_LOG if dry_run else FAILED_CALLS_LOG

    fragments_by_id, ambiguous = load_fragment_groups(path)
    if ambiguous:
        print(
            f"WARNING: {len(ambiguous)} report_id(s) map to more than one "
            f"fragment and are set aside from automatic processing: "
            f"{ambiguous}. These need a deliberate decision (e.g. a "
            f"disambiguating fragment_id) before they can be run safely."
        )
    if set(models) != set(MODEL_ORDER):
        print(f"  Running a restricted model set this session: {models}")

    execution_order = get_execution_order(sorted(fragments_by_id.keys()), log_path=order_log)
    completed_combos = get_completed_combos(db_path)
    exhausted_combos = get_exhausted_combos(failed_log)
    if exhausted_combos:
        print(
            f"  {len(exhausted_combos)} combo(s) already exhausted retries and "
            f"are sitting in {failed_log} -- not auto-retried here; see "
            f"retry.rerun_failed_calls to replay them deliberately."
        )
    skip_combos = completed_combos | exhausted_combos

    session_fragments = select_next_fragments(
        execution_order, fragments_by_id, skip_combos, n_fragments, models=models
    )

    if not session_fragments:
        print(
            "No automatically-processable fragments remain "
            f"(ambiguous report_ids set aside: {list(ambiguous.keys())})."
            if ambiguous else
            "Benchmark already complete -- no fragments remaining."
        )
        return None

    summary = run_benchmark_session(
        session_fragments, skip_combos, models=models,
        db_path=db_path, failed_log_path=failed_log, dry_run=dry_run,
    )
    _append_session_log(summary, session_log)
    print_status(db_path=db_path, failed_log_path=failed_log)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable, session-batched driver for the full benchmark run."
    )
    parser.add_argument(
        "--fragments", type=int, default=DEFAULT_FRAGMENTS_PER_SESSION,
        help=f"Not-yet-complete fragments to process this session "
             f"(default {DEFAULT_FRAGMENTS_PER_SESSION}; each fragment = "
             f"{CALLS_PER_FRAGMENT} calls).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use canned responses and fully isolated dry-run files -- never "
             "the real database, execution order, or logs.",
    )
    parser.add_argument(
        "--fragments-path", type=Path, default=None,
        help="Override the fragment corpus path (mainly for --dry-run smoke "
             "tests against tests/fixtures/sample_fragments.jsonl).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current progress and exit without running a session.",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="Print the full-run session plan and exit.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict this session to a subset of models (space-separated "
             "model keys, e.g. --models claude_opus_5 gpt_5_6_sol). Default: "
             "all models in MODEL_ORDER. Combos for excluded models are left "
             "genuinely untouched -- not marked done, not logged as failed.",
    )
    args = parser.parse_args()

    if args.plan:
        calculate_session_plan()
        return
    if args.status:
        print_status()
        return

    run_next_session(
        n_fragments=args.fragments,
        fragments_path=args.fragments_path,
        dry_run=args.dry_run,
        models=args.models or MODEL_ORDER,
    )


if __name__ == "__main__":
    _main()
