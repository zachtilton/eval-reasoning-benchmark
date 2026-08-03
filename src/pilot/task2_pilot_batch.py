"""
Task 2 — Pilot API batch.

Runs a fixed, pre-selected subset of the fragment corpus (pinned by
report_id in config.pilot_config.PILOT_PINNED_REPORT_IDS) through all 6
models under both prompt conditions, 1 run each, capturing full usage —
including hidden reasoning tokens on the effort-configured models — so
Task 3 can recompute the full 5,400-call cost projection from real numbers.

Reuses src.api_client.retry.make_api_call_with_retry for the actual HTTP
call and backoff (it's DB-agnostic — only touches the failed-calls JSONL
queue, not the real response_database.csv — so it's safe to share between
the pilot and the real 5,400-call run). Does NOT reuse
src.api_client.batch.process_fragments_batch, which is hardcoded to the
3-run production design and the real response database path.

Per the pilot spec's guardrails, outputs are isolated from the real
5,400-call dataset: logs/pilot/pilot_usage_log.csv and
data/responses/pilot/*.json, never data/responses/response_database.csv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.pilot_config import (
    FRAGMENTS_PATH,
    PILOT_MODELS,
    PILOT_PINNED_REPORT_IDS,
    PILOT_PROMPT_CONDITIONS,
    PILOT_RESPONSES_DIR,
    PILOT_RUN_NUMBER,
    PILOT_USAGE_LOG_PATH,
    compute_call_cost,
)
from src.api_client.retry import make_api_call_with_retry
from src.pilot.fragments import load_fragments, to_executor_fragment

PILOT_USAGE_LOG_COLUMNS: list[str] = [
    "model",
    "prompt_condition",
    "report_id",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost",
]

# Canned response used for --dry-run, so the orchestration path (fragment
# loading/adaptation, prompt rendering, CSV row shape) can be exercised
# without any real HTTP call.
_DRY_RUN_PARSED_RESPONSE: dict = {
    "classification": "sound",
    "rationale": "Dry-run rationale for smoke-testing the pilot pipeline.",
    "token_count_input": 800,
    "token_count_output": 120,
    "reasoning_tokens": None,
    "parse_method": "json",
    "latency_seconds": 0.0,
    "api_version": "dry-run",
}


def build_pilot_matrix(
    pinned_report_ids: list[str | int],
    models: list[str] = PILOT_MODELS,
    prompts: list[str] = PILOT_PROMPT_CONDITIONS,
) -> list[tuple[str, str, str]]:
    """
    Build the report_id x model x prompt_condition cross product.

    Pure function, no I/O — unit-testable in isolation.

    Args:
        pinned_report_ids: Fixed list of report_ids to run.
        models: Model identifiers to run each fragment against.
        prompts: Prompt conditions to run each fragment under.

    Returns:
        List of (report_id, model, prompt_type) tuples.
    """
    return [
        (report_id, model, prompt_type)
        for report_id in pinned_report_ids
        for model in models
        for prompt_type in prompts
    ]


def run_single_pilot_call(
    fragment_record: dict,
    model: str,
    prompt_type: str,
    dry_run: bool = False,
) -> dict:
    """
    Execute one pilot API call for a single (fragment, model, prompt) combo.

    Args:
        fragment_record: A fragments.jsonl record (report_id, fragment, ...).
        model: Model identifier from API_CONFIG.
        prompt_type: 'zero_shot' or 'few_shot'.
        dry_run: If True, skip the real HTTP call and return a canned result.

    Returns:
        Dict shaped like src.api_client.retry.make_api_call_with_retry's
        return value: {status, parsed_response, error, ...}.
    """
    fragment = to_executor_fragment(fragment_record)

    if dry_run:
        return {
            "fragment_id": fragment["id"],
            "model": model,
            "prompt_type": prompt_type,
            "run_number": PILOT_RUN_NUMBER,
            "status": "success",
            "parsed_response": dict(_DRY_RUN_PARSED_RESPONSE),
            "attempts": 1,
            "timestamp": datetime.now().isoformat(),
            "error": None,
        }

    return make_api_call_with_retry(
        fragment=fragment,
        model=model,
        prompt_type=prompt_type,
        run_number=PILOT_RUN_NUMBER,
    )


def process_pilot_result(result: dict, model: str, prompt_type: str, report_id: str) -> dict:
    """
    Build one pilot_usage_log.csv row from a call result.

    On failure, token/cost fields are None rather than the row being
    dropped — a failed pilot call is still worth an auditable row.

    Args:
        result: Return value of run_single_pilot_call.
        model: Model identifier.
        prompt_type: Prompt condition.
        report_id: The fragment's report_id.

    Returns:
        Dict with keys matching PILOT_USAGE_LOG_COLUMNS.
    """
    parsed = result.get("parsed_response") or {}
    input_tokens = parsed.get("token_count_input")
    output_tokens = parsed.get("token_count_output")
    reasoning_tokens = parsed.get("reasoning_tokens")

    cost = None
    if result["status"] == "success":
        cost = compute_call_cost(model, input_tokens or 0, output_tokens or 0, reasoning_tokens)

    return {
        "model": model,
        "prompt_condition": prompt_type,
        "report_id": report_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost": cost,
    }


def _append_row(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    pd.DataFrame([row], columns=PILOT_USAGE_LOG_COLUMNS).to_csv(
        output_path, mode="a", header=not file_exists, index=False
    )


def _save_raw_response(result: dict, model: str, prompt_type: str, report_id: str, response_dir: Path) -> None:
    response_dir.mkdir(parents=True, exist_ok=True)
    out_path = response_dir / f"{model}__{prompt_type}__{report_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def run_task2(
    pinned_report_ids: list[str | int] = PILOT_PINNED_REPORT_IDS,
    models: list[str] = PILOT_MODELS,
    prompts: list[str] = PILOT_PROMPT_CONDITIONS,
    output_path: Path = PILOT_USAGE_LOG_PATH,
    response_dir: Path = PILOT_RESPONSES_DIR,
    fragments_path: Path = FRAGMENTS_PATH,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run the Task 2 pilot API batch.

    Args:
        pinned_report_ids: Fixed report_ids to run — see
            config.pilot_config.PILOT_PINNED_REPORT_IDS's docstring for how
            to populate this after Task 1. Always required (including in
            dry runs) — a dry run against an explicit small fixture is how
            this gets smoke-tested before the real corpus/pins exist.
        models: Model identifiers to run.
        prompts: Prompt conditions to run.
        output_path: Where to append pilot_usage_log.csv rows.
        response_dir: Where to write raw per-call response JSON files.
        fragments_path: Path to the fragment corpus JSONL to read from.
        dry_run: If True, skip real HTTP calls and file writes; return the
            in-memory DataFrame only.

    Returns:
        DataFrame of all rows written (or that would have been written).

    Raises:
        ValueError: pinned_report_ids is empty.
    """
    if not pinned_report_ids:
        raise ValueError(
            "No pinned report_ids given. Run Task 1 first, inspect "
            "logs/corpus_token_audit.csv, and pin 3-5 report_ids "
            "(shortest, ~median, longest, plus 1-2 mid-range) in "
            "config/pilot_config.py (PILOT_PINNED_REPORT_IDS) before "
            "running Task 2 for real — or pass --report-ids explicitly "
            "for a --dry-run smoke test against a fixture corpus."
        )

    fragments = load_fragments(fragments_path)

    # Build report_id -> [matching records] first (not report_id -> record)
    # so an ambiguous report_id is caught explicitly rather than silently
    # resolving to whichever record happened to load last. The real corpus
    # has 2 report_ids that each cover 2 fragments (see
    # fragments.find_duplicate_report_ids) — spec says report_id is unique,
    # the actual data doesn't fully honor that.
    fragments_by_id: dict = {}
    for f in fragments:
        fragments_by_id.setdefault(f["report_id"], []).append(f)

    missing = [rid for rid in pinned_report_ids if rid not in fragments_by_id]
    if missing:
        raise ValueError(f"Pinned report_ids not found in fragments.jsonl: {missing}")

    ambiguous = {
        rid: [f["criterion"] for f in fragments_by_id[rid]]
        for rid in pinned_report_ids
        if len(fragments_by_id[rid]) > 1
    }
    if ambiguous:
        raise ValueError(
            f"Pinned report_id(s) match more than one fragment, refusing to "
            f"guess which one: {ambiguous}. Pick a different report_id, or "
            "disambiguate by criterion if you extend the schema."
        )

    fragments_by_id = {rid: records[0] for rid, records in fragments_by_id.items()}

    matrix = build_pilot_matrix(pinned_report_ids, models, prompts)
    total = len(matrix)
    rows: list[dict] = []

    print(
        f"Pilot batch: {len(pinned_report_ids)} fragments x {len(models)} models x "
        f"{len(prompts)} prompts x 1 run = {total} calls\n"
    )

    for i, (report_id, model, prompt_type) in enumerate(matrix, start=1):
        print(f"[{i}/{total}] {report_id} / {model} / {prompt_type}")
        result = run_single_pilot_call(fragments_by_id[report_id], model, prompt_type, dry_run=dry_run)
        row = process_pilot_result(result, model, prompt_type, report_id)
        rows.append(row)

        if not dry_run:
            _append_row(row, output_path)
            _save_raw_response(result, model, prompt_type, report_id, response_dir)

    df = pd.DataFrame(rows, columns=PILOT_USAGE_LOG_COLUMNS)
    if not dry_run:
        print(f"\nWrote {len(df)} rows to {output_path}")
    return df


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Task 2 — pilot API batch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model keys to run (default: all PILOT_MODELS)",
    )
    parser.add_argument(
        "--report-ids", nargs="+", default=None, type=int,
        help="Report IDs to run (default: PILOT_PINNED_REPORT_IDS). Required "
             "for --dry-run until PILOT_PINNED_REPORT_IDS is populated. "
             "report_id is an int in the real corpus (data/raw/fragments.jsonl) — "
             "pass bare numbers, e.g. --report-ids 4425 6782.",
    )
    parser.add_argument("--fragments-path", type=Path, default=FRAGMENTS_PATH)
    args = parser.parse_args()

    models = args.models if args.models else PILOT_MODELS
    report_ids = args.report_ids if args.report_ids else PILOT_PINNED_REPORT_IDS
    run_task2(
        pinned_report_ids=report_ids,
        models=models,
        fragments_path=args.fragments_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _main()
