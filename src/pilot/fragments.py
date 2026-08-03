"""
Fragment loading and adaptation for the pilot cost/token audit.

fragments.jsonl uses report_id/fragment field names. The spec describes
report_id as unique per fragment ("one fragment per report... no separate
fragment_id field"), but the real uploaded corpus (data/raw/fragments.jsonl)
does not fully honor that: 2 of 150 report_ids each cover two fragments
(different criteria extracted from the same report). find_duplicate_report_ids
surfaces this so Task 1/2 can flag it instead of silently resolving to the
wrong fragment. report_id itself is an int in the real corpus, not a string.

The existing api_client layer (executor.execute_api_request,
retry.make_api_call_with_retry) expects a fragment dict shaped
{"id": ..., "text": ...}. to_executor_fragment is the single place that
shape mismatch is bridged, so it isn't duplicated across task2/tests.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from config.pilot_config import FRAGMENTS_PATH


def load_fragments(path: Path = FRAGMENTS_PATH) -> list[dict]:
    """
    Load the fragment corpus from fragments.jsonl.

    Args:
        path: Path to the JSONL fragment corpus (one JSON object per line,
            each with report_id, criterion, paragraph_count, fragment).

    Returns:
        List of fragment record dicts, in file order.

    Raises:
        FileNotFoundError: The corpus file doesn't exist yet — expected
            until the real 150-fragment corpus is generated.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fragment corpus not found: {path}\n"
            "Expected a JSONL file with one JSON object per line, each "
            "shaped {\"report_id\": ..., \"criterion\": ..., "
            "\"paragraph_count\": ..., \"fragment\": ...}."
        )

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def to_executor_fragment(record: dict) -> dict:
    """
    Adapt a fragments.jsonl record to the {"id", "text"} shape expected by
    src/api_client/executor.py and src/api_client/retry.py.

    Args:
        record: A fragment dict with at least "report_id" and "fragment" keys.

    Returns:
        Dict with "id" (the report_id) and "text" (the fragment text).
    """
    return {"id": record["report_id"], "text": record["fragment"]}


def validate_fragment_count(fragments: list[dict], expected: int = 150) -> None:
    """
    Warn (not raise) if the loaded corpus doesn't match the expected size.

    A mismatch is expected when testing against a small fixture file, so
    this is a warning rather than a hard failure.

    Args:
        fragments: Loaded fragment records.
        expected: Expected corpus size (default 150, per spec).
    """
    if len(fragments) != expected:
        warnings.warn(
            f"Loaded {len(fragments)} fragments, expected {expected}. "
            "This is expected when running against a test fixture.",
            stacklevel=2,
        )


def find_duplicate_report_ids(fragments: list[dict]) -> dict[Any, list[str]]:
    """
    Find report_ids that appear more than once in the corpus.

    The spec describes report_id as unique per fragment, but the real
    corpus doesn't fully honor that (2 report_ids each cover 2 fragments
    under different criteria as of the Aug 2026 upload). Callers should
    refuse to resolve an ambiguous report_id to a single fragment rather
    than silently picking one (see task2_pilot_batch.run_task2).

    Args:
        fragments: Loaded fragment records.

    Returns:
        Dict of report_id -> list of criterion values, for every report_id
        that appears more than once. Empty if the corpus is fully unique.
    """
    counts = Counter(f["report_id"] for f in fragments)
    duplicated_ids = {rid for rid, n in counts.items() if n > 1}
    return {
        rid: [f["criterion"] for f in fragments if f["report_id"] == rid]
        for rid in duplicated_ids
    }
