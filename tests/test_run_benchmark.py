"""
Unit tests for src/api_client/run_benchmark.py.

Focus is on the properties that matter most for a real, money-spending run:
- duplicate report_ids are set aside, never silently dropped or merged
- resuming a session never redoes (and double-bills) an already-completed
  combo
- a session only ever touches the db_path it's given -- dry-run isolation
  is a correctness property, not just a convention
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.api_client.run_benchmark import (
    CALLS_PER_FRAGMENT,
    get_completed_combos,
    get_exhausted_combos,
    load_fragment_groups,
    run_benchmark_session,
    select_next_fragments,
)


@pytest.fixture
def fragments_with_duplicate(tmp_path):
    """3 unique report_ids + 1 duplicated report_id (2 fragments, 2 criteria)."""
    records = [
        {"report_id": 100, "criterion": "effectiveness", "paragraph_count": 1, "fragment": "Alpha."},
        {"report_id": 200, "criterion": "efficiency", "paragraph_count": 1, "fragment": "Beta."},
        {"report_id": 300, "criterion": "impact", "paragraph_count": 1, "fragment": "Gamma one."},
        {"report_id": 300, "criterion": "relevance", "paragraph_count": 1, "fragment": "Gamma two."},
    ]
    path = tmp_path / "fragments.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


class TestLoadFragmentGroups:
    def test_unambiguous_ids_loaded(self, fragments_with_duplicate):
        fragments_by_id, ambiguous = load_fragment_groups(fragments_with_duplicate)
        assert set(fragments_by_id.keys()) == {100, 200}

    def test_duplicate_id_set_aside_not_dropped(self, fragments_with_duplicate):
        fragments_by_id, ambiguous = load_fragment_groups(fragments_with_duplicate)
        assert 300 not in fragments_by_id
        assert ambiguous == {300: ["impact", "relevance"]}


class TestSelectNextFragments:
    def test_skips_ambiguous_report_id(self, fragments_with_duplicate):
        fragments_by_id, ambiguous = load_fragment_groups(fragments_with_duplicate)
        execution_order = [100, 200, 300]  # 300 is ambiguous -- not in fragments_by_id

        selected = select_next_fragments(
            execution_order, fragments_by_id, skip_combos=set(), n_fragments=10
        )

        selected_ids = [f["id"] for f in selected]
        assert 300 not in selected_ids
        assert set(selected_ids) == {100, 200}

    def test_skips_fully_completed_fragment(self, fragments_with_duplicate):
        from src.api_client.run_benchmark import MODEL_ORDER, PROMPT_CONDITIONS, RUNS_PER_COMBO

        fragments_by_id, _ = load_fragment_groups(fragments_with_duplicate)
        execution_order = [100, 200]

        # Fragment 100 fully done; 200 untouched.
        completed = {
            (100, model, prompt, run)
            for model in MODEL_ORDER
            for prompt in PROMPT_CONDITIONS
            for run in range(1, RUNS_PER_COMBO + 1)
        }

        selected = select_next_fragments(
            execution_order, fragments_by_id, skip_combos=completed, n_fragments=10
        )

        assert [f["id"] for f in selected] == [200]

    def test_respects_n_fragments_cap(self, fragments_with_duplicate):
        fragments_by_id, _ = load_fragment_groups(fragments_with_duplicate)
        execution_order = [100, 200]

        selected = select_next_fragments(
            execution_order, fragments_by_id, skip_combos=set(), n_fragments=1
        )

        assert len(selected) == 1


class TestGetCompletedCombos:
    def test_empty_when_db_missing(self, tmp_path):
        assert get_completed_combos(tmp_path / "nonexistent.csv") == set()

    def test_reads_back_written_combos(self, tmp_path):
        db_path = tmp_path / "response_database.csv"
        df = pd.DataFrame([
            {"fragment_id": 100, "model_family": "claude_opus_5", "prompt_condition": "zero_shot", "run_number": 1},
            {"fragment_id": 100, "model_family": "claude_opus_5", "prompt_condition": "zero_shot", "run_number": 2},
        ])
        df.to_csv(db_path, index=False)

        combos = get_completed_combos(db_path)

        assert combos == {
            (100, "claude_opus_5", "zero_shot", 1),
            (100, "claude_opus_5", "zero_shot", 2),
        }


class TestGetExhaustedCombos:
    def test_empty_when_log_missing(self, tmp_path):
        assert get_exhausted_combos(tmp_path / "nonexistent.jsonl") == set()

    def test_reads_back_logged_failures(self, tmp_path):
        log_path = tmp_path / "failed_calls.jsonl"
        records = [
            {"fragment_id": 2336, "model": "gemini_3_pro_preview_high", "prompt_type": "zero_shot", "run_number": 1},
            {"fragment_id": 2336, "model": "gemini_3_pro_preview_high", "prompt_type": "zero_shot", "run_number": 1},
        ]
        with log_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        combos = get_exhausted_combos(log_path)

        assert combos == {(2336, "gemini_3_pro_preview_high", "zero_shot", 1)}


class TestExhaustedFailureIsNotReattemptedOnResume:
    """
    Regression test for the real bug hit during the first live benchmark
    session (2026-10-20): a combo that genuinely exhausted its retries (and
    got logged to failed_calls.jsonl) was still "not in the response
    database," so every resumed session re-attempted it from scratch --
    burning another full retry budget (~16 minutes at the 300s/attempt,
    3-attempt schedule) before making any further progress.
    """

    def test_exhausted_combo_is_skipped_not_reattempted(self, tmp_path):
        from src.api_client.retry import log_failed_call

        db_path = tmp_path / "response_database.csv"
        failed_log_path = tmp_path / "failed_calls.jsonl"
        fragment = {"id": 2336, "text": "Alpha."}

        # Simulate one combo having already exhausted its retries in a
        # prior session (written directly, not via a real API call).
        log_failed_call(
            {
                "fragment_id": 2336,
                "model": "gemini_3_pro_preview_high",
                "prompt_type": "zero_shot",
                "run_number": 1,
                "status": "failed",
                "error": {"category": "timeout", "message": "timed out after 300s"},
                "attempts": 3,
                "timestamp": "2026-10-20T17:08:23",
            },
            log_path=failed_log_path,
        )

        completed = get_completed_combos(db_path)  # empty -- nothing succeeded yet
        exhausted = get_exhausted_combos(failed_log_path)
        skip_combos = completed | exhausted

        summary = run_benchmark_session(
            [fragment],
            skip_combos=skip_combos,
            db_path=db_path,
            failed_log_path=failed_log_path,
            dry_run=True,
        )

        # The exhausted combo must be skipped, not attempted again --
        # dry_run always "succeeds," so if this assertion ever fails, the
        # combo was silently re-run rather than respected as already handled.
        assert summary["attempted"] == CALLS_PER_FRAGMENT - 1
        assert summary["skipped"] == 1

        # The failed-calls log must not have grown -- the whole point is
        # that this combo isn't re-attempted (and so can't re-fail) here.
        with failed_log_path.open("r", encoding="utf-8") as fh:
            assert sum(1 for line in fh if line.strip()) == 1


class TestRunBenchmarkSessionDryRunResume:
    def test_session_writes_exactly_calls_per_fragment_rows(self, tmp_path):
        db_path = tmp_path / "response_database.csv"
        fragment = {"id": 100, "text": "Alpha."}

        summary = run_benchmark_session(
            [fragment], skip_combos=set(), db_path=db_path, dry_run=True
        )

        df = pd.read_csv(db_path)
        assert len(df) == CALLS_PER_FRAGMENT
        assert summary["succeeded"] == CALLS_PER_FRAGMENT
        assert summary["skipped"] == 0

    def test_resuming_skips_already_completed_combos(self, tmp_path):
        db_path = tmp_path / "response_database.csv"
        fragment = {"id": 100, "text": "Alpha."}

        # First "session": everything runs.
        run_benchmark_session([fragment], skip_combos=set(), db_path=db_path, dry_run=True)
        first_row_count = len(pd.read_csv(db_path))

        # Simulate re-invoking the driver after an interruption: recompute
        # completed_combos from the database exactly as run_next_session
        # would, then run the "same" fragment again.
        completed = get_completed_combos(db_path)
        summary = run_benchmark_session(
            [fragment], skip_combos=completed, db_path=db_path, dry_run=True
        )

        second_row_count = len(pd.read_csv(db_path))

        assert summary["attempted"] == 0
        assert summary["skipped"] == CALLS_PER_FRAGMENT
        assert second_row_count == first_row_count, (
            "Re-running an already-completed fragment must not write duplicate rows"
        )

    def test_dry_run_never_touches_a_path_other_than_db_path(self, tmp_path):
        """
        dry_run=True must not write anywhere except the db_path explicitly
        passed in -- this is what makes --dry-run safe to smoke-test with
        before any real API key is ever used.
        """
        db_path = tmp_path / "isolated.csv"
        fragment = {"id": 100, "text": "Alpha."}

        run_benchmark_session([fragment], skip_combos=set(), db_path=db_path, dry_run=True)

        written_files = list(tmp_path.rglob("*"))
        assert written_files == [db_path]
