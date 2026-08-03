"""
Unit tests for src/pilot/fragments.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pilot.fragments import (
    find_duplicate_report_ids,
    load_fragments,
    to_executor_fragment,
    validate_fragment_count,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_fragments.jsonl"


class TestLoadFragments:
    def test_loads_fixture(self):
        fragments = load_fragments(FIXTURE_PATH)
        assert len(fragments) == 5
        assert fragments[0]["report_id"] == "TEST-2020-001"
        assert "fragment" in fragments[0]

    def test_missing_file_raises_clear_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError, match="Fragment corpus not found"):
            load_fragments(missing)


class TestToExecutorFragment:
    def test_adapts_shape(self):
        record = {"report_id": "R-001", "fragment": "Some text.", "criterion": "relevance"}
        adapted = to_executor_fragment(record)
        assert adapted == {"id": "R-001", "text": "Some text."}


class TestValidateFragmentCount:
    def test_warns_on_mismatch(self):
        with pytest.warns(UserWarning, match="Loaded 5 fragments, expected 150"):
            validate_fragment_count(load_fragments(FIXTURE_PATH), expected=150)

    def test_no_warning_on_match(self, recwarn):
        validate_fragment_count(load_fragments(FIXTURE_PATH), expected=5)
        assert len(recwarn) == 0


class TestFindDuplicateReportIds:
    def test_no_duplicates_in_fixture(self):
        assert find_duplicate_report_ids(load_fragments(FIXTURE_PATH)) == {}

    def test_detects_duplicate_report_id(self):
        fragments = [
            {"report_id": 1359, "criterion": "relevance", "fragment": "a"},
            {"report_id": 1359, "criterion": "sustainability", "fragment": "b"},
            {"report_id": 42, "criterion": "impact", "fragment": "c"},
        ]
        duplicates = find_duplicate_report_ids(fragments)
        assert duplicates == {1359: ["relevance", "sustainability"]}
        assert 42 not in duplicates

    def test_against_real_corpus_has_no_duplicates(self):
        """
        Regression: an earlier upload of the real corpus had 2 duplicated
        report_ids (1359, 1383) — fixed by renaming the second occurrence
        of each to 1360/1361. This asserts the fix stuck; a future
        re-upload reintroducing duplicates should fail this test rather
        than silently corrupting PILOT_PINNED_REPORT_IDS resolution.
        """
        from config.pilot_config import FRAGMENTS_PATH

        if not FRAGMENTS_PATH.exists():
            pytest.skip("real fragments.jsonl not present in this environment")

        duplicates = find_duplicate_report_ids(load_fragments(FRAGMENTS_PATH))
        assert duplicates == {}
