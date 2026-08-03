"""
Unit tests for src/pilot/task1_corpus_audit.py.

Uses the real tiktoken cl100k_base encoding (no network dependency, so no
mocking needed) against tests/fixtures/sample_fragments.jsonl.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tiktoken

from config.pilot_config import CL100K_ENCODING_NAME
from src.pilot.task1_corpus_audit import (
    compute_fragment_stats,
    count_scaffold_tokens,
    count_tokens,
    run_task1,
    summarize_stats,
)
from src.pilot.fragments import load_fragments

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_fragments.jsonl"
ENCODING = tiktoken.get_encoding(CL100K_ENCODING_NAME)


class TestCountTokens:
    def test_counts_nonzero_tokens(self):
        assert count_tokens("The Project met its stated objectives.", ENCODING) > 0

    def test_empty_string_is_zero(self):
        assert count_tokens("", ENCODING) == 0


class TestComputeFragmentStats:
    def test_returns_sorted_ascending(self):
        fragments = load_fragments(FIXTURE_PATH)
        df = compute_fragment_stats(fragments, ENCODING)
        assert len(df) == 5
        assert list(df["token_count"]) == sorted(df["token_count"])
        assert set(df.columns) == {"report_id", "criterion", "paragraph_count", "token_count"}

    def test_shortest_and_longest_fragments_at_ends(self):
        fragments = load_fragments(FIXTURE_PATH)
        df = compute_fragment_stats(fragments, ENCODING)
        # TEST-2020-001 is the shortest fragment, TEST-2020-004 the longest
        assert df.iloc[0]["report_id"] == "TEST-2020-001"
        assert df.iloc[-1]["report_id"] == "TEST-2020-004"


class TestSummarizeStats:
    def test_summary_keys_and_values(self):
        df = pd.DataFrame({"token_count": [10, 20, 30, 40, 50]})
        summary = summarize_stats(df)
        assert summary["n_fragments"] == 5
        assert summary["mean"] == 30.0
        assert summary["median"] == 30.0
        assert summary["min"] == 10
        assert summary["max"] == 50
        assert summary["sd"] > 0

    def test_single_row_sd_is_zero(self):
        df = pd.DataFrame({"token_count": [42]})
        summary = summarize_stats(df)
        assert summary["sd"] == 0.0


class TestScaffoldTokens:
    def test_zero_shot_and_few_shot_scaffolds_nonzero(self):
        zero_shot = count_scaffold_tokens("zero_shot", ENCODING)
        few_shot = count_scaffold_tokens("few_shot", ENCODING)
        assert zero_shot > 0
        assert few_shot > 0
        # few-shot scaffold includes the 2 calibration examples, so it must
        # be substantially larger than zero-shot's bare checklist + instructions
        assert few_shot > zero_shot


class TestRunTask1DryRun:
    def test_dry_run_skips_file_writes(self, tmp_path):
        output_path = tmp_path / "corpus_token_audit.csv"
        calibration_path = tmp_path / "calibration_token_audit.csv"

        result = run_task1(
            fragments_path=FIXTURE_PATH,
            output_path=output_path,
            calibration_output_path=calibration_path,
            dry_run=True,
        )

        assert not output_path.exists()
        assert not calibration_path.exists()
        assert result["fragment_stats"]["n_fragments"] == 5
        assert result["calibration_tokens"] > 0
        assert result["scaffold_tokens"]["zero_shot"] > 0
        assert result["scaffold_tokens"]["few_shot"] > 0

    def test_real_run_writes_files(self, tmp_path):
        output_path = tmp_path / "corpus_token_audit.csv"
        calibration_path = tmp_path / "calibration_token_audit.csv"

        run_task1(
            fragments_path=FIXTURE_PATH,
            output_path=output_path,
            calibration_output_path=calibration_path,
            dry_run=False,
        )

        assert output_path.exists()
        assert calibration_path.exists()
        written = pd.read_csv(output_path)
        assert len(written) == 5
