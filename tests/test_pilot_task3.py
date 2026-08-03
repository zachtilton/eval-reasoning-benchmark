"""
Unit tests for src/pilot/task3_cost_projection.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pilot.task3_cost_projection import (
    compute_updated_projection,
    flag_high_reasoning_models,
)


def _task1_result():
    return {
        "fragment_stats": {"n_fragments": 150, "mean": 300.0, "median": 290.0, "min": 100, "max": 600, "sd": 80.0},
        "calibration_tokens": 220,
        "scaffold_tokens": {"zero_shot": 500, "few_shot": 720},
    }


class TestComputeUpdatedProjection:
    def test_projects_input_tokens_from_scaffold_plus_avg_fragment(self):
        df = pd.DataFrame([
            {"model": "claude_opus_5", "prompt_condition": "zero_shot", "report_id": "R1",
             "input_tokens": 800, "output_tokens": 40, "reasoning_tokens": None, "cost": 0.02},
            {"model": "claude_opus_5", "prompt_condition": "zero_shot", "report_id": "R2",
             "input_tokens": 850, "output_tokens": 60, "reasoning_tokens": None, "cost": 0.03},
        ])
        projection = compute_updated_projection(_task1_result(), df)
        row = projection.iloc[0]
        assert row["model"] == "claude_opus_5"
        assert row["prompt_condition"] == "zero_shot"
        # scaffold (500) + avg fragment tokens (300) = 800
        assert row["projected_input_tokens"] == 800.0
        assert row["avg_output_tokens"] == 50.0
        assert row["cost_per_call"] is not None  # claude_opus_5 has verified rates
        assert row["n_calls"] == 450
        # total_cost is intentionally rounded to 2 decimals in compute_updated_projection
        assert row["total_cost"] == pytest.approx(row["cost_per_call"] * 450, abs=0.01)

    def test_unverified_rate_model_has_none_cost(self):
        df = pd.DataFrame([
            {"model": "gpt_5_6_sol", "prompt_condition": "few_shot", "report_id": "R1",
             "input_tokens": 900, "output_tokens": 70, "reasoning_tokens": 200, "cost": None},
        ])
        projection = compute_updated_projection(_task1_result(), df)
        row = projection.iloc[0]
        assert row["cost_per_call"] is None
        assert row["total_cost"] is None

    def test_all_none_reasoning_tokens_does_not_crash(self):
        df = pd.DataFrame([
            {"model": "claude_opus_5", "prompt_condition": "zero_shot", "report_id": "R1",
             "input_tokens": 800, "output_tokens": 40, "reasoning_tokens": None, "cost": 0.02},
        ])
        projection = compute_updated_projection(_task1_result(), df)
        assert projection.iloc[0]["avg_reasoning_tokens"] is None


class TestFlagHighReasoningModels:
    def test_claude_family_reports_na(self):
        df = pd.DataFrame([
            {"model": "claude_opus_5", "output_tokens": 40, "reasoning_tokens": np.nan},
            {"model": "claude_opus_5", "output_tokens": 50, "reasoning_tokens": np.nan},
        ])
        statuses = flag_high_reasoning_models(df)
        assert statuses["claude_opus_5"].startswith("N/A")

    def test_high_ratio_flagged(self):
        df = pd.DataFrame([
            {"model": "gpt_5_6_sol", "output_tokens": 50, "reasoning_tokens": 300},
        ])
        statuses = flag_high_reasoning_models(df, threshold=3.0)
        assert statuses["gpt_5_6_sol"].startswith("FLAGGED")

    def test_low_ratio_ok(self):
        df = pd.DataFrame([
            {"model": "kimi_k3", "output_tokens": 50, "reasoning_tokens": 50},
        ])
        statuses = flag_high_reasoning_models(df, threshold=3.0)
        assert statuses["kimi_k3"].startswith("OK")

    def test_does_not_crash_on_mixed_none_and_values(self):
        df = pd.DataFrame([
            {"model": "gemini_3_pro_preview_high", "output_tokens": 40, "reasoning_tokens": 100},
            {"model": "gemini_3_pro_preview_high", "output_tokens": 60, "reasoning_tokens": np.nan},
        ])
        statuses = flag_high_reasoning_models(df, threshold=3.0)
        assert "gemini_3_pro_preview_high" in statuses
