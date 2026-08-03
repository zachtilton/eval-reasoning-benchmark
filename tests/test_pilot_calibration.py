"""
Unit tests for src/pilot/calibration.py.
"""

from __future__ import annotations

import pytest

from src.pilot.calibration import extract_calibration_block


class TestExtractCalibrationBlock:
    def test_extracts_block_between_markers(self):
        text = (
            "TASK: ...\n"
            "CALIBRATION EXAMPLES:\n"
            "Example 1 of 2 — Classification: sound\n"
            "...content...\n"
            "NOW EVALUATE THIS FRAGMENT:\n"
            "[Target fragment text will be inserted here]"
        )
        block = extract_calibration_block(text)
        assert "Example 1 of 2" in block
        assert "...content..." in block
        assert "NOW EVALUATE" not in block
        assert "CALIBRATION EXAMPLES:" not in block

    def test_missing_start_marker_raises(self):
        text = "TASK: ...\nNOW EVALUATE THIS FRAGMENT:\n[Fragment text will be inserted here]"
        with pytest.raises(ValueError, match="CALIBRATION EXAMPLES:"):
            extract_calibration_block(text)

    def test_missing_end_marker_raises(self):
        text = "TASK: ...\nCALIBRATION EXAMPLES:\n...content..."
        with pytest.raises(ValueError, match="NOW EVALUATE THIS FRAGMENT:"):
            extract_calibration_block(text)

    def test_against_real_few_shot_template(self):
        """Regression: the actual few_shot_template.txt survives extraction."""
        from src.api_client.config import PROMPTS_DIR

        template_text = (PROMPTS_DIR / "few_shot_template.txt").read_text(encoding="utf-8")
        block = extract_calibration_block(template_text)
        assert "Example 1 of 2" in block
        assert "Example 2 of 2" in block
        assert len(block) > 0
