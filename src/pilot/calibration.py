"""
Calibration-example token measurement for the pilot corpus token audit.

The 2 finalized calibration examples live embedded inside
config/prompts/few_shot_template.txt (between the "CALIBRATION EXAMPLES:"
and "NOW EVALUATE THIS FRAGMENT:" marker lines) rather than in a separate
fixture file. Slicing the live template — instead of duplicating the
examples into their own file — means this measurement can't drift out of
sync with whatever calibration text is actually in the template; the
trade-off is that it depends on those two marker lines staying present,
which is structural boilerplate unlikely to change independently of the
examples themselves.
"""

from __future__ import annotations

from pathlib import Path

from src.api_client.config import PROMPTS_DIR

CALIBRATION_START_MARKER = "CALIBRATION EXAMPLES:"
CALIBRATION_END_MARKER = "NOW EVALUATE THIS FRAGMENT:"


def extract_calibration_block(template_text: str) -> str:
    """
    Slice the calibration-examples block out of the few-shot template text.

    Args:
        template_text: Full text of config/prompts/few_shot_template.txt.

    Returns:
        The text between the CALIBRATION_START_MARKER and
        CALIBRATION_END_MARKER lines (exclusive of both markers).

    Raises:
        ValueError: Either marker is missing from the template text.
    """
    start = template_text.find(CALIBRATION_START_MARKER)
    if start == -1:
        raise ValueError(
            f"Marker '{CALIBRATION_START_MARKER}' not found in template text."
        )
    start += len(CALIBRATION_START_MARKER)

    end = template_text.find(CALIBRATION_END_MARKER, start)
    if end == -1:
        raise ValueError(
            f"Marker '{CALIBRATION_END_MARKER}' not found in template text."
        )

    return template_text[start:end].strip()


def count_calibration_tokens(
    encoding,
    prompts_dir: Path = PROMPTS_DIR,
) -> int:
    """
    Count tokens in the finalized calibration-examples block.

    Args:
        encoding: A tiktoken encoding object (e.g. from
            tiktoken.get_encoding("cl100k_base")).
        prompts_dir: Directory containing few_shot_template.txt.

    Returns:
        Token count of the calibration-examples block.
    """
    template_path = prompts_dir / "few_shot_template.txt"
    template_text = template_path.read_text(encoding="utf-8")
    block = extract_calibration_block(template_text)
    return len(encoding.encode(block))
