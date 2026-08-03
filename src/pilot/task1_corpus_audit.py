"""
Task 1 — Corpus token audit.

Measures real cl100k_base token counts across the fragment corpus and the
finalized calibration examples, so the pilot's downstream cost projection
(Task 3) is grounded in real numbers instead of the ~750-token estimate in
config/model_params.py's EXPECTED_TOKENS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tiktoken

from config.pilot_config import (
    CALIBRATION_TOKEN_AUDIT_PATH,
    CL100K_ENCODING_NAME,
    CORPUS_TOKEN_AUDIT_PATH,
    FRAGMENTS_PATH,
)
from src.api_client.config import PROMPTS_DIR
from src.api_client.executor import load_prompt_template
from src.pilot.calibration import count_calibration_tokens
from src.pilot.fragments import find_duplicate_report_ids, load_fragments, validate_fragment_count


def count_tokens(text: str, encoding) -> int:
    """Count tokens in a text string using the given tiktoken encoding."""
    return len(encoding.encode(text))


def compute_fragment_stats(fragments: list[dict], encoding) -> pd.DataFrame:
    """
    Compute per-fragment token counts, sorted ascending by token count.

    The ascending sort is the "ranked" order Task 2's fixed pin list
    (config.pilot_config.PILOT_PINNED_REPORT_IDS) is chosen from —
    shortest, ~median, longest, plus 1-2 mid-range.

    Args:
        fragments: Loaded fragment records (report_id, criterion,
            paragraph_count, fragment).
        encoding: A tiktoken encoding object.

    Returns:
        DataFrame with columns report_id, criterion, paragraph_count,
        token_count, sorted ascending by token_count.
    """
    rows = [
        {
            "report_id": f["report_id"],
            "criterion": f.get("criterion"),
            "paragraph_count": f.get("paragraph_count"),
            "token_count": count_tokens(f["fragment"], encoding),
        }
        for f in fragments
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("token_count", ascending=True).reset_index(drop=True)


def count_scaffold_tokens(
    prompt_type: str,
    encoding,
    prompts_dir: Path = PROMPTS_DIR,
) -> int:
    """
    Count tokens in a prompt template's fixed scaffold (everything except
    the target fragment text) — task instructions, the 22-checkpoint IERC
    block, and for few-shot, the 2 calibration examples.

    Reuses executor.load_prompt_template with an empty fragment_text so the
    scaffold token count can never drift out of sync with what the real
    executor actually renders.

    Args:
        prompt_type: 'zero_shot' or 'few_shot'.
        encoding: A tiktoken encoding object.
        prompts_dir: Directory containing the *_template.txt files.

    Returns:
        Token count of the scaffold (template with fragment_text="").
    """
    scaffold_text = load_prompt_template(prompt_type, "", prompts_dir=prompts_dir)
    return len(encoding.encode(scaffold_text))


def summarize_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean/median/min/max/SD over the fragment token counts.

    Args:
        df: DataFrame from compute_fragment_stats (must have a token_count column).

    Returns:
        Dict with keys: n_fragments, mean, median, min, max, sd.
    """
    tc = df["token_count"]
    return {
        "n_fragments": int(len(df)),
        "mean": float(tc.mean()),
        "median": float(tc.median()),
        "min": int(tc.min()),
        "max": int(tc.max()),
        "sd": float(tc.std()) if len(df) > 1 else 0.0,
    }


def run_task1(
    fragments_path: Path = FRAGMENTS_PATH,
    output_path: Path = CORPUS_TOKEN_AUDIT_PATH,
    calibration_output_path: Path = CALIBRATION_TOKEN_AUDIT_PATH,
    dry_run: bool = False,
) -> dict:
    """
    Run the Task 1 corpus token audit.

    Args:
        fragments_path: Path to fragments.jsonl.
        output_path: Where to write the per-fragment token audit CSV.
        calibration_output_path: Where to write the calibration-example
            token audit CSV.
        dry_run: If True, skip all file writes; only compute and return.

    Returns:
        Dict: {"fragment_stats": <summarize_stats dict>,
               "calibration_tokens": int,
               "scaffold_tokens": {"zero_shot": int, "few_shot": int},
               "duplicate_report_ids": {report_id: [criteria]}}.
    """
    encoding = tiktoken.get_encoding(CL100K_ENCODING_NAME)

    fragments = load_fragments(fragments_path)
    validate_fragment_count(fragments)

    duplicate_report_ids = find_duplicate_report_ids(fragments)
    if duplicate_report_ids:
        print(
            f"WARNING: {len(duplicate_report_ids)} report_id(s) appear more than "
            "once in the corpus — do not pin these in PILOT_PINNED_REPORT_IDS, "
            "Task 2 will refuse to run against an ambiguous report_id:"
        )
        for rid, criteria in duplicate_report_ids.items():
            print(f"  report_id={rid}: criteria={criteria}")

    df = compute_fragment_stats(fragments, encoding)
    summary = summarize_stats(df)

    calibration_tokens = count_calibration_tokens(encoding)
    scaffold_tokens = {
        "zero_shot": count_scaffold_tokens("zero_shot", encoding),
        "few_shot": count_scaffold_tokens("few_shot", encoding),
    }

    print("Corpus token audit:")
    print(f"  n_fragments: {summary['n_fragments']}")
    print(f"  mean:   {summary['mean']:.1f}")
    print(f"  median: {summary['median']:.1f}")
    print(f"  min:    {summary['min']}")
    print(f"  max:    {summary['max']}")
    print(f"  sd:     {summary['sd']:.1f}")
    print(
        f"Calibration examples: {calibration_tokens} tokens "
        "(fixed constant, not pooled with corpus stats)"
    )
    print(
        f"Scaffold tokens — zero_shot: {scaffold_tokens['zero_shot']}, "
        f"few_shot: {scaffold_tokens['few_shot']} (includes calibration examples)"
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        calibration_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"item": "calibration_examples_block", "token_count": calibration_tokens},
                {"item": "scaffold_zero_shot", "token_count": scaffold_tokens["zero_shot"]},
                {"item": "scaffold_few_shot", "token_count": scaffold_tokens["few_shot"]},
            ]
        ).to_csv(calibration_output_path, index=False)

        print(f"Wrote {output_path}")
        print(f"Wrote {calibration_output_path}")

    return {
        "fragment_stats": summary,
        "calibration_tokens": calibration_tokens,
        "scaffold_tokens": scaffold_tokens,
        "duplicate_report_ids": duplicate_report_ids,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Task 1 — corpus token audit")
    parser.add_argument("--fragments-path", type=Path, default=FRAGMENTS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_task1(fragments_path=args.fragments_path, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
