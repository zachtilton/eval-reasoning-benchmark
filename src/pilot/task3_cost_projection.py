"""
Task 3 — Updated cost projection.

Recomputes the full 5,400-call benchmark cost projection (900 calls/model:
450 zero-shot / 450 few-shot) using Task 1's real average fragment token
count + fixed scaffold token counts (task instructions + IERC checklist,
plus calibration examples for few-shot) and Task 2's real per-model
output/reasoning-token figures, superseding the pre-registered
$168-173 estimate built on the ~750-token EXPECTED_TOKENS assumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.pilot_config import (
    CALIBRATION_TOKEN_AUDIT_PATH,
    CORPUS_TOKEN_AUDIT_PATH,
    PILOT_COST_PROJECTION_PATH,
    PILOT_USAGE_LOG_PATH,
    REASONING_FLAG_THRESHOLD,
    compute_call_cost,
)
from src.pilot.task1_corpus_audit import run_task1, summarize_stats
from src.pilot.task2_pilot_batch import run_task2

N_CALLS_PER_MODEL_PROMPT: int = 450  # 150 fragments x 3 runs


def compute_updated_projection(
    task1_result: dict,
    pilot_usage_df: pd.DataFrame,
    n_calls_per_model_prompt: int = N_CALLS_PER_MODEL_PROMPT,
) -> pd.DataFrame:
    """
    Recompute the per-model, per-prompt-condition cost projection.

    projected_input_tokens = Task 1's fixed scaffold tokens (per prompt
    condition — few-shot's scaffold already includes the calibration
    examples) + Task 1's corpus-wide average fragment token count.
    avg_output_tokens / avg_reasoning_tokens are the real Task 2 averages
    per (model, prompt_condition), across whichever pinned fragments ran.

    Args:
        task1_result: Return value of task1_corpus_audit.run_task1.
        pilot_usage_df: Return value of task2_pilot_batch.run_task2 (or an
            equivalent DataFrame with model/prompt_condition/output_tokens/
            reasoning_tokens columns).
        n_calls_per_model_prompt: Calls per (model, prompt) in the real
            5,400-call run (default 450 = 150 fragments x 3 runs).

    Returns:
        DataFrame with one row per (model, prompt_condition): model,
        prompt_condition, projected_input_tokens, avg_output_tokens,
        avg_reasoning_tokens, cost_per_call, n_calls, total_cost.
    """
    avg_fragment_tokens = task1_result["fragment_stats"]["mean"]
    scaffold_tokens = task1_result["scaffold_tokens"]

    rows: list[dict] = []
    grouped = pilot_usage_df.groupby(["model", "prompt_condition"], dropna=False)
    for (model, prompt_condition), group in grouped:
        avg_output_tokens = group["output_tokens"].mean()
        avg_reasoning_tokens = group["reasoning_tokens"].mean()  # NaN if all None
        reasoning_for_cost = None if pd.isna(avg_reasoning_tokens) else avg_reasoning_tokens

        projected_input_tokens = scaffold_tokens.get(prompt_condition, 0) + avg_fragment_tokens

        cost_per_call = compute_call_cost(
            model, projected_input_tokens, avg_output_tokens or 0, reasoning_for_cost
        )
        total_cost = (
            round(cost_per_call * n_calls_per_model_prompt, 2)
            if cost_per_call is not None
            else None
        )

        rows.append({
            "model": model,
            "prompt_condition": prompt_condition,
            "projected_input_tokens": round(projected_input_tokens, 1),
            "avg_output_tokens": round(avg_output_tokens, 1) if pd.notna(avg_output_tokens) else None,
            "avg_reasoning_tokens": round(avg_reasoning_tokens, 1) if pd.notna(avg_reasoning_tokens) else None,
            "cost_per_call": cost_per_call,
            "n_calls": n_calls_per_model_prompt,
            "total_cost": total_cost,
        })

    return pd.DataFrame(rows)


def flag_high_reasoning_models(
    pilot_usage_df: pd.DataFrame,
    threshold: float = REASONING_FLAG_THRESHOLD,
) -> dict[str, str]:
    """
    Flag models where reasoning tokens are a large multiple of visible
    output tokens — the main source of uncertainty in the pre-registered
    cost estimate.

    Args:
        pilot_usage_df: Pilot usage rows (model, output_tokens,
            reasoning_tokens columns).
        threshold: Flag when mean(reasoning_tokens) / mean(output_tokens)
            meets or exceeds this multiple.

    Returns:
        Dict of model -> status string. Claude-family models (or any model
        that never reports reasoning_tokens) get an explicit "N/A" status
        rather than a crash or a silent 0x.
    """
    statuses: dict[str, str] = {}
    for model, group in pilot_usage_df.groupby("model"):
        if group["reasoning_tokens"].isna().all():
            statuses[model] = "N/A — reasoning tokens not separately reported by this provider"
            continue

        mean_reasoning = group["reasoning_tokens"].mean()
        mean_output = group["output_tokens"].mean()
        if not mean_output:
            statuses[model] = "N/A — no output tokens recorded"
            continue

        ratio = mean_reasoning / mean_output
        if ratio >= threshold:
            statuses[model] = f"FLAGGED — reasoning tokens are {ratio:.1f}x visible output tokens"
        else:
            statuses[model] = f"OK — reasoning tokens are {ratio:.1f}x visible output tokens"

    return statuses


def _load_task1_result_from_disk(
    corpus_path: Path = CORPUS_TOKEN_AUDIT_PATH,
    calibration_path: Path = CALIBRATION_TOKEN_AUDIT_PATH,
) -> dict:
    """Reconstruct run_task1's return shape from previously written CSVs."""
    if not corpus_path.exists() or not calibration_path.exists():
        raise FileNotFoundError(
            f"Task 1 outputs not found ({corpus_path}, {calibration_path}). "
            "Run Task 1 first (without --skip-task1)."
        )

    fragment_df = pd.read_csv(corpus_path)
    fragment_stats = summarize_stats(fragment_df)

    calibration_df = pd.read_csv(calibration_path).set_index("item")["token_count"]
    return {
        "fragment_stats": fragment_stats,
        "calibration_tokens": int(calibration_df["calibration_examples_block"]),
        "scaffold_tokens": {
            "zero_shot": int(calibration_df["scaffold_zero_shot"]),
            "few_shot": int(calibration_df["scaffold_few_shot"]),
        },
    }


def _load_pilot_usage_from_disk(usage_path: Path = PILOT_USAGE_LOG_PATH) -> pd.DataFrame:
    """Load a previously written pilot_usage_log.csv."""
    if not usage_path.exists():
        raise FileNotFoundError(
            f"Task 2 output not found ({usage_path}). Run Task 2 first "
            "(without --skip-task2)."
        )
    return pd.read_csv(usage_path)


def run_task3(
    output_path: Path = PILOT_COST_PROJECTION_PATH,
    skip_task1: bool = False,
    skip_task2: bool = False,
    task1_result: dict | None = None,
    pilot_usage_df: pd.DataFrame | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run the Task 3 updated cost projection, optionally chaining Task 1 -> 2 -> 3.

    Args:
        output_path: Where to write cost_projection_updated.csv.
        skip_task1: If True, task1_result must be supplied (reuse prior output).
        skip_task2: If True, pilot_usage_df must be supplied (reuse prior output).
        task1_result: Pre-computed Task 1 result (required if skip_task1).
        pilot_usage_df: Pre-computed Task 2 output (required if skip_task2).
        dry_run: If True, run Task 1/2 in dry-run mode and skip file writes.

    Returns:
        DataFrame from compute_updated_projection.
    """
    if skip_task1:
        task1_result = task1_result or _load_task1_result_from_disk()
    else:
        task1_result = run_task1(dry_run=dry_run)

    if skip_task2:
        pilot_usage_df = pilot_usage_df if pilot_usage_df is not None else _load_pilot_usage_from_disk()
    else:
        pilot_usage_df = run_task2(dry_run=dry_run)

    projection = compute_updated_projection(task1_result, pilot_usage_df)
    flags = flag_high_reasoning_models(pilot_usage_df)

    print("\nUpdated cost projection (per model, per prompt condition):")
    print(projection.to_string(index=False))

    totals = projection.groupby("model")["total_cost"].sum(min_count=1)
    print("\nProjected total cost per model (900 calls):")
    for model, total in totals.items():
        print(f"  {model}: {'$' + format(total, '.2f') if pd.notna(total) else 'N/A (rate unverified)'}")

    print("\nReasoning-token flags:")
    for model, status in flags.items():
        print(f"  {model}: {status}")

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        projection.to_csv(output_path, index=False)
        print(f"\nWrote {output_path}")

    return projection


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Task 3 — updated cost projection")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-task1", action="store_true",
        help="Reuse logs/corpus_token_audit.csv + calibration_token_audit.csv instead of re-running Task 1",
    )
    parser.add_argument(
        "--skip-task2", action="store_true",
        help="Reuse logs/pilot/pilot_usage_log.csv instead of re-running Task 2",
    )
    args = parser.parse_args()

    run_task3(skip_task1=args.skip_task1, skip_task2=args.skip_task2, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
