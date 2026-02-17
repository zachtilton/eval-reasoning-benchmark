"""
Performance matrix generation and summary tables (spec Section 3.2; Appendix H.1).

Corrections over the appendix:
- generate_performance_matrix now iterates in the canonical MODEL_ORDER ×
  PROMPT_ORDER sequence rather than unique() — prevents nondeterministic output.
- architecture_map moved to config.py; no longer duplicated here and in H.2.
- All output paths use pathlib from config rather than hardcoded 'results/'.
- wilson_confidence_interval handles n=0 defensively (was already there)
  and now clamps output to [0, 1] explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ARCHITECTURE_MAP,
    MODEL_ORDER,
    PROMPT_ORDER,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Wilson confidence interval (shared utility)
# ---------------------------------------------------------------------------

def wilson_confidence_interval(
    p: float,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    More accurate than the Wald interval for small samples or extreme
    proportions (spec specifies 95% CIs throughout).

    Args:
        p: Observed pass rate (proportion of successes).
        n: Sample size (number of fragments or runs).
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound), both clamped to [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom

    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# H.1: 12-cell performance matrix
# ---------------------------------------------------------------------------

def generate_performance_matrix(fragment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the 12-cell performance matrix (6 models × 2 prompts) with
    pass rates and 95% Wilson score confidence intervals (spec Section 3.2).

    Iterates in canonical MODEL_ORDER × PROMPT_ORDER so the output is
    deterministic regardless of DataFrame sort order.

    Args:
        fragment_df: Fragment-level outcomes DataFrame.
            Required columns: model_family, prompt_condition,
            fragment_outcome, unanimous_agreement.

    Returns:
        DataFrame with one row per model-prompt combination (≤ 12 rows).
    """
    records: list[dict] = []

    for model in MODEL_ORDER:
        for prompt in PROMPT_ORDER:
            subset = fragment_df[
                (fragment_df["model_family"] == model)
                & (fragment_df["prompt_condition"] == prompt)
            ]
            if subset.empty:
                continue

            n = len(subset)
            passes = int((subset["fragment_outcome"] == "pass").sum())
            fails = n - passes
            pass_rate = passes / n
            unanimous_n = int(subset["unanimous_agreement"].sum())
            unanimous_rate = unanimous_n / n
            ci_lo, ci_hi = wilson_confidence_interval(pass_rate, n)

            records.append({
                "model_family": model,
                "prompt_condition": prompt,
                "n_fragments": n,
                "passes": passes,
                "fails": fails,
                "fragment_pass_rate": round(pass_rate, 4),
                "ci_lower_95": round(ci_lo, 4),
                "ci_upper_95": round(ci_hi, 4),
                "ci_width": round(ci_hi - ci_lo, 4),
                "unanimous_agreement_rate": round(unanimous_rate, 4),
                "unanimous_count": unanimous_n,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# H.1: Collapsed summaries (by model, by prompt, by architecture)
# ---------------------------------------------------------------------------

def generate_collapsed_summaries(
    fragment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate summary tables collapsed by model, by prompt, and by architecture.

    Uses ARCHITECTURE_MAP from config rather than a locally defined dict
    (removes the duplication that existed between H.1 and H.2 in the appendix).

    Args:
        fragment_df: Fragment-level outcomes DataFrame.

    Returns:
        Tuple of (by_model_df, by_prompt_df, by_architecture_df), each sorted
        by fragment_pass_rate descending.
    """
    # --- By model (collapsed across prompts) ---
    by_model_records: list[dict] = []
    for model in MODEL_ORDER:
        subset = fragment_df[fragment_df["model_family"] == model]
        if subset.empty:
            continue
        n = len(subset)
        pass_rate = float((subset["fragment_outcome"] == "pass").mean())
        ci_lo, ci_hi = wilson_confidence_interval(pass_rate, n)
        by_model_records.append({
            "model_family": model,
            "architecture": ARCHITECTURE_MAP.get(model, "unknown"),
            "n_combinations": n,
            "fragment_pass_rate": round(pass_rate, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
            "unanimous_agreement_rate": round(
                float(subset["unanimous_agreement"].mean()), 4
            ),
        })
    by_model_df = pd.DataFrame(by_model_records).sort_values(
        "fragment_pass_rate", ascending=False
    )

    # --- By prompt (collapsed across models) ---
    by_prompt_records: list[dict] = []
    for prompt in PROMPT_ORDER:
        subset = fragment_df[fragment_df["prompt_condition"] == prompt]
        if subset.empty:
            continue
        n = len(subset)
        pass_rate = float((subset["fragment_outcome"] == "pass").mean())
        ci_lo, ci_hi = wilson_confidence_interval(pass_rate, n)
        by_prompt_records.append({
            "prompt_condition": prompt,
            "n_combinations": n,
            "fragment_pass_rate": round(pass_rate, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
            "unanimous_agreement_rate": round(
                float(subset["unanimous_agreement"].mean()), 4
            ),
        })
    by_prompt_df = pd.DataFrame(by_prompt_records).sort_values(
        "fragment_pass_rate", ascending=False
    )

    # --- By architecture (open vs. closed) ---
    df_arch = fragment_df.copy()
    df_arch["architecture"] = df_arch["model_family"].map(ARCHITECTURE_MAP)
    by_arch_records: list[dict] = []
    for arch in ["closed", "open"]:
        subset = df_arch[df_arch["architecture"] == arch]
        if subset.empty:
            continue
        n = len(subset)
        pass_rate = float((subset["fragment_outcome"] == "pass").mean())
        ci_lo, ci_hi = wilson_confidence_interval(pass_rate, n)
        by_arch_records.append({
            "architecture": arch,
            "n_models": int(subset["model_family"].nunique()),
            "n_combinations": n,
            "fragment_pass_rate": round(pass_rate, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
        })
    by_arch_df = pd.DataFrame(by_arch_records).sort_values(
        "fragment_pass_rate", ascending=False
    )

    return by_model_df, by_prompt_df, by_arch_df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_performance_tables(
    fragment_df: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and export all H.1 performance tables to CSV.

    Args:
        fragment_df: Fragment-level outcomes DataFrame.
        output_dir: Directory for output files.

    Returns:
        Tuple of (matrix_df, by_model_df, by_prompt_df, by_arch_df).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = generate_performance_matrix(fragment_df)
    matrix.to_csv(output_dir / "H1_performance_matrix.csv", index=False)

    by_model, by_prompt, by_arch = generate_collapsed_summaries(fragment_df)
    by_model.to_csv(output_dir / "H1_by_model.csv", index=False)
    by_prompt.to_csv(output_dir / "H1_by_prompt.csv", index=False)
    by_arch.to_csv(output_dir / "H1_by_architecture.csv", index=False)

    print(f"H.1 tables exported to {output_dir}")
    return matrix, by_model, by_prompt, by_arch
