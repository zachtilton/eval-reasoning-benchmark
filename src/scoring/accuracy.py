"""
Classification accuracy comparison, run-level outcome assignment,
fragment-level adjudication, and performance metrics (spec Section 2.4,
Steps 2–4; Appendix G.5).

Key corrections over the appendix code:

- load_gold_standard now globs the date-stamped filename
  (gold_standard_locked_YYYY-MM-DD.csv) instead of hardcoding a path.
- All row['coherence'] references updated to row['coherence_final'].
- assign_run_outcomes now assigns the failure_type column
  (type_1/type_2/type_3) required by the scored_database schema and
  needed for failure mode coding eligibility in Section 3.1.
- adjudicate_fragment_outcomes saves output to fragment_outcomes.csv.
- calculate_performance_metrics excludes infrastructure-excluded rows
  from run-level calculations.
- identify_failures_for_coding filters to Type 1 failures only (coherent
  but incorrect) per spec Section 3.1; documents Type 2/3 quantitatively.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from .config import (
    FRAGMENT_OUTCOMES_PATH,
    GOLD_STANDARD_DIR,
    PRIMARY_MODEL,
    RESPONSES_DIR,
)


# ---------------------------------------------------------------------------
# Gold standard
# ---------------------------------------------------------------------------

def load_gold_standard(gold_dir: Path = GOLD_STANDARD_DIR) -> pd.DataFrame:
    """
    Load the locked gold standard expert judgments.

    Globs for the date-stamped file matching the spec naming convention
    ``gold_standard_locked_YYYY-MM-DD.csv``.  Raises if the file is absent
    (it must be locked and present before scoring begins).

    Args:
        gold_dir: Directory containing the gold standard file.

    Returns:
        DataFrame with fragment_id and expert_classification columns.

    Raises:
        FileNotFoundError: No gold standard file found in the directory.
        ValueError: Multiple gold standard files found (ambiguous which to use).
    """
    candidates = sorted(gold_dir.glob("gold_standard_locked_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No gold standard file found in {gold_dir}.\n"
            "Expected: gold_standard_locked_YYYY-MM-DD.csv"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple gold standard files found in {gold_dir}:\n"
            + "\n".join(str(p) for p in candidates)
            + "\nRemove all but the current locked file."
        )

    gold_path = candidates[0]
    gold_df = pd.read_csv(gold_path)

    sound_n = (gold_df["expert_classification"] == "sound").sum()
    not_sound_n = (gold_df["expert_classification"] == "not_sound").sum()

    print(f"Gold standard loaded: {gold_path.name}")
    print(f"  Fragments: {len(gold_df)}")
    print(f"  Sound:     {sound_n}")
    print(f"  Not sound: {not_sound_n}")

    return gold_df


# ---------------------------------------------------------------------------
# Step 2: Classification accuracy (coherent responses only)
# ---------------------------------------------------------------------------

def compare_to_gold_standard(
    database_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare model classifications to the gold standard (Step 2).

    Only applies to coherent responses.  Incoherent and ambiguous responses
    bypass this step — they already fail regardless of classification match
    (spec Section 2.4, Step 2).

    Args:
        database_df: Scored database with coherence_final populated.
        gold_df: Gold standard DataFrame from :func:`load_gold_standard`.

    Returns:
        DataFrame with classification_accuracy column added.
    """
    merged = database_df.merge(
        gold_df[["fragment_id", "expert_classification"]],
        on="fragment_id",
        how="left",
    )

    def _accuracy(row: pd.Series) -> str | None:
        if row["coherence_final"] != "coherent":
            return None  # bypass — will fail regardless
        if row["classification_output"] == row["expert_classification"]:
            return "correct"
        return "incorrect"

    merged["classification_accuracy"] = merged.apply(_accuracy, axis=1)
    return merged


# ---------------------------------------------------------------------------
# Step 3: Run-level outcome assignment
# ---------------------------------------------------------------------------

def assign_run_outcomes(database_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Pass/Fail and failure type at the run level (Step 3).

    Decision rule (spec Section 2.4, Step 3 + G.6.1):
      Pass = coherent AND correct
      Fail = incoherent OR incorrect (or both)

    Also assigns failure_type per spec Section 3.1 — required to identify
    which failures are eligible for failure mode coding:
      type_1: coherent but incorrect  → eligible for coding
      type_2: incoherent but correct  → documented quantitatively only
      type_3: incoherent and incorrect → documented quantitatively only
      None:   pass — no failure

    The appendix did not assign failure_type at all; it is added here.

    Args:
        database_df: Scored database with coherence_final and
                     classification_accuracy populated.

    Returns:
        DataFrame with run_outcome and failure_type columns added.
    """
    outcomes: list[str] = []
    failure_types: list[str | None] = []

    for _, row in database_df.iterrows():
        coherence = row["coherence_final"]
        accuracy = row.get("classification_accuracy")

        coherent = coherence == "coherent"
        correct = accuracy == "correct"

        if coherent and correct:
            outcomes.append("pass")
            failure_types.append(None)
        elif coherent and not correct:
            outcomes.append("fail")
            failure_types.append("type_1")  # coherent but wrong — codeable
        elif not coherent and correct:
            outcomes.append("fail")
            failure_types.append("type_2")  # incoherent but accidentally correct
        else:
            outcomes.append("fail")
            failure_types.append("type_3")  # incoherent and wrong

    database_df = database_df.copy()
    database_df["run_outcome"] = outcomes
    database_df["failure_type"] = failure_types

    pass_n = outcomes.count("pass")
    fail_n = outcomes.count("fail")
    total = len(outcomes)

    t1 = failure_types.count("type_1")
    t2 = failure_types.count("type_2")
    t3 = failure_types.count("type_3")

    print(f"\nRun-Level Outcomes ({total:,} runs):")
    print(f"  Pass:   {pass_n:,} ({pass_n/total:.1%})")
    print(f"  Fail:   {fail_n:,} ({fail_n/total:.1%})")
    print(f"  Breakdown: Type 1={t1}, Type 2={t2}, Type 3={t3}")

    return database_df


# ---------------------------------------------------------------------------
# Step 4: Fragment-level adjudication
# ---------------------------------------------------------------------------

def adjudicate_fragment_outcomes(
    database_df: pd.DataFrame,
    output_path: Path = FRAGMENT_OUTCOMES_PATH,
) -> pd.DataFrame:
    """
    Apply majority rule across 3 runs for fragment-level outcomes (Step 4).

    A fragment-model-prompt combination passes if ≥ 2 of 3 runs pass.
    Unanimous agreement (0/3 or 3/3) is tracked separately (spec Section 2.4).

    Saves result to fragment_outcomes.csv per the data schema.

    Args:
        database_df: Scored database with run_outcome populated.
        output_path: Path to save fragment-level outcomes CSV.

    Returns:
        DataFrame of fragment-level outcomes (≈ 1,800 rows).
    """
    groups = database_df.groupby(["fragment_id", "model_family", "prompt_condition"])

    records: list[dict] = []
    for (frag_id, model, prompt), group in groups:
        n_runs = len(group)
        if n_runs != 3:
            print(
                f"WARNING: {frag_id}/{model}/{prompt} has {n_runs} runs "
                f"(expected 3)"
            )

        pass_count = int((group["run_outcome"] == "pass").sum())
        fail_count = int((group["run_outcome"] == "fail").sum())
        fragment_outcome = "pass" if pass_count >= 2 else "fail"
        unanimous = (pass_count == 3) or (fail_count == 3)

        records.append({
            "fragment_id": frag_id,
            "model_family": model,
            "prompt_condition": prompt,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "fragment_outcome": fragment_outcome,
            "unanimous_agreement": unanimous,
        })

    fragment_df = pd.DataFrame(records)

    # Save to data/responses/fragment_outcomes.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_df.to_csv(output_path, index=False)

    total = len(fragment_df)
    pass_n = (fragment_df["fragment_outcome"] == "pass").sum()
    fail_n = (fragment_df["fragment_outcome"] == "fail").sum()
    unanimous_n = fragment_df["unanimous_agreement"].sum()

    print(f"\nFragment-Level Outcomes ({total} combinations):")
    print(f"  Pass:               {pass_n} ({pass_n/total:.1%})")
    print(f"  Fail:               {fail_n} ({fail_n/total:.1%})")
    print(f"  Unanimous agreement:{unanimous_n} ({unanimous_n/total:.1%})")
    print(f"  Saved to:           {output_path}")

    return fragment_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _wilson_ci(
    pass_rate: float,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        pass_rate: Observed proportion of successes.
        n: Sample size.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    center = (pass_rate + z**2 / (2 * n)) / denom
    margin = z * ((pass_rate * (1 - pass_rate) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (center - margin, center + margin)


def calculate_performance_metrics(
    fragment_df: pd.DataFrame,
    database_df: pd.DataFrame,
) -> dict:
    """
    Calculate primary, secondary, and tertiary performance metrics.

    Primary:   fragment-level pass rate (% of 1,800 combinations passing)
    Secondary: unanimous agreement rate (% where all 3 runs pass)
    Tertiary:  run-level pass rate (% of 5,400 runs passing)

    All metrics stratified by: model family, prompt condition, interaction.
    Run-level metrics exclude rows where exclude_from_analysis == True
    (infrastructure failures; spec Section 2.4).

    Args:
        fragment_df: Fragment-level outcomes from :func:`adjudicate_fragment_outcomes`.
        database_df: Run-level scored database.

    Returns:
        Nested dict with 'overall', 'by_model', 'by_prompt', 'by_interaction'.
    """
    # Exclude infrastructure failures from run-level calculations
    run_df = (
        database_df[database_df["exclude_from_analysis"] != True]  # noqa: E712
        if "exclude_from_analysis" in database_df.columns
        else database_df
    )

    metrics: dict = {}

    metrics["overall"] = {
        "fragment_pass_rate": float((fragment_df["fragment_outcome"] == "pass").mean()),
        "unanimous_agreement_rate": float(fragment_df["unanimous_agreement"].mean()),
        "run_pass_rate": float((run_df["run_outcome"] == "pass").mean()),
    }

    metrics["by_model"] = {}
    for model in sorted(fragment_df["model_family"].unique()):
        m_frag = fragment_df[fragment_df["model_family"] == model]
        metrics["by_model"][model] = {
            "fragment_pass_rate": float((m_frag["fragment_outcome"] == "pass").mean()),
            "unanimous_agreement_rate": float(m_frag["unanimous_agreement"].mean()),
            "n_combinations": len(m_frag),
        }

    metrics["by_prompt"] = {}
    for prompt in sorted(fragment_df["prompt_condition"].unique()):
        p_frag = fragment_df[fragment_df["prompt_condition"] == prompt]
        metrics["by_prompt"][prompt] = {
            "fragment_pass_rate": float((p_frag["fragment_outcome"] == "pass").mean()),
            "unanimous_agreement_rate": float(p_frag["unanimous_agreement"].mean()),
            "n_combinations": len(p_frag),
        }

    metrics["by_interaction"] = {}
    for model in sorted(fragment_df["model_family"].unique()):
        metrics["by_interaction"][model] = {}
        for prompt in sorted(fragment_df["prompt_condition"].unique()):
            sub = fragment_df[
                (fragment_df["model_family"] == model)
                & (fragment_df["prompt_condition"] == prompt)
            ]
            metrics["by_interaction"][model][prompt] = {
                "fragment_pass_rate": float((sub["fragment_outcome"] == "pass").mean()),
                "unanimous_agreement_rate": float(sub["unanimous_agreement"].mean()),
                "n_combinations": len(sub),
            }

    return metrics


def print_performance_summary(metrics: dict) -> None:
    """
    Print a formatted performance metrics summary to stdout.

    Args:
        metrics: Dict from :func:`calculate_performance_metrics`.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("PERFORMANCE METRICS SUMMARY")
    print(sep)

    o = metrics["overall"]
    print(f"\nOVERALL:")
    print(f"  Fragment-level pass rate: {o['fragment_pass_rate']:.1%}")
    print(f"  Unanimous agreement rate: {o['unanimous_agreement_rate']:.1%}")
    print(f"  Run-level pass rate:      {o['run_pass_rate']:.1%}")

    print(f"\nBY MODEL FAMILY:")
    for model, m in metrics["by_model"].items():
        print(f"  {model}:")
        print(f"    Fragment pass rate: {m['fragment_pass_rate']:.1%}")
        print(f"    Unanimous:          {m['unanimous_agreement_rate']:.1%}")

    print(f"\nBY PROMPT CONDITION:")
    for prompt, p in metrics["by_prompt"].items():
        print(f"  {prompt}:")
        print(f"    Fragment pass rate: {p['fragment_pass_rate']:.1%}")
        print(f"    Unanimous:          {p['unanimous_agreement_rate']:.1%}")

    print(f"\nMODEL × PROMPT INTERACTION (Fragment Pass Rates):")
    print(f"{'Model':<22} {'Zero-Shot':>12} {'Few-Shot':>12}")
    print("-" * 48)
    for model in sorted(metrics["by_interaction"]):
        zs = metrics["by_interaction"][model].get("zero_shot", {}).get(
            "fragment_pass_rate", float("nan")
        )
        fs = metrics["by_interaction"][model].get("few_shot", {}).get(
            "fragment_pass_rate", float("nan")
        )
        print(f"  {model:<20} {zs:>11.1%} {fs:>11.1%}")


def export_performance_matrix(
    fragment_df: pd.DataFrame,
    output_path: Path = RESPONSES_DIR / "performance_matrix.csv",
) -> pd.DataFrame:
    """
    Export the 12-combination performance matrix with 95% Wilson CIs.

    Covers all 6 models × 2 prompts, matching Appendix H.1 requirements.

    Args:
        fragment_df: Fragment-level outcomes from :func:`adjudicate_fragment_outcomes`.
        output_path: Path for the output CSV.

    Returns:
        Performance matrix DataFrame.
    """
    matrix = (
        fragment_df.groupby(["model_family", "prompt_condition"])
        .agg(
            fragment_pass_rate=("fragment_outcome", lambda x: (x == "pass").mean()),
            unanimous_agreement_rate=("unanimous_agreement", "mean"),
            n_fragments=("fragment_id", "count"),
        )
        .reset_index()
    )

    ci_lower, ci_upper = zip(
        *matrix.apply(
            lambda row: _wilson_ci(row["fragment_pass_rate"], row["n_fragments"]),
            axis=1,
        )
    )
    matrix["ci_lower_95"] = ci_lower
    matrix["ci_upper_95"] = ci_upper

    matrix = matrix.sort_values(["model_family", "prompt_condition"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)
    print(f"\nPerformance matrix ({len(matrix)} combinations) exported to {output_path}")

    return matrix


# ---------------------------------------------------------------------------
# Failure identification for coding (GPT 5.2, Type 1 only)
# ---------------------------------------------------------------------------

def identify_failures_for_coding(
    fragment_df: pd.DataFrame,
    database_df: pd.DataFrame,
    primary_model: str = PRIMARY_MODEL,
    output_path: Path = RESPONSES_DIR / f"{PRIMARY_MODEL}_failures_for_coding.csv",
) -> pd.DataFrame:
    """
    Identify primary-model fragment-level failures eligible for failure mode coding.

    Per spec Section 3.1:
    - Only Type 1 failures (coherent but incorrect) are eligible for coding.
    - Type 2 (incoherent but correct) and Type 3 (incoherent and incorrect)
      are documented quantitatively only.

    For each Type 1 eligible fragment-level failure the function identifies
    the representative failed run: a Type 1 run is preferred; if none exists
    among the failed runs, the first failed run is used.

    The appendix filtered only by fragment_outcome == 'fail' without checking
    failure type.  This is corrected here.

    Args:
        fragment_df: Fragment-level outcomes from :func:`adjudicate_fragment_outcomes`.
        database_df: Run-level scored database with failure_type populated.
        primary_model: Model identifier receiving full diagnostic treatment.
        output_path: Path for the coding export CSV.

    Returns:
        DataFrame of Type 1 eligible fragment-level failures with
        representative_run column added.
    """
    # Fragment-level failures for the primary model
    primary_failures = fragment_df[
        (fragment_df["model_family"] == primary_model)
        & (fragment_df["fragment_outcome"] == "fail")
    ].copy()

    if primary_failures.empty:
        print(f"\n{primary_model}: No fragment-level failures found.")
        return primary_failures

    # Determine failure type at the fragment level and identify representative run
    run_subset = database_df[
        (database_df["model_family"] == primary_model)
        & (database_df["run_outcome"] == "fail")
    ]

    representative_runs: list[int | None] = []
    fragment_failure_types: list[str | None] = []

    for _, frag_row in primary_failures.iterrows():
        runs = run_subset[
            (run_subset["fragment_id"] == frag_row["fragment_id"])
            & (run_subset["prompt_condition"] == frag_row["prompt_condition"])
        ]

        # Prefer Type 1 runs for representative selection
        type1_runs = runs[runs["failure_type"] == "type_1"]
        if not type1_runs.empty:
            rep_run = int(type1_runs.iloc[0]["run_number"])
            frag_type = "type_1"
        elif not runs.empty:
            # Determine dominant failure type across failed runs
            dominant_type = runs["failure_type"].mode()
            frag_type = dominant_type.iloc[0] if not dominant_type.empty else None
            rep_run = int(runs.iloc[0]["run_number"])
        else:
            rep_run = None
            frag_type = None

        representative_runs.append(rep_run)
        fragment_failure_types.append(frag_type)

    primary_failures["representative_run"] = representative_runs
    primary_failures["dominant_failure_type"] = fragment_failure_types

    # Split by eligibility for coding
    eligible = primary_failures[primary_failures["dominant_failure_type"] == "type_1"]
    type2 = primary_failures[primary_failures["dominant_failure_type"] == "type_2"]
    type3 = primary_failures[primary_failures["dominant_failure_type"] == "type_3"]

    total_primary = len(
        fragment_df[
            (fragment_df["model_family"] == primary_model)
        ]
    )
    failure_n = len(primary_failures)

    print(f"\n{primary_model.upper()} Failure Analysis:")
    print(f"  Fragment-model-prompt combinations: {total_primary}")
    print(f"  Fragment-level failures:            {failure_n} ({failure_n/total_primary:.1%})")
    print(f"  Type 1 (eligible for coding):       {len(eligible)}")
    print(f"  Type 2 (incoherent but correct):    {len(type2)}")
    print(f"  Type 3 (incoherent and incorrect):  {len(type3)}")

    for prompt in sorted(eligible["prompt_condition"].unique()):
        n = len(eligible[eligible["prompt_condition"] == prompt])
        print(f"    {prompt}: {n} Type 1 failures")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(output_path, index=False)
    print(f"  Type 1 failures exported to: {output_path}")

    return eligible
