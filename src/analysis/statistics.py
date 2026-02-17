"""
Statistical tests (spec Section 3.3; Appendix H.2).

Corrections over the appendix:

- stats.binom_test is deprecated in scipy ≥ 1.7; replaced with
  stats.binomtest(...).pvalue throughout.
- Added overall McNemar's test collapsing across all models (spec calls
  for a single test, not only per-model breakdowns).
- ARCHITECTURE_MAP imported from config.py (was duplicated in H.2).
- JSON serializer extended to handle np.bool_ (was missing, would TypeError).
- All output paths use pathlib from config rather than hardcoded 'results/'.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ALPHA,
    ARCHITECTURE_MAP,
    MCNEMAR_SMALL_N,
    MODEL_ORDER,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Test 1: McNemar's test — prompt condition effects (spec Section 3.3)
# ---------------------------------------------------------------------------

def _mcnemar_pair(
    paired: pd.DataFrame,
    label: str,
) -> dict:
    """
    Compute McNemar's test statistics for a paired zero-shot / few-shot table.

    Args:
        paired: DataFrame with columns 'zero_shot' and 'few_shot',
                both containing 'pass' or 'fail'.
        label: Identifier string for this comparison (model name or 'overall').

    Returns:
        Dict with contingency table, test statistic, p-value, odds ratio,
        and directional interpretation.
    """
    a = int(((paired["zero_shot"] == "pass") & (paired["few_shot"] == "pass")).sum())
    b = int(((paired["zero_shot"] == "pass") & (paired["few_shot"] == "fail")).sum())
    c = int(((paired["zero_shot"] == "fail") & (paired["few_shot"] == "pass")).sum())
    d = int(((paired["zero_shot"] == "fail") & (paired["few_shot"] == "fail")).sum())
    n_total = a + b + c + d
    n_discordant = b + c

    base = {
        "contingency_table": {
            "both_pass": a, "zero_only": b,
            "few_only": c, "both_fail": d,
        },
        "n_paired": n_total,
        "n_discordant": n_discordant,
        "zero_shot_pass_rate": round((a + b) / n_total, 4) if n_total else None,
        "few_shot_pass_rate": round((a + c) / n_total, 4) if n_total else None,
    }

    if n_discordant == 0:
        return {
            **base,
            "test_type": None,
            "statistic": None,
            "p_value": None,
            "significant_at_05": None,
            "odds_ratio": None,
            "odds_ratio_ci": (None, None),
            "direction": "no_difference",
            "note": "No discordant pairs — test not applicable",
        }

    if n_discordant < MCNEMAR_SMALL_N:
        # Exact binomial (two-tailed) — scipy ≥ 1.7 API
        p_value = stats.binomtest(b, n_discordant, 0.5).pvalue
        test_stat = None
        test_type = "exact_binomial"
    else:
        # Chi-squared with continuity correction
        test_stat = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - stats.chi2.cdf(test_stat, df=1))
        test_type = "chi_squared_corrected"

    # Odds ratio and 95% CI (log method)
    odds_ratio = b / c if c > 0 else float("inf")
    if b > 0 and c > 0:
        log_or = np.log(odds_ratio)
        se_log = np.sqrt(1 / b + 1 / c)
        ci_lo = float(np.exp(log_or - 1.96 * se_log))
        ci_hi = float(np.exp(log_or + 1.96 * se_log))
    else:
        ci_lo, ci_hi = None, None

    direction = (
        "few_shot_better" if c > b
        else ("zero_shot_better" if b > c else "no_difference")
    )

    return {
        **base,
        "test_type": test_type,
        "statistic": test_stat,
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < ALPHA,
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else None,
        "odds_ratio_ci": (
            round(ci_lo, 4) if ci_lo is not None else None,
            round(ci_hi, 4) if ci_hi is not None else None,
        ),
        "direction": direction,
    }


def mcnemars_test_prompt_effects(fragment_df: pd.DataFrame) -> dict:
    """
    McNemar's test for the prompt condition effect (spec Test 1).

    Runs two complementary analyses:

    1. **Overall** (spec-defined test): pairs each (fragment_id, model_family)
       combination's zero-shot vs. few-shot outcome, yielding n=900 pairs
       (150 fragments × 6 models). This is the single test called for in
       Section 3.3 H₀: no difference in pass rates between conditions.

    2. **Per-model breakdowns**: each model's 150 paired fragments separately,
       providing diagnostic granularity beyond the spec minimum.

    The appendix provided per-model results only; the overall test is added here.

    Args:
        fragment_df: Fragment-level outcomes with columns fragment_id,
                     model_family, prompt_condition, fragment_outcome.

    Returns:
        Dict with keys 'overall' and one key per model family.
    """
    results: dict = {}

    # --- Overall test (spec-defined) ---
    zero_df = (
        fragment_df[fragment_df["prompt_condition"] == "zero_shot"]
        [["fragment_id", "model_family", "fragment_outcome"]]
        .rename(columns={"fragment_outcome": "zero_shot"})
        .set_index(["fragment_id", "model_family"])
    )
    few_df = (
        fragment_df[fragment_df["prompt_condition"] == "few_shot"]
        [["fragment_id", "model_family", "fragment_outcome"]]
        .rename(columns={"fragment_outcome": "few_shot"})
        .set_index(["fragment_id", "model_family"])
    )
    overall_paired = zero_df.join(few_df, how="inner").reset_index()
    results["overall"] = _mcnemar_pair(overall_paired, "overall")

    # --- Per-model breakdowns ---
    for model in MODEL_ORDER:
        model_data = fragment_df[fragment_df["model_family"] == model]
        zero = (
            model_data[model_data["prompt_condition"] == "zero_shot"]
            [["fragment_id", "fragment_outcome"]]
            .rename(columns={"fragment_outcome": "zero_shot"})
            .set_index("fragment_id")
        )
        few = (
            model_data[model_data["prompt_condition"] == "few_shot"]
            [["fragment_id", "fragment_outcome"]]
            .rename(columns={"fragment_outcome": "few_shot"})
            .set_index("fragment_id")
        )
        paired = zero.join(few, how="inner").reset_index()
        if paired.empty:
            continue
        results[model] = _mcnemar_pair(paired, model)

    return results


# ---------------------------------------------------------------------------
# Test 2: Chi-squared — model family differences (spec Section 3.3)
# ---------------------------------------------------------------------------

def chi_squared_model_comparison(fragment_df: pd.DataFrame) -> dict:
    """
    Chi-squared test of independence across 6 model families (spec Test 2).

    Omnibus test assesses whether pass rates differ significantly across models.
    If significant, pairwise comparisons with Bonferroni correction are run.

    Args:
        fragment_df: Fragment-level outcomes DataFrame.

    Returns:
        Dict with 'omnibus' and 'pairwise' sub-dicts.
    """
    contingency = pd.crosstab(
        fragment_df["model_family"],
        fragment_df["fragment_outcome"],
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    n_total = int(contingency.to_numpy().sum())
    k = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n_total * k))) if k > 0 else 0.0

    omnibus = {
        "chi_squared": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "n_total": n_total,
        "significant_at_05": bool(p_value < ALPHA),
        "cramers_v": round(cramers_v, 4),
        "contingency_table": contingency.to_dict(),
        "expected_frequencies": pd.DataFrame(
            expected, index=contingency.index, columns=contingency.columns
        ).to_dict(),
    }

    pairwise: dict = {}
    if p_value < ALPHA:
        model_list = [m for m in MODEL_ORDER if m in fragment_df["model_family"].unique()]
        n_comparisons = len(list(combinations(model_list, 2)))
        bonferroni_alpha = round(ALPHA / n_comparisons, 6)

        for model_a, model_b in combinations(model_list, 2):
            sub = fragment_df[fragment_df["model_family"].isin([model_a, model_b])]
            pair_ct = pd.crosstab(sub["model_family"], sub["fragment_outcome"])
            chi2_p, p_p, _, _ = stats.chi2_contingency(pair_ct)

            rate_a = float(
                (fragment_df[fragment_df["model_family"] == model_a]["fragment_outcome"] == "pass").mean()
            )
            rate_b = float(
                (fragment_df[fragment_df["model_family"] == model_b]["fragment_outcome"] == "pass").mean()
            )
            pairwise[f"{model_a}_vs_{model_b}"] = {
                "chi_squared": round(float(chi2_p), 4),
                "p_value": round(float(p_p), 6),
                "bonferroni_alpha": bonferroni_alpha,
                "significant_bonferroni": bool(p_p < bonferroni_alpha),
                "pass_rate_a": round(rate_a, 4),
                "pass_rate_b": round(rate_b, 4),
                "difference": round(abs(rate_a - rate_b), 4),
                "higher_performer": model_a if rate_a > rate_b else model_b,
            }

    return {"omnibus": omnibus, "pairwise": pairwise}


# ---------------------------------------------------------------------------
# Test 3: Z-test — architecture comparison (spec Section 3.3)
# ---------------------------------------------------------------------------

def architecture_z_test(fragment_df: pd.DataFrame) -> dict:
    """
    Independent proportions z-test: open vs. closed architecture (spec Test 3).

    Uses ARCHITECTURE_MAP from config.py (was duplicated in the appendix).

    Args:
        fragment_df: Fragment-level outcomes DataFrame.

    Returns:
        Dict with z-statistic, p-value, Cohen's h, and interpretation.
    """
    df = fragment_df.copy()
    df["architecture"] = df["model_family"].map(ARCHITECTURE_MAP)

    closed = df[df["architecture"] == "closed"]
    open_m = df[df["architecture"] == "open"]

    n_closed, n_open = len(closed), len(open_m)
    p_closed = float((closed["fragment_outcome"] == "pass").mean())
    p_open   = float((open_m["fragment_outcome"] == "pass").mean())

    n_pass_closed = int((closed["fragment_outcome"] == "pass").sum())
    n_pass_open   = int((open_m["fragment_outcome"] == "pass").sum())
    p_pooled = (n_pass_closed + n_pass_open) / (n_closed + n_open)

    se = float(np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_closed + 1 / n_open)))
    z_stat = float((p_closed - p_open) / se) if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Cohen's h = 2·arcsin(√p₁) − 2·arcsin(√p₂)
    cohens_h = float(2 * np.arcsin(np.sqrt(p_closed)) - 2 * np.arcsin(np.sqrt(p_open)))
    h_abs = abs(cohens_h)
    h_label = "small" if h_abs < 0.2 else ("medium" if h_abs < 0.5 else "large")

    return {
        "closed_pass_rate": round(p_closed, 4),
        "open_pass_rate": round(p_open, 4),
        "difference": round(p_closed - p_open, 4),
        "n_closed": n_closed,
        "n_open": n_open,
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": bool(p_value < ALPHA),
        "cohens_h": round(cohens_h, 4),
        "cohens_h_magnitude": h_label,
        "higher_performer": "closed" if p_closed > p_open else "open",
    }


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_statistical_tests(
    fragment_df: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Execute all three statistical tests and export results (spec Section 3.3).

    Args:
        fragment_df: Fragment-level outcomes DataFrame.
        output_dir: Directory for output files.

    Returns:
        Dict with test_1_mcnemar, test_2_chi_squared, test_3_z_test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "=" * 70

    print(f"\n{sep}")
    print("STATISTICAL ANALYSIS — APPENDIX H.2")
    print(f"{sep}")

    # Test 1
    print("\n--- Test 1: McNemar's Test (Prompt Condition Effects) ---")
    mcnemar = mcnemars_test_prompt_effects(fragment_df)
    overall = mcnemar["overall"]
    print(f"  OVERALL: p={overall.get('p_value')}, "
          f"significant={overall.get('significant_at_05')}, "
          f"direction={overall.get('direction')}")
    for model, res in mcnemar.items():
        if model == "overall":
            continue
        print(f"  {model}: p={res.get('p_value')}, "
              f"sig={res.get('significant_at_05')}, "
              f"direction={res.get('direction')}")

    # Test 2
    print("\n--- Test 2: Chi-Squared (Model Family Differences) ---")
    chi2_res = chi_squared_model_comparison(fragment_df)
    omni = chi2_res["omnibus"]
    print(f"  Omnibus: χ²={omni['chi_squared']:.3f}, "
          f"p={omni['p_value']:.4f}, Cramér's V={omni['cramers_v']:.3f}")
    if omni["significant_at_05"] and chi2_res["pairwise"]:
        sig_pairs = [
            k for k, v in chi2_res["pairwise"].items() if v["significant_bonferroni"]
        ]
        print(f"  Bonferroni-significant pairs ({len(sig_pairs)}): "
              f"{', '.join(sig_pairs) or 'none'}")

    # Test 3
    print("\n--- Test 3: Z-Test (Open vs. Closed Architecture) ---")
    z_res = architecture_z_test(fragment_df)
    print(f"  Closed: {z_res['closed_pass_rate']:.1%}, "
          f"Open: {z_res['open_pass_rate']:.1%}")
    print(f"  z={z_res['z_statistic']:.3f}, p={z_res['p_value']:.4f}, "
          f"Cohen's h={z_res['cohens_h']:.3f} ({z_res['cohens_h_magnitude']})")

    all_results = {
        "test_1_mcnemar": mcnemar,
        "test_2_chi_squared": chi2_res,
        "test_3_z_test": z_res,
    }

    # JSON export — handle numpy types including np.bool_
    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    out_path = output_dir / "H2_statistical_tests.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=_json_default)

    print(f"\nExported H.2 results to {out_path}")
    return all_results
