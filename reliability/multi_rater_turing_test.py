"""
Multi-Rater Agreement Statistical Turing Test
==============================================

Implements the equivalence test from:
Kljajic, J., O'Toole, J. M., Hogan, R., & Skoric, T. (2025). Honest and
reliable evaluation and expert equivalence testing of automated neonatal
seizure detection. arXiv:2508.04899.

WHAT THIS IS FOR (your immediate use case)
-------------------------------------------
Right now this is being used to validate your single-coder gold-standard
ANCHOR, not to test an LLM. The "AI/model under test" slot in the original
paper is filled by your anchor codes; the "human expert panel" slot is
filled by your two colleague raters. The question being answered: if the
anchor stands in for one of the two colleagues, does the panel's agreement
hold up as well as it would with an actual second human rater?

The exact same script can be reused later to test an LLM's outputs against
the human panel, by putting the LLM's ratings in the `candidate` column
instead of the anchor's. Nothing else about the procedure changes.

HOW IT WORKS
------------
1. Compute Fleiss' kappa among the human-only raters (here: rater_1 &
   rater_2) on the rated subsample. This is the human-only inter-rater
   agreement (IRA).
2. Bootstrap that human-only IRA (resample fragments with replacement, B
   times) to see how much it naturally wobbles due to sampling alone.
   The "margin" is defined as (mean of the bootstrap distribution) minus
   (its 2.5th percentile) -- i.e., a positive number representing the
   typical downward fluctuation you'd see from resampling noise alone,
   with nothing to do with the candidate/anchor at all.
3. For each human rater, substitute the candidate (anchor or LLM) in their
   place and recompute IRA with the remaining rater(s). This gives
   delta-kappa = kappa_with_candidate_substituted - kappa_human_only for
   that substitution.
4. Bootstrap steps 1 and 3 TOGETHER, using the same resampled fragments for
   both computations each iteration. This is what makes it a paired
   bootstrap rather than two independent estimates being compared -- it
   respects that both numbers are computed from the same underlying data,
   which a simpler two-sample z-test would not.
5. Average delta-kappa across all substitutions in each bootstrap
   iteration (the paper's "Average kappa" variant -- there are other
   variants; see NOTES at the bottom).
6. Decision rule: the candidate is considered statistically equivalent to
   a human rater if the 5th percentile of the delta-kappa bootstrap
   distribution is NOT worse than the margin from step 2 -- i.e., even in
   a pessimistic scenario, substituting the candidate doesn't hurt
   agreement more than ordinary sampling noise would.

A NOTE ON FIDELITY TO THE SOURCE PAPER
---------------------------------------
VERIFIED against the authors' own reference implementation, published
alongside the paper at:
https://github.com/jovanak1/annotation-generation-and-expert-level-testing
(file: expert-level-tests/Multi_Rater_Agreement_Statistical_Turing_Tests.py,
commit 189ee476a9dfad07835180ddd003adab20f0b496).

The paper's own prose ("the 5th percentile of delta-kappa exceeds the
margin") reads as comparing against a positive-valued margin, which is
ambiguous on its own. Their actual code resolves it: the decision rule is
`delta_p5 > -margin` -- i.e. the standard non-inferiority comparison this
script already implements. No correction to the decision logic was needed.

One thing worth knowing if you ever implement the "All raters" / "Majority
raters" / "Any rater" variants (not implemented here -- see NOTES at the
bottom): the authors' own per-annotator code branch compares against
`margin` (positive) rather than `-margin` in two places, which is
inconsistent with their own docstring and with their "Average" branch (the
variant this script matches). That looks like a bug in their reference
implementation specific to the per-annotator branch -- if you build those
variants later, use `-margin`, not the sign convention in that branch.

Their bootstrap also resamples at a "baby" (subject-cluster) level, since
their EEG data has multiple annotated segments nested within each infant.
Your fragments have no such nesting, so plain fragment-level resampling
(what this script does) is the correct analog, not a simplification.

USAGE
-----
Prepare a CSV with one row per rated fragment and these columns:
    fragment_id, anchor, rater_1, rater_2
Each rating column should contain the same category labels (e.g. "pass"/
"fail", or 0/1). Then run:

    python multi_rater_turing_test.py your_ratings.csv --candidate anchor

Swap --candidate to whichever column you're testing (e.g. an LLM's column,
once you have one).
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd


def fleiss_kappa(ratings):
    """
    Fleiss' kappa for a set of subjects rated by a fixed number of raters.

    ratings : 2D array-like, shape (n_subjects, n_raters)
        Categorical labels (any hashable type). Works for n_raters >= 2 and
        any number of categories; reduces to (a close cousin of) Cohen's
        kappa when n_raters == 2.

    Returns kappa (float). Returns np.nan if the computation is degenerate
    (e.g. only one category appears across the whole sample).
    """
    ratings = np.asarray(ratings)
    n_subjects, n_raters = ratings.shape
    categories = np.unique(ratings)
    k = len(categories)
    if k < 2:
        return np.nan

    # n_ij: count of raters assigning subject i to category j
    n_ij = np.zeros((n_subjects, k))
    for j, cat in enumerate(categories):
        n_ij[:, j] = (ratings == cat).sum(axis=1)

    N, n = n_subjects, n_raters
    p_j = n_ij.sum(axis=0) / (N * n)
    P_i = (np.square(n_ij).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    Pe_bar = np.square(p_j).sum()

    if Pe_bar == 1:
        return np.nan
    return (P_bar - Pe_bar) / (1 - Pe_bar)


def bootstrap_indices(n_items, n_boot, rng):
    """Yield n_boot arrays of resampled row indices (sampling with replacement)."""
    for _ in range(n_boot):
        yield rng.integers(0, n_items, size=n_items)


def run_turing_test(
    human_ratings: dict,
    candidate_ratings: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
    variant: str = "average",
):
    """
    human_ratings : dict of {rater_name: 1D array of ratings}, e.g.
        {"rater_1": [...], "rater_2": [...]}. Must all be the same length.
    candidate_ratings : 1D array of ratings from the anchor (or LLM) to test,
        same length and category scheme as the human ratings.
    n_boot : number of bootstrap iterations.
    variant : "average" (paper's variant 1 -- AI must match average human
        substitution effect). "all", "majority", and "any" are noted but
        not implemented here; see NOTES at bottom of file for how to adapt.

    Returns a dict with the human-only IRA, the margin, the delta-kappa
    bootstrap distribution, the test statistic, and the pass/fail decision.
    """
    rng = np.random.default_rng(seed)
    rater_names = list(human_ratings.keys())
    n_items = len(candidate_ratings)
    for name in rater_names:
        assert len(human_ratings[name]) == n_items, (
            f"Rater '{name}' has a different number of ratings than the "
            f"candidate ({len(human_ratings[name])} vs {n_items})."
        )

    human_matrix = np.column_stack([human_ratings[n] for n in rater_names])

    # Point estimates (full data, no resampling) -- for reference/reporting.
    kappa_human_point = fleiss_kappa(human_matrix)
    substitution_deltas_point = {}
    for held_out in rater_names:
        remaining = [r for r in rater_names if r != held_out]
        sub_matrix = np.column_stack(
            [human_ratings[r] for r in remaining] + [candidate_ratings]
        )
        kappa_sub = fleiss_kappa(sub_matrix)
        substitution_deltas_point[held_out] = kappa_sub - kappa_human_point

    # --- Step 1-2: bootstrap the human-only IRA to get the natural-noise margin ---
    human_boot = np.empty(n_boot)
    for b, idx in enumerate(bootstrap_indices(n_items, n_boot, rng)):
        human_boot[b] = fleiss_kappa(human_matrix[idx])
    margin = np.nanmean(human_boot) - np.nanpercentile(human_boot, 2.5)

    # --- Step 3-5: paired bootstrap of delta-kappa (candidate substituted in) ---
    rng2 = np.random.default_rng(seed + 1)  # separate stream, still reproducible
    delta_boot = np.empty(n_boot)
    for b, idx in enumerate(bootstrap_indices(n_items, n_boot, rng2)):
        resampled_human = human_matrix[idx]
        resampled_candidate = candidate_ratings[idx]
        kappa_human_b = fleiss_kappa(resampled_human)

        per_substitution = []
        for held_out_pos, held_out in enumerate(rater_names):
            remaining_cols = [
                resampled_human[:, i]
                for i in range(len(rater_names))
                if i != held_out_pos
            ]
            sub_matrix_b = np.column_stack(remaining_cols + [resampled_candidate])
            per_substitution.append(fleiss_kappa(sub_matrix_b) - kappa_human_b)

        if variant == "average":
            delta_boot[b] = np.nanmean(per_substitution)
        else:
            raise NotImplementedError(
                f"Variant '{variant}' not implemented -- see NOTES at bottom "
                f"of file for 'all' / 'majority' / 'any'."
            )

    # --- Step 6: decision rule ---
    p5_delta = np.nanpercentile(delta_boot, 5)
    passes = bool(p5_delta > -margin)

    return {
        "human_only_kappa_point_estimate": kappa_human_point,
        "substitution_deltas_point_estimate": substitution_deltas_point,
        "margin": margin,
        "delta_kappa_bootstrap_mean": float(np.nanmean(delta_boot)),
        "delta_kappa_5th_percentile": float(p5_delta),
        "delta_kappa_95th_percentile": float(np.nanpercentile(delta_boot, 95)),
        "decision": "EQUIVALENT (passes)" if passes else "NOT EQUIVALENT (fails)",
        "n_boot": n_boot,
        "n_items": n_items,
        "raters_compared": rater_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Rater Agreement Statistical Turing Test")
    parser.add_argument("csv_path", help="CSV with fragment_id + rater columns")
    parser.add_argument(
        "--candidate", required=True,
        help="Name of the column to test (e.g. 'anchor', or an LLM's column name)",
    )
    parser.add_argument(
        "--humans", nargs="*", default=None,
        help="Column names of the human-only comparison panel. Defaults to "
             "every column except fragment_id and --candidate.",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    id_col = "fragment_id" if "fragment_id" in df.columns else df.columns[0]
    other_cols = [c for c in df.columns if c not in (id_col, args.candidate)]
    human_cols = args.humans if args.humans else other_cols

    human_ratings = {c: df[c].to_numpy() for c in human_cols}
    candidate_ratings = df[args.candidate].to_numpy()

    result = run_turing_test(
        human_ratings, candidate_ratings, n_boot=args.n_boot, seed=args.seed
    )

    print(f"\nMulti-Rater Agreement Statistical Turing Test")
    print(f"  Candidate column   : {args.candidate}")
    print(f"  Human panel        : {result['raters_compared']}")
    print(f"  N fragments        : {result['n_items']}")
    print(f"  Bootstrap iters    : {result['n_boot']}\n")
    print(f"  Human-only kappa (point estimate) : {result['human_only_kappa_point_estimate']:.3f}")
    for rater, delta in result["substitution_deltas_point_estimate"].items():
        print(f"    delta-kappa substituting for {rater:>8s} (point est.): {delta:+.3f}")
    print(f"  Natural-noise margin (from human-only bootstrap) : {result['margin']:.3f}")
    print(f"  Delta-kappa bootstrap mean   : {result['delta_kappa_bootstrap_mean']:+.3f}")
    print(f"  Delta-kappa 5th percentile   : {result['delta_kappa_5th_percentile']:+.3f}")
    print(f"  Delta-kappa 95th percentile  : {result['delta_kappa_95th_percentile']:+.3f}")
    print(f"\n  DECISION: {result['decision']}\n")


if __name__ == "__main__":
    main()


# NOTES
# -----
# - "all" variant: candidate must beat/match EVERY human rater, not just the
#   average. Compute delta_boot separately per held-out rater (don't average
#   across per_substitution) and require ALL of them to individually pass
#   their own 5th-percentile-vs-margin check.
# - "majority" variant: require a strict majority of the per-rater checks
#   above to pass.
# - "any" variant: require at least one per-rater check to pass. This is the
#   most permissive variant -- use with caution, it's easy to satisfy by
#   chance with only two human raters.
# - With only two human raters (rater_1, rater_2), Fleiss' kappa on the
#   human-only panel and on each substituted panel reduces to a close cousin
#   of Cohen's kappa -- this is expected and fine, not a bug.
# - If you add a third human rater later, this script needs no changes:
#   fleiss_kappa() and the substitution loop are already written generically
#   for any panel size >= 2.
