# Multi-Rater Agreement Statistical Turing Test — Disposition

**Date:** 2026-11-03
**Prior reports:**
`reliability/turing_test_report_2026-09-05.md` (original failure),
`reliability/turing_test_report_2026-09-22.md` (pass 1, "anchor 2.0"),
`reliability/turing_test_report_2026-09-22_pass2.md` (pass 2, "anchor 3.0")
**Raw data for this note:** `reliability/blind_retest_40frag_pass2.csv`
(pass 2 / "anchor3" column added here; `rater_1`, `rater_2`, `anchor1`
columns unchanged from `multi_rater_output_data.csv`)

## Summary of the arc

The anchor's gold-standard classification did not reach statistical
equivalence with the two-person human panel on the original 40-fragment
equivalence test (2026-09-05; Δκ 5th percentile -0.774 against a -0.292
margin). A calibration rule was derived from the raters' own written
rationale on the 14 disagreement fragments, confirmed by both raters, and
reapplied blind in two independent passes to test whether the correction
was real or coding noise. Both passes narrowed the gap substantially
(Δκ 5th percentile -0.547, then -0.619) but **neither reached
equivalence**. Pass-to-pass self-agreement (κ = 0.846) confirmed the
correction's effect is real and stable, not noise -- and in the process
surfaced two further, narrower gaps in the rule (a length-relative
evidence-sufficiency standard; no treatment for on-topic but
unstructured evidence) that were drafted as a further revision but
deliberately left untested, to avoid tuning the rule against the same
40-fragment sample repeatedly -- doing so would have stopped being
validation and started being overfitting to that specific held-out set.

## Disposition

This was raised with the dissertation committee. The committee's
direction was to proceed rather than continue iterating; the
overfitting rationale above was not part of their stated basis for that
call, and should not be attributed to them -- it is the author's own
supporting reasoning for why stopping here, rather than continuing to
revise against the same 40 fragments, was independently the right
methodological choice as well as the practical one.

**Decision:** proceeded with the remaining 110-fragment audit using the
*validated* rule (the version confirmed by both raters and tested
stable across passes 1 and 2) -- not the further-revised, untested
version. The full 150-fragment gold standard was subsequently completed
and locked: `data/gold_standard/gold_standard_locked_2026-10-03.csv`.

## What this means going forward

The gold standard this dissertation's model comparisons are built on is
a documented, imperfect approximation of expert judgment, not an
infallible ground truth -- confirmed by two independent lines of
evidence: RC1's intra-rater test-retest (κ = 0.70, see
`reliability/rc1_expert_temporal_consistency.csv` and
`results/reliability_checks.json`) and this Turing test's persistent
(though substantially narrowed) equivalence gap against the human
panel. The human panel itself is not a clean ceiling either -- rater_1
and rater_2's own inter-rater agreement was κ = 0.500.

Any later claim that a model falls short of (or matches) "human-level"
evaluative reasoning should be read against this baseline: the anchor's
own measured distance from the human panel is not small, so a model's
distance from the anchor is only interpretable relative to that
baseline, not against an assumed-perfect ground truth. This belongs in
the dissertation's methodological limitations section (spec Section
3.6) alongside the other documented limitations.
