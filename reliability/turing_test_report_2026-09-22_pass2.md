# Multi-Rater Agreement Statistical Turing Test — Second Blind Pass Report

**Date:** 2026-09-22
**Prior reports:** `reliability/turing_test_report_2026-09-05.md` (original failure),
`reliability/turing_test_report_2026-09-22.md` (first retest, "anchor 2.0")
**Input data:** `reliability/multi_rater_output_data.csv` (rater_1, rater_2, anchor 1.0) +
`reliability/blind_retest_40frag_completed.xlsx` (anchor 2.0, pass 1) +
`reliability/blind_retest_40frag_v2.xlsx` (anchor 3.0, pass 2)
**Script:** `reliability/multi_rater_turing_test.py`
**Method:** Kljajic, J., O'Toole, J. M., Hogan, R., & Skoric, T. (2025). *Honest
and reliable evaluation and expert equivalence testing of automated neonatal
seizure detection.* arXiv:2508.04899. ("Average kappa" variant.)

## What was tested

The first retest (anchor 2.0) narrowed the equivalence gap but did not close
it, and its self-stability against the original anchor (κ = 0.458) sat below
the anchor's established 0.70 intra-rater baseline — leaving open whether
that gap reflected noise in a single coding pass or a real, stable property
of the rule. This report reruns the identical blind procedure a second time
(same rule text, same 40 fragments, same shuffle), producing "anchor 3.0,"
to distinguish those two possibilities directly by comparing pass 1 against
pass 2, not just each pass against the original anchor.

## Test parameters

- N fragments: 40
- Human panel: `rater_1`, `rater_2` (unchanged throughout)
- Candidate: `anchor3` (second blind pass, same rule as anchor 2.0)
- Bootstrap iterations: 5,000
- Seed: 42

## Results — equivalence test, all three anchor versions

| Quantity | 1.0 (2026-09-05) | 2.0 / pass 1 (2026-09-22) | 3.0 / pass 2 (this report) |
|---|---|---|---|
| Human-only kappa | 0.500 | 0.500 | 0.500 |
| Δκ substituting for `rater_1` | -0.465 | -0.300 | -0.355 |
| Δκ substituting for `rater_2` | -0.451 | -0.207 | -0.274 |
| Margin | 0.292 | 0.292 | 0.292 |
| Δκ bootstrap mean | -0.462 | -0.257 | -0.319 |
| Δκ 5th percentile | -0.774 | -0.547 | -0.619 |
| Δκ 95th percentile | -0.153 | +0.026 | -0.028 |
| Decision | NOT EQUIVALENT | NOT EQUIVALENT | NOT EQUIVALENT |

Pass 2 sits between the original anchor and pass 1 — still a substantial
improvement over 1.0, slightly worse than pass 1 on this bootstrap, both
consistent with sampling variation around a stable underlying effect rather
than a trend in either direction.

## The real finding: pass 1 vs. pass 2 (self-stability)

| Comparison | Agreement | Cohen's kappa | Fragments flipped |
|---|---|---|---|
| Anchor 1.0 vs. pass 1 | 72.5% (29/40) | 0.458 | 11 |
| Anchor 1.0 vs. pass 2 | 75.0% (30/40) | 0.518 | 10 |
| **Pass 1 vs. pass 2** | **92.5% (37/40)** | **0.846** | **3** |

Pass 1 and pass 2 agree with each other far more than either agrees with
the original anchor, and above the 0.70 intra-rater baseline. This resolves
the open question from the first retest report: the low agreement against
anchor 1.0 is not primarily coding noise — the rule is being applied
consistently across independent blind passes. It genuinely produces
different judgments than the anchor's original, pre-calibration standard.
All 3 pass-to-pass flips fall inside the 14 fragments that were already the
site of the original anchor-vs-consensus disagreement (1071, 2019, 5946);
the 16 full-agreement and 10 rater-disagreement fragments were coded
identically in both passes.

## Disagreement breakdown

### The 14 anchor-vs-consensus fragments (original failure set)

| report_id | consensus | anchor 1.0 | pass 1 | pass 2 | pass 1 match | pass 2 match |
|---|---|---|---|---|---|---|
| 703 | not sound | sound | not sound | not sound | yes | yes |
| 939 | not sound | sound | sound | sound | no | no |
| 1001 | not sound | sound | not sound | not sound | yes | yes |
| 1058 | not sound | sound | sound | sound | no | no |
| 1071 | sound | not sound | sound | not sound | yes | no |
| 1367 | not sound | sound | not sound | not sound | yes | yes |
| 1942 | sound | not sound | sound | sound | yes | yes |
| 2019 | sound | not sound | sound | not sound | yes | no |
| 2517 | not sound | sound | sound | sound | no | no |
| 4116 | not sound | sound | sound | sound | no | no |
| 4425 | sound | not sound | not sound | not sound | no | no |
| 5285 | sound | not sound | not sound | not sound | no | no |
| 5946 | not sound | sound | sound | not sound | no | yes |
| 6279 | not sound | sound | not sound | not sound | yes | yes |

Pass 1: 7/14 match consensus (κ = 0.039). Pass 2: 6/14 (κ = -0.244) — a net
wash driven by 1071 and 2019 reverting to the original "not sound" call
while 5946 newly corrected. 1071 and 2019 behaving as swing cases across
passes is itself informative (see Interpretation).

### The 16 full-agreement fragments

Both passes: 13/16 match (κ = 0.636). The same 3 fragments are wrong in
both passes — 256, 2978, 6782 — confirming this is a stable effect, not
one-off noise.

### The 10 rater_1-vs-rater_2 disagreement fragments

Pass 2 sides with rater_1 on 6 and rater_2 on 4, matching pass 1's split.
No change; this subset still awaits direct adjudication between rater_1
and rater_2.

## Interpretation

Two things carry forward from this second pass. First, the rule's effect
is real and stable, not measurement noise — pass-to-pass agreement (0.846)
exceeds the anchor's own established reliability baseline (0.70), and every
instance of instability is confined to fragments that were already
genuinely contested, not spread across the full 40. Second, two specific,
bounded gaps in the rule text are now identifiable rather than speculative:
(1) 256, 2978, and 6782 fail consistently on a sufficiency standard that
does not account for a fragment's length — the anchor's own original
rationale on these three explicitly reasoned in relative terms ("sparse,
but sufficient," "sufficient... relative to segment size") that the current
rule text does not encode; (2) 1071 and 2019 swing between passes on a
different axis entirely — evidence that is on-topic but organizationally
dense (1071) or lacking explicitly stated warrants/caveats (2019) — which
the rule does not currently address at all.

## Suggested next step

Revise the rule to add both missing provisions (a length-relative
sufficiency standard; tolerance for evidence that is on-topic but formally
unstructured or spread across multiple sources) and send the revised text
back to rater_1 and rater_2 for confirmation before treating it as final —
it now differs from the version they already signed off on.
