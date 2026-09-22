# Multi-Rater Agreement Statistical Turing Test — Retest Report

**Date:** 2026-09-22
**Prior report:** `reliability/turing_test_report_2026-09-05.md`
**Input data:** `reliability/multi_rater_output_data.csv` (rater_1, rater_2, anchor 1.0) +
`reliability/blind_retest_40frag_completed.xlsx` (anchor 2.0)
**Script:** `reliability/multi_rater_turing_test.py`
**Method:** Kljajic, J., O'Toole, J. M., Hogan, R., & Skoric, T. (2025). *Honest
and reliable evaluation and expert equivalence testing of automated neonatal
seizure detection.* arXiv:2508.04899. ("Average kappa" variant.)

## What was tested

The 2026-09-05 report found the anchor NOT EQUIVALENT to the human panel
(rater_1, rater_2). Following that failure, a calibration rule was derived
from rater_1's and rater_2's written rationale on the 14 fragments where the
anchor disagreed with their consensus, confirmed by both raters, and then
reapplied by the anchor to all 40 fragments — blind, in randomized order,
with no indication of the original codes, the raters' codes, or which
comparison group each fragment belonged to. This report reruns the same
equivalence test with the recalibrated anchor ("anchor 2.0") substituted for
the original, and adds a self-stability check the first report did not need.

## Test parameters

- N fragments: 40
- Human panel: `rater_1`, `rater_2` (unchanged from 2026-09-05)
- Candidate: `anchor2` (recalibrated, blind reapplication)
- Bootstrap iterations: 5,000
- Seed: 42

## Results — equivalence test

| Quantity | 2026-09-05 (anchor 1.0) | 2026-09-22 (anchor 2.0) |
|---|---|---|
| Human-only kappa (point estimate) | 0.500 | 0.500 |
| Δκ substituting for `rater_1` | -0.465 | -0.300 |
| Δκ substituting for `rater_2` | -0.451 | -0.207 |
| Natural-noise margin | 0.292 | 0.292 |
| Δκ bootstrap mean | -0.462 | -0.257 |
| Δκ 5th percentile | -0.774 | -0.547 |
| Δκ 95th percentile | -0.153 | +0.026 |

**Decision rule:** pass requires `Δκ 5th percentile > -margin`, i.e.
`-0.547 > -0.292`. This is false.

**DECISION: NOT EQUIVALENT (fails).**

The gap narrowed substantially — the average drop in agreement from
substituting the anchor fell from ~-0.46 to ~-0.26, and the 5th-percentile
bootstrap bound moved from -0.774 to -0.547 — but it remains roughly 0.25
short of the non-inferiority margin. This is a real, measurable improvement,
not a fix.

## Self-stability: anchor 1.0 vs. anchor 2.0

Not part of the original test design; added because the recalibration
involved a second, independent coding pass by the same single coder, and
that pass's own consistency needs to be established before its results can
be trusted at face value.

| Quantity | Value |
|---|---|
| N fragments | 40 |
| Simple agreement | 72.5% (29/40) |
| Cohen's kappa | 0.458 |
| Fragments that flipped | 11/40 |

This sits notably below the anchor's established intra-rater test-retest
baseline (κ = 0.70, from prior stability testing). A gap of this size means
some portion of the change observed below is attributable to the anchor
coding less consistently on the second pass, not solely to the calibration
rule doing corrective work. The results in the next two sections should be
read with that caveat attached, not as a clean measure of the rule's effect
in isolation.

## Disagreement breakdown

### The 14 anchor-vs-consensus fragments (2026-09-05 failure set)

Anchor 2.0 now matches the rater_1/rater_2 consensus on 7/14 (50%,
up from 0/14 by construction). Cohen's kappa on this subset alone is 0.039 —
close to chance once the subset's base rates are accounted for.

| report_id | consensus | anchor 1.0 | anchor 2.0 | now matches? |
|---|---|---|---|---|
| 703 | not sound | sound | not sound | yes |
| 939 | not sound | sound | sound | no |
| 1001 | not sound | sound | not sound | yes |
| 1058 | not sound | sound | sound | no |
| 1071 | sound | not sound | sound | yes |
| 1367 | not sound | sound | not sound | yes |
| 1942 | sound | not sound | sound | yes |
| 2019 | sound | not sound | sound | yes |
| 2517 | not sound | sound | sound | no |
| 4116 | not sound | sound | sound | no |
| 4425 | sound | not sound | not sound | no |
| 5285 | sound | not sound | not sound | no |
| 5946 | not sound | sound | sound | no |
| 6279 | not sound | sound | not sound | yes |

### The 16 full-agreement fragments (anchor 1.0 already matched consensus)

Anchor 2.0 still matches on 13/16 (81.25%; kappa = 0.636). Three fragments
that were correct under anchor 1.0 are now wrong:

| report_id | consensus | anchor 1.0 | anchor 2.0 |
|---|---|---|---|
| 256 | sound | sound | not sound |
| 2978 | sound | sound | not sound |
| 6782 | sound | sound | not sound |

### The 10 rater_1-vs-rater_2 disagreement fragments

Anchor 2.0 sides with rater_1 on 6 and rater_2 on 4 — a similar split to
anchor 1.0's original 5/5 on this same subset (2026-09-05 report). This
subset has no "correct" answer to test against; it awaits direct
adjudication between rater_1 and rater_2.

## Interpretation

The recalibration rule produced a real, replicable improvement — roughly
halving the average agreement penalty and narrowing the equivalence gap by
a comparable margin — but the anchor still fails the equivalence test, and
two findings complicate a simple "the rule worked" reading:

First, self-stability (κ = 0.458) sits well below the anchor's own 0.70
baseline. An 11/40 flip rate on a second blind pass is more churn than pure
measurement noise would predict, which means part of the apparent fix on
the 14-fragment set may reflect inconsistency in this specific coding pass
rather than the rule's corrective content. The 14-set's near-chance kappa
(0.039) once matched (7/14) is consistent with this — it looks more like a
coin flip than a rule doing systematic work.

Second, the rule cost 3 fragments that were previously correct (256, 2978,
6782, all in the 16-set). All three failures are consistent with a risk
flagged before this retest was run: loosening the evidence-sufficiency
requirement (so that evidence "need not be exhaustive") to fix the anchor's
prior over-strictness may have reopened a narrower version of the original
leniency problem on fragments where the looser standard was not warranted.

## Suggested next step

Before this rule is used to scope the 110-fragment re-audit, two things are
worth resolving rather than treating this result as a pass: (1) whether the
self-stability gap reflects genuine noise in this specific pass (arguing for
a second blind pass to check) or a property of the rule itself that would
recur; and (2) whether 256, 2978, and 6782 share a textual pattern that
would let the evidence-sufficiency clause be tightened without reopening the
strictness problem it was written to fix.
