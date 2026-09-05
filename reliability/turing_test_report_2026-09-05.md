# Multi-Rater Agreement Statistical Turing Test — Report

**Date:** 2026-09-05
**Input data:** `reliability/multi_rater_output_data.csv`
**Script:** `reliability/multi_rater_turing_test.py`
**Method:** Kljajic, J., O'Toole, J. M., Hogan, R., & Skoric, T. (2025). *Honest
and reliable evaluation and expert equivalence testing of automated neonatal
seizure detection.* arXiv:2508.04899. ("Average kappa" variant.)

## What was tested

Whether the single-coder gold-standard **anchor** is a statistically valid
stand-in for a human rater, relative to the two-person human panel
(`rater_1`, `rater_2`). The anchor was substituted into the panel in place of
each human rater in turn, and the resulting change in Fleiss' kappa was
compared against the natural sampling noise of the human-only panel.

## Data note

The raw CSV had inconsistent capitalization across columns (`rater_1` /
`rater_2` used `"Sound"` / `"Not sound"`; `anchor` used lowercase
`"sound"` / `"not sound"`; one cell in `rater_2` was also lowercase). Values
were normalized (lowercased, whitespace-stripped) before analysis — this was
done in-memory only and did not modify the source CSV. After normalization,
all three columns resolved cleanly to the two expected categories:
`sound` / `not sound`.

## Test parameters

- N fragments: 40
- Human panel: `rater_1`, `rater_2`
- Candidate: `anchor`
- Bootstrap iterations: 5,000
- Seed: 42

## Results

| Quantity | Value |
|---|---|
| Human-only kappa (point estimate) | 0.500 |
| Δκ substituting for `rater_1` (point estimate) | -0.465 |
| Δκ substituting for `rater_2` (point estimate) | -0.451 |
| Natural-noise margin (human-only bootstrap) | 0.292 |
| Δκ bootstrap mean | -0.462 |
| Δκ 5th percentile | -0.774 |
| Δκ 95th percentile | -0.153 |

**Decision rule:** pass requires `Δκ 5th percentile > -margin`, i.e.
`-0.774 > -0.292`. This is false.

**DECISION: NOT EQUIVALENT (fails).**

The drop in agreement from substituting the anchor for a human rater
(~-0.46 on average) is more than 2.5x larger than the margin attributable to
ordinary sampling noise alone (0.292). This is not a borderline result.

## Disagreement breakdown

### Anchor vs. each rater individually

| | vs `rater_1` | vs `rater_2` |
|---|---|---|
| Simple agreement | 21/40 = 52.5% | 21/40 = 52.5% |
| Anchor says "sound", rater says "not sound" (anchor too lenient) | 12 | 10 |
| Anchor says "not sound", rater says "sound" (anchor too strict) | 7 | 9 |

Both raw agreement rates sit close to chance level for a roughly balanced
binary variable (base rates ~45-57% "sound" across all three columns).

### Performance on fragments where the human raters already agree

- 30/40 fragments have `rater_1` == `rater_2` (75% raw human-human
  agreement; kappa 0.500).
- On those 30 consensus fragments, the anchor agrees with the human
  consensus in only **16/30 = 53.3%** of cases — near coin-flip, even on
  fragments the human panel found unambiguous.
- Of the 14 consensus fragments the anchor disagrees on:
  - **9 are anchor-too-lenient** (humans: "not sound"; anchor: "sound"):
    report_ids 703, 939, 1001, 1058, 1367, 2517, 4116, 5946, 6279
  - **5 are anchor-too-strict** (humans: "sound"; anchor: "not sound"):
    report_ids 1071, 1942, 2019, 4425, 5285

This is a directional skew: when the anchor errs on a clear-consensus
fragment, it over-calls "sound" roughly 2x as often as it over-calls "not
sound." This is consistent with the anchor's overall "sound" rate (57.5%)
running higher than either human rater's.

### Fragments where the human raters disagree with each other (N=10)

The anchor's calls split evenly on these ambiguous fragments — it sides with
`rater_1` on 5 and with `rater_2` on 5 — showing no consistent alignment
with either rater when the panel itself is split. This subset is not where
the failure is concentrated.

| report_id | rater_1 | rater_2 | anchor |
|---|---|---|---|
| 1286 | not sound | sound | sound |
| 1977 | not sound | sound | sound |
| 2172 | not sound | sound | sound |
| 2515 | not sound | sound | not sound |
| 2988 | sound | not sound | sound |
| 3515 | not sound | sound | not sound |
| 3760 | sound | not sound | not sound |
| 4576 | sound | not sound | not sound |
| 5896 | not sound | sound | not sound |
| 6834 | not sound | sound | not sound |

## Interpretation

The anchor fails the equivalence test, and the failure is not primarily a
boundary-calibration problem on hard/ambiguous fragments — it is
concentrated in the 30 fragments the human panel already agrees on, where
the anchor should be performing best. The failure also has a direction: the
anchor over-calls "sound" about twice as often as it over-calls "not sound"
on those consensus fragments.

## Suggested next step

Pull the anchor's rationale text for a sample of the 9 lenient-error
consensus fragments (e.g. 703, 939, 1001, 1058, 4116) to check for a shared
pattern in what the anchor's coding rule is letting slide relative to the
human raters' standard for "sound."
