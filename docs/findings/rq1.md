# RQ1 — Findings

**Research question (verbatim, proposal):** *To what extent can large language
models accurately assess the quality of evaluative reasoning?*

**Status:** working answer, agreed 2026-11-11. Corrected (accuracy-only)
scoring against the locked gold standard
(`data/gold_standard/gold_standard_locked_2026-10-03.csv`).

## Mainline answer

> **LLMs agreed with the expert 60% of the time, barely better than a coin
> flip but about as often as trained human evaluators did, and they erred
> toward leniency.**

## Expanded answer

> Under corrected scoring, LLMs agreed with expert judgments of evaluative
> reasoning on ~60% of fragments (95% CI 54–66%). That is modestly above
> chance (κ ≈ 0.20), below the expert's own test-retest consistency (85%),
> and comparable to trained human evaluators' agreement with the same
> expert's independent ratings. It rises to ~74% when contested labels are
> resolved by an independent third reviewer. That adjustment can only raise
> the figure, and it applies to fragments where qualified evaluators
> themselves disagree. The most robust result is directional: models
> affirmed sound reasoning far more reliably than they detected flawed
> reasoning (75% vs 44%). Expert review of the primary model's errors
> confirmed most lenient errors (27 of 43) as genuine and attributed nearly
> all strict errors (32 of 33) to the gold standard.

## Evidence

| Claim | Figure | Source |
|---|---|---|
| Fragment-level agreement (pooled, 6 models × 2 conditions × 150) | 59.9%, 95% CI 54.3–65.5% (fragment-clustered bootstrap, 5,000 resamples, seed 42) | `data/responses/fragment_outcomes_corrected.csv` |
| Run-level agreement | 59.6% (3,220 / 5,400) | `data/responses/scored_database_corrected.csv` |
| Chance-corrected agreement, model vs anchor | κ mean 0.20 (range 0.10–0.29); raw 55–65% | scratch `rq1_agreement.py` (not yet folded into `src/`) |
| Accuracy by gold label | sound 75% vs not sound 44% | same |
| No model differences | χ²(5) = 3.24, p = .66, V = 0.04; closed 60.9% vs open 58.9%, z = 0.87, p = .39 | `results/H2_statistical_tests_corrected.json` |
| Run-to-run consistency | 86.8% of 1,800 combinations unanimous across 3 runs | `fragment_outcomes_corrected.csv` |
| Expert test-retest (RC1) | 85% agreement, κ = 0.70, n = 20 | `results/reliability_checks.json` |
| Human–human (RC2 raters) | 75% agreement, κ = 0.50, n = 40 | `reliability/blind_retest_40frag_pass2.csv` |
| Humans vs anchor's independent (pass-2) ratings, 38 shared fragments | 55.3% / 60.5% (κ 0.13 / 0.19) | same |
| Models vs anchor pass-2 ratings, same 38 fragments | mean 57.3% (range 42–76%) | same + corrected DB majority labels |
| Humans vs locked gold, same 38 (partly circular: gold revised toward raters) | 68.4% / 73.7% (κ 0.37 / 0.47) | same |
| Triangulated sensitivity (35 contested labels resolved by third reviewer) | 73.6%, 95% CI 69.1–78.1% | `data/responses/fragment_outcomes_triangulated.csv` |
| Third reviewer sided with models | 25 / 35 (lenient side 19/25, strict side 6/10) | `data/responses/contested_fragments_review_resolved.csv` |
| GPT error confirmation (failure coding) | lenient 27/43 genuine; strict 1/33 genuine | `results/failure_coding_synthesis.json` |
| Pre-lock revisions vs models | all models agree more with pre-revision ratings (+1.4 to +8.0 pp, mean +5.5) | same |

## Notes on the contested set

- Selection: fragments where ≥5 of 6 models' majority labels agreed with
  each other and disagreed with the gold (20 at 6/6, 15 at 5/6). The
  reviewer instructions said "all six"; the write-up should say "≥5 of 6".
- The external reviewer was a third person, not one of the RC2 raters.
- 10 of the 35 were pre-lock revisions. Four (703, 1001, 1367, 2019) were
  rated directly by the RC2 raters; each is a 2–2 human split (RC2 raters
  vs the anchor's original rating + third reviewer), with the locked gold
  following the RC2 raters and the models following the other two. The
  other six were flipped by the anchor applying the calibration rule.
- Roughly one in four fragments split qualified evaluators (human–human
  agreement 75%; 35/150 contested).

## Caveats

1. **Pre-registration deviation.** Corrected scoring removed the coherence
   gate after RC3 found 0 of 217 sampled flagged runs incoherent. The
   finding concerns classification agreement, not rationale coherence.
   RC3 was a sample (64/216 incoherent, 153/1,217 ambiguous reviewed); the
   rest were scored by extension.
2. **Human comparison is small and not held out.** 38 fragments (±~15 pp);
   the anchor's pass-2 ratings followed calibration on these fragments.
   "Comparable", not "equivalent".
3. **Single-expert gold standard,** calibrated against a two-person panel
   but not statistically equivalent to it (RC2).
4. **Scope.** 150 UN evaluation fragments; binary sound / not sound; six
   specific 2026 model versions; possible training-data exposure.
5. **Rounding.** Report 60%, not 59.9%: the sampling interval spans 11
   points and reference-standard uncertainty widens it further.

## Methods note (deviation disclosure)

> The pre-registered scoring treated rationales flagged as incoherent by an
> automated pipeline as failures (pass rate 43.3%). Human validation of a
> stratified sample of flagged runs (RC3; n = 217) found none incoherent,
> so the gate was removed and all usable responses were scored on
> classification accuracy.

The 43.3% figure belongs here only, not alongside the 60% / 74% estimates.

## Poster

> **LLMs agreed with the expert 60% of the time, barely better than a coin
> flip but about as often as trained human evaluators did, and they erred
> toward leniency.**
>
> *Robustness: 60% (95% CI 54–66%); 74% when contested labels are
> independently resolved. Expert's own test-retest: 85%. Roughly 1 in 4
> fragments split qualified evaluators. Human comparison: two independent
> evaluators, n = 38.*
