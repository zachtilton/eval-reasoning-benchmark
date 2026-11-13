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
> chance (κ ≈ 0.20), below the expert's own test-retest consistency (85%, κ = 0.70),
> and comparable to trained human evaluators' agreement with the same
> expert's independent ratings. It rises to ~74% when contested labels are
> resolved by an independent third reviewer. That adjustment can only raise
> the figure, and it applies to fragments where qualified evaluators
> themselves disagree. The most robust result is directional: models
> affirmed sound reasoning far more reliably than they detected flawed
> reasoning (75% vs 44%), a lean that also appeared against two
> independent trained evaluators. The expert's own re-review of the primary
> model's errors confirmed most lenient errors (27 of 43) as genuine and
> attributed nearly all strict errors (32 of 33) to the gold standard.

## Findings

The RQ1 answer is supported by three findings (structure agreed 2026-11-13).
Headline convention: each finding leads with a local implication, followed
by the observation that warrants it.

### Finding 1.1 — Accuracy (agreed 2026-11-13)

> **Models are not yet reliable judges of evaluative reasoning: they agreed
> with the expert 60% of the time, modestly above chance, and caught
> flawed reasoning far less reliably than they affirmed sound reasoning.**

*Previous (descriptive) headline:* Models agreed with the expert 60% of the time, modestly above chance and well below the expert's own consistency, and they caught flawed reasoning far less reliably than they affirmed sound reasoning.

Pooled across six models and two conditions, models agreed with the expert
on 59.9% of passages (95% CI 54.3–65.5%; 59.6% at the run level).
Chance-corrected agreement was κ ≈ 0.20 (0.10–0.29 across model ×
condition), against the expert's own blind test-retest of 85% (κ = 0.70,
n = 20). The six models were statistically indistinguishable (χ²(5) = 3.24,
p = .66; closed vs open p = .39). Models got 75% of gold-sound passages
right but only 44% of gold-not-sound passages. All three runs agreed in
86.8% of model × condition × passage combinations.

### Finding 1.2 — The human yardstick (agreed 2026-11-13)

> **Expert judgment is a contested yardstick: models agreed with the expert
> about as often as two trained evaluators did (about 57% vs 55–61%).**

*Previous (descriptive) headline:* Matching the expert was hard for everyone: models agreed with the expert about as often as two trained evaluators did (about 57% vs 55–61%).

On the 38 RC2 passages in the benchmark, models agreed with the expert's
blind pass-2 ratings 57.5% of the time on average (42–76% across model ×
condition); the two trained evaluators agreed with the same ratings 60.5%
and 55.3% of the time. The evaluators agreed with each other on 75% of
passages (κ = 0.50), disagreeing on one in four, and disagreed with the
expert's pass-2 ratings on about two in five (about one in two against the
expert's original ratings). Against the locked gold the evaluators agreed
68–74%, partly circular because the gold was revised toward them. A third
reviewer, rating the 35 passages where at least five models agreed against
the expert, sided with the models on 25; substituting those judgments raised
overall model agreement from 60% to 74% (95% CI 69–78%), an adjustment that
can only raise the figure.

### Finding 1.3 — Re-review changed the expert's mind (agreed 2026-11-12 as 2.4; re-homed and headline revised 2026-11-13)

#### Headline

> **The benchmark likely understates model capability: on re-review, most of the primary model's apparent failures rested on labels the expert no longer defended.**

*Earlier headline (2.4):* Re-examining the primary model's disagreements
changed the expert's mind more often than not: on re-review, most of its
apparent failures — nearly all of its stricter calls — rested on labels the
expert no longer defended.

#### Expanded

> The expert re-reviewed every passage the primary model (GPT) failed, with
> the model's rationale in view, and either confirmed the failure or doubted
> the original label. Of 76 failed passages, 28 were confirmed and 48 rested
> on labels the expert no longer defended, cutting the model's failure rate
> from 37% to 14% of its 300 fragment-condition cases. The doubts were not
> spread evenly: the expert sided with the model on 32 of 33 cases where the
> model was stricter than the original label, but on only 16 of 43 where it
> was more lenient. That selectivity argues against simple deference to
> fluent model reasoning, and it moved the gold standard stricter — the
> second time reconsidering the standard did so (the first was the RC2
> calibration before the lock). An independent third reviewer pointed the
> same way, siding with the models on 25 of 35 contested passages. Because
> only the model's disagreements were re-examined, the direction is clear
> (the benchmark understates defensible model verdicts) but the size is not:
> errors where the model agreed with a mistaken label were never looked for.

#### Interpretation: argumentative capability

The model's reasoning was good enough to change an expert's mind. That is a
capability the benchmark score does not measure: classification accuracy is
scored against a fixed label, while argumentative capability is the
capacity of a model's rationale to move a qualified human's judgment. It
carries a promise and a risk for evaluation practice:

- **Promise:** a model as a second reader that surfaces weaknesses in expert
  judgment — here, mainly where the expert's first pass was too generous.
- **Risk:** fluency that persuades whether or not it is right. A
  well-argued rationale can pull a reviewer toward its verdict; the
  asymmetry above is evidence against pure persuasion here, but a blind
  check is needed (RC4 persuasion check).

#### Reflexive note

The expert's view of many of their own labels changed during the
re-review. This is reported as analytic data, not as proof that the model
was right; the asymmetry, the third reviewer and the planned blind re-rating
are what test it.

#### Evidence

| | Failed passages | Confirmed | Gold doubted |
|---|---|---|---|
| All | 76 | 28 (37%) | 48 (63%) |
| Model stricter than label | 33 | 1 | 32 (97%) |
| Model more lenient than label | 43 | 27 | 16 (37%) |

- Failure rate: 37.3% against the locked gold → 14.0% counting only
  confirmed failures (of 300 GPT fragment × condition cases).
- Third reviewer on the 35 contested passages: sided with the models on 25.
- Reconsiderations of the standard: RC2 calibration (15 of 18 pre-lock
  revisions sound → not sound; blind pass 2 called 36.8% of 38 passages
  sound vs 57.9% originally) and this re-review (net 32 stricter vs 16 more
  lenient).
- `results/failure_coding_synthesis.json`, `results/findings_metrics.json`.

#### Caveats

- Not independent: the expert reconsidered the expert's own labels with the
  model's rationale in view (persuasion is a live rival explanation).
- One-sided: only failures were re-reviewed, so 14% understates the model's
  error rate by an unknown amount, and 63% describes inspected failures, not
  the gold's overall error rate.
- Coding stability untested (RC4 pending); one model; one coder.
- The expert and the third reviewer barely agreed on which contested labels
  were wrong (κ = 0.04).

---

## Evidence

| Claim | Figure | Source |
|---|---|---|
| Fragment-level agreement (pooled, 6 models × 2 conditions × 150) | 59.9%, 95% CI 54.3–65.5% (fragment-clustered bootstrap, 5,000 resamples, seed 42) | `data/responses/fragment_outcomes_corrected.csv` |
| Run-level agreement | 59.6% (3,220 / 5,400) | `data/responses/scored_database_corrected.csv` |
| Chance-corrected agreement, model vs anchor | κ mean 0.20 (range 0.10–0.29); raw 55–65% | `results/findings_metrics.json` → `model_anchor_agreement` |
| Accuracy by gold label | sound 75% vs not sound 44% | same |
| No model differences | χ²(5) = 3.24, p = .66, V = 0.04; closed 60.9% vs open 58.9%, z = 0.87, p = .39 | `results/H2_statistical_tests_corrected.json` |
| Run-to-run consistency | 86.8% of 1,800 combinations unanimous across 3 runs | `fragment_outcomes_corrected.csv` |
| Expert test-retest (RC1) | 85% agreement, κ = 0.70, n = 20 | `results/reliability_checks.json` |
| Human–human (RC2 raters) | 75% agreement, κ = 0.50, n = 40 | `reliability/blind_retest_40frag_pass2.csv` |
| Humans vs anchor's independent (pass-2) ratings, 38 shared fragments | 55.3% / 60.5% (κ 0.13 / 0.19) | same |
| Models vs anchor pass-2 ratings, same 38 fragments | mean 57.5% (range 42–76%) | same + corrected DB majority labels |
| Humans vs locked gold, same 38 (partly circular: gold revised toward raters) | 68.4% / 73.7% (κ 0.37 / 0.47) | same |
| Triangulated sensitivity (35 contested labels resolved by third reviewer) | 73.6%, 95% CI 68.9–78.1% | `data/responses/fragment_outcomes_triangulated.csv` |
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
- Roughly one in four fragments split qualified evaluators: the two RC2
  raters disagreed on 25% (primary estimate). The contested set (35/150,
  23%) is a looser proxy; the third reviewer disagreed with the expert on 25
  of those (17% of all passages, a floor).

## Caveats

1. **Pre-registration deviation.** Corrected scoring removed the coherence
   gate after RC3 found 0 of 217 sampled flagged runs incoherent. The
   finding concerns classification agreement, not rationale coherence.
   RC3 was a sample (64/216 incoherent, 153/1,217 ambiguous reviewed; the 64
   ambiguous runs a second model could not call were fully reviewed); the
   rest were scored by extension. Rule-of-three bound: at most ~43 genuinely
   incoherent runs (0.8% of 5,400) among those never reviewed. The rater
   never assigned "incoherent", including to 94 controls.
2. **Human comparison is small and not held out.** 38 fragments (±~15 pp);
   the anchor's pass-2 ratings followed calibration on these fragments.
   "Comparable", not "equivalent".
3. **Single-expert gold standard,** calibrated against a two-person panel
   but not statistically equivalent to it (RC2), with test-retest κ = 0.70
   (below the 0.80 target). All accuracy and error-direction figures are
   agreement with this expert's standard, not with an independent truth;
   claims are worded accordingly. The expert's re-review of GPT's errors is
   a reconsideration of the expert's own labels, not independent
   confirmation.
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
> independently resolved. Expert's own test-retest: 85% (κ = 0.70). Roughly 1 in 4
> fragments split qualified evaluators. Human comparison: two independent
> evaluators, n = 38.*

---

*Provenance:* every figure above is computed by
`src/analysis/findings_metrics.py` (→ `results/findings_metrics.json`) or an
existing results file; `docs/findings/findings_registry.json` maps each claim
to its metric, procedure and data sources.
