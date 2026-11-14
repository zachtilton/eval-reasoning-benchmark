# Analysis log — exploratory and unplanned analyses

Lab notes for analyses that were **not specified in the proposal** but arose
from inspecting results ("why is that there?") or from testing assumptions
behind a finding. Each entry records what prompted it, what was tested,
what came out, and what it changed. Results from these analyses are
**exploratory**: they generate and qualify findings but are not
confirmatory tests, and the number of such looks should be kept in mind
when weighing any single p-value (forking paths).

Planned analyses (proposal Sections 3.1–3.6; Appendix H) are not logged here
except where their execution departed from the plan.

Conventions: **Status** = planned-but-modified / exploratory / correction.
**Metric** = key in `results/findings_metrics.json` unless stated.
Commits are on `main`.

**Numbering note (2026-11-13):** Finding 2.4 (re-review) was re-homed to RQ1 as 1.3, and Finding 2.5 (missing judgment, F5.1) renumbered 2.4; then Finding 2.3 (calibration examples) moved to RQ4 as 4.2 and the missing-judgment finding became 2.3. Earlier entries and commit messages use the old numbers.

---

## Pre-specifications (written before running)

### P1 — Output-side ("reading") corrections for model leniency (2026-11-13; adopted by the author before any results)

**Question.** Can rules applied to the existing model outputs, without new
runs, improve model judgment of evaluative reasoning?

**Primary outcome.** Balanced accuracy against the locked gold (mean of
accuracy on gold-sound and on gold-not-sound passages). It decides whether a
rule "improves" judgment and is the criterion used to choose any tuned
parameter inside training folds.

**Secondary outcomes (reported whatever they show).** Plain agreement with
the gold; leniency ratio (missed-flaw rate ÷ wrongly-rejected rate).

**Rules (two only).**
- **R1 Stricter vote:** a model's verdict on a passage is "sound" only if all
  of its usable runs (at least two) say sound; otherwise "not sound".
  No tuned parameter. Compared with the baseline 2-of-3 majority label on the
  same fragment × model × condition units.
- **R2 Pooling across models:** one verdict per passage × condition: "sound"
  only if at least k of the six models' majority labels say sound. k ∈ {1,…,6}
  chosen by balanced accuracy in training folds. Compared with the mean of the
  six individual models' balanced accuracy on the same test passages.

**Conditions.** Zero-shot and few-shot, reported separately.

**Validation.** R2: 5-fold cross-validation over passages, stratified by
gold label, repeated 20 times (seed 42); report the mean test-fold result,
the spread across repeats, and the distribution of chosen k. R1 has no tuned
parameter and is evaluated on all 150 passages, with a fragment-clustered
bootstrap 95% CI (5,000 resamples, seed 42) for the difference from
baseline.

**Status.** Exploratory. Any improvement is measured against the expert's
labels and on the passages used here; it would need testing on new passages
or prompts before being claimed generally.

---

## Deviations from planned analyses

| ID | Date | Planned analysis | Departure | Why | Effect |
|---|---|---|---|---|---|
| D1 | 2026-11-05 | Coherence gate in scoring (Tier 1 + Tier 2) | Gate removed; all usable responses scored on classification accuracy ("corrected scoring") | RC3: 0 of 217 sampled flagged runs rated incoherent | Pass rate 43.3% → 59.9%; coherence no longer measured. Metric: `rc3_deviation` |
| D2 | 2026-11-12 | H.4 challenge cases (never previously run) | Run on corrected outcomes with confirmed coded rows only | Proposal's Type 1 coding became a two-layer disposition + code | Criterion 2 almost fully nested in Criterion 1; module's "low overlap" label misleading. Metric: `challenge_cases` |
| D3 | 2026-11-11 | RC3 sample 50 responses (20/20/10) | Enlarged stratified sample (314 sampled, 311 rated) | Precision (~±10 pp) | See D1 |

---

## Exploratory analyses

| ID | Date | Trigger (what prompted it) | Question / test | Result | Effect on findings | Metric | Commit |
|---|---|---|---|---|---|---|---|
| E1 | 2026-11-11 | Author: "did I see all 1,433 non-coherent runs?" | RC3 coverage by stratum; rule-of-three bound on unreviewed runs | 217 of 1,433 reviewed; ≤ ~43 incoherent among unreviewed (0.8% of runs) | RQ1 caveat 1 | `rc3_deviation.unreviewed_bound` | 317d517 |
| E2 | 2026-11-11 | Need a human yardstick in % (not only κ) | Rater–rater and rater–anchor % agreement; models vs anchor on the 38 shared RC2 fragments | Humans 75% with each other; humans vs anchor pass 2 55–61%; models 57.5% | RQ1 "comparable to trained evaluators" | `human_comparison` | c67972d |
| E3 | 2026-11-11 | Is 59.9% vs 73.6% sampling noise? | Fragment-clustered bootstrap CIs for pass rates | 54–66% vs 69–78%, non-overlapping | RQ1 robustness line | `pass_rate` | c67972d |
| E4 | 2026-11-11 | "How were the 35 contested labels contested?" | Contested-set composition; overlap with pre-lock revisions and RC2 ratings | ≥5/6 (not 6/6) model agreement; 10 overlap revisions; 4 directly rated cases all 2–2 human splits | RQ1 notes; "≈1 in 4 fragments split evaluators" | `contested_review`, `contested_overlap`, `evaluator_split` | 317d517 |
| E5 | 2026-11-12 | "Difficulty is a property of the fragment" — is that beyond chance? | Observed vs independence-expected count of models failing each fragment; η² by fragment / model / condition | 31 all-six and 35 none vs ~2 each expected; η² fragment .52, model .002 | Finding 2.1 | `shared_failures` | c67972d |
| E6 | 2026-11-12 | Do the revisions manufacture leniency? | Error asymmetry under locked, pre-revision and triangulated labels | ~2:1 under all three | 2.2 robustness | `error_asymmetry` | c67972d |
| E7 | 2026-11-12 | Author: findings rest on a shaky gold standard | Leniency against each human judge on the 38 RC2 fragments | Lean present against every judge (pooled) | 2.2 scoped; see C1 for the confound in the first pass | `leniency_vs_human_raters` | 3bc56f6 |
| E8 | 2026-11-12 | Tempting implication: a model's "not sound" is more trustworthy | Predictive value of each verdict under three label versions | 64% vs 58% (locked), reversed pre-revision | Implication **not** adopted | (session only; not registered) | — |
| E9 | 2026-11-12 | Accuracy barely moved with few-shot — why? | Label flips between conditions, exact McNemar per model; flips that fixed vs created an error | 142 of 156 flips toward sound; 90 fixed / 66 created | Finding 2.3 | `few_shot_shift` | 7be88a0, 138fe0f |
| E10 | 2026-11-12 | What could teach leniency? | Qualitative read of the two calibration examples (`config/prompts/few_shot_template.txt`) | Sound example concedes a flaw; not-sound example has neither judgment nor evidence | Mechanism hypothesis (2.3); stimulus-sampling caveat | — | — |
| E11 | 2026-11-12 | Author: is the effect few-shot per se or these examples? | Asymmetry split by condition | Zero-shot 1.52 vs few-shot 3.79 | 2.2 revised (Option A) | `error_asymmetry.*.by_condition` | fa254ff |
| E12 | 2026-11-12 | Option A needs zero-shot robustness | Per-condition human-judge check | Zero-shot ratio < 1 against every judge on the RC2 subset | 2.2 qualified; led to E15–E16 | `leniency_vs_human_raters.by_condition` | fa254ff |
| E13 | 2026-11-12 | RQ2 #4: impact looks hardest | Fail rate by criterion with clustered CIs; within-label split | Fail rates track gold not-sound share | Led to E14 | `criterion_variation` | 639b9ac |
| E14 | 2026-11-12 | Is criterion difficulty just composition? Is the impact pattern real? | Stratified nested F + permutation; gold composition χ²; models' own not-sound share; RC2 raters by criterion; per condition | Criterion adds nothing given label (p = .94; .91 / .95 by condition); gold composition p = .18; corroboration suggestive only | #4 folded into 2.2; impact hypothesis parked for RQ3 | `criterion_checks`, `criterion_by_condition` | 1126f70, 8ef32e0 |
| E15 | 2026-11-12 | Is the RC2-subset reversal noise? | Bootstrap CI for ratios; 5,000 random 38-fragment subsets | Zero-shot CI 1.14–2.01; subset ratio matched by only 0.2–0.4% of random subsets | Subset judged unusual, not noise | `leniency_ratio_uncertainty` | 2125041 |
| E16 | 2026-11-12 | Author: RC2 fragments were stratified to 1–3 paragraphs | Leniency by paragraph-count bin (bins chosen **after** this prompt); word count vs models' sound calls | No zero-shot lean on 1–3 paragraphs, clear on 7+; models' sound calls rise with length, gold's do not | 2.2 headline (length); RC2 reversal explained | `length_moderation` | cbe69c7 |
| E17 | 2026-11-12 | Does long text hide missing judgments (F5.1)? | GPT confirmed failures by length; F5.1 share by length | F5.1 ≈ half at every length; strict failures concentrate in short passages but gold doubted at every length | Hypothesis not supported; lexicon test parked for RQ3 | `gpt_failures_by_length` | 22cd31e |
| E18 | 2026-11-12 | The not-sound calibration example was itself a "no judgment" passage | F5.1 by prompt condition (rows); conditions in which F5.1 fragments failed | F5.1 in 7 zero-shot vs 13 few-shot confirmed rows; all 13 F5.1 fragments failed with examples, none only without | Pending discussion (finding on F5.1) | `f51_by_condition` | 8ca56e6 |
| E19 | 2026-11-12 | Calibration-boundary hypothesis predicts missed F5.1 passages have evidence | Evidence codes (F2.x) among the 13 confirmed F5.1 fragments, by conditions failed | 8 of 13 lack an evidence code; but 5 of the 6 missed only with examples DO carry F2.2, while all 7 missed in both conditions lack one | Prediction (as stated) not supported for the example-induced misses; hypothesis not adopted as stated | `f51_by_condition` | 3ce2fc8 |

| E20 | 2026-11-12 | RQ3: proposal's construct-validity questions; author asked to raise Stairway propagation now | D4 vs D2 codes; D6 codes by condition; challenge Criterion 1 share by criterion; D5 co-occurrence with D1–D4 codes in confirmed fragments | D2 13 vs D4 10; D6 1 vs 4 rows; sustainability 54% vs relevance 41%; 16 of 20 D5 fragments also carry lower-level codes | Queued in `rq3.md` for interpretation | `theory_checks` | be9b0d1 |
| E21 | 2026-11-13 | RQ4: leniency by family and family × prompting not covered by H.2 | Per family × condition: share called sound, missed/rejected rates, ratio, pass rate; zero→few changes and closed − open contrasts with fragment-clustered bootstrap CIs | Open models more lenient in both conditions (called sound −6 / −8 pp closed vs open); no accuracy difference; examples shifted both families similarly (13 vs 15 pp) | `rq4.md` | `family_checks` | a7fc799 |
| E22 | 2026-11-13 | Conclusions (Cluster A): would we trust RC2 raters and models equally at equal agreement? | Each RC2 rater's missed-flaw vs wrongly-rejected rates against the expert's pass-2 ratings and the locked gold (38 fragments), beside the models' | vs pass 2: rater 1 0.88 (37.5% vs 42.9%), rater 2 1.40 (50.0% vs 35.7%); models zero-shot 0.85, few-shot 3.05, pooled 1.50 | Zero-shot models within the human range on short passages; with examples, beyond both raters. Pending conclusions | `rater_leniency` | 48cb313 |
| E23 | 2026-11-13 | Pre-specified P1 (see Pre-specifications) | R1 stricter vote; R2 pooling across six models with k tuned by cross-validation; primary outcome balanced accuracy | Zero-shot: no gain from either (R1 −0.7 pp, CI −2.7 to 1.2; R2 +0.2 pp), though R1 removed the lean (ratio 1.52 → 1.03). Few-shot: R1 +0.8 pp (CI −0.9 to 2.6), ratio 3.79 → 2.48; R2 chose k = 6 in every fold (all six models must agree to call sound): balanced accuracy 66.9% vs 61.2% for individual models (+5.7 pp, 5.6–5.9 across repeats), leniency ratio ≈ 1.0 | Pending conclusions (correctability) | `reading_corrections` | 5893904 |
| E24 | 2026-11-13 | Author asked for a CI on the P1 pooling gain (not pre-specified) | Fragment-clustered bootstrap of pooled − individual balanced accuracy: (a) rule fixed at k = 6; (b) whole cross-validated procedure re-run per resample | Few-shot: (a) +5.7 pp, CI 0.8 to 10.7; (b) +3.9 pp, CI −5.0 to 10.4. Zero-shot: (a) −1.6 pp, CI −6.3 to 3.5; (b) −0.4 pp, CI −8.0 to 6.7 | Few-shot pooling gain is suggestive, not established, once uncertainty in choosing the rule is included | `pooling_bootstrap` | 7a9485c |
| E25 | 2026-11-13 | C4 lacked a denominator (share of failures ≠ miss rate); author: not-sound rationales always name a missing judgment | Classified 74 not-sound gold rationales (phrase match + 7 author-confirmed; buried and stale kept separate; author decisions 2026-11-13), saved to `data/analysis/gold_not_sound_reasons.csv`; model miss rate by category and condition | 48 judgment-absent (71% of not-sound with known reasons; 32% of all passages), 14 other flaw, 6 buried, 6 stale. Miss rate on judgment-absent: 51% zero-shot, 61% few-shot (other flaw 42%, 55%; buried 61%, 81%). By model 31–77% | Judgment-absent passages are missed about as often as other flaws; F5.1 dominates confirmed failures largely because it dominates the flaws. Pending: C4 landing, 2.3 wording, C-a | `judgment_absent_rates` | aa8dcc7 |
| E26 | 2026-11-13 | C4's "comparable to raters" standard had no rater benchmark; author asked to check the RC2 overlap | RC2 fragments classified judgment-absent in `gold_not_sound_reasons.csv`; how often each RC2 rater called them sound vs models' majority labels by condition | 6 overlapping passages (19 of 40 RC2 fragments have a not-sound reason). Raters missed 5 of 12 ratings (42%; rater 1 2/6, rater 2 3/6). Models missed 12 of 36 zero-shot (33%), 20 of 36 few-shot (56%) | Too small to settle C4's parity level (6 passages); direction: zero-shot models no worse than raters, few-shot worse. Pending C4 standard decision | ad hoc (not yet in `findings_metrics`) | — |
| E27 | 2026-11-13 | Evidence-rubric rating of Domain 5 part (b): the model–rater comparison had no CI; author asked for one after tightening the precision descriptor | Fragment-clustered bootstrap (38 shared RC2 fragments, 5,000 resamples, seed 42) of model mean agreement (over model × condition) minus each rater and the rater mean, against blind pass 2 and locked gold | vs pass 2: models − rater mean −0.4 pp (CI −15.4 to 14.8); − rater 1 −3.1 (−17.9 to 12.7); − rater 2 +2.2 (−16.6 to 20.9). vs locked: −11.6 (−25.9 to 3.8); −14.2 (−28.8 to 1.4); −8.9 (−27.3 to 10.2) | No detectable difference against either reference, but intervals of about ±15 pp cannot show a match; against the locked gold the estimate leans below the raters. Pending rating of (b) | `human_comparison_ci` | 9768d72 |
| E28 | 2026-11-13 | Evidence-rubric rating of Domain 5 part (c): no CI on the missed-flaw rate; author asked for one | Fragment-clustered bootstrap (150 fragments, 5,000 resamples, seed 42); outputs fixed before running: missed-flaw rate vs locked and triangulated (pooled, per condition), judgment-absent miss rate, judgment-absent share of misses | Locked: 55.8% (CI 47.7–64.1); zero-shot 50.0 (41.6–58.6); few-shot 61.6 (53.3–70.1). Triangulated: 40.1 (32.4–48.0); 32.2 (25.0–40.0); 48.1 (39.4–57.0). Judgment-absent miss 55.7 (45.0–66.4). Judgment-absent share of misses 64.6 (51.5–77.1) | "About half" fits the locked gold (CI 48–64) but not the triangulated reference (32–48). Pending rating of (c) | `missed_flaw_ci` | 6d2eb77 |
---

## Corrections caught during analysis

| ID | Date | What was stated | Correct | Cause |
|---|---|---|---|---|
| C1 | 2026-11-12 | Leniency "smaller against other evaluators" | Similar across judges on the same 38 fragments | Compared a 150-fragment ratio with 38-fragment ratios |
| C2 | 2026-11-12 | Locked gold matches pass 2 on 82.5% | 86.8% (38 shared) | Two non-benchmark fragments counted as mismatches |
| C3 | 2026-11-12 | ≤ ~48 unreviewed incoherent runs | ≤ ~43 | Wrong reviewed-n for one stratum |
| C4 | 2026-11-12 | 9 of 10 revised contested fragments matched models originally | 10 of 10 | Miscounted fragment 2019 |
| C5 | 2026-11-12 | Few-shot flips: 79 fixed / 63 created | 90 / 66 (all flips; 79/63 was toward-sound only) | Subset reported as total |
| C6 | 2026-11-12 | Triangulated pass rate 74.3% | 73.6% | Misread figure |
| C7 | 2026-11-12 | The expert's standard moved stricter "three times" | Twice (RC2 calibration incl. pass 2; failure re-review) | Counted pass 2 separately from the calibration |
| C8 | 2026-11-12 | RC2 subset reversal "probably sampling noise" | Unusual (0.2–0.4% of random subsets); explained by length | Assumption stated before testing |
