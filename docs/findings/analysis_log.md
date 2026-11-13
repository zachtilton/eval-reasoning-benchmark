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

**Numbering note (2026-11-13):** Finding 2.4 (re-review) was re-homed to RQ1 as 1.3, and Finding 2.5 (missing judgment, F5.1) renumbered 2.4. Earlier entries and commit messages use the old numbers.

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
