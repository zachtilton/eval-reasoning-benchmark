# RQ2 — Findings (in progress)

**Research question (verbatim, proposal):** *What systematic failure patterns
emerge from the LLM artifacts?*

**Status:** working, started 2026-11-12. Corrected (accuracy-only) scoring;
majority labels per fragment × model × condition. Structure: all-model
patterns first, then GPT (primary diagnostic model) close-up.

---

## Finding 2.1 — Shared failures (agreed 2026-11-12)

### Headline

> **The six models got the same passages wrong: what mattered was which
> passage was being judged, not which model judged it.**

### Expanded

> The six models largely passed and failed the same passages. Thirty-one
> passages were failed by every model and 35 by none; if each model erred
> independently at its own rate, about two of each would be expected. Which
> passage was being judged accounted for about half the variation in
> outcomes; which model judged it accounted for almost none, consistent with
> the absence of significant differences between models. Nearly all of the
> primary model's failures (107 of 112) were shared with at least two other
> models. Shared failure is not the same as shared error, however: of the 31
> passages every model failed, 26 were among those where the models jointly
> disagreed with the expert, and on expert review 13 were confirmed as
> genuine model errors while 18 rested on reference labels open to question.

### Plain-language version

> The six AI models tended to get the same passages right and the same
> passages wrong. Thirty-one passages tripped up every model, and 35 were
> handled correctly by all of them; by chance alone, we'd expect only two or
> three of each. What mattered was *which passage* was being judged, not
> *which model* was judging it. When the expert re-examined the passages that
> every model got "wrong," about 4 in 10 were genuine model mistakes; the
> other 6 in 10 were cases where the expert's original answer was itself
> open to question.

**Implications (candidates):** switching models is unlikely to fix these
errors, since they share the same blind spots; and when several models agree
against a human expert, that is a signal worth checking, not a verdict.

### Evidence

| Models failing a fragment (failure in either condition) | Observed | Expected if independent* |
|---|---|---|
| 0 | 35 | 2.7 |
| 1–5 | 84 | 145.3 |
| 6 | 31 | 2.0 |

\*Exact (Poisson-binomial) from each model's own any-condition fail rate
(46–51%).

- Share of pass/fail variation explained (η², 1,800 combinations): fragment
  0.52 (chance baseline ≈ 0.08 with 150 groups), model 0.002, prompt
  condition 0.001.
- Of the 31 all-six fragments: 26 are in the 35-fragment contested set; 26
  are gold not sound; GPT coding confirmed 13 as genuine failures and doubted
  the gold on 18.
- Challenge cases (H.4, first run on corrected outcomes + confirmed coded
  rows): Criterion 1 holds for 107 of GPT's 112 failures (96%; 45 with all
  five comparison models failing; 72 fragments). Criterion 2 qualifying
  codes: F5.1 47.6%, F5.2 19.0%, F4.2 11.9% of 42 coded rows. 32 of 33
  Criterion 2 cases fall inside Criterion 1 — the module's "low overlap"
  label reflects Criterion 1's size, not independence.

### Caveat

"Difficulty" means disagreement with the locked gold. Shared failure can
reflect a genuinely tricky fragment, a questionable gold label, or a
borderline fragment that splits qualified evaluators — all fragment-level
properties the outcome data alone cannot separate.

**Link to 2.2:** 26 of the 31 all-six fragments are gold not sound — the
clusters are largely missed flaws.

---

## Finding 2.2 — Leniency (agreed 2026-11-12; Option A; length and criterion checks added; headline settled 2026-11-12)

### Headline

> **Models were lenient judges: they missed flawed reasoning more often
> than they rejected sound reasoning. Without examples, the lean depended on
> passage length: none on short passages, about 2 to 1 on long ones. The
> study's calibration examples amplified it at every length, to about 5 to 1
> on long passages.**

### Expanded

> Without examples, models called flawed passages sound 50% of the time and
> sound passages flawed 33% of the time, a ratio of 1.5 to 1 (95% CI 1.1–2.0).
> That lean depended on passage length: there was no clear lean on short
> passages (1–3 paragraphs: 0.87, CI 0.56–1.28) and a clear one on long
> passages (7+ paragraphs: 2.26, CI 1.41–4.18). The models' own share of
> "sound" verdicts rose with length (47% → 62% → 68%) while the expert's did
> not (49%, 59%, 45%), so it was the models, not the expert, that treated
> longer passages as sounder. Length also explains why the 38 passages rated
> by two independent evaluators — all of them short — showed no zero-shot
> lean against any judge. Without examples the lean came mainly from Gemini,
> GLM and Kimi; GPT, Claude and DeepSeek leaned strict, and on re-review the
> expert judged GPT's strict calls defensible in 32 of 33 cases. With the
> study's two calibration examples, the gap widened to 62% vs 16%, nearly 4
> to 1 (CI 2.7–5.9), as all three strict models turned lenient; that lean
> held under every label version (3.6–4.7), against every human judge on the
> short subset (2.0–3.1), within every evaluation criterion, and at every
> length (2.6 short → 5.4 long). Because only one set of examples was tested,
> the amplification shows what a particular calibration set can do, not what
> few-shot prompting does in general. Leniency also accounts for apparent
> differences between evaluation criteria: impact looked hardest and
> relevance easiest, but once the share of flawed passages was taken into
> account, criterion explained nothing (p = .94).

### Evidence

Error rates are conditional on the reference label (rate within each group),
not shares of all errors. Ratio = missed flaws ÷ wrongly rejected (> 1
lenient, < 1 strict). CIs are fragment-clustered bootstraps (5,000, seed 42).

**By reference-label version and condition (all 150 fragments)**

| Reference labels | Zero-shot: missed vs rejected (ratio) | Few-shot: missed vs rejected (ratio) | Pooled ratio |
|---|---|---|---|
| Locked gold | 50.0% vs 33.0% (1.52; CI 1.14–2.01) | 61.6% vs 16.3% (3.79; CI 2.67–5.89) | 2.27 (CI 1.73–3.07) |
| Pre-revision ratings (18 pre-lock revisions undone) | 44.1% vs 31.1% (1.42) | 56.5% vs 15.7% (3.59) | 2.15 |
| Triangulated (35 contested → third reviewer) | 32.2% vs 23.3% (1.39) | 48.1% vs 10.3% (4.67) | 2.39 |

**By passage length (locked gold)**

| Length | Passages | Median words | Gold not sound | Zero-shot ratio (CI) | Few-shot ratio (CI) | Models call sound, zero → few |
|---|---|---|---|---|---|---|
| 1–3 paragraphs | 55 | 191 | 51% | 0.87 (0.56–1.28) | 2.64 (1.59–5.33) | 47% → 67% |
| 4–6 paragraphs | 46 | 495 | 41% | 1.80 (0.96–3.48) | 3.86 (1.98–9.47) | 62% → 73% |
| 7+ paragraphs | 49 | 815 | 55% | 2.26 (1.41–4.18) | 5.43 (2.74–20.49) | 68% → 80% |

Word count vs models' share of sound calls per fragment: Spearman ρ = .32
(p = .0001) zero-shot, .29 (p = .0003) few-shot. All 38 RC2 passages are
1–3 paragraphs; the other 17 short passages show a zero-shot ratio of 1.15.
In 5,000 random 38-passage subsets (any length), only 0.2–0.4% had a
zero-shot ratio as low as the RC2 subset's 0.78.

**Against independent human judges (38 RC2 fragments, all 1–3 paragraphs)**

| Reference | Calls sound | Zero-shot ratio | Few-shot ratio | Pooled ratio |
|---|---|---|---|---|
| Expert (pass 2, independent) | 36.8% | 0.85 | 3.05 | 1.50 |
| Rater 1 | 44.7% | 0.80 | 2.81 | 1.39 |
| Locked gold | 50.0% | 0.78 | 2.63 | 1.32 |
| Rater 2 | 55.3% | 0.83 | 1.98 | 1.25 |

**By evaluation criterion (locked gold; fragment-level fail rate)**

| Criterion | n | Gold not sound | Zero-shot: flawed vs sound | Few-shot: flawed vs sound |
|---|---|---|---|---|
| Impact | 17 | 71% | 53% vs 40% | 68% vs 23% |
| Coherence | 7 | 57% | 63% vs 33% | 71% vs 11% |
| Other | 22 | 59% | 46% vs 33% | 65% vs 17% |
| Sustainability | 35 | 46% | 49% vs 40% | 55% vs 23% |
| Efficiency | 25 | 56% | 44% vs 24% | 61% vs 6% |
| Effectiveness | 17 | 35% | 50% vs 35% | 58% vs 17% |
| Relevance | 27 | 33% | 57% vs 28% | 59% vs 15% |

Lenient gap present in 7/7 criteria in both conditions. Criterion given gold
label: F(6, 142) = 0.29, p = .94 (stratified permutation p = .94); zero-shot
p = .91, few-shot p = .95. Gold not-sound share across criteria: χ²(6) =
8.8, p = .18 (impact vs relevance Fisher p = .029, chosen post hoc as the
extremes — not interpreted). Models' own not-sound share by criterion
(gold-independent): permutation p = .94; correlation with the gold's
criterion pattern r = .43, p = .33. RC2 raters: same direction for impact vs
relevance on 4 and 8 fragments.

**By model and condition (locked gold)**

| Model | Zero-shot: missed vs rejected | Few-shot: missed vs rejected |
|---|---|---|
| Kimi K3 | 62% vs 20% | 82% vs 4% |
| GLM 5.2 | 70% vs 12% | 76% vs 5% |
| Gemini 3 Pro | 72% vs 18% | 66% vs 16% |
| GPT 5.6 Sol | 32% vs 43% | 58% vs 16% |
| Claude Opus 5 | 34% vs 43% | 46% vs 26% |
| DeepSeek V4 Pro | 30% vs 61% | 41% vs 30% |

9 of 12 combinations lean lenient; the 3 strict ones are GPT, Claude and
DeepSeek at zero-shot, and all three flip lenient with few-shot.

**Share of fragments called "sound" (gold: 50.7%)**

| Model | Zero-shot | Few-shot |
|---|---|---|
| GPT 5.6 Sol | 44.7% | 71.3% |
| Claude Opus 5 | 45.3% | 60.0% |
| DeepSeek V4 Pro | 34.7% | 55.3% |
| Gemini 3 Pro | 76.7% | 75.3% |
| GLM 5.2 | 79.2% | 85.3% |
| Kimi K3 | 71.3% | 89.3% |

**Expert re-review (GPT failure coding)** — lenient errors 27/43
fragments confirmed genuine (62.8%); strict 1/33 (3.0%). By length:
confirmed lenient as share of gold-not-sound passages 36% / 26% / 44%
(short / medium / long); strict failures 17 / 9 / 7 with the gold doubted on
16 / 9 / 7; F5.1 in about half of confirmed failures at every length (5/11,
2/5, 6/12). `results/failure_coding_synthesis.json`,
`results/findings_metrics.json` → `gpt_failures_by_length`.

### Caveats

- One calibration set was tested (two fixed examples, fixed order). The
  few-shot amplification is a property of that set; it does not establish
  what few-shot prompting does in general (stimulus sampling).
- The zero-shot lean is length-dependent and comes from three of six
  models. On short passages, models' apparent strict errors were largely
  judged defensible on re-review (GPT only), so "no clear lean" there may
  understate model leniency.
- Length may stand in for features that grow with it (more evidence, more
  structure, more evaluative language). The "models reward text that looks
  evaluative" explanation is untested; F5.1 did not concentrate in long
  passages. A pre-specified lexicon test is parked for RQ3.
- Independent-judge checks exist only for short passages (RC2 subset).
- The GPT re-review is the expert reconsidering the expert's own labels
  with GPT's rationale in view; it is not independent confirmation, and it
  exists for GPT only.
- The triangulated row is a one-directional sensitivity (contested set
  selected where models disagreed with gold).

---

## Finding 2.3 — Calibration examples shifted verdicts, not accuracy (agreed 2026-11-12)

### Headline

> **A balanced, expert-written pair of calibration examples shifted five of
> six models toward calling reasoning sound, without reliably improving
> accuracy.**

### Expanded

> Adding two calibration examples (one sound, one not sound) changed models'
> verdicts on the same passages almost entirely in one direction: 142 of 156
> changes (91%) went from "not sound" to "sound." The shift was strong for
> four models (GPT, DeepSeek, Kimi, Claude; p < .0001), modest for GLM
> (p = .04), and absent for Gemini. GPT, for example, changed 40 verdicts, all
> toward "sound." Accuracy barely moved because the changes cut both ways: of
> GPT's 40 changes, 21 corrected an unfair rejection and 19 let a flawed
> passage through; across all 156 changes, 90 fixed an error and 66 created
> one. Only DeepSeek gained meaningfully (+10 points). This is why the planned
> pass/fail comparisons found no overall prompting effect: they register how
> many verdicts are right, not which way they move. With examples, models
> also became more alike and more consistent run to run, converging at a
> more lenient level. Because one fixed pair of examples was tested, this
> demonstrates what a calibration set *can* do, not what few-shot prompting
> does in general.

### Plain-language version

> Showing the AI models two worked examples didn't make them better judges.
> It made them more generous ones. They fixed some unfair rejections but let
> through about as many flawed passages, so their scores barely changed while
> their judgments shifted noticeably.

### Practical implications

1. **Calibration examples can introduce bias that accuracy scores hide.**
   Anyone building AI evaluation tools should check which way errors fall,
   not just how many there are.
2. **Examples don't necessarily carry their author's standard.** Examples
   written by the expert moved models *away* from that standard: with
   examples, models called 55–89% of passages sound, against the expert's 51%.
3. **"Balanced" examples aren't neutral.** One example of each label still
   produced a strong one-way push. Balancing the labels doesn't balance the
   lessons.

### Possible mechanism (hypothesis, for RQ3)

The "sound" example's rationale openly concedes a flaw ("There may be a
mismatch in the evidence… but there is a grand judgment here"); the "not
sound" example lacks both a judgment and evidence. Together they may teach
that only passages missing *everything* are "not sound." Recency bias would
predict a pull toward "not sound" (the last example), so it does not explain
the shift. Source: `config/prompts/few_shot_template.txt`.

### Evidence

| Model | Sound, zero-shot | Sound, few-shot | → sound | → not sound | Exact McNemar p | Changes fixing / creating an error |
|---|---|---|---|---|---|---|
| GPT 5.6 Sol | 44.7% | 71.3% | 40 | 0 | < .0001 | 21 / 19 |
| DeepSeek V4 Pro | 34.7% | 55.3% | 31 | 0 | < .0001 | 23 / 8 |
| Kimi K3 | 71.3% | 89.3% | 28 | 1 | < .0001 | 13 / 16 |
| Claude Opus 5 | 45.3% | 60.0% | 24 | 2 | < .0001 | 15 / 11 |
| GLM 5.2 | 78.9% | 85.7% | 15 | 5 | .041 | 10 / 10 |
| Gemini 3 Pro | 76.7% | 75.3% | 4 | 6 | .75 | 8 / 2 |

- Pooled pass rate (H.2 McNemar on pass/fail): 58.6% → 61.2%, p = .067;
  DeepSeek 54.7% → 64.7%, p = .012 (the only significant model).
- Convergence: spread in share called sound 44 → 34 points (35–79% → 55–89%);
  model–model κ 0.45 → 0.49; run-to-run unanimity 85.9% → 87.7%.
- GLM paired n = 147 (3 fragments lack a majority in one condition).

### Caveats

- Existence proof only: presence of examples, their content, their order and
  prompt length all change together (stimulus sampling).
- The verdict shift does not use the gold standard; the fixed/created
  counts and pass rates do.
- GLM's p = .041 would not survive a correction for six tests.

---

## Still to discuss

- GPT close-up: disposition (28/76 confirmed), failure-code profile
  (Domain 5 = 64% of primary codes; F5.1 most common), condition pattern
  (few-shot-only failures 68% confirmed vs zero-shot-only 5%), criterion.
- Mainline RQ2 answer.

*Provenance:* every figure above is computed by
`src/analysis/findings_metrics.py` (→ `results/findings_metrics.json`) or an
existing results file; `docs/findings/findings_registry.json` maps each claim
to its metric, procedure and data sources.
