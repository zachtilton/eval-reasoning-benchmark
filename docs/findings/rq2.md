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

## Finding 2.2 — Leniency (agreed 2026-11-12; revised to Option A 2026-11-12)

### Headline

> **Without examples, models missed flawed reasoning somewhat more often than
> they wrongly rejected sound reasoning (1.5 to 1); the study's calibration
> examples widened that gap to nearly 4 to 1.**

### Expanded

> Without examples, models called flawed passages sound 50% of the time and
> sound passages flawed 33% of the time, a ratio of 1.5 to 1. That baseline
> held under all three versions of the reference labels (1.4 to 1.5) but
> came from half the models: Gemini, GLM and Kimi were lenient, while GPT,
> Claude and DeepSeek leaned strict. It is also sample-dependent: on the 38
> passages also rated by two independent trained evaluators, models without
> examples leaned slightly strict against every judge, the locked gold
> included (0.8 to 1). With the study's two calibration examples, the gap
> widened to 62% vs 16%, nearly 4 to 1, as all three strict models turned
> lenient; that lean held under every label version (3.6 to 4.7) and
> against every human judge on the 38-passage subset (2.0 to 3.1). Because
> only one set of examples was tested, the amplification shows what a
> particular calibration set can do, not what few-shot prompting does in
> general. The expert's own re-review of the primary model's errors points
> toward leniency as well (27 of 43 lenient errors confirmed, against 1 of
> 33 strict errors), though that review is not independent of the expert's
> labels.

### Evidence

Error rates are conditional on the reference label (rate within each group),
not shares of all errors. Ratio = missed flaws ÷ wrongly rejected (> 1
lenient, < 1 strict).

**By reference-label version and condition (all 150 fragments)**

| Reference labels | Zero-shot: missed vs rejected (ratio) | Few-shot: missed vs rejected (ratio) | Pooled ratio |
|---|---|---|---|
| Locked gold | 50.0% vs 33.0% (1.52) | 61.6% vs 16.3% (3.79) | 2.27 |
| Pre-revision ratings (18 pre-lock revisions undone) | 44.1% vs 31.1% (1.42) | 56.5% vs 15.7% (3.59) | 2.15 |
| Triangulated (35 contested → third reviewer) | 32.2% vs 23.3% (1.39) | 48.1% vs 10.3% (4.67) | 2.39 |

**Against independent human judges (38 RC2 fragments in the benchmark)**

| Reference | Calls sound | Zero-shot ratio | Few-shot ratio | Pooled ratio |
|---|---|---|---|---|
| Expert (pass 2, independent) | 36.8% | 0.85 | 3.05 | 1.50 |
| Rater 1 | 44.7% | 0.80 | 2.81 | 1.39 |
| Locked gold | 50.0% | 0.78 | 2.63 | 1.32 |
| Rater 2 | 55.3% | 0.83 | 1.98 | 1.25 |

Models' share called sound on these fragments: 44.6% zero-shot, 66.6%
few-shot (mean across models). Within each condition the ratios are similar
whichever judge is the reference, so the judge matters little; the
condition and the sample matter a lot.

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
fragments confirmed genuine (62.8%); strict 1/33 (3.0%).
`results/failure_coding_synthesis.json`.

### Caveats

- One calibration set was tested (two fixed examples, fixed order). The
  few-shot amplification is a property of that set; it does not establish
  what few-shot prompting does in general (stimulus sampling).
- The zero-shot baseline is modest, comes from three of six models, and
  reverses (slightly) on the 38-passage RC2 subset against every judge.
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
