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

## Finding 2.2 — Leniency (agreed 2026-11-12; scoped 2026-11-12)

### Headline

> **Models missed flawed reasoning about twice as often as they wrongly
> rejected sound reasoning, judged against the expert; the same lean
> appeared against two independent trained evaluators.**

### Expanded

> When the expert judged a passage's reasoning flawed, models called it
> sound 56% of the time. When the expert judged it sound, models called it
> flawed 25% of the time. The roughly 2-to-1 ratio held under all three
> versions of the reference labels: the locked gold (56% vs 25%), the
> expert's pre-revision ratings (50% vs 23%), and contested labels resolved
> by an independent reviewer (40% vs 17%). Three models (Gemini, GLM, Kimi)
> were lenient with or without examples. The other three (GPT, Claude,
> DeepSeek) leaned strict without examples but turned lenient once given
> them, so few-shot prompting increased leniency rather than correcting it.
> The lean is not specific to the expert's standard: on the 38 passages
> also rated by two independent trained evaluators, models leaned lenient
> against every judge to a similar degree (missed-to-rejected ratios
> 1.25–1.5), even though the judges themselves called between 37% and 55%
> of those passages sound. The expert's own re-review of the primary
> model's errors points the same way (27 of 43 lenient errors confirmed,
> against 1 of 33 strict errors), though that review is not independent of
> the expert's labels.

### Evidence

Error rates are conditional on the reference label (rate within each group),
not shares of all errors.

**By reference-label version (pooled, all models and conditions)**

| Reference labels | Sound labels (of 150) | Missed flaws (ref. not sound) | Wrongly rejected (ref. sound) |
|---|---|---|---|
| Locked gold | 76 | 55.8% | 24.6% |
| Pre-revision ratings (18 pre-lock revisions undone) | 88 | 50.3% | 23.4% |
| Triangulated (35 contested → third reviewer) | 89 | 40.1% | 16.8% |

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

**Against independent human judges (38 RC2 fragments in the benchmark)**

| Reference | Calls sound | Model missed flaws | Model wrongly rejected | Ratio |
|---|---|---|---|---|
| Expert (pass 2, independent) | 36.8% | 48.4% | 32.3% | 1.50 |
| Rater 1 | 44.7% | 46.6% | 33.5% | 1.39 |
| Locked gold | 50.0% | 46.1% | 35.0% | 1.32 |
| Rater 2 | 55.3% | 58.1% | 46.6% | 1.25 |

Models' own share called sound on these fragments: mean 55.6% (range
13.2–89.5% across the 12 model × condition combinations). The ratio is
smaller on this subset than on all 150 fragments (2.27) whichever judge is
used, so the subset speaks to the direction of the lean, not its size.

**Expert re-review (GPT failure coding)** — lenient errors 27/43
fragments confirmed genuine (62.8%); strict 1/33 (3.0%).
`results/failure_coding_synthesis.json`.

### Caveats

- "About twice as often" is the full-sample estimate against the locked
  gold. The independent-judge check (n = 38) supports the direction but
  cannot confirm the size.
- The GPT re-review is the expert reconsidering the expert's own labels
  with GPT's rationale in view; it is not independent confirmation.
- Confirmation of error direction exists only for GPT; whether Claude's and
  DeepSeek's zero-shot strict errors are likewise mostly gold artifacts is
  unverified.
- The 15 sound→not-sound pre-lock revisions could have manufactured
  leniency; the pre-revision row shows the asymmetry survives without them.
- The triangulated row is a one-directional sensitivity (contested set
  selected where models disagreed with gold).

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
