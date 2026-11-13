# RQ4 — Findings

**Research question (verbatim, proposal):** *How do failure patterns vary
across model families (frontier vs. open-source) and prompting conditions
(zero-shot vs. few-shot)?*

**Sample note (author, 2026-11-13):** the six models are a purposive sample
of the top-performing models in each family according to a public
leaderboard at the time of the run, chosen to test the frontier of both
family types. "Family" comparisons are therefore comparisons of three
specific models against three others, not a sample of families.

---

## Evidence assembled

| Factor | Evidence | Source |
|---|---|---|
| Family: accuracy | Closed 60.9% vs open 58.9%, p = .39 (Cohen's h = 0.04); no differences among the six models, p = .66 | H.2 tests (pre-specified, run) |
| Prompting: accuracy | 58.6% → 61.2%, p = .067; only DeepSeek gained significantly | H.2 McNemar (pre-specified, run) |
| Prompting: verdicts and leniency | Calibration examples shifted verdicts toward "sound" (142 of 156 changes; no gold standard needed), widening leniency from 1.5 to 3.8 against the expert's standard | 2.2, 4.2 |
| Family: leniency | Open models more lenient than closed in both conditions (below) | `family_checks` (exploratory, E21) |
| Family × prompting | Examples moved both families toward "sound" by similar amounts; accuracy changed equally (below) | `family_checks` (exploratory, E21) |

### Leniency by family and condition (locked gold; majority labels)

| Family × condition | Called sound | Missed flaws | Wrongly rejected | Ratio | Pass rate |
|---|---|---|---|---|---|
| Closed, zero-shot | 55.6% | 45.9% | 35.1% | 1.31 | 59.6% |
| Closed, few-shot | 68.9% | 56.8% | 19.3% | 2.94 | 62.2% |
| Open, zero-shot | 61.7% | 54.1% | 30.8% | 1.75 | 57.6% |
| Open, few-shot | 76.8% | 66.5% | 13.2% | 5.03 | 60.2% |

### Contrasts (fragment-clustered bootstrap 95% CIs, 5,000 resamples)

| Contrast | Pass rate (pp) | Called sound (pp) | Leniency ratio |
|---|---|---|---|
| Closed − open, zero-shot | +2.0 (−2.2 to 6.2) | −6.1 (−10.2 to −2.2) | −0.44 (−0.81 to −0.13) |
| Closed − open, few-shot | +2.0 (−1.8 to 5.8) | −7.9 (−11.5 to −4.4) | −2.09 (−4.01 to −0.96) |
| Effect of examples, closed | +2.7 (−1.3 to 6.9) | +13.3 (9.8 to 16.9) | +1.63 (0.94 to 3.14) |
| Effect of examples, open | +2.7 (−1.6 to 7.1) | +15.1 (11.5 to 18.9) | +3.28 (1.99 to 5.89) |
| Crossing (closed − open difference in the effect) | 0.0 (−4.2 to 4.2) | −1.8 (−6.0 to 2.4) | −1.65 (−3.50 to −0.59) |

Reading notes:
- Families did not differ in accuracy, with or without examples.
- The open models called more passages sound and were more lenient in both
  conditions.
- Examples shifted both families toward "sound" by similar amounts (13 vs.
  15 points; difference not distinguishable from zero). The leniency
  *ratio* rose more for open models, but a ratio grows faster from a more
  lenient starting point, so the linear measure (share called sound) is the
  better guide to whether the families responded differently: they did not
  clearly do so.

---

## Findings

Headline convention: each finding leads with a local implication, followed
by the observation that warrants it. Structure agreed 2026-11-13.

### Finding 4.1 — Model family (agreed 2026-11-13)

#### Headline

> **Choosing an open-source or closed model is unlikely to change accuracy,
> though it may change leniency: the two families matched on accuracy, but
> the open-source models called 6–8 points more passages sound, with and
> without examples.**

#### Expanded

> The two families matched on accuracy: closed models agreed with the expert
> on 60.9% of passages and open models on 58.9% (p = .39, Cohen's h = 0.04),
> and the six models were statistically indistinguishable (p = .66). They
> differed in leniency: the open-source models called 6.1 points more
> passages sound without examples (95% CI 2.2–10.2) and 7.9 points more with
> them (4.4–11.5), and their missed-flaw / wrongly-rejected ratio was higher
> in both conditions (1.75 vs 1.31 zero-shot; 5.03 vs 2.94 few-shot).
> Because each family is three purposively chosen top-performing models, the
> gap may reflect particular models (GLM and Kimi were the most lenient
> overall) rather than open-source models as such.

### Finding 4.2 — Calibration examples shifted verdicts, not accuracy (agreed 2026-11-12 as 2.3; moved to RQ4 2026-11-13)

#### Headline

> **Calibration examples can introduce bias that accuracy scores hide: a
> balanced, expert-written pair shifted five of six models toward calling
> reasoning sound, without reliably improving accuracy.**

*Previous (descriptive) headline:* A balanced, expert-written pair of calibration examples shifted five of six models toward calling reasoning sound, without reliably improving accuracy.

#### Expanded

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

#### Plain-language version

> Showing the AI models two worked examples didn't make them better judges.
> It made them more generous ones. They fixed some unfair rejections but let
> through about as many flawed passages, so their scores barely changed while
> their judgments shifted noticeably.

#### Practical implications

1. **Calibration examples can introduce bias that accuracy scores hide.**
   Anyone building AI evaluation tools should check which way errors fall,
   not just how many there are.
2. **Examples don't necessarily carry their author's standard.** Examples
   written by the expert moved models *away* from that standard: with
   examples, models called 55–89% of passages sound, against the expert's 51%.
3. **"Balanced" examples aren't neutral.** One example of each label still
   produced a strong one-way push. Balancing the labels doesn't balance the
   lessons.

#### Possible mechanism (hypothesis, for RQ3)

The "sound" example's rationale openly concedes a flaw ("There may be a
mismatch in the evidence… but there is a grand judgment here"); the "not
sound" example lacks both a judgment and evidence. Together they may teach
that only passages missing *everything* are "not sound." Recency bias would
predict a pull toward "not sound" (the last example), so it does not explain
the shift. Source: `config/prompts/few_shot_template.txt`.

#### Evidence

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

#### Caveats

- Existence proof only: presence of examples, their content, their order and
  prompt length all change together (stimulus sampling).
- The verdict shift does not use the gold standard; the fixed/created
  counts and pass rates do.
- GLM's p = .041 would not survive a correction for six tests.

### Finding 4.3 — Examples affected both families alike (agreed 2026-11-13)

#### Headline

> **Calibration examples affected both families alike: they shifted closed
> and open models toward "sound" by similar amounts (13 vs 15 points) and
> improved accuracy in neither.**

#### Expanded

> The calibration examples moved closed models 13.3 points and open models
> 15.1 points toward calling passages sound; the difference (−1.8 points,
> 95% CI −6.0 to 2.4) is not distinguishable from zero. Accuracy rose 2.7
> points in each family, a change not distinguishable from zero in either
> (pooled McNemar p = .067). The missed-flaw / wrongly-rejected ratio rose
> more for open models (+3.3 vs +1.6), but a ratio climbs faster from a more
> lenient starting point, so the share called sound is the fairer measure of
> whether the families responded differently. Effects are those of the one
> calibration set tested.

---

## Limitations specific to RQ4

- **Three models per family.** The CIs resample passages, not models, so
  they do not capture model-to-model variation within a family; the family
  gap may reflect particular models (e.g., GLM and Kimi are the most lenient
  overall; DeepSeek, an open model, was among the strictest without
  examples; Gemini, a closed model, was lenient without them).
- **Purposive sample** of top-performing models per family (see sample
  note); not generalizable to families as such.
- **Failure-mode coding exists only for the primary (closed) model**, so
  whether the *kinds* of failure (e.g., missing judgments, F5.1) vary by
  family cannot be examined. RQ4 can speak to accuracy, verdicts and
  leniency by family, not to failure modes by family.
- **One calibration set** (see 2.2 / 4.2): prompting effects are effects of
  that set.

---

## RQ4 answer (agreed 2026-11-13)

> **Model families did not differ in accuracy, but the open-source models were somewhat more lenient. The calibration examples increased leniency in both families alike, without significantly improving accuracy.**

Supporting findings: 4.1 (family), 4.2 (calibration examples) and 4.3
(the crossing), with 2.2 (leniency amplified by examples). "Somewhat" reflects three models per family; "calibration
examples" rather than "prompting" reflects the single calibration set.
