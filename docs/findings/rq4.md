# RQ4 — Findings (in progress)

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
| Prompting: verdicts and leniency | Calibration examples shifted verdicts toward "sound" (142 of 156 changes; no gold standard needed), widening leniency from 1.5 to 3.8 against the expert's standard | 2.2, 2.3 |
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
- **One calibration set** (see 2.2 / 2.3): prompting effects are effects of
  that set.

---

## RQ4 answer

*(to be drafted by the author)*
