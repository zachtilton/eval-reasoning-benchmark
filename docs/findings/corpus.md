# Corpus finding — the reasoning being judged

Context for RQ1–RQ4: a finding about the UN evaluation conclusions
themselves, not about LLMs. Cited by conclusion C-a.

## Finding 0.1 — Missing judgments are common (agreed 2026-11-13)

### Headline

> **Missing judgments are common in UN evaluation conclusions: about a third
> of the sampled conclusion passages (32%, 95% CI 25–40%) discussed
> performance without ever saying how good it was.**

### Expanded

> The 150 passages are one conclusion passage per report, drawn at random
> from roughly 5,000 UN evaluation reports. By the expert's ratings, 74 were
> not sound; in 48 of them the stated reason was that no evaluative judgment
> was reached — 71% of not-sound passages with a recorded reason and 32% of
> all passages (95% CI 25–40%). If the six revised passages whose reasons
> were not recorded also lacked a judgment, the figure would be 36% (CI
> 29–44%), so "about a third" holds either way. This is consistent with
> empirical studies of evaluation reports that find evaluative conclusions
> frequently omitted or unsupported (Nunns et al., 2015; Hurteau et al., 2009).

### Qualifiers

- **Screening:** the sampling protocol excluded fragments that consisted
  "mainly of descriptive or prescriptive content" (proposal [218]), close
  cousins of judgment-absent passages; the estimate applies to screened,
  eligible conclusion passages and is likely conservative for UN conclusion
  sections as a whole.
- **Rater:** the classification rests on the expert's rationales and standard
  (stricter than the two RC2 raters; test-retest κ = 0.70).
- **Unit:** one passage per report; a report may make judgments elsewhere.
- **Precision:** 95% CI ±7 points (Wilson); sampling fraction ~3%, no
  finite-population correction needed.

### Evidence

- Classification: `data/analysis/gold_not_sound_reasons.csv` (phrase match on
  rationales, seven author-confirmed cases, buried and stale kept separate;
  author decisions 2026-11-13). Exploratory E25.
- Counts: judgment absent 48, other flaw 14, buried judgment 6, stale
  rationale 6.
- `results/findings_metrics.json` → `judgment_absent_rates`.
