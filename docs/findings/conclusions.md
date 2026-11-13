# Conclusions — candidates (the "so what")

Structure (decided 2026-11-12): answers to research questions take the form
of observational findings; conclusions stand on their own and draw across
questions; recommendations follow. Conclusions are to be rated with the
evidence rubric (pending).

## Candidate conclusions

### C-a. Models carry a blind spot for a flaw common in real reports (kept 2026-11-12)
Models most often missed passages that never reach an evaluative judgment
(2.3). Empirical studies of evaluation reports find that omitted or
unsupported evaluative conclusions are common (Nunns et al., 2015; see also
Hurteau et al., 2009 on absent synthesis). Models used for meta-evaluation
would therefore carry a blind spot for one of the flaws a meta-evaluator
most needs to catch, which calls for added caution in using them for that
purpose.

- Grounds: 2.3 (F5.1, primary model; generalized via 2.1); 2.2 (leniency).
- Qualifier: Nunns et al.'s sample (30 New Zealand reports) is narrower than
  the UN corpus; prevalence in UN reports not estimated here.
- Rebuttal to address: F5.1 rests on one model's coded failures and one
  coder; RC4 pending.

---

## Title-level conclusion (in development, 2026-11-13)

**Question (study title, unstated as an RQ):** *To what extent can LLMs do
evaluative reasoning?* Operationalized as rating evaluative conclusions with
rationales (meta-evaluation as the vehicle; text as evidence). The
conclusion synthesizes across RQ1–RQ4 and is structured through the six
checklist domains ("dog-fooding"), so that it reaches an explicit judgment
(Domain 5) rather than committing the failure the study documents (F5.1).

**Scope (proposal, [162] and [135]):** claims concern reasoning
*artifacts*, not internal processes. The proposal names *epistemia* (Loru
et al., 2025) — "the appearance of coherent and authoritative judgment
[arising] from statistical patterning alone" — as the failure mode the study
is designed to detect. Related findings: length effect (2.2), evaluative
veneer (2.3), calibration-example shift (4.2); counter-signal: rationales
that held up on re-review (1.3).

### Domain 1 — Criteria and standards (proposed 2026-11-13; post hoc)

Levels: Falls short / Partial / Human parity / Expert level.

| Criterion | Standards (anchors) | Evidence lands | Provenance |
|---|---|---|---|
| C1 Verdict agreement with the expert | Chance (κ ≈ 0) / above chance, below raters / within raters' range (55–61% with expert's blind ratings) / expert's own consistency (85%, κ = 0.70) | Human parity (≈57%; κ ≈ 0.20 overall) | Pre-specified (primary outcome) |
| C2 Error structure | Strong one-way bias outside human range / at its edge / within it (raters 0.88, 1.40) / no lean | Parity without examples (0.85, short passages); falls short with them (3.05) | Emergent (leniency analyses) |
| C3 Robustness to irrelevant features | Substantial / moderate / comparable to humans / unaffected | Falls short (length; calibration examples); humans not tested | Emergent (length, calibration-example analyses) |
| C4 Detecting an absent judgment | Routinely misses / inconsistent / comparable to raters / reliable | Falls short to partial (F5.1 most common genuine failure; worse with examples); no rater comparison | Pre-specified domain (Domain 5); emphasis post hoc |
| C5 Checklist evidence in rationales (artifacts) | Lacks elements / some, inconsistently / arguments an expert accepts / improve on the expert's | Insufficient evidence, promising signal (GPT failure rationales often persuaded the expert; strict ones nearly all defensible) | Pre-specified instrument; use as a standard post hoc |

### Safeguards for post hoc criteria (adopted 2026-11-13)

1. **Disclose.** Criteria and standards for the overall judgment were
   articulated after the analyses; provenance is shown above.
2. **Ground in theory independently of results.** E.g., C4 rests on the
   checklist's Domain 5 basis, stated in the proposal before data:
   "Evaluative judgment is the endpoint — consolidation of all prior
   elements into an explicit statement of merit, worth, significance"
   [512].
3. **Report the overall verdict both ways:** with C4 as a bar (Davidson,
   2005: a minimum that must be cleared regardless of performance
   elsewhere) and with no bars (a weighted profile). If the two agree, the
   post hoc choice matters little; if they differ, the dependence is
   visible.

### Remaining steps
- Domain 2–3: link each criterion to its findings and warrants.
- Domain 4: synthesis, both ways (bar / no bar).
- Domain 5: explicit judgment — to what extent.
- Domain 6: qualifications; evidence-rubric rating.
