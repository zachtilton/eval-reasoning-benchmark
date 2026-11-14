# Conclusions — candidates (the "so what")

Structure (decided 2026-11-12): answers to research questions take the form
of observational findings; conclusions stand on their own and draw across
questions; recommendations follow. Conclusions are to be rated with the
evidence rubric (pending).

## Candidate conclusions

### C-a. Models carry a blind spot for a flaw common in real reports (kept 2026-11-12)
Models missed about half of the passages that never reach an evaluative
judgment (2.3), and such passages made up about a third of the sampled UN
conclusions (Finding 0.1). Empirical studies of evaluation reports find that omitted or
unsupported evaluative conclusions are common (Nunns et al., 2015; see also
Hurteau et al., 2009 on absent synthesis). Models used for meta-evaluation
would therefore carry a blind spot for one of the flaws a meta-evaluator
most needs to catch, which calls for added caution in using them for that
purpose.

- Grounds: 2.3 (F5.1, primary model; generalized via 2.1); 2.2 (leniency).
- Grounds (added 2026-11-13): corpus Finding 0.1 — about a third of sampled
  UN conclusion passages (32%, CI 25–40%) lack a judgment; models missed
  about half of them (51% zero-shot, 61% few-shot; E25).
- Qualifier: Nunns et al.'s sample (30 New Zealand reports) is narrower; the
  UN estimate rests on the expert's ratings of screened passages and is
  likely conservative (mainly descriptive fragments were excluded).
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
| C3 Robustness to irrelevant features | Substantial / moderate / comparable to humans / unaffected | Falls short. On length, models were swayed where the expert was not (one human comparator, who is also the yardstick; E16). On calibration examples, there's no human comparison (revised 2026-11-13) | Emergent (length, calibration-example analyses) |
| C4 Detecting an absent judgment | **Absolute** (revised 2026-11-13): routinely misses (most) / inconsistent (about half) / mostly reliable (a few) / reliable (rarely). Human raters are context, not the anchor | **Partial (inconsistent)**: models missed 51% of judgment-absent passages without examples and 61% with them, edging toward falls short; similar to other flaws (E25). Context: on 6 shared passages RC2 raters missed 42% (E26, too small to compare) | Pre-specified domain (Domain 5); emphasis and absolute standard post hoc |
| C5 Checklist evidence in rationales (artifacts) | Lacks elements / some, inconsistently / arguments an expert accepts / improve on the expert's | Insufficient evidence, promising signal (GPT failure rationales often persuaded the expert; strict ones nearly all defensible) | Pre-specified instrument; use as a standard post hoc |

### Safeguards for post hoc criteria (adopted 2026-11-13)

1. **Disclose.** Criteria and standards for the overall judgment were
   articulated after the analyses; provenance is shown above. C4's standard
   was first relative ("comparable to raters"). No rater benchmark existed
   (E26 found 6 shared passages), so it was changed to an absolute standard
   after the model results were known. Rationale (author, 2026-11-13): a
   conclusion that never reaches a judgment cannot count as evaluative
   reasoning, so detecting one is a requirement in its own right, not a
   comparison with humans.
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

### Domain 4 — Synthesis (first pass 2026-11-13; post hoc)

Method: qualitative weight and sum (Scriven; Davidson, 2005). Importance
levels and the synthesis rule were set by the author before they were applied.

| Criterion | Importance | Landing |
|---|---|---|
| C4 Detecting an absent judgment | Extremely important | Partial |
| C1 Verdict agreement | Very important | Human parity |
| C5 Checklist evidence in rationales | Very important | Insufficient evidence, promising signal |
| C2 Error structure | Important | Parity zero-shot; falls short few-shot |
| C3 Robustness to irrelevant features | Important | Falls short |

**Rule (weighted, no bar):**
- *Can do (human parity)*: parity on all extremely and very important
  criteria, and no "falls short" on any important one.
- *Partly can*: very important criteria at parity or partial; extremely
  important at least partial.
- *Cannot yet*: extremely important falls short, or very important fall short.

**Bar version:** C4 must reach "mostly reliable" (its third level; human
parity before the switch to an absolute standard) for the overall verdict
to exceed "partly can."

**Results:**
- Weighted: **partly can.** "Can do" fails on C4 (partial) and C3 (falls
  short). "Cannot yet" is not triggered: neither C4 nor C1 falls short.
  This holds in both conditions; examples move C2 to "falls short" and
  C4 toward it, but neither crosses a threshold in the rule.
- Bar: **partly can** (capped by C4 at partial).
- The two agree, so by safeguard 3 the post hoc emphasis on C4 does not
  drive the verdict.

**Open issues (author to decide):**
1. *C5 "insufficient evidence" is not a level the rule names.* As applied
   above it is left unrated, and the verdict rests on the other criteria. The
   alternative is to count the promising signal as partial. The verdict
   is the same either way.
2. *The agreement is partly structural.* As worded, the bar can only cap the
   verdict at "partly can," and the weighted rule already landed there.
   A bar that could have changed the result would be a hurdle: if C4 falls
   short, the verdict is "cannot yet." C4 clears that hurdle in both
   conditions, but only narrowly with examples (61% missed).
3. *The bar's validity (author's concern, 2026-11-13):* no single
   checkpoint is sufficient on its own. A bar claims only that C4 is
   necessary: a conclusion without a judgment cannot count as evaluative
   reasoning (author, 2026-11-13). The missing rater benchmark was
   resolved by switching C4 to an absolute standard (safeguard 1). The
   verdict is unchanged: partly can, both ways.

### Remaining steps
- Domain 2–3: link each criterion to its findings and warrants.
- Domain 5: explicit judgment — to what extent.
- Domain 6: qualifications; evidence-rubric rating.
