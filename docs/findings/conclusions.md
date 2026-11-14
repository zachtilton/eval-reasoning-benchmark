# Conclusions — candidates (the "so what")

Structure (decided 2026-11-12): answers to research questions take the form
of observational findings; conclusions stand on their own and draw across
questions; recommendations follow. Conclusions are to be rated with the
evidence rubric (pending).

## Candidate conclusions

### C-a. Models would pass a flaw common in real reports (kept 2026-11-12; reframed 2026-11-13)
Models missed about half of the passages that never reach an evaluative
judgment (2.3), and such passages made up about a third of the sampled UN
conclusions (Finding 0.1). Empirical studies of evaluation reports find that omitted or
unsupported evaluative conclusions are common (Nunns et al., 2015; see also
Hurteau et al., 2009 on absent synthesis). Models missed these passages
about as often as other flaws (E25); the consequence comes from how common
the flaw is, not from special susceptibility to it. Models used for
meta-evaluation would therefore pass many conclusions that never reach a
judgment, one of the flaws a meta-evaluator most needs to catch, which
calls for added caution in using them for that purpose. ("Blind spot"
wording dropped 2026-11-13: it implied special susceptibility.)

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

Levels (combined labels, adopted 2026-11-13): **Poor** (falls short) /
**Fair** (partial) / **Good** (human parity) / **Excellent** (expert level).
The same four-level scale is used for each criterion and for the overall
verdict. Each level keeps its descriptor as its anchor (e.g., good = within
the human raters' range); for C4, whose standard is absolute, the levels
order its four anchors. Always show the scale beside a rating.

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
   visible. *Updated 2026-11-13:* the bar version is now **primary**,
   because the bridging warrant gives it a logical basis; the weighted
   version is reported as a sensitivity check.

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

**Rule (weighted, no bar):** overall levels relabeled 2026-11-13 (cannot
yet → poor; partly can → fair; can do → good; excellent added).
- *Excellent (expert level)*: expert level on all extremely and very
  important criteria, and at least good on every important one.
- *Good (human parity)*: good or better on all extremely and very important
  criteria, and none poor on any important one.
- *Fair (partial)*: very important criteria at good or fair; extremely
  important at least fair.
- *Poor (falls short)*: extremely important poor, or very important poor.

**Bar version:** C4 must reach good ("mostly reliable", its third level)
for the overall verdict to exceed fair.

**Results (bar primary, weighted as sensitivity check; order set 2026-11-13):**
- **Primary, with C4 as a bar: fair.** C4 is at fair, so the verdict cannot
  exceed fair, whatever the other criteria show. Rationale: the bridging
  warrant makes detecting an absent judgment a requirement for judging
  evaluative reasoning, not one weighted consideration among others.
- **Sensitivity check, weighted: fair.** "Good" fails on C4 (fair) and C3
  (poor). "Poor" is not triggered, since neither C4 nor C1 is poor. This
  holds in both conditions: examples move C2 to poor and C4 toward it,
  but neither crosses a threshold in the rule.
- The two agree, but the agreement is partly structural (open issue 2):
  the bar did not bind, because the weighted rule already landed at fair.
  The bar would bind if models reached good on the other criteria while
  C4 stayed at fair; weighting could then rate them good, but the bar
  would hold them at fair. It also states the threshold for "good":
  models must reliably catch missing judgments.

**Open issues (author to decide):**
1. *C5 "insufficient evidence" is not a level the rule names.* As applied
   above it is left unrated, and the verdict rests on the other criteria. The
   alternative is to count the promising signal as partial. The verdict
   is the same either way.
2. *The agreement is partly structural.* As worded, the bar can only cap the
   verdict at fair, and the weighted rule already landed there.
   A bar that could have changed the result would be a hurdle: if C4 falls
   short, the verdict is poor. C4 clears that hurdle in both
   conditions, but only narrowly with examples (61% missed).
3. *The bar's validity (author's concern, 2026-11-13):* no single
   checkpoint is sufficient on its own. A bar claims only that C4 is
   necessary: a conclusion without a judgment cannot count as evaluative
   reasoning (author, 2026-11-13). The missing rater benchmark was
   resolved by switching C4 to an absolute standard (safeguard 1). The
   verdict is unchanged: fair, both ways.

### Bridging warrant for C4 as a bar (agreed 2026-11-13; logical, not empirical)

> Judging evaluative reasoning requires knowing what evaluative reasoning
> must contain. A judgment is its required endpoint, so a judge who cannot
> tell whether one is present does not yet know what they are judging.

- Holds for the narrow claim (models *as judges* of evaluative
  conclusions). Fails for the broad claim (evaluative reasoning in general):
  producing a judgment and recognizing an absent one are separable abilities
  (author's counterexample, 2026-11-13). Hence the scoped Domain 5 wording.
- Passage-level premise (author): a conclusion that never reaches a judgment
  cannot count as evaluative reasoning.

### Domain 5 — Judgment (recorded 2026-11-13)

> **Frontier LLMs' evaluative reasoning, as judges of evaluative conclusions,
> rated fair (poor–fair–good–excellent):
> they matched trained evaluators' modest agreement with an expert, but
> passed about half of the conclusions the expert judged flawed, most often
> ones that never reach a judgment. Their errors leaned lenient, more so on
> longer passages and with calibration examples.**

Drafting notes:
- "As judges of evaluative conclusions" (added 2026-11-13): the study tested
  models judging others' conclusions, not producing their own. Whether they
  can produce sound evaluative reasoning was not tested (Domain 6).
- "Frontier": three closed and three open models chosen as the top of each
  family at the time of the run (rq4.md sample note).
- "Modest": the two trained evaluators agreed with the expert 55–61% of
  the time (C1).
- "About half … most often ones that never reach a judgment": models missed
  flaws generally at similar rates (E25); missing judgments are the most
  common flaw (48 of 74 not-sound passages), so most misses fall there.
  About half without examples, about 6 in 10 with them.
- Left to the findings: balanced errors on short passages without examples
  (C2 good); the split by condition.
- Moderators (length, calibration examples) are content; the limits on
  them (one calibration set; one human comparator for length) go to
  Domain 6.

### Domain 6 — Qualifications (tiers agreed 2026-11-13)

All qualifications are reported (transparency); the tiers set where each
appears and how much space it gets. Ranking also feeds the evidence-rubric
rating. Author's top two: #1 and #5 (scope, for its clause on reasoning not
assessed); Claude's: #1 and #8. Agreed: #1 and "reasoning rated from
verdicts" in Tier 2; #8 heads Tier 3.

**Tier 1 — in the judgment itself:** "frontier"; "as judges of evaluative
conclusions"; "the expert judged flawed."

**Tier 2 — beside the verdict (poster; immediately after the judgment):**
1. **The yardstick is one expert's contested standard.** Re-review
   overturned most of the primary model's apparent failures, so the
   benchmark likely understates the models by an unknown amount.
2. **Reasoning quality was rated from verdicts, not rationales.** Whether
   models reasoned well, as distinct from concluding correctly, was not
   systematically assessed; the only rationale analysis was the review of
   GPT's failures, which produced qualification 1.

**Tier 3 — full list (Domain 6 / limitations chapter), with likely bearing on
"fair":**

| # | Qualification | Likely direction | Could it change "fair"? |
|---|---|---|---|
| 8 | Criteria, weights, bar and C4's absolute standard set after results (safeguard 1) | Unknown | Possibly: a framework set in advance might land elsewhere |
| 1 | Contested yardstick (Tier 2) | Understates models on strict calls; unknown where models agreed with a mistaken label (never reviewed) | Unlikely under the bar: C4 rests on lenient misses, and the expert upheld most lenient failures (27 of 43) |
| 5 | Scope: six frontier models, one prompt, UN conclusion passages; producing evaluative reasoning and rationale quality not assessed (Tier 2) | Unknown | Not for what was tested; limits generalization. If rationales were rated, C5 could lower the verdict (poor on a very important criterion), not raise it (the bar caps at fair) |
| 7 | C5 insufficient evidence; persuasion check likely after the poster | Unknown | See #5 |
| 2 | One calibration set | Few-shot effects specific to that set | Unlikely: verdict is fair in both conditions |
| 3 | Length: only human comparator is the expert, who is also the yardstick | Unknown | Unlikely: affects C3 only |
| 4 | Judgment-absent classification (E25) by one coder | Unknown | Unlikely: models missed other flaws at similar rates |
| 6 | Human comparison rests on two raters and 38 short passages | Unknown | Unlikely: C1 would need to fall to poor |

### Evidence rubric (scaffold agreed 2026-11-13)

Summary layer: IPCC evidence × agreement → confidence (Mastrandrea et al.,
2010). Rate the claim as worded, **by parts** (a claim made of several parts
is no more certain than its weakest part).

| Dimension | Strong | Moderate | Weak |
|---|---|---|---|
| Transparency & precision | Pre-specified, or post hoc and logged and disclosed; uncertainty quantified, with precision adequate to the claim as worded | Logged and disclosed, but uncertainty unquantified, or quantified but too imprecise for the claim as worded | Not logged or disclosed; uncertainty unquantified |
| Reference independence | Holds against several independent references (locked gold, blind pass 2, RC2 raters, third reviewer) | Holds against more than one, but they share a source | Rests on a single reference |
| Consistency across checks | Same direction across models, conditions, analyses | Mostly consistent, exceptions explained | Mixed or contradicted |
| Rival explanations | Plausible rivals tested and ruled out | Some tested; others named, untested | Unaddressed |
| Scope | Within what was sampled | Slight stretch, signalled in wording | Beyond the sample |

Descriptor revision (author, 2026-11-13): post hoc work is not
discredited outright. The proposal claims an abductive approach, and lines
of inquiry emerge as results are interpreted, so logged and disclosed post
hoc work can rate strong. Precision is judged against the claim's wording:
"matched" needs a tight interval around zero, while "no detectable
difference" tolerates a wide one.

- **Evidence** (limited / medium / robust) ← transparency & precision,
  rival explanations, scope.
- **Agreement** (low / medium / high) ← reference independence,
  consistency across checks.
- **Confidence = the lower of the two** (conservative; IPCC leaves the
  mapping partly to judgment).

Parts of the Domain 5 judgment: (a) fair overall; (b) matched trained
evaluators' agreement; (c) passed about half of conclusions the expert
judged flawed, mostly judgment-absent ones; (d) leaned lenient, more so
with length and calibration examples. Ratings: *(author, pending)*.

### Remaining steps
- Domain 2–3: link each criterion to its findings and warrants.
- Domain 6: write up Tier 3 entries in prose; evidence-rubric rating.
