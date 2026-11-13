# Results roadmap (working, 2026-11-12)

## Structure
Corpus finding 0.1 (`corpus.md`, context before the RQs): about a third of sampled UN conclusion passages lack a judgment.
One topline conclusion per RQ, supported by findings; 2–3 supporting findings per RQ on the poster.

## RQ1 — done (answer + findings 1.1–1.3; 2.4 re-homed as 1.3, 2026-11-13)
`rq1.md`. Resolved 2026-11-13: the leniency clause stays as a pooled statement (≈2.3:1, CI 1.7–3.1); RQ2's answer carries the length and calibration-example nuance.

## RQ2 — in progress (`rq2.md`)
- [x] 2.1 Shared failures
- [x] 2.2 Leniency (headline settled: leniency / length / calibration examples, three sentences)
- [x] Former 2.3 Calibration examples → moved to RQ4 as 4.2 (2026-11-13)
- [x] #4 Failure by evaluation criterion — folded into 2.2 (criterion explains nothing once gold label is held constant)
- [x] Passage-length moderation — folded into 2.2 (leniency grows with length; explains RC2-subset reversal)
- [x] #5 / #9 absorbed into 2.1, with a bridging sentence: GPT's coded flaws likely describe what other models missed (shared failures); reasoning coded for GPT only
- [x] 2.4 Disposition → re-homed to RQ1 as 1.3 (2026-11-13); named concept: argumentative capability
- [x] 2.3 (formerly 2.5, then 2.4) Missing the missing judgment (F5.1; examples made it worse); named phenomena: evaluative veneer (preferred), description–evaluation gap
- [x] RQ2 answer agreed (2026-11-13): lenient, esp. longer passages and calibration examples, most often missed judgments; supporting 2.2, 2.3

## RQ3 — tabled (2026-11-13); provisional answer in `rq3.md`; Stairway marked dead end; conclusions candidates in `conclusions.md`
**Scope decision (2026-11-12):** the proposal's named frameworks lead (Scriven's general logic and Nunns et al.'s variation; JCSEE; Toulmin; Hurteau et al.; Gullickson; Meldrum & Gullickson), with abductive extensions accompanying (verbosity bias, shortcut learning, in-context-learning literature). Proposal commitments 1–3 (named-framework interpretation, checklist as sensitizing concepts, emergent-code / framework-adequacy analysis) are met procedurally by the coding; the interpretive relating to theory remains to be written. Construct-validity questions (D4 vs D2; few-shot and D6; criterion clustering) to be reported as weak/null checks. Candidate connections: Gullickson's (Stake's) "fully describe / fully judge" for the description–evaluation gap (conjecture; revisit data); Nunns et al. on omitted conclusions as a practical stake for meta-evaluation; Stairway propagation within a passage (suggestive only).
Develop the named phenomena from RQ2: **evaluative veneer** (2.3; links to verbosity bias, shortcut learning), the **description–evaluation gap** (Scriven's logic of evaluation), and **argumentative capability** (1.3).
Domain profile and theory linkage; **parked lexicon test**: do surface
evaluative features (evaluative-term density, rating scales, criterion
references; pre-specified) predict models' sound calls beyond length and
gold label? Hypothesis about impact conclusions being weaker (gold 71% vs
33% relevance; suggestive only); framework adequacy (0 emergent codes);
"looks evaluative" mechanism hypothesis; human-and-model first-pass leniency hypothesis (expert's reconsiderations moved the gold stricter twice; MRST gap narrowed after the stricter calibration) — conjecture needing testing; contested reference standard;
proposal construct-validity questions (Domain 4 vs 2; few-shot and Domain 6;
challenge cases by criterion). Mostly interpretive writing by the author.

## RQ4 — answer agreed 2026-11-13 (`rq4.md`); findings 4.1 family, 4.2 calibration examples (former 2.3), 4.3 crossing
Prompting (2.2, 4.2); model family (H2 chi-squared, closed vs open
z-test); check whether leniency aligns with family; conclusion and rating.

## Reliability
- [ ] **RC4** failure-coding temporal consistency (proposal: 20% of GPT
  failures, blind re-code after ≥ 1 week, Cohen's κ ≥ 0.80). Decide: unit
  (rows / 76 fragments / 28 coded), layers (disposition and/or codes),
  κ basis (primary code, domain). Earliest date: ~2026-11-18 (last coding
  session 2026-11-11).
  **Persuasion check (added 2026-11-12):** re-rate a sample blind to BOTH the
  gold label and GPT's rationale, including GPT passes as well as failures.
  Tests (a) stability of codes/dispositions, (b) whether re-review
  dispositions were driven by exposure to GPT's rationale, and (c) gold
  errors on the side never re-reviewed (passes), which bias the 14%
  confirmed failure rate downward.

## Evidence rubric — pinned
Adapted rubric with IPCC summary layer (evidence × agreement → low / medium
/ high confidence); five dimensions as the traceable account (transparency
& precision, reference independence, consistency across checks, rival
explanations, scope); "rate the claim as worded." For now apply at the
conclusion level per RQ, plus the 2–3 load-bearing findings. To do: verify
Mastrandrea et al. (2010); write descriptors and the evidence × agreement
rule; add bootstrap CIs for the leniency ratios.

## Findings review (author request, 2026-11-13)
- [x] Headline review done (2026-11-13): implication-first headlines applied to 1.1, 1.2, 2.1–2.4 (1.3 already); former 2.5 renumbered 2.4, then 2.3 when the former 2.3 moved to RQ4 as 4.2.

## Conclusions (in progress, `conclusions.md`)
- [x] C-a blind spot for missing judgments (Nunns)
- [ ] Title-level conclusion via the six checklist domains: Domain 1 criteria/standards proposed (post hoc; safeguards adopted: disclose provenance, ground in theory, report verdict with and without C4 as a bar); Domains 2–6 to do
- [ ] Cluster conclusions (models as judges incl. bias vs noise / correctability — P1 found leniency shiftable, accuracy gain suggestive only; the yardstick; practitioner choices)
- [ ] Verify Dell'Acqua et al. (2023) jagged frontier, Kahneman et al. (2021) Noise, Turpin et al. (2023), Loru et al. (2025)

## After findings
Per-RQ conclusions (rated); study-level limitations (ethics, representation,
design); poster (one headline + one figure per RQ; new figures for RQ1/RQ2).

## Post-poster extensions
- Automated coding of all six models' rationales against Appendix D codes,
  validated against the author's GPT coding (κ); tests whether models miss
  the same flaws for the same reasons. Risks: LLM-codes-LLM circularity,
  small single-coder validation set.
- Alternative calibration sets for the few-shot condition (stimulus
  sampling), ~$50 per variant across all models.

## Documentation
- [ ] Backfill `analysis_log.md` with unplanned work from earlier sessions
  (e.g., external-reviewer triangulation of the 35 contested fragments,
  blind Haiku gold-label audit, pre-lock rationale restoration, Luna
  triage of ambiguous runs, corrected-scoring decision record).

## Housekeeping (not poster-critical)
- [x] RC numbering fixed in repo (2026-11-12): coherence references RC2 → RC3 in `src/scoring/*`, `coding_synthesis.py`, `runner.py`, `meta_evaluation.py` (RC1/RC3/RC4 κ checks; RC2 MRST separate), `.gitignore`, `CLAUDE.md`. Remaining: proposal paragraph on failure-coding recode (says RC3) — author to update.
- Proposal text says 4 calibration fragments; study used 2 — author to update proposal.
Rename RC2 → RC3 in code references (`corrected_scoring.py`, `*_corrected`).
