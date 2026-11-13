# Results roadmap (working, 2026-11-12)

## Structure
One topline conclusion per RQ, supported by findings; 2–3 supporting findings per RQ on the poster.

## RQ1 — done
`rq1.md`. Open: revisit the mainline leniency clause in light of RQ2.2
(zero-shot vs few-shot split).

## RQ2 — in progress (`rq2.md`)
- [x] 2.1 Shared failures
- [x] 2.2 Leniency (headline settled: leniency / length / calibration examples, three sentences)
- [x] 2.3 Calibration examples shifted verdicts, not accuracy
- [x] #4 Failure by evaluation criterion — folded into 2.2 (criterion explains nothing once gold label is held constant)
- [x] Passage-length moderation — folded into 2.2 (leniency grows with length; explains RC2-subset reversal)
- [x] #5 / #9 absorbed into 2.1, with a bridging sentence: GPT's coded flaws likely describe what other models missed (shared failures); reasoning coded for GPT only
- [x] 2.4 Disposition: re-review changed the expert's mind (#6, #7); named concept: argumentative capability
- [x] 2.5 Missing the missing judgment (F5.1; examples made it worse); named phenomena: evaluative veneer (preferred), description–evaluation gap
- [ ] RQ2 mainline answer + conclusion-level rating

## RQ3 — workspace started (`rq3.md`); conclusions candidates in `conclusions.md`
**Scope decision (2026-11-12):** the proposal's named frameworks lead (Scriven's general logic and Nunns et al.'s variation; JCSEE; Toulmin; Hurteau et al.; Gullickson; Meldrum & Gullickson), with abductive extensions accompanying (verbosity bias, shortcut learning, in-context-learning literature). Proposal commitments 1–3 (named-framework interpretation, checklist as sensitizing concepts, emergent-code / framework-adequacy analysis) are met procedurally by the coding; the interpretive relating to theory remains to be written. Construct-validity questions (D4 vs D2; few-shot and D6; criterion clustering) to be reported as weak/null checks. Candidate connections: Gullickson's (Stake's) "fully describe / fully judge" for the description–evaluation gap (conjecture; revisit data); Nunns et al. on omitted conclusions as a practical stake for meta-evaluation; Stairway propagation within a passage (suggestive only).
Develop the named phenomena from RQ2: **evaluative veneer** (2.5; links to verbosity bias, shortcut learning), the **description–evaluation gap** (Scriven's logic of evaluation), and **argumentative capability** (2.4).
Domain profile and theory linkage; **parked lexicon test**: do surface
evaluative features (evaluative-term density, rating scales, criterion
references; pre-specified) predict models' sound calls beyond length and
gold label? Hypothesis about impact conclusions being weaker (gold 71% vs
33% relevance; suggestive only); framework adequacy (0 emergent codes);
"looks evaluative" mechanism hypothesis; human-and-model first-pass leniency hypothesis (expert's reconsiderations moved the gold stricter twice; MRST gap narrowed after the stricter calibration) — conjecture needing testing; contested reference standard;
proposal construct-validity questions (Domain 4 vs 2; few-shot and Domain 6;
challenge cases by criterion). Mostly interpretive writing by the author.

## RQ4 — not started
Prompting (largely 2.2/2.3); model family (H2 chi-squared, closed vs open
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
