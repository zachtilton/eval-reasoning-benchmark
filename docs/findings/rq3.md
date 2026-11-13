# RQ3 — Workspace (author's interpretive work)

**Research question (verbatim, proposal):** *How do observed failure patterns
relate to evaluation theories of evaluative reasoning?* (summary wording:
"…known evaluation and reasoning theoretical frameworks?")

**Scope (decided 2026-11-12):** the proposal's named frameworks lead —
Scriven's general logic of evaluation and Nunns et al.'s (2015) variation;
JCSEE (Yarbrough et al., 2010); Toulmin; Hurteau et al. (2009); Gullickson
(2020); Meldrum & Gullickson (2025). Abductive extensions accompany
(verbosity bias, shortcut learning, in-context-learning literature;
`docs/references/literature_verified.md`). Procedural commitments (named-
framework coding, checklist as sensitizing concepts, emergent-code /
framework-adequacy analysis) are met by the failure coding; this file is
for the interpretive relating.

---

## Queued for interpretation

### Findings (from `rq1.md`, `rq2.md`)
| Finding | Headline (short) | Prompts for theory |
|---|---|---|
| RQ1 | ~60% agreement; ~as often as trained humans; lenient | What does "accurately" mean when qualified judges split ~1 in 4? |
| 2.1 | Same passages wrong; passage, not model | What property of a passage makes evaluative reasoning hard to judge? |
| 2.2 | Lenient; grows with length; examples amplify | Verbosity bias vs. "more checkpoints met": which does theory predict? |
| 2.3 | Examples shifted verdicts, not accuracy | What do examples transmit — a standard, or surface features? (Fournier: general vs. working logic) |
| 2.4 | Re-review changed the expert's mind | Argumentative capability; Toulmin warrants; first-pass vs. considered judgment |
| 2.5 | Missing the missing judgment (F5.1); examples worsened it | Evaluative veneer; description–evaluation gap |

### Named phenomena (conjecture unless stated)
- **Evaluative veneer** (2.5): surface of evaluation without the verdict.
- **Description–evaluation gap** (2.5): candidate link to Gullickson's (2020)
  use of Stake's "fully describe / fully judge"; revisit data before
  developing.
- **Argumentative capability** (2.4): a model rationale's capacity to move a
  qualified human's judgment; promise and risk.
- **First-pass leniency in humans and models** (2.4, RQ3 roadmap): the
  expert's reconsiderations moved the gold stricter twice; untested.

---

## Construct-validity questions (proposal; weak / null checks)

Source: `results/findings_metrics.json` → `theory_checks.construct_validity`.

1. **Synthesis (D4) harder than evidence (D2)?** Among 28 confirmed GPT
   failures: evidence codes in 13 fragments vs. synthesis in 10 (primary: 3
   vs. 3). Not borne out; tiny numbers.
2. **Few-shot helps tacit-judgment domain (D6)?** D6 codes in 1 of 15
   zero-shot vs. 4 of 27 few-shot confirmed rows — too few to answer (if
   anything, not a benefit).
3. **Challenge cases cluster on complex criteria (sustainability vs.
   relevance)?** Share of fragments meeting challenge Criterion 1:
   sustainability 54%, relevance 41% (impact 59%, coherence 57%, efficiency
   32%). Direction as expected, but the criterion check (2.2) found
   criterion explains nothing once the gold label is held constant.

## Stairway (Meldrum & Gullickson, 2025): propagation within a passage

Source: `theory_checks.stairway_within_passage`. Codes describe flaws the
primary model missed in a single passage, so this is within-passage only,
not report-level propagation.

- 20 of 28 confirmed fragments carry a conclusion-level (D5) code; **16 of
  those 20 also carry a lower-level code** (D1–D4); 4 are D5 alone.
- 14 fragments have D5 as the primary code with a lower-level secondary.
- 8 fragments have lower-level codes without D5.

Question for interpretation: does a conclusion-level failure that travels
with lower-level flaws read as propagation (Stairway), as the coder tracing
causes downward, or as passages that are weak throughout? What would
distinguish these?

---

## Space for the author's interpretation

*(to be written)*
