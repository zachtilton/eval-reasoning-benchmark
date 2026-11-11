# Appendix D: Failure Mode Coding Schema

Extracted verbatim from `docs/tilton_dissertation_proposal-26-11-7.docx` (Appendix D,
2026-11-07) so the coding schema is available without reopening the Word
doc. If the proposal's Appendix D is ever revised, re-extract from the
proposal rather than editing this file independently — it should stay a
faithful copy, not a second source of truth.

Used for qualitative coding of GPT 5.6 Sol's Type 1 failures (coherent but
incorrect), per spec Section 3.1. Scope: GPT 5.6 Sol only — comparison
models receive aggregate pass/fail rates and initial failure codes only,
not individual coding.

Official output schema: `data/analysis/failure_codes_gpt5.csv` — columns
`fragment_id, prompt_condition, representative_run, primary_code,
secondary_code, tertiary_code, domain, code_type, coding_justification,
coding_date, coding_session`. `domain` is 1-6 for deductive codes, 0 for
emergent (D.2) codes. `code_type` is `deductive` or `inductive`.

## Appendix D.1: Failure Coding Schema (Inverted Integrated Evaluative Reasoning Checklist)

Deductive codes — the 22 checkpoints from the Integrated Evaluative
Reasoning Checklist (Appendix A), inverted to describe how each checkpoint
*fails* rather than how it is satisfied.

### Domain 1 (Evaluative Framing): Checkpoints 1-3

| Code | Checkpoint failure |
|---|---|
| F1.1 | Checkpoint 1 failure (evaluative vs. descriptive confusion) |
| F1.2 | Checkpoint 2 failure (criteria/comparators absent or unjustified) |
| F1.3 | Checkpoint 3 failure (standards not articulated) |

### Domain 2 (Evidence Base): Checkpoints 4-6

| Code | Checkpoint failure |
|---|---|
| F2.1 | Checkpoint 4 failure (performance evidence absent) |
| F2.2 | Checkpoint 5 failure (evidence insufficient or irrelevant) |
| F2.3 | Checkpoint 6 failure (evidence base indefensible) |

### Domain 3 (Argument Structure): Checkpoints 7-10

| Code | Checkpoint failure |
|---|---|
| F3.1 | Checkpoint 7 failure (reasoning chain unclear) |
| F3.2 | Checkpoint 8 failure (warrants implicit or absent) |
| F3.3 | Checkpoint 9 failure (backing for warrants missing) |
| F3.4 | Checkpoint 10 failure (inappropriate authority appeal) |

### Domain 4 (Synthesis & Integration): Checkpoints 11-13

| Code | Checkpoint failure |
|---|---|
| F4.1 | Checkpoint 11 failure (criteria and evidence not integrated) |
| F4.2 | Checkpoint 12 failure (multiple dimensions not integrated) |
| F4.3 | Checkpoint 13 failure (algorithmic rather than synthetic judgment) |

### Domain 5 (Evaluative Conclusion): Checkpoints 14-16

| Code | Checkpoint failure |
|---|---|
| F5.1 | Checkpoint 14 failure (evaluative judgment absent) |
| F5.2 | Checkpoint 15 failure (conclusion doesn't culminate reasoning) |
| F5.3 | Checkpoint 16 failure (incoherent element connections) |

### Domain 6 (Qualifications & Transparency): Checkpoints 17-22

| Code | Checkpoint failure |
|---|---|
| F6.1 | Checkpoint 17 failure (qualifiers inappropriate for certainty) |
| F6.2 | Checkpoint 18 failure (conditions/boundaries not acknowledged) |
| F6.3 | Checkpoint 19 failure (unwarranted overgeneralization) |
| F6.4 | Checkpoint 20 failure (assumptions/values not explicit) |
| F6.5 | Checkpoint 21 failure (limitations not acknowledged) |
| F6.6 | Checkpoint 22 failure (alternative interpretations not considered) |

## Appendix D.2: Emergent Code Log Template

Purpose: document failure patterns not captured by the 22-checkpoint
framework. Inductive codes — `domain = 0`, `code_type = inductive` in the
output schema.

For each emergent code:

- **Code:** [Brief label, e.g., "Circular Reasoning"]
- **Definition:** [1-2 sentences describing the failure pattern]
- **Why checkpoints are insufficient:** [1 sentence]
- **Example fragment(s):** [Fragment ID(s) where this appears]
- **Frequency:** [X instances / Y% of total failures]

(Repeat for each emergent code.)

## Appendix D.3: Coding Memo Template

Purpose: track coding decisions and pattern observations across sessions.

Per session:

- **Date:**
- **Fragments coded:** [IDs or range, e.g., F_015-F_023]
- **Key patterns observed:** [What failure types dominated? Any co-occurrences?]
- **Difficult coding decisions:**
  - Fragment [ID]: [What was ambiguous? How resolved?]
- **Potential emergent patterns:** [Anything not fitting checkpoint framework? Track here before formalizing in D.2]
- **Notes:** [Any other observations about failures, consistency, or the coding process]

(New entry for each coding session.)

### Final Summary (written after all coding)

- **Saturation:** [Yes/No — did new patterns stop emerging?]
- **Framework adequacy:** [Did 22 checkpoints capture most failures?]
- **Total emergent codes:** [X codes developed]
- **Key insights:** [2-3 sentences on main takeaways about GPT 5.6 Sol's reasoning breakdowns]
