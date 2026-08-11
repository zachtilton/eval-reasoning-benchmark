# reliability/

Standalone rater-agreement analysis tools. Nothing in this folder is called
by, or depended on by, the orchestration pipeline (`src/`) — each script
here takes a CSV of ratings as input and is run manually.

## multi_rater_turing_test.py

Implements the Multi-Rater Agreement Statistical Turing Test from:

> Kljajic, J., O'Toole, J. M., Hogan, R., & Skoric, T. (2025). *Honest and
> reliable evaluation and expert equivalence testing of automated neonatal
> seizure detection.* arXiv:2508.04899.

It tests whether a candidate rater is statistically equivalent to a human
expert rater: bootstrap the human-only inter-rater agreement to get a
natural-noise margin, then check whether substituting the candidate into
the rater panel degrades agreement more than that margin allows.

Current use: validating the single-coder gold-standard anchor against the
two-person human rater panel (i.e., does the anchor hold up as a stand-in
for one of the two human raters?). The same script can later test an LLM's
outputs by putting them in the `--candidate` column instead — nothing else
about the procedure changes.

See the module docstring in `multi_rater_turing_test.py` for the full
step-by-step method and usage.

### Fidelity to the source paper

The decision rule this script implements (`delta-kappa 5th percentile >
-margin`) was verified directly against the authors' own reference
implementation, not just inferred from the paper's prose (which is
ambiguous about the sign of the comparison on its own):

- Repo: https://github.com/jovanak1/annotation-generation-and-expert-level-testing
- File: `expert-level-tests/Multi_Rater_Agreement_Statistical_Turing_Tests.py`
- Commit: `189ee476a9dfad07835180ddd003adab20f0b496`

This script implements only the paper's "Average kappa" variant. If the
"All raters" / "Majority raters" / "Any rater" variants are ever added: the
authors' own per-annotator code branch compares against `margin` (positive)
rather than `-margin` in two places, which is inconsistent with their own
docstring and with their "Average" branch (the variant this script
matches). That appears to be a sign-convention bug specific to their
per-annotator branch — use `-margin` for consistency with their documented
method and their "Average" branch, not the sign convention in that branch.

### Ratings CSV — no real names

If a ratings CSV is added to this folder, do not commit real names for
rater columns. Use generic column names (`rater_1`, `rater_2`, etc.)
instead of colleagues' names before committing.
