"""
Scoring-layer configuration: indicator lists, LLM prompt, path constants,
and pipeline parameters.

All constants used across G.4–G.6 scoring modules are centralized here so
that configuration is separated from logic (CLAUDE.md code standards).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
GOLD_STANDARD_DIR = DATA_DIR / "gold_standard"
RESPONSES_DIR = DATA_DIR / "responses"
LOGS_DIR = PROJECT_ROOT / "logs"

# Output file paths
RESPONSE_DB_PATH = RESPONSES_DIR / "response_database.csv"
SCORED_DB_PATH = RESPONSES_DIR / "scored_database.csv"
FRAGMENT_OUTCOMES_PATH = RESPONSES_DIR / "fragment_outcomes.csv"
RC3_SAMPLE_PATH = LOGS_DIR / "rc3_manual_review_sample.csv"

# ---------------------------------------------------------------------------
# RC3-corrected scoring (provisional-final; see src/scoring/corrected_scoring.py)
# ---------------------------------------------------------------------------
# RC3's blind human review (2026-11-05, logs/human_review_blinded.csv +
# human_review_key.csv) found 0% agreement between the automated coherence
# pipeline's "incoherent"/"ambiguous" calls and human judgment across every
# tested sub-stratum (incoherent_confident 64/64 overridden, all three
# ambiguous sub-groups 153/153 overridden) -- including the 7 cases where
# Tier 2 (Haiku) independently agreed with Tier 1, which were overridden too.
# Only the coherent_confident bucket held up (94/94). No sub-component of
# the pipeline's "not coherent" determination survived validation, so these
# outputs stop using coherence_final as a scoring gate (Step 2) and instead
# score classification accuracy for every response. Written to separate
# *_corrected paths -- the original scored_database.csv / fragment_outcomes
# .csv / performance_matrix.csv are left untouched as the as-designed,
# pre-RC3 historical results.
SCORED_DB_CORRECTED_PATH = RESPONSES_DIR / "scored_database_corrected.csv"
FRAGMENT_OUTCOMES_CORRECTED_PATH = RESPONSES_DIR / "fragment_outcomes_corrected.csv"
PERFORMANCE_MATRIX_CORRECTED_PATH = RESPONSES_DIR / "performance_matrix_corrected.csv"

# ---------------------------------------------------------------------------
# Coherence validation parameters (spec Section 2.4, Step 1)
# ---------------------------------------------------------------------------

# Minimum indicator difference for Tier 1 to be considered "confident."
# If |strength_count - weakness_count| >= threshold → assign automatically.
# If < threshold → route to Tier 2 LLM screen.
CONFIDENCE_THRESHOLD: int = 3

# Haiku model used for Tier 2 LLM coherence screening (temperature 0).
# Must match the claude-haiku model ID current at execution time.
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"

# Keep in sync with src/api_client/config.ANTHROPIC_API_VERSION
ANTHROPIC_API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Triangulation check (supplementary — not part of the official scoring
# pipeline; see src/scoring/triangulation.py)
# ---------------------------------------------------------------------------
# Deliberately NOT claude-haiku-4-5-20251001 (already Tier 2) and NOT
# gpt_5_6_sol (one of the 6 benchmarked models under study — using it would
# have a model grade its own study's outputs). gpt-5.6-luna is the cheapest
# tier of the same model generation as the primary diagnostic model, from a
# vendor entirely outside Tier 2's Anthropic lineage — maximizes
# independence from both the existing automated judge and the subject
# models. CONFIRMED live (2026-11-04): /v1/responses, 200 OK, clean
# single-word output, 0 reasoning tokens on a trivial coherence judgment.
TRIANGULATION_MODEL_ID = "gpt-5.6-luna"
TRIANGULATION_ENDPOINT = "https://api.openai.com/v1/responses"
TRIANGULATION_RESULTS_PATH = RESPONSES_DIR / "triangulation_luna_ambiguous.csv"

# ---------------------------------------------------------------------------
# Secondary (label-substitution) analyses -- see reviewer_substitution.py.
# Never overwrite the locked gold standard or the *_corrected primary outputs.
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
SCORED_DB_CORRECTED_INPUT_PATH = RESPONSES_DIR / "scored_database_corrected.csv"
CONTESTED_REVIEW_RESOLVED_PATH = RESPONSES_DIR / "contested_fragments_review_resolved.csv"
SCORED_DB_TRIANGULATED_PATH = RESPONSES_DIR / "scored_database_triangulated.csv"
FRAGMENT_OUTCOMES_TRIANGULATED_PATH = RESPONSES_DIR / "fragment_outcomes_triangulated.csv"
H2_TRIANGULATED_RESULTS_PATH = RESULTS_DIR / "H2_statistical_tests_triangulated.json"

# ---------------------------------------------------------------------------
# RC3 stratified sample sizes (spec Section 2.4, Step 1)
# ---------------------------------------------------------------------------

# RAISED from the original spec's flat 20/20/10, 2026-11-04: those sizes
# give a ~±22pp worst-case margin of error (95% CI, p=0.5) on each bucket's
# population (coherent N=3,967; incoherent N=216; ambiguous N=1,217) -- not
# enough precision to support a validity claim. Sized instead to a ~±10pp
# margin (95% CI, p=0.5, finite-population-corrected):
#   coherent:   n = 1.96^2*0.25/0.01, FPC-adjusted for N=3,967  -> ~94
#   incoherent: n = 1.96^2*0.25/0.01, FPC-adjusted for N=216    -> ~67
# The ambiguous arm is no longer a flat random draw of RC3_AMBIGUOUS_MAX --
# see triangulation.py::select_stratified_human_review_sample, which
# stratifies it using the Luna triangulation triage (full census of the
# 64 three-way-split cases + a precision-sized proportional sample of the
# rest) instead of treating "ambiguous" as one undifferentiated bucket.
RC3_COHERENT_SAMPLE = 94
RC3_INCOHERENT_SAMPLE = 67
# Kept for select_rc3_sample's original spec-literal behavior (and its
# existing tests) -- not used by the triangulation-informed ambiguous
# sampling actually run in practice; see comment above.
RC3_AMBIGUOUS_MAX = 10    # "or all ambiguous if < 10"
RC3_KAPPA_TARGET = 0.80

# ---------------------------------------------------------------------------
# Primary diagnostic model (spec Section 2.2)
# ---------------------------------------------------------------------------

# RENAMED 2026-11-02: was "gpt_5" (the proposal-stage model generation name).
# Must match the real model_family value in response_database.csv -- see
# src/analysis/config.py's ARCHITECTURE_MAP comment for the full context.
PRIMARY_MODEL = "gpt_5_6_sol"   # full diagnostic treatment: performance + failure coding

# ---------------------------------------------------------------------------
# Strength indicators — signals of quality for "sound" classifications
# (spec Section 2.4, Tier 1; Appendix G.4)
# ---------------------------------------------------------------------------

STRENGTH_INDICATORS: list[str] = [
    # Affirmations of reasoning quality
    "clear", "strong", "robust", "comprehensive", "thorough", "well-supported",
    "effective", "appropriate", "adequate", "sufficient", "sound", "valid",
    "defensible", "convincing", "compelling", "coherent", "logical",
    # Satisfied checkpoints
    "establishes", "demonstrates", "provides", "includes", "presents",
    "addresses", "considers", "integrates", "synthesizes", "acknowledges",
    "articulates", "specifies", "identifies", "supports",
    # Positive evaluative language
    "successfully", "effectively", "appropriately", "clearly", "explicitly",
    "adequately", "properly", "satisfactorily", "well", "good",
]

# ---------------------------------------------------------------------------
# Weakness indicators — signals of flaws for "not sound" classifications
# (spec Section 2.4, Tier 1; Appendix G.4)
# ---------------------------------------------------------------------------

WEAKNESS_INDICATORS: list[str] = [
    # Violated or missing checkpoints
    "missing", "absent", "lacks", "fails to", "does not", "omits",
    "neglects", "ignores", "overlooks", "insufficient", "inadequate",
    "incomplete", "unclear", "vague", "ambiguous",
    # Reasoning flaws
    "weak", "unsupported", "unsubstantiated", "unjustified", "unfounded",
    "problematic", "flawed", "inconsistent", "contradictory", "circular",
    "fallacious", "invalid", "indefensible",
    # Negative evaluative language
    "poorly", "inadequately", "insufficiently", "weakly", "vaguely",
    "unclearly", "inappropriately", "without", "never", "no evidence",
]

# ---------------------------------------------------------------------------
# LLM coherence screening prompt (fixed — spec Section 2.4, Tier 2)
# Input: classification + rationale ONLY (no fragment, no gold standard,
# no study metadata) to preserve independence from the benchmarked task.
# ---------------------------------------------------------------------------

LLM_COHERENCE_PROMPT: str = """\
You are checking whether a rationale logically supports its classification.

Classification: {classification}
Rationale: {rationale}

Does the rationale logically support the classification? Consider:
- Does the rationale describe strengths if classified as "sound"?
- Does the rationale describe weaknesses if classified as "not sound"?
- Is there any contradiction between the rationale and the classification?

Respond with ONLY one word: "coherent" or "incoherent"\
"""

# ---------------------------------------------------------------------------
# Gold-standard rating/rationale audit (see gold_label_audit.py)
# Blind: Haiku sees ONLY the expert's rationale -- not the expert's label,
# the fragment, or any model output -- and reports which label it implies.
# ---------------------------------------------------------------------------

GOLD_AUDIT_PROMPT: str = """\
An evaluation expert read a fragment of a program evaluation report and rated \
the quality of its evaluative reasoning as either SOUND or NOT SOUND, then \
wrote the rationale below explaining that rating. You are shown only the \
rationale, not the rating.

Based solely on what the rationale says, which rating does it support?

Rationale:
\"\"\"{rationale}\"\"\"

On the first line, respond with exactly one of: SOUND, NOT_SOUND, UNCLEAR \
(use UNCLEAR only if the rationale genuinely does not indicate either rating). \
On the second line, give one short sentence explaining why.\
"""
GOLD_AUDIT_MAX_TOKENS = 120
GOLD_AUDIT_RAW_PATH = LOGS_DIR / "gold_label_audit_haiku.csv"
GOLD_ADJUDICATION_PATH = DATA_DIR / "analysis" / "gold_label_adjudication.csv"
FAILURE_CODES_GPT5_PATH = DATA_DIR / "analysis" / "failure_codes_gpt5.csv"
