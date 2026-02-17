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
RC2_SAMPLE_PATH = LOGS_DIR / "rc2_manual_review_sample.csv"

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
# RC2 stratified sample sizes (spec Section 2.4, Step 1)
# ---------------------------------------------------------------------------

RC2_COHERENT_SAMPLE = 20
RC2_INCOHERENT_SAMPLE = 20
RC2_AMBIGUOUS_MAX = 10    # "or all ambiguous if < 10"
RC2_KAPPA_TARGET = 0.80

# ---------------------------------------------------------------------------
# Primary diagnostic model (spec Section 2.2)
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "gpt_5"   # full diagnostic treatment: performance + failure coding

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
