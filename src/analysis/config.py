"""
Analysis-layer configuration: shared constants, model metadata, and output paths.

Centralizes constants that were duplicated across H.1 (generate_collapsed_summaries)
and H.2 (architecture_z_test) in the appendix.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR       = PROJECT_ROOT / "data"
ANALYSIS_DIR   = DATA_DIR / "analysis"          # failure codes, analysis outputs
RESPONSES_DIR  = DATA_DIR / "responses"
GOLD_STD_DIR   = DATA_DIR / "gold_standard"
RESULTS_DIR    = PROJECT_ROOT / "results"       # H.1-H.5 exported tables/reports

# Input file paths (written by scoring pipeline)
FRAGMENT_OUTCOMES_PATH = RESPONSES_DIR / "fragment_outcomes.csv"
SCORED_DB_PATH         = RESPONSES_DIR / "scored_database.csv"
FAILURE_CODES_PATH     = ANALYSIS_DIR  / "failure_codes_gpt5.csv"

# ---------------------------------------------------------------------------
# Model metadata (spec Section 2.2)
# ---------------------------------------------------------------------------

# Centralized map — do NOT duplicate in individual analysis modules.
ARCHITECTURE_MAP: dict[str, str] = {
    "gpt_5":          "closed",
    "claude_opus_4_6": "closed",
    "gemini_3_pro":   "closed",
    "deepseek_v3":    "open",
    "kimi_k2":        "open",
    "glm_4_7":        "open",
}

# Canonical display order for tables and charts
MODEL_ORDER: list[str] = [
    "gpt_5", "claude_opus_4_6", "gemini_3_pro",   # closed
    "deepseek_v3", "kimi_k2", "glm_4_7",           # open
]

PROMPT_ORDER: list[str] = ["zero_shot", "few_shot"]

PRIMARY_MODEL: str = "gpt_5"

# ---------------------------------------------------------------------------
# Domain labels (spec Section 3.1 / Appendix D.1)
# ---------------------------------------------------------------------------

# Centralized — do NOT repeat inline in H.3 functions.
DOMAIN_LABELS: dict[int, str] = {
    1: "Evaluative Framing",
    2: "Evidence Base",
    3: "Argument Structure",
    4: "Synthesis & Integration",
    5: "Evaluative Conclusion",
    6: "Qualifications & Transparency",
}

# ---------------------------------------------------------------------------
# Challenge case thresholds (spec Section 3.5)
# ---------------------------------------------------------------------------

COMPARISON_MODEL_THRESHOLD: int   = 2     # Criterion 1: ≥2 comparison models fail
RECURRING_FAILURE_THRESHOLD: float = 0.10  # Criterion 2: ≥10% of primary model failures

# ---------------------------------------------------------------------------
# Reliability check targets (spec Sections 3.1, 3.6)
# ---------------------------------------------------------------------------

KAPPA_TARGET: float = 0.80

# ---------------------------------------------------------------------------
# Statistical test parameters (spec Section 3.3)
# ---------------------------------------------------------------------------

ALPHA: float = 0.05
MCNEMAR_SMALL_N: int = 25   # Use exact binomial when n_discordant < 25
