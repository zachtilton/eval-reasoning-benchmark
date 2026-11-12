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
# RENAMED 2026-11-02: these were the model generation named in the original
# proposal (gpt_5, claude_opus_4_6, gemini_3_pro, deepseek_v3, kimi_k2,
# glm_4_7); the real run executed against the current-generation models
# actually available at execution time (per api_config.py's "models are
# selected as top-3 SOTA -- IDs may change between now and execution").
# These keys must match response_database.csv's model_family values
# exactly, or every lookup here (ARCHITECTURE_MAP.get, MODEL_ORDER
# iteration, PRIMARY_MODEL targeting) silently fails to match real data.
ARCHITECTURE_MAP: dict[str, str] = {
    "gpt_5_6_sol":                "closed",
    "claude_opus_5":              "closed",
    "gemini_3_pro_preview_high":  "closed",
    "deepseek_v4_pro":            "open",
    "kimi_k3":                    "open",
    "glm_5_2":                    "open",
}

# Canonical display order for tables and charts
MODEL_ORDER: list[str] = [
    "gpt_5_6_sol", "claude_opus_5", "gemini_3_pro_preview_high",  # closed
    "kimi_k3", "deepseek_v4_pro", "glm_5_2",                       # open
]

PROMPT_ORDER: list[str] = ["zero_shot", "few_shot"]

PRIMARY_MODEL: str = "gpt_5_6_sol"

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

# ---------------------------------------------------------------------------
# Failure-coding synthesis (see coding_synthesis.py)
# ---------------------------------------------------------------------------

SCORED_DB_CORRECTED_PATH = RESPONSES_DIR / "scored_database_corrected.csv"
FRAGMENT_OUTCOMES_CORRECTED_PATH = RESPONSES_DIR / "fragment_outcomes_corrected.csv"
CONTESTED_REVIEW_RESOLVED_PATH = RESPONSES_DIR / "contested_fragments_review_resolved.csv"
GOLD_RATIONALE_SUPPLEMENT_PATH = GOLD_STD_DIR / "gold_standard_rationale_supplement.csv"
CODING_SYNTHESIS_RESULTS_PATH = RESULTS_DIR / "failure_coding_synthesis.json"
FIGURES_DIR = RESULTS_DIR / "figures"

# DAC criteria reported individually; anything else is grouped as "other".
CORE_CRITERIA: list[str] = [
    "relevance", "coherence", "effectiveness", "efficiency", "impact", "sustainability",
]
FIGURE_DPI: int = 300

# ---------------------------------------------------------------------------
# Findings metrics and registry (see findings_metrics.py)
# ---------------------------------------------------------------------------

FRAGMENT_OUTCOMES_TRIANGULATED_PATH = RESPONSES_DIR / "fragment_outcomes_triangulated.csv"
RC2_BLIND_RETEST_PATH = PROJECT_ROOT / "reliability" / "blind_retest_40frag_pass2.csv"
RELIABILITY_CHECKS_PATH = RESULTS_DIR / "reliability_checks.json"
H2_CORRECTED_RESULTS_PATH = RESULTS_DIR / "H2_statistical_tests_corrected.json"
H2_TRIANGULATED_RESULTS_PATH = RESULTS_DIR / "H2_statistical_tests_triangulated.json"
FINDINGS_METRICS_PATH = RESULTS_DIR / "findings_metrics.json"
FINDINGS_REGISTRY_PATH = PROJECT_ROOT / "docs" / "findings" / "findings_registry.json"

FINDINGS_SEED: int = 42
BOOTSTRAP_RESAMPLES: int = 5000
KAPPA_BOOTSTRAP_RESAMPLES: int = 2000
