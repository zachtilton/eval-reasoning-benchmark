"""
Shared pytest fixtures for scoring pipeline tests.

Indicator words are chosen from src/scoring/config.py STRENGTH_INDICATORS and
WEAKNESS_INDICATORS.  Each fixture comment notes which list each word belongs
to so tests remain self-documenting when indicators change.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Indicator word constants (sourced from src/scoring/config.py lists)
# Strength: "establishes", "demonstrates", "synthesizes", "logical",
#            "comprehensive", "appropriately", "identifies", "addresses"
# Weakness: "missing", "absent", "flawed", "inadequate", "circular",
#            "indefensible", "omits", "fails to"
# ---------------------------------------------------------------------------

# A rationale whose strength count >> weakness count (diff >= 3, Tier-1 confident)
STRONG_SOUND_RATIONALE = (
    "The evaluation demonstrates comprehensive evidence and establishes a "
    "logical argument. It appropriately synthesizes findings and addresses "
    "all salient criteria with sufficient depth."
)
# strength words: demonstrates, comprehensive, establishes, logical,
#                 appropriately, synthesizes, addresses, sufficient  (~8)
# weakness words: 0  →  diff = 8 ≥ 3 ✓

# A rationale whose weakness count >> strength count (diff >= 3, Tier-1 confident)
STRONG_NOT_SOUND_RATIONALE = (
    "The evaluation is missing key evidence and omits central criteria. "
    "The reasoning is flawed and circular, rendering the conclusion "
    "indefensible and wholly inadequate."
)
# weakness words: missing, omits, flawed, circular, indefensible, inadequate (~6)
# strength words: 0  →  diff = 6 ≥ 3 ✓

# A rationale with ONE strength and ONE weakness word (diff = 0 < 3, routes to LLM)
BALANCED_RATIONALE = (
    "The evaluation demonstrates some progress toward addressing the criterion, "
    "but the evidence base is missing in key areas."
)
# strength: demonstrates (1)
# weakness: missing (1)
# diff = 0 < 3 → routes to Tier 2

# A rationale with ONLY weakness words for "sound" classification (should be incoherent)
WEAK_SOUND_MISMATCH_RATIONALE = (
    "The evaluation is missing evidence and omits key criteria. "
    "The reasoning is flawed, circular, and wholly inadequate for judgment."
)
# weakness: missing, omits, flawed, circular, inadequate (~5), strength: 0
# For "sound": strength (0) not > weakness (5) → incoherent ✓

# A rationale with ONLY strength words for "not_sound" classification (should be incoherent)
STRONG_NOT_SOUND_MISMATCH_RATIONALE = (
    "The evaluation demonstrates comprehensive and logical reasoning. "
    "It synthesizes evidence appropriately and addresses all criteria."
)
# strength: demonstrates, comprehensive, logical, synthesizes, appropriately, addresses (~6)
# weakness: 0
# For "not_sound": weakness (0) not > strength (6) → incoherent ✓


# ---------------------------------------------------------------------------
# Response dict fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sound_coherent_response():
    """Well-formed 'sound' response with clear strength indicators."""
    return {
        "classification_output": "sound",
        "rationale_text": STRONG_SOUND_RATIONALE,
        "error_details": None,
        "error_flag": False,
    }


@pytest.fixture
def not_sound_coherent_response():
    """Well-formed 'not_sound' response with clear weakness indicators."""
    return {
        "classification_output": "not_sound",
        "rationale_text": STRONG_NOT_SOUND_RATIONALE,
        "error_details": None,
        "error_flag": False,
    }


@pytest.fixture
def normal_response_dict():
    """Generic valid response dict (no edge case)."""
    return {
        "classification_output": "sound",
        "rationale_text": (
            "The evaluation demonstrates clear evidence-based reasoning "
            "and appropriately synthesizes all criteria with logical rigor."
        ),
        "error_details": None,
    }


# ---------------------------------------------------------------------------
# Gold standard fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def gold_df():
    """Minimal gold standard DataFrame for accuracy tests."""
    return pd.DataFrame([
        {"fragment_id": "F_001", "expert_classification": "sound"},
        {"fragment_id": "F_002", "expert_classification": "not_sound"},
        {"fragment_id": "F_003", "expert_classification": "sound"},
        {"fragment_id": "F_004", "expert_classification": "not_sound"},
    ])


# ---------------------------------------------------------------------------
# Scored database builder helpers
# ---------------------------------------------------------------------------

def make_run_row(
    fragment_id: str = "F_001",
    model: str = "gpt_5",
    prompt: str = "zero_shot",
    run: int = 1,
    coherence: str = "coherent",
    accuracy: str | None = "correct",
    run_outcome: str = "pass",
    failure_type: str | None = None,
    exclude: bool = False,
) -> dict:
    """Build a single run-level row for the scored database."""
    return {
        "fragment_id": fragment_id,
        "model_family": model,
        "prompt_condition": prompt,
        "run_number": run,
        "coherence_final": coherence,
        "classification_accuracy": accuracy,
        "run_outcome": run_outcome,
        "failure_type": failure_type,
        "exclude_from_analysis": exclude,
    }


def make_three_runs(
    fragment_id: str = "F_001",
    model: str = "gpt_5",
    prompt: str = "zero_shot",
    outcomes: tuple[str, str, str] = ("pass", "pass", "pass"),
) -> pd.DataFrame:
    """Build a 3-run group with specified per-run pass/fail outcomes."""
    rows = []
    for i, outcome in enumerate(outcomes, start=1):
        failure_type = None if outcome == "pass" else "type_1"
        rows.append(make_run_row(
            fragment_id=fragment_id,
            model=model,
            prompt=prompt,
            run=i,
            run_outcome=outcome,
            failure_type=failure_type,
        ))
    return pd.DataFrame(rows)
