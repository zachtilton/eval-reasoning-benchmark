"""Tests for src.analysis.coding_synthesis (failure-coding synthesis)."""

import pandas as pd
import pytest

from src.analysis.coding_synthesis import (
    condition_summary,
    direction_summary,
    disposition_summary,
    load_coding_sheet,
    reviewer_concordance,
    to_fragment_level,
    validate_consistency,
)

COLUMNS = [
    "fragment_id", "prompt_condition", "gpt_classification", "gold_classification",
    "criterion", "primary_code", "secondary_code", "tertiary_code", "coder_doubts_gold_standard",
]


def _sheet(tmp_path, rows):
    """Write synthetic coding rows to CSV and load them through the module."""
    path = tmp_path / "codes.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return load_coding_sheet(path)


@pytest.fixture
def rows(tmp_path):
    """Three fragments: one lenient genuine failure in both conditions, one
    strict doubted in zero-shot, one lenient doubted in few-shot."""
    return _sheet(tmp_path, [
        [1, "zero_shot", "sound", "not_sound", "impact", "F5.1", "F.2.2", "", "n"],
        [1, "few_shot", "sound", "not_sound", "impact", "F5.1", "F.2.2", "", "n"],
        [2, "zero_shot", "not_sound", "sound", "relevance", "", "", "", "y"],
        [3, "few_shot", "sound", "not_sound", "efficiency", "", "", "", "y"],
    ])


def test_load_normalizes_malformed_codes_and_flags(rows):
    """F.2.2 becomes F2.2; genuine and direction derive from the sheet."""
    assert (rows["secondary_code"].iloc[:2] == "F2.2").all()
    assert rows["genuine"].tolist() == [True, True, False, False]
    assert rows["direction"].tolist() == ["lenient", "lenient", "strict", "lenient"]


def test_validate_consistency_clean(rows):
    """A consistent sheet reports no problems."""
    assert validate_consistency(rows) == []


def test_validate_consistency_flags_condition_disagreement(rows):
    """A fragment coded differently across conditions is reported."""
    rows.loc[1, "primary_code"] = "F4.2"
    assert any("primary_code differs" in p for p in validate_consistency(rows))


def test_validate_consistency_flags_code_without_genuine(rows):
    """A doubted row carrying a code breaks the coded-iff-genuine rule."""
    rows.loc[2, "primary_code"] = "F2.2"
    assert any("iff genuine" in p for p in validate_consistency(rows))


def test_disposition_and_direction(rows):
    """Fragment-level splits and failure rates use the right denominators."""
    frags = to_fragment_level(rows)
    d = disposition_summary(rows, frags, n_gpt_cases=10)
    assert d["fragments"] == {"n": 3, "genuine": 1, "doubt_gold": 2, "genuine_pct": 33.3, "doubt_gold_pct": 66.7}
    assert d["failure_rate_locked_gold_pct"] == 40.0
    assert d["failure_rate_coder_confirmed_pct"] == 20.0
    r = direction_summary(frags)
    assert (r["lenient"]["genuine"], r["lenient"]["n"]) == (1, 2)
    assert (r["strict"]["genuine"], r["strict"]["n"]) == (0, 1)


def test_condition_summary(rows):
    """Per-condition counts and confirmed failure rates."""
    c = condition_summary(rows, {"zero_shot": 5, "few_shot": 5})
    assert c["zero_shot"]["failures"] == 2 and c["zero_shot"]["genuine"] == 1
    assert c["few_shot"]["by_direction"]["lenient"]["n"] == 2
    assert c["few_shot"]["coder_confirmed_failure_rate_pct"] == 20.0


def test_reviewer_concordance_perfect_agreement(rows):
    """Coder doubt paired with reviewer siding with models gives kappa 1."""
    frags = to_fragment_level(rows)
    resolved = pd.DataFrame({
        "fragment_id": [1, 2, 3],
        "expert_classification": ["not_sound", "sound", "not_sound"],
        "reviewer_classification": ["not_sound", "not_sound", "sound"],
    })
    out = reviewer_concordance(frags, resolved)
    assert out["n"] == 3
    assert out["percent_agreement"] == 100.0
    assert out["cohens_kappa"] == 1.0
