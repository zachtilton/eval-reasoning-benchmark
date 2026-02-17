"""
Schema validation tests for the scoring pipeline outputs.

Covers spec Section 2.4 data schemas (Section 5 and 6 of specifications.md):
  - Scored database column presence and valid values
  - Fragment outcomes column presence and valid values
  - Structural invariants: pass_count + fail_count == 3
  - Null constraints: run_outcome and fragment_outcome never null
  - Cross-column constraints: failure_type vs. run_outcome
  - Classification accuracy vs. coherence_final consistency
  - Majority-rule and unanimous-agreement correctness
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.scoring.accuracy import adjudicate_fragment_outcomes, assign_run_outcomes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_RUN_OUTCOMES = {"pass", "fail"}
VALID_FAILURE_TYPES = {"type_1", "type_2", "type_3", None}
VALID_COHERENCE_FINAL = {"coherent", "incoherent", "ambiguous"}
VALID_CLASSIFICATION_ACCURACY = {"correct", "incorrect", None}
VALID_FRAGMENT_OUTCOMES = {"pass", "fail"}

# Required columns in the fragment_outcomes schema (spec Section 2.4 §6)
FRAGMENT_OUTCOMES_REQUIRED_COLUMNS = {
    "fragment_id",
    "model_family",
    "prompt_condition",
    "pass_count",
    "fail_count",
    "fragment_outcome",
    "unanimous_agreement",
}


def _make_run_df(
    fragment_id: str = "F_001",
    model: str = "gpt_5",
    prompt: str = "zero_shot",
    coherence_final: str = "coherent",
    classification_accuracy: str | None = "correct",
) -> pd.DataFrame:
    """Single-row DataFrame suitable for assign_run_outcomes input."""
    return pd.DataFrame([{
        "fragment_id": fragment_id,
        "model_family": model,
        "prompt_condition": prompt,
        "run_number": 1,
        "coherence_final": coherence_final,
        "classification_accuracy": classification_accuracy,
    }])


def _make_scored_df_all_combos() -> pd.DataFrame:
    """
    Build a 6-row DataFrame covering all coherent/accuracy combinations
    that are reachable in the real pipeline.

    Combinations:
      coherent + correct       → pass   / None
      coherent + incorrect     → fail   / type_1
      incoherent + correct     → fail   / type_2
      incoherent + incorrect   → fail   / type_3
      incoherent + None        → fail   / type_3  (incoherent bypasses accuracy)
      ambiguous  + None        → fail   / type_3  (ambiguous treated as incoherent)
    """
    rows = [
        {"coherence_final": "coherent",   "classification_accuracy": "correct"},
        {"coherence_final": "coherent",   "classification_accuracy": "incorrect"},
        {"coherence_final": "incoherent", "classification_accuracy": "correct"},
        {"coherence_final": "incoherent", "classification_accuracy": "incorrect"},
        {"coherence_final": "incoherent", "classification_accuracy": None},
        {"coherence_final": "ambiguous",  "classification_accuracy": None},
    ]
    return pd.DataFrame(rows)


def _three_run_df(
    outcomes: tuple[str, str, str],
    fragment_id: str = "F_001",
    model: str = "gpt_5",
    prompt: str = "zero_shot",
) -> pd.DataFrame:
    """Build a 3-run DataFrame for adjudicate_fragment_outcomes input."""
    rows = [
        {
            "fragment_id": fragment_id,
            "model_family": model,
            "prompt_condition": prompt,
            "run_number": i + 1,
            "run_outcome": outcome,
        }
        for i, outcome in enumerate(outcomes)
    ]
    return pd.DataFrame(rows)


def _build_full_scored_df(tmp_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a scored DataFrame and the adjudicated fragment outcomes."""
    rows = []
    for i, outcome in enumerate(["pass", "pass", "fail"], start=1):
        rows.append({
            "fragment_id": "F_001",
            "model_family": "gpt_5",
            "prompt_condition": "zero_shot",
            "run_number": i,
            "run_outcome": outcome,
        })
    df = pd.DataFrame(rows)
    frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "frag.csv")
    return df, frag_df


# ===========================================================================
# Class 1: Scored database — column presence after assign_run_outcomes
# ===========================================================================

class TestScoredDatabaseColumnPresence:
    """assign_run_outcomes adds the two required columns."""

    def test_run_outcome_column_added(self):
        df = _make_run_df()
        result = assign_run_outcomes(df)
        assert "run_outcome" in result.columns

    def test_failure_type_column_added(self):
        df = _make_run_df()
        result = assign_run_outcomes(df)
        assert "failure_type" in result.columns

    def test_input_columns_preserved(self):
        df = _make_run_df()
        original_cols = set(df.columns)
        result = assign_run_outcomes(df)
        assert original_cols.issubset(set(result.columns))

    def test_row_count_unchanged(self):
        df = pd.concat([_make_run_df("F_001"), _make_run_df("F_002")], ignore_index=True)
        result = assign_run_outcomes(df)
        assert len(result) == 2


# ===========================================================================
# Class 2: Scored database — valid values per schema
# ===========================================================================

class TestScoredDatabaseValidValues:
    """run_outcome and failure_type only contain schema-permitted values."""

    def test_run_outcome_only_valid_values(self):
        df = _make_scored_df_all_combos()
        result = assign_run_outcomes(df)
        observed = set(result["run_outcome"].unique())
        assert observed.issubset(VALID_RUN_OUTCOMES)

    def test_failure_type_only_valid_values(self):
        df = _make_scored_df_all_combos()
        result = assign_run_outcomes(df)
        # Convert to Python objects so None comparison works
        observed = {v if not pd.isna(v) else None for v in result["failure_type"]}
        assert observed.issubset(VALID_FAILURE_TYPES)

    def test_run_outcome_no_nulls(self):
        df = _make_scored_df_all_combos()
        result = assign_run_outcomes(df)
        assert result["run_outcome"].notna().all(), "run_outcome must never be null"

    def test_run_outcome_is_string_type(self):
        df = _make_run_df(coherence_final="coherent", classification_accuracy="correct")
        result = assign_run_outcomes(df)
        assert isinstance(result["run_outcome"].iloc[0], str)


# ===========================================================================
# Class 3: failure_type ↔ run_outcome cross-column constraint
# ===========================================================================

class TestFailureTypeConstraint:
    """Schema rule: failure_type is None iff run_outcome is 'pass'."""

    def test_failure_type_none_when_pass(self):
        df = _make_run_df(coherence_final="coherent", classification_accuracy="correct")
        result = assign_run_outcomes(df)
        row = result.iloc[0]
        assert row["run_outcome"] == "pass"
        assert row["failure_type"] is None or pd.isna(row["failure_type"])

    def test_failure_type_type1_when_coherent_incorrect(self):
        df = _make_run_df(coherence_final="coherent", classification_accuracy="incorrect")
        result = assign_run_outcomes(df)
        row = result.iloc[0]
        assert row["run_outcome"] == "fail"
        assert row["failure_type"] == "type_1"

    def test_failure_type_type2_when_incoherent_correct(self):
        df = _make_run_df(coherence_final="incoherent", classification_accuracy="correct")
        result = assign_run_outcomes(df)
        row = result.iloc[0]
        assert row["run_outcome"] == "fail"
        assert row["failure_type"] == "type_2"

    def test_failure_type_type3_when_incoherent_incorrect(self):
        df = _make_run_df(coherence_final="incoherent", classification_accuracy="incorrect")
        result = assign_run_outcomes(df)
        row = result.iloc[0]
        assert row["run_outcome"] == "fail"
        assert row["failure_type"] == "type_3"

    def test_failure_type_type3_when_incoherent_null_accuracy(self):
        """Incoherent responses bypass accuracy — treated as type_3 not pass."""
        df = _make_run_df(coherence_final="incoherent", classification_accuracy=None)
        result = assign_run_outcomes(df)
        row = result.iloc[0]
        assert row["run_outcome"] == "fail"
        assert row["failure_type"] == "type_3"

    def test_failure_type_not_none_for_every_fail_row(self):
        """Every 'fail' run must have a failure_type assigned."""
        df = _make_scored_df_all_combos()
        result = assign_run_outcomes(df)
        fail_rows = result[result["run_outcome"] == "fail"]
        null_mask = fail_rows["failure_type"].isna()
        assert not null_mask.any(), (
            f"Fail rows have null failure_type: {fail_rows[null_mask]}"
        )

    def test_exhaustive_mapping_consistent(self):
        """All six reachable combinations produce schema-valid (outcome, type) pairs."""
        df = _make_scored_df_all_combos()
        result = assign_run_outcomes(df)
        for _, row in result.iterrows():
            outcome = row["run_outcome"]
            ft = row["failure_type"] if not pd.isna(row["failure_type"]) else None
            assert outcome in VALID_RUN_OUTCOMES
            assert ft in VALID_FAILURE_TYPES
            if outcome == "pass":
                assert ft is None
            else:
                assert ft is not None, f"fail row has null failure_type: {row}"


# ===========================================================================
# Class 4: classification_accuracy ↔ coherence_final constraint
# ===========================================================================

class TestClassificationAccuracyConstraint:
    """classification_accuracy is None when coherence_final != 'coherent'."""

    def test_incoherent_accuracy_remains_none(self):
        """compare_to_gold_standard sets accuracy=None for incoherent rows;
        assign_run_outcomes must not overwrite those Nones with a non-None type."""
        df = _make_run_df(coherence_final="incoherent", classification_accuracy=None)
        result = assign_run_outcomes(df)
        # The classification_accuracy column is not touched by assign_run_outcomes
        # — verify the column value is still None (not altered).
        val = result.iloc[0].get("classification_accuracy")
        assert val is None or pd.isna(val)

    def test_ambiguous_accuracy_remains_none(self):
        df = _make_run_df(coherence_final="ambiguous", classification_accuracy=None)
        result = assign_run_outcomes(df)
        val = result.iloc[0].get("classification_accuracy")
        assert val is None or pd.isna(val)

    def test_coherent_with_correct_accuracy_passes(self):
        df = _make_run_df(coherence_final="coherent", classification_accuracy="correct")
        result = assign_run_outcomes(df)
        assert result.iloc[0]["classification_accuracy"] == "correct"

    def test_coherent_with_incorrect_accuracy_preserved(self):
        df = _make_run_df(coherence_final="coherent", classification_accuracy="incorrect")
        result = assign_run_outcomes(df)
        assert result.iloc[0]["classification_accuracy"] == "incorrect"


# ===========================================================================
# Class 5: Fragment outcomes — column presence
# ===========================================================================

class TestFragmentOutcomesColumnPresence:
    """adjudicate_fragment_outcomes returns all required schema columns."""

    def test_all_required_columns_present(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        missing = FRAGMENT_OUTCOMES_REQUIRED_COLUMNS - set(frag_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_unexpected_columns(self, tmp_path):
        """The schema defines exactly 7 columns; no bonus columns should appear."""
        df = _three_run_df(("pass", "pass", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        extra = set(frag_df.columns) - FRAGMENT_OUTCOMES_REQUIRED_COLUMNS
        assert not extra, f"Unexpected extra columns: {extra}"


# ===========================================================================
# Class 6: Fragment outcomes — valid values
# ===========================================================================

class TestFragmentOutcomesValidValues:
    """Fragment outcomes columns only contain schema-permitted values."""

    def test_fragment_outcome_only_pass_or_fail(self, tmp_path):
        df = pd.concat([
            _three_run_df(("pass", "pass", "pass"), fragment_id="F_001"),
            _three_run_df(("fail", "fail", "fail"), fragment_id="F_002"),
            _three_run_df(("pass", "pass", "fail"), fragment_id="F_003"),
            _three_run_df(("pass", "fail", "fail"), fragment_id="F_004"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        observed = set(frag_df["fragment_outcome"].unique())
        assert observed.issubset(VALID_FRAGMENT_OUTCOMES)

    def test_pass_count_in_range_0_to_3(self, tmp_path):
        df = pd.concat([
            _three_run_df(("pass", "pass", "pass"), fragment_id="F_001"),
            _three_run_df(("fail", "fail", "fail"), fragment_id="F_002"),
            _three_run_df(("pass", "fail", "fail"), fragment_id="F_003"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df["pass_count"].between(0, 3).all()

    def test_fail_count_in_range_0_to_3(self, tmp_path):
        df = pd.concat([
            _three_run_df(("pass", "pass", "pass"), fragment_id="F_001"),
            _three_run_df(("fail", "fail", "fail"), fragment_id="F_002"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df["fail_count"].between(0, 3).all()

    def test_fragment_outcome_no_nulls(self, tmp_path):
        df = pd.concat([
            _three_run_df(("pass", "pass", "fail"), fragment_id="F_001"),
            _three_run_df(("fail", "fail", "pass"), fragment_id="F_002"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df["fragment_outcome"].notna().all()

    def test_unanimous_agreement_is_boolean(self, tmp_path):
        df = _three_run_df(("pass", "pass", "pass"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        val = frag_df.iloc[0]["unanimous_agreement"]
        # numpy.bool_ is not a subclass of Python int in numpy 2.x, so use
        # membership test rather than isinstance to handle all bool-like types.
        assert val in (True, False), (
            f"unanimous_agreement should be boolean-like, got {type(val)}: {val!r}"
        )


# ===========================================================================
# Class 7: pass_count + fail_count == 3 invariant
# ===========================================================================

class TestPassFailCountInvariant:
    """Every fragment-model-prompt row: pass_count + fail_count == 3."""

    def test_3_pass_0_fail(self, tmp_path):
        df = _three_run_df(("pass", "pass", "pass"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        row = frag_df.iloc[0]
        assert row["pass_count"] + row["fail_count"] == 3

    def test_2_pass_1_fail(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        row = frag_df.iloc[0]
        assert row["pass_count"] + row["fail_count"] == 3

    def test_1_pass_2_fail(self, tmp_path):
        df = _three_run_df(("pass", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        row = frag_df.iloc[0]
        assert row["pass_count"] + row["fail_count"] == 3

    def test_0_pass_3_fail(self, tmp_path):
        df = _three_run_df(("fail", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        row = frag_df.iloc[0]
        assert row["pass_count"] + row["fail_count"] == 3

    def test_invariant_holds_across_multiple_combinations(self, tmp_path):
        """All four possible pass/fail distributions satisfy the invariant."""
        df = pd.concat([
            _three_run_df(("pass", "pass", "pass"), fragment_id="F_001"),
            _three_run_df(("pass", "pass", "fail"), fragment_id="F_002"),
            _three_run_df(("pass", "fail", "fail"), fragment_id="F_003"),
            _three_run_df(("fail", "fail", "fail"), fragment_id="F_004"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        totals = frag_df["pass_count"] + frag_df["fail_count"]
        assert (totals == 3).all(), (
            f"Rows violating pass+fail==3:\n{frag_df[totals != 3]}"
        )


# ===========================================================================
# Class 8: Majority rule correctness
# ===========================================================================

class TestMajorityRuleCorrectness:
    """fragment_outcome follows the ≥ 2/3 majority-pass rule."""

    def test_3_of_3_pass_yields_pass(self, tmp_path):
        df = _three_run_df(("pass", "pass", "pass"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df.iloc[0]["fragment_outcome"] == "pass"

    def test_2_of_3_pass_yields_pass(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df.iloc[0]["fragment_outcome"] == "pass"

    def test_1_of_3_pass_yields_fail(self, tmp_path):
        df = _three_run_df(("pass", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df.iloc[0]["fragment_outcome"] == "fail"

    def test_0_of_3_pass_yields_fail(self, tmp_path):
        df = _three_run_df(("fail", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert frag_df.iloc[0]["fragment_outcome"] == "fail"

    def test_majority_rule_threshold_is_exactly_2(self, tmp_path):
        """Boundary: exactly 2 passes → 'pass'; exactly 1 pass → 'fail'."""
        df_2 = _three_run_df(("pass", "pass", "fail"), fragment_id="F_001")
        df_1 = _three_run_df(("pass", "fail", "fail"), fragment_id="F_002")
        df = pd.concat([df_2, df_1], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")

        outcome_map = dict(zip(frag_df["fragment_id"], frag_df["fragment_outcome"]))
        assert outcome_map["F_001"] == "pass", "2/3 passes must yield pass"
        assert outcome_map["F_002"] == "fail", "1/3 passes must yield fail"


# ===========================================================================
# Class 9: unanimous_agreement definition
# ===========================================================================

class TestUnanimousAgreementDefinition:
    """unanimous_agreement is True iff pass_count == 0 or pass_count == 3."""

    def test_unanimous_when_all_pass(self, tmp_path):
        df = _three_run_df(("pass", "pass", "pass"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert bool(frag_df.iloc[0]["unanimous_agreement"]) is True

    def test_unanimous_when_all_fail(self, tmp_path):
        df = _three_run_df(("fail", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert bool(frag_df.iloc[0]["unanimous_agreement"]) is True

    def test_not_unanimous_when_2_of_3(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert bool(frag_df.iloc[0]["unanimous_agreement"]) is False

    def test_not_unanimous_when_1_of_3(self, tmp_path):
        df = _three_run_df(("pass", "fail", "fail"))
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert bool(frag_df.iloc[0]["unanimous_agreement"]) is False

    def test_unanimous_agreement_consistent_with_counts(self, tmp_path):
        """unanimous_agreement must agree with pass_count/fail_count semantics."""
        df = pd.concat([
            _three_run_df(("pass", "pass", "pass"), fragment_id="F_001"),
            _three_run_df(("pass", "pass", "fail"), fragment_id="F_002"),
            _three_run_df(("pass", "fail", "fail"), fragment_id="F_003"),
            _three_run_df(("fail", "fail", "fail"), fragment_id="F_004"),
        ], ignore_index=True)
        frag_df = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")

        for _, row in frag_df.iterrows():
            expected_unanimous = (row["pass_count"] == 0) or (row["pass_count"] == 3)
            actual_unanimous = bool(row["unanimous_agreement"])
            assert actual_unanimous == expected_unanimous, (
                f"{row['fragment_id']}: pass_count={row['pass_count']}, "
                f"expected unanimous={expected_unanimous}, got {actual_unanimous}"
            )


# ===========================================================================
# Class 10: Output returns a copy (pipeline immutability)
# ===========================================================================

class TestPipelineImmutability:
    """assign_run_outcomes and adjudicate_fragment_outcomes return copies."""

    def test_assign_run_outcomes_does_not_mutate_input(self):
        df = _make_run_df()
        original_cols = set(df.columns)
        assign_run_outcomes(df)
        assert set(df.columns) == original_cols, (
            "assign_run_outcomes must not add columns to the input DataFrame"
        )

    def test_adjudicate_fragment_outcomes_returns_dataframe(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        result = adjudicate_fragment_outcomes(df, output_path=tmp_path / "out.csv")
        assert isinstance(result, pd.DataFrame)

    def test_adjudicate_saves_csv_to_output_path(self, tmp_path):
        df = _three_run_df(("pass", "pass", "fail"))
        out = tmp_path / "fragment_outcomes.csv"
        adjudicate_fragment_outcomes(df, output_path=out)
        assert out.exists(), "adjudicate_fragment_outcomes must write the CSV file"

    def test_csv_contents_match_returned_dataframe(self, tmp_path):
        df = _three_run_df(("pass", "fail", "fail"))
        out = tmp_path / "fragment_outcomes.csv"
        frag_df = adjudicate_fragment_outcomes(df, output_path=out)
        csv_df = pd.read_csv(out)
        assert list(frag_df.columns) == list(csv_df.columns)
        assert len(frag_df) == len(csv_df)
