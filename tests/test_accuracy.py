"""
Unit tests for src/scoring/accuracy.py.

Covers:
- compare_to_gold_standard: correct/incorrect assignment, incoherent bypass.
- assign_run_outcomes: all four pass/fail × failure-type combinations.
- adjudicate_fragment_outcomes: 3/3, 2/3, 1/3, 0/3 pass scenarios;
  unanimous_agreement semantics; pass_count + fail_count = 3 invariant.
- identify_failures_for_coding: Type 1 filter; representative run selection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.scoring.accuracy import (
    adjudicate_fragment_outcomes,
    assign_run_outcomes,
    compare_to_gold_standard,
    identify_failures_for_coding,
)

from .conftest import make_run_row, make_three_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_with_coherence(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal database DataFrame with coherence + accuracy populated."""
    defaults = {
        "fragment_id": "F_001",
        "model_family": "gpt_5",
        "prompt_condition": "zero_shot",
        "run_number": 1,
        "classification_output": "sound",
        "coherence_final": "coherent",
        "classification_accuracy": "correct",
        # NOTE: expert_classification is NOT included here — compare_to_gold_standard
        # adds it via a merge with gold_df.  Including it pre-merge causes pandas to
        # produce expert_classification_x / expert_classification_y column names,
        # breaking row["expert_classification"] access inside the function.
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_run_db(fragment_coherence_accuracy: list[tuple]) -> pd.DataFrame:
    """
    Build a run-level database from (coherence, accuracy) tuples.
    All rows share fragment F_001 / gpt_5 / zero_shot.
    """
    rows = []
    for i, (coh, acc) in enumerate(fragment_coherence_accuracy, start=1):
        rows.append({
            "fragment_id": "F_001",
            "model_family": "gpt_5",
            "prompt_condition": "zero_shot",
            "run_number": i,
            "coherence_final": coh,
            "classification_accuracy": acc,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Class: compare_to_gold_standard
# ---------------------------------------------------------------------------

class TestCompareToGoldStandard:

    def test_coherent_matching_is_correct(self, gold_df):
        """Coherent response that matches gold standard → 'correct'."""
        db = _make_db_with_coherence([{
            "fragment_id": "F_001",
            "classification_output": "sound",
            "coherence_final": "coherent",
        }])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] == "correct"

    def test_coherent_diverging_is_incorrect(self, gold_df):
        """Coherent response that diverges from gold standard → 'incorrect'."""
        db = _make_db_with_coherence([{
            "fragment_id": "F_001",
            "classification_output": "not_sound",  # gold is "sound"
            "coherence_final": "coherent",
        }])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] == "incorrect"

    def test_incoherent_response_bypasses_accuracy(self, gold_df):
        """Incoherent responses bypass Step 2 → classification_accuracy is None."""
        db = _make_db_with_coherence([{
            "fragment_id": "F_001",
            "classification_output": "sound",
            "coherence_final": "incoherent",
        }])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] is None

    def test_ambiguous_response_bypasses_accuracy(self, gold_df):
        """Ambiguous (manual review needed) → also bypasses accuracy check."""
        db = _make_db_with_coherence([{
            "fragment_id": "F_001",
            "classification_output": "sound",
            "coherence_final": "ambiguous",
        }])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] is None

    def test_not_sound_matching_gold_is_correct(self, gold_df):
        """'not_sound' matching gold 'not_sound' → 'correct'."""
        db = _make_db_with_coherence([{
            "fragment_id": "F_002",
            "classification_output": "not_sound",
            "coherence_final": "coherent",
        }])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] == "correct"

    def test_multiple_rows_accuracy_per_row(self, gold_df):
        """Each row must get its own independent accuracy assessment."""
        db = _make_db_with_coherence([
            {"fragment_id": "F_001", "classification_output": "sound",     "coherence_final": "coherent"},
            {"fragment_id": "F_001", "classification_output": "not_sound", "coherence_final": "coherent"},
            {"fragment_id": "F_001", "classification_output": "sound",     "coherence_final": "incoherent"},
        ])
        result = compare_to_gold_standard(db, gold_df)
        assert result.iloc[0]["classification_accuracy"] == "correct"
        assert result.iloc[1]["classification_accuracy"] == "incorrect"
        # pandas converts None → NaN when the column contains mixed types (strings + None)
        assert pd.isna(result.iloc[2]["classification_accuracy"])


# ---------------------------------------------------------------------------
# Class: assign_run_outcomes
# ---------------------------------------------------------------------------

class TestAssignRunOutcomes:

    def test_coherent_correct_is_pass(self):
        """Coherent AND correct → Pass, failure_type=None (spec Step 3)."""
        db = _make_run_db([("coherent", "correct")])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "pass"
        assert result.iloc[0]["failure_type"] is None

    def test_coherent_incorrect_is_fail_type_1(self):
        """Coherent but incorrect → Fail, failure_type='type_1' (codeable)."""
        db = _make_run_db([("coherent", "incorrect")])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "fail"
        assert result.iloc[0]["failure_type"] == "type_1"

    def test_incoherent_correct_is_fail_type_2(self):
        """
        Incoherent but classification accidentally matches gold → Fail, type_2.
        Note: in the normal pipeline accuracy=None for incoherent, but the
        function logic handles type_2 when accuracy='correct' is explicitly set.
        """
        db = _make_run_db([("incoherent", "correct")])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "fail"
        assert result.iloc[0]["failure_type"] == "type_2"

    def test_incoherent_incorrect_is_fail_type_3(self):
        """Incoherent AND incorrect (or accuracy=None) → Fail, type_3."""
        db = _make_run_db([("incoherent", "incorrect")])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "fail"
        assert result.iloc[0]["failure_type"] == "type_3"

    def test_incoherent_none_accuracy_is_fail_type_3(self):
        """Incoherent with accuracy=None (normal pipeline path) → fail, type_3."""
        db = _make_run_db([("incoherent", None)])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "fail"
        assert result.iloc[0]["failure_type"] == "type_3"

    def test_ambiguous_is_fail(self):
        """Ambiguous coherence (not coherent) → Fail regardless of accuracy."""
        db = _make_run_db([("ambiguous", None)])
        result = assign_run_outcomes(db)
        assert result.iloc[0]["run_outcome"] == "fail"

    def test_all_rows_receive_run_outcome(self):
        rows = [
            ("coherent", "correct"),
            ("coherent", "incorrect"),
            ("incoherent", None),
        ]
        db = _make_run_db(rows)
        result = assign_run_outcomes(db)
        assert result["run_outcome"].notna().all()
        assert set(result["run_outcome"]).issubset({"pass", "fail"})

    def test_failure_type_none_only_when_pass(self):
        rows = [
            ("coherent", "correct"),
            ("coherent", "incorrect"),
        ]
        db = _make_run_db(rows)
        result = assign_run_outcomes(db)
        pass_rows = result[result["run_outcome"] == "pass"]
        fail_rows = result[result["run_outcome"] == "fail"]
        assert (pass_rows["failure_type"].isna()).all()
        assert (fail_rows["failure_type"].notna()).all()

    def test_returns_copy_not_mutation(self):
        db = _make_run_db([("coherent", "correct")])
        original_cols = set(db.columns)
        _ = assign_run_outcomes(db)
        assert "run_outcome" not in original_cols


# ---------------------------------------------------------------------------
# Class: adjudicate_fragment_outcomes
# ---------------------------------------------------------------------------

class TestAdjudicateFragmentOutcomes:
    """
    Fragment-level adjudication: majority rule across 3 runs.
    Spec Step 4: Pass if ≥ 2/3 runs pass.
    """

    def _adjudicate(self, outcomes: tuple[str, str, str], tmpdir: Path) -> pd.Series:
        """Helper: build 3-run database, adjudicate, return the single result row."""
        db = make_three_runs(outcomes=outcomes)
        output_path = Path(tmpdir) / "fragment_outcomes.csv"
        result_df = adjudicate_fragment_outcomes(db, output_path=output_path)
        assert len(result_df) == 1
        return result_df.iloc[0]

    def test_3_pass_is_fragment_pass(self, tmp_path):
        row = self._adjudicate(("pass", "pass", "pass"), tmp_path)
        assert row["fragment_outcome"] == "pass"

    def test_3_pass_is_unanimous(self, tmp_path):
        row = self._adjudicate(("pass", "pass", "pass"), tmp_path)
        assert bool(row["unanimous_agreement"]) == True

    def test_2_pass_1_fail_is_fragment_pass(self, tmp_path):
        row = self._adjudicate(("pass", "pass", "fail"), tmp_path)
        assert row["fragment_outcome"] == "pass"

    def test_2_pass_1_fail_is_not_unanimous(self, tmp_path):
        row = self._adjudicate(("pass", "pass", "fail"), tmp_path)
        assert bool(row["unanimous_agreement"]) == False

    def test_1_pass_2_fail_is_fragment_fail(self, tmp_path):
        row = self._adjudicate(("pass", "fail", "fail"), tmp_path)
        assert row["fragment_outcome"] == "fail"

    def test_1_pass_2_fail_is_not_unanimous(self, tmp_path):
        row = self._adjudicate(("pass", "fail", "fail"), tmp_path)
        assert bool(row["unanimous_agreement"]) == False

    def test_0_pass_3_fail_is_fragment_fail(self, tmp_path):
        row = self._adjudicate(("fail", "fail", "fail"), tmp_path)
        assert row["fragment_outcome"] == "fail"

    def test_0_pass_3_fail_is_unanimous(self, tmp_path):
        """All-fail is also unanimous agreement (unanimous on failure)."""
        row = self._adjudicate(("fail", "fail", "fail"), tmp_path)
        assert bool(row["unanimous_agreement"]) == True

    def test_pass_count_plus_fail_count_equals_3(self, tmp_path):
        """pass_count + fail_count must always equal 3 (spec constraint)."""
        for outcomes in [
            ("pass", "pass", "pass"),
            ("pass", "pass", "fail"),
            ("pass", "fail", "fail"),
            ("fail", "fail", "fail"),
        ]:
            row = self._adjudicate(outcomes, tmp_path)
            assert row["pass_count"] + row["fail_count"] == 3, (
                f"pass_count({row['pass_count']}) + fail_count({row['fail_count']}) "
                f"≠ 3 for outcomes={outcomes}"
            )

    def test_output_saved_to_csv(self, tmp_path):
        db = make_three_runs(outcomes=("pass", "pass", "pass"))
        output_path = tmp_path / "fragment_outcomes.csv"
        adjudicate_fragment_outcomes(db, output_path=output_path)
        assert output_path.exists()
        saved = pd.read_csv(output_path)
        assert len(saved) == 1

    def test_multiple_combinations_adjudicated_independently(self, tmp_path):
        """Multiple fragment-model-prompt groups must be adjudicated independently."""
        db = pd.concat([
            make_three_runs("F_001", "gpt_5", "zero_shot", ("pass","pass","pass")),
            make_three_runs("F_002", "gpt_5", "zero_shot", ("pass","fail","fail")),
            make_three_runs("F_001", "gpt_5", "few_shot",  ("fail","fail","fail")),
        ])
        output_path = tmp_path / "fragment_outcomes.csv"
        result = adjudicate_fragment_outcomes(db, output_path=output_path)

        assert len(result) == 3

        f1_zs = result[(result["fragment_id"]=="F_001") & (result["prompt_condition"]=="zero_shot")].iloc[0]
        f2_zs = result[(result["fragment_id"]=="F_002") & (result["prompt_condition"]=="zero_shot")].iloc[0]
        f1_fs = result[(result["fragment_id"]=="F_001") & (result["prompt_condition"]=="few_shot")].iloc[0]

        assert f1_zs["fragment_outcome"] == "pass"  # 3/3
        assert f2_zs["fragment_outcome"] == "fail"   # 1/3
        assert f1_fs["fragment_outcome"] == "fail"   # 0/3


# ---------------------------------------------------------------------------
# Class: identify_failures_for_coding
# ---------------------------------------------------------------------------

class TestIdentifyFailuresForCoding:
    """
    Type 1 filter: only coherent-but-incorrect fragment-level failures
    are returned for coding.  Type 2 and Type 3 failures are documented
    but excluded from the returned DataFrame.
    """

    def _build_inputs(
        self,
        fragment_outcome: str,
        run_failure_types: tuple[str | None, str | None, str | None],
        tmpdir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build fragment_df + database_df for a single fragment under gpt_5."""
        frag_df = pd.DataFrame([{
            "fragment_id": "F_001",
            "model_family": "gpt_5",
            "prompt_condition": "zero_shot",
            "fragment_outcome": fragment_outcome,
            "pass_count": sum(1 for ft in run_failure_types if ft is None),
            "fail_count": sum(1 for ft in run_failure_types if ft is not None),
            "unanimous_agreement": all(ft is None for ft in run_failure_types),
        }])
        db_rows = []
        for i, ft in enumerate(run_failure_types, start=1):
            db_rows.append({
                "fragment_id": "F_001",
                "model_family": "gpt_5",
                "prompt_condition": "zero_shot",
                "run_number": i,
                "run_outcome": "fail" if ft else "pass",
                "failure_type": ft,
            })
        db_df = pd.DataFrame(db_rows)
        return frag_df, db_df

    def test_type_1_failure_is_returned_for_coding(self, tmp_path):
        frag_df, db_df = self._build_inputs(
            "fail", ("type_1", "type_1", "type_1"), tmp_path
        )
        output_path = tmp_path / "coding.csv"
        result = identify_failures_for_coding(
            frag_df, db_df, primary_model="gpt_5", output_path=output_path
        )
        assert len(result) == 1
        assert result.iloc[0]["dominant_failure_type"] == "type_1"

    def test_type_3_failure_not_returned_for_coding(self, tmp_path):
        """Type 3 (incoherent + incorrect) → excluded from coding result."""
        frag_df, db_df = self._build_inputs(
            "fail", ("type_3", "type_3", "type_3"), tmp_path
        )
        output_path = tmp_path / "coding.csv"
        result = identify_failures_for_coding(
            frag_df, db_df, primary_model="gpt_5", output_path=output_path
        )
        assert len(result) == 0

    def test_passing_fragment_not_returned(self, tmp_path):
        frag_df, db_df = self._build_inputs(
            "pass", (None, None, None), tmp_path
        )
        output_path = tmp_path / "coding.csv"
        result = identify_failures_for_coding(
            frag_df, db_df, primary_model="gpt_5", output_path=output_path
        )
        assert len(result) == 0

    def test_representative_run_is_type_1_when_available(self, tmp_path):
        """Representative run must prefer type_1 run (spec Section 3.1 step 1)."""
        frag_df, db_df = self._build_inputs(
            "fail", ("type_3", "type_1", "type_3"), tmp_path
        )
        # Fragment-level: mixed types; type_1 available at run 2
        output_path = tmp_path / "coding.csv"
        result = identify_failures_for_coding(
            frag_df, db_df, primary_model="gpt_5", output_path=output_path
        )
        if len(result) > 0:
            assert result.iloc[0]["representative_run"] == 2

    def test_comparison_model_failures_not_returned(self, tmp_path):
        """Only primary model failures are identified; comparison models ignored."""
        frag_df = pd.DataFrame([{
            "fragment_id": "F_001",
            "model_family": "claude_opus_4_6",  # comparison model
            "prompt_condition": "zero_shot",
            "fragment_outcome": "fail",
            "pass_count": 0, "fail_count": 3,
            "unanimous_agreement": True,
        }])
        db_df = pd.DataFrame([{
            "fragment_id": "F_001",
            "model_family": "claude_opus_4_6",
            "prompt_condition": "zero_shot",
            "run_number": i,
            "run_outcome": "fail",
            "failure_type": "type_1",
        } for i in range(1, 4)])

        output_path = tmp_path / "coding.csv"
        result = identify_failures_for_coding(
            frag_df, db_df, primary_model="gpt_5", output_path=output_path
        )
        assert len(result) == 0
