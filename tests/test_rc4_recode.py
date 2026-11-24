"""Tests for the RC4 blinded recode sheet builder (src/analysis/rc4_recode.py)."""

import pandas as pd

from src.analysis.rc4_recode import (
    ENTRY_COLUMNS,
    HIDDEN_COLUMNS,
    SHOWN_COLUMNS,
    select_recode_rows,
    write_recode_sheet,
)


def _coding(n_confirmed: int = 6, n_doubted: int = 10) -> pd.DataFrame:
    """Synthetic first-pass coding sheet with every shown and hidden column."""
    n = n_confirmed + n_doubted
    df = pd.DataFrame({c: [f"{c}_{i}" for i in range(n)] for c in SHOWN_COLUMNS + HIDDEN_COLUMNS})
    df["fragment_id"] = range(n)
    df["representative_run"] = 1
    df["coder_doubts_gold_standard"] = ["n"] * n_confirmed + ["y"] * n_doubted
    return df


def test_selects_all_confirmed_and_sampled_doubted():
    """All confirmed rows plus exactly n_doubted doubted rows, each once."""
    rows = select_recode_rows(_coding(), n_doubted=3, seed=1)
    assert len(rows) == 9
    assert (rows["coder_doubts_gold_standard"] == "n").sum() == 6
    assert rows["fragment_id"].is_unique
    assert rows["recode_id"].tolist() == [f"R{i}" for i in range(1, 10)]


def test_selection_is_reproducible():
    """Same seed gives the same sample and order."""
    a = select_recode_rows(_coding(), n_doubted=3, seed=7)["fragment_id"].tolist()
    b = select_recode_rows(_coding(), n_doubted=3, seed=7)["fragment_id"].tolist()
    assert a == b


def test_blinded_sheet_hides_first_pass_fields(tmp_path):
    """The Recode tab shows materials and blank entry columns, never first-pass judgments."""
    rows = select_recode_rows(_coding(), n_doubted=3, seed=1)
    codes = pd.DataFrame({"code": ["F5.1"], "checkpoint_failure": ["x"]})
    sheet, key = write_recode_sheet(rows, codes, tmp_path / "s.xlsx", tmp_path / "k.csv")
    recode = pd.read_excel(sheet, sheet_name="Recode")
    assert list(recode.columns) == ["recode_id", *SHOWN_COLUMNS, *ENTRY_COLUMNS]
    first_pass_values = set(rows["coder_notes"]) | set(rows["coding_justification"])
    assert not first_pass_values & set(recode.astype(str).values.ravel())
    assert recode[ENTRY_COLUMNS].isna().all().all()
    assert len(pd.read_csv(key)) == len(rows)
