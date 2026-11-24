"""
RC4 failure-coding temporal consistency: build the blinded recode sheet.

Design (pre-specified, docs/findings/analysis_log.md P2 / D4): 56
fragment x condition rows from the first-pass coding sheet (all 42
confirmed-failure rows plus 14 of the 70 rows where the coder doubted the
gold label, seed 42), presented in random order with every first-pass
coding field hidden. The recoder decides disposition first, then codes the
rows judged confirmed.

Outputs:
    reliability/rc4/rc4_recode_blinded.xlsx  - what the recoder sees and fills in
    reliability/rc4/rc4_recode_key.csv       - recode_id -> fragment_id, condition
                                               (do not open until recoding is done)

Run: python3 -m src.analysis.rc4_recode
"""

import re
from pathlib import Path

import pandas as pd

from src.analysis.config import (
    APPENDIX_D_PATH,
    FAILURE_CODES_PATH,
    RC4_BLINDED_SHEET_PATH,
    RC4_DIR,
    RC4_KEY_PATH,
    RC4_N_DOUBTED_SAMPLE,
    RC4_SEED,
)

# Columns the recoder sees (first-pass materials only)
SHOWN_COLUMNS: list[str] = [
    "criterion", "prompt_condition", "fragment_text",
    "gpt_classification", "gpt_rationale", "gold_classification", "gold_rationale",
]

# First-pass judgments and anything written after first coding: never shown
HIDDEN_COLUMNS: list[str] = [
    "primary_code", "secondary_code", "tertiary_code", "domain", "code_type",
    "coding_justification", "coding_date", "coding_session", "coder_doubts_gold_standard",
    "coder_notes", "rating_rationale_mismatch", "rating_revised_prelock", "gold_rationale_updated",
]

# Blank columns the recoder fills in
ENTRY_COLUMNS: list[str] = [
    "disposition", "primary_code", "secondary_code", "tertiary_code",
    "coding_justification", "recoder_notes",
]

DISPOSITIONS: list[str] = ["confirmed_failure", "doubts_gold"]


def select_recode_rows(coding: pd.DataFrame, n_doubted: int = RC4_N_DOUBTED_SAMPLE,
                       seed: int = RC4_SEED) -> pd.DataFrame:
    """
    Select the RC4 sample and shuffle it.

    Args:
        coding: First-pass coding sheet (failure_codes_gpt5.csv).
        n_doubted: Number of doubted-gold rows to sample.
        seed: Random seed for sampling and ordering.

    Returns:
        Selected rows in presentation order, with a `recode_id` column (R01...).
    """
    doubt = coding["coder_doubts_gold_standard"].astype(str).str.strip().str.lower()
    confirmed = coding[doubt == "n"]
    doubted = coding[doubt == "y"].sample(n=n_doubted, random_state=seed)
    chosen = pd.concat([confirmed, doubted]).sample(frac=1, random_state=seed).reset_index(drop=True)
    width = len(str(len(chosen)))
    chosen.insert(0, "recode_id", [f"R{i + 1:0{width}d}" for i in range(len(chosen))])
    return chosen


def load_code_list(path: Path = APPENDIX_D_PATH) -> pd.DataFrame:
    """
    Read the deductive code list (F1.1-F6.6) from the Appendix D markdown copy.

    Args:
        path: docs/appendix_d_failure_coding_schema.md.

    Returns:
        DataFrame with columns code, checkpoint_failure.
    """
    rows = re.findall(r"^\|\s*(F[1-6]\.\d+)\s*\|\s*(.+?)\s*\|\s*$", path.read_text(), flags=re.M)
    return pd.DataFrame(rows, columns=["code", "checkpoint_failure"])


INSTRUCTIONS: list[str] = [
    "RC4 blind recode (failure-coding temporal consistency). Design: analysis_log.md P2.",
    "",
    "Do not open data/analysis/failure_codes_gpt5.csv or reliability/rc4/rc4_recode_key.csv until every row is done.",
    "",
    "For each row, follow the first-pass procedure:",
    "1. Read the fragment, GPT's classification and rationale, and the gold classification and rationale.",
    "2. Disposition: 'confirmed_failure' if GPT's verdict is a genuine failure against the gold standard;",
    "   'doubts_gold' if you doubt the gold label instead.",
    "3. Only if confirmed_failure: assign primary (required), secondary and tertiary (optional) codes",
    "   from the Codes tab (F1.1-F6.6), or an emergent code (write it out and describe it in the notes).",
    "4. Write a short justification. Use recoder_notes for anything else.",
    "",
    "Rows are in random order. Some rows are the same fragment in the other prompt condition; code each row on its own.",
]


def write_recode_sheet(rows: pd.DataFrame, codes: pd.DataFrame,
                       sheet_path: Path = RC4_BLINDED_SHEET_PATH, key_path: Path = RC4_KEY_PATH) -> list[Path]:
    """
    Write the blinded workbook (Instructions, Recode, Codes tabs) and the unblinding key.

    Args:
        rows: Output of select_recode_rows.
        codes: Output of load_code_list.
        sheet_path: Workbook destination.
        key_path: Key CSV destination.

    Returns:
        Paths written.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    RC4_DIR.mkdir(parents=True, exist_ok=True)
    blinded = rows[["recode_id", *SHOWN_COLUMNS]].copy()
    for col in ENTRY_COLUMNS:
        blinded[col] = ""
    with pd.ExcelWriter(sheet_path, engine="openpyxl") as xw:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(xw, sheet_name="Instructions", index=False, header=False)
        blinded.to_excel(xw, sheet_name="Recode", index=False)
        codes.to_excel(xw, sheet_name="Codes", index=False)

        ws = xw.sheets["Recode"]
        widths = {"recode_id": 9, "criterion": 14, "prompt_condition": 12, "fragment_text": 80,
                  "gpt_classification": 12, "gpt_rationale": 60, "gold_classification": 12,
                  "gold_rationale": 50, "disposition": 18, "primary_code": 11, "secondary_code": 11,
                  "tertiary_code": 11, "coding_justification": 45, "recoder_notes": 35}
        entry_fill = PatternFill("solid", fgColor="FDF4C4")
        for j, col in enumerate(blinded.columns, start=1):
            letter = ws.cell(row=1, column=j).column_letter
            ws.column_dimensions[letter].width = widths.get(col, 15)
            ws.cell(row=1, column=j).font = Font(bold=True)
            for i in range(2, len(blinded) + 2):
                cell = ws.cell(row=i, column=j)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if col in ENTRY_COLUMNS:
                    cell.fill = entry_fill
        ws.freeze_panes = "B2"

        last = len(blinded) + 1
        disp_col = blinded.columns.get_loc("disposition") + 1
        dv = DataValidation(type="list", formula1='"' + ",".join(DISPOSITIONS) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        letter = ws.cell(row=1, column=disp_col).column_letter
        dv.add(f"{letter}2:{letter}{last}")

        xw.sheets["Instructions"].column_dimensions["A"].width = 110
        xw.sheets["Codes"].column_dimensions["A"].width = 8
        xw.sheets["Codes"].column_dimensions["B"].width = 70

    rows[["recode_id", "fragment_id", "prompt_condition", "representative_run"]].to_csv(key_path, index=False)
    return [sheet_path, key_path]


def run_build_recode_sheet() -> list[Path]:
    """
    Build the RC4 blinded recode sheet and key from the first-pass coding sheet.

    Returns:
        Paths written.
    """
    coding = pd.read_csv(FAILURE_CODES_PATH)
    return write_recode_sheet(select_recode_rows(coding), load_code_list())


if __name__ == "__main__":
    for p in run_build_recode_sheet():
        print(p)
