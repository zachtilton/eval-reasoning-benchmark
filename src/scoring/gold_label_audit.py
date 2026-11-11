"""
Blinded audit of the locked gold standard for rating/rationale mismatches.

During GPT failure coding the coder found gold rows whose expert_classification
contradicts the expert's own written rationale -- a transcription error in the
source spreadsheet, not a change of judgment. Because those were found only
where models disagreed (detection bias), this audit checks all 150 rows
independently of model outputs: Claude Haiku (temperature 0, as in Tier 2)
reads ONLY each rationale and reports the label it implies. Disagreements,
coder-flagged rows, and verbatim-duplicate rationales carrying conflicting
labels are queued for human adjudication. The locked gold file is never
modified; adjudicated errata feed a separate sensitivity analysis.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .accuracy import load_gold_standard
from .config import (
    ANTHROPIC_API_VERSION,
    FAILURE_CODES_GPT5_PATH,
    GOLD_ADJUDICATION_PATH,
    GOLD_AUDIT_MAX_TOKENS,
    GOLD_AUDIT_PROMPT,
    GOLD_AUDIT_RAW_PATH,
    HAIKU_MODEL_ID,
)

LABEL_TOKENS = {"NOT_SOUND": "not_sound", "NOT SOUND": "not_sound", "SOUND": "sound", "UNCLEAR": "unclear"}

# Rows flagged in an unblinded first-pass read (2026-11-10, Claude, label
# visible) beyond the coder's own flags -- kept separate from the blind call.
UNBLINDED_SCREEN_FLAGS = {1977, 5946, 1877, 1071}


def parse_audit_response(text: str) -> str:
    """
    Extract the implied label from the first line of a Haiku audit response.

    NOT_SOUND is checked before SOUND because "SOUND" is a substring of it.

    Args:
        text: Raw response text.

    Returns:
        'sound', 'not_sound', 'unclear', or 'parse_error'.
    """
    first = text.strip().splitlines()[0].strip().upper().strip("*.: ") if text.strip() else ""
    for token in ("NOT_SOUND", "NOT SOUND", "UNCLEAR", "SOUND"):
        if first.startswith(token):
            return LABEL_TOKENS[token]
    return "parse_error"


def implied_label_from_rationale(rationale: str, api_key: str) -> tuple[str, str]:
    """
    Ask Haiku which label a gold rationale implies, seeing the rationale only.

    Args:
        rationale: Expert rationale text.
        api_key: Anthropic API key.

    Returns:
        Tuple (implied_label, raw_response). On API failure implied_label is
        'api_error' so the row is queued for adjudication, never silently
        assigned.
    """
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
            json={
                "model": HAIKU_MODEL_ID,
                "messages": [{"role": "user", "content": GOLD_AUDIT_PROMPT.format(rationale=rationale)}],
                "max_tokens": GOLD_AUDIT_MAX_TOKENS,
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["content"][0]["text"].strip()
        return parse_audit_response(raw), raw
    except Exception as exc:
        return "api_error", f"error: {exc}"


def run_blind_audit(raw_path=GOLD_AUDIT_RAW_PATH) -> pd.DataFrame:
    """
    Run the blind Haiku audit over every gold rationale, resuming if partial.

    Writes after each call so an interruption loses nothing; rows already
    present in raw_path (without api_error) are not re-requested.

    Args:
        raw_path: CSV of raw audit results (fragment_id, implied label, raw).

    Returns:
        DataFrame of audit results for all gold rows.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- run: set -a; source .env; set +a")

    gold = load_gold_standard()
    done = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame(
        columns=["fragment_id", "haiku_implied_label", "haiku_raw_response", "model_id", "audited_at"]
    )
    done = done[done["haiku_implied_label"] != "api_error"]
    todo = gold[~gold["fragment_id"].isin(done["fragment_id"])]
    print(f"Blind audit: {len(done)} already done, {len(todo)} to request")

    for i, row in enumerate(todo.itertuples(), 1):
        label, raw = implied_label_from_rationale(str(row.rationale), api_key)
        done = pd.concat([done, pd.DataFrame([{
            "fragment_id": row.fragment_id,
            "haiku_implied_label": label,
            "haiku_raw_response": raw,
            "model_id": HAIKU_MODEL_ID,
            "audited_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }])], ignore_index=True)
        done.to_csv(raw_path, index=False)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(0.2)

    print(done["haiku_implied_label"].value_counts().to_string())
    return done


def build_adjudication_sheet(
    audit_df: pd.DataFrame,
    output_path=GOLD_ADJUDICATION_PATH,
    failure_codes_path=FAILURE_CODES_GPT5_PATH,
) -> pd.DataFrame:
    """
    Merge the blind audit with all other mismatch signals into one sheet.

    A row needs adjudication if any of: Haiku's implied label differs from
    the gold label (or is unclear/error); the coder flagged
    rating_rationale_mismatch; the unblinded screen flagged it; or its
    rationale is a verbatim duplicate of another row's carrying a different
    label (the rationale itself may be the pasted-in error).

    Args:
        audit_df: Output of run_blind_audit.
        output_path: Where to write the adjudication sheet.
        failure_codes_path: Coding sheet carrying rating_rationale_mismatch.

    Returns:
        Adjudication DataFrame (all 150 rows, queued rows first).
    """
    gold = load_gold_standard()[["fragment_id", "criterion", "expert_classification", "rationale", "fragment"]]
    df = gold.merge(audit_df[["fragment_id", "haiku_implied_label", "haiku_raw_response"]], on="fragment_id", how="left")
    df["haiku_reason"] = df["haiku_raw_response"].fillna("").str.split("\n").str[1:].str.join(" ").str.strip()

    codes = pd.read_csv(failure_codes_path)
    coder_flags = set(codes.loc[codes["rating_rationale_mismatch"] == "y", "fragment_id"])
    df["in_gpt_failure_subsample"] = df["fragment_id"].isin(set(codes["fragment_id"]))
    df["coder_flagged_mismatch"] = df["fragment_id"].isin(coder_flags)
    df["unblinded_screen_flag"] = df["fragment_id"].isin(UNBLINDED_SCREEN_FLAGS)
    df["haiku_disagrees"] = df["haiku_implied_label"] != df["expert_classification"]

    norm = df["rationale"].fillna("").str.split().str.join(" ").str.lower()
    df["duplicate_rationale_group"] = norm.map(
        {t: i for i, t in enumerate(norm[norm.duplicated(keep=False)].unique(), 1)}
    ).astype("Int64")
    labels_per_group = df.groupby("duplicate_rationale_group")["expert_classification"].nunique()
    df["duplicate_label_conflict"] = df["duplicate_rationale_group"].map(labels_per_group).fillna(0) > 1

    df["needs_adjudication"] = (
        df["haiku_disagrees"] | df["coder_flagged_mismatch"]
        | df["unblinded_screen_flag"] | df["duplicate_label_conflict"]
    )

    # Human adjudication columns, filled by the coder.
    df["adjudicated_label"] = ""        # sound | not_sound | unresolvable
    df["error_location"] = ""           # label | rationale | none | unresolvable
    df["adjudication_notes"] = ""

    df = df.sort_values(["needs_adjudication", "coder_flagged_mismatch", "fragment_id"], ascending=[False, False, True])
    cols = [
        "fragment_id", "needs_adjudication", "expert_classification", "haiku_implied_label",
        "rationale", "haiku_reason", "coder_flagged_mismatch", "unblinded_screen_flag",
        "haiku_disagrees", "duplicate_rationale_group", "duplicate_label_conflict",
        "in_gpt_failure_subsample", "criterion", "adjudicated_label", "error_location",
        "adjudication_notes", "fragment",
    ]
    df = df[cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nAdjudication sheet: {output_path}")
    print(f"  Rows needing adjudication: {int(df['needs_adjudication'].sum())} / {len(df)}")
    return df


if __name__ == "__main__":
    build_adjudication_sheet(run_blind_audit())
