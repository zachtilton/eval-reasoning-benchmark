"""
Supplementary triangulation check for the ambiguous-coherence bucket.

NOT part of the official scoring pipeline (see pipeline.py / coherence.py
for that) -- this is a one-time diagnostic added after the scoring run
to inform how to size and target the human RC3 check on the 1,217
responses the pipeline flagged 'ambiguous' (Tier 1 rule-based and Tier 2
Haiku disagreed on coherence).

Design: a THIRD independent automated judge, gpt-5.6-luna, is asked the
exact same minimal-information question Tier 2 asks (classification +
rationale only -- see config.LLM_COHERENCE_PROMPT). Deliberately not
claude-haiku-4-5-20251001 (already Tier 2 -- re-asking it is not
independent triangulation) and not gpt_5_6_sol (one of the 6 benchmarked
models under study -- using it to grade study outputs is self-referential).
Luna is the cheapest tier of the same generation as the primary diagnostic
model, from a vendor entirely outside Tier 2's Anthropic lineage.

Because every ambiguous row by construction already has
coherence_rule_based != coherence_llm, Luna's vote always resolves the
pair into a 2-of-3 majority, unless Luna itself returns 'ambiguous' (a
genuine 3-way impasse).

IMPORTANT interpretive limit: this measures RELIABILITY (do independent
automated methods agree with each other), not VALIDITY (does any of them
match a human's judgment of actual coherence). Three automated judges can
converge on the same wrong answer by sharing systematic biases. This
triages which ambiguous cases most need the human RC3 check -- it does
not substitute for it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import requests

from .config import (
    LLM_COHERENCE_PROMPT,
    LOGS_DIR,
    RC3_COHERENT_SAMPLE,
    RC3_INCOHERENT_SAMPLE,
    SCORED_DB_PATH,
    TRIANGULATION_ENDPOINT,
    TRIANGULATION_MODEL_ID,
    TRIANGULATION_RESULTS_PATH,
)

MERGE_KEYS = ["fragment_id", "model_family", "prompt_condition", "run_number"]

HUMAN_REVIEW_BLINDED_PATH = LOGS_DIR / "human_review_blinded.csv"
HUMAN_REVIEW_KEY_PATH = LOGS_DIR / "human_review_key.csv"


def luna_coherence_check(
    classification: str | None,
    rationale_text: str | None,
    api_key: str,
) -> tuple[str, str]:
    """
    Query gpt-5.6-luna with the same minimal-information prompt Tier 2 uses.

    Args:
        classification: Normalized classification ('sound' or 'not_sound').
        rationale_text: Rationale text from model output.
        api_key: OpenAI API key (Responses API, Bearer auth).

    Returns:
        Tuple of (coherence_result, raw_response_text).
        coherence_result is 'coherent', 'incoherent', or 'ambiguous'
        ('ambiguous' on a call failure or an unparseable response, same
        failure-mode convention as coherence.llm_coherence_check).
    """
    if not classification or not rationale_text:
        return "incoherent", "missing_input"

    display_class = classification.replace("_", " ")
    prompt = LLM_COHERENCE_PROMPT.format(
        classification=display_class,
        rationale=rationale_text,
    )

    try:
        response = requests.post(
            TRIANGULATION_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": TRIANGULATION_MODEL_ID,
                "input": prompt,
                "max_output_tokens": 20,
            },
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        # Responses API: "output" is a list of items; the message item's
        # content holds an "output_text" block (see live diagnostic,
        # 2026-11-04, confirming this exact shape for gpt-5.6-luna).
        text = ""
        for item in body.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        text = block.get("text", "")
                        break

        result_text = text.strip().lower()

        if "incoherent" in result_text:
            return "incoherent", result_text
        if "coherent" in result_text:
            return "coherent", result_text
        return "ambiguous", result_text or "unparseable_response"

    except Exception as exc:
        print(f"  Luna triangulation call failed: {exc}")
        return "ambiguous", f"error: {exc}"


def run_triangulation(
    scored_db_path: Path = SCORED_DB_PATH,
    output_path: Path = TRIANGULATION_RESULTS_PATH,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    """
    Run the triangulation judge over every ambiguous-flagged response.

    Resumable: any (fragment_id, model_family, prompt_condition, run_number)
    already present in output_path is skipped, so an interrupted run can
    simply be re-invoked without re-paying for completed calls.

    Args:
        scored_db_path: Path to the scored database (post coherence.py).
        output_path: Path to write/append triangulation results.
        checkpoint_every: Write progress to disk every N new calls.

    Returns:
        DataFrame of triangulation results for all ambiguous cases.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — load .env first.")

    db = pd.read_csv(scored_db_path)
    ambiguous = db[db["coherence_final"] == "ambiguous"].reset_index(drop=True)

    if output_path.exists():
        done = pd.read_csv(output_path)
        done_keys = set(
            done[MERGE_KEYS].itertuples(index=False, name=None)
        )
    else:
        done = pd.DataFrame()
        done_keys = set()

    already_done_mask = (
        ambiguous[MERGE_KEYS]
        .apply(tuple, axis=1)
        .isin(done_keys)
    )
    todo = ambiguous[~already_done_mask]

    print(
        f"Triangulation: {len(ambiguous)} ambiguous cases, "
        f"{len(done_keys)} already done, {len(todo)} remaining."
    )

    new_records: list[dict] = []
    for pos, (_, row) in enumerate(todo.iterrows(), start=1):
        verdict, raw = luna_coherence_check(
            row["classification_output"], row["rationale_text"], api_key
        )
        new_records.append({
            **{k: row[k] for k in MERGE_KEYS},
            "coherence_rule_based": row["coherence_rule_based"],
            "coherence_llm": row["coherence_llm"],
            "luna_verdict": verdict,
            "luna_raw": raw,
        })

        if pos % checkpoint_every == 0 or pos == len(todo):
            combined = pd.concat(
                [done, pd.DataFrame(new_records)], ignore_index=True
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)
            print(f"  Checkpointed {pos}/{len(todo)} new calls ({len(combined)} total)")

    return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(new_records)


def summarize_triangulation(triangulation_df: pd.DataFrame) -> dict:
    """
    Summarize the triage split for the ambiguous bucket.

    Every ambiguous row by definition already has
    coherence_rule_based != coherence_llm (that disagreement is what
    produced the 'ambiguous' flag), so Luna's vote always breaks the tie
    into a 2-of-3 majority, unless Luna itself returns 'ambiguous'.

    Args:
        triangulation_df: Output of :func:`run_triangulation`.

    Returns:
        Dict with counts for each triage bucket and the majority-coherent
        fraction (the pass-rate sensitivity signal this check was for —
        NOT a validity claim; see module docstring).
    """
    df = triangulation_df
    total = len(df)

    sides_with_llm = (df["luna_verdict"] == df["coherence_llm"]).sum()
    sides_with_rule = (df["luna_verdict"] == df["coherence_rule_based"]).sum()
    three_way_split = (df["luna_verdict"] == "ambiguous").sum()

    majority_coherent = (
        ((df["luna_verdict"] == df["coherence_llm"]) & (df["coherence_llm"] == "coherent"))
        | ((df["luna_verdict"] == df["coherence_rule_based"]) & (df["coherence_rule_based"] == "coherent"))
    ).sum()

    summary = {
        "total_ambiguous_checked": int(total),
        "luna_sides_with_tier2_llm": int(sides_with_llm),
        "luna_sides_with_tier1_rule": int(sides_with_rule),
        "three_way_split_unresolved": int(three_way_split),
        "majority_leans_coherent": int(majority_coherent),
        "majority_leans_coherent_pct": round(majority_coherent / total, 4) if total else None,
    }

    print("\nTriangulation Triage Summary (ambiguous bucket):")
    print(f"  Total checked:                  {total}")
    print(f"  Luna sides with Tier 2 (Haiku):   {sides_with_llm} ({sides_with_llm/total:.1%})")
    print(f"  Luna sides with Tier 1 (rule):    {sides_with_rule} ({sides_with_rule/total:.1%})")
    print(f"  3-way split (Luna='ambiguous'):   {three_way_split} ({three_way_split/total:.1%})")
    print(f"  Majority leans COHERENT:          {majority_coherent} ({majority_coherent/total:.1%})")
    print(f"  (reliability triage only — not a validity claim; human RC3 still required)")

    return summary


# ---------------------------------------------------------------------------
# Stratified human review sample (RC3, ambiguous arm informed by triage)
# ---------------------------------------------------------------------------

def _margin_of_error_sample_size(
    population: int,
    margin: float = 0.10,
    confidence: float = 0.95,
) -> int:
    """
    Minimum sample size for a proportion estimate at a given margin of error.

    Worst-case variance (p=0.5), with finite-population correction since
    none of this project's strata are large relative to the sample needed.

    Args:
        population: Size of the stratum being sampled from.
        margin: Desired margin of error (e.g. 0.10 for ±10 points).
        confidence: Confidence level (default 0.95).

    Returns:
        Required sample size, capped at the population size.
    """
    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2)
    n_infinite = (z ** 2) * 0.25 / (margin ** 2)
    n_adjusted = n_infinite / (1 + (n_infinite - 1) / population)
    return min(population, math.ceil(n_adjusted))


def select_stratified_human_review_sample(
    scored_db_path: Path = SCORED_DB_PATH,
    triangulation_path: Path = TRIANGULATION_RESULTS_PATH,
    blinded_output_path: Path = HUMAN_REVIEW_BLINDED_PATH,
    key_output_path: Path = HUMAN_REVIEW_KEY_PATH,
    seed: int = 42,
    coherent_n: int = RC3_COHERENT_SAMPLE,
    incoherent_n: int = RC3_INCOHERENT_SAMPLE,
    ambiguous_margin: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the RC3 human-review sample, sized to a ~10pp margin of error
    per stratum, with the ambiguous arm stratified by the Luna
    triangulation triage rather than drawn as one flat random sample.

    Coherent and incoherent strata: simple random samples sized by
    :func:`_margin_of_error_sample_size` (RC3_COHERENT_SAMPLE=94,
    RC3_INCOHERENT_SAMPLE=67 at the default 10pp margin).

    Ambiguous stratum, three sub-groups from triangulation.py's triage:
      - 3-way split (Luna itself said 'ambiguous'): FULL CENSUS. Smallest
        group (64) and the hardest cases -- most informative per rating.
      - Luna sides with Tier 1 rule-based (73): sampled to the same 10pp
        margin, proportionally allocated against the next group.
      - Luna sides with Tier 2 Haiku (1,080): sampled to the same 10pp
        margin, proportionally allocated.
    The latter two are sampled together as one pool so the ~10pp target
    is hit on their combined size, then split proportionally -- avoids
    over-sampling the much larger "sides with Haiku" group at the expense
    of the much smaller "sides with rule" group.

    Writes two files:
      - blinded_output_path: shuffled, reviewer-facing. Only
        classification_output + rationale_text (the same minimal
        information Tier 2 and Luna were given) + a row_id for merge-back
        and a blank rating column. No stratum label, no automated
        verdicts, no model/prompt/run identity -- preserves the same
        blind-independence design as every other coherence check in this
        pipeline.
      - key_output_path: internal mapping (row_id -> merge keys, stratum,
        and all automated verdicts) for merging ratings back and
        computing kappa per stratum. Not for the reviewer to consult
        before rating.

    Args:
        scored_db_path: Path to the scored database.
        triangulation_path: Path to the triangulation results CSV.
        blinded_output_path: Output path for the reviewer-facing sheet.
        key_output_path: Output path for the internal key.
        seed: Random seed for reproducibility.
        coherent_n: Sample size for the coherent stratum.
        incoherent_n: Sample size for the incoherent stratum.
        ambiguous_margin: Margin of error target for the two sampled
            ambiguous sub-groups (the 3-way-split group is always a
            full census regardless of this value).

    Returns:
        Tuple of (blinded_df, key_df).
    """
    db = pd.read_csv(scored_db_path)
    tri = pd.read_csv(triangulation_path)

    coherent_pool = db[db["coherence_final"] == "coherent"]
    incoherent_pool = db[db["coherence_final"] == "incoherent"]

    coherent_sample = coherent_pool.sample(
        n=min(coherent_n, len(coherent_pool)), random_state=seed
    )
    incoherent_sample = incoherent_pool.sample(
        n=min(incoherent_n, len(incoherent_pool)), random_state=seed
    )

    # Triangulation triage sub-groups (mutually exclusive, exhaustive --
    # confirmed in summarize_triangulation: counts sum to len(tri)).
    three_way = tri[tri["luna_verdict"] == "ambiguous"]
    sides_with_llm = tri[
        (tri["luna_verdict"] == tri["coherence_llm"]) & (tri["luna_verdict"] != "ambiguous")
    ]
    sides_with_rule = tri[
        (tri["luna_verdict"] == tri["coherence_rule_based"]) & (tri["luna_verdict"] != "ambiguous")
    ]

    remainder_pool_size = len(sides_with_llm) + len(sides_with_rule)
    remainder_n = _margin_of_error_sample_size(remainder_pool_size, margin=ambiguous_margin)
    n_from_llm = round(remainder_n * len(sides_with_llm) / remainder_pool_size)
    n_from_rule = remainder_n - n_from_llm

    sample_llm = sides_with_llm.sample(n=min(n_from_llm, len(sides_with_llm)), random_state=seed)
    sample_rule = sides_with_rule.sample(n=min(n_from_rule, len(sides_with_rule)), random_state=seed)

    # Pull full rows (classification_output, rationale_text) from the
    # scored database -- triangulation.csv only carries merge keys +
    # automated verdicts, not the response content itself.
    def _with_stratum(tri_subset: pd.DataFrame, stratum: str) -> pd.DataFrame:
        merged = db.merge(tri_subset[MERGE_KEYS], on=MERGE_KEYS, how="inner")
        merged["stratum"] = stratum
        return merged

    ambiguous_census = _with_stratum(three_way, "ambiguous_3way_split")
    ambiguous_llm = _with_stratum(sample_llm, "ambiguous_sides_with_haiku")
    ambiguous_rule = _with_stratum(sample_rule, "ambiguous_sides_with_rule")

    coherent_sample = coherent_sample.copy()
    coherent_sample["stratum"] = "coherent_confident"
    incoherent_sample = incoherent_sample.copy()
    incoherent_sample["stratum"] = "incoherent_confident"

    full_sample = pd.concat(
        [coherent_sample, incoherent_sample, ambiguous_census, ambiguous_llm, ambiguous_rule],
        ignore_index=True,
    )
    full_sample = full_sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    full_sample["row_id"] = range(1, len(full_sample) + 1)

    blinded_df = full_sample[["row_id", "classification_output", "rationale_text"]].copy()
    blinded_df["your_rating"] = ""

    key_df = full_sample[
        MERGE_KEYS + ["row_id", "stratum", "coherence_rule_based", "coherence_llm", "coherence_final"]
    ].copy()

    blinded_output_path.parent.mkdir(parents=True, exist_ok=True)
    blinded_df.to_csv(blinded_output_path, index=False)
    key_df.to_csv(key_output_path, index=False)

    print("\nStratified Human Review Sample:")
    print(f"  Coherent (confident):          {len(coherent_sample)} / {len(coherent_pool)} available")
    print(f"  Incoherent (confident):        {len(incoherent_sample)} / {len(incoherent_pool)} available")
    print(f"  Ambiguous — 3-way split:       {len(ambiguous_census)} / {len(three_way)} available (full census)")
    print(f"  Ambiguous — sides with Haiku:  {len(ambiguous_llm)} / {len(sides_with_llm)} available")
    print(f"  Ambiguous — sides with rule:   {len(ambiguous_rule)} / {len(sides_with_rule)} available")
    print(f"  TOTAL:                         {len(full_sample)}")
    print(f"\n  Blinded rating sheet: {blinded_output_path}")
    print(f"  Internal key:         {key_output_path}")

    return blinded_df, key_df
