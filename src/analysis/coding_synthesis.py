"""
Synthesis of the GPT failure-coding exercise (secondary findings).

The coder reviewed every GPT fragment x prompt-condition failure against the
locked gold standard and either coded it as a genuine failure (Appendix D.1)
or marked coder_doubts_gold_standard = y. This module quantifies that
disposition and what it implies: error direction (lenient vs. strict),
prompt-condition shifts, the effect of the anchor's pre-lock rating
revisions, and concordance with the external reviewer on the contested
subset. Domain/co-occurrence analysis of genuine failures reuses H.3
(failure_patterns.synthesize_failure_patterns).

All figures here are secondary to the locked-gold primary results: the
coder's doubts are reconsiderations made with GPT's rationale in view.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.scoring.reviewer_substitution import rescore_with_label_overrides

from .config import (
    CODING_SYNTHESIS_RESULTS_PATH,
    CONTESTED_REVIEW_RESOLVED_PATH,
    CORE_CRITERIA,
    DOMAIN_LABELS,
    FAILURE_CODES_PATH,
    FIGURE_DPI,
    FIGURES_DIR,
    FRAGMENT_OUTCOMES_CORRECTED_PATH,
    GOLD_RATIONALE_SUPPLEMENT_PATH,
    MODEL_ORDER,
    PRIMARY_MODEL,
    PROMPT_ORDER,
    SCORED_DB_CORRECTED_PATH,
)
from .failure_patterns import synthesize_failure_patterns

CODE_COLS = ["primary_code", "secondary_code", "tertiary_code"]
LENIENT, STRICT = "lenient", "strict"


def load_coding_sheet(path: Path = FAILURE_CODES_PATH) -> pd.DataFrame:
    """
    Load the coding sheet with normalized codes and derived flags.

    Args:
        path: failure_codes_gpt5.csv.

    Returns:
        Row-level (fragment x condition) DataFrame with `genuine` (coder agrees
        with gold) and `direction` (lenient = GPT said sound against a
        not_sound gold; strict = the reverse).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in CODE_COLS:
        df[c] = df[c].str.strip().str.replace(r"^F\.(\d)", r"F\1", regex=True)
    df["fragment_id"] = df["fragment_id"].astype(int)
    df["genuine"] = df["coder_doubts_gold_standard"].str.strip().str.lower() == "n"
    df["direction"] = np.where(df["gpt_classification"] == "sound", LENIENT, STRICT)
    return df


def to_fragment_level(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse fragment x condition rows to one row per fragment.

    Disposition and codes are identical across conditions for every fragment
    in the sheet (checked in validate_consistency), so the first row stands
    for the fragment.

    Args:
        rows: Output of load_coding_sheet.

    Returns:
        Fragment-level DataFrame with a `conditions` column listing the
        prompt conditions in which GPT failed.
    """
    agg = rows.groupby("fragment_id").agg(
        conditions=("prompt_condition", lambda s: "+".join(sorted(s))),
        n_conditions=("prompt_condition", "count"),
    )
    first = rows.drop_duplicates("fragment_id").set_index("fragment_id")
    return first.join(agg).reset_index()


def validate_consistency(rows: pd.DataFrame) -> list[str]:
    """
    List fragments whose disposition or codes differ between conditions.

    Args:
        rows: Output of load_coding_sheet.

    Returns:
        Human-readable problems (empty when consistent).
    """
    problems = []
    for fid, g in rows.groupby("fragment_id"):
        for col in ["genuine"] + CODE_COLS:
            if g[col].nunique() > 1:
                problems.append(f"{fid}: {col} differs across conditions")
    coded = rows["primary_code"] != ""
    if (coded != rows["genuine"]).any():
        problems.append("primary_code present iff genuine is violated")
    return problems


def _split(series: pd.Series) -> dict:
    """Count and percentage of genuine vs. doubted in a boolean series."""
    n = len(series)
    g = int(series.sum())
    return {
        "n": n,
        "genuine": g,
        "doubt_gold": n - g,
        "genuine_pct": round(100 * g / n, 1) if n else None,
        "doubt_gold_pct": round(100 * (n - g) / n, 1) if n else None,
    }


def disposition_summary(rows: pd.DataFrame, frags: pd.DataFrame, n_gpt_cases: int) -> dict:
    """
    Genuine-vs-doubted split, plus failure rates against all GPT cases.

    Args:
        rows: Row-level coding sheet.
        frags: Fragment-level coding sheet.
        n_gpt_cases: Total GPT fragment x condition cases scored (denominator).

    Returns:
        Dict of fragment- and row-level splits and failure rates.
    """
    return {
        "fragments": _split(frags["genuine"]),
        "rows": _split(rows["genuine"]),
        "gpt_cases_total": n_gpt_cases,
        "failure_rate_locked_gold_pct": round(100 * len(rows) / n_gpt_cases, 1),
        "failure_rate_coder_confirmed_pct": round(100 * rows["genuine"].sum() / n_gpt_cases, 1),
    }


def direction_summary(frags: pd.DataFrame) -> dict:
    """
    Disposition by error direction (lenient vs. strict), fragment level.

    Args:
        frags: Fragment-level coding sheet.

    Returns:
        {direction: split} for lenient and strict errors.
    """
    return {d: _split(frags.loc[frags["direction"] == d, "genuine"]) for d in (LENIENT, STRICT)}


def condition_summary(rows: pd.DataFrame, n_per_condition: dict) -> dict:
    """
    Error direction and disposition per prompt condition, row level.

    Args:
        rows: Row-level coding sheet.
        n_per_condition: {condition: GPT fragments scored in that condition}.

    Returns:
        Per condition: lenient/strict error counts, genuine/doubted splits,
        and the coder-confirmed failure rate.
    """
    out = {}
    for cond in PROMPT_ORDER:
        sub = rows[rows["prompt_condition"] == cond]
        out[cond] = {
            "failures": len(sub),
            "by_direction": {d: _split(sub.loc[sub["direction"] == d, "genuine"]) for d in (LENIENT, STRICT)},
            "genuine": int(sub["genuine"].sum()),
            "coder_confirmed_failure_rate_pct": round(100 * sub["genuine"].sum() / n_per_condition[cond], 1),
        }
    return out


def failure_only_condition_summary(frags: pd.DataFrame) -> dict:
    """
    Disposition by which conditions GPT failed in (zero-shot only, few-shot only, both).

    Args:
        frags: Fragment-level coding sheet.

    Returns:
        {conditions: split}.
    """
    return {c: _split(g["genuine"]) for c, g in frags.groupby("conditions")}


def code_profile(rows: pd.DataFrame, frags: pd.DataFrame) -> dict:
    """
    Code frequencies for genuine failures, plus the H.3 synthesis.

    Args:
        rows: Row-level coding sheet.
        frags: Fragment-level coding sheet.

    Returns:
        Fragment-level primary code and domain counts, any-position counts,
        codes per fragment, and H.3 output on genuine rows.
    """
    gf = frags[frags["genuine"]]
    n = len(gf)
    primary_domain = gf["primary_code"].str.extract(r"^F(\d)")[0].astype(int).value_counts()
    any_codes = gf[CODE_COLS].stack()
    any_codes = any_codes[any_codes != ""]
    domain_presence = {
        DOMAIN_LABELS[d]: int(gf[CODE_COLS].apply(lambda r: any(c.startswith(f"F{d}.") for c in r), axis=1).sum())
        for d in DOMAIN_LABELS
    }
    return {
        "n_genuine_fragments": n,
        "primary_codes": gf["primary_code"].value_counts().to_dict(),
        "primary_domain": {
            DOMAIN_LABELS[d]: {"n": int(primary_domain.get(d, 0)), "pct": round(100 * primary_domain.get(d, 0) / n, 1)}
            for d in DOMAIN_LABELS
        },
        "any_position_codes": any_codes.value_counts().to_dict(),
        "any_position_domain_presence": domain_presence,
        "codes_per_fragment": gf[CODE_COLS].apply(lambda r: int((r != "").sum()), axis=1).value_counts().sort_index().to_dict(),
        "h3_row_level": synthesize_failure_patterns(rows[rows["genuine"]].copy()),
    }


def criterion_summary(frags: pd.DataFrame) -> dict:
    """
    Disposition by evaluation criterion (non-DAC criteria grouped as other).

    Args:
        frags: Fragment-level coding sheet.

    Returns:
        {criterion: split}.
    """
    crit = frags["criterion"].str.lower().str.strip()
    crit = crit.where(crit.isin(CORE_CRITERIA), "other")
    return {c: _split(frags.loc[crit == c, "genuine"]) for c in CORE_CRITERIA + ["other"]}


def revision_summary(
    frags: pd.DataFrame,
    supplement: pd.DataFrame,
    scored_db: pd.DataFrame,
) -> dict:
    """
    Effect of the anchor's pre-lock rating revisions.

    Reports how many ratings were revised, how the coder dispositioned the
    revised fragments in GPT's failure set, and every model's fragment pass
    rate under the locked (revised) vs. original pre-revision ratings.

    Args:
        frags: Fragment-level coding sheet.
        supplement: gold_standard_rationale_supplement.csv.
        scored_db: RC2-corrected scored database.

    Returns:
        Dict with revision counts, coder disposition, and pass-rate table.
    """
    revised_ids = set(supplement["fragment_id"])
    in_set = frags[frags["fragment_id"].isin(revised_ids)]
    directions = (supplement["original_rating"] + "->" + supplement["final_rating_locked"]).value_counts().to_dict()

    rates = {}
    for name, overrides in [("locked", {}), ("pre_revision", dict(zip(supplement["fragment_id"], supplement["original_rating"])))]:
        db = rescore_with_label_overrides(scored_db, overrides, "syn")
        db["p"] = (db["run_outcome_syn"] == "pass").astype(int)
        frag_pass = db.groupby(["fragment_id", "model_family", "prompt_condition"])["p"].sum().ge(2)
        rates[name] = frag_pass.groupby(["model_family", "prompt_condition"]).mean() * 100

    table = pd.DataFrame(rates)
    table["delta"] = table["pre_revision"] - table["locked"]
    return {
        "n_revised": len(supplement),
        "revision_directions": directions,
        "revised_in_gpt_failure_set": _split(in_set["genuine"]),
        "pass_rate_locked_vs_pre_revision": {
            f"{m}|{c}": {k: round(float(v), 1) for k, v in table.loc[(m, c)].items()}
            for m, c in table.index
        },
        "mean_delta_pp": round(float(table["delta"].mean()), 1),
        "min_delta_pp": round(float(table["delta"].min()), 1),
        "max_delta_pp": round(float(table["delta"].max()), 1),
        "_table": table,
    }


def reviewer_concordance(frags: pd.DataFrame, resolved: pd.DataFrame) -> dict:
    """
    Compare coder disposition with the external reviewer on contested fragments.

    "Coder doubts gold" is paired with "reviewer sides with the models"
    (reviewer's label differs from the anchor's).

    Args:
        frags: Fragment-level coding sheet.
        resolved: contested_fragments_review_resolved.csv.

    Returns:
        Crosstab counts, percent agreement, and Cohen's kappa.
    """
    m = frags.merge(resolved[["fragment_id", "reviewer_classification", "expert_classification"]], on="fragment_id")
    coder_doubt = (~m["genuine"]).astype(int)
    reviewer_models = (m["reviewer_classification"] != m["expert_classification"]).astype(int)
    xt = pd.crosstab(coder_doubt.map({1: "coder_doubt", 0: "coder_genuine"}),
                     reviewer_models.map({1: "reviewer_sides_models", 0: "reviewer_sides_gold"}))
    return {
        "n": len(m),
        "crosstab": {r: xt.loc[r].to_dict() for r in xt.index},
        "reviewer_sides_models_pct": round(100 * reviewer_models.mean(), 1),
        "percent_agreement": round(100 * float((coder_doubt == reviewer_models).mean()), 1),
        "cohens_kappa": round(float(cohen_kappa_score(coder_doubt, reviewer_models)), 3),
    }


def _label_stacks(ax, lower: list, upper: list, min_inside: int = 3) -> None:
    """Annotate two-segment stacked bars; segments too thin for a label get it beside the bar."""
    for i, (lo, up) in enumerate(zip(lower, upper)):
        if lo >= min_inside:
            ax.text(i, lo / 2, str(lo), ha="center", va="center", color="white", fontweight="bold")
        elif lo > 0:
            ax.text(i + 0.42, lo / 2, str(lo), ha="left", va="center", color="#c0392b", fontweight="bold")
        if up > 0:
            ax.text(i, lo + up / 2, str(up), ha="center", va="center", fontweight="bold")


def make_figures(results: dict, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """
    Write poster-ready figures (PNG, FIGURE_DPI).

    Args:
        results: Output of run_coding_synthesis (before JSON export).
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    output_dir.mkdir(parents=True, exist_ok=True)
    c_gen, c_doubt = "#c0392b", "#95a5a6"
    paths = []

    # 1. Disposition by error direction
    d = results["direction"]
    labels = ["GPT lenient\n(said sound)", "GPT strict\n(said not sound)"]
    gen = [d[LENIENT]["genuine"], d[STRICT]["genuine"]]
    dbt = [d[LENIENT]["doubt_gold"], d[STRICT]["doubt_gold"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, gen, color=c_gen, label="Confirmed GPT failure")
    ax.bar(labels, dbt, bottom=gen, color=c_doubt, label="Coder doubts gold standard")
    _label_stacks(ax, gen, dbt)
    ax.set_ylabel("Fragments")
    ax.set_title("GPT disagreements with the gold standard, by direction")
    ax.legend(frameon=False)
    paths.append(output_dir / "fig1_disposition_by_direction.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=FIGURE_DPI); plt.close(fig)

    # 2. Prompt condition x direction, confirmed vs doubted
    cs = results["prompt_condition"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, cond in zip(axes, PROMPT_ORDER):
        bd = cs[cond]["by_direction"]
        g = [bd[LENIENT]["genuine"], bd[STRICT]["genuine"]]
        b = [bd[LENIENT]["doubt_gold"], bd[STRICT]["doubt_gold"]]
        ax.bar(["Lenient", "Strict"], g, color=c_gen, label="Confirmed GPT failure")
        ax.bar(["Lenient", "Strict"], b, bottom=g, color=c_doubt, label="Coder doubts gold")
        _label_stacks(ax, g, b)
        ax.set_title(f"{cond.replace('_', '-')}: {cs[cond]['genuine']} confirmed of {cs[cond]['failures']}")
    axes[0].set_ylabel("GPT failures (fragments)")
    axes[0].legend(frameon=False)
    fig.suptitle("Few-shot prompting shifts GPT's errors toward leniency")
    paths.append(output_dir / "fig2_prompt_condition_direction.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=FIGURE_DPI); plt.close(fig)

    # 3. Primary domain of confirmed failures
    pdm = results["codes"]["primary_domain"]
    names = list(pdm.keys())[::-1]
    vals = [pdm[n]["n"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color=[c_gen if v == max(vals) else "#e6b0aa" for v in vals])
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v} ({pdm[names[i]]['pct']:.0f}%)", va="center")
    ax.set_xlabel(f"Confirmed failures (n = {results['codes']['n_genuine_fragments']} fragments)")
    ax.set_title("Primary failure domain, confirmed GPT failures")
    paths.append(output_dir / "fig3_primary_domain.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=FIGURE_DPI); plt.close(fig)

    # 4. Pass rates under locked vs pre-revision anchor ratings
    t = results["revisions"]["_table"]
    models = [m for m in MODEL_ORDER if m in t.index.get_level_values(0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, cond in zip(axes, PROMPT_ORDER):
        x = np.arange(len(models))
        ax.bar(x - 0.2, [t.loc[(m, cond), "locked"] for m in models], 0.4, color="#7f8c8d", label="Locked (revised)")
        ax.bar(x + 0.2, [t.loc[(m, cond), "pre_revision"] for m in models], 0.4, color="#2980b9", label="Original (pre-revision)")
        ax.set_xticks(x, [m.split("_")[0] for m in models], rotation=30)
        ax.set_title(cond.replace("_", "-"))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Fragment pass rate (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Every model agrees more with the anchor's original ratings")
    paths.append(output_dir / "fig4_pass_rate_pre_revision.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=FIGURE_DPI); plt.close(fig)

    return paths


def _json_default(obj):
    """Serialize numpy scalars for json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def run_coding_synthesis(
    coding_path: Path = FAILURE_CODES_PATH,
    output_path: Path = CODING_SYNTHESIS_RESULTS_PATH,
    figures_dir: Path | None = FIGURES_DIR,
) -> dict:
    """
    Run the full coding synthesis and export JSON (and figures).

    Args:
        coding_path: Completed coding sheet.
        output_path: JSON results path.
        figures_dir: Figure directory, or None to skip figures.

    Returns:
        Results dict.

    Raises:
        ValueError: The coding sheet is internally inconsistent.
    """
    rows = load_coding_sheet(coding_path)
    problems = validate_consistency(rows)
    if problems:
        raise ValueError("Coding sheet inconsistencies:\n" + "\n".join(problems))
    frags = to_fragment_level(rows)

    outcomes = pd.read_csv(FRAGMENT_OUTCOMES_CORRECTED_PATH)
    gpt = outcomes[outcomes["model_family"] == PRIMARY_MODEL]
    n_per_condition = gpt["prompt_condition"].value_counts().to_dict()

    results = {
        "disposition": disposition_summary(rows, frags, len(gpt)),
        "direction": direction_summary(frags),
        "prompt_condition": condition_summary(rows, n_per_condition),
        "failed_conditions": failure_only_condition_summary(frags),
        "codes": code_profile(rows, frags),
        "criterion": criterion_summary(frags),
        "revisions": revision_summary(
            frags,
            pd.read_csv(GOLD_RATIONALE_SUPPLEMENT_PATH),
            pd.read_csv(SCORED_DB_CORRECTED_PATH),
        ),
        "reviewer_concordance": reviewer_concordance(frags, pd.read_csv(CONTESTED_REVIEW_RESOLVED_PATH)),
    }

    if figures_dir is not None:
        results["figures"] = [str(p) for p in make_figures(results, figures_dir)]

    export = {k: v for k, v in results.items()}
    export["revisions"] = {k: v for k, v in results["revisions"].items() if k != "_table"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export, f, indent=2, default=_json_default)

    d, r = results["disposition"], results["direction"]
    print(f"\nFailure-coding synthesis ({d['fragments']['n']} fragments, {d['rows']['n']} rows)")
    print(f"  Confirmed GPT failures: {d['fragments']['genuine']} fragments ({d['fragments']['genuine_pct']}%)")
    print(f"  Failure rate: locked gold {d['failure_rate_locked_gold_pct']}% -> coder-confirmed {d['failure_rate_coder_confirmed_pct']}%")
    print(f"  Strict errors confirmed: {r[STRICT]['genuine']}/{r[STRICT]['n']}; lenient: {r[LENIENT]['genuine']}/{r[LENIENT]['n']}")
    print(f"  Pre-revision pass-rate delta: mean {results['revisions']['mean_delta_pp']}pp")
    print(f"  Reviewer concordance kappa: {results['reviewer_concordance']['cohens_kappa']}")
    print(f"Saved {output_path}")
    return results


if __name__ == "__main__":
    run_coding_synthesis()
