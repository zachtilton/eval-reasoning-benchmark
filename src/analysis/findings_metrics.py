"""
Metrics behind the written RQ findings (docs/findings/*.md).

Every number quoted in a finding is computed here and exported to
results/findings_metrics.json under a stable key, so each claim in
docs/findings/findings_registry.json can point at the exact metric, data
source and procedure that corroborates it. Inputs are the corrected
(accuracy-only) scoring outputs, the locked gold standard, the triangulated
outputs, the RC2 blind-retest file and the completed GPT coding sheet.

Unit of analysis: majority label per fragment x model x prompt condition
(>= 2 of 3 usable runs agree); combinations without a majority are dropped
from label-based metrics and counted.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.scoring.accuracy import load_gold_standard

from .challenge_cases import identify_challenge_cases
from .coding_synthesis import _json_default, load_coding_sheet, to_fragment_level
from .config import (
    BOOTSTRAP_RESAMPLES,
    CONTESTED_REVIEW_RESOLVED_PATH,
    FAILURE_CODES_PATH,
    FINDINGS_METRICS_PATH,
    FINDINGS_SEED,
    FRAGMENT_OUTCOMES_CORRECTED_PATH,
    FRAGMENT_OUTCOMES_PATH,
    FRAGMENT_OUTCOMES_TRIANGULATED_PATH,
    GOLD_RATIONALE_SUPPLEMENT_PATH,
    KAPPA_BOOTSTRAP_RESAMPLES,
    LUNA_TRIAGE_PATH,
    MODEL_ORDER,
    PRIMARY_MODEL,
    PROMPT_ORDER,
    RC2_BLIND_RETEST_PATH,
    RC3_REVIEW_BLINDED_PATH,
    RC3_REVIEW_KEY_PATH,
    SCORED_DB_CORRECTED_PATH,
    SCORED_DB_PATH,
)

LABELS = ("sound", "not_sound")
KEYS = ["model_family", "prompt_condition", "fragment_id"]


def _pct(x: float) -> float:
    """Proportion to percent, 1 dp."""
    return round(100 * float(x), 1)


def _norm_label(s: pd.Series) -> pd.Series:
    """Normalize 'Not sound' / 'not sound' / 'not_sound' to sound|not_sound."""
    return s.astype(str).str.strip().str.lower().str.replace(" ", "_")


def majority_labels(db: pd.DataFrame) -> pd.DataFrame:
    """
    Majority label per fragment x model x condition from usable runs.

    Args:
        db: Scored database (corrected).

    Returns:
        DataFrame with KEYS + `pred` in {sound, not_sound, none}; 'none' when
        no label has >= 2 usable runs. Covers every model x condition x
        fragment present in db.
    """
    ok = db[~db["error_flag"].fillna(False).astype(bool) & db["classification_output"].isin(LABELS)]
    cnt = ok.groupby(KEYS)["classification_output"].value_counts().unstack(fill_value=0)
    for lab in LABELS:
        if lab not in cnt:
            cnt[lab] = 0
    pred = np.where(cnt["sound"] >= 2, "sound", np.where(cnt["not_sound"] >= 2, "not_sound", "none"))
    out = pd.Series(pred, index=cnt.index, name="pred")
    full = db.set_index(KEYS).index.unique()
    return out.reindex(full, fill_value="none").reset_index()


def reference_labels(gold: pd.DataFrame, supplement: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """
    The three reference-label versions used in sensitivity checks.

    Args:
        gold: Locked gold (fragment_id, expert_classification).
        supplement: Pre-lock revision supplement (fragment_id, original_rating).
        resolved: Contested-fragment review (fragment_id, reviewer_classification).

    Returns:
        DataFrame: fragment_id, locked, pre_revision, triangulated.
    """
    ref = gold[["fragment_id", "expert_classification"]].rename(columns={"expert_classification": "locked"})
    pre = dict(zip(supplement["fragment_id"], supplement["original_rating"]))
    tri = dict(zip(resolved["fragment_id"], resolved["reviewer_classification"]))
    ref["pre_revision"] = ref["fragment_id"].map(pre).fillna(ref["locked"])
    ref["triangulated"] = ref["fragment_id"].map(tri).fillna(ref["locked"])
    return ref


def cluster_bootstrap_pass_rate(outcomes: pd.DataFrame, n: int = BOOTSTRAP_RESAMPLES, seed: int = FINDINGS_SEED) -> dict:
    """
    Pooled fragment-level pass rate with a 95% CI resampling fragments.

    Fragments, not the 1,800 combinations, are resampled because the 12
    model x condition outcomes for one fragment are correlated.

    Args:
        outcomes: Fragment outcomes (fragment_id, fragment_outcome).
        n: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Dict with estimate, ci_95 (percent), n_combinations, n_fragments.
    """
    rng = np.random.default_rng(seed)
    by = outcomes.assign(p=(outcomes["fragment_outcome"] == "pass").astype(int)).groupby("fragment_id")["p"].agg(["sum", "count"])
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    boots = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "pass_rate_pct": _pct(s.sum() / c.sum()),
        "ci_95_pct": [_pct(lo), _pct(hi)],
        "n_combinations": int(c.sum()),
        "n_fragments": int(len(s)),
        "method": f"fragment-clustered bootstrap, {n} resamples, seed {seed}",
    }


def model_anchor_agreement(maj: pd.DataFrame, gold: pd.DataFrame, seed: int = FINDINGS_SEED) -> dict:
    """
    Per model x condition agreement with the locked gold, plus summaries.

    Args:
        maj: Output of majority_labels.
        gold: Locked gold.
        seed: RNG seed for kappa bootstrap CIs.

    Returns:
        Dict with `by_combination` rows and pooled summaries.
    """
    rng = np.random.default_rng(seed)
    m = maj.merge(gold[["fragment_id", "expert_classification"]], on="fragment_id")
    rows = []
    for model in MODEL_ORDER:
        for cond in PROMPT_ORDER:
            g = m[(m["model_family"] == model) & (m["prompt_condition"] == cond)]
            v = g[g["pred"] != "none"]
            a, b = v["expert_classification"].to_numpy(), v["pred"].to_numpy()
            ks = [cohen_kappa_score(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(KAPPA_BOOTSTRAP_RESAMPLES))]
            sound, unsound = v[v["expert_classification"] == "sound"], v[v["expert_classification"] == "not_sound"]
            rows.append({
                "model_family": model, "prompt_condition": cond, "n": len(v),
                "no_majority": int((g["pred"] == "none").sum()),
                "agreement_pct": _pct((v["pred"] == v["expert_classification"]).mean()),
                "kappa": round(cohen_kappa_score(a, b), 3),
                "kappa_ci_95": [round(x, 3) for x in np.percentile(ks, [2.5, 97.5])],
                "accuracy_on_gold_sound_pct": _pct((sound["pred"] == "sound").mean()),
                "accuracy_on_gold_not_sound_pct": _pct((unsound["pred"] == "not_sound").mean()),
                "missed_flaws_pct": _pct((unsound["pred"] == "sound").mean()),
                "wrongly_rejected_pct": _pct((sound["pred"] == "not_sound").mean()),
                "calls_sound_pct": _pct((v["pred"] == "sound").mean()),
            })
    t = pd.DataFrame(rows)
    lenient = t[t["missed_flaws_pct"] > t["wrongly_rejected_pct"]]
    return {
        "by_combination": {f"{r['model_family']}|{r['prompt_condition']}": r for r in rows},
        "kappa_mean": round(t["kappa"].mean(), 3),
        "kappa_range": [round(t["kappa"].min(), 3), round(t["kappa"].max(), 3)],
        "agreement_range_pct": [t["agreement_pct"].min(), t["agreement_pct"].max()],
        "accuracy_on_gold_sound_mean_pct": round(t["accuracy_on_gold_sound_pct"].mean(), 1),
        "accuracy_on_gold_not_sound_mean_pct": round(t["accuracy_on_gold_not_sound_pct"].mean(), 1),
        "gold_sound_pct": _pct((gold["expert_classification"] == "sound").mean()),
        "n_combinations_lenient": int(len(lenient)),
        "strict_combinations": [f"{r.model_family}|{r.prompt_condition}" for r in t[t["missed_flaws_pct"] <= t["wrongly_rejected_pct"]].itertuples()],
        "few_shot_more_lenient_models": [
            mdl for mdl in MODEL_ORDER
            if t.query("model_family == @mdl and prompt_condition == 'few_shot'")["calls_sound_pct"].iloc[0]
            > t.query("model_family == @mdl and prompt_condition == 'zero_shot'")["calls_sound_pct"].iloc[0]
        ],
    }


def model_model_kappa(maj: pd.DataFrame) -> dict:
    """
    Pairwise model-vs-model kappa within each condition.

    Args:
        maj: Output of majority_labels.

    Returns:
        Dict per condition: mean, range, n_pairs.
    """
    out = {}
    for cond in PROMPT_ORDER:
        ks = []
        for a, b in itertools.combinations(MODEL_ORDER, 2):
            x = maj[(maj["prompt_condition"] == cond) & (maj["model_family"] == a)].set_index("fragment_id")["pred"]
            y = maj[(maj["prompt_condition"] == cond) & (maj["model_family"] == b)].set_index("fragment_id")["pred"]
            keep = (x != "none") & (y.reindex(x.index) != "none")
            ks.append(cohen_kappa_score(x[keep], y.reindex(x.index)[keep]))
        out[cond] = {"mean": round(float(np.mean(ks)), 3), "range": [round(min(ks), 3), round(max(ks), 3)], "n_pairs": len(ks)}
    return out


def error_asymmetry_by_reference(maj: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """
    Pooled missed-flaw vs wrongly-rejected rates under each reference version.

    Rates are conditional on the reference label (within-group rates);
    reported pooled and per prompt condition.

    Args:
        maj: Output of majority_labels.
        ref: Output of reference_labels.

    Returns:
        Dict per reference version.
    """
    m = maj[maj["pred"] != "none"].merge(ref, on="fragment_id")

    def rates(d: pd.DataFrame, col: str) -> dict:
        missed = (d.loc[d[col] == "not_sound", "pred"] == "sound").mean()
        rejected = (d.loc[d[col] == "sound", "pred"] == "not_sound").mean()
        return {"missed_flaws_pct": _pct(missed), "wrongly_rejected_pct": _pct(rejected), "ratio": round(missed / rejected, 2)}

    out = {}
    for col in ("locked", "pre_revision", "triangulated"):
        out[col] = {"n_sound_labels": int((ref[col] == "sound").sum()), **rates(m, col),
                    "by_condition": {c: rates(m[m["prompt_condition"] == c], col) for c in PROMPT_ORDER}}
    return out


def human_comparison(maj: pd.DataFrame, gold: pd.DataFrame, retest: pd.DataFrame) -> dict:
    """
    Human raters vs models against the anchor on the shared RC2 fragments.

    Args:
        maj: Output of majority_labels.
        gold: Locked gold.
        retest: RC2 blind retest (report_id, rater_1, rater_2, anchor1, anchor3_pass2).

    Returns:
        Dict of percent agreement / kappa for human pairs and model summaries.
    """
    h = retest.rename(columns={"report_id": "fragment_id"}).copy()
    for c in ("rater_1", "rater_2", "anchor1", "anchor3_pass2"):
        h[c] = _norm_label(h[c])

    def pair(a: pd.Series, b: pd.Series) -> dict:
        return {"agreement_pct": _pct((a == b).mean()), "kappa": round(cohen_kappa_score(a, b), 3), "n": int(len(a))}

    out = {"all_40": {
        "rater_1_vs_rater_2": pair(h["rater_1"], h["rater_2"]),
        "rater_1_vs_anchor_pass2": pair(h["rater_1"], h["anchor3_pass2"]),
        "rater_2_vs_anchor_pass2": pair(h["rater_2"], h["anchor3_pass2"]),
        "rater_1_vs_anchor_original": pair(h["rater_1"], h["anchor1"]),
        "rater_2_vs_anchor_original": pair(h["rater_2"], h["anchor1"]),
    }}
    shared = h.merge(gold[["fragment_id", "expert_classification"]].rename(columns={"expert_classification": "locked"}), on="fragment_id")
    out["not_in_benchmark"] = sorted(set(h["fragment_id"]) - set(shared["fragment_id"]))
    out["shared"] = {
        "n": int(len(shared)),
        "locked_matches_anchor_pass2_pct": _pct((shared["locked"] == shared["anchor3_pass2"]).mean()),
        "rater_1_vs_anchor_pass2": pair(shared["rater_1"], shared["anchor3_pass2"]),
        "rater_2_vs_anchor_pass2": pair(shared["rater_2"], shared["anchor3_pass2"]),
        "rater_1_vs_locked": pair(shared["rater_1"], shared["locked"]),
        "rater_2_vs_locked": pair(shared["rater_2"], shared["locked"]),
        "rater_1_vs_rater_2": pair(shared["rater_1"], shared["rater_2"]),
    }
    x = shared.merge(maj[maj["pred"] != "none"], on="fragment_id")
    per = x.groupby(["model_family", "prompt_condition"]).apply(
        lambda d: pd.Series({"vs_pass2": (d["pred"] == d["anchor3_pass2"]).mean(), "vs_locked": (d["pred"] == d["locked"]).mean()}),
        include_groups=False,
    )
    out["shared"]["models_vs_anchor_pass2_pct"] = {"mean": _pct(per["vs_pass2"].mean()), "range": [_pct(per["vs_pass2"].min()), _pct(per["vs_pass2"].max())]}
    out["shared"]["models_vs_locked_pct"] = {"mean": _pct(per["vs_locked"].mean()), "range": [_pct(per["vs_locked"].min()), _pct(per["vs_locked"].max())]}
    return out


def leniency_vs_human_raters(maj: pd.DataFrame, gold: pd.DataFrame, retest: pd.DataFrame) -> dict:
    """
    Is model leniency relative to the anchor, or to human judges generally?

    On the RC2 fragments in the benchmark, compares how often each human
    judge and the models call fragments sound, and recomputes the models'
    missed-flaw vs wrongly-rejected rates against each RC2 rater in place of
    the anchor.

    Args:
        maj: Output of majority_labels.
        gold: Locked gold.
        retest: RC2 blind retest.

    Returns:
        Dict: sound-call rates per judge, model mean and range, and model
        error rates and ratio against each reference judge.
    """
    h = retest.rename(columns={"report_id": "fragment_id"}).copy()
    for c in ("rater_1", "rater_2", "anchor3_pass2"):
        h[c] = _norm_label(h[c])
    h = h.merge(gold[["fragment_id", "expert_classification"]].rename(columns={"expert_classification": "locked"}), on="fragment_id")
    judges = {"anchor_pass2": "anchor3_pass2", "rater_1": "rater_1", "rater_2": "rater_2", "locked": "locked"}
    m = maj[maj["pred"] != "none"].merge(h, on="fragment_id")
    per = m.groupby(["model_family", "prompt_condition"])["pred"].apply(lambda s: (s == "sound").mean())

    def rates(d: pd.DataFrame, col: str) -> dict:
        missed = (d.loc[d[col] == "not_sound", "pred"] == "sound").mean()
        rejected = (d.loc[d[col] == "sound", "pred"] == "not_sound").mean()
        return {"missed_flaws_pct": _pct(missed), "wrongly_rejected_pct": _pct(rejected), "ratio": round(missed / rejected, 2)}

    against = {name: rates(m, col) for name, col in judges.items()}
    by_condition = {
        c: {
            "models_calls_sound_pct_mean": _pct(per.xs(c, level="prompt_condition").mean()),
            "model_errors_against": {name: rates(m[m["prompt_condition"] == c], col) for name, col in judges.items()},
        }
        for c in PROMPT_ORDER
    }
    return {
        "n_fragments": int(len(h)),
        "calls_sound_pct": {name: _pct((h[col] == "sound").mean()) for name, col in judges.items()},
        "models_calls_sound_pct": {"mean": _pct(per.mean()), "range": [_pct(per.min()), _pct(per.max())]},
        "model_errors_against": against,
        "by_condition": by_condition,
    }


def few_shot_shift(maj: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """
    How the two calibration examples change each model's verdicts.

    Paired by fragment within model: label flips between conditions, an exact
    McNemar (binomial) test on flip direction, whether each flip fixed or
    created an error against the gold, change in share called sound,
    accuracy within each gold label by condition, and between-model spread.
    The label-shift statistics do not use the gold standard.

    Args:
        maj: Output of majority_labels.
        gold: Locked gold.

    Returns:
        Dict per model plus pooled and spread summaries.
    """
    from scipy.stats import binomtest

    w = maj.pivot_table(index=["model_family", "fragment_id"], columns="prompt_condition", values="pred", aggfunc="first").reset_index()
    w = w[(w["zero_shot"] != "none") & (w["few_shot"] != "none")].merge(gold[["fragment_id", "expert_classification"]], on="fragment_id")
    per = {}
    for model in MODEL_ORDER:
        d = w[w["model_family"] == model]
        to_sound = int(((d["zero_shot"] == "not_sound") & (d["few_shot"] == "sound")).sum())
        to_not = int(((d["zero_shot"] == "sound") & (d["few_shot"] == "not_sound")).sum())
        n_flip = to_sound + to_not
        flipped = d[d["zero_shot"] != d["few_shot"]]
        fixed = int((flipped["few_shot"] == flipped["expert_classification"]).sum())
        acc = {}
        for cond in PROMPT_ORDER:
            for lab in LABELS:
                sub = d[d["expert_classification"] == lab]
                acc[f"{cond}_accuracy_on_gold_{lab}_pct"] = _pct((sub[cond] == lab).mean())
        per[model] = {
            "n_paired": int(len(d)),
            "zero_shot_calls_sound_pct": _pct((d["zero_shot"] == "sound").mean()),
            "few_shot_calls_sound_pct": _pct((d["few_shot"] == "sound").mean()),
            "shift_pp": round(100 * ((d["few_shot"] == "sound").mean() - (d["zero_shot"] == "sound").mean()), 1),
            "flips_to_sound": to_sound,
            "flips_to_not_sound": to_not,
            "flip_rate_pct": _pct(n_flip / len(d)),
            "flips_fixed_error": fixed,
            "flips_created_error": int(len(flipped) - fixed),
            "mcnemar_exact_p": round(binomtest(to_sound, n_flip, 0.5).pvalue, 6) if n_flip else 1.0,
            **acc,
        }
    zs = [v["zero_shot_calls_sound_pct"] for v in per.values()]
    fs = [v["few_shot_calls_sound_pct"] for v in per.values()]
    tot_s = sum(v["flips_to_sound"] for v in per.values())
    tot_n = sum(v["flips_to_not_sound"] for v in per.values())
    return {
        "by_model": per,
        "pooled": {"flips_to_sound": tot_s, "flips_to_not_sound": tot_n, "share_of_flips_to_sound_pct": _pct(tot_s / (tot_s + tot_n)),
                   "flips_fixed_error": sum(v["flips_fixed_error"] for v in per.values()),
                   "flips_created_error": sum(v["flips_created_error"] for v in per.values())},
        "models_more_lenient_with_examples": [m for m, v in per.items() if v["shift_pp"] > 0],
        "models_significant_at_05": [m for m, v in per.items() if v["mcnemar_exact_p"] < 0.05],
        "spread_calls_sound_pct": {"zero_shot": [min(zs), max(zs)], "few_shot": [min(fs), max(fs)],
                                   "zero_shot_range_pp": round(max(zs) - min(zs), 1), "few_shot_range_pp": round(max(fs) - min(fs), 1)},
    }


def criterion_variation(outcomes: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """
    All-model failure rates by evaluation criterion.

    Reports, per criterion, fragment count, share of gold labels not sound
    (leniency makes not-sound fragments fail more), the pooled fail rate,
    the fail rate within each gold label, and a fragment-clustered bootstrap
    CI. A chi-squared test on fragment-level fail counts (each fragment's 12
    outcomes summed) asks whether criteria differ overall.

    Args:
        outcomes: Corrected fragment outcomes.
        gold: Locked gold with criterion.

    Returns:
        Dict per criterion plus the omnibus test.
    """
    from scipy.stats import chi2_contingency

    from .config import CORE_CRITERIA

    g = gold[["fragment_id", "expert_classification", "criterion"]].copy()
    g["crit"] = g["criterion"].where(g["criterion"].isin(CORE_CRITERIA), "other")
    o = outcomes.merge(g, on="fragment_id").assign(f=lambda d: (d["fragment_outcome"] == "fail").astype(int))
    rng = np.random.default_rng(FINDINGS_SEED)
    out = {}
    for crit, d in o.groupby("crit"):
        by = d.groupby("fragment_id")["f"].agg(["sum", "count"])
        s, c = by["sum"].to_numpy(), by["count"].to_numpy()
        boots = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(BOOTSTRAP_RESAMPLES))]
        out[crit] = {
            "n_fragments": int(len(by)),
            "gold_not_sound_pct": _pct((g.loc[g["crit"] == crit, "expert_classification"] == "not_sound").mean()),
            "fail_pct": _pct(d["f"].mean()),
            "fail_ci_95_pct": [_pct(x) for x in np.percentile(boots, [2.5, 97.5])],
            "fail_on_gold_sound_pct": _pct(d.loc[d["expert_classification"] == "sound", "f"].mean()) if (d["expert_classification"] == "sound").any() else None,
            "fail_on_gold_not_sound_pct": _pct(d.loc[d["expert_classification"] == "not_sound", "f"].mean()) if (d["expert_classification"] == "not_sound").any() else None,
        }
    frag = o.groupby(["crit", "fragment_id"])["f"].agg(["sum", "count"]).reset_index()
    table = frag.groupby("crit")[["sum", "count"]].sum()
    table["pass"] = table["count"] - table["sum"]
    chi2, p, dof, _ = chi2_contingency(table[["sum", "pass"]].to_numpy())
    return {"by_criterion": out, "omnibus_chi2_combinations": {"chi2": round(float(chi2), 3), "dof": int(dof), "p_value": round(float(p), 4),
            "note": "Treats the 12 outcomes per fragment as independent; anti-conservative. Read with the clustered CIs."}}


def _nested_f(y: np.ndarray, x_reduced: np.ndarray, x_full: np.ndarray) -> float:
    """F statistic comparing nested OLS models (full adds columns to reduced)."""
    def rss(x):
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return float(((y - x @ beta) ** 2).sum())
    r0, r1 = rss(x_reduced), rss(x_full)
    df1 = x_full.shape[1] - x_reduced.shape[1]
    df2 = len(y) - x_full.shape[1]
    return ((r0 - r1) / df1) / (r1 / df2)


def criterion_checks(outcomes: pd.DataFrame, maj: pd.DataFrame, gold: pd.DataFrame, retest: pd.DataFrame,
                     n_perm: int = BOOTSTRAP_RESAMPLES, seed: int = FINDINGS_SEED) -> dict:
    """
    Is apparent criterion difficulty more than gold-label composition, and is
    the gold's criterion pattern corroborated without the gold?

    (a) Stratified check: fragment-level fail proportion (12 outcomes each;
        fragments are independent units) regressed on gold label, then on
        label + criterion; nested F test, with a permutation p-value that
        shuffles criterion within gold-label strata.
    (b) Gold composition: share of not-sound labels by criterion; chi-squared
        and a permutation p-value (small expected counts), plus impact vs
        relevance Fisher exact.
    (c) Gold-independent: models' own share of not-sound majority labels per
        fragment, by criterion (permutation test across criteria), and its
        correlation with the gold's criterion pattern.
    (d) RC2 raters' not-sound share by criterion on shared fragments.

    Args:
        outcomes: Corrected fragment outcomes.
        maj: Output of majority_labels.
        gold: Locked gold with criterion.
        retest: RC2 blind retest.
        n_perm: Permutations.
        seed: RNG seed.

    Returns:
        Dict with the four checks.
    """
    from scipy.stats import chi2_contingency, f as f_dist, fisher_exact, pearsonr

    from .config import CORE_CRITERIA

    rng = np.random.default_rng(seed)
    g = gold[["fragment_id", "expert_classification", "criterion"]].copy()
    g["crit"] = g["criterion"].where(g["criterion"].isin(CORE_CRITERIA), "other")
    crits = sorted(g["crit"].unique())

    # (a) stratified check
    fr = outcomes.assign(f=(outcomes["fragment_outcome"] == "fail").astype(float)).groupby("fragment_id")["f"].mean().rename("fail").reset_index().merge(g, on="fragment_id")
    y = fr["fail"].to_numpy()
    label = (fr["expert_classification"] == "not_sound").astype(float).to_numpy()
    x_red = np.column_stack([np.ones(len(y)), label])

    def design(c):
        return np.column_stack([x_red] + [(c == k).astype(float) for k in crits[1:]])

    crit_arr = fr["crit"].to_numpy()
    f_obs = _nested_f(y, x_red, design(crit_arr))
    df1, df2 = len(crits) - 1, len(y) - (2 + len(crits) - 1)
    perm = []
    for _ in range(n_perm):
        c = crit_arr.copy()
        for lab in (0.0, 1.0):
            idx = np.where(label == lab)[0]
            c[idx] = rng.permutation(c[idx])
        perm.append(_nested_f(y, x_red, design(c)))
    f_label_only = _nested_f(y, np.ones((len(y), 1)), x_red)
    stratified = {
        "unit": "fragment (mean of 12 model x condition outcomes)",
        "label_effect_F": round(f_label_only, 2),
        "label_effect_p": float(f"{f_dist.sf(f_label_only, 1, len(y) - 2):.2e}"),
        "criterion_given_label_F": round(f_obs, 3),
        "criterion_given_label_df": [df1, df2],
        "criterion_given_label_p": round(float(f_dist.sf(f_obs, df1, df2)), 4),
        "criterion_given_label_permutation_p": round(float((np.array(perm) >= f_obs).mean()), 4),
        "fail_pct_gold_not_sound_vs_sound": [_pct(y[label == 1].mean()), _pct(y[label == 0].mean())],
    }

    # (b) gold composition
    tab = pd.crosstab(g["crit"], g["expert_classification"]).reindex(crits)
    chi2, p_chi, dof, _ = chi2_contingency(tab.to_numpy())
    perm_chi = []
    labs = g["expert_classification"].to_numpy()
    for _ in range(n_perm):
        pt = pd.crosstab(g["crit"].to_numpy(), rng.permutation(labs))
        perm_chi.append(chi2_contingency(pt.to_numpy())[0])
    imp, rel = tab.loc["impact"], tab.loc["relevance"]
    odds, p_fish = fisher_exact([[imp["not_sound"], imp["sound"]], [rel["not_sound"], rel["sound"]]])
    gold_comp = {
        "not_sound_pct": {k: _pct(tab.loc[k, "not_sound"] / tab.loc[k].sum()) for k in crits},
        "n": {k: int(tab.loc[k].sum()) for k in crits},
        "chi2": round(float(chi2), 2), "dof": int(dof), "chi2_p": round(float(p_chi), 4),
        "permutation_p": round(float((np.array(perm_chi) >= chi2).mean()), 4),
        "impact_vs_relevance_fisher_p": round(float(p_fish), 4),
        "impact_vs_relevance_odds_ratio": round(float(odds), 2),
    }

    # (c) models' own not-sound calls (gold-independent)
    mm = maj[maj["pred"] != "none"].assign(ns=lambda d: (d["pred"] == "not_sound").astype(float))
    fm = mm.groupby("fragment_id")["ns"].mean().rename("model_ns").reset_index().merge(g, on="fragment_id")
    by_crit = fm.groupby("crit")["model_ns"].mean()
    def between_var(vals, groups):
        return float(pd.Series(vals).groupby(groups).mean().var())
    obs_bv = between_var(fm["model_ns"].to_numpy(), fm["crit"].to_numpy())
    perm_bv = [between_var(fm["model_ns"].to_numpy(), rng.permutation(fm["crit"].to_numpy())) for _ in range(n_perm)]
    r, p_r = pearsonr([gold_comp["not_sound_pct"][k] for k in crits], [100 * by_crit[k] for k in crits])
    within = fm.groupby(["crit", "expert_classification"])["model_ns"].mean().unstack()
    models_view = {
        "model_not_sound_pct": {k: _pct(by_crit[k]) for k in crits},
        "model_not_sound_pct_within_gold_label": {k: {lab: _pct(within.loc[k, lab]) for lab in within.columns} for k in crits},
        "criterion_permutation_p": round(float((np.array(perm_bv) >= obs_bv).mean()), 4),
        "correlation_with_gold_pattern_r": round(float(r), 3),
        "correlation_p": round(float(p_r), 4),
        "n_criteria": len(crits),
    }

    # (d) RC2 raters
    h = retest.rename(columns={"report_id": "fragment_id"}).copy()
    for c in ("rater_1", "rater_2"):
        h[c] = _norm_label(h[c])
    h = h.merge(g, on="fragment_id")
    raters = {
        k: {"n": int(len(d)), "gold_not_sound": int((d["expert_classification"] == "not_sound").sum()),
            "rater_1_not_sound": int((d["rater_1"] == "not_sound").sum()), "rater_2_not_sound": int((d["rater_2"] == "not_sound").sum())}
        for k, d in h.groupby("crit")
    }
    return {"stratified": stratified, "gold_composition": gold_comp, "models_view": models_view, "rc2_raters": raters}


def criterion_by_condition(outcomes: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """
    Within-criterion leniency gap and stratified criterion test, per condition.

    For each prompt condition: fail rate on gold-not-sound vs gold-sound
    fragments within each criterion, the count of criteria where the gap is
    positive (lenient), and the nested F test of criterion given gold label
    on fragment-level fail proportions (6 outcomes per fragment).

    Args:
        outcomes: Corrected fragment outcomes.
        gold: Locked gold with criterion.

    Returns:
        Dict per condition.
    """
    from scipy.stats import f as f_dist

    from .config import CORE_CRITERIA

    g = gold[["fragment_id", "expert_classification", "criterion"]].copy()
    g["crit"] = g["criterion"].where(g["criterion"].isin(CORE_CRITERIA), "other")
    crits = sorted(g["crit"].unique())
    out = {}
    for cond in PROMPT_ORDER:
        o = outcomes[outcomes["prompt_condition"] == cond].assign(f=lambda d: (d["fragment_outcome"] == "fail").astype(float))
        fr = o.groupby("fragment_id")["f"].mean().rename("fail").reset_index().merge(g, on="fragment_id")
        gaps = {}
        for k in crits:
            d = fr[fr["crit"] == k]
            ns = d.loc[d["expert_classification"] == "not_sound", "fail"]
            s = d.loc[d["expert_classification"] == "sound", "fail"]
            gaps[k] = {"fail_on_not_sound_pct": _pct(ns.mean()), "fail_on_sound_pct": _pct(s.mean()),
                       "gap_pp": round(100 * (ns.mean() - s.mean()), 1), "n_not_sound": int(len(ns)), "n_sound": int(len(s))}
        y = fr["fail"].to_numpy()
        label = (fr["expert_classification"] == "not_sound").astype(float).to_numpy()
        x_red = np.column_stack([np.ones(len(y)), label])
        x_full = np.column_stack([x_red] + [(fr["crit"].to_numpy() == k).astype(float) for k in crits[1:]])
        f_obs = _nested_f(y, x_red, x_full)
        df1, df2 = len(crits) - 1, len(y) - x_full.shape[1]
        out[cond] = {
            "by_criterion": gaps,
            "criteria_with_lenient_gap": int(sum(v["gap_pp"] > 0 for v in gaps.values())),
            "gap_range_pp": [min(v["gap_pp"] for v in gaps.values()), max(v["gap_pp"] for v in gaps.values())],
            "criterion_given_label_F": round(f_obs, 3),
            "criterion_given_label_p": round(float(f_dist.sf(f_obs, df1, df2)), 4),
        }
    return out


def leniency_ratio_uncertainty(maj: pd.DataFrame, gold: pd.DataFrame, retest: pd.DataFrame,
                               n: int = BOOTSTRAP_RESAMPLES, seed: int = FINDINGS_SEED) -> dict:
    """
    How certain is the leniency ratio, and is the RC2-subset reversal unusual?

    (a) Fragment-clustered bootstrap 95% CI for the missed-flaw / wrongly-
        rejected ratio against the locked gold, per condition and pooled.
    (b) For zero-shot, the distribution of the ratio across random subsets of
        the same size as the RC2 overlap (38), drawn (i) freely and (ii) with
        the same sound / not-sound mix, giving the share of subsets with a
        ratio below 1 and the percentile of the observed RC2-subset ratio.

    Args:
        maj: Output of majority_labels.
        gold: Locked gold.
        retest: RC2 blind retest (defines the observed subset).
        n: Resamples.
        seed: RNG seed.

    Returns:
        Dict with CIs and the subset simulation.
    """
    rng = np.random.default_rng(seed)
    m = maj[maj["pred"] != "none"].merge(gold[["fragment_id", "expert_classification"]], on="fragment_id")
    m = m.assign(missed=(m["expert_classification"] == "not_sound") & (m["pred"] == "sound"),
                 rejected=(m["expert_classification"] == "sound") & (m["pred"] == "not_sound"),
                 is_ns=m["expert_classification"] == "not_sound")

    def per_fragment(d: pd.DataFrame) -> pd.DataFrame:
        return d.groupby("fragment_id").agg(missed=("missed", "sum"), rejected=("rejected", "sum"), n=("pred", "size"),
                                            is_ns=("is_ns", "first"))

    def ratio(fr: pd.DataFrame) -> float:
        ns, s = fr[fr["is_ns"]], fr[~fr["is_ns"]]
        missed = ns["missed"].sum() / ns["n"].sum()
        rejected = s["rejected"].sum() / s["n"].sum()
        return float(missed / rejected) if rejected > 0 else float("inf")

    cis = {}
    for name, d in [("zero_shot", m[m["prompt_condition"] == "zero_shot"]), ("few_shot", m[m["prompt_condition"] == "few_shot"]), ("pooled", m)]:
        fr = per_fragment(d)
        arr = fr.reset_index(drop=True)
        boots = [ratio(arr.iloc[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
        cis[name] = {"ratio": round(ratio(fr), 2), "ci_95": [round(x, 2) for x in np.percentile(boots, [2.5, 97.5])],
                     "share_of_resamples_below_1_pct": _pct(np.mean(np.array(boots) < 1))}

    zs = per_fragment(m[m["prompt_condition"] == "zero_shot"])
    rc2_ids = set(retest["report_id"]) & set(zs.index)
    k = len(rc2_ids)
    observed = ratio(zs.loc[sorted(rc2_ids)])
    ns_ids, s_ids = zs.index[zs["is_ns"]].to_numpy(), zs.index[~zs["is_ns"]].to_numpy()
    k_ns = int(zs.loc[sorted(rc2_ids), "is_ns"].sum())
    free, strat = [], []
    for _ in range(n):
        free.append(ratio(zs.loc[rng.choice(zs.index.to_numpy(), k, replace=False)]))
        strat.append(ratio(zs.loc[np.concatenate([rng.choice(ns_ids, k_ns, replace=False), rng.choice(s_ids, k - k_ns, replace=False)])]))
    free, strat = np.array(free), np.array(strat)
    subset = {
        "subset_size": k, "subset_not_sound": k_ns,
        "observed_rc2_subset_ratio": round(observed, 2),
        "random_subsets_free": {"share_below_1_pct": _pct((free < 1).mean()), "share_at_or_below_observed_pct": _pct((free <= observed).mean()),
                                "middle_95": [round(x, 2) for x in np.percentile(free, [2.5, 97.5])]},
        "random_subsets_same_label_mix": {"share_below_1_pct": _pct((strat < 1).mean()), "share_at_or_below_observed_pct": _pct((strat <= observed).mean()),
                                          "middle_95": [round(x, 2) for x in np.percentile(strat, [2.5, 97.5])]},
    }
    return {"bootstrap_ci_locked_gold": cis, "zero_shot_subset_simulation": subset}


def unanimity(outcomes: pd.DataFrame) -> dict:
    """
    Run-to-run consistency: share of combinations with 3/3 agreement.

    Args:
        outcomes: Fragment outcomes with pass_count and unanimous_agreement.

    Returns:
        Dict with overall rate and pass-count distribution.
    """
    return {
        "unanimous_pct": _pct(outcomes["unanimous_agreement"].astype(bool).mean()),
        "pass_count_distribution": outcomes["pass_count"].value_counts().sort_index().to_dict(),
    }


def shared_failures(outcomes: pd.DataFrame, gold: pd.DataFrame, resolved: pd.DataFrame, frags: pd.DataFrame) -> dict:
    """
    Do failures cluster on the same fragments beyond chance?

    A model "fails" a fragment if it fails in either condition. The expected
    distribution assumes independent models, each failing at its own observed
    rate (exact Poisson-binomial). Also reports the share of pass/fail
    variance explained (eta squared) by fragment, model and condition.

    Args:
        outcomes: Corrected fragment outcomes (1,800 rows).
        gold: Locked gold.
        resolved: Contested-fragment review (35 rows).
        frags: Fragment-level GPT coding (to_fragment_level output).

    Returns:
        Dict of observed vs expected counts, eta squared, and composition of
        the fragments failed by all six models.
    """
    o = outcomes.assign(f=(outcomes["fragment_outcome"] == "fail").astype(int))
    pm = o.groupby(["fragment_id", "model_family"])["f"].max().unstack()
    p = pm.mean()
    dist = np.array([1.0])
    for pi in p.to_numpy():
        dist = np.convolve(dist, [1 - pi, pi])
    n_frag = len(pm)
    observed = pm.sum(axis=1).value_counts().reindex(range(7), fill_value=0)

    total = ((o["f"] - o["f"].mean()) ** 2).sum()
    eta = {}
    for k in ("fragment_id", "model_family", "prompt_condition"):
        means = o.groupby(k)["f"].transform("mean")
        eta[k] = round(float(((means - o["f"].mean()) ** 2).sum() / total), 3)
    n_groups = o["fragment_id"].nunique()

    all6 = set(pm.index[pm.sum(axis=1) == 6])
    labels = gold.set_index("fragment_id")["expert_classification"]
    genuine = frags.set_index("fragment_id")["genuine"]
    return {
        "per_model_any_condition_fail_rate_pct": {m: _pct(p[m]) for m in MODEL_ORDER},
        "observed": {str(k): int(v) for k, v in observed.items()},
        "expected_if_independent": {str(k): round(float(dist[k] * n_frag), 1) for k in range(7)},
        "observed_grouped": {"0": int(observed[0]), "1-5": int(observed[1:6].sum()), "6": int(observed[6])},
        "expected_grouped": {"0": round(dist[0] * n_frag, 1), "1-5": round(dist[1:6].sum() * n_frag, 1), "6": round(dist[6] * n_frag, 1)},
        "eta_squared": eta,
        "eta_squared_chance_baseline_fragment": round((n_groups - 1) / (len(o) - 1), 3),
        "all_six": {
            "n": len(all6),
            "in_contested_set": len(all6 & set(resolved["fragment_id"])),
            "gold_not_sound": int((labels.reindex(sorted(all6)) == "not_sound").sum()),
            "gpt_coding_genuine": int(genuine.reindex(sorted(all6)).fillna(False).astype(bool).sum()),
            "gpt_coding_doubt_gold": int((~genuine.reindex(sorted(all6)).fillna(True).astype(bool)).sum()),
        },
    }


def challenge_case_summary(outcomes: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """
    H.4 challenge cases on corrected outcomes and confirmed coded rows.

    Args:
        outcomes: Corrected fragment outcomes.
        rows: Row-level coding sheet (load_coding_sheet output).

    Returns:
        Dict with Criterion 1/2 counts and the nesting of Criterion 2 in 1.
    """
    gen = rows[rows["genuine"]]
    r = identify_challenge_cases(outcomes, gen)
    c1 = r["criterion_1"]["cases"]
    c1_keys = set(zip(c1["fragment_id"], c1["prompt_condition"]))
    codes = r["criterion_2"]["codes"]
    c2_rows = gen[gen["primary_code"].isin(set(codes["failure_code"]))]
    c2_keys = set(zip(c2_rows["fragment_id"], c2_rows["prompt_condition"]))
    gpt_fails = int(((outcomes["model_family"] == PRIMARY_MODEL) & (outcomes["fragment_outcome"] == "fail")).sum())
    return {
        "gpt_failures": gpt_fails,
        "criterion_1_cases": int(len(c1)),
        "criterion_1_pct_of_gpt_failures": _pct(len(c1) / gpt_fails),
        "criterion_1_all_five_comparison_fail": int((c1["n_comparison_failures"] == 5).sum()),
        "criterion_1_fragments": int(c1["fragment_id"].nunique()),
        "criterion_2_codes": {row.failure_code: row.percentage for row in codes.itertuples()},
        "criterion_2_coded_rows": int(len(gen)),
        "criterion_2_cases": len(c2_keys),
        "criterion_2_inside_criterion_1": len(c2_keys & c1_keys),
        "module_overlap_label": r["overlap"]["interpretation"],
    }


def rc3_deviation(
    outcomes_as_designed: pd.DataFrame,
    scored_db_as_designed: pd.DataFrame,
    review_key: pd.DataFrame,
    review_blinded: pd.DataFrame,
    luna: pd.DataFrame,
) -> dict:
    """
    Figures behind the pre-registration deviation (coherence gate removed).

    Args:
        outcomes_as_designed: Fragment outcomes under the pre-registered scoring.
        scored_db_as_designed: Scored DB with coherence_final.
        review_key: RC3 stratified sample key (row_id, stratum, coherence_final).
        review_blinded: RC3 blinded sheet with the author's ratings (your_rating).
        luna: Secondary-model triage of all ambiguous runs (luna_verdict).

    Returns:
        Dict: as-designed pass rate, coherence population, RC3 sample by
        stratum and rating, Luna triage counts, and a rule-of-three bound
        on genuinely incoherent runs among those never reviewed.
    """
    pop = scored_db_as_designed["coherence_final"].value_counts().to_dict()
    rev = review_key.merge(review_blinded[["row_id", "your_rating"]], on="row_id", how="left")
    rated = rev[rev["your_rating"].notna()]
    flagged = rated[rated["coherence_final"] != "coherent"]
    by_stratum = {
        s: {"sampled": int(len(g)), "rated": int(g["your_rating"].notna().sum()),
            "rated_incoherent": int((g["your_rating"] == "incoherent").sum())}
        for s, g in rev.groupby("stratum")
    }
    n_incoh = int((rated["coherence_final"] == "incoherent").sum())
    n_amb = int((rated["coherence_final"] == "ambiguous").sum())
    return {
        "as_designed_pass_rate_pct": _pct((outcomes_as_designed["fragment_outcome"] == "pass").mean()),
        "coherence_population": {k: int(v) for k, v in pop.items()},
        "non_coherent_total": int(pop.get("incoherent", 0) + pop.get("ambiguous", 0)),
        "review": {
            "sampled": int(len(rev)),
            "rated": int(len(rated)),
            "rated_coherent": int((rated["your_rating"] == "coherent").sum()),
            "rated_incoherent": int((rated["your_rating"] == "incoherent").sum()),
            "flagged_rated": int(len(flagged)),
            "flagged_rated_incoherent": int((flagged["your_rating"] == "incoherent").sum()),
            "incoherent_rated": n_incoh,
            "ambiguous_rated": n_amb,
            "by_stratum": by_stratum,
        },
        "luna_triage": luna["luna_verdict"].value_counts().to_dict(),
        "unreviewed_bound": _unreviewed_bound(rev, pop, luna),
    }


def _unreviewed_bound(rev: pd.DataFrame, pop: dict, luna: pd.DataFrame) -> dict:
    """
    Rule-of-three bound on genuinely incoherent runs left unreviewed.

    Populations follow the RC3 strata: confident-incoherent runs, ambiguous
    runs Luna judged coherent (sampled), and ambiguous runs Luna could not
    call (fully censused). With 0 incoherent ratings in n reviewed, the 95%
    upper bound on the rate is ~3/n, applied to the unreviewed remainder.

    Args:
        rev: RC3 key merged with ratings.
        pop: Coherence population counts.
        luna: Luna triage of all ambiguous runs.

    Returns:
        Dict per population plus the total bound on unreviewed incoherent runs.
    """
    luna_counts = luna["luna_verdict"].value_counts()
    rated = rev[rev["your_rating"].notna()]
    groups = {
        "incoherent": (int(pop.get("incoherent", 0)), int((rated["stratum"] == "incoherent_confident").sum())),
        "ambiguous_luna_coherent": (int(luna_counts.get("coherent", 0)),
                                    int(rated["stratum"].isin(["ambiguous_sides_with_haiku", "ambiguous_sides_with_rule"]).sum())),
        "ambiguous_luna_ambiguous": (int(luna_counts.get("ambiguous", 0)), int((rated["stratum"] == "ambiguous_3way_split").sum())),
    }
    out, total = {}, 0.0
    for name, (n_pop, n_rev) in groups.items():
        unreviewed = n_pop - n_rev
        rate = 3 / n_rev if n_rev else None
        bound = unreviewed * rate if rate is not None else None
        total += bound or 0.0
        out[name] = {"population": n_pop, "reviewed": n_rev, "unreviewed": unreviewed,
                     "upper_rate_pct": round(100 * rate, 1) if rate else None,
                     "max_incoherent_unreviewed": round(bound, 1) if bound is not None else None}
    out["max_incoherent_unreviewed_total"] = round(total, 1)
    out["max_incoherent_unreviewed_pct_of_all_runs"] = round(100 * total / sum(pop.values()), 2)
    return out


def contested_overlap(resolved: pd.DataFrame, supplement: pd.DataFrame, retest: pd.DataFrame) -> dict:
    """
    Overlap of the contested set with the pre-lock revisions and RC2 ratings.

    Args:
        resolved: Contested-fragment review (35 rows).
        supplement: Pre-lock revision supplement (18 rows).
        retest: RC2 blind retest (report_id, rater_1, rater_2, anchor1, ...).

    Returns:
        Dict: overlap counts, how the reviewer sided on revised fragments, and
        the human vote split on fragments also rated by the RC2 raters.
    """
    x = resolved.merge(supplement[["fragment_id", "original_rating", "final_rating_locked"]], on="fragment_id")
    h = retest.rename(columns={"report_id": "fragment_id"}).copy()
    for c in ("rater_1", "rater_2", "anchor1"):
        h[c] = _norm_label(h[c])
    direct = x.merge(h, on="fragment_id")
    splits = {}
    for r in direct.itertuples():
        votes = [r.rater_1, r.rater_2, r.original_rating, r.reviewer_classification]
        splits[str(r.fragment_id)] = {
            "rc2_raters": [r.rater_1, r.rater_2],
            "anchor_original": r.original_rating,
            "third_reviewer": r.reviewer_classification,
            "locked": r.final_rating_locked,
            "model_majority": r.model_majority,
            "human_sound_votes": int(sum(v == "sound" for v in votes)),
            "human_votes": len(votes),
        }
    return {
        "revised_in_contested": int(len(x)),
        "revised_in_contested_reviewer_sides_models": int((x["reviewer_classification"] == x["model_majority"]).sum()),
        "revised_in_contested_original_matches_models": int((x["original_rating"] == x["model_majority"]).sum()),
        "directly_rated_by_rc2": int(len(direct)),
        "directly_rated_all_2_2": bool(all(s["human_sound_votes"] == 2 for s in splits.values())),
        "directly_rated": splits,
    }


def evaluator_split_share(human: dict, contested_n: int, n_fragments: int) -> dict:
    """
    The two estimates behind "roughly one in four fragments split evaluators".

    Args:
        human: Output of human_comparison.
        contested_n: Size of the contested set.
        n_fragments: Benchmark size.

    Returns:
        Dict with human-human disagreement and contested share (percent).
    """
    return {
        "human_human_disagreement_pct": round(100 - human["all_40"]["rater_1_vs_rater_2"]["agreement_pct"], 1),
        "contested_share_pct": _pct(contested_n / n_fragments),
    }


def run_findings_metrics(output_path: Path = FINDINGS_METRICS_PATH) -> dict:
    """
    Compute every metric cited in docs/findings and export JSON.

    Args:
        output_path: Where to write findings_metrics.json.

    Returns:
        Metrics dict.
    """
    gold = load_gold_standard()
    db = pd.read_csv(SCORED_DB_CORRECTED_PATH)
    outcomes = pd.read_csv(FRAGMENT_OUTCOMES_CORRECTED_PATH)
    tri = pd.read_csv(FRAGMENT_OUTCOMES_TRIANGULATED_PATH)
    resolved = pd.read_csv(CONTESTED_REVIEW_RESOLVED_PATH)
    supplement = pd.read_csv(GOLD_RATIONALE_SUPPLEMENT_PATH)
    rows = load_coding_sheet(FAILURE_CODES_PATH)
    frags = to_fragment_level(rows)
    retest = pd.read_csv(RC2_BLIND_RETEST_PATH)
    maj = majority_labels(db)
    ref = reference_labels(gold, supplement, resolved)

    usable = db[~db["error_flag"].fillna(False).astype(bool)]
    correct = (db["classification_accuracy_corrected"] == "correct").sum()
    metrics = {
        "pass_rate": {
            "corrected": cluster_bootstrap_pass_rate(outcomes),
            "triangulated": cluster_bootstrap_pass_rate(tri),
            "run_level_corrected": {"correct": int(correct), "n_runs": int(len(db)), "pct": _pct(correct / len(db)), "n_unusable": int(len(db) - len(usable))},
        },
        "model_anchor_agreement": model_anchor_agreement(maj, gold),
        "model_model_kappa": model_model_kappa(maj),
        "error_asymmetry": error_asymmetry_by_reference(maj, ref),
        "human_comparison": human_comparison(maj, gold, retest),
        "leniency_vs_human_raters": leniency_vs_human_raters(maj, gold, retest),
        "unanimity_corrected": unanimity(outcomes),
        "unanimity_by_condition_pct": {c: _pct(outcomes.loc[outcomes["prompt_condition"] == c, "unanimous_agreement"].astype(bool).mean()) for c in PROMPT_ORDER},
        "few_shot_shift": few_shot_shift(maj, gold),
        "criterion_variation": criterion_variation(outcomes, gold),
        "criterion_checks": criterion_checks(outcomes, maj, gold, retest),
        "criterion_by_condition": criterion_by_condition(outcomes, gold),
        "leniency_ratio_uncertainty": leniency_ratio_uncertainty(maj, gold, retest),
        "shared_failures": shared_failures(outcomes, gold, resolved, frags),
        "challenge_cases": challenge_case_summary(outcomes, rows),
        "contested_review": {
            "n": int(len(resolved)),
            "majority_strength": resolved["majority_strength"].value_counts().sort_index().to_dict(),
            "reviewer_sides_models": int((resolved["reviewer_classification"] == resolved["model_majority"]).sum()),
            "lenient_side": {"n": int((resolved["expert_classification"] == "not_sound").sum()),
                             "reviewer_sides_models": int(((resolved["expert_classification"] == "not_sound") & (resolved["reviewer_classification"] == "sound")).sum())},
            "strict_side": {"n": int((resolved["expert_classification"] == "sound").sum()),
                            "reviewer_sides_models": int(((resolved["expert_classification"] == "sound") & (resolved["reviewer_classification"] == "not_sound")).sum())},
        },
    }
    metrics["rc3_deviation"] = rc3_deviation(
        pd.read_csv(FRAGMENT_OUTCOMES_PATH),
        pd.read_csv(SCORED_DB_PATH, usecols=["coherence_final"]),
        pd.read_csv(RC3_REVIEW_KEY_PATH),
        pd.read_csv(RC3_REVIEW_BLINDED_PATH),
        pd.read_csv(LUNA_TRIAGE_PATH),
    )
    metrics["contested_overlap"] = contested_overlap(resolved, supplement, retest)
    metrics["evaluator_split"] = evaluator_split_share(metrics["human_comparison"], len(resolved), len(gold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"Saved {output_path}")
    return metrics


if __name__ == "__main__":
    run_findings_metrics()
