"""
Poster figures built from existing results files (no new analysis).

Figure: calibration-example slope chart. Each model's share of passages
called "sound" without vs with the two calibration examples
(findings_metrics.json -> few_shot_shift), against the expert's share.
Agreement with the expert (H2_statistical_tests_corrected.json ->
test_1_mcnemar) is loaded for the caption, not plotted (author, 2026-11-15).

Run: python3 -m src.analysis.poster_figures
"""

import json
from pathlib import Path

from src.analysis.config import (
    ARCHITECTURE_MAP,
    FAMILY_COLORS,
    POSTER_ACCENT,
    POSTER_ACCENT_DARK,
    POSTER_ACCENT_PALE,
    POSTER_LABEL_COLOR,
    FIGURE_DPI,
    FINDINGS_METRICS_PATH,
    H2_CORRECTED_RESULTS_PATH,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    POSTER_FIGURES_DIR,
)


def load_slope_data(
    metrics_path: Path = FINDINGS_METRICS_PATH,
    h2_path: Path = H2_CORRECTED_RESULTS_PATH,
) -> dict:
    """
    Collect per-model sound-call shares and accuracy by condition.

    Args:
        metrics_path: findings_metrics.json.
        h2_path: H2_statistical_tests_corrected.json.

    Returns:
        Dict with per-model rows, pooled accuracy and the expert's sound share.
    """
    metrics = json.loads(metrics_path.read_text())
    h2 = json.loads(h2_path.read_text())["test_1_mcnemar"]
    shift = metrics["few_shot_shift"]["by_model"]
    rows = []
    for m in MODEL_ORDER:
        rows.append({
            "model": m,
            "label": MODEL_DISPLAY_NAMES[m],
            "family": ARCHITECTURE_MAP[m],
            "sound_zero": shift[m]["zero_shot_calls_sound_pct"],
            "sound_few": shift[m]["few_shot_calls_sound_pct"],
            "acc_zero": 100 * h2[m]["zero_shot_pass_rate"],
            "acc_few": 100 * h2[m]["few_shot_pass_rate"],
        })
    locked = metrics["error_asymmetry"]["locked"]
    n_fragments = metrics["pass_rate"]["corrected"]["n_fragments"]
    return {
        "rows": rows,
        "pooled_acc_zero": 100 * h2["overall"]["zero_shot_pass_rate"],
        "pooled_acc_few": 100 * h2["overall"]["few_shot_pass_rate"],
        "pooled_p": h2["overall"]["p_value"],
        "expert_sound_pct": 100 * locked["n_sound_labels"] / n_fragments,
        "flips": metrics["few_shot_shift"]["pooled"],
    }


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """
    Nudge label positions apart so no two are closer than min_gap.

    Args:
        values: Target y positions.
        min_gap: Minimum vertical spacing (data units).

    Returns:
        Adjusted positions in the original order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap)
    shift = (sum(placed) - sum(values[i] for i in order)) / len(placed)
    placed = [p - shift for p in placed]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + min_gap)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = placed[k]
    return out


def make_calibration_slope_chart(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw the calibration-example slope chart (PNG and PDF).

    Args:
        data: Output of load_slope_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = data["rows"]
    x = [0, 1]
    fig, ax_s = plt.subplots(figsize=(6.8, 5.8))

    # Left: share called sound
    for r in rows:
        ax_s.plot(x, [r["sound_zero"], r["sound_few"]], color=FAMILY_COLORS[r["family"]],
                  lw=2.6, marker="o", ms=7, ls="-" if r["family"] == "closed" else "--")
    label_y = _spread_labels([r["sound_few"] for r in rows], min_gap=4.2)
    for r, y in zip(rows, label_y):
        ax_s.text(1.06, y, f"{r['label']} {r['sound_few']:.0f}%", va="center", fontsize=11,
                  color=POSTER_LABEL_COLOR)
    left_y = _spread_labels([r["sound_zero"] for r in rows], min_gap=4.2)
    for r, y in zip(rows, left_y):
        ax_s.text(-0.06, y, f"{r['sound_zero']:.0f}%", va="center", ha="right", fontsize=11,
                  color=POSTER_LABEL_COLOR)
    ax_s.axhline(data["expert_sound_pct"], color="0.45", lw=1.2, ls=":")
    ax_s.text(1.06, data["expert_sound_pct"] - 1.5, f"Expert: {data['expert_sound_pct']:.0f}%",
              ha="left", va="top", fontsize=11, color="0.35", style="italic")
    ax_s.set_title('Called the conclusion "sound"', fontsize=14, pad=10)

    ax_s.set_xticks(x, ["Without\nexamples", "With\nexamples"])
    ax_s.set_xlim(-0.35, 1.75)
    ax_s.set_ylim(20, 100)
    ax_s.tick_params(axis="x", length=0)
    ax_s.spines["bottom"].set_visible(False)
    ax_s.set_ylabel("% of 150 conclusions")

    handles = [
        Line2D([], [], color=FAMILY_COLORS["closed"], lw=2.2, ls="-", label="Closed models"),
        Line2D([], [], color=FAMILY_COLORS["open"], lw=2.2, ls="--", label="Open models"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    paths = [output_dir / "calibration_slope.png", output_dir / "calibration_slope.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def load_missed_judgment_data(metrics_path: Path = FINDINGS_METRICS_PATH) -> dict:
    """
    Collect the share of no-judgment conclusions among flawed conclusions and among misses.

    Corpus share: judgment-absent passages / all passages the expert judged
    not sound (author's classification of the expert's reasons, E25).
    Miss share: judgment-absent units / all missed units (model x passage
    majority verdicts called "sound" on not-sound passages), pooled across
    conditions, with its fragment-clustered bootstrap CI (E28).

    Args:
        metrics_path: findings_metrics.json.

    Returns:
        Dict with counts, corpus share and miss share (pct and CI).
    """
    m = json.loads(metrics_path.read_text())
    counts = m["judgment_absent_rates"]["counts"]
    n_flawed = sum(counts.values())
    share = m["missed_flaw_ci"]["judgment_absent_share_of_misses"]["pooled"]
    return {
        "n_flawed": n_flawed,
        "n_judgment_absent": counts["judgment_absent"],
        "corpus_pct": 100 * counts["judgment_absent"] / n_flawed,
        "miss_pct": share["pct"],
        "miss_ci": share["ci95"],
    }


def make_missed_judgment_chart(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw two stacked bars: no-judgment share of flawed conclusions vs of models' misses.

    Args:
        data: Output of load_missed_judgment_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        (f"Conclusions the expert\njudged flawed ({data['n_flawed']})", data["corpus_pct"]),
        ("Flawed conclusions\nthe models passed", data["miss_pct"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, (_, pct) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, pct, 0.62, color=POSTER_ACCENT, edgecolor=POSTER_ACCENT_DARK, lw=1.2)
        ax.barh(y, 100 - pct, 0.62, left=pct, color=POSTER_ACCENT_PALE, edgecolor=POSTER_ACCENT_DARK, lw=1.2)
        ax.text(pct / 2, y, f"{pct:.0f}%", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(pct + (100 - pct) / 2, y, f"{100 - pct:.0f}%", ha="center", va="center", fontsize=13,
                color="0.3")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0, len(rows) - 0.35, "No judgment", fontsize=12, fontweight="bold", va="bottom")
    ax.text(100, len(rows) - 0.35, "Other flaws", fontsize=12, color="0.3", va="bottom", ha="right")
    lo, hi = data["miss_ci"]
    ax.text(0, -0.55, f"Models' share: all six models, both conditions (95% CI {lo:.0f}–{hi:.0f}%).",
            fontsize=10, color="0.4", va="top")
    fig.tight_layout()

    paths = [output_dir / "missed_judgment.png", output_dir / "missed_judgment.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def load_unit_chart_data() -> dict:
    """
    Count, for each flawed conclusion, how many of the six models passed it (E30).

    A model "passes" a conclusion when its majority label (>= 2 of 3 runs) is
    "sound" on a conclusion the expert's locked label rates not sound. No
    majority label counts as not passed.

    Returns:
        Dict: for each condition, lists of per-conclusion pass counts (0-6) for
        no-judgment conclusions and for all other flawed conclusions, sorted
        high to low; plus summary counts.
    """
    import pandas as pd

    from src.analysis.config import GOLD_NOT_SOUND_REASONS_PATH, SCORED_DB_CORRECTED_PATH
    from src.analysis.findings_metrics import majority_labels

    maj = majority_labels(pd.read_csv(SCORED_DB_CORRECTED_PATH))
    reasons = pd.read_csv(GOLD_NOT_SOUND_REASONS_PATH)[["fragment_id", "not_sound_reason"]]
    reasons["group"] = (reasons["not_sound_reason"] == "judgment_absent").map(
        {True: "no_judgment", False: "other"})
    passed = (maj.assign(passed=(maj["pred"] == "sound").astype(int))
                 .groupby(["prompt_condition", "fragment_id"])["passed"].sum().reset_index())
    out = {"conditions": {}}
    for cond in ("zero_shot", "few_shot"):
        d = reasons.merge(passed[passed["prompt_condition"] == cond], on="fragment_id", how="left")
        d["passed"] = d["passed"].fillna(0).astype(int)
        groups = {}
        for g in ("no_judgment", "other"):
            vals = sorted(d.loc[d["group"] == g, "passed"].tolist(), reverse=True)
            groups[g] = {
                "counts": vals,
                "n": len(vals),
                "distribution": {k: vals.count(k) for k in range(7)},
                "majority_passed": sum(v >= 4 for v in vals),
            }
        out["conditions"][cond] = groups
    return out


def make_unit_chart(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw one square per flawed conclusion, shaded by how many models passed it.

    Two panels (without / with examples); in each, no-judgment conclusions and
    other flawed conclusions are separate blocks, sorted from most to fewest
    models passing.

    Args:
        data: Output of load_unit_chart_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    ramp = LinearSegmentedColormap.from_list("accent", [POSTER_ACCENT_PALE, POSTER_ACCENT, POSTER_ACCENT_DARK])
    shade = {0: "#e6e6e6", **{k: to_hex(ramp((k - 1) / 5)) for k in range(1, 7)}}
    cols, size, gap = 8, 1.0, 0.18
    titles = {"zero_shot": "Without examples", "few_shot": "With examples"}
    labels = {"no_judgment": "No judgment", "other": "Other flaws"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.6))
    for ax, cond in zip(axes, ("zero_shot", "few_shot")):
        y0 = 0.0
        for g in ("no_judgment", "other"):
            vals = data["conditions"][cond][g]["counts"]
            n_rows = -(-len(vals) // cols)
            ax.text(0, y0 + 0.35, f"{labels[g]} ({len(vals)})", fontsize=12, fontweight="bold", va="bottom")
            for i, v in enumerate(vals):
                r, c = divmod(i, cols)
                ax.add_patch(Rectangle((c * (size + gap), y0 - (r + 1) * (size + gap)), size, size,
                                       fc=shade[v], ec="white", lw=0.5))
            y0 -= n_rows * (size + gap) + 1.6
        ax.set_xlim(-0.1, cols * (size + gap))
        ax.set_ylim(y0 + 0.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[cond], fontsize=14, pad=4)

    # Legend: models that passed the conclusion
    lax = fig.add_axes([0.2, 0.02, 0.6, 0.06])
    lax.axis("off")
    lax.set_xlim(0, 7)
    lax.set_ylim(0, 1)
    for k in range(7):
        lax.add_patch(Rectangle((k + 0.1, 0.25), 0.8, 0.6, fc=shade[k], ec="white"))
        lax.text(k + 0.5, 0.55, str(k), ha="center", va="center", fontsize=11,
                 color="white" if k >= 5 else "black")
    fig.text(0.5, 0.11, "Number of the six models that called it sound", ha="center", fontsize=11,
             color="0.3")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.14, wspace=0.12)

    paths = [output_dir / "unit_chart.png", output_dir / "unit_chart.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


# Domain 1 / Domain 4 synthesis as recorded in docs/findings/conclusions.md
# (2026-11-13). Text, not computed: the ratings are the author's judgments.
RUBRIC_ROWS: list[dict] = [
    {"criterion": "Detecting a missing judgment", "importance": "Extremely",
     "rating": ["Fair"], "evidence": "Passed 51% of them without examples, 61% with"},
    {"criterion": "Agreement with the expert", "importance": "Very",
     "rating": ["Good"], "evidence": "57% vs trained evaluators' 55–61% (38 passages)"},
    {"criterion": "Reasoning in rationales", "importance": "Very",
     "rating": [None], "evidence": "Not rated: too little evidence"},
    {"criterion": "Balance of errors", "importance": "Important",
     "rating": ["Good", "Poor"], "evidence": "Like trained evaluators without examples (short\npassages); far more lenient with them"},
    {"criterion": "Unswayed by irrelevant features", "importance": "Important",
     "rating": ["Poor"], "evidence": '"Sound" calls rose with length; the expert\'s did not'},
]
# Gradations of the poster accent colour (#f2c503), light to dark.
RUBRIC_LEVEL_COLORS: dict[str, str] = {
    "Poor": "#fdf4c4", "Fair": "#f8df6e", "Good": "#f2c503", "Excellent": "#a88900",
}


def make_rubric_table(rows: list[dict] = RUBRIC_ROWS, overall: str = "Fair",
                      output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw the synthesis rubric as a near-square figure for a narrow poster column.

    Each row shows the criterion (importance beneath it), a rating chip and
    wrapped evidence text.

    Args:
        rows: Criterion rows, most important first.
        overall: Overall verdict.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = {"criterion": 0.0, "rating": 0.40, "evidence": 0.64}
    chip_w = 0.21
    row_h = 1.25
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-(n + 3.3) * row_h, 0.6)
    ax.axis("off")

    for key, label in [("criterion", "Criterion"), ("rating", "Rating"), ("evidence", "Evidence")]:
        ax.text(cols[key], 0.1, label, fontsize=13, fontweight="bold", va="center")
    ax.plot([0, 1], [-0.45, -0.45], color="black", lw=1.2)

    def chip(x, y, level, w=chip_w):
        """Draw one rating chip centred vertically on y."""
        if level is None:
            ax.add_patch(Rectangle((x, y - 0.33), w, 0.66, fc="white", ec="0.55", lw=1, ls="--"))
            ax.text(x + w / 2, y, "Not rated", ha="center", va="center", fontsize=11, color="0.4")
            return
        ax.add_patch(Rectangle((x, y - 0.33), w, 0.66, fc=RUBRIC_LEVEL_COLORS[level], ec="none"))
        ax.text(x + w / 2, y, level, ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if level == "Excellent" else "black")

    def wrap(text, width):
        """Wrap text to a character width, keeping explicit line breaks out."""
        return textwrap.fill(text.replace("\n", " "), width)

    for i, r in enumerate(rows):
        y = -(i + 1) * row_h
        ax.text(cols["criterion"], y + 0.17, wrap(r["criterion"], 22), va="center", fontsize=12.5,
                linespacing=1.1)
        ax.text(cols["criterion"], y - 0.42, f"{r['importance']} important".replace("Important important", "Important"),
                va="center", fontsize=10, color="0.45", style="italic")
        if len(r["rating"]) == 1:
            chip(cols["rating"], y, r["rating"][0])
        else:
            half = (chip_w - 0.01) / 2
            chip(cols["rating"], y, r["rating"][0], w=half)
            chip(cols["rating"] + half + 0.01, y, r["rating"][1], w=half)
        ax.text(cols["evidence"], y, wrap(r["evidence"], 26), va="center", fontsize=11, color="0.2",
                linespacing=1.15)
        if i < n - 1:
            ax.plot([0, 1], [y - 0.62, y - 0.62], color="0.85", lw=0.8)

    y = -(n + 1) * row_h - 0.05
    ax.plot([0, 1], [y + 0.62, y + 0.62], color="black", lw=1.2)
    ax.text(cols["criterion"], y, "Overall verdict", va="center", fontsize=13.5, fontweight="bold")
    chip(cols["rating"], y, overall)
    ax.text(cols["evidence"], y, wrap("Capped at Fair until models reliably catch missing judgments", 26),
            va="center", fontsize=11, color="0.2", linespacing=1.15)

    ky = y - 1.15
    ax.text(0, ky, "Scale:", fontsize=10, color="0.35", va="center")
    key = [("Poor", "falls short"), ("Fair", "partial"), ("Good", "human parity"),
           ("Excellent", "expert level")]
    for j, (level, desc) in enumerate(key):
        kx = 0.11 + (j % 2) * 0.40
        kyj = ky - (j // 2) * 0.6
        ax.add_patch(Rectangle((kx, kyj - 0.2), 0.035, 0.4, fc=RUBRIC_LEVEL_COLORS[level], ec="none"))
        ax.text(kx + 0.05, kyj, f"{level} ({desc})", fontsize=10, color="0.35", va="center")
    ax.text(0, ky - 1.3, "Split rating: without | with examples.", fontsize=10, color="0.35", va="center")

    fig.tight_layout()
    paths = [output_dir / "rubric_table.png", output_dir / "rubric_table.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def run_poster_figures() -> list[Path]:
    """
    Build all poster figures.

    Returns:
        Paths written.
    """
    return (make_calibration_slope_chart(load_slope_data())
            + make_missed_judgment_chart(load_missed_judgment_data())
            + make_rubric_table()
            + make_unit_chart(load_unit_chart_data()))


if __name__ == "__main__":
    for p in run_poster_figures():
        print(p)
