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
                  lw=2.2, marker="o", ms=6, ls="-" if r["family"] == "closed" else "--")
    label_y = _spread_labels([r["sound_few"] for r in rows], min_gap=4.2)
    for r, y in zip(rows, label_y):
        ax_s.text(1.06, y, f"{r['label']} {r['sound_few']:.0f}%", va="center", fontsize=11,
                  color=FAMILY_COLORS[r["family"]])
    left_y = _spread_labels([r["sound_zero"] for r in rows], min_gap=4.2)
    for r, y in zip(rows, left_y):
        ax_s.text(-0.06, y, f"{r['sound_zero']:.0f}%", va="center", ha="right", fontsize=11,
                  color=FAMILY_COLORS[r["family"]])
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
    Collect models' miss rates on flawed passages by flaw type and condition.

    Miss rate = share of model x passage units (majority verdicts) called
    "sound" among passages the expert judged not sound, by the author's
    classification of the expert's reasons (E25). Buried-judgment and
    stale-rationale passages (6 each) are left out as boundary categories.

    Args:
        metrics_path: findings_metrics.json.

    Returns:
        Dict with miss rates, passage counts and the share of misses.
    """
    m = json.loads(metrics_path.read_text())
    ja = m["judgment_absent_rates"]
    rates = ja["pooled_miss_rate"]
    return {
        "conditions": ["zero_shot", "few_shot"],
        "groups": {
            "judgment_absent": {
                "n": ja["counts"]["judgment_absent"],
                "rates": [rates[c]["judgment_absent"]["miss_rate_pct"] for c in ("zero_shot", "few_shot")],
            },
            "other_flaw": {
                "n": ja["counts"]["other_flaw"],
                "rates": [rates[c]["other_flaw"]["miss_rate_pct"] for c in ("zero_shot", "few_shot")],
            },
        },
        "share_of_misses": m["missed_flaw_ci"]["judgment_absent_share_of_misses"]["pooled"],
    }


def make_missed_judgment_chart(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw grouped bars of miss rates by flaw type and condition (PNG and PDF).

    Args:
        data: Output of load_missed_judgment_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    g = data["groups"]
    styles = [
        ("judgment_absent", f"No judgment ({g['judgment_absent']['n']} conclusions)", FAMILY_COLORS["closed"]),
        ("other_flaw", f"Other flaws ({g['other_flaw']['n']} conclusions)", "#a6a6a6"),
    ]
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for i, (key, label, color) in enumerate(styles):
        xs = [k + (i - 0.5) * width for k in range(2)]
        bars = ax.bar(xs, g[key]["rates"], width * 0.94, color=color, label=label)
        for b, v in zip(bars, g[key]["rates"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold" if key == "judgment_absent" else "normal")
    ax.set_xticks([0, 1], ["Without\nexamples", "With\nexamples"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of flawed conclusions called "sound"')
    ax.set_title("Flawed conclusions the models passed", fontsize=14, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()

    paths = [output_dir / "missed_judgment.png", output_dir / "missed_judgment.pdf"]
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
RUBRIC_LEVEL_COLORS: dict[str, str] = {
    "Poor": "#d7e3f4", "Fair": "#9fbde3", "Good": "#4f7fc4", "Excellent": "#1f4e9c",
}


def make_rubric_table(rows: list[dict] = RUBRIC_ROWS, overall: str = "Fair",
                      output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw the synthesis rubric (criteria, importance, rating, evidence) as a figure.

    Args:
        rows: Criterion rows, most important first.
        overall: Overall verdict.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = {"criterion": 0.0, "importance": 0.33, "rating": 0.46, "evidence": 0.61}
    row_h = 1.0
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12.5, 0.62 * (n + 3.2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-(n + 2.4) * row_h, 1.0)
    ax.axis("off")

    head_y = 0.2
    for key, label in [("criterion", "Criterion"), ("importance", "Importance"),
                       ("rating", "Rating"), ("evidence", "Evidence")]:
        ax.text(cols[key], head_y, label, fontsize=12.5, fontweight="bold", va="center")
    ax.plot([0, 1], [-0.35, -0.35], color="black", lw=1.2)

    def chip(x, y, level, w=0.135):
        """Draw one rating chip at (x, y)."""
        if level is None:
            ax.add_patch(Rectangle((x, y - 0.3), w, 0.6, fc="white", ec="0.55", lw=1, ls="--"))
            ax.text(x + w / 2, y, "Not rated", ha="center", va="center", fontsize=11, color="0.4")
            return
        fc = RUBRIC_LEVEL_COLORS[level]
        tc = "white" if level in ("Good", "Excellent") else "black"
        ax.add_patch(Rectangle((x, y - 0.3), w, 0.6, fc=fc, ec="none"))
        ax.text(x + w / 2, y, level, ha="center", va="center", fontsize=11.5, color=tc, fontweight="bold")

    for i, r in enumerate(rows):
        y = -(i + 1) * row_h
        ax.text(cols["criterion"], y, r["criterion"], va="center", fontsize=12)
        ax.text(cols["importance"], y, r["importance"], va="center", fontsize=12, color="0.25")
        if len(r["rating"]) == 1:
            chip(cols["rating"], y, r["rating"][0])
        else:
            chip(cols["rating"], y, r["rating"][0], w=0.065)
            chip(cols["rating"] + 0.07, y, r["rating"][1], w=0.065)
        ax.text(cols["evidence"], y, r["evidence"], va="center", fontsize=11, color="0.2")
        if i < n - 1:
            ax.plot([0, 1], [y - 0.5, y - 0.5], color="0.85", lw=0.8)

    y = -(n + 1) * row_h - 0.1
    ax.plot([0, 1], [y + 0.5, y + 0.5], color="black", lw=1.2)
    ax.text(cols["criterion"], y, "Overall verdict", va="center", fontsize=13, fontweight="bold")
    chip(cols["rating"], y, overall)
    ax.text(cols["evidence"], y, "Capped at Fair until models reliably\ncatch missing judgments",
            va="center", fontsize=11, color="0.2")
    ax.text(0, y - 1.05, "Scale: Poor (falls short) · Fair (partial) · Good (human parity) · "
            "Excellent (expert level). Split rating: without | with examples.",
            fontsize=10, color="0.35", va="center")

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
            + make_rubric_table())


if __name__ == "__main__":
    for p in run_poster_figures():
        print(p)
