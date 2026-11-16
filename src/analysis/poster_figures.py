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
    POSTER_FONT_DIR,
    POSTER_FONT_FAMILY,
)


def _use_poster_font() -> None:
    """Register the bundled Montserrat files with matplotlib and make it the default family."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in sorted(POSTER_FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = POSTER_FONT_FAMILY


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
    _use_poster_font()
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
    _use_poster_font()

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
    _use_poster_font()
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


def make_funnel_chart(unit: dict, n_total: int = 150, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw a narrowing funnel: all conclusions -> flawed -> no judgment -> passed by most models.

    Counts come from load_unit_chart_data (E30); "most models" = at least 4
    of 6 majority labels "sound". The last stage is split by condition.

    Args:
        unit: Output of load_unit_chart_data.
        n_total: Number of benchmark conclusions.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_poster_font()

    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    zs, fs = unit["conditions"]["zero_shot"], unit["conditions"]["few_shot"]
    n_flawed = zs["no_judgment"]["n"] + zs["other"]["n"]
    n_nj = zs["no_judgment"]["n"]
    stages = [
        (n_total, "UN evaluation conclusions", POSTER_ACCENT_PALE),
        (n_flawed, "judged flawed by the expert", "#f8df6e"),
        (n_nj, "never reach a judgment", POSTER_ACCENT),
    ]
    scale = 1.0 / n_total
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    h, step = 0.72, 1.15
    for i, (n, label, color) in enumerate(stages):
        y = -i * step
        w = n * scale
        ax.barh(y, w, h, left=0.5 - w / 2, color=color, edgecolor=POSTER_ACCENT_DARK, lw=1.2)
        ax.text(0.5, y + 0.02, f"{n}", ha="center", va="center", fontsize=17, fontweight="bold")
        ax.text(0.5, y - h / 2 - 0.08, label, ha="center", va="top", fontsize=12, color="0.25")

    # Final stage, split by condition
    y = -len(stages) * step - 0.25
    final = [(zs["no_judgment"]["majority_passed"], "without examples"),
             (fs["no_judgment"]["majority_passed"], "with examples")]
    for (n, cond), cx in zip(final, (0.33, 0.67)):
        w = n * scale
        ax.barh(y, w, h, left=cx - w / 2, color=POSTER_ACCENT_DARK, edgecolor=POSTER_ACCENT_DARK, lw=1.2)
        ax.text(cx, y + 0.02, f"{n}", ha="center", va="center", fontsize=17, fontweight="bold", color="white")
        ax.text(cx, y - h / 2 - 0.08, cond, ha="center", va="top", fontsize=12, color="0.25")
    ax.text(0.5, y + h / 2 + 0.12, "passed by most models (4+ of 6)", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="0.15")

    ax.set_xlim(0, 1)
    ax.set_ylim(y - 1.0, 0.6)
    ax.axis("off")
    fig.tight_layout()
    paths = [output_dir / "funnel.png", output_dir / "funnel.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_length_chart(metrics_path: Path = FINDINGS_METRICS_PATH,
                      output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw the share of conclusions called "sound" by passage length: models vs the expert.

    Values from findings_metrics.json -> length_moderation (Finding 2.2):
    models' share of majority verdicts "sound" per condition, and the
    expert's share (100 - gold not-sound %).

    Args:
        metrics_path: findings_metrics.json.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_poster_font()

    lm = json.loads(metrics_path.read_text())["length_moderation"]["by_length"]
    bins = ["1-3 paragraphs", "4-6 paragraphs", "7+ paragraphs"]
    xlabels = [f"Short\n1–3 paragraphs", f"Medium\n4–6", f"Long\n7+"]
    series = [
        ("Models, with examples", [lm[b]["few_shot"]["models_call_sound_pct"] for b in bins],
         POSTER_ACCENT_DARK, "-", "o"),
        ("Models, without examples", [lm[b]["zero_shot"]["models_call_sound_pct"] for b in bins],
         POSTER_ACCENT, "-", "o"),
        ("Expert", [100 - lm[b]["gold_not_sound_pct"] for b in bins], "#404040", "--", "s"),
    ]
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    x = list(range(len(bins)))
    for label, ys, color, ls, mk in series:
        ax.plot(x, ys, color=color, ls=ls, marker=mk, lw=3, ms=9)
        below = label == "Expert"
        for xi, yi in zip(x, ys):
            if xi == 0 and label != "Models, with examples":
                # Short bin: expert and zero-shot points nearly coincide; label to the left
                ax.text(xi - 0.1, yi + (1.6 if below else -1.6), f"{yi:.0f}%", ha="right",
                        va="center", fontsize=11, color=POSTER_LABEL_COLOR)
                continue
            ax.text(xi, yi - 2.4 if below else yi + 2.2, f"{yi:.0f}%", ha="center",
                    va="top" if below else "bottom", fontsize=11, color=POSTER_LABEL_COLOR)
        ax.text(x[-1] + 0.12, ys[-1], label, va="center", fontsize=12, color=POSTER_LABEL_COLOR,
                fontweight="bold" if label.startswith("Models") else "normal")
    ax.set_xticks(x, xlabels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 3.35)
    ax.set_ylim(30, 90)
    ax.set_ylabel('% of conclusions called "sound"')
    ax.set_title("Longer conclusions looked sounder to models, not to the expert", fontsize=13.5, pad=12)
    fig.tight_layout()
    paths = [output_dir / "length.png", output_dir / "length.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_length_chart_compact(metrics_path: Path = FINDINGS_METRICS_PATH,
                              output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Compact, larger-type variant of the length chart, sized to match human_comparison_compact.

    Same values as make_length_chart; no title (set in the poster layout);
    legend inside the empty lower band of the plot. Written to length_compact.*.

    Args:
        metrics_path: findings_metrics.json.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_poster_font()
    from matplotlib.lines import Line2D

    lm = json.loads(metrics_path.read_text())["length_moderation"]["by_length"]
    bins = ["1-3 paragraphs", "4-6 paragraphs", "7+ paragraphs"]
    series = [
        ("Models, with examples", [lm[b]["few_shot"]["models_call_sound_pct"] for b in bins],
         POSTER_ACCENT_DARK, "-", "o"),
        ("Models, without examples", [lm[b]["zero_shot"]["models_call_sound_pct"] for b in bins],
         POSTER_ACCENT, "-", "o"),
        ("Expert", [100 - lm[b]["gold_not_sound_pct"] for b in bins], "#404040", "--", "s"),
    ]
    plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.6, 5.3))
    ax = fig.add_axes((0.17, 0.20, 0.80, 0.77))
    x = list(range(len(bins)))
    for label, ys, color, ls, mk in series:
        ax.plot(x, ys, color=color, ls=ls, marker=mk, lw=4, ms=12)
        below = label == "Expert"
        for xi, yi in zip(x, ys):
            if xi == 0 and label != "Models, with examples":
                ax.text(xi - 0.13, yi + (1.8 if below else -1.8), f"{yi:.0f}%", ha="right", va="center",
                        fontsize=15, color=POSTER_LABEL_COLOR)
                continue
            ax.text(xi, yi - 2.6 if below else yi + 2.4, f"{yi:.0f}%", ha="center",
                    va="top" if below else "bottom", fontsize=15, color=POSTER_LABEL_COLOR)
    ax.set_xticks(x, ["Short\n1–3", "Medium\n4–6", "Long\n7+"])
    ax.set_xlabel("Conclusion length (paragraphs)", fontsize=15, labelpad=8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.35)
    ax.set_ylim(25, 88)
    ax.set_yticks([30, 40, 50, 60, 70, 80])
    ax.set_ylabel('% called "sound"')
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, lw=4, ms=10, label=l) for l, _, c, ls, mk in series]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=13.5, handlelength=1.8,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    paths = [output_dir / "length_compact.png", output_dir / "length_compact.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def load_human_comparison_data() -> dict:
    """
    Agreement with the expert's blind second-pass ratings on the 38 shared RC2 passages.

    Recomputes the per model x condition values behind
    findings_metrics.json -> human_comparison.shared (mean 57.5%, range
    42.1-76.3%) for display; raters from the same entry.

    Returns:
        Dict with per-model rows and rater agreement.
    """
    import pandas as pd

    from src.analysis.config import RC2_BLIND_RETEST_PATH, SCORED_DB_CORRECTED_PATH
    from src.analysis.findings_metrics import _norm_label, majority_labels
    from src.scoring.accuracy import load_gold_standard

    maj = majority_labels(pd.read_csv(SCORED_DB_CORRECTED_PATH))
    h = pd.read_csv(RC2_BLIND_RETEST_PATH).rename(columns={"report_id": "fragment_id"})
    for c in ("rater_1", "rater_2", "anchor3_pass2"):
        h[c] = _norm_label(h[c])
    gold = load_gold_standard()
    h = h[h["fragment_id"].isin(gold["fragment_id"])]
    x = h.merge(maj[maj["pred"] != "none"], on="fragment_id")
    rows = []
    for (m, c), d in x.groupby(["model_family", "prompt_condition"]):
        rows.append({"model": m, "condition": c, "pct": 100 * (d["pred"] == d["anchor3_pass2"]).mean()})
    raters = {r: 100 * (h[r] == h["anchor3_pass2"]).mean() for r in ("rater_1", "rater_2")}
    return {"rows": rows, "raters": raters, "n": int(len(h))}


def make_human_comparison_chart(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Draw a dot plot: each model's agreement with the expert vs two trained evaluators.

    Args:
        data: Output of load_human_comparison_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_poster_font()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["model"], r["condition"]): r["pct"] for r in data["rows"]}
    models = [m for m in MODEL_ORDER if (m, "zero_shot") in by]
    lo, hi = min(data["raters"].values()), max(data["raters"].values())
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    n_rows = len(models)
    ax.add_patch(Rectangle((lo, -1.5), hi - lo, n_rows + 1.1, fc=POSTER_ACCENT_PALE, ec="none", zorder=0))
    ax.text((lo + hi) / 2, -0.95, f"Trained\nevaluators\n{lo:.0f}–{hi:.0f}%", ha="center",
            va="center", fontsize=10.5, fontweight="bold", color=POSTER_LABEL_COLOR, linespacing=1.1)
    mean = sum(r["pct"] for r in data["rows"]) / len(data["rows"])
    ax.plot([mean, mean], [-0.4, n_rows - 0.45], color=POSTER_LABEL_COLOR, ls=(0, (2, 2)), lw=1.4, zorder=1)
    ax.text(mean, n_rows - 0.4, f"Model average {mean:.0f}%", ha="center", va="bottom", fontsize=10.5,
            color=POSTER_LABEL_COLOR)
    for i, m in enumerate(models[::-1]):
        z, f = by[(m, "zero_shot")], by.get((m, "few_shot"))
        ax.plot([z, f], [i, i], color="0.75", lw=1.5, zorder=1)
        ax.plot(z, i, "o", ms=11, mfc="white", mec=POSTER_ACCENT_DARK, mew=2.2, zorder=2)
        ax.plot(f, i, "o", ms=11, color=POSTER_ACCENT_DARK, zorder=2)
    ax.set_yticks(range(n_rows), [MODEL_DISPLAY_NAMES[m] for m in models[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(35, 85)
    ax.set_ylim(-1.55, n_rows + 0.1)
    ax.set_title("On average, models agreed with the expert about as often\n"
                 "as trained evaluators did, but individual models varied widely",
                 fontsize=13.5, pad=12)
    ax.set_xlabel(f"% agreement with the expert's blind ratings ({data['n']} short conclusions)")
    handles = [Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec=POSTER_ACCENT_DARK, mew=2,
                      label="Without examples"),
               Line2D([], [], marker="o", ls="", ms=10, color=POSTER_ACCENT_DARK, label="With examples")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    paths = [output_dir / "human_comparison.png", output_dir / "human_comparison.pdf"]
    fig.savefig(paths[0], dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def make_human_comparison_chart_compact(data: dict, output_dir: Path = POSTER_FIGURES_DIR) -> list[Path]:
    """
    Narrower, larger-type variant of the human-comparison dot plot for a slim poster column.

    Same data and encodings as make_human_comparison_chart; the evaluator band
    is explained in the legend rather than labelled in the plot. Written to
    human_comparison_compact.* so the original is kept. Layout is fixed by
    hand (no tight_layout) so the title sits directly above the plot.

    Args:
        data: Output of load_human_comparison_data.
        output_dir: Destination directory.

    Returns:
        Paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_poster_font()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["model"], r["condition"]): r["pct"] for r in data["rows"]}
    models = [m for m in MODEL_ORDER if (m, "zero_shot") in by]
    lo, hi = min(data["raters"].values()), max(data["raters"].values())
    n_rows = len(models)

    fig = plt.figure(figsize=(5.6, 5.3))
    ax = fig.add_axes((0.40, 0.30, 0.56, 0.62))
    # Title omitted: the author sets it in the poster layout (2026-11-16).

    ax.add_patch(Rectangle((lo, -0.6), hi - lo, n_rows + 0.2, fc=POSTER_ACCENT_PALE, ec="none", zorder=0))
    mean = sum(r["pct"] for r in data["rows"]) / len(data["rows"])
    ax.plot([mean, mean], [-0.6, n_rows - 0.45], color=POSTER_LABEL_COLOR, ls=(0, (2, 2)), lw=1.6, zorder=1)
    ax.text(mean, n_rows - 0.4, f"Model average {mean:.0f}%", ha="center", va="bottom", fontsize=13,
            color=POSTER_LABEL_COLOR)
    for i, m in enumerate(models[::-1]):
        z, f = by[(m, "zero_shot")], by.get((m, "few_shot"))
        ax.plot([z, f], [i, i], color="0.75", lw=2, zorder=1)
        ax.plot(z, i, "o", ms=14, mfc="white", mec=POSTER_ACCENT_DARK, mew=2.6, zorder=2)
        ax.plot(f, i, "o", ms=14, color=POSTER_ACCENT_DARK, zorder=2)
    ax.set_yticks(range(n_rows), [MODEL_DISPLAY_NAMES[m] for m in models[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(40, 80)
    ax.set_xticks([40, 50, 60, 70, 80])
    ax.set_ylim(-0.7, n_rows + 0.1)
    ax.set_xlabel(f"% agreement with the expert\n({data['n']} short conclusions)")

    handles = [Line2D([], [], marker="o", ls="", ms=12, mfc="white", mec=POSTER_ACCENT_DARK, mew=2.4,
                      label="Without examples"),
               Line2D([], [], marker="o", ls="", ms=12, color=POSTER_ACCENT_DARK, label="With examples"),
               Patch(fc=POSTER_ACCENT_PALE, ec="none", label=f"Trained evaluators\n{lo:.0f}–{hi:.0f}%")]
    # Legend fills the empty block under the model names, left of the x-axis label
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.265), ncol=1, frameon=False,
               fontsize=13, handletextpad=0.35, labelspacing=0.35, handlelength=1.2, borderaxespad=0)

    paths = [output_dir / "human_comparison_compact.png", output_dir / "human_comparison_compact.pdf"]
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
    _use_poster_font()
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
            + make_unit_chart(unit := load_unit_chart_data())
            + make_funnel_chart(unit)
            + make_length_chart()
            + make_length_chart_compact()
            + make_human_comparison_chart(hc := load_human_comparison_data())
            + make_human_comparison_chart_compact(hc))


if __name__ == "__main__":
    for p in run_poster_figures():
        print(p)
