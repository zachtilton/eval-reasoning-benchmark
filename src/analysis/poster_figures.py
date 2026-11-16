"""
Poster figures built from existing results files (no new analysis).

Figure: calibration-example slope chart. Left panel: each model's share of
passages called "sound" without vs with the two calibration examples
(findings_metrics.json -> few_shot_shift). Right panel: each model's
agreement with the expert's locked ratings in the same two conditions
(H2_statistical_tests_corrected.json -> test_1_mcnemar), with the pooled
change. The pairing shows verdicts shifting while accuracy barely moves.

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
    Draw the two-panel calibration-example slope chart (PNG and PDF).

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
    fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)

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

    # Right: agreement with the expert
    for r in rows:
        ax_a.plot(x, [r["acc_zero"], r["acc_few"]], color=FAMILY_COLORS[r["family"]], lw=1.2,
                  alpha=0.55, ls="-" if r["family"] == "closed" else "--")
    ax_a.plot(x, [data["pooled_acc_zero"], data["pooled_acc_few"]], color="black", lw=3.2,
              marker="o", ms=7)
    ax_a.text(1.06, data["pooled_acc_few"], f"All models\n{data['pooled_acc_few']:.0f}%",
              va="center", fontsize=11, fontweight="bold")
    ax_a.text(-0.06, data["pooled_acc_zero"], f"{data['pooled_acc_zero']:.0f}%", va="center",
              ha="right", fontsize=11, fontweight="bold")
    ax_a.set_title("Agreed with the expert", fontsize=14, pad=10)

    for ax in (ax_s, ax_a):
        ax.set_xticks(x, ["Without\nexamples", "With\nexamples"])
        ax.set_xlim(-0.35, 1.75)
        ax.set_ylim(20, 100)
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
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


def run_poster_figures() -> list[Path]:
    """
    Build all poster figures.

    Returns:
        Paths written.
    """
    return make_calibration_slope_chart(load_slope_data())


if __name__ == "__main__":
    for p in run_poster_figures():
        print(p)
