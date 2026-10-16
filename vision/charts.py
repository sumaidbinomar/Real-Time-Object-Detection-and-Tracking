"""Charts for the frame-by-frame metrics.

Colors come from a validated palette (blue / orange / aqua). Single-series charts use one color;
the three object classes also differ by line style, so color is never the only cue.
"""
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import CLASSES, STAGES

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CLASS_STYLE = {"person-like": (BLUE, "-"), "large blob": (ORANGE, "--"), "small object": (AQUA, ":")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _style(ax, ylabel: str = "", xlabel: str = "") -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _col(rows: List[Dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def plot_stage_timing(summary: Dict, path: Path) -> None:
    """Mean time per pipeline stage (one series -> one color)."""
    means = summary["stage_ms_mean"]
    order = sorted(STAGES, key=lambda s: means[s])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(order, [means[s] for s in order], color=BLUE, height=0.55)
    for i, s in enumerate(order):
        ax.text(means[s] + max(means.values()) * 0.01, i, f"{means[s]:.2f} ms  ({summary['stage_share'][s]:.0%})",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(means.values()) * 1.3)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean time per frame (ms)", fontsize=9)
    ax.set_title("Where the time goes: mean per-stage cost", fontsize=11, weight="bold", loc="left")
    fig.tight_layout()
    _save(fig, path)


def plot_fps_over_time(rows: List[Dict], path: Path, window: int = 25) -> None:
    fps, t = _col(rows, "fps"), _col(rows, "time_s")
    smooth = np.convolve(fps, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t, fps, color=BLUE, alpha=0.25, linewidth=1, label="Per-frame FPS")
    ax.plot(t[window - 1:], smooth, color=BLUE, linewidth=2, label=f"{window}-frame moving average")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    ax.set_title("Processing speed over the video", fontsize=11, weight="bold", loc="left")
    _style(ax, "Frames per second", "Video time (s)")
    fig.tight_layout()
    _save(fig, path)


def plot_latency_histogram(rows: List[Dict], summary: Dict, path: Path) -> None:
    totals = _col(rows, "total_ms")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(totals, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    p = summary["pipeline_ms"]
    for value, label in ((p["median"], f"median {p['median']:.1f} ms"), (p["p95"], f"p95 {p['p95']:.1f} ms")):
        ax.axvline(value, color=INK_2, linewidth=1)
        ax.text(value, ax.get_ylim()[1] * 0.95, " " + label, fontsize=9, color=INK_2, va="top")
    ax.set_title("Per-frame inference time distribution", fontsize=11, weight="bold", loc="left")
    _style(ax, "Frames", "Total pipeline time per frame (ms)")
    fig.tight_layout()
    _save(fig, path)


def plot_counts_over_time(rows: List[Dict], path: Path) -> None:
    t = _col(rows, "time_s")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for name in CLASSES:
        color, ls = CLASS_STYLE[name]
        top.plot(t, _col(rows, f"count_{name}"), color=color, linestyle=ls, linewidth=2, label=name)
    top.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3, loc="upper right")
    top.set_title("Live object counts per class", fontsize=11, weight="bold", loc="left")
    _style(top, "Objects in frame")
    bottom.plot(t, _col(rows, "unique_objects"), color=BLUE, linewidth=2)
    bottom.set_title("Cumulative unique objects tracked", fontsize=11, weight="bold", loc="left")
    _style(bottom, "Unique objects", "Video time (s)")
    fig.tight_layout()
    _save(fig, path)


def plot_all(result: Dict, out_dir: Path) -> None:
    rows, summary = result["rows"], result["summary"]
    plot_stage_timing(summary, out_dir / "chart_stage_timing.png")
    plot_fps_over_time(rows, out_dir / "chart_fps_over_time.png")
    plot_latency_histogram(rows, summary, out_dir / "chart_latency_histogram.png")
    plot_counts_over_time(rows, out_dir / "chart_counts_over_time.png")


def plot_sweeps(sweeps: Dict[str, List[Dict]], defaults: Dict[str, float], path: Path) -> None:
    """One-at-a-time parameter sweeps. Columns = parameter, rows = metric (separate axes, no dual-axis)."""
    metrics = [("mean_ms", "Mean time per frame (ms)"), ("mean_detections", "Mean detections per frame"),
               ("unique_objects", "Unique objects counted")]
    names = list(sweeps)
    fig, axes = plt.subplots(len(metrics), len(names), figsize=(3.3 * len(names), 8.2), squeeze=False)
    for j, name in enumerate(names):
        rows = sweeps[name]
        xs = list(range(len(rows)))
        for i, (key, ylabel) in enumerate(metrics):
            ax = axes[i][j]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
            d = [k for k, r in enumerate(rows) if r["value"] == defaults[name]]
            if d:  # ring the default setting
                ax.plot(d, [ys[d[0]]], marker="o", markersize=13, markerfacecolor="none", markeredgecolor=ORANGE, markeredgewidth=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(r["value"]) for r in rows])
            ax.set_ylim(0, max(ys) * 1.2 if max(ys) > 0 else 1)
            _style(ax, ylabel if j == 0 else "", name if i == len(metrics) - 1 else "")
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=ORANGE,
                                   markeredgewidth=2, markersize=11, label="default setting")],
               loc="upper center", frameon=False, fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Parameter sensitivity (one parameter varied at a time)", fontsize=12, weight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
