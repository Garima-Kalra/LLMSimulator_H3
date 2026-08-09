#!/usr/bin/env python3
"""Render H3 simulation results in the same chart format as the H3 paper's
Figs. 5 and 6, so simulated vs. paper deviation is visible at a glance.

Usage:
    python3 plot_results.py [results.log]

Produces two PNGs in the current directory:
    fig5_batch_size.png       -- max batch size, HBM-only vs H3 (paper Fig. 5 style)
    fig6_throughput_power.png -- throughput & throughput/power, normalized to
                                  HBM-only=1 (paper Fig. 6 style), incl. the
                                  halved-HBF-bandwidth sensitivity variant

Requires matplotlib (see .venv in this repo, or `pip install matplotlib`).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_results import PAPER, load_rows, pick_latest, scenario_key

# Palette (dataviz skill, validated categorical order for 3 series)
BLUE = "#2a78d6"    # simulated, full HBF bandwidth
AQUA = "#1baf7a"    # simulated, half HBF bandwidth
ORANGE = "#eb6834"  # paper-reported

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def fmt_x(context_window):
    return f"{context_window // 1_000_000}M" if context_window >= 1_000_000 else str(context_window)


def get(rows, context_window, use_hbf, scale, amortized=True):
    """Fetch one run. `amortized` selects the attention cost model: True =
    shared-context work charged once per batch (paper-aligned), False =
    charged per sequence (this codebase's original model). Within the
    requested mode, prefers the calibrated run (hbm_reserve_fraction > 0)
    since that is the headline configuration; falls back to uncalibrated,
    then to the other cost model only if nothing else was run."""
    cands = [r for r in rows
             if r["context_window"] == context_window
             and r["use_hbf"] == use_hbf
             and r["hbf_bandwidth_scale"] == scale]
    for want in (amortized, not amortized):
        pool = [r for r in cands if r["cag_amortize_shared_compute"] == want]
        if pool:
            return max(pool, key=lambda r: (r["hbm_reserve_fraction"],
                                            r["allreduce_hierarchical"]))
    return None


def plot_fig5(rows, contexts):
    """Paper Fig. 5 style: vertical bars, max batch size, one subplot per
    context size. Paper's ratio is annotated as text (we only have their
    ratio, not their absolute batch numbers, so we don't fabricate a bar)."""
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.5 * len(contexts), 4.6))
    if len(contexts) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, contexts):
        hbm = get(rows, ctx, False, 1.0)
        h3 = get(rows, ctx, True, 1.0)
        paper_ratio = PAPER.get(ctx, {}).get("batch_size_ratio")

        labels, values, colors, infeasible = [], [], [], []
        labels.append("HBM-only")
        colors.append(BLUE)
        if hbm is None:
            values.append(0)
            infeasible.append(True)
        else:
            values.append(hbm["final_max_batch_size"])
            infeasible.append(False)

        labels.append("H3")
        colors.append(ORANGE)
        values.append(h3["final_max_batch_size"] if h3 else 0)
        infeasible.append(h3 is None)

        x = range(len(labels))
        bars = ax.bar(x, values, width=0.55, color=colors, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Maximum Batch Size")
        ax.set_title(f"{fmt_x(ctx)} Case", fontsize=13, color=INK, pad=12)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        top = max(values) if max(values) > 0 else 1
        for bar, val, infeas in zip(bars, values, infeasible):
            if infeas:
                ax.text(bar.get_x() + bar.get_width() / 2, top * 0.05, "✗",
                        ha="center", va="bottom", fontsize=22, color=INK_MUTED,
                        fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2, -top * 0.09,
                        "infeasible\n(weights+shared\ncache > HBM)",
                        ha="center", va="top", fontsize=8, color=INK_MUTED)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.015,
                        f"{val:,}", ha="center", va="bottom", fontsize=10, color=INK)

        if h3 and hbm:
            sim_ratio = h3["final_max_batch_size"] / hbm["final_max_batch_size"]
            note = f"simulated: {sim_ratio:.2g}x"
            if paper_ratio:
                note += f"\npaper: {paper_ratio:.2g}x"
            ax.text(1, max(values) * 1.18, note, ha="center", va="bottom",
                    fontsize=9.5, color=INK_SECONDARY,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                             ec=BASELINE, lw=0.8))
        elif h3 and paper_ratio:
            ax.text(1, values[1] * 1.15, f"paper: {paper_ratio:.2g}x\n(no HBM-only\nbaseline here)",
                    ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                             ec=BASELINE, lw=0.8))
        ax.set_ylim(0, top * 1.45)

    fig.suptitle("H3 vs. HBM-only: Maximum Batch Size (cf. paper Fig. 5)",
                 fontsize=14, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig("fig5_batch_size.png", dpi=180, bbox_inches="tight")
    print("Wrote fig5_batch_size.png")


def plot_fig6(rows, contexts):
    """Paper Fig. 6 style: horizontal bars, normalized to HBM-only=1,
    one subplot for throughput and one for throughput-per-power. The
    throughput-per-power metric gets two distinct paper reference bars
    (full HBF bandwidth vs. halved) since the paper reports both, e.g.
    2.69x / 2.09x -- both are 10M-specific, so the 1M column stays "n/a"
    for both (see ASSUMPTIONS.md and extract_results.py's PAPER dict)."""
    metrics = [
        ("throughput_tps", "throughput_ratio", None, "Normalized Throughput (TPS)"),
        ("throughput_per_power", "throughput_per_power_ratio",
         "throughput_per_power_ratio_half_bw", "Normalized Throughput per Power"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    bar_h = 0.17
    for ax, (field, paper_key, paper_key_half, xlabel) in zip(axes, metrics):
        y_positions = list(range(len(contexts)))
        # (label, color, use_hbf, scale, paper_key, amortized)
        series = [
            ("Simulated H3 (full HBF BW)", BLUE, True, 1.0, paper_key, True),
            ("Simulated H3 (1/2 HBF BW)", AQUA, True, 0.5, paper_key_half, True),
            ("Simulated H3 (full BW, per-seq cost model)", INK_MUTED, True, 1.0,
             paper_key, False),
            ("Paper (reported, full BW)", ORANGE, None, None, paper_key, None),
        ]
        if paper_key_half is not None:
            series.append(("Paper (reported, 1/2 BW)", "#c9803f", None, None,
                           paper_key_half, None))

        for i, (label, color, use_hbf, scale, series_paper_key, amort) in enumerate(series):
            offsets = [y - bar_h * (len(series) - 1) / 2 + i * bar_h for y in y_positions]
            vals, missing = [], []
            for ctx in contexts:
                if use_hbf is None:
                    ratio = PAPER.get(ctx, {}).get(series_paper_key) if series_paper_key else None
                else:
                    hbm = get(rows, ctx, False, 1.0, amort)
                    h3 = get(rows, ctx, True, scale, amort)
                    # only compare runs sharing the same cost model
                    if (hbm and h3 and (
                            hbm["cag_amortize_shared_compute"]
                            != h3["cag_amortize_shared_compute"]
                            or hbm["hbm_reserve_fraction"]
                            != h3["hbm_reserve_fraction"]
                            or hbm["allreduce_hierarchical"]
                            != h3["allreduce_hierarchical"])):
                        hbm = None
                    ratio = (h3[field] / hbm[field]) if (hbm and h3 and hbm[field] > 0) else None
                vals.append(ratio if ratio is not None else 0)
                missing.append(ratio is None)

            bars = ax.barh(offsets, vals, height=bar_h * 0.92, color=color,
                           label=label, zorder=3)
            for bar, val, miss in zip(bars, vals, missing):
                y = bar.get_y() + bar.get_height() / 2
                if miss:
                    ax.text(0.05, y, "n/a", ha="left", va="center",
                            fontsize=8.5, color=INK_MUTED, style="italic")
                else:
                    ax.text(val + 0.08, y, f"{val:.2g}x", ha="left", va="center",
                            fontsize=9, color=INK)

        ax.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=2)
        ax.text(1.0, len(contexts) - 0.5, " HBM-only baseline", fontsize=8,
               color=INK_MUTED, va="bottom", ha="left", rotation=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{fmt_x(c)}" for c in contexts])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlim(0, max(2.0, ax.get_xlim()[1]))

    # Collect handles/labels across both subplots (the throughput-per-power
    # subplot has an extra "Paper (reported, 1/2 BW)" series the throughput
    # subplot doesn't), deduping by label while preserving first-seen order.
    seen = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle("H3 vs. HBM-only: Throughput & Throughput/Power (cf. paper Fig. 6)",
                 fontsize=14, color=INK, y=1.14)
    fig.tight_layout()
    fig.savefig("fig6_throughput_power.png", dpi=180, bbox_inches="tight")
    print("Wrote fig6_throughput_power.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.log"
    rows = pick_latest(load_rows(path))
    # results.log accumulates every run ever made (including plain
    # regression checks with CAG disabled); only context_window > 0 runs
    # are the H3/CAG scenarios this chart compares against the paper.
    rows = [r for r in rows if r["context_window"] > 0]
    contexts = sorted({r["context_window"] for r in rows})
    if not contexts:
        print(f"No CAG scenario data (context_window > 0) in {path}")
        sys.exit(1)

    plot_fig5(rows, contexts)
    plot_fig6(rows, contexts)


if __name__ == "__main__":
    main()
