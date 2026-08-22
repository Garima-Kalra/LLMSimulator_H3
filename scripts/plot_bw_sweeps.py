#!/usr/bin/env python3
"""
Plot the HBF bandwidth sensitivity sweep.

Reads results.log -- the CSV written by eval/test.cpp -- and produces
four presentation-ready figures in results/.

Simulator metrics are read directly from results.log. The script only
derives presentation metrics such as speedup relative to HBM-only.
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results.log")
OUTDIR = os.path.join(ROOT, "results")


def load(path):
    """Return (hbm_only_row, [h3_rows sorted by bandwidth]).

    The sweep may have been run more than once, so rows are de-duplicated on
    (use_hbf, hbf_bandwidth_scale) keeping the most recent by timestamp.
    """
    if not os.path.isfile(path):
        sys.exit(f"Not found: {path}\nRun the sweep first.")

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has a header but no data rows.")

    need = ["use_hbf", "hbf_bandwidth_scale", "throughput_tps",
            "attn_compute_ms", "attn_memory_ms", "final_max_batch_size",
            "total_power_w", "throughput_per_power"]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        sys.exit("results.log is missing columns: " + ", ".join(missing) +
                 "\nIt was probably written by an older build. Archive it and "
                 "re-run the sweep so a fresh header is written.")

    best = {}
    for r in rows:
        try:
            key = (int(r["use_hbf"]), float(r["hbf_bandwidth_scale"]))
            ts = int(r["timestamp"])
        except (ValueError, KeyError):
            continue
        if key not in best or ts >= int(best[key]["timestamp"]):
            best[key] = r

    hbm = next((v for k, v in best.items() if k[0] == 0), None)
    h3 = sorted((v for k, v in best.items() if k[0] == 1),
                key=lambda r: float(r["hbf_bandwidth_scale"]))
    if not h3:
        sys.exit("No H3 rows (use_hbf=1) found.")
    if hbm is None:
        print("WARNING: no HBM-only row (use_hbf=0); baseline lines omitted.")
    return hbm, h3


def f(row, col):
    return float(row[col])


def main():
    hbm, h3 = load(CSV_IN)
    os.makedirs(OUTDIR, exist_ok=True)

    pct = [f(r, "hbf_bandwidth_scale") * 100 for r in h3]
    tps = [f(r, "throughput_tps") for r in h3]
    comp = [f(r, "attn_compute_ms") for r in h3]
    mem = [f(r, "attn_memory_ms") for r in h3]
    batch = [f(r, "final_max_batch_size") for r in h3]
    tpp = [f(r, "throughput_per_power") for r in h3]

    # H3 speedup relative to HBM-only
    speedup = None
    if hbm:
        hbm_tps = f(hbm, "throughput_tps")
        speedup = [t / hbm_tps for t in tps]

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("                    H3 HBF BANDWIDTH SENSITIVITY")
    print("=" * 90)

    print(
        f"{'HBF BW':>8} "
        f"{'Batch':>8} "
        f"{'TPS':>10} "
        f"{'Speedup':>10} "
        f"{'Comp ms':>10} "
        f"{'Mem ms':>10} "
        f"{'Tok/s/W':>10}"
    )

    print("-" * 90)

    if hbm:
        print(
            f"{'HBM-only':>8} "
            f"{f(hbm,'final_max_batch_size'):>8.0f} "
            f"{f(hbm,'throughput_tps'):>10.1f} "
            f"{'1.000x':>10} "
            f"{f(hbm,'attn_compute_ms'):>10.1f} "
            f"{f(hbm,'attn_memory_ms'):>10.1f} "
            f"{f(hbm,'throughput_per_power'):>10.4f}"
        )

    for i, r in enumerate(h3):
        sp_text = f"{speedup[i]:.3f}x" if speedup else "N/A"

        print(
            f"{f(r,'hbf_bandwidth_scale') * 100:>7.0f}% "
            f"{f(r,'final_max_batch_size'):>8.0f} "
            f"{f(r,'throughput_tps'):>10.1f} "
            f"{sp_text:>10} "
            f"{f(r,'attn_compute_ms'):>10.1f} "
            f"{f(r,'attn_memory_ms'):>10.1f} "
            f"{f(r,'throughput_per_power'):>10.4f}"
        )

    print("=" * 90)

    # ------------------------------------------------------------------
    # Import matplotlib
    # ------------------------------------------------------------------

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed.")
        print("Install it with: pip3 install matplotlib")
        return

    saved = []

    # ------------------------------------------------------------------
    # Figure 1: Throughput vs HBF bandwidth
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(pct, tps, "o-", lw=2, label="H3")

    for x, y in zip(pct, tps):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    if hbm:
        ax.axhline(
            f(hbm, "throughput_tps"),
            ls="--",
            color="0.4",
            label="HBM-only baseline"
        )

    ax.set_xlabel("HBF bandwidth (% of nominal)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("H3 Throughput vs HBF Bandwidth")
    ax.set_ylim(0, max(tps) * 1.25)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    p = os.path.join(
        OUTDIR,
        "fig1_throughput_vs_hbf_bw.png"
    )
    fig.savefig(p, dpi=150)
    saved.append(p)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Attention compute vs memory
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 4))

    x = range(len(pct))

    ax.bar(
        x,
        comp,
        label="Attention compute"
    )

    ax.bar(
        x,
        mem,
        bottom=comp,
        label="Attention memory"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{p_:.0f}%" for p_ in pct]
    )

    ax.set_xlabel("HBF bandwidth")
    ax.set_ylabel("Attention time (ms)")
    ax.set_title("Attention Compute vs Memory Time")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()

    p = os.path.join(
        OUTDIR,
        "fig2_attn_compute_vs_memory.png"
    )
    fig.savefig(p, dpi=150)
    saved.append(p)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Speedup vs HBM-only
    # ------------------------------------------------------------------

    if hbm:

        fig, ax = plt.subplots(figsize=(6.5, 4))

        ax.plot(
            pct,
            speedup,
            "o-",
            lw=2
        )

        # HBM-only = 1x
        ax.axhline(
            1.0,
            ls="--",
            color="0.4",
            label="HBM-only (1x)"
        )

        for x, y in zip(pct, speedup):
            ax.annotate(
                f"{y:.2f}x",
                (x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

        ax.set_xlabel("HBF bandwidth (% of nominal)")
        ax.set_ylabel("Speedup vs HBM-only")
        ax.set_title("H3 Speedup vs HBM-only")
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        p = os.path.join(
            OUTDIR,
            "fig3_speedup_vs_hbm.png"
        )
        fig.savefig(p, dpi=150)
        saved.append(p)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Energy efficiency
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 4))

    labels = [
        f"H3 {p_:.0f}%"
        for p_ in pct
    ]

    vals = list(tpp)

    if hbm:
        labels.insert(0, "HBM-only")
        vals.insert(
            0,
            f(hbm, "throughput_per_power")
        )

    ax.bar(
        range(len(vals)),
        vals
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_ylabel(
        "Throughput per power (tokens/s/W)"
    )

    ax.set_title(
        "Energy Efficiency: H3 vs HBM-only"
    )

    ax.grid(
        alpha=0.3,
        axis="y"
    )

    fig.tight_layout()

    p = os.path.join(
        OUTDIR,
        "fig4_throughput_per_power.png"
    )

    fig.savefig(p, dpi=150)
    saved.append(p)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Batch size vs HBF bandwidth
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(
        pct,
        batch,
        "o-",
        lw=2
    )

    for x, y in zip(pct, batch):
        ax.annotate(
            f"{y:.0f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    ax.set_xlabel(
        "HBF bandwidth (% of nominal)"
    )

    ax.set_ylabel(
        "Final maximum batch size"
    )

    ax.set_title(
        "Maximum Batch Size vs HBF Bandwidth"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    p = os.path.join(
        OUTDIR,
        "fig5_batch_vs_hbf_bw.png"
    )

    fig.savefig(p, dpi=150)
    saved.append(p)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print("\nWrote:")

    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()