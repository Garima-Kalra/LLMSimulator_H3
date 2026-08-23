#!/usr/bin/env python3
"""
Compare HBM/HBF architectures across context lengths.

Reads results.log -- the CSV eval/test.cpp writes itself. Computes nothing
but ratios.

Each architecture is run at several context lengths. Growing the context
grows the shared KV cache (which lives in HBF) without changing HBM
capacity, so this sweep isolates HBF bandwidth pressure from capacity.

Usage:
    python3 scripts/compare_workload.py
    python3 scripts/compare_workload.py --results results.log
"""

import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results")

STYLE = {
    "cascaded":     ("Cascaded (H3)",   "#4C72B0", "D-"),
    "shared_base":  ("Shared-base",     "#DD8452", "s-"),
    "side_by_side": ("Side-by-side",    "#55A868", "o-"),
    "hbf_only":     ("HBF-only",        "#C44E52", "^-"),
}
ORDER = ["cascaded", "shared_base", "side_by_side", "hbf_only"]


def num(r, k, d=0.0):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


def fmt_ctx(c):
    if c >= 1_000_000:
        v = c / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    return f"{c/1000:.0f}K"


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("results.log has no data rows.")
    for col in ("architecture", "context_window"):
        if col not in rows[0]:
            raise SystemExit(f"results.log has no '{col}' column.")
    # Key includes context_window: without it, every context length for one
    # architecture collapses into a single row.
    best = {}
    for r in rows:
        try:
            key = (r["architecture"], round(num(r, "hbm_gb")),
                   round(num(r, "context_window")))
            ts = int(r["timestamp"])
        except (KeyError, ValueError):
            continue
        if key not in best or ts >= int(best[key]["timestamp"]):
            best[key] = r
    return list(best.values())


def organise(rows):
    fam = {}
    ctxs = set()
    for r in rows:
        a = r["architecture"]
        if a not in STYLE:
            continue
        c = round(num(r, "context_window"))
        fam.setdefault(a, {})[c] = r
        ctxs.add(c)
    return fam, sorted(ctxs)


def table(fam, ctxs, metric, title, fmt="{:.0f}"):
    names = [a for a in ORDER if a in fam]
    w0, w = 22, 12
    print(f"\n  {title}")
    print("  " + "-" * (w0 + w * len(ctxs) + 12))
    print("  " + "ARCHITECTURE".ljust(w0)
          + "".join(fmt_ctx(c).rjust(w) for c in ctxs)
          + "CHANGE".rjust(12))
    print("  " + "-" * (w0 + w * len(ctxs) + 12))
    for a in names:
        vals = [num(fam[a][c], metric) if c in fam[a] else None for c in ctxs]
        cells = "".join((fmt.format(v) if v is not None else "-").rjust(w)
                        for v in vals)
        first, last = vals[0], vals[-1]
        chg = (f"{(last/first - 1)*100:+.0f}%"
               if first and last else "-")
        print("  " + STYLE[a][0].ljust(w0) + cells + chg.rjust(12))
    print("  " + "-" * (w0 + w * len(ctxs) + 12))


def pooled_vs_partitioned(fam, ctxs):
    if "shared_base" not in fam or "side_by_side" not in fam:
        return []
    print("\n  POOLED vs PARTITIONED LINK (identical capacity and aggregate BW)")
    print(f"  {'context':>10}{'side-by-side':>16}{'shared-base':>14}{'delta':>10}")
    print("  " + "-" * 50)
    pts = []
    for c in ctxs:
        if c not in fam["shared_base"] or c not in fam["side_by_side"]:
            continue
        a = num(fam["side_by_side"][c], "throughput_tps")
        b = num(fam["shared_base"][c], "throughput_tps")
        if not a:
            continue
        d = (b / a - 1) * 100
        pts.append((c, d))
        print(f"  {fmt_ctx(c):>10}{a:>16.0f}{b:>14.0f}{d:>9.1f}%")
    print("\n  Both carry the same 8 TB/s aggregate; only its division differs.")
    return pts


def plots(fam, ctxs, cont):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; tables only.")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    names = [a for a in ORDER if a in fam]
    labels = [fmt_ctx(c) for c in ctxs]
    x = range(len(ctxs))

    panels = [("throughput_tps", "Throughput", "tokens/s"),
              ("throughput_per_power", "Efficiency", "tokens/s/W"),
              ("final_max_batch_size", "Batch size", "requests"),
              ("attn_memory_ms", "Attention memory time", "ms")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, ylab) in zip(axes.flat, panels):
        for a in names:
            lbl, col, mk = STYLE[a]
            ys = [num(fam[a][c], key) if c in fam[a] else None for c in ctxs]
            xs = [i for i, y in enumerate(ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if ys:
                ax.plot(xs, ys, mk, color=col, label=lbl, lw=2, ms=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("HBM/HBF architectures vs context length  -  8 shoreline sites",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, "workload_families.png")
    fig.savefig(p1, dpi=150)
    print(f"\n  Wrote {p1}")

    # Normalised view: how well does each architecture hold up as context grows?
    fig2, ax = plt.subplots(figsize=(8, 5))
    for a in names:
        lbl, col, mk = STYLE[a]
        ys = [num(fam[a][c], "throughput_tps") if c in fam[a] else None
              for c in ctxs]
        if not ys or not ys[0]:
            continue
        base = ys[0]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [y / base * 100 for y in ys if y is not None], mk,
                color=col, label=lbl, lw=2, ms=7)
    ax.axhline(100, ls="--", color="0.4", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Context length")
    ax.set_ylabel(f"Throughput, % of {fmt_ctx(ctxs[0])} value")
    ax.set_title("Robustness to context growth", fontweight="bold")
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig2.tight_layout()
    p2 = os.path.join(OUTDIR, "workload_robustness.png")
    fig2.savefig(p2, dpi=150)
    print(f"  Wrote {p2}")

    if cont:
        fig3, ax = plt.subplots(figsize=(8, 5))
        xs = [ctxs.index(c) for c, _ in cont]
        ys = [d for _, d in cont]
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.plot(xs, ys, "o-", color="#4C72B0", lw=2, ms=7)
        ax.fill_between(xs, 0, ys, alpha=.15, color="#4C72B0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Shared-base advantage (% throughput)")
        ax.set_title("Pooled vs partitioned link at equal capacity",
                     fontweight="bold")
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig3.tight_layout()
        p3 = os.path.join(OUTDIR, "workload_contention.png")
        fig3.savefig(p3, dpi=150)
        print(f"  Wrote {p3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results.log"))
    a = ap.parse_args()
    fam, ctxs = organise(load(a.results))
    if not ctxs:
        raise SystemExit("No recognised architectures found.")
    table(fam, ctxs, "throughput_tps", "THROUGHPUT (tokens/s)")
    table(fam, ctxs, "throughput_per_power", "EFFICIENCY (tokens/s/W)",
          "{:.4f}")
    table(fam, ctxs, "final_max_batch_size", "BATCH SIZE (requests)")
    table(fam, ctxs, "attn_memory_ms", "ATTENTION MEMORY TIME (ms)", "{:.1f}")
    plots(fam, ctxs, pooled_vs_partitioned(fam, ctxs))