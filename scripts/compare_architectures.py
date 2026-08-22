#!/usr/bin/env python3
"""
Compare HBM/HBF architectures from the simulator's own results.log.

Side-by-side and shared-base are swept over the same HBM/HBF split, so at
every point they have identical capacity and identical aggregate bandwidth
(8 TB/s). Side-by-side PARTITIONS that bandwidth between the tiers;
shared-base POOLS it behind one link. The gap between the two curves is
therefore the value of partitioning vs pooling -- nothing else differs.

Cascaded and HBF-only are fixed reference points, drawn as single markers.

Reads only what eval/test.cpp wrote. Computes nothing but ratios.

Usage:
    python3 scripts/compare_architectures.py
"""

import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results")

STYLE = {
    "side_by_side": ("Side-by-side (partitioned BW)", "#55A868", "o-"),
    "shared_base":  ("Shared-base (pooled BW)",       "#DD8452", "s-"),
    "cascaded":     ("H3 cascaded",                   "#4C72B0", "D"),
    "hbf_only":     ("HBF-only",                      "#C44E52", "^"),
}


def num(r, k, d=0.0):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("results.log has no data rows.")
    if "architecture" not in rows[0]:
        raise SystemExit(
            "results.log has no 'architecture' column -- it predates the\n"
            "topology work. Archive it and re-run so a fresh header is written.")
    best = {}
    for r in rows:
        try:
            key = (r["architecture"], round(num(r, "hbm_gb")),
                   round(num(r, "hbf_gb")))
            ts = int(r["timestamp"])
        except (KeyError, ValueError):
            continue
        if key not in best or ts >= int(best[key]["timestamp"]):
            best[key] = r
    return list(best.values())


def split(rows):
    fam = {k: [] for k in STYLE}
    for r in rows:
        a = r["architecture"]
        if a in fam:
            fam[a].append(r)
    for k in fam:
        fam[k].sort(key=lambda r: num(r, "hbm_gb"))
    return fam


def table(fam):
    w = [30, 9, 9, 8, 8, 8, 9, 10]
    head = ["CONFIGURATION", "HBM GB", "HBF GB", "HBM BW", "HBF BW",
            "BATCH", "TPS", "tok/s/W"]
    line = "+" + "+".join("-" * x for x in w) + "+"
    print("\n" + line)
    print("|" + "|".join(h.center(x) for h, x in zip(head, w)) + "|")
    print(line)
    for key in ("cascaded", "hbf_only", "side_by_side", "shared_base"):
        for r in fam[key]:
            cells = [
                f"{STYLE[key][0].split(' (')[0]} "
                f"{num(r,'hbm_gb'):.0f}+{num(r,'hbf_gb'):.0f}",
                f"{num(r,'hbm_gb'):.0f}", f"{num(r,'hbf_gb'):.0f}",
                f"{num(r,'hbm_bw_tbs'):.1f}T", f"{num(r,'hbf_bw_tbs'):.1f}T",
                f"{num(r,'final_max_batch_size'):.0f}",
                f"{num(r,'throughput_tps'):.0f}",
                f"{num(r,'throughput_per_power'):.4f}",
            ]
            print("|" + "|".join(c.rjust(x - 1) + " "
                                 for c, x in zip(cells, w)) + "|")
        if fam[key]:
            print(line)


def contention(fam):
    """Pooled vs partitioned bandwidth at matched capacity."""
    sb = {round(num(r, "hbm_gb")): r for r in fam["side_by_side"]}
    sh = {round(num(r, "hbm_gb")): r for r in fam["shared_base"]}
    common = sorted(set(sb) & set(sh))
    if not common:
        return []
    print("\n  POOLED vs PARTITIONED BANDWIDTH (identical capacity at each row)")
    print(f"  {'HBM GB':>8}{'HBF GB':>9}{'side-by-side':>15}"
          f"{'shared-base':>14}{'delta':>9}")
    print("  " + "-" * 55)
    pts = []
    for c in common:
        a, b = num(sb[c], "throughput_tps"), num(sh[c], "throughput_tps")
        d = (b / a - 1) * 100 if a else 0.0
        pts.append((c, d))
        print(f"  {c:>8.0f}{num(sb[c],'hbf_gb'):>9.0f}{a:>15.0f}"
              f"{b:>14.0f}{d:>8.1f}%")
    print("\n  Positive delta = pooling the link beats partitioning it.")
    print("  Both rows carry the same 8 TB/s aggregate; only its division"
          " differs.\n")
    return pts


def plots(fam, cont):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; tables only.\n"
              "    pip3 install --user matplotlib")
        return
    os.makedirs(OUTDIR, exist_ok=True)

    panels = [("final_max_batch_size", "Batch size", "requests"),
              ("throughput_tps", "Throughput", "tokens/s"),
              ("throughput_per_power", "Efficiency", "tokens/s/W"),
              ("total_power_w", "Power", "watts")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    seen = set()
    for ax, (key, title, ylab) in zip(axes.flat, panels):
        for a in ("side_by_side", "shared_base"):
            rs = fam[a]
            if not rs:
                continue
            lbl, col, mk = STYLE[a]
            ax.plot([num(r, "hbm_gb") for r in rs], [num(r, key) for r in rs],
                    mk, color=col, label=lbl, lw=2, ms=6)
        for a in ("cascaded", "hbf_only"):
            for r in fam[a]:
                lbl, col, mk = STYLE[a]
                ax.plot(num(r, "hbm_gb"), num(r, key), mk, color=col,
                        ms=13, label=lbl, mec="white", mew=1.5)
        ax.set_xlabel("HBM capacity per device (GB)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    h, l = axes.flat[0].get_legend_handles_labels()
    uniq = [(x, y) for i, (x, y) in enumerate(zip(h, l)) if y not in l[:i]]
    if uniq:
        axes.flat[0].legend([x for x, _ in uniq], [y for _, y in uniq],
                            fontsize=8, loc="best")
    fig.suptitle("HBM/HBF architecture families  -  8 shoreline sites",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, "arch_families.png")
    fig.savefig(p1, dpi=150)
    print(f"  Wrote {p1}")

    if cont:
        fig2, ax = plt.subplots(figsize=(8, 5))
        xs, ys = zip(*cont)
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.plot(xs, ys, "o-", color="#4C72B0", lw=2, ms=7)
        ax.fill_between(xs, 0, ys, alpha=.15, color="#4C72B0")
        ax.set_xlabel("HBM capacity per device (GB)")
        ax.set_ylabel("Shared-base advantage (% throughput)")
        ax.set_title("Pooled vs partitioned bandwidth at equal capacity",
                     fontweight="bold")
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        p2 = os.path.join(OUTDIR, "arch_contention.png")
        fig2.savefig(p2, dpi=150)
        print(f"  Wrote {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results.log"))
    a = ap.parse_args()
    fam = split(load(a.results))
    table(fam)
    plots(fam, contention(fam))