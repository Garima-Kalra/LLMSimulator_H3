#!/usr/bin/env python3
"""Compare LLMSimulator H3 results.log against the H3 paper's reported figures.

Usage:
    python3 extract_results.py [results.log]

Reads the CSV rows appended by eval/test.cpp (one row per run: config +
final batch size + throughput + power), pairs up matching HBM-only / H3
scenarios by (model, context_window, hbf_bandwidth_scale), computes ratios,
and prints them next to the paper's headline numbers from:

  Ha, Kim & Kim, "H3: Hybrid Architecture Using High Bandwidth Memory and
  High Bandwidth Flash for Cost-Efficient LLM Inference", IEEE CAL 2026.
  - Fig. 5: max batch size, HBM-only vs H3, at 1M and 10M context (8 / 32 GPUs)
  - Fig. 6: throughput and throughput-per-power, incl. halved HBF bandwidth

Paper reference numbers are for their own (closed) simulator/assumptions;
this script's job is to show the simulated ratio next to them, not to force
a match. See the "notes" column for known reasons the simulated ratio can
diverge (e.g. compute-bound saturation at large batch sizes).
"""
import csv
import sys
from collections import defaultdict

PAPER = {
    # context_window -> {metric: paper's reported H3/HBM-only ratio}
    1_000_000: {
        "batch_size_ratio": 2.6,
        "throughput_ratio": 1.25,
        # Paper states no explicit 1M throughput-per-power number in text --
        # the 2.69x/2.09x figures below are 10M-specific (confirmed by the
        # 6.14/2.28~=2.69 arithmetic check, where 2.28 is this simulator's
        # own H3/HBM-only per-device power ratio). Do not compare 1M against
        # them -- see ASSUMPTIONS.md.
        "throughput_per_power_ratio": None,
        "throughput_per_power_ratio_half_bw": None,
    },
    10_000_000: {
        "batch_size_ratio": 18.8,
        "throughput_ratio": 6.14,
        "throughput_per_power_ratio": 2.69,  # paper's headline number (10M, full HBF bandwidth)
        "throughput_per_power_ratio_half_bw": 2.09,  # paper's number (10M, halved HBF bandwidth)
    },
}

FIELDS = [
    "timestamp", "model", "gpu_gen", "use_hbf", "hbf_bandwidth_scale",
    "num_node", "num_device", "ne_tp_dg", "context_parallel_degree",
    "fp8_compute_doubling", "cag_amortize_shared_compute",
    "hbm_reserve_fraction", "allreduce_hierarchical",
    "context_window", "input_len",
    "output_len", "requested_max_batch_size", "final_max_batch_size",
    "throughput_tps", "per_device_power_w", "total_power_w",
    "throughput_per_power",
]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["use_hbf"] = bool(int(row["use_hbf"]))
            row["fp8_compute_doubling"] = bool(int(row.get("fp8_compute_doubling", 1)))
            row["cag_amortize_shared_compute"] = bool(
                int(row.get("cag_amortize_shared_compute") or 0))
            row["hbm_reserve_fraction"] = float(
                row.get("hbm_reserve_fraction") or 0.0)
            row["allreduce_hierarchical"] = bool(
                int(row.get("allreduce_hierarchical") or 0))
            for key in ("hbf_bandwidth_scale", "throughput_tps",
                        "per_device_power_w", "total_power_w",
                        "throughput_per_power"):
                row[key] = float(row[key])
            for key in ("num_node", "num_device", "ne_tp_dg",
                        "context_window", "input_len", "output_len",
                        "requested_max_batch_size", "final_max_batch_size",
                        "timestamp"):
                row[key] = int(row[key])
            row["context_parallel_degree"] = int(row.get("context_parallel_degree", 1))
            row["total_device"] = row["num_node"] * row["num_device"]
            rows.append(row)
    return rows


def scenario_key(row):
    """Groups runs that should be compared against each other: same
    model/context/hardware setup, differing only in use_hbf. HBF-bandwidth
    sensitivity variants (scale != 1.0) are kept as their own H3 scenario,
    always compared against the full-bandwidth HBM-only baseline. Uses
    total device count (num_node * num_device), not num_device alone --
    the 10M scenario spans multiple nodes (e.g. num_node=4, num_device=8 =
    32 total), and num_device alone would collide with an unrelated
    single-node, 8-total-device scenario. cag_amortize_shared_compute is part
    of the key too: it changes the attention cost model for BOTH the HBM-only
    baseline and H3, so the two modes form separate, self-consistent
    comparisons and must never share a baseline."""
    return (row["model"], row["context_window"], row["total_device"],
            row["cag_amortize_shared_compute"], row["hbm_reserve_fraction"],
            row["allreduce_hierarchical"])


def pick_latest(rows):
    """If a scenario was rerun, keep only the most recent row per exact config.
    fp8_compute_doubling is part of the key so the FP8-doubling-off sensitivity
    run doesn't silently overwrite the primary (doubling-on) comparison row."""
    latest = {}
    for row in rows:
        key = (scenario_key(row), row["use_hbf"], row["hbf_bandwidth_scale"],
               row["fp8_compute_doubling"])
        if key not in latest or row["timestamp"] > latest[key]["timestamp"]:
            latest[key] = row
    return list(latest.values())


def fmt_ratio(x):
    return f"{x:.3g}x" if x is not None else "n/a"


def fmt_num(x):
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.3g}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.log"
    try:
        rows = load_rows(path)
    except FileNotFoundError:
        print(f"No results.log found at '{path}'. Run the simulator first "
              f"(each run appends a row).")
        sys.exit(1)

    if not rows:
        print(f"'{path}' has no data rows yet.")
        sys.exit(1)

    rows = pick_latest(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[scenario_key(row)].append(row)

    print(f"Loaded {len(rows)} run(s) from {path}\n")

    for key in sorted(groups, key=lambda k: (k[3], k[4], k[5], k[0], k[1], k[2])):
        (model, context_window, total_device, amortized, reserve,
         hier_ar) = key
        scenario_rows = groups[key]
        baseline = next((r for r in scenario_rows if not r["use_hbf"]), None)
        h3_variants = sorted(
            (r for r in scenario_rows if r["use_hbf"]),
            key=lambda r: (r["hbf_bandwidth_scale"], r["fp8_compute_doubling"]),
            reverse=True,
        )

        mode = ("shared-attention compute AMORTIZED per batch (paper-aligned)"
                if amortized else
                "shared-attention compute per-sequence (codebase original)")
        if reserve:
            mode += f"; HBM reserve {reserve:.2%} (CALIBRATED)"
        if total_device > 8:
            mode += ("; hierarchical all-reduce" if hier_ar
                     else "; flat-ring all-reduce over scale-out fabric")
        print("=" * 78)
        print(f"Scenario: {model}, context_window={context_window:,}, "
              f"total_device={total_device}")
        print(f"  cost model: {mode}")
        print("=" * 78)

        if baseline is None:
            print("  HBM-only: NO DATA (either not run, or the simulator aborted --")
            print("            it exits without writing a row when weights + shared")
            print("            KV cache alone exceed HBM capacity on this many GPUs)")
        else:
            print(f"  HBM-only : batch={baseline['final_max_batch_size']:>6}  "
                  f"throughput={fmt_num(baseline['throughput_tps']):>10} tok/s  "
                  f"power={baseline['total_power_w']:>7.0f} W  "
                  f"tput/power={baseline['throughput_per_power']:.4g}")

        if not h3_variants:
            print("  H3: NOT RUN")

        paper_ref = PAPER.get(context_window)

        for h3 in h3_variants:
            scale_tag = ("full HBF bandwidth" if h3["hbf_bandwidth_scale"] == 1.0
                         else f"{h3['hbf_bandwidth_scale']:g}x HBF bandwidth")
            if not h3["fp8_compute_doubling"]:
                scale_tag += ", FP8-doubling OFF (sensitivity check)"
            print(f"  H3 ({scale_tag}): "
                  f"batch={h3['final_max_batch_size']:>6}  "
                  f"throughput={fmt_num(h3['throughput_tps']):>10} tok/s  "
                  f"power={h3['total_power_w']:>7.0f} W  "
                  f"tput/power={h3['throughput_per_power']:.4g}")

            if baseline is None:
                print(f"    -> ratio vs HBM-only: n/a (no baseline data on "
                      f"{total_device} GPUs -- if it aborted rather than just not "
                      f"being run yet, that itself matches the paper's point that "
                      f"HBM-only needs far more GPUs at this context size)")
                continue

            batch_ratio = h3["final_max_batch_size"] / baseline["final_max_batch_size"]
            tput_ratio = (h3["throughput_tps"] / baseline["throughput_tps"]
                         if baseline["throughput_tps"] > 0 else None)
            power_eff_ratio = (h3["throughput_per_power"] / baseline["throughput_per_power"]
                               if baseline["throughput_per_power"] > 0 else None)

            print(f"    -> ratio vs HBM-only: "
                  f"batch {fmt_ratio(batch_ratio)}, "
                  f"throughput {fmt_ratio(tput_ratio)}, "
                  f"throughput/power {fmt_ratio(power_eff_ratio)}")

            if h3["hbf_bandwidth_scale"] == 1.0 and paper_ref is not None:
                print(f"    -> paper reports: "
                      f"batch {fmt_ratio(paper_ref['batch_size_ratio'])}, "
                      f"throughput {fmt_ratio(paper_ref['throughput_ratio'])}, "
                      f"throughput/power {fmt_ratio(paper_ref['throughput_per_power_ratio'])}")
            elif h3["hbf_bandwidth_scale"] == 0.5 and paper_ref is not None:
                print(f"    -> paper reports (half HBF bandwidth): "
                      f"throughput/power "
                      f"{fmt_ratio(paper_ref['throughput_per_power_ratio_half_bw'])}")

        print()

    print("Notes:")
    print("  - Paper ratios are read off Figs. 5/6 of the H3 paper; the exact")
    print("    simulator/assumptions behind those numbers are not published, so")
    print("    exact matches aren't expected -- direction and rough magnitude are")
    print("    the meaningful comparison.")
    print("  - If throughput/power ratio is well below the paper's despite a real")
    print("    batch-size gain, check whether the dominant GEMM/attention op is")
    print("    compute-bound at that batch size (Op/B in the simulator's verbose")
    print("    per-layer log) -- HBM vs HBF placement cannot help once compute-bound.")


if __name__ == "__main__":
    main()
