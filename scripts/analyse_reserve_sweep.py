#!/usr/bin/env python3
"""
Phase 2, Experiment 1 -- analyse the HBM reserve sweep.

Reads results/p2_reserve_raw/*.log, computes every derived quantity from the
raw batch and total figures, and writes results/p2_reserve_summary.csv.

Throughput = batch * iterations / total. Ratios are H3 over the baseline at the
SAME reserve fraction and the SAME collective. Power ratio is fixed at 2.28
(680 + 8*40 = 1000 W against 680 + 8*40 + 8*160 = 2280 W) and does not depend
on the reserve fraction.
"""
import os, re, csv, sys

RAW = "results/p2_reserve_raw"
OUT = "results/p2_reserve_summary.csv"
POWER_RATIO = 2.28

def read(name):
    p = os.path.join(RAW, name + ".log")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    if "ERROR" in txt or "Out of Memory" in txt:
        return {"failed": re.search(r"ERROR: (.*)", txt).group(1)
                          if "ERROR" in txt else "OOM"}
    b = re.findall(r"max_batch_size to (\d+)", txt)
    t = re.findall(r"^Total: ([\d.]+)", txt, re.M)
    if not b or not t:
        return None
    return {"batch": int(b[-1]), "total": float(t[-1])}

def tag(r):
    return "r" + ("%g" % r).replace("0.", "", 1)

R_1M  = [0.05, 0.06, 0.07, 0.08, 0.0821, 0.09, 0.10, 0.12]
R_10M = [0.0821, 0.10, 0.12]

rows = []

for r in R_1M:
    b = read(f"1M_hbmonly_{tag(r)}")
    h = read(f"1M_h3_{tag(r)}")
    if not b or not h or "failed" in b or "failed" in h:
        print(f"  1M r={r}: missing or failed", file=sys.stderr); continue
    tb = b["batch"] / b["total"]
    th = h["batch"] / h["total"]
    rows.append({
        "context": "1M", "collective": "-", "reserve": r,
        "hbm_batch": b["batch"], "h3_batch": h["batch"],
        "batch_ratio": round(h["batch"] / b["batch"], 4),
        "hbm_total_ms": round(b["total"] / 1e6, 2),
        "h3_total_ms":  round(h["total"] / 1e6, 2),
        "throughput_ratio": round(th / tb, 4),
        "tput_per_watt_ratio": round(th / tb / POWER_RATIO, 4),
    })

for r in R_10M:
    for coll in ("flat", "hier"):
        b = read(f"10M_hbmonly_{coll}_{tag(r)}")
        h = read(f"10M_h3_{coll}_{tag(r)}")
        if not b or not h or "failed" in b or "failed" in h:
            print(f"  10M {coll} r={r}: missing or failed", file=sys.stderr); continue
        tb = b["batch"] / b["total"]
        th = h["batch"] / h["total"]
        rows.append({
            "context": "10M", "collective": coll, "reserve": r,
            "hbm_batch": b["batch"], "h3_batch": h["batch"],
            "batch_ratio": round(h["batch"] / b["batch"], 4),
            "hbm_total_ms": round(b["total"] / 1e6, 2),
            "h3_total_ms":  round(h["total"] / 1e6, 2),
            "throughput_ratio": round(th / tb, 4),
            "tput_per_watt_ratio": round(th / tb / POWER_RATIO, 4),
        })

if not rows:
    print("No results found. Run scripts/run_reserve_sweep.sh first.", file=sys.stderr)
    sys.exit(1)

os.makedirs("results", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

hdr = f"{'ctx':>4}{'coll':>6}{'reserve':>9}{'HBM-only':>10}{'H3':>7}{'batch':>8}{'tput':>8}{'t/W':>8}"
print(hdr); print("-" * len(hdr))
base = {}
for row in rows:
    key = (row["context"], row["collective"])
    if abs(row["reserve"] - 0.0821) < 1e-9:
        base[key] = row
for row in rows:
    mark = "  <- Phase 1" if abs(row["reserve"] - 0.0821) < 1e-9 else ""
    print(f"{row['context']:>4}{row['collective']:>6}{row['reserve']*100:8.2f}%"
          f"{row['hbm_batch']:10d}{row['h3_batch']:7d}"
          f"{row['batch_ratio']:8.3f}{row['throughput_ratio']:8.3f}"
          f"{row['tput_per_watt_ratio']:8.3f}{mark}")

print("\nDeviation from the Phase 1 value at 8.21%:")
for row in rows:
    key = (row["context"], row["collective"])
    if key not in base or abs(row["reserve"] - 0.0821) < 1e-9:
        continue
    b0 = base[key]
    print(f"  {row['context']:>4} {row['collective']:>5} r={row['reserve']*100:5.2f}%  "
          f"batch {(row['batch_ratio']/b0['batch_ratio']-1)*100:+6.1f}%  "
          f"tput {(row['throughput_ratio']/b0['throughput_ratio']-1)*100:+6.1f}%  "
          f"t/W {(row['tput_per_watt_ratio']/b0['tput_per_watt_ratio']-1)*100:+6.1f}%")

print(f"\nWrote {OUT}")
