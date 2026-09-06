#!/usr/bin/env bash
# TIER 1 FIX #2 -- iso-batch comparison.
#
# THE PROBLEM
#   Every design is currently measured at its OWN maximum batch. So
#   "cascaded is 47% faster than shared-base at 10M" mixes two effects:
#   cascaded has twice the HBM (bigger batch) AND different wiring. The
#   half-flash control isolates this for one pair; this generalises it.
#
# THE FIX
#   Re-run all four designs at a SET of fixed batch sizes. Because
#   Cluster only ever REDUCES max_batch_size to fit, setting a small
#   value pins the batch exactly -- the cap never fires.
#
#   Batch points, and why:
#     175  every design can reach it (HBF-only's ceiling at 0.0821 reserve)
#     400  mid-range
#     725  shared-base / side-by-side ceiling; HBF-only INFEASIBLE here,
#          which is itself the finding rather than a missing data point
#
#   Speed differences at a fixed batch are attributable to memory design
#   alone. The gap between the iso-batch and max-batch tables is the
#   capacity contribution.
#
# Usage:  bash scripts/gen_isobatch_configs.sh

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BATCHES="175 400 725"
DESIGNS="casc_8H8F shbase_4d4d side_4H4F hbfonly_1H7F"

made=0
for d in $DESIGNS; do
  src="configs_h3/arch_$d.yaml"
  [ -f "$src" ] || { echo "MISSING: $src (run gen_arch_configs.sh first)"; exit 1; }
  grep -q "^  hbm_reserve_fraction:" "$src" || {
    echo "ERROR: $src has no explicit hbm_reserve_fraction."
    echo "       Run fix_calibration.sh first, or iso-batch runs will not be"
    echo "       comparable with the calibrated replication runs."; exit 1; }

  for b in $BATCHES; do
    out="configs_h3/iso${b}_$d.yaml"
    sed "s/^  max_batch_size: .*/  max_batch_size: $b # ISO-BATCH: pinned, capacity effect removed/" \
        "$src" > "$out"
    printf '  %-34s batch %s\n' "$(basename "$out")" "$b"
    made=$((made+1))
  done
done

echo
echo "Generated $made iso-batch configs."
echo "Run:      for f in configs_h3/iso*.yaml; do (cd build && ./run \"../\$f\"); done"
echo "Expect:   HBF-only at batch 725 reports OOM -- that is the result, not a failure."
