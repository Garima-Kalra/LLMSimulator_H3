#!/usr/bin/env bash
# TIER 1 FIX #3 -- bounding the amortisation assumption.
#
# THE PROBLEM
#   cag_amortize_shared_compute charges shared-span attention once per
#   BATCH rather than once per SEQUENCE. It is not physically exact: real
#   shared-prefix attention does the arithmetic for every sequence. What
#   it stands in for is far better hardware utilisation -- one large
#   well-shaped operation instead of thousands of tiny ones.
#
#   With the switch OFF, compute dominates memory by 28x and NO memory
#   design matters. So every conclusion in the architecture study rests
#   on this one idealisation. That is the study's weakest point.
#
# THE FIX
#   Stop asserting the idealisation and BOUND it instead.
#   attn_compute_efficiency scales attention's achievable peak FLOPS
#   (attention_gen_impl.cpp:54). Turn amortisation OFF and sweep it:
#   the arithmetic then scales with batch size honestly, and efficiency
#   carries the utilisation effect.
#
#   The deliverable is a threshold: "H3's advantage holds provided
#   shared-attention achieves better than X% of peak." That is a
#   falsifiable engineering claim, not a modelling assumption.
#
# Usage:  bash scripts/gen_efficiency_sweep.sh

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_H3="configs_h3/run18_10M_h3_fullbw_calh.yaml"
BASE_HBM="configs_h3/run17_10M_hbmonly_calh.yaml"
for f in "$BASE_H3" "$BASE_HBM"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# 1.0 = today's implicit assumption. Below that, attention runs at a
# fraction of peak, which is what many small operations actually achieve.
EFFS="1.0 0.5 0.25 0.10 0.05 0.02 0.01"

made=0
for e in $EFFS; do
  tag=$(echo "$e" | tr -d '.')
  for base in "$BASE_H3" "$BASE_HBM"; do
    stem=$(basename "$base" .yaml)
    out="configs_h3/eff${tag}_${stem}.yaml"
    sed -e "s/^  cag_amortize_shared_compute: .*/  cag_amortize_shared_compute: false # SWEEP: per-sequence compute, honest scaling/" \
        "$base" > "$out"
    # insert the efficiency knob under system:
    sed -i "/^  use_hbf:/a\\  attn_compute_efficiency: $e # SWEEP: achieved fraction of attention peak FLOPS" "$out"
    grep -q "attn_compute_efficiency: $e" "$out" || { echo "ERROR: insert failed in $out"; exit 1; }
    printf '  %-46s eff %s\n' "$(basename "$out")" "$e"
    made=$((made+1))
  done
done

echo
echo "Generated $made configs (H3 and HBM-only at each efficiency)."
echo
echo "Run:  for f in configs_h3/eff*.yaml; do (cd build && ./run \"../\$f\"); done"
echo
echo "Then for each efficiency compute  (batch_H3/time_H3) / (batch_HBM/time_HBM)."
echo "The efficiency at which that ratio crosses 1.0 is the threshold to report."
echo "If it crosses ABOVE any plausible achieved efficiency, the amortised model"
echo "is doing the work and the study must say so plainly."
