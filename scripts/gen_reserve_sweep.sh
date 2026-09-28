#!/usr/bin/env bash
# Phase 2, Experiment 1 -- HBM reserve fraction sensitivity.
#
# NO CODE CHANGE. hbm_reserve_fraction is already parsed (test.cpp:105),
# whitelisted (validate_config.h:51) and written to results.log (test.cpp:715).
# experiment_mode: h3_paper only WARNS when it is absent; it does not forbid
# setting it. So the sweep runs under the frozen paper mode.
#
# Phase 1 configs are not touched. New files are written under a distinct
# prefix so the baseline remains reproducible.
#
# Usage: bash scripts/gen_reserve_sweep.sh

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT=configs_h3/p2_reserve
mkdir -p "$OUT"

R_1M="0.05 0.06 0.07 0.08 0.0821 0.09 0.10 0.12"
R_10M="0.0821 0.10 0.12"

gen () {   # $1 src config   $2 reserve   $3 tag
  local tag; tag=$(echo "$2" | sed 's/0\.//;s/^/r/')
  local out="$OUT/${3}_${tag}.yaml"
  sed -e "s/^  hbm_reserve_fraction: .*/  hbm_reserve_fraction: $2/" "$1" > "$out"
  grep -q "^  hbm_reserve_fraction: $2" "$out" || { echo "FAILED: $out"; exit 1; }
  # Every other parameter is inherited verbatim from the Phase 1 config.
  echo "  $out"
}

echo "1M (8 GPUs): baseline + H3 at each reserve fraction"
for r in $R_1M; do
  gen configs_h3/run14_1M_hbmonly_cal.yaml    "$r" 1M_hbmonly
  gen configs_h3/run15_1M_h3_fullbw_cal.yaml  "$r" 1M_h3
done

echo
echo "10M (32 GPUs): both collectives, at 8.21 / 10 / 12 percent"
for r in $R_10M; do
  gen configs_h3/run17_10M_hbmonly_cal.yaml   "$r" 10M_hbmonly_flat
  gen configs_h3/run17_10M_hbmonly_calh.yaml  "$r" 10M_hbmonly_hier
  gen configs_h3/run18_10M_h3_fullbw_cal.yaml "$r" 10M_h3_flat
  gen configs_h3/run18_10M_h3_fullbw_calh.yaml "$r" 10M_h3_hier
done

echo
echo "$(ls "$OUT" | wc -l) configs in $OUT"
echo "Sanity: only hbm_reserve_fraction differs from the Phase 1 source."
diff <(sed '/hbm_reserve_fraction/d' configs_h3/run15_1M_h3_fullbw_cal.yaml) \
     <(sed '/hbm_reserve_fraction/d' "$OUT/1M_h3_r12.yaml") \
  && echo "  confirmed: no other key differs"
