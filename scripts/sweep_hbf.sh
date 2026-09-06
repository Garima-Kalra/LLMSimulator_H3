#!/usr/bin/env bash
# H3 Action Item 2 -- HBF write onset sweep.
#
# Two axes, because they answer different halves of the question:
#
#   PHASE 1 (batch)    private-KV footprint vs batch size, HBF budget fixed.
#                      Establishes that footprint scales linearly and lets us
#                      predict the crossover batch. Writes stay at zero here --
#                      that IS the result: H3 never spills at realistic batch.
#
#   PHASE 2 (budget)   hbm_reserve_fraction swept at fixed batch, shrinking the
#                      HBM available to private KV until it binds. This crosses
#                      the threshold that phase 1 cannot reach, and measures
#                      actual spill volume, write amplification and HBF time.
#
# Phase 2 is the equivalent of raising batch past the crossover, but costs
# seconds instead of hours: the per-tensor LRU in TransientKvManager is O(n^2)
# in registered tensors, so batch >512 (>128k tensors) does not terminate.
#
# Usage:   ./sweep_hbf.sh            from the build/ directory
# Output:  sweep_results.csv

set -u
CONFIG="../config.yaml"
RUN="./run"
OUT="sweep_results.csv"
TMP="/tmp/h3_sweep_cfg.yaml"

if [[ ! -x "$RUN" ]]; then
  echo "error: $RUN not found. Run this from the build/ directory." >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "error: $CONFIG not found." >&2
  exit 1
fi

echo "phase,batch,reserve_fraction,budget_gib,peak_gib,static_weight_gib,static_shared_gib,wr_logical_gib,wr_physical_gib,waf,hbf_rd_gib,mig_out,mig_in,wr_time_s" > "$OUT"

run_one() {
  local phase="$1" batch="$2" reserve="$3"

  sed -e "s/^\( *max_batch_size:\).*/\1 ${batch}/" \
      -e "s/^\( *hbm_reserve_fraction:\).*/\1 ${reserve}/" \
      -e "s/^\( *use_hbf:\).*/\1 on/" \
      -e "s/^\( *print_log:\).*/\1 false/" \
      "$CONFIG" > "$TMP"

  # hbm_reserve_fraction may be absent from the config entirely; append it.
  if ! grep -q "hbm_reserve_fraction" "$TMP"; then
    sed -i "0,/^system:/s//system:\n  hbm_reserve_fraction: ${reserve}/" "$TMP"
  fi

  printf "  batch=%-6s reserve=%-8s ... " "$batch" "$reserve"
  local log="/tmp/h3_sweep_run.log"
  if ! timeout 3600 "$RUN" "$TMP" > "$log" 2>&1; then
    printf "FAILED/TIMEOUT\n"
    echo "${phase},${batch},${reserve},,,,,,,,,,," >> "$OUT"
    return
  fi

  # Pull the dump block. grep -m1 so only device0's block is read.
  local budget peak sw ss wl wp waf rd mo mi wt
  budget=$(grep -m1 "HBM KV budget"     "$log" | awk '{print $5}')
  peak=$(  grep -m1 "HBM KV peak"       "$log" | awk '{print $7}')
  sw=$(    grep -m1 "HBF static  weights"  "$log" | awk '{print $5}')
  ss=$(    grep -m1 "HBF static  sharedKV" "$log" | awk '{print $5}')
  wl=$(    grep -m1 "HBF writes  logical"  "$log" | awk '{print $5}')
  wp=$(    grep -m1 "HBF writes  physical" "$log" | awk '{print $5}')
  waf=$(   grep -m1 "HBF writes  physical" "$log" | sed 's/.*WAF \([0-9.]*\).*/\1/')
  rd=$(    grep -m1 "HBF reads"            "$log" | awk '{print $4}')
  mo=$(    grep -m1 "migrations out / in"  "$log" | awk '{print $6}')
  mi=$(    grep -m1 "migrations out / in"  "$log" | awk '{print $8}')
  wt=$(    grep -m1 "HBF time  wr / rd"    "$log" | awk '{print $7}')

  echo "${phase},${batch},${reserve},${budget},${peak},${sw},${ss},${wl},${wp},${waf},${rd},${mo},${mi},${wt}" >> "$OUT"
  printf "peak=%-8s writes=%-8s mig=%s\n" "${peak:-?}" "${wp:-?}" "${mo:-?}"
}

echo "=== PHASE 1: batch sweep (budget fixed, reserve=0) ==="
echo "    expect: peak grows linearly, writes stay 0"
for b in 4 8 16 32 64 128 256; do
  run_one batch "$b" 0.0
done

echo
echo "=== PHASE 2: budget sweep (batch fixed at 64) ==="
echo "    expect: writes switch on once budget < peak"
for r in 0.0 0.90 0.99 0.995 0.997 0.998 0.999 0.9995; do
  run_one budget 64 "$r"
done

echo
echo "done -> $OUT"
column -s, -t < "$OUT"