#!/usr/bin/env bash
# HBF bandwidth sensitivity sweep.
# Runs ./build/run once per config. Computes nothing itself --
# the simulator appends its own CSV row to results.log each run.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/build" || exit 1        # binary writes ../results.log

OUT="$ROOT/results/bw_sweep"
mkdir -p "$OUT"

CONFIGS=(
  "run4_10M_hbmonly.yaml     hbm_only"
  "sweep_10M_h3_bw100.yaml   h3_bw100"
  "sweep_10M_h3_bw75.yaml    h3_bw75"
  "sweep_10M_h3_bw50.yaml    h3_bw50"
  "sweep_10M_h3_bw25.yaml    h3_bw25"
  "sweep_10M_h3_bw10.yaml    h3_bw10"
  "sweep_10M_h3_bw05.yaml    h3_bw05"
)

for entry in "${CONFIGS[@]}"; do
  set -- $entry
  cfg="$ROOT/configs_h3/$1"; label=$2
  [ -f "$cfg" ] || { echo "SKIP $label: $1 not found"; continue; }

  scale=$(grep -m1 hbf_bandwidth_scale "$cfg" | awk '{print $2}')
  echo ">>> $label   (hbf_bandwidth_scale=$scale)"

  before=$( [ -f "$ROOT/results.log" ] && wc -l < "$ROOT/results.log" || echo 0 )
  start=$(date +%s)
  ./run "$cfg" > "$OUT/$label.log" 2>&1
  rc=$?
  echo "    exit=$rc  wall=$(( $(date +%s) - start ))s"

  if [ $rc -ne 0 ]; then echo "    !! FAILED:"; tail -5 "$OUT/$label.log"; continue; fi

  after=$(wc -l < "$ROOT/results.log")
  if [ "$after" -gt "$before" ]; then
    tail -n $(( after - before )) "$ROOT/results.log" | tee "$OUT/$label.row" | sed 's/^/    row: /'
  else
    echo "    !! simulator appended no row to results.log"
  fi
  echo
done

echo "Done. results.log now has $(wc -l < "$ROOT/results.log") lines."
