#!/usr/bin/env bash
# Run every arch_*.yaml through the simulator.
# Runs from build/ so the binary's "../results.log" lands in the project root.
# Computes nothing: each run appends its own CSV row.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/arch"
mkdir -p "$OUT"
cd "$ROOT/build" || { echo "ERROR: no build/ directory. Build first."; exit 1; }
[ -x ./run ] || { echo "ERROR: ./run not found. Build first."; exit 1; }

cfgs=$(ls "$ROOT"/configs_h3/arch_*.yaml 2>/dev/null)
[ -n "$cfgs" ] || { echo "ERROR: no arch_*.yaml. Run gen_arch_configs.sh."; exit 1; }

# The results.log header is written once, so a stale header from an older
# build would silently misalign every new row.
if [ -f "$ROOT/results.log" ] && ! head -1 "$ROOT/results.log" | grep -q architecture; then
  ts=$(date +%Y%m%d_%H%M%S)
  mv "$ROOT/results.log" "$ROOT/results_pre_arch_$ts.log"
  echo "Archived old results.log (no 'architecture' column) -> results_pre_arch_$ts.log"
fi

total=$(echo "$cfgs" | wc -l); i=0; ok=0; failed=""
echo "Running $total configurations."
echo

for f in $cfgs; do
  i=$((i+1))
  n=$(basename "$f" .yaml)
  printf '[%2d/%2d] %-28s ' "$i" "$total" "$n"
  before=$( [ -f "$ROOT/results.log" ] && wc -l < "$ROOT/results.log" || echo 0 )
  start=$(date +%s)
  ./run "$f" > "$OUT/$n.log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))

  if [ $rc -ne 0 ]; then
    # A capacity assert is a legitimate result: that machine cannot serve
    # this workload. Record it rather than treating it as a script failure.
    reason=$(grep -m1 -i "error\|assert\|fail" "$OUT/$n.log" | cut -c1-60)
    printf 'INFEASIBLE (%ds)  %s\n' "$dur" "$reason"
    failed="$failed $n"
    continue
  fi

  after=$(wc -l < "$ROOT/results.log")
  if [ "$after" -gt "$before" ]; then
    tail -n $(( after - before )) "$ROOT/results.log" > "$OUT/$n.row"
    ok=$((ok+1))
    printf 'ok (%ds)\n' "$dur"
  else
    printf 'ran but appended no row (%ds)\n' "$dur"
  fi
done

echo
echo "$ok/$total produced results."
[ -n "$failed" ] && echo "Infeasible:$failed"
echo "Next:  python3 scripts/compare_architectures.py"