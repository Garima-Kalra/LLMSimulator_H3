#!/usr/bin/env bash
# Run the reserve sweep and collect raw logs.
# Usage: bash scripts/run_reserve_sweep.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
RAW=results/p2_reserve_raw
mkdir -p "$RAW"

cp results.log "$RAW/results_before_sweep.log" 2>/dev/null || true

cd build
n=0; total=$(ls ../configs_h3/p2_reserve/*.yaml | wc -l)
for f in ../configs_h3/p2_reserve/*.yaml; do
  name=$(basename "$f" .yaml); n=$((n+1))
  printf '[%2d/%2d] %-28s ' "$n" "$total" "$name"
  ./run "$f" > "../$RAW/$name.log" 2>&1 || true
  if grep -q "ERROR\|Out of Memory" "../$RAW/$name.log"; then
    echo "FAILED: $(grep -oP 'ERROR: \K.*' "../$RAW/$name.log" | head -1)"
  else
    b=$(grep -oP 'max_batch_size to \K[0-9]+' "../$RAW/$name.log" | tail -1)
    t=$(grep -oP '^Total: \K[0-9.]+'          "../$RAW/$name.log" | tail -1)
    echo "batch=${b:-none} total=${t:-none}"
  fi
done
cd ..
echo
echo "Raw logs: $RAW/"
echo "Next: python3 scripts/analyse_reserve_sweep.py"
