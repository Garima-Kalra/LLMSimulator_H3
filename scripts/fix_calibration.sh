#!/usr/bin/env bash
# TIER 1 FIX #1 -- calibration consistency.
#
# THE PROBLEM
#   eval/test.cpp reads hbm_reserve_fraction with a default of 0.0:
#       config["system"]["hbm_reserve_fraction"].as<double>(0.0);
#   Only run14-run20 set it to 0.0821. Every arch_*, topo_* and sweep_*
#   config omits the key entirely and therefore silently runs at ZERO
#   reserve. The architecture study's batch sizes (1588 / 790 / 190)
#   are the zero-reserve numbers; the replication's (1457) is the
#   0.0821 number. Same simulator, two calibrations, no warning.
#
# THE FIX
#   Insert the key into every config that lacks it, so the value is
#   always explicit and never inherited from a default.
#
# Usage:  bash scripts/fix_calibration.sh [fraction]      (default 0.0821)

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FRAC="${1:-0.0821}"

echo "Setting hbm_reserve_fraction = $FRAC in every config that lacks it."
echo

added=0; already=0
for f in configs_h3/*.yaml config.yaml; do
  [ -f "$f" ] || continue
  if grep -q "^  hbm_reserve_fraction:" "$f"; then
    cur=$(grep "^  hbm_reserve_fraction:" "$f" | awk '{print $2}')
    printf '  %-34s already set (%s)\n' "$(basename "$f")" "$cur"
    already=$((already+1))
    continue
  fi
  grep -q "^  use_hbf:" "$f" || {
    printf '  %-34s SKIPPED (no use_hbf anchor)\n' "$(basename "$f")"; continue; }
  sed -i "/^  use_hbf:/a\\  hbm_reserve_fraction: $FRAC # framework/workspace/fragmentation headroom; calibrated to paper Fig.5" "$f"
  printf '  %-34s added\n' "$(basename "$f")"
  added=$((added+1))
done

echo
echo "$added added, $already already explicit."
echo
echo "Verify no config is left implicit:"
echo "  grep -L 'hbm_reserve_fraction' configs_h3/*.yaml"
