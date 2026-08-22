#!/usr/bin/env bash
# Generate architecture configs for the HBM/HBF design-space study.
#
#   cascaded 8H+8F   H3 as published            (fixed reference point)
#   hbf_only 1H+7F   Son et al.                 (fixed reference point)
#   side_by_side     7 points, site split swept
#   shared_base      7 points, die split swept
#
# At each sweep point the two families have IDENTICAL capacity and IDENTICAL
# aggregate bandwidth (8 TB/s). Side-by-side partitions that bandwidth between
# the tiers; shared-base pools it. That is the only difference, which is what
# makes the pair a controlled comparison.

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${1:-configs_h3/sweep_10M_h3_bw100.yaml}"
[ -f "$BASE" ] || { echo "ERROR: base config not found: $BASE"; exit 1; }
grep -q "^  use_hbf:" "$BASE" || {
  echo "ERROR: '$BASE' has no '  use_hbf:' line to anchor the insert."; exit 1; }

mk () {  # name arch hbm_sites hbf_sites hbm_dies hbf_dies
  f="configs_h3/arch_$1.yaml"
  cp "$BASE" "$f"
  sed -i "/^  use_hbf:/a\\
\\  architecture: $2\\
\\  hbm_sites: $3\\
\\  hbf_sites: $4\\
\\  hbm_dies_per_site: $5\\
\\  hbf_dies_per_site: $6\\
\\  shoreline_slots: 8" "$f"
  grep -q "architecture: $2" "$f" || { echo "ERROR: insert failed for $f"; exit 1; }
  printf '  %-28s %-13s sites %dH+%dF  dies %d/%d\n' "$(basename "$f")" "$2" "$3" "$4" "$5" "$6"
}

echo "Base config: $BASE"
echo
echo "Fixed reference points:"
mk casc_8H8F    cascaded     8 0 8 8
mk hbfonly_1H7F hbf_only     1 7 8 8

echo
echo "Side-by-side (site split swept):"
for h in 7 6 5 4 3 2 1; do
  mk "side_${h}H$((8-h))F" side_by_side "$h" "$((8-h))" 8 8
done

echo
echo "Shared-base (die split swept):"
for h in 7 6 5 4 3 2 1; do
  mk "shbase_${h}d$((8-h))d" shared_base 8 0 "$h" "$((8-h))"
done

echo
echo "Generated $(ls configs_h3/arch_*.yaml | wc -l) configs."
echo "Sanity check one:  grep -A6 '^  use_hbf:' configs_h3/arch_shbase_4d4d.yaml"