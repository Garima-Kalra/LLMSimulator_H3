#!/usr/bin/env bash
# Multi-model sweep across the cascaded mixed-die designs.
#
# WHAT AND WHY
#   Section 8.6 established that flash capacity is unused for eight of nine
#   models, so chained-stack dies are better spent on HBM. This tests that
#   across every model the simulator supports, at 10M context.
#
#   Designs (8 sites, first stack 8 HBM dies, chained stack mixed):
#     hbmonly  reference, no flash at all
#     8H8F     H3's published configuration
#     12H4F    mixed, balanced
#     14H2F    mixed, efficiency optimum for Llama-3.1-405B
#
#   deepseekV3 needs 1311 GB/device: 12H4F (1536 GB) fits, 14H2F (768 GB)
#   does NOT and is expected to report out-of-memory. That is the result --
#   it is the one model that justifies H3's flash depth.
#
# TWO MODELS NEED CODE FIRST -- see PREFLIGHT below.
#
# Usage:  bash scripts/gen_model_sweep.sh          # generate
#         bash scripts/gen_model_sweep.sh --run    # generate and run

set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS="llama3_405B mixtral grok1 llama7bMoE openMoE llama4_scout llama4_maverick deepseekV3 glam"
BASE="configs_h3/arch_casc_8H8F.yaml"
[ -f "$BASE" ] || { echo "MISSING $BASE"; exit 1; }

# ---- PREFLIGHT -----------------------------------------------------------
echo "PREFLIGHT"
if ! grep -q '"glam"' eval/test.cpp; then
  echo "  [BLOCK] glam is not in the model dispatch chain in eval/test.cpp."
  echo "          Add, next to the other else-if branches:"
  echo "            } else if (!model_name.compare(\"glam\")) {"
  echo "              model_config = glam;"
  echo "          Skipping glam for now."
  MODELS=$(echo "$MODELS" | sed 's/ glam//')
fi
echo "  [WARN]  deepseekV3 uses MLA (qk_rope_head_dim=64). The shared-cache"
echo "          path is wired only into SelfAttentionGen, not"
echo "          MultiLatentAttentionGen, so it will run WITHOUT ERROR but"
echo "          treat the whole 10M context as private KV. Its numbers are"
echo "          NOT comparable until that path is wired. Included so the"
echo "          capacity result is on record; exclude it from timing claims."
echo

gen () {  # $1 name  $2 model  $3 hbm_dies  $4 hbf_dies  $5 use_hbf
  out="configs_h3/mm_${2}_${1}.yaml"
  sed -e "s/^  model_name: .*/  model_name: $2/" \
      -e "s/^  hbm_dies_per_site: .*/  hbm_dies_per_site: $3/" \
      -e "s/^  hbf_dies_per_site: .*/  hbf_dies_per_site: $4/" \
      -e "s/^  use_hbf: .*/  use_hbf: $5/" \
      "$BASE" > "$out"
  # every key explicit: the sweep configs have defaulted silently before
  grep -q "^  hbm_reserve_fraction:" "$out" || \
    sed -i "/^  use_hbf:/a\\  hbm_reserve_fraction: 0.0821" "$out"
  grep -q "^  cag_amortize_shared_compute:" "$out" || \
    sed -i "/^  context_window:/a\\  cag_amortize_shared_compute: true" "$out"
  echo "  $out"
}

echo "GENERATING"
for m in $MODELS; do
  gen hbmonly "$m"  8 0 off
  gen 8H8F    "$m"  8 8 on
  gen 12H4F   "$m" 12 4 on
  gen 14H2F   "$m" 14 2 on
done
n=$(ls configs_h3/mm_*.yaml | wc -l)
echo
echo "$n configs generated ($(echo $MODELS | wc -w) models x 4 designs)."

if [ "${1:-}" = "--run" ]; then
  echo
  echo "RUNNING -- roughly 200s each, $((n*200/60)) minutes total."
  cd build
  for f in ../configs_h3/mm_*.yaml; do
    printf '%-46s ' "$(basename "$f")"
    out=$(./run "$f" 2>&1)
    if echo "$out" | grep -q "Out of Memory"; then echo "OOM"
    else
      b=$(echo "$out" | grep -oP 'max_batch_size to \K[0-9]+' | tail -1)
      t=$(echo "$out" | grep -oP '^Total: \K[0-9.]+' | tail -1)
      echo "batch=${b:-?}  total=${t:-?}"
    fi
  done
  cd ..
  echo
  echo "Rows appended to results.log. Compare with scripts/compare_architectures.py"
else
  echo "Add --run to execute, or run them yourself:"
  echo "  cd build && for f in ../configs_h3/mm_*.yaml; do ./run \"\$f\"; done"
fi
