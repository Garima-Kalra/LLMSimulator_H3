# H3 replication: paper facts vs. simulator assumptions

This documents, for every number the H3 replication (`results.log`,
`extract_results.py`, `plot_results.py`) depends on, whether it comes
directly from the paper or is an assumption I made because the paper's own
(closed-source, "in-house") simulator internals aren't published. Read this
before citing the results as validated against the paper.

## Facts taken directly from the paper

| Value | Paper source | Where used |
|---|---|---|
| Llama 3.1 405B weights ≈ 405GB in **FP8** (1 byte) | Sec. II-A | `precision_byte: 1` |
| Shared KV cache ≈ 540GB at 1M tokens, ≈ 5.4TB at 10M | Sec. II-A | derived: only matches arithmetically at **FP16 (2 bytes)**, not FP8 → `kv_cache_precision_byte: 2` |
| ISL = OSL = 1K tokens each | Sec. IV-A.2 | `input_len: 1000, output_len: 1000` |
| Shared cache ≈ 35% / 84% of total capacity at 1M / 10M | Sec. IV-A.2 | used to derive that the denominator is `num_GPUs(baseline) × 192GB`, confirming 8 GPUs (1M) / 32 GPUs (10M) as the relevant device counts |
| B200: HBM3E 192GB / 8TB/s per GPU, 24GB / 1TB/s per cube (8 cubes) | Sec. IV-A.3 | matches this codebase's pre-existing B200 preset |
| HBF: 3TB capacity (~16x HBM3E), 8TB/s bandwidth (same as HBM3E) | Sec. IV-A.3 | `hbf_capacity`, `hbf_bandwidth` in the `B200_H3` preset |
| HBM3E TDP 40W/cube, HBF TDP 160W/cube (4x, matching Sec. II-B's "up to 4x") | Sec. IV-A.3 | `hbm_tdp_per_cube`, `hbf_tdp_per_cube` |
| GPU package TDP 680W (excl. HBM/HBF) | Sec. IV-B.3 | `gpu_tdp` |
| LHB: BW_HBF = 1TB/s/cube, latency = 20µs → capacity = 40MB | Sec. III-C | not separately modeled (see "not modeled" below) — used only as a sanity check that HBF is modeled bandwidth-only, no added latency, consistent with the LHB fully hiding it |
| 1M case: 8 GPUs; 10M case: 32 GPUs ("minimum ... for the HBM-only case") | Sec. IV-A.4 | 1M matches my setup (`num_device: 8`); 10M does **not** — see limitation below |
| 10M uses scale-out (InfiniBand), not scale-up (NVLink), between GPU servers | Sec. IV-A.4 | consistent with 32 GPUs spanning multiple 8-GPU NVLink domains — not itself load-bearing for any number I report |
| Headline ratios: batch 2.6x/18.8x, throughput 1.25x/6.14x, throughput/power up to 2.69x | Sec. IV-B.1–3 | the comparison targets in `extract_results.py`'s `PAPER` dict |

## Assumptions I made (paper doesn't specify)

These are places the paper's prose doesn't give a formula or exact number,
so I used this codebase's pre-existing (pre-H3) conventions, inherited from
the MICRO 2024 "Duplex" paper this simulator was originally built to
validate — **not verified against H3's own methodology**:

1. **Attention FLOPs formula** (scale/mask/softmax constant, Q/K/V/O
   projection costs) — this codebase's existing roofline formulas,
   unchanged by me except for the CAG amortization split.
2. **"Shared KV attention amortizes memory access overhead" (paper's own
   words, Sec. II-A) means only the *memory read* of the shared cache is
   charged once per batch iteration; compute (FLOPs) still scales with the
   full context per sequence.** The paper doesn't say whether compute is
   also reduced. I chose "memory only" because that's literally what their
   sentence says, but a specialized attention kernel could plausibly reduce
   compute too — if so, my throughput numbers understate H3's benefit.
3. **FP8 doubles GPU compute-peak-FLOPs** (`compute_peak_flops *= 2` when
   `precision_byte == 1`) — a pre-existing convention in this codebase, not
   something the paper confirms or denies for their own simulator. This
   directly sets where the compute/memory roofline crossover sits, which is
   the reason throughput stays flat in my results (see below) — if the
   paper doesn't apply this doubling, their crossover point differs from
   mine and that could be the whole explanation for the throughput gap.
4. **Activation-memory formula** (per-layer intermediate tensor sizes counted
   toward HBM capacity) — this codebase's existing formula, used as-is.
5. **Communication/collective (all-reduce) overhead model** for tensor
   parallelism — this codebase's existing NVLink/InfiniBand roofline model,
   used as-is; not validated against whatever the paper assumes.
6. **CAG shared-cache wiring is only implemented for the plain-GQA decode
   path** (`SelfAttentionGen`), not prefill or MLA/DeepSeek-style variants.
   Irrelevant to this specific Llama-3.1-405B/decode-mode comparison, but a
   real gap if you test other models/modes.
7. **Weight and shared-cache sharding both use this codebase's
   `ne_tp_dg`-based tensor-parallelism, capped at `num_kv_heads` (8) for
   attention.** This is the gap explained above — the paper's 32-GPU/10M
   case requires distributing the shared cache (and probably weights)
   across more GPUs than there are KV heads, which needs context-parallel
   sharding this simulator doesn't implement. **This is the most consequential
   open gap** — it's why the 10M comparison has no HBM-only baseline data
   point at all in my results.

## What this means for the two comparisons

- **1M / 8 GPUs**: 8 GPUs happens to equal `num_kv_heads` exactly, so the
  existing TP model shards everything the way the 35%/84% check implies it
  should, with no extra assumption needed. This is the comparison to trust
  most: batch-size ratio (2.37x vs. paper's 2.6x) is a genuine,
  paper-grounded result. Throughput (1.02x vs. 1.25x) and throughput/power
  (0.45x vs. paper's implied >1x) are **not** matched, and the diagnosed
  cause (assumption #3, FP8 compute-doubling pushing the roofline crossover
  down to ~280 tokens for this model, an order of magnitude below any
  capacity-derived batch size) is the one remaining lever I haven't tuned.

- **10M / 32 GPUs**: cannot be properly replicated without implementing
  context-parallel KV-cache sharding (assumption #7). My 10M numbers use
  8 GPUs (the max this simulator supports for this model), so they
  understate what 32 GPUs would achieve — not comparable to the paper's
  10M figures as-is.
