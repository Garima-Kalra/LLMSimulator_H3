# Notes C: HBF systems (HBFlex, HBF Sucks?, TileLens) and HBF simulators (HBFSim, HBF-Sim)

Written for the BTP "HBM + HBF integration for LLM inference" (IIT Guwahati).
Sources: plain-text extractions in `scratchpad/txt/`, all five read in full. Page and section numbers refer to the arXiv versions named below.
Conventions:
- "[paper]" means the number is copied from the paper.
- "[derived]" means I computed it from the paper's own numbers. Check it before you cite it.
- "unknown" means the paper does not state it.

Context from the student's report (`BTP_Report.tex`). It extends an analytical roofline simulator (LLMSimulator, SNU) to a second tier and evaluates H3's placement of read-only weights and shared KV in flash. Its claims:
- Flash is over-provisioned by 17–67×.
- 12H4F mixed stacks give 1.75× concurrency at 21% lower power.
- GC does not engage at H3's provisioning. It is "essentially free" if KV blocks are allocated per request.
- Wear concentration and static wear levelling decide lifetime by up to 10.7×.
- Retention refresh is negligible (995 years), assuming SLC with 100K P/E cycles.

Several findings below bear directly on those claims. They are flagged **[BTP]**.

---

## 1. HBFlex

### 1.1 Bibliographic
- **Title:** "HBFlex: A Flexible Memory System for Bridging Fine-Grained LLM States and Coarse-Grained HBF Parallel Execution"
- **Authors:** Shuzhang Zhong* (Peking University), Weikai Xu* (HKUST), Yifan Zhou (PKU), Tongbin Zhao (PKU), Tenghao Zhao (PKU), Yifei Kang (Alibaba Group), Cunyin Chang (Alibaba Group), Shu Li (Alibaba Group), Guangyu Sun (PKU), Meng Li† (PKU, meng.li@pku.edu.cn). *equal contribution; †corresponding.
- **Venue:** unknown. It is an arXiv preprint in ACM format, and no venue is named.
- **Year / id:** 2026, arXiv:2609.18675v1 [cs.AR], 16 Sep 2026.

### 1.2 Problem
Serving LLMs entirely from HBF causes three problems:
1. Fine-grained KV reads cause plane placement and access imbalance.
2. Incremental KV writes (about 75 µs, non-preemptible programs) interfere with foreground reads.
3. Mixed KV lifetimes inside a block amplify GC. The paper measured about 30× WAF for DeepSeek-V4-Pro.

Hybrid HBM/HBF designs avoid these problems, but under a fixed package budget the retained HBM takes away HBF planes and so HBF bandwidth. HBFlex argues for a **full-HBF** package with no HBM at all, and co-designs placement, writeback and reclamation.

### 1.3 Architecture and mechanisms
- **Attachment:** GPU compute die → D2D → HBF base die → TSV → NAND core dies (Fig. 11).
  - The base die holds the NAND controller, TSV PHY and a **40 MB SRAM buffer per stack** (240 MB over six stacks).
  - The SRAM is shared by weight prefetch, buffered decode KV, hot-plane offload and GC migration (Eq. 2).
  - 40 MB of SRAM at 3 nm with 20% peripheral overhead is ≈ 8.06 mm², about 6.7% of a 121 mm² base die (the base-die area is taken from H3).
- **No HBM.** Prefill and decode are **co-located** on one GPU+HBF package, with no P/D disaggregation. Long prefill compute windows are used to drain buffered decode KV writes.
- **Four granularities are kept separate:**
  - logical KV page: the allocation and prefix-reuse unit;
  - physical stripe: G adjacent 4 KiB pages across complementary planes;
  - work tile: the KV subset read by one CTA;
  - erase block.
- **Three runtime modules:** Paged Logical KV Manager, Placement-Aware Read Scheduler, Write/Erase-Aware Controller.
- **Mechanism 1: HBF-aware placement and scheduling.**
  - Planes are grouped into Plane Groups, and adjacent physical pages are striped across the planes of a group.
  - Hot shared prefixes (for example system prompts) are replicated across PGs, so reads can be redirected to less-loaded PGs.
  - Excess pages on the hottest planes are cached in base-die SRAM. Offloading the top 10% of planes is chosen, costing 6–12 MB/stack.
  - CTA chunks are chosen greedily to cover as many distinct planes as possible, preferring the same request so the query vector can be reused.
- **Mechanism 2: window-aware writeback.**
  - Weights of consecutive operations are prefetched into SRAM, so compute runs back-to-back without HBF reads. For DeepSeek-V4-Pro in INT8, three expert projections are about 63 MB and the two Q projections about 106 MB.
  - Decode KV is buffered and drained during prefill. Most prefill-to-prefill intervals hold fewer than 4K tokens, about 20 MB at ≈5 KB/token for DeepSeek-V4-Pro.
  - Writeback is forced, stalling decode, when SRAM nears capacity.
- **Mechanism 3: lifetime-aware reclamation.**
  - Cross-layer KV pages of the same request are packed into the same block.
  - A lifetime-interval label, taken from the root request's start time, is propagated to descendants. Start and end times correlate strongly: Pearson 1.000 for To-B, 0.994 for Thinking, 0.939 for Coding (Fig. 14).
  - Deferred reclamation: eviction starts at 90% utilisation and GC at 95%.
- **FTL / GC / wear levelling:**
  - GC is modelled as valid-page migration before block erase.
  - **Wear levelling is not discussed.**
  - The OCP host/device split (host-managed zone remapping; see §2 and §5) is not discussed.

### 1.4 Device parameters (all from §3 / §6.1)
**HBF**
| Parameter | Value | Source |
|---|---|---|
| Stacks per GPU | 6 | "industry-provided HBF configuration" |
| Dies per stack | 16 | same |
| Planes per die | 32 (3,072 planes in total) | same |
| Blocks per plane | 512 | same |
| Pages per block | 512 | same |
| Page size | 4 KiB | same |
| Read / program latency | tR ≈ 4 µs, tPROG ≈ 75 µs | cited to FlashAccel [31] |
| Read bandwidth | 488 GB/s per stack (2.93 TB/s over 6) | paper |
| Write bandwidth | 27.2 GB/s per stack | paper |

- Access granularity: array reads and programs work on 4 KiB pages. "The host interface may return the requested portion of a buffered page."
- Capacity [derived]: 16 × 32 × 512 × 512 × 4 KiB = 512 GiB per stack, 3 TiB per GPU.
- Bandwidth consistency check [derived]: 512 planes × 4 KiB / 4 µs ≈ 524 GB/s, and 512 × 4 KiB / 75 µs ≈ 28 GB/s. The quoted 488 and 27.2 GB/s are consistent with plane-parallel array limits.
- Endurance: "A 1-TB HBF with 100K P/E cycles supports approximately 54.8 TB/day over five years." **No citation for 100K P/E.** 100K is an SLC-class figure.
- Not given: power, thermal, ECC, retention.
- Spec: cites "[20] Open Compute Project. 2023. OCP HBF Architecture Specification v0.7.0 FINAL". Every other paper dates v0.7.0 to **August 2026**, so the 2023 year looks like a bibliography error.

**HBM**
- HBM3e stack: 8 DRAM dies, 24 GB, 0.8 TB/s (JEDEC JESD238A).
- GPU: H200-class. Compute is "2,000 TOPS" (ideal, Fig. 9).

### 1.5 Evaluation methodology
- **Simulator:** custom trace-driven simulator that "models individual read and write operations". It is **not validated** against hardware and no release is mentioned.
- **Models:**
  - DeepSeek-V4-Pro (1.6T params), run as TP2 × DP4;
  - DeepSeek-V3, Hunyuan3 ("Hy3") and Qwen3-235B, each run as DP4.
- **Workloads:**
  - four Alibaba Bailian production traces: To-B, To-C, Coding, Thinking;
  - SWE-bench traces collected by running Qwen3, with concurrency from 128/256 up to 4K/8K tasks.
  - The ablation uses 4,096 concurrent SWE-bench tasks on four DP GPUs.
- **Baselines:**
  - H3 (cascaded, prefix KV in HBF);
  - FlashAccel FA-CLI (co-located: 5 HBF + 1 HBM stacks);
  - FA-CSI (cascaded);
  - IS (integrated DRAM+flash stack; Yin et al.).
- **Package normalisation for cascaded designs:** ¼ of the budget goes to HBM and ¾ to HBF. HBM capacity is scaled to ¼, and HBF capacity and bandwidth to ¾. HBM bandwidth is unchanged.
- All designs get the same per-stack SRAM.
- Batch sizes and context lengths are set by the traces and are not listed explicitly.

### 1.6 Headline results
- **Bailian latency (geomean over model × trace):**
  - vs FA-CLI / FA-CSI: TTFT 1.20× / 1.33×, TPOT 1.21× / 1.35×.
  - **vs H3: TTFT 22.65×, TPOT 1.89×.** The paper attributes the TTFT gain to admission queueing under KV-capacity pressure.
- **SWE-bench throughput:**
  - lowest concurrency: 1.31× over FA-CLI and 1.47× over FA-CSI;
  - highest concurrency: 1.48× and 1.58×;
  - vs H3: 1.56× at the lowest concurrency, rising to **3.30×** at the highest.
- **Ablation (Table 2), cumulative throughput vs Naive HBF, DeepSeek-V3 / Qwen3-235B:**

  | Configuration | DeepSeek-V3 | Qwen3-235B |
  |---|---|---|
  | + plane-balanced placement | 2.94× | 2.59× |
  | + runtime read balancing | 4.08× | 4.25× |
  | + window-aware writeback | 4.10× | 4.27× |
  | + lifetime-aware reclamation | 4.15× | 4.32× |

  Almost all of the gain comes from read-side balancing.
- **Read-wave amplification:** 5.97× (random) → 3.95× → 2.25×, a 62.21% reduction.
- **Write stall per iteration (DeepSeek-V4-Pro):** 4.54 ms (immediate) → 69.12 µs (stripe writeback) → ≈0.003 µs (window-aware).
- **GC WAF (DeepSeek-V3, Qwen3):** 11.98× → 1.24× with lifetime-guided packing → 1.04× with deferral, an 89.39% average reduction. In motivation, DeepSeek-V4-Pro reached about 30× (Coding 31×, Reasoning 29×).
- **Endurance arithmetic (§3.3):**
  - 434B new-KV tokens/day × 35 KB/token (DeepSeek-V3) = 15.19 PB/day over 1,814 GPUs, or 8.37 TB/day per GPU.
  - Qwen3 at 95 KB/token gives 22.73 TB/day per GPU.
  - These are 15.3% and 41.5% of the 54.8 TB/day budget, leaving headroom for WAF of 6.5× and 2.4×.

### 1.7 Stated limitations and future work
- **There is no limitations or future-work section.**
- Only implicit caveats:
  - writeback gains are small because "relatively small per-token KV footprints ... limit write traffic";
  - forced writeback "temporarily stalls decoding";
  - the endurance analysis holds "under our workload assumptions".

### 1.8 Critical view
- **Endurance rests on an unsourced 100K P/E.**
  - Take a TLC-class budget instead, such as the ≈3.3K P/E implied by HBF Sucks' 21.7 TB/day for 12 TB [derived].
  - Raw KV writes alone (8.37–22.73 TB/day per GPU) would then exceed the budget several-fold, before any GC.
  - The "full-HBF is endurance-safe" argument therefore depends entirely on SLC-class cells. SLC cells cut the stated capacity advantage by about 3× compared with TLC.
- **Latencies are SLC-like.** tR = 4 µs and tPROG = 75 µs come, through FlashAccel, from a 1-bit/cell die (the Kouchi JSSC 2021 part, cited by HBFSim). This is the most optimistic latency point of any paper here. There is **no sensitivity sweep** over tR or tPROG.
- **The H3 comparison is unfair.**
  - H3 is normalised to ¼ HBM capacity, which cripples its HBM-resident KV. That produces the 22.65× TTFT.
  - The original H3 chains HBF *behind* HBM without giving up HBM sites. **[BTP]** The student found H3 wins precisely because it preserves HBM sites. HBFlex's normalisation removes that advantage by assumption.
- **Not modelled:**
  - power and thermal, even though a write-heavy all-flash package is the worst thermal case (see HBF Sucks' 202 GB/s throttle point);
  - ECC and read-retry;
  - read disturb and retention refresh (OCP requires host refresh);
  - tail latency (only means are reported);
  - TCO.
- **Validation:** the custom simulator is unvalidated and unreleased.
- **Deployment:** evaluated only as DP4 / TP2. Multi-node scaling is not studied.
- **Granularity:** the host 64 B read path from OCP is not modelled. KV reads are counted at page level.
- **[BTP] Direct relevance.**
  - HBFlex states that request-isolated blocks make GC trivial but "severely restricts plane-level striping". That is the tension the student's "per-request block allocation makes GC essentially free" claim must address.
  - HBFlex's lifetime-guided packing reaches WAF 1.04 while keeping striping. That is a design point the student could adopt or compare against.
  - HBFlex models no HBM, so it cannot evaluate a hybrid placement policy.

### 1.9 BibTeX
```bibtex
@misc{zhong2026hbflex,
  title         = {{HBFlex}: A Flexible Memory System for Bridging Fine-Grained {LLM} States and Coarse-Grained {HBF} Parallel Execution},
  author        = {Zhong, Shuzhang and Xu, Weikai and Zhou, Yifan and Zhao, Tongbin and Zhao, Tenghao and Kang, Yifei and Chang, Cunyin and Li, Shu and Sun, Guangyu and Li, Meng},
  year          = {2026},
  eprint        = {2609.18675},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v1, 16 Sep 2026}
}
```

---

## 2. HBF Sucks? A Full-Stack Characterization of HBF for KV-Centric LLM Serving

### 2.1 Bibliographic
- **Title (as printed in v4):** "HBF Sucks? A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving".
  - The local filename, and HBFlex's citation [16], use "HBF Sucks!". That is probably an earlier version's title.
  - HBFSim and HBF-Sim both cite it with "?". Use the "?" form.
- **Authors:** Zhuoran Li¹, Zhuohang Bian¹, Xin Huang², Yibo Zhao¹, Guangyu Sun¹, Youwei Zhuo¹*.
  - ¹School of Integrated Circuits, Peking University; ²Fudan Institute of Systems for Advanced Computing, Fudan University.
  - *Corresponding author.
- **Venue:** unknown (arXiv preprint).
- **Year / id:** 2026, arXiv:2608.11668v4 [cs.AR], 14 Sep 2026.
- **Code:** simulator at https://github.com/pku-lemonade/TokenSim/tree/hbf (base TokenSim: https://github.com/pku-lemonade/TokenSim).

### 2.2 Problem
The paper asks whether swapping the SSD tier of a Mooncake-style KV-offloading stack for package-local HBF speeds up serving, keeping the runtime otherwise unchanged. It finds the opposite. It then explains why with a cost-benefit model built on three necessary conditions:
- **C1:** read I/O is on the critical path;
- **C2:** reads outweigh writes;
- **C3:** delivered bandwidth is sustainable, both thermally and in endurance.

### 2.3 Architecture and simulator
- **Organizations** come from Joungho Kim's KAIST HBF roadmap presentation (10 Feb 2026). These are projections.
  - **HBF-1 (2028):** the interposer is filled with HBF (6 × 512 GB); the near tier is demoted to off-interposer GDDR7. Table 2 gives "12 GDDR7, 48 GB" for H100. Fig. 1 draws 15 GDDR7 parts of 3 GB each, which contradicts the table.
  - **HBF-2 (2030):** half the sites are HBM (3 HBM, 48 GB) and half HBF (3 stacks, 1.5 TB).
  - **HBF-3 (2032):** advanced interposer; not evaluated.
- **Data placement:**
  - weights sit in the near tier (GDDR7 or HBM);
  - transient KV is saved to HBF by the Mooncake connector (16-token hashed blocks, saved during prefill after existing-block filtering) and restored to the near tier before reuse.
  - KV is **never read in place** from HBF.
- **Simulator:** extended TokenSim models arrival, prefill/decode, continuous batching, paged KV, admission, preemption, recompute, KV save/restore, tier transfers, queueing and SLO. The extensions add Mooncake lookup, per-tier counters and HBF-1/HBF-2 profiles.
- **Validation:** TokenSim's base was validated against real machines in its original paper. For the extensions, only regression checks were run (byte conservation, ordering, deterministic replay).
- **Thermal model:**
  - 3D-ICE steady state for a 16-Hi stack similar to HBM4, with 128-layer TLC 3D-NAND dies;
  - driven by read/write energy per 16-token KV block;
  - 80 °C is chosen as the "safe junction temperature";
  - throttling closes the hottest planes and redirects writes to cooler ones.
- **Endurance model:**
  - daily write rate normalised from 2-hour HBF-2 8×H100 replays;
  - linear exhaustion of the budget, with unit WAF assumed in HBF's favour.
- **Base-die NMP (Finding 2), configured to favour HBF:** 24 FP16 MAC arrays at 600 MHz, 896 GB/s internal bandwidth, no area or power penalty.
- **OCP host/device contract as reported (spec v0.7.0, "dated August 3, 2026", Ballapuram & Lee, Sandisk and SK hynix):**
  - AXI over UCIe; non-coherent; host maps global addresses across independent channels;
  - base die handles ECC, bad blocks and command scheduling;
  - 64-byte reads; two page-cache buffers per bank; ordered sensing per bank; batch-read hints;
  - writes must accumulate a full 4 KiB page and complete only after programming;
  - pages are written sequentially within a block; a page-0 write triggers automatic block erase;
  - optional host-managed SRAM scratchpad (§11.3);
  - wear levelling on the base die or the host;
  - **reclamation and live-data movement are delegated to the host** (zone remapping needs invalid data and quiesced reads);
  - host refresh for data age and read-disturb counts;
  - **24-hour powered-on retention at 85 °C** (§§9, 11.5);
  - normal / light / severe throttling / shutdown states with configurable thresholds (§9);
  - product-specific maximum P/E, with average P/E exposed to the host;
  - separate weight and KV channels are described (§13.3.3).

### 2.4 Device parameters
| Item | Value | Source |
|---|---|---|
| HBF latency (media-only sweep) | 8/80, 12/120, 20/200, 30/300 µs read/write | "sourced or swept" |
| NMP study latency | 12 µs read, 120 µs write | Table 6 |
| HBF stack | 16 × 128–256-layer NAND dies + TSV + base die; 512 GB | roadmap / HAVEN |
| HBF-1/H100 (8 GPUs) | 48 GB GDDR7 at 1.344 TB/s per GPU; shared 24 TB HBF at 1.2 TB/s | Table 3(c) |
| HBF-1/B200 | 96 GB GDDR7 at 2.688 TB/s; 48 TB HBF at 2.4 TB/s | Table 3(c) |
| HBF-2/H100 | 48 GB HBM3e at 1.5 TB/s; 12 TB HBF at 0.6 TB/s | Table 3(c) |
| HBF-2/B200 | 96 GB HBM3e at 4.0 TB/s; 24 TB HBF at 1.2 TB/s | Table 3(c) |
| SSD baselines | 96 GB HBM at 3.0 TB/s (H100) or 192 GB at 8.0 TB/s (B200); 12/24/48 TB SSD at 54.7–218.8 GB/s read; KIOXIA CM7-V 3.2 TB, 14.0/6.75 GB/s seq. R/W, 3 TB usable | Table 3(c), [13] |
| Density ratio | 512 GB HBF vs 24 GB HBM ≈ 21.3 | §2.3.1 |
| Power (cited) | H3 assumes ≈160 W per HBF cube, so 6 stacks ≈ 960 W | [7] H3 |
| Thermal (own model) | hits 80 °C at **202.27 GB/s and 53.72 W** per stack | Fig. 10 |
| Endurance | HBF (TLC, unit WAF, 5-year target) 21.7 TB/day; capacity-matched SSD (4 × CM7-V at 3 DWPD) 38.4 TB/day | §4.4 |

- Other points on the Fig. 10 power/bandwidth line: 50 GB/s → 13 W, 100 → 27 W, 150 → 40 W, 250 → 66 W, 300 → 80 W, 400 → 106 W.
- Device latency, energy and bandwidth are said to come from HAVEN's NAND model [9].
- Per-stack HBF bandwidth [derived, ambiguous]:
  - the table labels 1.2 TB/s as the "shared secondary tier" for 24 TB = 48 stacks over 8 GPUs;
  - if that is the aggregate, per-stack bandwidth is only ≈25 GB/s; if it is per GPU, it is ≈200 GB/s;
  - either way it is far below the spec's 384 GB/s–3.072 TB/s per cube. **Ask the authors, or check the TokenSim hbf config.**
- Implied P/E [derived]: 21.7 TB/day × 1,826 days / 12 TB ≈ 3.3K P/E cycles.
- The spec date is stated explicitly: **OCP HBF base-die spec v0.7.0, August 3, 2026.**

### 2.5 Evaluation methodology
- **Traces:** four complete 2-hour Aliyun Qwen-Bailian traces (Table 3a):

  | Trace | Requests | QPS | Multi-turn | Reused |
  |---|---|---|---|---|
  | traceA | 43,058 | 5.98 | 46.3% | 46.8% |
  | traceB | 172,800 | 24.0 | 0% | 6.0% |
  | coder | 43,011 | 5.97 | 38.6% | 39.0% |
  | thinking | 10,812 | 1.50 | 11.1% | 14.3% |

- **Models:** Qwen3-4B, Qwen3-32B (dense); DeepSeek-V3.2-685B, GLM-5.2-753B, Kimi-K2.7-Code-1.1T (MoE). Maximum context runs from 40,960 to 1,048,576 tokens.
- **System:** eight GPUs, TP8, with EP for MoE, on H100 and B200 profiles.
- **Load:** offered load 0.05–32 QPS, set by rescaling timestamps.
- **Metrics:** TTFT, TBT, E2E, throughput, SLO goodput, per-tier byte counters.
- **No real HBF hardware.** The comparison is simulator output against a simulated SSD baseline.

### 2.6 Headline results
- **Finding 1 (C1).**
  - HBF is worse than its capacity-matched SSD pair on every request metric: mean E2E latency 2–5.5×, throughput down 4–34%, maximum SLO goodput down 1.1–2.7×. H100 is at the severe end.
  - Making media latency k = 3.75× faster moves E2E by only 0.75% (HBF-1) and 0.88% (HBF-2), which gives **f ≈ 1%**.
  - Example (GLM-5.2, traceA, B200, 32 QPS): connector time 399 s vs 1,293 s (3.2× lower), yet E2E 4,386 s vs 1,682 s (2.6× higher).
  - HBF-1 is 16–64% slower than HBF-2 despite having twice the flash.
  - B200 Table 5: e.g. Kimi-K2.7 HBF-1 +285.2% E2E and −51.7% goodput.
- **Finding 2.** A base-die NMP on HBF gives −0.03% average decode-attention gain (range −29.05% to +20.49%). An HBM-side NMP gives 42.84%.
  - HBF-resident KV fraction is only 15.48% (per trace: 26.62%, 13.63%, 17.64%, 4.02%).
  - End-to-end effect: decode p50 +0.39%, p99 +0.82%.
- **Finding 3 (C2).**
  - Write/read ratio: traceA 2.14×, traceB 4.90×, coder 1.14×, thinking 2.20×.
  - That gives ρ = 0.47 / 0.20 / 0.88 / 0.45.
  - Prefix hit rate is 38–53%.
  - The top decile of blocks supplies 64–100% of reuse (Gini 0.80–1.0).
- **Finding 4.** Mooncake write batching improves SSD-12 p50 by 8.9–21.1%, but HBF-2 by only 2.69% overall (+9.2% at 16 tokens, +0.72% at 128, −1.85% at 512).
- **Finding 5 (C3).** The thermal limit is reached at 202.27 GB/s and 53.72 W, and the throttled replay holds about 202 GB/s at about 80 °C.
- **Finding 6.**
  - Every trace writes 48–140 TB/day. HBF lifetime is 0.56× the SSD pool's on every trace.
  - Fig. 11 lifetimes, in days: SSD 763/501/826/1460 vs HBF 431/283/466/825.
  - Coarser 16→128→512-token blocks raise HBF writes 188→237→258 TB (1.38×).

### 2.7 Stated limitations and future work
- §9.3: the hardware profiles "follow the HBF roadmap's near-term organizations". The architecture comparison "measures the joint effect of the near tier and backing tier under the same pooling policy".
- Thermal and endurance results "use the stated temperature policy and TLC write budget, while cost is expressed through package-resource trade-offs".
- The placement map "gives the axes and the break-even rule ... rather than a calibrated boundary".
- Future direction (explicit "Challenge to the Reader"): design "an HBF organization and runtime that accelerates critical-path reads, earns enough useful reads per write, sustains bandwidth within thermal and endurance limits".
- Required runtime: "critical-path-aware access, reuse-aware admission, and thermal-aware write budgeting".

### 2.8 Critical view
- **The headline conflates the medium with the package trade.**
  - The damage comes from halving near-tier capacity and bandwidth (96→48 GB, 3.0→1.5 TB/s on H100). The authors say as much ("C_pkg exceeds fΔt_media by two orders of magnitude").
  - The paper never evaluates HBF **added without removing HBM**, for example H3-style cascading behind the HBM base die. It also never evaluates weights in HBF, which frees near-tier capacity.
  - **[BTP]** The student's H3 cascade is exactly the untested configuration.
- **KV is always restored to the near tier.** Reading KV in place (HBFlex) and streaming weights from HBF (H3, TileLens, FLINT) are excluded by construction, so the negative result covers only one runtime design.
- **Bandwidth looks under-provisioned.** Per-stack HBF bandwidth in Table 3(c) appears far below the spec [derived, see §2.4].
- **The thermal model is simplified:**
  - steady-state, single stack, not package-coupled;
  - the 80 °C limit is the authors' choice, while OCP and HBFSim use 105 °C as the operating junction;
  - a single energy per 16-token block;
  - no GPU heat.
  - HBFSim reaches the opposite conclusion (no throttling at 384 GB/s read-only). See the cross-paper notes.
- **The endurance model is simplified:** linear exhaustion, unit WAF (favourable to HBF), no wear-levelling model, no retention refresh traffic even though OCP's 24 h at 85 °C implies refresh.
- **Not studied:**
  - tail latency beyond SLO goodput;
  - ECC and read-retry;
  - multi-tenant interference;
  - TCO in dollars;
  - long-context (≥128K) sweeps, even though model maximum contexts reach 1M, traces have mean prompts of 574–4,540 tokens.
- **Positive value for the BTP:**
  - C1–C3 plus Eq. (8), ρ > ρ★ = AC/S, give a clean, device-agnostic test for any HBF placement.
  - Fig. 12's placement map (weights and shared prefixes: strong; private KV: poor) supports H3 / TileLens-style read-only placement, which is what the student evaluates.

### 2.9 BibTeX
```bibtex
@misc{li2026hbfsucks,
  title         = {{HBF} Sucks? {A} Full-Stack Characterization of High-Bandwidth Flash for {KV}-Centric {LLM} Serving},
  author        = {Li, Zhuoran and Bian, Zhuohang and Huang, Xin and Zhao, Yibo and Sun, Guangyu and Zhuo, Youwei},
  year          = {2026},
  eprint        = {2608.11668},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v4, 14 Sep 2026. Simulator: \url{https://github.com/pku-lemonade/TokenSim/tree/hbf}}
}
```

---

## 3. HBFSim (simulation under real GPU execution)

### 3.1 Bibliographic
- **Title:** "HBFSim: Fast and Faithful Simulation of High-Bandwidth Flash Under Real GPU Execution".
- **Authors:** Yanpeng Hu¹*, Yiwei Yang²*, Yuanwu Zhu³, Kexin Chu⁴, Yusheng Zheng², Andi Quinn², Wei Zhang⁴.
  - ¹ShanghaiTech University; ²UC Santa Cruz; ³University of Science and Technology of China; ⁴University of Connecticut.
  - *equal contribution.
- **Venue:** unknown (arXiv preprint).
- **Year / id:** 2026, arXiv:2609.09800v2 [cs.AR], 18 Sep 2026. HBF Sucks cites v1.
- **Open source:** the paper says "the first open-source HBF simulator". **No repository URL appears in the text, and I found none embedded in the PDF's link annotations.**

### 3.2 Problem
- HBF behaviour must be evaluated before silicon exists.
- Cycle-level GPU simulators are far too slow for production LLMs. Accel-Sim runs at 12,500 warp instructions/s in trace mode, and a lavaMD run took more than 5 days with 10,748,031× slowdown.
- Trace replay cannot capture how different HBM:HBF splits change allocation, migration and kernels.
- Key idea: run the real LLM on a real GPU and inject only the *program-visible* effects of HBF (timing, thermal, retention) into the running program without destroying GPU asynchrony.

### 3.3 Simulator design (D1–D6)
- **D1: explicit, fail-closed boundary.**
  - Applications register virtual-address ranges as HBF timing or capacity ranges.
  - A PTX pass rewrites memory instructions, and device helpers classify each effective address.
  - TMA TensorMaps are hashed and decoded, so tiles crossing the HBM/HBF boundary are split byte-by-byte.
  - Incomplete instrumentation refuses the launch.
  - Precompiled cubins (no PTX) are not observed. NVBit (SASS) was rejected at more than 10× the overhead.
- **D2: preserving asynchrony.** Issue and consumption are split. The rewritten access creates a "device future" holding its modelled completion time, and the wait happens at the first consumer instruction, mimicking the scoreboard. For TMA, native and modelled completion must both happen before the barrier releases.
- **D3: closed-loop thermal controller.**
  - Uses real GPU telemetry: temperature and power, published by the CUDA process.
  - HBF heat comes from media events: E_NAND = E_command + e_byte × bytes, in 10 ms bins.
  - Whole-package 3D-ICE 4.0 operator covering GPU, HBM and HBF with a shared G/C matrix. It is reduced by ERA to a 224-state model, which agrees with the full model to 0.0308 K.
  - Hysteretic states Normal / Light / Severe / Shutdown, following OCP; the thresholds RTT/LTT/STT are configurable.
  - Service fraction q scales service time: s_T = ⌈s·10⁶/q⌉.
- **D4: retention, refresh and wear as injected latency.**
  - Per-block retention damage follows an Arrhenius model (Ea = 1.04 eV, taken from HeatWatch for 3D NAND after 10K P/E).
  - Each block also tracks a read-disturb count, a zone and its P/E count.
  - Refresh read–rewrite pairs enter the same MQSim queues as foreground traffic, so the workload pays for them.
  - Reliability sweeps over activation energy.
- **D5: capacity mode.** A bounded HBM frame cache backed by a sparse host file. The paper demonstrates 110 GiB mapped through a 2 GiB page cache.
- **D6: speculative prefetcher.** Next-page after a fault, with a pluggable interface.
  - On a synthetic MoE stream, accesses served without a media read rise from 17.97% to 89.16%.
  - Waited reads fall 840→111 (−86.79%). Total reads rise 16.55%.
  - It is **not in the timing path.** The paper says "this paper claims no run-time performance benefit from D6".
- **Timing sources:**
  1. an online MQSim media model on the host, on the reference path;
  2. a calibrated end-to-end "vmem" curve measured on a Dell CD8P NVMe through a host paging path, with an 11,133 ns single-page cost.
- **Implementation:**
  - 9,302 lines of C/C++/CUDA, plus 3,250 lines for calibration;
  - reuses MQSim, bpftime (CUDA interception), llama.cpp and vLLM, all pinned by commit.
- **Abstracted away or absent:**
  - no GPU simulation, since the GPU is real;
  - no cycle-level UCIe or AXI;
  - FTL and GC are not described beyond MQSim's media model;
  - the three synthetic timing profiles (conservative, nominal, aggressive) share 1 TiB and 128–1,000 GB/s, and are **not anchored to the spec** (the paper says re-anchoring "edits a data file").

### 3.4 Device parameters
- **Table 1, first-generation HBF:**
  - 512 GB/stack; 0.4–3.0 TB/s read; UCIe;
  - read latency "∼20 µs [13]", where [13] is **TileLens**;
  - 64 B read granularity [4 = OCP spec];
  - limited P/E; unpowered retention "not guaranteed".
- **HBM4 (Table 1):** 64 GB/stack, ∼2 TB/s, 10–100 ns, 32 B granularity.
- **Evaluation stack (§6.3):**
  - one HBF stack, 16 host channels, 8-hi, which is the height Speed Grade 1 allows;
  - read power 5 W idle + 8.4×10⁻¹¹ J/B, giving 37.256 W at 384 GB/s;
  - 384 GB/s is the OCP Table 4 per-cube maximum for Speed Grade 1. Speed Grades 2 and 3 are 1.536 and 3.072 TB/s at 16-hi and are not modelled;
  - 64-bit host channel (fixed by the spec);
  - ECC engines assumed not to cap bandwidth, since the spec leaves this open.
- **Temperature limits (the authors' own choices):**
  - HBF 105 °C (spec operating junction range);
  - HBM 105 °C (Micron HBM3E "TOPER");
  - compute die 90 °C (RTX 5090 published maximum, not the server part).
- **Capacity study:**
  - 512 GiB per HBF unit (spec cube);
  - 36 GiB per HBM stack (the spec's illustrative worked example);
  - package budget 861.1 mm², a "project assumption".
- **Prefetch lead time (§2):** on H100, computing layer i leaves about 130.2 µs to read layer i+1's weights, which is 6.5× the assumed 20 µs page read.
- **Spec date:** "Sandisk and SK hynix introduced the HBF specification through the OCP in August 2026, with Google and Tenstorrent taking part in validation". Samples are expected in 2027.
- **Not given:** tPROG and tBERS numbers.

### 3.5 Evaluation methodology
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (CC 12.0, 97,887 MiB GDDR7), driver 595.84, CUDA 13.0.88. The flash and thermal reference is a Dell DC NVMe CD8P E3.S 1.92 TB on PCIe 5.0 x4.
- **Workload:**
  - Qwen3-30B-A3B under vLLM v0.15.1, 32-token prompt, seed 0, 8 output tokens;
  - a TinyLlama run under llama.cpp.
- **Coverage is tiny.** Under vLLM the registered range is only the **first 16,384 bytes of `model.layers.0.mlp.experts.w13_weight`**: 24 modelled launches out of 2,304, and 10,339 coverage decisions. The full 61,064,245,248 bytes cannot be registered because of vLLM's profiling pass.
- **Validation performed:**
  - functional: 42 checks, plus 128 identical checksums in capacity mode;
  - timing: the injection *increment* (1,000 − 200 µs = 800 µs) across 6 TMA cells;
  - thermal: reduced model vs full model ("model against model, not against silicon").

### 3.6 Headline results
- The run completes: 60.988 s load, then 44.469 s generation for 8 tokens. Output tokens are identical to the baseline.
- **Injection error is 0.152%** on the 800 µs increment, in the 8,192 B cell (798,784.0 ns measured). The median delivered increment is about 400 ns below the request, with the middle 90% between −4,800 and +4,160 ns. **"no physical HBF latency was measured."**
- **Thermal:**
  - Over 600, 3,600 and 86,400 s horizons, all five policies deliver the full 384.0 GB/s with no throttling. Peak temperature is 79.8 °C.
  - The package limit is reached at the **compute die at 86.6 W** external load, before HBM (121.4 W) or HBF (105.9 W).
  - At that point the HBF layers span 44.4 K, from 91.6 °C in layer 0 (package base) to 47.2 °C in layer 7.
  - Layer 0 stays flat up to 65.09 W.
- **Measured GPU board power:** 314.988–431.370 W. Marginal board power reaches 45.707 W at 383.649 GB/s of GDDR7 traffic, which is **not** HBF.
- **Capacity layouts (861.1 mm² budget):**

  | Layout | Stacks | Installed capacity | HBM-resident KV |
  |---|---|---|---|
  | A | 6 HBM + 0 HBF | 216 GiB | ≈1.72M tokens |
  | B | 3 HBM + 2 HBF | 1,132 GiB | ≈0.54M tokens |
  | C | 1 HBM + 3 HBF | 1,572 GiB | experts don't fit in HBM |

  - At C, 30.9879 GiB is available, 57.4% of the 54.000 GiB expert set.
  - The per-stack expert-residency threshold is 59.012 GiB.
- **MoE demand:** in a Qwen3-30B-A3B routing trace (48 layers, 128 experts, top-8), deduplicated expert demand at batch 8 and 16 is 2.487× and 3.584× batch 1, against 8× and 16× routing decisions.

### 3.7 Stated limitations and future work
- The prefetcher entering the timing path "is future work".
- Results "carry to accesses inside registered ranges of rewritable kernels".
- Timing profiles are synthetic, and "re-anchoring the profiles to the published figures edits a data file".
- The thermal study uses "one GPU and one Dell CD8P" and the simulated-warning crossings are "computed values".
- Reliability outputs are "a reproducible, parameterized model outcome—not a claim that HBFSim measured the lifetime".
- "This work does not measure the timing benefit of prefetching."
- The layouts "are representative points rather than a sweep"; fitting the area budget "settles nothing about routing, power delivery, PHY placement or manufacturability".

### 3.8 Critical view
- **No end-to-end HBF performance result.** The paper validates the injection mechanism, not HBF. With only 16 KB of one tensor registered, no serving-level claim (throughput, TTFT) can be drawn.
- **The latency citation looks wrong.** The "∼20 µs" HBF read latency is attributed to TileLens [13]. TileLens gives 1–10 µs as the typical range, sweeps 1–20 µs and uses 5 µs as its headline. The 130.2 µs / 6.5× prefetch argument is built on that figure.
- **The thermal conclusion is unrealistic for data-centre GPUs.**
  - "HBF is not the binding component" holds inside a package envelope where the compute die hits 90 °C at only **86.6 W**. Server GPUs run at 300–700+ W (the card's own limits are 300–600 W).
  - The package model evidently has far less cooling than a real system, or a very different geometry.
  - The heat model is read-only (read power only). HBF Sucks shows writes are the thermal problem.
- **Not calibrated:** the timing calibration is a host NVMe paging path (Dell CD8P), not an in-package medium.
- **Not modelled in results:** write-heavy workloads, GC, ECC and read-retry latency, tail latency under refresh, multi-GPU, TCO. Refresh and read disturb are *implemented* but not evaluated with numbers.
- **Strength for the BTP:** the HBM:HBF split is a first-class runtime knob (register tensors as HBF ranges, capacity mode), which is exactly the placement question. The retention/refresh model is the only one among the five papers that makes the program pay for refresh.

### 3.9 BibTeX
```bibtex
@misc{hu2026hbfsim,
  title         = {{HBFSim}: Fast and Faithful Simulation of High-Bandwidth Flash Under Real {GPU} Execution},
  author        = {Hu, Yanpeng and Yang, Yiwei and Zhu, Yuanwu and Chu, Kexin and Zheng, Yusheng and Quinn, Andi and Zhang, Wei},
  year          = {2026},
  eprint        = {2609.09800},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v2, 18 Sep 2026}
}
```

---

## 4. HBF-Sim (Accel-Sim-integrated HBF simulator)

### 4.1 Bibliographic
- **Title:** "HBF-Sim: An Extensible HBF Simulator for Large-scale GPU Memory Systems".
- **Authors:** Yaqi Li¹², Jing Wang¹, Junfeng Wang¹, Long Yang¹², Han Yan¹, Xiaohu Chai¹, Liang Shi*¹².
  - ¹East China Normal University; ²Shanghai Innovation Institute.
  - *corresponding.
- **Venue:** unknown (arXiv preprint).
- **Year / id:** 2026, arXiv:2609.29246v1 [cs.AR], 24 Sep 2026.
- **Open source:** the contribution list says "release it as open source (§4)". **No URL is given in the text or in the PDF link annotations.**

### 4.2 Problem
HBF is "neither a large HBM nor a fast NVMe SSD". Its usable bandwidth depends on:
- how 64 B / 128 B GPU cache-line requests map onto 4 KiB NAND pages;
- how concurrency spreads over channel-affine die sets;
- how non-posted 4 KiB write aggregation and backpressure interact with the GPU memory pipeline.

Existing tools cover only part of this. GPU simulators treat off-chip memory as DRAM, SSD simulators implement NVMe/CXL block semantics, and published HBF studies use internal models. None closes the loop from GPU issue through HBF service to GPU completion.

### 4.3 Simulator design
- **Structure:**
  - built as an extension to Accel-Sim / GPGPU-Sim in C++, with the HBF sources in `hbf/`;
  - a setup script applies a patch to a pinned Accel-Sim revision;
  - the ordinary DRAM path is retained;
  - supports Make and CMake builds.
- **GPU–HBF interaction controller:**
  - cache-line-to-page alignment: p(a) = ⌊(a−B)/4096⌋;
  - channel-oriented mapping: 1/2/4/8/16 channels per stack (more is experimental), with no cross-channel sharing;
  - three placement policies: page interleave, contiguous, and an **explicit mapping table** that can place data classes such as weights and KV pages;
  - a per-channel bandwidth-credit host link, and outstanding-operation limits that model backpressure.
- **Page-based multi-stack flash manager:**
  - merge-based write module: a per-page write buffer with byte-coverage bits; flush when full, under pressure, on deadline, on a same-page read, or on idle drain;
  - strict mode (incomplete page is an error) or opt-in pad-and-program mode (`hbf_write_timeout_policy`);
  - erase-before-write and sequential in-block allocation;
  - **non-posted ACK after PROGRAM**;
  - page-keyed MSHRs merge reads, in demand mode or aggregation mode (`hbf_read_mode`, `hbf_read_agg_window`, `hbf_read_agg_threshold`);
  - two NAND page buffers per subarray, and an optional LRU logic-die page cache;
  - per-subarray READ / PROGRAM / ERASE state machines, clocked on the DRAM clock;
  - scheduling: FCFS, read-priority or write-drain.
- **Trace:**
  - request states INGRESS → MAPPED → (BUFFERED) → QUEUED → MEDIAOP → RETURN → COMPLETED;
  - 15-column CSV schema, frozen as `hbf-trace-v1`;
  - a device-only replay tool.
- **FTL / GC / wear:**
  - following the spec, "there's no device-managed garbage collection"; wear management is host-side zone remapping, with no background migration;
  - each stack "owns ... FTL". Statistics include per-zone P/E spread.
- **Abstracted away:**
  - "ECC, read retry, refresh, and thermal effects are not modeled";
  - no transaction-level AXI/UCIe (bandwidth-and-credit model only);
  - not calibrated against hardware.

### 4.4 Device parameters
| Item | Value | Source |
|---|---|---|
| Aggregate interface | up to 16 UCIe host channels per cube, **3.072 TB/s** | OCP v0.7.0 [24] |
| Capacity | 512 GiB per cube | OCP v0.7.0 [24] |
| Granularity | 64 B host reads; 4 KiB NAND pages; writes must fill 4 KiB; sequential from page 0 | OCP v0.7.0 [24] |
| Default timing | **tR = 15 µs, tPROG = 200 µs, tBERS = 2 ms** | "model parameters rather than hardware measurements" (no source) |
| Placement diagnostic timing | 20/40/80 DRAM ticks | paper |
| Table 1 HBF | latency 1–15 µs; GC "no" | Table 1 |
| Table 1 HBM | 100 ns; 32–64 B; 80–192 GB; 1–8 TB/s | Table 1 |
| MQSim cross-check | nominal channel bandwidth 192 GB/s | §5.3 |
| Clocks | GPU core 1.132 GHz, DRAM 850 MHz; 80-SM scaling config; the Qwen3 replay uses an SM80 profile at 1.410/1.512 GHz | §5.1, §5.5 |

- Array bandwidth ceiling [derived]: 512 subarrays × 4 KiB / 15 µs ≈ 139.8 GB/s. This matches the measured 139.285 GB/s at 16 channels.
- Implication [derived]: at tR = 15 µs, **about 11,250 concurrently busy planes** would be needed to fill 3.072 TB/s (3.072e12 × 15e-6 / 4096).
- The spec date is not stated explicitly (cites "OCP ... Version 0.7.0 ... 2026. SK hynix and SanDisk").
- Not given: power, thermal, endurance numbers.

### 4.5 Evaluation methodology
Only microbenchmarks and one traced decode step are evaluated.
- **Functional:**
  - 64 request/completion pairs are preserved at 1–16 channels;
  - a shared-staging probe gives 1 page read + 31 MSHR hits;
  - a remapped LUD trace preserves 158 pairs.
- **Closed vs open loop:** 24 configurations (16 pages, 1 or 32 entries, 32/128/4,096 admission slots, 1/15 µs reads, merge on/off), 8 SMs, 4 channels.
- **MQSim cross-check** on read-only service (MQSim revision 51f0f2d).
- **Media scaling:** 1→16 channels, 1,024 producers.
- **Qwen3-1.7B decode:**
  - one decode step, batch 1, BF16, 29-token context;
  - 1,819 kernels traced on an RTX 4090 with the NVBit tracer;
  - weights remapped into HBF, with KV and activations left in GPU memory;
  - 8 SMs; one stack with 16 channels, 512 subarrays and 512 active slots.
- **Case studies:** placement (9 configurations, then 36 runs) and read/write isolation (37 configurations, 36 of which complete).

### 4.6 Headline results
- **Closed-loop vs frozen-ingress replay, MSHR merging benefit at 15 µs:**

  | Admission slots | Closed loop | Frozen ingress |
  |---|---|---|
  | 128 | 61.1% (87,703 → 34,101 cycles) | 0.8% |
  | 32 | 75.8% | 0.1% |
  | 4,096 | 10.0% | 10.0% |

  Replay "understates the benefit of merging under admission pressure".
- **MQSim cross-check:** MQSim windows exceed HBF-Sim's by 32.9% / 32.6% at 1 µs and 2.2% at 15 µs. Both scale as 64·ΔtR/C.
- **Media scaling:**
  - 8.738 → 139.285 GB/s array bandwidth (15.94×) at 99.6–99.99% occupancy;
  - with total slots fixed at 32, it falls back to 8.738 GB/s;
  - GPU sector bandwidth at 16 channels is only 1.088 GB/s, a large read-amplification effect in this microbenchmark (4 B loads, one page per lane).
- **Qwen3-1.7B decode step:** 1,819/1,819 kernels completed; 352,434,530 cycles (249.95 ms); 107,540,992 read requests and completions; 3.441 GB returned = 3.441 GB array reads.
- **Simulator cost:** idle path 6.768 s (HBF off) vs 6.964 s (HBF on); RSS about 58.8–59.0 MB. A 4-stack 512 GiB sparse probe passes.
- **Placement:**
  - contiguous placement cuts page-service amplification by 41.9% (8.0625 → 4.6875) compared with interleave, but kernel time rises 0.69%;
  - active-page striping: 3.94× speedup at stride 32 (1,092,373 → 277,353 cycles), 0.51× at stride 4.
- **Read/write isolation (3R/1W vs shared read priority; 4 written pages, 100 µs delay):** read p95 −27.0% (232,456 → 169,730 cycles), kernel time +17.5%, write p95 4.00×. Which option wins depends on arrival phase.

### 4.7 Stated limitations and future work
- "ECC, read retry, refresh, and thermal effects are not modeled in the current version."
- The evaluation does "not establish cycle-level accuracy against physical HBF silicon". "Hardware calibration remains outside this evaluation."
- Future work:
  - "extend the static policies ... to adaptive placement and channel allocation";
  - "Transaction-level AXI/UCIe modeling";
  - "Hardware measurements ... could calibrate timing parameters".

### 4.8 Critical view
- **Timing is unsourced.** 15/200/2000 µs is not sourced and has no sensitivity study beyond 1 vs 15 µs.
- **Scale is tiny.** One decode step of a 1.7B model on 8 SMs takes 250 ms of simulated time. There are no serving-level, multi-request, prefill, long-context or MoE results. Accel-Sim speed means realistic models (tens to hundreds of billions of parameters) and serving traces are out of reach.
- **Not modelled:** thermal, power, endurance, ECC, read-retry, refresh. There is no GC, by spec assumption, but host-side zone remapping is also not evaluated.
- **The read-amplification microbenchmark** (4 B loads per page) is extreme. TileLens-style layout issues are not examined.
- **Strengths:**
  - faithful OCP semantics (64 B / 4 KiB, non-posted writes, channel affinity);
  - clean policy hooks (mapping table per data class, schedulers, partitioning);
  - request-level CSV traces;
  - retains the DRAM path, so a mixed HBM+HBF address space is natively supported.

### 4.9 BibTeX
```bibtex
@misc{li2026hbfsim,
  title         = {{HBF-Sim}: An Extensible {HBF} Simulator for Large-scale {GPU} Memory Systems},
  author        = {Li, Yaqi and Wang, Jing and Wang, Junfeng and Yang, Long and Yan, Han and Chai, Xiaohu and Shi, Liang},
  year          = {2026},
  eprint        = {2609.29246},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v1, 24 Sep 2026}
}
```
(The citation keys `li2026hbfsucks` and `li2026hbfsim` differ; keep them distinct.)

---

## 5. TileLens

### 5.1 Bibliographic
- **Title:** "TileLens: Efficiently Using Large-Granularity Memory Systems with Transparent Two-Dimensional Memory Layout".
- **Authors:** Jae Hyung Ju*, Euijun Chung*, Hritvik Taneja, Anish Saxena, Shinnung Jeong, Hyesoon Kim, Moinuddin K. Qureshi. All at Georgia Institute of Technology. *equal contribution.
- **Venue:** unknown (arXiv preprint, ACM format).
- **Year / id:** 2026, arXiv:2607.04031v1 [cs.AR], 4 Jul 2026. This predates the OCP spec release of August 2026.

### 5.2 Problem
Large-Granularity Memory Systems (LGMS) such as HBF and RoMe have a minimum access of about 4 KB, against HBM's 32 B. With row-major or column-major weights, each 4 KB access fetches a 1-D strip that extends far beyond the 2-D compute tile (64–512 B wide). The result is:
- **read amplification**, which wastes bandwidth;
- **straggler-induced CTA stalls**, because CTAs sharing a 4 KB request wait for the slowest one.

Tiled matmul, the dominant LLM operation, slows down by up to an order of magnitude as a result.

### 5.3 Mechanism
- **Tile-major layout.** Each contiguous 4 KB block is reshaped into an a×b "memory tile" with a·b·s = 4096, for example 64×64 for FP8 or 64×32 for BF16. The memory tile is decoupled from the compute tile, and must divide it.
  - Address offset (Eq. 2): (T_j·K/a + T_i)·4 KB + (j′·a + i′)·s. Because a and b are powers of two, this needs only shifts and masks.
  - Requires 4 KB-aligned allocation and translation (huge pages), and dimensions divisible by the tile. Recent LLM dimensions are multiples of at least 256; otherwise pad.
- **TileLens-SW.** Extends CuTe so the global layout is a 4-D tensor, shape (a, b, K/a, N/b) and stride (1, a, ab, Kb). DSL kernels (CUTLASS, FlashAttention) change only the layout descriptor.
- **TileLens-HW.**
  - Adds the memory-tile dimensions (a, b) and the leading stride K to the TMA descriptor.
  - A split-and-sum unit classifies coordinate × stride terms using the leading stride. Nested counters replace the stride-K counter, and a bit permutation handles the case a > u.
  - Cost: about **3–4K gates** and about 5–7 cycles. The extra latency is not modelled, on the grounds that TMA 2-D loads already cost about 170 cycles.
  - Non-TMA, non-DSL kernels: edit the address code, or use binary instrumentation.
- **System support for HBF-augmented GPUs:**
  - **weights only in HBF; KV and activations in HBM** (because of endurance), chosen per tensor at allocation time;
  - mixed-granularity L2 (full 4 KB inserts for HBF);
  - HBF MSHRs partitioned at 4 KB granularity, tracking 32× more pages;
  - **adaptive stride prefetcher** in the memory controller, d = min(⌈BL/p_wave⌉, K/TILE_K). The first term is doubled to allow for plane collisions. With the H3 SRAM, near prefetches go to L2 and far prefetches to SRAM.
- **Out of scope:** FTL, GC, wear and writes. Weights are written once at load time.

### 5.4 Device parameters (§2.2, §6.1)
- **HBF (background):** "∼1.6 TB/s per stack for Gen1" [37, Sandisk fact sheet 2025]; 8–16× the capacity per stack; read latency 1–10 µs; granularity "on the order of kilobytes, constrained by factors such as NAND page size and ECC".
- **Simulated HBF:**
  - 16 channels per stack with the **same total bandwidth as HBM** (6 stacks = 4.915 TB/s), at 4 KB granularity;
  - buses run at 3,200 MHz (6.4 Gbps/pin) with 64 pins per channel;
  - tR swept over **1, 2, 5, 10 and 20 µs**, headline 5 µs;
  - internal HBF bandwidth is set to 2.5× bus bandwidth to model concurrent plane access.
- **HBM3e:** 16 channels per stack, 64 B granularity (32 B per pseudo-channel), row-buffer hit/miss 14 / 42 ns, 6 stacks = 4.915 TB/s. Timing comes from H200 and HBM3E specifications.
- **Configurations:** HBM-only; daisy-chain (6 HBF behind 6 HBM, **time-sharing the data bus**); daisy-chain + 40 MB SRAM (H3).
- **RoMe:** 18 channels per stack, 5.53 TB/s, 4 KB.
- **Not given:** capacity, tPROG, endurance, power, thermal.

### 5.5 Evaluation methodology
- **Simulator:** Macsim (cycle-level) with DRAM, RoMe and HBF models. Kernel traces come from a real H200 via an NVBit SASS tracer.
- **Simulated GPU:** 132 SMs at 2.0 GHz, 4 warps per SM (round-robin), 256 KB L1 per SM (30-cycle hit), 50 MB L2 (270-cycle hit).
- **Workloads:**
  - Qwen-3 30B `fused_moe` (BF16 128×256 compute tile);
  - Llama-3.1 70B FFN (64×128 tile);
  - batch 16 / 64 / 256; Llama becomes compute-bound at 256;
  - memory tile 64×32.
- **Scope:** kernel-level only. No end-to-end inference, no attention or KV kernels, no validation of the HBF model against hardware.

### 5.6 Headline results
- **Motivation (Qwen `fused_moe`):** column-major gives 10.1× read amplification, 14.8× stall cycles and 11.2× execution time. Row-major gives 3.9× amplification and 3.3× time.
- **Main result at 5 µs:**
  - column-major is 3–10× slower even with SRAM;
  - row-major has a **1.61× geomean** slowdown;
  - **tile-major + prefetch 1.01×**, and with SRAM 1.00×, so the SRAM is unnecessary.
  - Abstract: "from 1.61–6.49× ... to within 1% of an HBM-only baseline".
  - Without prefetch, tile-major can be worse than row-major, because overfetch acts as implicit prefetch.
- **Effective bandwidth:** column-major 11–26% of peak; row-major 33–92%; tile-major 70–98%.
- **Stragglers:** 128 vs 16 outstanding requests per tile. The 112 extra requests (448 KB) would take about 0.09 µs to transfer, yet latency rises by 3.2 µs. Tile-major cuts per-K-iteration memory latency by 1.5×. Row-major's tail reaches 15 µs.
- **Latency sweep:** tile-major stays near HBM up to 10 µs and degrades to **3–4× at 20 µs**, the prefetcher's coverage limit. Column-major is 8–10× slower at every latency.
- **Tile-shape sweep:** narrower, taller tiles (for example 512×4 vs 64×32) reintroduce amplification and stragglers.
- **RoMe:** row-major 1.02–1.13× and tile-major 1.05–1.12× speedup over HBM-only. There is no straggler effect at DRAM latency.

### 5.7 Stated limitations and future work
- The prefetcher "is not a general-purpose mechanism ... Designing a prefetcher that handles irregular access patterns on LGMS is an interesting direction ... we leave it for future work."
- Assumes aligned allocation and divisible dimensions.
- The paper "does not advocate for any specific LGMS technology".

### 5.8 Critical view
- **The granularity premise conflicts with OCP.** The spec, as described by HBF Sucks, HBFSim and HBF-Sim, gives **64 B host reads** with page buffers and a page cache. So the 4 KB "minimum access granularity" may apply to array sensing, not to bus transfer. If that is right, read amplification costs internal plane bandwidth and energy rather than interface bandwidth. HBF-Sim's Qwen3 run shows array bytes equal returned bytes, with no amplification.
  - TileLens predates the spec, so its conclusions should be re-checked under 64 B reads with page buffers.
  - The straggler argument still holds wherever sensing latency dominates.
- **Bandwidth is generous.** HBF bandwidth is set equal to HBM's (about 0.82 TB/s per stack [derived: 4.915/6]) with internal bandwidth at 2.5× bus. That exceeds OCP Speed Grade 1 (384 GB/s) and HBF Sucks' 202 GB/s thermal point.
- **Only weights are in HBF**, and only decode GEMMs are evaluated. There are no attention, KV, prefill, write, endurance, power or thermal results, and nothing end-to-end or multi-GPU.
- **Latency tail and plane collisions** come from the simulator's queueing model (with an empirical ×2 factor). ECC and read-retry tails are not modelled.
- **The daisy-chain means HBF reads consume HBM bus time.** The effect on concurrent HBM-resident KV reads (attention) is not measured.
- **Relevance:** layout is an orthogonal but necessary piece for any HBF weight placement. The student's roofline model implicitly assumes 100% useful bytes, which TileLens shows needs tile-major layout plus prefetch.

### 5.9 BibTeX
```bibtex
@misc{ju2026tilelens,
  title         = {{TileLens}: Efficiently Using Large-Granularity Memory Systems with Transparent Two-Dimensional Memory Layout},
  author        = {Ju, Jae Hyung and Chung, Euijun and Taneja, Hritvik and Saxena, Anish and Jeong, Shinnung and Kim, Hyesoon and Qureshi, Moinuddin K.},
  year          = {2026},
  eprint        = {2607.04031},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v1, 4 Jul 2026}
}
```

---

## 6. The two simulators: open source and suitability for a hybrid HBM+HBF placement study

| | HBFSim (2609.09800) | HBF-Sim (2609.29246) | For reference: TokenSim-hbf (from HBF Sucks) |
|---|---|---|---|
| Open source? | Claimed; **no URL in paper** | Claimed; **no URL in paper** | **Yes**: github.com/pku-lemonade/TokenSim/tree/hbf |
| GPU | Real GPU (RTX PRO 6000 Blackwell, CUDA 13, PTX rewrite, bpftime) | Simulated (Accel-Sim / GPGPU-Sim) | None (serving-level analytical/event model) |
| HBF media | Online MQSim + calibrated NVMe curve; thermal (3D-ICE ROM); retention/refresh/read-disturb | Own page/channel/subarray state machines; MSHR merge; write buffer; no thermal, ECC or refresh | Latency/bandwidth tier model; offline 3D-ICE thermal; endurance budget |
| HBM:HBF split | First-class (register tensors as HBF ranges; capacity mode) | DRAM path retained; explicit mapping table per data class | Tier capacities and bandwidths per profile |
| Scale | Real vLLM model, but only 16 KB of one tensor modelled | One decode step of Qwen3-1.7B on 8 SMs takes about 250 ms simulated | Full 2-hour traces, 8 GPUs, 5 models up to 1.1T |
| Hardware needed | A recent NVIDIA GPU with PTX-rewritable kernels | CPU only | CPU only |

**Recommendation.**
1. **To evaluate a hybrid HBM+HBF *placement policy* at serving level** (batch size and concurrency, KV and weight residency, admission), neither of the two HBF simulators fits well.
   - The most practical base is the student's existing analytical LLMSimulator extension.
   - Alternatively, use **TokenSim's `hbf` branch**, the only one with a public URL. It already has HBF tiers, per-tier byte counters and production traces. Its runtime is SSD-style save/restore, so an in-place read path and an H3-style cascade would need to be added.
2. **Of the two named HBF simulators, HBF-Sim is the easier one for a student to extend:**
   - C++ module in `hbf/`;
   - no special GPU needed;
   - an explicit mapping-table hook that already distinguishes weights from KV pages;
   - read, write and scheduler policies selectable by configuration;
   - CSV traces;
   - the DRAM path is kept, so a mixed HBM+HBF address space is native.

   Use it for **micro-validation** of per-access costs your roofline model abstracts away: read/write interference, page-service amplification, and channel or subarray placement. Do not use it for end-to-end serving, because Accel-Sim speed rules that out.
3. **HBFSim is the more realistic tool for "which tensors live in HBM vs HBF"** under real vLLM execution, with thermal and refresh coupling.
   - It needs matching GPU hardware and CUDA 13, and its coverage is currently tiny (16 KB of one tensor).
   - There is also no public code link yet.
   - It is higher risk for a BTP timeline.

---

## 7. Cross-paper observations

### 7.1 Contradictions in assumptions
1. **NAND timing varies by 5× or more with no common source.** The OCP spec defers all timing to vendor datasheets, according to HBFSim §3.

   | Paper | tR | tPROG | Other | Basis |
   |---|---|---|---|---|
   | HBFlex | 4 µs | 75 µs | — | Via FlashAccel; originates in a 1-bit/cell 96-layer die (Kouchi JSSC 2021, cited by HBFSim [20]) |
   | TileLens | sweep 1–20 µs, headline 5 µs | — | — | — |
   | HBF Sucks? | sweep 8/80 to 30/300 µs (read/write); 12/120 µs for NMP | — | — | — |
   | HBF-Sim | 15 µs | 200 µs | tBERS 2 ms | Unsourced |
   | HBFSim | ∼20 µs | — | — | Cited to TileLens, which says 1–10 µs: probable mis-citation |

2. **Per-stack bandwidth assumptions range roughly 20×:**
   - HBFlex: 488 GB/s read, 27.2 GB/s write;
   - HBFSim: 384 GB/s (OCP Speed Grade 1, 8-hi);
   - HBF-Sim: 3.072 TB/s interface (16 channels, SG3 at 16-hi), but its own media model gives only ≈139 GB/s array bandwidth at 15 µs with 512 subarrays;
   - TileLens: ≈0.82 TB/s per stack, set equal to HBM [derived], and cites ∼1.6 TB/s Gen1 from the Sandisk fact sheet;
   - HBF Sucks: the Table 3(c) figures imply ≈25–200 GB/s per stack [derived, ambiguous].
3. **Endurance (P/E) budget differs about 30×, and it flips the verdict on KV-in-HBF:**
   - HBFlex assumes 100K P/E (unsourced, SLC-class) and concludes raw KV writes use only 15–42% of budget.
   - HBF Sucks assumes a TLC budget (≈3.3K P/E [derived]) and concludes HBF wears out in 0.56× of SSD life.
   - The per-GPU write *rates* are similar in magnitude: HBFlex 8.4–22.7 TB/day per GPU; HBF Sucks 48–140 TB/day per 8-GPU system [derived: ≈6–17.5 TB/day per GPU].
   - **[BTP]** The student's 995-year retention-refresh result and GC analysis assume SLC with 100K P/E, matching H3 and HBFlex. Under a TLC budget, lifetime scales down by about 30× [derived: roughly 33 years for the same refresh load]. A sensitivity line should be added.
4. **Thermal verdicts disagree:**
   - HBF Sucks: a 16-Hi TLC stack throttles at 202.27 GB/s and 53.72 W against an 80 °C limit the authors chose, with write-heavy traffic.
   - HBFSim: an 8-hi stack sustains 384 GB/s with no throttling (peak 79.8 °C), and the package limit falls at the compute die at 86.6 W, against a 105 °C HBF limit taken from the spec. Its traffic is read-only.
   - Energy per byte differs: HBFSim 0.084 nJ/B for reads; HBF Sucks' Fig. 10 slope ≈0.26 nJ/B [derived: 106 W / 400 GB/s].
   - Power per cube in H3 is ≈160 W (quoted by HBF Sucks), far above both.
   - **[BTP]** The student's "4× power per stack" follows H3 and is disputed by both thermal models.
5. **What belongs in HBF:**
   - HBF Sucks: transient or private KV is the worst fit; weights and shared prefixes are the best.
   - HBFlex: all KV in HBF, and it beats hybrids.
   - TileLens: weights only.
   - These are reconciled by the *access path*. HBF Sucks restores KV to the near tier through an SSD-style connector; HBFlex reads KV in place with plane-aware attention. They also differ in tR/tPROG (4/75 vs 8/80–30/300 µs) and P/E (100K vs TLC). HBFlex cites HBF Sucks [16] but does not address its C2/C3 findings directly.
6. **Host access granularity:**
   - TileLens assumes 4 KB is the minimum access, including on the bus.
   - OCP, as reported by HBF Sucks, HBFSim and HBF-Sim, gives 64 B host reads with two page buffers per bank or subarray.
   - HBFlex: "host interface may return the requested portion of a buffered page".
   - The read-amplification cost is on the array side, not the bus side, under OCP. TileLens's bus-bandwidth-waste numbers may overstate the penalty. Its straggler and sensing-concurrency argument remains valid.
7. **GC ownership:**
   - HBF-Sim and HBF Sucks (citing OCP): no device GC; reclamation and zone remapping are host-managed; wear levelling is base-die or host.
   - HBFlex models GC migration with WAF up to 30× but does not frame it under the OCP host/device split.
   - **[BTP]** The student notes static wear levelling is "a policy the architecture does not specify". OCP does allocate it (base die or host) and exposes average P/E to the host. Cite this.
8. **Spec dating:** HBF Sucks and HBFSim give OCP HBF v0.7.0 as August 2026 (HBF Sucks: "August 3, 2026"); HBF-Sim gives 2026; HBFlex's bibliography says 2023, which is probably an error; TileLens (July 2026) predates it. Retention: OCP gives "24-hour powered-on retention at 85 °C" (HBF Sucks), and HBFSim's table says unpowered retention is "not guaranteed". **[BTP]** The student's daily-refresh assumption is consistent with this, but the refresh *bandwidth contention*, which HBFSim models, is not in the student's model.
9. **HBM baseline per stack:** HBFlex 24 GB / 0.8 TB/s (HBM3e 8-Hi); HBF Sucks 16 GB per HBM stack (HBF-2); HBFSim 36 GiB (spec example) and HBM4 64 GB / ∼2 TB/s; TileLens ≈0.82 TB/s per stack. Capacity ratios quoted: 21.3× (HBF Sucks), "at least 8×" (HBFSim/Sandisk), 8–16× (TileLens).
10. **Package normalisation:** each paper uses a different metric:
    - HBFlex: ¼ HBM, ¾ HBF, bandwidth scaled;
    - HBF Sucks: site counts, half and half;
    - HBFSim: 861.1 mm² area, 2-D fit only.

    None models PHY shoreline, UCIe beachfront or power delivery. HBFSim says explicitly that area fit "settles nothing about routing, power delivery, PHY placement".

### 7.2 Design points nobody explored
- **A hybrid in which HBM is kept *and* HBF serves KV in place, with reuse-aware, dynamic migration between tiers.**
  - HBF Sucks calls for reuse-aware admission, write budgeting and thermal coordination but does not build them.
  - HBFlex removes HBM entirely; H3 and TileLens keep KV out of HBF.
  - This is the natural BTP niche: use HBF Sucks' ρ > ρ★ test (Eq. 8) as the placement rule, per object class.
- **H3-style cascade (HBF behind the HBM base die, no loss of HBM sites) under production traces.** HBF Sucks tests only HBF-1 and HBF-2; HBFlex handicaps H3 by area normalisation.
- **Mixed dies within one stack** (the student's 12H4F) is only touched as the "IS" baseline in HBFlex. No simulator here models it.
- **Tile-major layout (TileLens) for KV and attention, combined with plane-striped KV pages (HBFlex).** Nobody combines layout with KV paging.
- **Write-heavy thermal behaviour in a package-coupled, transient model.** HBF Sucks has write-heavy but steady-state single-stack; HBFSim has package-coupled but read-only.
- **Prefill-heavy and long-context (≥128K) workloads.** TileLens covers decode GEMMs only; HBF Sucks' traces have short mean prompts; HBFlex relies on prefill windows but doesn't sweep context.
- **Multi-GPU.** No TP/EP scaling study with HBF: HBFlex is DP4/TP2 on one package; HBF Sucks is TP8 but fixed.
- **Multi-tenant or multi-model interference** on shared HBF channels.
- **MoE expert placement in HBF with routing-aware prefetch into timing.** HBFSim measures deduplicated demand (2.487× / 3.584×) but not timing; TileLens covers one `fused_moe` kernel.

### 7.3 Research gaps (common to all five)
- **No hardware validation anywhere.** Every HBF number is modelled; HBFSim and HBF-Sim say so explicitly.
- **ECC, read-retry and their tail latency:** none of the five models them (HBF-Sim says so explicitly; TileLens's "long tail" is queueing only).
- **Read disturb and retention refresh cost:** implemented only in HBFSim, with no reported numbers. Weights are read billions of times, so read-disturb refresh for "read-only" data is an open question. **[BTP]** This matters for the student's "read-only means no writes" argument.
- **TCO in dollars per token:** none quantify it. HBF Sucks expresses cost as package-resource trade-offs; HBFSim cites Sandisk's "8× capacity at same price".
- **Tail latency (p99 TPOT/TTFT) under GC or refresh:** HBFlex reports means; HBF Sucks reports goodput; HBF-Sim reports p95 only in microbenchmarks.
- **Wear-levelling policy (static vs dynamic) and wear concentration:** none of the five evaluate it. **[BTP]** The student's 10.7× wear-concentration finding appears to be novel relative to this set.
