# HBF Systems Papers — Reading Notes (Batch B1)

Papers covered: **FlashAccel** (ICT-CAS), **FLINT** (Huawei / ETH Zürich), **DASH** (KAIST).
All three were read in full from the text extractions in `scratchpad/txt/`. Numbers are copied from the papers. Where I did my own arithmetic or inference, the note is marked **[my calc]** or **[my inference]**.

Context from the student's report (`BTP_Report.tex`): it evaluates H³ (Ha, Kim, Kim, IEEE CAL 2026). Its claims: flash is over-provisioned 17–67× for 8 of 9 models; 12H4F gives 1.75× concurrency at 21% lower power; pooling a memory link beats partitioning it; read-only placement makes GC structurally absent at H³ depth; wear concentration (static wear levelling) changes lifetime by up to 10.7×; "retention refresh is negligible at 995 years". Where these papers bear on those claims, I point it out, especially FLINT's read-disturb result.

---

## 1. FlashAccel

### 1.1 Bibliographic
- **Title:** "FlashAccel: Leveraging High-Bandwidth Flash for High-Throughput LLM Inference"
- **Authors:** Xinyu Wang, Yalong Xue, Xiaotian Sun, Xiaoyu Zhang, Chunmeng Dou, Xueqi Li, Xiaoming Chen (corresponding author). Affiliations: Institute of Computing Technology, CAS; University of Chinese Academy of Sciences; Institute of Microelectronics, CAS (Dou).
- **Venue:** unknown. The paper uses an ACM-style template and the running header has no venue. The only mention of ASPLOS is an example question in Fig. 2 ("Where will ASPLOS 27 be held?"), which does not count as a venue statement.
- **Year:** 2026 (arXiv v1 dated 11 Jul 2026)
- **arXiv:** 2607.10186v1 [cs.AR]

### 1.2 Problem
HBM capacity limits decode batch size, forces early KV-cache eviction (so multi-turn reuse is lost and recomputation goes up), and pushes deployments to multiple GPUs. That adds cost, collective communication ("up to 20% of total latency" [20]) and failures ("every 7.9 hours" in a 1024-GPU system [31]). HBF adds capacity, but three things stop it from turning into throughput: (i) high access latency (tR ≈ 4 µs against about 100 ns for HBM, a 40× gap); (ii) low effective bandwidth unless thousands of planes are kept busy; (iii) no system software for persistent, heterogeneous HBM/HBF/SRAM resources.

### 1.3 Core mechanism / architecture
- **Attachment.** Two options (Fig. 6a):
  - **CLI (co-located integration)** [46 = SanDisk blog]: HBF sits next to the GPU and *replaces* HBM stacks. In the evaluation, 5 of 6 HBM stacks are replaced and 1 is kept.
  - **CSI (cascaded integration)** [23 = H³]: HBF is daisy-chained through the HBM base die. In the evaluation, 6 HBF stacks sit alongside the 6 HBM stacks of an H200, and "each HBM stack is reduced to 2 DRAM layers to minimize cost". The resulting HBM capacity is **not stated**.
  - Section 3.3 argues that "treating Flash as a lower tier behind HBM becomes increasingly inefficient because it forces unnecessary staging through HBM. Instead, HBF should be exposed as a directly accessible resource for GPUs". The GPU reads HBF directly through a unified virtual address space, even in CSI.
- **Placement.** HBM holds "small, frequently updated intermediate data". HBF holds **both model weights and KV cache**, which is unusual: FLINT and H³ keep KV out of flash. New decode KV is buffered in HBM and flushed to HBF in full hyper-pages. Overloaded-plane KV pages are *offloaded* (copied) to HBM for rebalancing.
- **HBF stack.** 8 flash dies plus a base die with TSVs. Each flash die is an array die hybrid-bonded to a circuit die, scaled to **96 planes/die** (following Lincoln [52] and SanDisk [46]). Cells are SLC, chosen "for its lower read latency and higher write endurance".
- **Latency hiding (HW + runtime).**
  - SRAM goes into unused circuit-die area: **32 KB per plane**, plus **8 MB on the base die**, for 32 MB per stack. The stack SRAM is sized "> 2 × Peak_Bandwidth × Read_Latency" to allow double buffering.
  - Runtime API: `SramPrefetch`/`SramRelease`. Prefetches are pulled from the compute graph, queued in topological order and run asynchronously. Kernels that start before the prefetch completes read from the flash address; the page table is updated to point at SRAM once data arrives.
- **Data layout (compiler/offline + runtime).**
  - *Hyper page* = one page from every plane. It is about **19 MB** for a 4.8 TB/s HBF [their number]. A single Qwen3-235B weight matrix is only 12 MB, so it cannot fill a hyper page.
  - *Weights:* split into page-sized units and placed round-robin across all planes and channels **in execution order**. The order is fixed offline from the compute graph.
  - *KV cache:* modelled as balls-into-bins. For 100 GB of Qwen3-235B KV over 4,916 planes, the max plane load is 52% above average, which means 52% bandwidth loss. Mitigations:
    1. Aggregate non-resident KV into full hyper pages before writing; leftovers go to random planes.
    2. **Selective offloading:** excess pages on overloaded planes are copied to HBM until no plane exceeds the average (`GroupArrange`).
    3. HBF-friendly attention reads at hyper-page granularity, with partial FlashAttention results reduced per request.
    4. Decode KV is buffered in HBM and flushed once full hyper pages form (`GroupWrite`).
    5. **256 KB KV blocks.** 512 KB would need at least 8 GB of HBM at batch 256 for Qwen3-235B, which is too much pressure for CLI.
- **Storage management / FTL.** The FTL is **eliminated**. Their key observation is that "both model weights and KV cache follow append-only write patterns". Physical addresses are stored directly in per-object indexing metadata, which is persisted in flash. **Isolated block allocation** gives weights and each request's KV their own blocks, so erasing one request's KV causes no write amplification. The paper describes no GC, wear levelling, read-disturb handling or refresh.
- **Programming model.** `NandMmap` maps weights. `GroupCreate`/`GroupMmap`/`GroupWrite`/`GroupArrange` handle the per-step "group object" of all active requests' KV.

### 1.4 Device parameters (Table 2 and text)
| Parameter | Value | Source cited |
|---|---|---|
| Cell type | SLC, 96 word-line layers | based on XL-Flash [32] (Kouchi et al., JSSC 2020, "128Gb 1-bit/cell 96-WL-layer ... tprog=75µs and tr=4µs") |
| Page size | 4 KB | Table 2 |
| Pages/block, blocks/plane, planes/die | 256 / 256 / 96 | Table 2 |
| Plane capacity | 256 MB. They keep tR and tProg the same as [32] "even though the capacity of each plane is reduced by 4×" | text |
| Die capacity | 192 Gb | Table 2 |
| tR / tProg | 4 µs / 75 µs | [32] |
| Per-plane read BW | "a plane reads a 4KB page in 4 µs, 1 GB/s" | text |
| Stack | 8 flash dies + 1 base die @ **768 GB/s**; **192 GB** flash; 32 MB SRAM | Table 2 |
| Area | Array die 149 mm² (plane 1.43 mm², TSV 12 mm²); circuit die 143 mm²; HBF 149 mm² vs HBM3e 121 mm² (1.23×); "6.5× higher density than HBM3e" | derived from the die shot in [32] |
| Endurance | SLC "about 100K P/E cycles" [30,39,59]. **Assumed 1M P/E** via retention relaxation (a conservative 10×, where [12,37] report "up to 50×" for 3-year to 3-day retention) | text |
| System write BW | CSI peak write **245.8 GB/s** against read **4.6 TB/s** | §7.4 |
| Read energy | **8 pJ/bit** from "a real hybrid-bonded Flash prototype" [62], compared with HBM3e **2.99 pJ/bit** [4] | §7.6 |
| HBM latency | ~100 ns [27] | §3.1 |
| Baseline HBM | H200: 6 HBM3e stacks, 141 GB, 4.8 TB/s | [38,41] |

[my calc] Checks: 8 dies × 96 planes × 1 GB/s ≈ 768 GB/s per stack; 6 stacks give about 4.6 TB/s and 1,152 GB, matching the stated "1152GB capacity". 4,608 planes × 4 KB / 75 µs ≈ 246 GiB/s, matching 245.8 GB/s. Also, ref [62] is "A 1Tb **3b/Cell** 3D-Flash Memory ..." (ISSCC 2025). The 8 pJ/bit read-energy figure therefore comes from a **TLC** chip but is applied to an **SLC** design.

### 1.5 Evaluation methodology
- **Simulator:** event-driven, built on **LLMCompass** [69], which searches GEMM tilings and uses ScaleSim for latency. It is extended with a NAND simulator that models page access latency at plane granularity.
- **Models (Table 1):** Qwen3-235B (GQA/DP, 188 KB/token, MoE/EP); Qwen3-Coder-480B (GQA/DP, 248 KB/token, MoE/EP); LLaMA3.1-405B (GQA/TP, 502 KB/token, dense/TP); DeepSeek-V3-671B (MLA/DP, 70 KB/token, MoE/EP). All FP16 except DeepSeek-V3 weights, which are FP8.
- **Sequence lengths:** long-context 8.11K in / 2.53K out (LongProc averages); agentic 15K in / 6K out (from [58]).
- **SLOs:** decode latency of 50 ms and 100 ms. For each configuration they use the largest batch that meets the SLO and fits in capacity. Ablation uses batch 256.
- **Baseline:** DGX-H200, 8 GPUs, 900 GB/s NVLink. Configurations: 8×H200, 8×CSI, 8×CLI, 4×CSI, 4×CLI, and 16-GPU variants that fall back to RDMA with "about 9×" lower bandwidth.
- **KV hit-rate study:** multi-turn traces from pi-mono [6].

### 1.6 Headline results
- **2.54× throughput/GPU and 1.93× energy efficiency** over HBM-only under a 100 ms SLO (abstract, six HBF stacks). In the body, 8×CSI averages **2.15×** "across all models, sequence lengths, and SLOs" and 8×CLI reaches **2.04×** at 100 ms. CLI has 5 HBF stacks and therefore 16.7% less bandwidth; it "slightly underperform[s] the baseline" on DeepSeek-V3 and Qwen3-235B at 50 ms.
- Example: LLaMA3.1-405B at 15K/6K needs 10 GB of KV per request. 8×H200 tops out at batch 30, while 8×CSI supports a theoretical batch of 110 under the 50 ms SLO.
- 4×CSI/CLI: LLaMA-405B and Qwen3-480B cannot meet 50 ms because "loading weights alone takes close to 50ms". At 100 ms they still beat 8×H200 per GPU on LLaMA-405B, Qwen3-235B and Qwen3-480B.
- 8× HBF-GPU gives higher throughput/GPU than 16× HBM-GPU, because the 16-GPU system has to use RDMA.
- **Ablation** (Qwen3-235B, batch 256, 100 ms): the full design is within about 4% of HBM latency (including 2% offload overhead). Removing prefetch costs −55% throughput, weight layout −7%, KV layout −15%. Plain HBF is −65% and lands below the HBM GPU.
- **Writes:** worst case 276 MB/s of decode KV (8×CSI, Qwen3-480B, 100 ms, 1138 tok/s) plus 712 MB/s from prefill, for **988 MB/s**. That takes 3.9 ms of each second (4% overhead). Over 5 years this is **148,570 TB**, against CSI's **1,125,000 TBW** (1152 GB × 1M P/E).
- **Prefill/KV reuse:** 8×CLI/CSI achieve the ideal hit rate. 16×H200 is still 50% lower. Tokens needing computation drop by up to **89%**.
- **Energy/TDP:** TDP rises 1.31× (CSI) and 1.23× (CLI). Tokens/J is 1.93× (CSI) and 1.66× (CLI).

### 1.7 Stated limitations and future work
- The paper has no explicit limitations or future-work section. The conclusion only summarises.
- Implicit admissions:
  - "it still cannot match the nearly unlimited write endurance of DRAM"
  - Endurance relaxation is assumed "Conservatively ... up to 10×"
  - The area overhead "stems from our conservative assumptions based on the design in [32], which uses only 96 wordline layers", and "Using array dies with more wordline layers allows HBF to achieve area parity"
  - The KV block size "cannot be arbitrarily large"
  - CLI "slightly underperform[s]" at 50 ms

### 1.8 Critical view
1. **Retention relaxation conflicts with persistence of weights.** They assume 1M P/E by cutting retention from years to days ([12,37]). The same device also holds *weights* as "persistent objects", and the storage layer is designed for persistence. Short retention would need periodic rewrites of weights, which is itself write traffic and wear. This interaction is not discussed.
2. **Read disturb is ignored entirely.** Weight pages are read every decode step. FLINT (below) shows that SLC read-disturb (about 10⁶ reads/block) forces refresh of always-on blocks and wears them out in weeks at 10⁵ P/E. FlashAccel has no read counters, refresh or ECC design.
3. **The endurance maths is aggregate, not per block.** 1,125,000 TBW assumes perfectly uniform wear. The paper has no wear levelling, and its "random placement" of residual blocks plus per-request isolated blocks does not guarantee it. The student's report shows that wear concentration changes lifetime by up to 10.7×. Hot/cold separation (static weights never erased versus KV blocks cycling) will concentrate wear on the KV region. [my inference]
4. **GC is not analysed.** Isolated per-request blocks avoid write amplification only if KV blocks never share an erase block. With 256 KB KV blocks against a 4 KB × 256-page = 1 MB erase block [my calc from Table 2], one erase block holds four KV blocks. Unless all four belong to the same request, partial-block invalidation forces relocation. The paper claims isolation but does not reconcile these granularities.
5. **Write interference with reads.** The 4% overhead is time-share only. A 75 µs program occupies a plane, and in a design whose whole premise is balancing the *max plane load*, programs on some planes create read stragglers. Tail latency from this is not evaluated.
6. **MoE prefetch is not addressed.** The prefetch queue follows the static topological order of the compute graph. For MoE models (three of the four models) the routed experts are not known ahead of time. FLINT shows static layer-ahead prefetch wastes 86–96% of HBF traffic on MoE. The paper does not say how expert weights are prefetched under EP; it may prefetch all experts or only local ones.
7. **Energy parameter borrowed from TLC.** 8 pJ/bit comes from a TLC ISSCC chip, while FLINT uses 20 pJ/bit (swept 10–25). Energy-efficiency conclusions (1.93×) are sensitive to this, and no sweep is given.
8. **CSI HBM capacity is never stated** (the "2 DRAM layers" stacks). CSI bandwidth is quoted as 4.6 TB/s, the HBF-only figure. That suggests HBF traffic and HBM traffic share the same GPU-facing shoreline, but whether HBM bandwidth adds to it is unspecified. This matters directly for the student's "pooling vs partitioning" finding.
9. **Not evaluated:** thermal (8 flash dies plus SRAM plus TDP up 1.31×), tail latency (only mean SLO), multi-tenant/multi-model, ECC/BER, bad blocks, power-loss consistency of the metadata-in-flash scheme, the cost of balls-into-bins rebalancing each step when the batch changes (only "2%" is quoted), and cost/TCO (claimed "cost efficiency" without numbers).
10. **Prefill is only analysed for the KV hit rate.** The prefill node's HBF write rate (712 MB/s) is *derived* from an input/output ratio rather than simulated.

### 1.9 BibTeX
```bibtex
@misc{wang2026flashaccel,
  title         = {{FlashAccel}: Leveraging High-Bandwidth Flash for High-Throughput {LLM} Inference},
  author        = {Wang, Xinyu and Xue, Yalong and Sun, Xiaotian and Zhang, Xiaoyu and Dou, Chunmeng and Li, Xueqi and Chen, Xiaoming},
  year          = {2026},
  eprint        = {2607.10186},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv:2607.10186v1}
}
```

---

## 2. FLINT

### 2.1 Bibliographic
- **Title:** "FLINT: Efficiently Leveraging High Bandwidth Flash for Capacity-Scalable LLM Inference Acceleration"
- **Authors (in order):** Geraldo F. Oliveira*, Arash Tavakkol* (co-primary), Xiangyu Zhu, Ahmet Caner Yüzügüler, Vamanan Arulchelvan, Lukas Cavigelli, Renzo Andri, Mohammad Sadrosadati, Jia Xinglei, Onur Mutlu, Zhou Ke, Shai Bergman, Ji Zhang. Affiliations: Huawei Technologies Switzerland AG; Huawei Technologies Co., Ltd.; ETH Zürich (Mutlu); HUST (Zhou Ke). Names are given exactly as printed; "Jia Xinglei" and "Zhou Ke" appear family-name-first.
- **Venue:** unknown
- **Year:** 2026 (arXiv v1 dated 25 Aug 2026)
- **arXiv:** 2608.25062v1 [cs.AR]

### 2.2 Problem
The target is **single-accelerator and small-node** inference, where adding GPU packages only for capacity is impractical. HBF is used as a **read-only weight tier** daisy-chained behind HBM. Prior HBF designs (H³ [34], KAIST presentations [35,36]) have three problems:
- (i) They rely on "coarse-grained static prefetching" into an SRAM staging buffer (the LHB), which cannot track runtime access order, particularly MoE routing.
- (ii) They "expose NAND flash management tasks (e.g., refresh operations) to the accelerator-visible critical inference path".
- (iii) They import SSD-class FTL machinery for data that is never rewritten.

### 2.3 Core mechanism / architecture
- **Attachment.** GPU (xPU) → D2D → HBM base die → D2D → HBF base die, "daisy-chaining both tiers behind a single xPU-facing shoreline budget". A unified physical address space: the HBM base-die address decoder routes LLC misses either to HBM banks or across the HBM→HBF D2D link.
- **Placement.** HBM holds the KV cache, activations and runtime metadata. HBF holds **weights only**, preloaded at deployment and read-only during inference.
- **Mechanism 1: burst-buffer controller (hardware, HBF base die).** Parts: cache request queue, cl-to-page mapping table and burst scheduler (e.g., FR-FCFS).
  - It coalesces fine-grained LLC-miss requests into a **burst**: the same (block, page) sensed across all 512 planes of a stack, giving **2 MB = BDP (1 TB/s × 2 µs)**.
  - It pipelines using the **existing per-plane page buffer and cache buffer**. The next burst is sensed while the current one drains, and a lookahead window issues the next burst.
  - There is **no dedicated SRAM LHB** and **no compiler prefetch hints**.
- **Mechanism 2: phantom-plane refresh.**
  - Each die gets **N+1 physical planes** for N logical planes. One plane at a time is offline as the "phantom".
  - Per-block read counters trip at the read-disturb threshold. Trip points are staggered so blocks come due as a trickle rather than all at once.
  - Pages read in the foreground pass through the ECC engine and are "forked": one copy goes to the accelerator, one to a program queue into the phantom plane. A migration bitmap deduplicates them, and a scrubber covers cold pages.
  - When a source plane is fully copied, the block relocation table is updated, the vacated plane is erased, and the phantom role rotates round-robin.
  - Rotation step **T_rot ≥ C_plane/BW_prog = 12.5 s**. A full pass over the N+1 planes takes "about seven minutes, four orders of magnitude inside the SLC retention margin".
  - Hot-block refresh demand is at most **0.86 GB/s per stack**, against **1.28 GB/s** of program capacity from 16 phantom planes.
  - The paper describes wear levelling as "provided implicitly by phantom-plane refresh".
- **Mechanism 3: read-only FTL.** A burst translation table maps each logical burst to a physical (block, page), plus a block-granular relocation table and one read counter per block. For a 512 GB stack with 2 MB bursts that is 256K entries (about 1 MB), **1.8 MB per stack** in total. The paper says this is "512× fewer than a conventional page-level SSD FTL". No GC or out-of-place updates.
- **Write path.** The model is written once at deployment with a fixed striping function: block(b) = ⌊b/1024⌋, page(b) = b mod 1024, and slice s goes to die ⌊s/32⌋, plane s mod 32.
- **Power loss.** Resident weights are *not* preserved. "FLINT treats every power-up as a clean installation", estimated at t_erase + t_reload ≈ **6.7 min**.

### 2.4 Device parameters
| Parameter | Value | Source cited |
|---|---|---|
| HBM | 12–16 DRAM dies, 24–36 GB, 1.2–2 TB/s per stack [23 JEDEC]. Simulated: **HBM3e 192 GB, 8 × 1 TB/s stacks**, calibrated 0.70 sustained/peak (Ramulator 2.0 gives 67.8–69.9%), row hit/miss 70/105 ns, **20 GB KV reserve** | Table 2 |
| HBF (background) | e.g. 16 dies over a base die, "512–1024 GB of capacity and 1.6–2 TB/s of aggregate NAND bandwidth per stack" | [33 SanDisk, 34 H³] |
| HBF simulated | **Gen-1 SLC, 4 TB total = 8 × (512 GB, 1 TB/s) stacks**; 16 dies/stack, **32 planes/die**, 256 blocks/plane, 1024 × **4 kB** pages/block | Table 2, [33,34] |
| tR / tPROG / tERASE | **2 µs / 50 µs / 3 ms**. Background text gives tR of "1–2 µs in modern SLC NAND" | [33,35,65,66], [51] |
| Per-plane BW | read ≈ 4 KB/2 µs ≈ 2 GB/s; program ≈ 4 KB/50 µs ≈ 80 MB/s | footnote 1 |
| Burst | 2 MB (512 page-senses) | Table 2 |
| Reliability | read-disturb **1×10⁶ senses/block**; **1×10⁵ P/E/block**; ">1yr end-of-life retention"; in-place block refresh **54.2 ms** (= 3 ms + 1024 × 50 µs, excluding read-out) | Table 2, [34,51] |
| Refresh trigger | "roughly 10⁵–10⁶ reads to that block due to read-disturb, or after several years due to retention loss" | [51,52] |
| D2D | 1 TB/s per stack [41 UCIe] | Table 2 |
| Energy (pJ/bit) | HBM 4; **HBF read 20 (swept 10–25)**; HBF program 100; D2D 1; NVLink 1.3; SSD 100; ALU 0.32 pJ/B; GPU 1000 W TDP / 400 W sync spin / 100 W idle; LHB leakage 50 mW | Table 2, [71 AttAcc] |
| GPU | B200: 148 SMs @ 2.5 GHz, 11,000 fp16 MACs/SM/cycle, MFU 0.5, L2 126 MB @ 21 TB/s | Table 2 |
| SSD baseline | PCIe Gen5 ×4, 14 GB/s, 40 µs read | Table 2 |
| Area | phantom plane +3.1% HBF die; base die 14.3 mm² at 22 nm, **3.9 mm² at 7 nm** (a page-level FTL table of 512 MB would be 180 mm² at 7 nm) | Table 3, CACTI |

[my calc] 512 planes × 4 kB / 2 µs ≈ 1.05 TB/s, consistent with 1 TB/s per stack. Plane capacity is 256 × 1024 × 4 kB = 1 GB, and 1 GB / 80 MB/s = 12.5 s, consistent with T_rot.

### 2.5 Evaluation methodology
- **Simulator:** in-house, trace-driven. It covers five tiers (SMs, L2, HBM, staging buffers, HBF) and consumes per-SM event streams from Huawei's **WSE workload generator** [69]. Compute is analytical: peak × utilisation. Multi-package runs charge every cross-package transfer and a per-layer barrier. The HBM model is calibrated against Ramulator 2.0.
- **Models:** DeepSeek-V3, DeepSeek-V4-Pro, Qwen3-235B-A22B, Llama-4 Maverick and Kimi K2 (MoE), plus Llama-3.1-405B (dense). Native precision throughout. Routed-expert distributions were captured from a real GPU deployment serving MMLU prompts.
- **Batch sizes** {1, 4, 16, 64}; **contexts** {2, 8, 32, 128}K. Headline results are at 128K. **Decode only.**
- **Baselines:**
  - (i) HBM+SSD: 1 GPU, spilling to NVMe.
  - (ii) HBM-only min-fit: fewest B200s that hold the weights. Cells where KV overflows are marked infeasible rather than re-sharded.
  - (iii) H³: 1 GPU, 2 LHB slots with FIFO eviction and compiler layer-ahead hints, issued only where the address is knowable.
  - Also a "capacity-lifted oracle".
- **SLO:** 50 ms TPOT.

### 2.6 Headline results
- **Motivation (min-fit HBM-only, 128K):**
  - Communication plus synchronisation is 52–85% of per-token time at bsz 1–4 and 22–61% at bsz 64. Barrier waits alone are 51–79%.
  - The busiest package reads 1.4–4× the mean routed-expert bytes (1.1× at bsz 64).
  - Compute is 0.1–1.9% of time.
  - The capacity constraint over-provisions packages by 4.6× on average at bsz 1 and 1.9× at bsz 64.
- **H³ static prefetch** wastes **86–96%** of HBF traffic on MoE models. An oracle prefetcher changes serving throughput by at most 3% and *hurts* DeepSeek-V3 (0.83–0.92×). Re-fetches are "too late" at bsz 1 and "thrash" at bsz 64.
- **Refresh in-place:**
  - Burst refresh stalls 26 s to 22 min whenever it triggers.
  - Distributed refresh inserts 0.7 s per decode step on average (0.09–1.8 s).
  - In Fig. 13, in-place refresh costs 20× on average (13–24×). The burst-refresh tail is 3,700× (303–6,771×) and its average is 4.8×.
- **FLINT:**
  - Consumes 90–97% of fetched HBF traffic and sustains 1.9–3.6 TB/s of *useful* bandwidth on MoE models, **6.2×** H³ on average (4.0–14.3×).
  - Dense Llama is the boundary case: both deliver 2.6 TB/s.
- **Decode throughput per GPU:** **1,205× / 2.2× / 6.2×** over HBM+SSD / HBM-only / H³ (2.2× over HBM-only is 1.5–3.1× by model, up to 3.7× at bsz 1). FLINT+R (with refresh) gives throughput *identical* to FLINT.
- **Energy:** **408× / 1.1× / 6.8×** reduction. MoE models use 0.72–0.90× of HBM-only energy (0.45–0.73× at bsz 1). **Dense Llama uses 1.64× *more* energy than HBM-only**, because weights are re-streamed at a higher pJ/bit. Refresh adds ≤0.31% energy.
- **Packages at the 50 ms SLO:** **3.1× fewer** GPU packages than HBM-only (up to 8×). A single FLINT package serves every MoE model at bsz 1. At bsz 64: 2 vs 4 (Maverick), 32 vs 64 (Qwen3), 16 vs 96 (Kimi-K2), 16 vs 128 (DSv3). Llama ties at 128. DSv4-Pro at bsz 64 misses by 0.7 ms and needs 16 packages against 6.
- **Lifetime of an always-on weight block (bsz 1, 128K, 50 tok/s):**
  - Without refresh it corrupts in about **25 s** (geomean).
  - With refresh: **29 days at 10⁵ P/E** (7–33 days on MoE models, 1.0 year on dense Llama), **0.8 years at 10⁶**, **8.0 years at a projected 10⁷**.
  - The most-read block refreshes 1.1–7.9× more often than the average one.
- **Area:** +3.1% HBF die; 3.9 mm² base die at 7 nm.

### 2.7 Stated limitations and future work
- There is no explicit limitations or future-work section. The conclusion is a summary.
- Admissions in the text:
  - "vendors have not yet published an HBF endurance figure, we sweep the P/E budget from the commodity-SLC floor of 10⁵ cycles to a projected 10⁷"
  - Power-loss: "FLINT does not rely on preserving resident weights across an unplanned power loss"
  - Refresh read-out cost: "Reading the block's 1024 pages out before the erase adds up to 1024×tR on top, which we conservatively omit"
  - The hard SLO threshold makes DSv4-Pro at bsz 64 need more packages

### 2.8 Critical view
1. **The lifetime result undermines "read-only means no wear".** This is the key point for the student. At commodity SLC 10⁵ P/E, an always-on weight block dies in about **29 days** of continuous 50 tok/s decode. [my inference] Weights are read once per *decode step* regardless of batch, so lifetime is set by step rate (steps/s), not by tokens/s. A batched server running at 20 ms/step has roughly the same lifetime. The authors report this but frame FLINT as "significantly" increasing lifetime (against 25 s). They do not say that 29 days is not deployable at 10⁵. The student's report ("retention refresh is negligible at 995 years"; weights "written once") omits read-disturb-driven rewrites. This is a first-order gap to address.
2. **The KV cache sits in HBM with a "20 GB KV reserve", yet results are at 128K context and bsz up to 64 "from a single package".** For large-KV models (e.g., Llama-3.1-405B GQA) 128K × 64 requests cannot fit in 192 GB. [my inference; FlashAccel lists LLaMA3.1-405B at 502 KB/token, i.e. about 64 GB per 128K request.] The text does not explain how FLINT's KV fits. The HBM-only baseline is marked "capacity-infeasible" in those cells, but FLINT's feasibility is not discussed. This is a methodological gap.
3. **Link sharing is under-specified.** HBF traffic crosses the xPU→HBM D2D link, 1 TB/s per stack. FLINT claims to beat the capacity-lifted oracle because it "streams weights over the otherwise-idle HBF channel in parallel with the HBM KV reads and serializes only at the D2D links". How the shared xPU-facing link is arbitrated between HBM-local and HBF-relayed traffic is not detailed. This is exactly the student's pooling-vs-partitioning question, and DASH's motivation.
4. **Refresh is ~0.86 of ~1.28 GB/s per-stack program capacity**, a 67% utilisation margin [my calc]. At higher step rates (bigger GPUs, smaller models, speculative decoding), refresh demand scales with the read rate, and the phantom plane could fall behind. No sensitivity to step rate is shown. The 1024 × tR read-out is omitted, although the forked-read design mostly hides it.
5. **Phantom-plane "implicit wear levelling"** rotates *all* planes through reprogramming, cold ones included. T_rot is only a *lower bound* (≥ 12.5 s), which gives a ~7-minute full pass at the minimum. [my calc/inference] If rotation ran at that minimum, it would mean ~75,000 reprograms per block per year, or ~1.3 years at 10⁵ P/E even for *cold* blocks. The paper does not say what rotation rate it uses in operation or what P/E the rotation itself costs. It should be checked whether the lifetime figures include rotation wear.
6. **Energy per bit.** HBF read is 20 pJ/bit here against FlashAccel's 8 pJ/bit. FLINT sweeps 10–25 but reports the main result at 20. Dense models lose on energy (1.64×).
7. **Power loss means a clean reinstall (~6.7 min).** That adds P/E wear on every boot (a full-device erase and program) and adds cold-start downtime. Neither is counted in the lifetime or availability figures.
8. **Not evaluated:** prefill (decode only); KV in HBF (explicitly out of scope); thermal; tail latency (no percentiles); multi-tenant or model swapping (each swap is a full rewrite plus P/E); ECC strength/BER versus read count (the ECC engine is assumed, and read-disturb is a single threshold of 10⁶ with no BER curve); retention as a function of temperature; cost/TCO beyond package count; bad-block management (a "KB table" is listed but not modelled); expert load imbalance *across HBF planes* (bursts are striped across all planes, so there is no intra-stack imbalance by construction, but inter-stack "block-striped" imbalance is not discussed).
9. **Compute is analytical (MFU 0.5).** Energy for the HBM-only baseline is dominated by 400 W "sync spin" at barriers, so the 1.1× energy win depends on that barrier-power assumption.
10. **Burst granularity is 2 MB.** Fine for weights. With MoE, the expert weight slice per burst matters, and small experts (e.g., DeepSeek, 2048 intermediate dim) may under-fill a burst. The paper reports 90–97% useful traffic, but padding and fragmentation are not analysed.

### 2.9 BibTeX
```bibtex
@misc{oliveira2026flint,
  title         = {{FLINT}: Efficiently Leveraging High Bandwidth Flash for Capacity-Scalable {LLM} Inference Acceleration},
  author        = {Oliveira, Geraldo F. and Tavakkol, Arash and Zhu, Xiangyu and Y{\"u}z{\"u}g{\"u}ler, Ahmet Caner and Arulchelvan, Vamanan and Cavigelli, Lukas and Andri, Renzo and Sadrosadati, Mohammad and Jia, Xinglei and Mutlu, Onur and Zhou, Ke and Bergman, Shai and Zhang, Ji},
  year          = {2026},
  eprint        = {2608.25062},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv:2608.25062v1. Oliveira and Tavakkol are co-primary authors}
}
```
The paper prints the names as "Jia Xinglei" and "Zhou Ke". I parsed them as family name first ("Jia, Xinglei"; "Zhou, Ke"). Verify against the arXiv listing before submission.

---

## 3. DASH

### 3.1 Bibliographic
- **Title:** "Beyond Capacity: Scalable MoE LLM Inference via High-Bandwidth Flash with Direct GPU and HBM Paths". DASH stands for "Direct Attachment of HBF to the GPU as the main memory tier, with a Separate path to HBM". The PDF filename's shortened title differs slightly; the text is authoritative.
- **Authors:** Seeyeon Kim*, Juhyeong Jin* (equal contribution), Joo-Young Kim. KAIST, Daejeon.
- **Venue:** unknown (IEEE-style template, no venue given)
- **Year:** 2026 (arXiv v1 dated 14 Aug 2026)
- **arXiv:** 2608.14333v1 [cs.AR]

### 3.2 Problem
MoE weights are 281 GB to 1.5 TB, with experts making up 94.1–98.8% of the weight. They exceed HBM, and the expert set must stay addressable. Cascaded GPU–HBM–HBF designs (H³ [9], Park et al. [33]) push HBF traffic through HBM-side routing, so HBF bandwidth is never exposed as an independent resource. Direct attachment alone has two problems: (i) tR is exposed because MoE experts are only known after routing, and prior work [33] keeps experts in HBM to avoid this; (ii) slow tPROG, with small decode KV writes stalling reads.

### 3.3 Core mechanism / architecture
- **Attachment.** Three UCIe 3.0 UCIe-A links: GPU–HBM, GPU–HBF and HBM–HBF. Each is **4 modules × x64 at 64 GT/s = 2.048 TB/s raw per direction**, modelled as **1.6 TB/s usable** (~22% headroom). HBF reads can take the **Direct path** (HBF→GPU) or the **Relay path** (HBF→HBM base-die router→GPU, bypassing the HBM controller and DRAM cells). Both run concurrently.
- **Base-die SRAM.**
  - HBF base die: **18 MiB** per stack (16 MiB usable as two interleaved 8 MiB transfer regions matching one "page wave", plus 2 MiB for ECC check bits at 12.5%).
  - HBM base die: **9 MiB** (8 MiB usable as two 4 MiB relay regions).
  - Double buffering between shared-TSV fill and D2D drain. Writes are backpressured when both regions are full.
- **Placement (runtime/scheduler, by size, mutability and reuse):**
  - HBF: expert weights and write-once/read-many **prefill KV**.
  - HBM: activations and **decode KV**. Decode KV accumulates in HBM and is written back to HBF in page-aligned "page waves" when HBM nears capacity. The transfer starts after the QKV projection so tPROG overlaps attention.
  - Both: attention QKV/out-projection weights are **replicated** and read in parallel from both.
  - Inside HBF, each expert is chunked across dies and planes.
  - **Weight-owned vs KV-owned erase blocks**, so KV GC never relocates weight pages.
- **Execution modes:** (1) parallel HBM/HBF read; (2) dual-path HBF read; (3) direct prefill write (GPU→HBF, programmed while attention runs); (4) HBM→HBF KV writeback.
- **Lookahead expert selection (exact).** Router logits are rewritten as P = α(P_base + P_attn), with P_base = X·diag(γ)·W_r computed before attention and P_attn = O·Ŵ_r, where Ŵ_r = W_out·diag(γ)·W_r is precomputed. The positive scalar α = 1/RMS(·) does not change the ranking, so top-k is exact. Selection runs in FP32. It applies **only to routers that are scale-invariant and have no expert-specific additive bias**. DeepSeek-V3's router falls back to conventional late selection.
- **Who manages placement:** a GPU-side scheduler plus base-die schedulers. The FTL is not described beyond block isolation and a GC reserve.
- **Scalability:** n HBM–HBF pairs use 2n GPU-facing connections plus n pair-local links, which is O(n) links.

### 3.4 Device parameters
| Parameter | Value | Source cited |
|---|---|---|
| HBF stack | **512 GB, 16 dies × 32 planes/die**, "reported stack-level read bandwidth of 1.6 TB/s" | [37] SanDisk HBF Fact Sheet (Jul 2025) |
| Page | **4 KiB** | [51] SanDisk patent US 2025/0259685 A1 |
| Sub-array parallelism | **Assumed 4 independently accessible subarrays per plane** ("public HBF disclosures do not specify"). Page wave = **8 MiB per stack**, 16 MiB over two stacks | assumption |
| tR | **3 µs nominal**, background range "1–30 µs", swept **1–32 µs** | [20] KIOXIA serial NAND datasheet |
| tPROG | **100 µs nominal**, "tens to hundreds of microseconds", swept **50 µs–5 ms** | [20] |
| HBM | **24 GB HBM3E stack** | [25] Micron HBM3E product brief |
| D2D | 1.6 TB/s usable per link; UCIe-A **~0.5 pJ/bit @0.5 V, 0.6 pJ/bit @0.7 V**, i.e. **6.4–7.7 W per fully used direction**; module width 0.389 mm, so **~1.6 mm die edge per endpoint** | [41,42,46] UCIe 3.0 spec |
| Endurance | **E_SLC = 100,000 P/E** | "Following prior work [21]" (Kyung et al., CAL 2026) |
| GC reserve | 16 GB | §VI-H |
| HBF energy per bit | **not modelled** | — |
| Cost | not reported by vendor. Parametric r = C_HBF/C_HBM with anchor r = 1 ("fact sheet's qualitative similar-cost statement") | [37] |

[my calc] NAND-side rate per stack = 8 MiB / 3 µs ≈ 2.8 TB/s. In DASH one HBF stack drains over *both* a direct link (1.6 TB/s) and a relay link (1.6 TB/s), up to 3.2 TB/s. That is **double the SanDisk 1.6 TB/s stack target** DASH itself cites. The design can only exceed the vendor figure because of the assumed 4 subarrays per plane.

### 3.5 Evaluation methodology
- **Simulator:** SNU SCALE Lab's **LLMSimulator** [38], extended with HBF read/program latency, die/plane parallelism, SRAM readiness, per-link D2D availability and KV writeback. This is the same base simulator the student uses.
  - GPU operator latencies were **measured on an H100 PCIe 80 GB**. Unmeasured shapes are interpolated. Across 1,107 validation points the median error is 0.51% and 90% are under 3.52%.
  - Memory is the modelled DASH system, not the H100's HBM2e.
- **Configurations (Fig. 8):**
  - DASH: 2 × 512 GB HBF + 2 × 24 GB HBM.
  - Compact-DASH: 1 × 1024 GB HBF + 1 × 48 GB HBM.
  - RelayOnly: no direct GPU–HBF path.
  - DirectOnly: HBM–HBF peer link disabled.
- **Models:** Qwen3-235B-A22B (BF16, 128/8), Mixtral-8×22B (BF16, 8/2), Grok-1 (INT8-mixed, 8/2), Llama-4 Maverick (BF16, 128+1/1), DeepSeek-V3 (FP8-mixed, 256+1/8). DeepSeek-V2 (160+2/6) is used only for the lookahead study.
- **Default:** B/L_in/L_out = **4/1K/128**. Batch sweep {1, 4, 16, 64}. Sequence sweep at B = 1: 512/16, 4K/16, 512/20K, 20K/20K; Maverick additionally at 4K/256K and 4K/1M (KV overflow into HBF).
- **Continuous batching (Qwen3):** 4096-token prompts, 128 output tokens, B_max = 32, 8192-token iteration budget, 512-token chunks. Mixed vs Serial scheduling. Poisson arrivals at 50/75/90% of RelayOnly saturation, 5 seeds, 1,024 measured requests. Routing is "analytical ... with balanced per-expert rows", not real routing traces.
- **CPU offload comparison:** a real Xeon Platinum 8452Y measurement of a Qwen3 expert layer, compared with a Hybrid oracle.

### 3.6 Headline results
- **Batch sweep:** geomean throughput **1.90×** over RelayOnly and **1.84×** over DirectOnly; E2E latency −42.2% and −40.8%. The abstract's representative workload gives "1.94× higher throughput and 1.90× end-to-end speedup" over relay-only.
- **Sequence sweep (20 combinations):** 1.79× and 1.63× throughput; −40.1% and −35.6% E2E latency.
- **Maverick, 197.413 GB KV** (24.46 GB in HBM, 172.96 GB in HBF): **1.92×** throughput and **−48.0%** E2E versus RelayOnly.
- **Compact-DASH** (half the stacks) is "comparable" to RelayOnly and DirectOnly.
- **CPU Hybrid oracle** is **8.22–12.32×** slower than DASH.
- **Lookahead expert selection:** at 3 µs, E2E −3.33% (Qwen3) and −1.99% (DSv2); TPOT −3.86% and −2.45%. At 32 µs, E2E −9.50% and −8.69%; TPOT −10.88% and −10.53%.
- **Continuous batching, Qwen3 P90 (Table III):**

  | Load | P90 E2E | P90 TPOT | Reduction vs RelayOnly |
  |---|---|---|---|
  | 50% | 20.982 s | 148.0 ms | ~61–62% |
  | 75% | 28.847 s | 208.5 ms | E2E −53.5% (−53.3% vs DirectOnly) |
  | 90% | 31.859 s | 232.0 ms | ~50% |

  Peak throughput is 34.1–37.1% higher than single-route. Mixed scheduling adds 10.3% (DASH).
- **Sensitivity:**
  - Larger page size and lower tR reduce E2E.
  - All program time is hidden up to **500 µs** tPROG; at **5 ms** there is 19.94 s of stall.
  - Latency rises when D2D drops below 1.6 TB/s and keeps improving above it.
- **Cost:** G(r, δ) = 44.67/(2 + 2r + δ), which is **11.17×** at (1, 0). G > 1 whenever r < 21.33 (δ = 0). The capacity comparison is 1,072 GB against 96 GB for four HBM stacks.
- **Endurance:** Maverick at 32K/128, B_max = 32. R_HBF = **1,015.13 MB/s** (1,648.34 GB written over 1,623.76 s). C_KV = 206.58 GB (1,024 GB − 801.42 GB weights − 16 GB GC reserve). **Lifetime = 0.645 years** at WAF 1 and u = 1, assuming uniform wear.

### 3.7 Stated limitations and future work (quotes)
- "A deployed-lifetime claim additionally requires target-device measurements of total WAF, per-block erase counts, bad-block growth, and device aging, together with a deployment-specific duty cycle"
- "larger configurations remain subject to GPU-edge PHY beachfront, package area, and interposer routability"
- "physical design further accounts for routing, signal- and power-integrity, and power-delivery costs"
- "Because concrete HBF specifications are not yet available, we further sweep the HBF read latency"
- "public HBF disclosures do not specify the exact subarray-level parallelism"
- "the public HBF fact sheet reports neither a numerical per-stack price nor package-integration cost"
- Early selection only "when routing is scale-invariant and free of expert-specific additive bias"
- There is no explicit future-work section.

### 3.8 Critical view
1. **The speedup is largely extra link bandwidth.** DASH has 6.4 TB/s of expert-delivery bandwidth against 3.2 TB/s for either baseline (stated in §VI-E), so a roughly 1.9× gain is close to what 2× link bandwidth predicts for bandwidth-bound expert streaming. The fair comparison would hold total GPU-facing bandwidth or shoreline constant. Compact-DASH is a partial answer (one pair, comparable to the baselines). [my inference]
2. **HBF internal bandwidth assumed above the vendor target.** Feeding both paths needs about 3.2 TB/s per stack. The 8 MiB/3 µs ≈ 2.8 TB/s rate rests on an *assumed* 4 subarrays per plane, whereas SanDisk's figure [37] is 1.6 TB/s per stack. No sensitivity to subarray count is shown; the page-size sweep partly proxies for it.
3. **The relay path consumes the GPU–HBM link.** Relayed expert bytes compete with KV and attention-weight reads from HBM on the same 1.6 TB/s GPU–HBM link. This is the student's pooling-vs-partitioning issue. The paper claims concurrency but does not break down contention on the GPU–HBM link.
4. **Endurance of 0.645 years at 10⁵ P/E** in a prefill-heavy case is not deployable. The authors call it a "projection", not a claim, but they still conclude that DASH "addresses the endurance challenges". Read disturb on expert weights is not considered; FLINT's result suggests it would further limit lifetime for hot experts.
5. **The simulation setting is small.** The default batch is 4 with 1K/128. TPOT P90 of 148–232 ms is far above typical 50 ms SLOs. The GPU is an H100 PCIe (compute-profiled) paired with only 48 GB of HBM. Routing in continuous batching is "analytical ... balanced per-expert rows", so no expert skew, which is precisely what FLINT found hurts caching and prefetch.
6. **The lookahead gain is small at realistic tR** (2–4% at 3 µs). It is inapplicable to DeepSeek-V3-style biased routers, which are an increasingly common design.
7. **Not evaluated:** energy (only a UCIe link-power estimate; HBF read/program energy is not modelled at all), thermal, read disturb, retention, ECC beyond SRAM check-bits, GC behaviour and WAF (a 16 GB reserve is assumed and WAF is set to 1), bad blocks, multi-GPU scaling (single GPU; O(n) topology only argued), multi-tenant, and cost (parametric only, anchored at r = 1).
8. **Shoreline.** DASH uses 2n GPU-facing connections for n pairs, the same as a 2n-HBM reference, so an HBM stack is traded for an HBF stack at the GPU edge. This is the CLI-like trade that the student's report finds H³'s chaining avoids ("wins because it preserves HBM sites"). DASH also adds a peer UCIe PHY on each HBM base die (1.6 mm edge), which non-standard HBM would require.
9. **The replicated attention weights** use HBF capacity and HBM capacity at the same time. The cost in HBM capacity (and hence KV/batch) is not quantified.

### 3.9 BibTeX
```bibtex
@misc{kim2026dash,
  title         = {Beyond Capacity: Scalable {MoE} {LLM} Inference via High-Bandwidth Flash with Direct {GPU} and {HBM} Paths},
  author        = {Kim, Seeyeon and Jin, Juhyeong and Kim, Joo-Young},
  year          = {2026},
  eprint        = {2608.14333},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv:2608.14333v1. Kim and Jin contributed equally}
}
```

---

## 4. Side-by-side assumptions

| | FlashAccel | FLINT | DASH | (H³, as cited by these) |
|---|---|---|---|---|
| Attachment | CLI (replaces HBM) or CSI (chain); GPU accesses HBF "directly" | Chain: xPU→HBM→HBF over D2D, single shoreline | Direct GPU–HBF **and** HBF→HBM relay (3 UCIe links) | Chain behind HBM |
| HBF holds | Weights **plus all KV** | Weights only | Experts, prefill KV, spilled decode KV, replicated attention weights | Read-only (weights, shared KV) |
| Dies/stack, planes/die | 8, 96 | 16, 32 | 16, 32 (×4 subarrays assumed) | — |
| Capacity/stack | 192 GB | 512 GB | 512 GB | — |
| BW/stack | 768 GB/s | 1 TB/s (background 1.6–2) | 1.6 TB/s per link; ~2.8 TB/s NAND-side [my calc] | — |
| Page | 4 KB | 4 kB | 4 KiB | — |
| tR | 4 µs | 2 µs | 3 µs (1–32 sweep) | — |
| tPROG | 75 µs | 50 µs | 100 µs (50 µs–5 ms sweep) | — |
| tERASE | — | 3 ms | — | — |
| P/E | 100K, assumed 1M via retention relaxation | 10⁵ (sweep to 10⁷) | 100K | — |
| Read disturb | not modelled | 10⁶ reads/block | not modelled | — |
| Retention | relaxed to days (implied by the P/E boost) | >1 yr end-of-life | not modelled | — |
| HBF read energy | 8 pJ/bit (TLC chip) | 20 pJ/bit (10–25) | not modelled | — |
| HBM energy | 2.99 pJ/bit | 4 pJ/bit | — | — |
| Latency hiding | Per-plane SRAM (32 KB) + 8 MB base-die SRAM + runtime prefetch in graph order | Hardware burst-buffer using NAND page/cache latches; no SRAM LHB, no hints | 18 MiB HBF + 9 MiB HBM base-die SRAM double-buffering + exact early top-k | MB-scale SRAM LHB + compiler hints |
| FTL | none (physical addresses in object metadata; isolated blocks) | read-only burst FTL, 1.8 MB/stack | block isolation + 16 GB GC reserve | — |
| GPU | H200 ×8 | B200 | H100 PCIe (profiled) | — |
| Simulator | LLMCompass + NAND model | in-house trace-driven + WSE + Ramulator calibration | LLMSimulator (SNU) — same as the student | — |
| Batch | up to ~256 (SLO-limited) | 1–64 | 1–64, default 4 | — |
| Context | 8.11K/2.53K, 15K/6K | 2–128K (report 128K) | 512–20K, to 1M for Maverick | — |
| Phase | decode (plus KV hit rate for prefill) | decode only | prefill + decode, continuous batching | — |

---

## 5. Cross-paper observations

### 5.1 Contradictions in assumptions
1. **Should KV go in flash?**
   - FlashAccel puts *all* KV in HBF and says endurance is "sufficient" (1,125,000 TBW against 148,570 TB over 5 years), but only after assuming 1M P/E.
   - DASH, with 100K P/E and a similar write rate (~1,015 MB/s against FlashAccel's 988 MB/s), projects **0.645 years**.
   - FLINT refuses KV in HBF.

   [my calc] The ~6.5× gap between FlashAccel and DASH comes mostly from two things: the P/E assumption (10×) and writable capacity (FlashAccel's 1,152 GB across 6 stacks against DASH's 206.58 GB after weights). The *same* write rate leads to opposite conclusions depending on unvalidated endurance numbers.
2. **Read-only weights and wear.**
   - DASH: "Read-only model weights cause little write wear".
   - FlashAccel: weights written once, no refresh.
   - FLINT: always-on weight blocks need refresh every ~10⁶ reads and **die in ~29 days at 10⁵ P/E**.

   FLINT directly contradicts the other two, and the student's own "995-year retention" framing. Read disturb, not retention or GC, may be the binding endurance mechanism for a weight tier.
3. **Latency hiding: SRAM or no SRAM.** FlashAccel adds SRAM (32 MB/stack) and DASH adds 18 + 9 MiB. FLINT argues dedicated SRAM is "avoidable area overhead" because the NAND page/cache latches suffice. FLINT also shows static graph-order prefetch (FlashAccel's scheme, and H³'s) wastes 86–96% of traffic on MoE. FlashAccel evaluates three MoE models with static topological-order prefetch and reports near-HBM latency (~4% gap). The two results cannot both hold without an explanation of FlashAccel's expert handling, possibly EP with all local experts streamed.
4. **Chain vs direct attachment.**
   - FLINT: a single shoreline and chain is the right choice.
   - DASH: chain-only (RelayOnly) wastes a path, so use both.
   - FlashAccel: calls HBF behind HBM "increasingly inefficient", yet its best configuration (CSI) is the chain, and CLI (direct, replacing HBM) loses at 50 ms.
   - The student's report: chain wins "because it preserves HBM sites, not because it supplies flash bandwidth".

   No paper holds GPU-edge shoreline constant across all three topologies.
5. **Device numbers differ widely.** tR ranges 2–4 µs, tPROG 50–100 µs, planes/die 32–96, capacity/stack 192–512 GB, BW/stack 0.77–1.6 TB/s (DASH implicitly ~2.8), HBF read energy 8 or 20 pJ/bit or not modelled. The papers cite different sources: XL-Flash JSSC 2020; SanDisk blog, fact sheet and patent; KIOXIA serial NAND datasheet. None has a real HBF part, and every result is sensitive to these choices.
6. **Where the benefit comes from.** FlashAccel: batch size (throughput regime, batch ~100–256). FLINT: removing multi-package synchronisation at *small* batch (bsz 1–4). DASH: link bandwidth for expert streaming at batch 4. These are three different operating regimes, and none compares against the others. FLINT's biggest win is where FlashAccel does not evaluate, and the reverse.
7. **Energy direction.** FLINT finds dense models use *more* energy with HBF (1.64×). FlashAccel finds 1.93× *better* tokens/J, including for dense LLaMA-405B. The difference comes from the pJ/bit (8 vs 20) and the batch regime (weight reads amortised over big batches in FlashAccel).

### 5.2 Design points nobody explored
- **Read-disturb-aware placement or replication of hot weights.** FLINT hides refresh but does not *reduce* disturb. Options include replicating always-on blocks (attention weights, shared experts) in HBM, as DASH does for attention weights but for bandwidth reasons, or rotating the physical copies of hot experts. Combining DASH-style replication with FLINT-style counters would directly extend lifetime.
- **Per-block (not aggregate) wear analysis with mixed weights and KV in one device.** FlashAccel and DASH mix both. None models the resulting hot/cold skew or static wear levelling. The student's 10.7× wear-concentration finding is exactly this gap.
- **Iso-shoreline comparison** of chain, direct and dual-path under the same GPU-edge budget and same total link bandwidth.
- **Link arbitration and pooling** between HBM-local and HBF-relayed traffic on the shared xPU–HBM link. FLINT and DASH both depend on it and neither details it. This is the student's pooling result.
- **Prefill-decode disaggregation with HBF.** Only FlashAccel touches it, and analytically (712 MB/s of prefill writes derived from a ratio).
- **Multi-tenant or model swapping.** Every swap is a full-device program. FLINT's power-up reinstall (~6.7 min) is the only related number. Swap frequency against endurance is unstudied.
- **Thermal coupling.** Flash retention and read-disturb worsen with temperature, and HBF sits next to a 1000 W GPU. Nobody models temperature.
- **Tail latency from program and erase interference with reads** at the plane level. DASH reports P90 at the request level only; the others report only averages.
- **Speculative decoding or larger step rates.** Refresh and disturb scale with the read rate per step, which none sweeps.
- **KV compression or quantisation** to reduce HBF write volume. Only attention variants (GQA/MLA) are cited as reducing writes.

### 5.3 Research gaps directly relevant to the student's BTP
1. Add a **read-disturb and refresh model** to the endurance analysis. Use FLINT's parameters: 10⁶ reads/block, 54.2 ms in-place refresh, 10⁵ P/E. Note that lifetime scales with decode *step* rate, not batch. The current "weights written once / 995-year retention" conclusion is incomplete without it.
2. The student's **12H4F (fewer flash dies) recommendation concentrates reads on fewer planes and blocks**. [my inference] If each block is read more often per unit time, read-disturb refresh comes sooner. The die-split decision gains a fourth axis, read-disturb wear, on top of the report's bandwidth, power and collection pressure.
3. The student's **pooling-vs-partitioning result** fills a gap all three papers leave open: FLINT's shared D2D link, DASH's relay contending on the GPU–HBM link, and FlashAccel's unspecified CSI bandwidth sharing.
4. The student's **flash over-provisioning finding (17–67×)** contrasts with FLINT's 4 TB of HBF (8 × 512 GB) on one GPU and DASH's 1,024 GB. The weight footprints (281 GB–1.5 TB) suggest these papers also over-provision capacity. The extra capacity does help endurance (larger C_KV in DASH's formula) and FLINT's phantom rotation.
5. **Device-parameter sensitivity.** Since tR, tPROG, planes/die and pJ/bit are all unvalidated, results should be swept across the union of these papers' ranges: tR 1–32 µs, tPROG 50 µs–5 ms, 8–20 pJ/bit, P/E 10⁵–10⁷.
6. The student uses **LLMSimulator, the same base as DASH**. DASH's HBF extensions (page wave, SRAM readiness, per-link D2D availability, KV writeback) are directly comparable, and the student's results can be positioned against DASH's.
