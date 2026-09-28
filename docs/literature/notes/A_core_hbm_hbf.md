# Core HBM + HBF papers: reading notes (set A)

Context: BTP report `/Users/garimakalra/Desktop/BTP/BTP_Report.tex` ("Memory Tiering for Long-Context LLM Inference") evaluates H3, compares it with the physical arrangements H3 dismisses, and proposes reallocating unused flash dies to HBM dies (12H4F), plus endurance/GC/wear-concentration analysis. The notes below were written with that in mind. They mark what matters for (a) where HBF sits, (b) what lives where, and (c) endurance/GC/wear.

Conventions:
- "Quoted" = verbatim from the paper text.
- **[my arithmetic]** = something I derived. It is not stated in the paper, so do not cite it as the authors' claim.
- All numbers come from the plain-text extractions. Figure-only values (bar heights, heatmap cells) were not readable and are not reported.

---

## 1. H3 (Ha, Kim, Kim — SK hynix, IEEE CAL 2026)

### 1.1 Bibliographic
- **Title:** "H3: Hybrid Architecture Using High Bandwidth Memory and High Bandwidth Flash for Cost-Efficient LLM Inference" (the superscript is typeset as H^3).
- **Authors:** Minho Ha, Euiseok Kim, Hoshik Kim (all SK hynix Inc., Icheon-Si, Korea). Corresponding author: Minho Ha.
- **Venue:** IEEE Computer Architecture Letters (CAL), vol. 25, no. 1, Jan.–Jun. 2026, pp. 49–52.
- **DOI:** 10.1109/LCA.2026.3660969.
- **Dates:** received 4 Jan 2026; revised 31 Jan 2026; accepted 1 Feb 2026; published 4 Feb 2026.
- **arXiv:** none given.

### 1.2 Problem
Long-sequence LLM inference, and especially cache-augmented generation (CAG) with a shared pre-computed KV cache, needs far more capacity than HBM provides. For Llama 3.1 405B the shared pre-computed KV is "approximately 540 GB and 5.4 TB" at 1M and 10M tokens, "requiring dozens of GPUs just to store these values." HBF gives capacity but has long latency, poor endurance and high power, so it needs an architecture and a use case that hide these weaknesses.

### 1.3 Mechanism / architecture
- **Placement:** HBF is **daisy-chained behind HBM**. HBMs sit on the GPU shoreline, and each HBF connects to an HBM base die. H3 explicitly rejects placing HBM and HBF side by side on the shoreline because "this method has the disadvantage of reducing the number of HBMs due to the limited shoreline space on the GPU."
- **Access path:** inside the HBM base die an "address decoder & router" splits accesses into an HBM path and an HBF path. The GPU therefore reaches HBF *directly through the HBM base die*, and both HBM and HBF are main memory in a **unified address space with divided regions**.
- **Links:** GPU, HBM base die and HBF base die are connected by D2D interfaces. The paper assumes "the bandwidth between the GPU and the HBM base die is equal to the bandwidth between each base die and the core die." The HBM and HBF controllers sit on their own base dies ("subject to change depending on silicon area availability").
- **Data placement:** **model weights + shared pre-computed KV cache go to HBF** (read-only). **Generated KV + other data go to HBM.**
- **Who manages placement:** the software stack. "we expect that it will be possible to allocate data suitable for the HBM and the HBF with the support of deep learning framework". "tensor-level scheduling is possible, making contention between the HBM and HBF manageable."
- **Latency Hiding Buffer (LHB):** an SRAM prefetch buffer on the HBM base die (or the HBF base die). It relies on the "deterministic and sequential data pattern" of LLM inference. The DL framework must be modified to issue prefetch hints at "coarse-grained tensor-level".
  - LHB size: Capacity_LHB = 2 × BW_HBF × Latency_HBF, with double buffering.
  - With 1 TB/s and 20 µs this gives **40 MB**.
  - Area: 40 MB of 3 nm SRAM at 0.021 µm²/bit gives a 6.72 mm² core, **8.06 mm²** with 20% overhead, which is "approximately 6.7% of the base die area of 121 mm²".
- **Hop latency:** HBF accesses add "hop latency", but the paper says this "can be sufficiently hidden" because inference is bandwidth-bound.
- **Claimed side benefit:** read-only use "reduc[es] garbage collection and wear-leveling overhead" (no numbers).

### 1.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| GPU | NVIDIA B200 | [8] NVIDIA Blackwell brief |
| HBM | HBM3e, 192 GB and 8 TB/s per GPU (24 GB, 1 TB/s per cube) | [8] |
| HBF capacity | 3 TB per GPU, "approximately 16x larger than the HBM3e" | from SanDisk target of "up to 16x" [7] |
| HBF bandwidth | 8 TB/s per GPU, same as HBM3e (BW_HBF = 1 TB/s per cube in Eq. 1) | assumption |
| HBF read latency | 20 µs, "based on the read access latency of publicly available SLC NAND flash [11]" ([11] = RL-assisted convertible SSD, DAC'23) | |
| Program latency / page size / P/E | **not specified** | — |
| TDP per cube | HBM3e 40 W, **HBF 160 W** ("considering the per-cube capacity and the power consumption per bit of current NAND flash") | assumption |
| GPU TDP | 680 W without HBM/HBF | [8] |
| Qualitative targets | "up to 16x larger capacity", "similar bandwidth", "slower access (ns vs. µs)", "lower write endurance", "up to 4x higher power consumption than HBM" | [7] SanDisk blog |

Note on per-cube HBF capacity: 3 TB over 8 cubes is about 384 GB per cube **[my arithmetic]**. The paper never states a per-cube capacity.

### 1.5 Evaluation methodology
- **Simulator:** an in-house analytical simulator "proven in previous studies [3], [4], [10]". It models:
  - compute time: operation count / GPU performance;
  - data-transfer time: data size / device BW, with TP and DP partitioning/duplication;
  - ring all-reduce for TP.
- **Workload:** Llama 3.1 405B, FP8 (405 GB weights). Context windows of 1M and 10M are *assumed* (the native context is 128K). ISL = OSL = 1K, and the rest of the context is shared pre-computed KV, which is about 35% (1M) and 84% (10M) of total capacity.
- **System:** DGX-style. 1M runs on 8 GPUs and 10M on 32 GPUs, which are the HBM-only minimum GPU counts. At 32 GPUs scale-out goes over InfiniBand.
- **Metrics:** max batch size, TPS per request, throughput per power (throughput / total TDP).
- **Sensitivity:** HBF bandwidth halved.

### 1.6 Headline results
- Max batch: up to **2.6×** (1M) and **18.8×** (10M) versus HBM-only.
- One GPU with H3 can run the 1M case and two GPUs can run the 10M case.
- Throughput (TPS per request): **1.25×** (1M), **6.14×** (10M).
- Throughput/power: **up to 2.69×**.
- With HBF BW halved: 1M throughput is still above HBM-only, and 10M throughput/power is still **2.09×**.
- LHB: 40 MB SRAM, 8.06 mm² (6.7% of a 121 mm² base die).

### 1.7 Stated limitations / future work
- HBF is "not yet a commercially available device", so the paper uses target specs and "actual measurements are impossible".
- Power "is based on current NAND flash technology", and the authors "expect that power consumption of HBF will be further reduced".
- Bandwidth: "practical issues during productization may lead to supporting a lower bandwidth".
- The 20 µs latency is "expected to improve", so the LHB would be smaller than 40 MB.
- Future work: "There are various ways to use HBM and HBF together (e.g., hybrid stacking of HBM and HBF), and in certain cases, it may be possible to use HBF alone. Therefore, future work will focus on exploring optimal architectures that can exploit the benefits of HBF and other applications suitable for HBF."

### 1.8 Critical view
- **Link bandwidth is assumed away.** The GPU–HBM-base-die link is assumed to equal each base-die-to-core-die bandwidth, and the paper reports 8 TB/s of HBF *in addition to* 8 TB/s of HBM. Behind a daisy chain, however, all HBF traffic crosses the same shoreline link as HBM traffic. The paper never says whether the GPU-side link is 16 TB/s or a shared 8 TB/s. This is exactly the point the BTP report makes: the benefit comes from preserving HBM sites, not from flash bandwidth, and pooling the link matters.
- **The shoreline alternative is dismissed in one sentence.** There is no quantitative comparison with side-by-side placement, hybrid stacking, or a separate link (acknowledged only as future work).
- **The 16× HBF capacity is never sized against demand.** The BTP report finds it over-provisioned by 17–67× for most models.
- **Power is TDP-based and static** (160 W per HBF cube, which is 4× HBM). There is no activity-based energy (pJ/bit), no read-versus-program energy, and no thermal analysis of a 160 W stack chained beside a 40 W HBM next to the GPU.
- **No latency or SLO metrics** (TTFT/TPOT) and no tail latency. "Hop latency can be sufficiently hidden" is asserted, not measured. The LHB is sized, but its effectiveness is not simulated (no hit/miss model).
- **Endurance, GC and wear leveling are claimed to be reduced, not quantified.** No P/E budget, no WAF, no retention or read-disturb analysis. Read disturb is a real concern for SLC blocks read at TB/s rates for years.
- **Write path is unspecified.** Weights and the shared KV must still be written into HBF initially and on corpus/model updates. Update frequency and cost are not discussed.
- **One model (Llama 3.1 405B, dense, GQA).** No MoE (where weights-in-HBF prefetch is harder), no MLA, a single ISL/OSL, and no prefill/decode distinction or disaggregation.
- **Fixed 1:1 HBM:HBF stack pairing.** The HBM/HBF die mix is not explored.
- **Contention between HBM and HBF traffic** is called "manageable" without evidence. The HBM/HBF controller location is left open.
- **No cost model.** Throughput/TDP stands in for cost, with no $/GB and no stack or packaging cost.

### 1.9 BibTeX
```bibtex
@article{ha2026h3,
  author  = {Ha, Minho and Kim, Euiseok and Kim, Hoshik},
  title   = {{H$^3$}: Hybrid Architecture Using High Bandwidth Memory and High Bandwidth Flash for Cost-Efficient {LLM} Inference},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {1},
  pages   = {49--52},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3660969}
}
```

---

## 2. HBM-HBF-Centric Memory Pooling with Custom Base Die (Park et al. — KAIST, IEEE CAL 2026)

### 2.1 Bibliographic
- **Title:** "HBM-HBF-Centric Memory Pooling Architecture With Custom Base Die for Terabyte-Scale LLM Inference".
- **Authors:** Junho Park (Graduate Student Member, IEEE), Hyowon An, Haeseok Suh (GSM, IEEE), Youngsu Yoon (Member, IEEE), Hyuni Lee, Joungho Kim (Fellow, IEEE). Dept. of EE, KAIST, Daejeon. Corresponding author: Joungho Kim.
- **Venue:** IEEE Computer Architecture Letters, vol. 25, no. 2, Jul.–Dec. 2026, pp. 259–262.
- **DOI:** 10.1109/LCA.2026.3703982.
- **Dates:** received 6 May 2026; revised 11 Jun 2026; accepted 11 Jun 2026; published 16 Jun 2026.
- **Funding:** IITP grant IITP-2026-RS-2024-00436765.
- **arXiv:** none given.

### 2.2 Problem
Weights plus KV for terabyte-scale models exceed HBM. H3-style read-only HBF has two failure points:
- It cannot absorb *newly generated* KV.
- MoE's "non-deterministic expert routing ... degrades prefetch accuracy".

The paper therefore wants a read-*write* HBM+HBF pool managed entirely in hardware.

### 2.3 Mechanism / architecture
- **Placement:** a "cascaded memory configuration where HBF units are stacked behind the HBM modules within a single package". HBF stacks are "interconnected with the HBM through the silicon interposer" (so HBF is behind HBM topologically, not vertically stacked on it). GPU, HBM and HBF talk over high-speed D2D links.
- **Custom Base Die (CBD):** replaces the passive HBM base die with logic that manages the HBM-HBF pool "with full hardware autonomy", "without host-side intervention".
- **System:** 8 GPU-HBM-HBF modules. Each has **192 GB HBM (24 GB × 8 stacks)** and **4 TB HBF (512 GB × 8 stacks)**.
- **HBM partition per module** (Llama-4 Maverick FP8):
  - 50 GB model-weight region;
  - 2 GB system metadata;
  - 140 GB KV buffers, split into about **60 GB KV write buffer + 80 GB KV read buffer** at batch 256. The split is "statically configurable".
- **What lives where:**
  - Weights are kept **resident in HBM** and are not staged from HBF, because "non-deterministic expert routing in MoE architectures makes prefetch accuracy unreliable".
  - KV lives in HBF and is staged through HBM buffers.
- **Addressing:** the GPU MMU generates Unified Physical Addresses (UPAs) covering the HBF space, the HBM weight region and the HBM reserved area. The KV read/write buffers are *excluded* from UPA space and managed transparently by the CBD.
- **CBD blocks:**
  1. **UCMC (Unified Central Memory Controller):**
     - The prefetch controller tracks GPU layer progress and prefetches later layers' KV from HBF into HBM.
     - The request scheduler uses fixed priority: **P1** on-demand fetch after a prefetch miss, **P2** BWB flush, **P3** prefetch.
     - A DMA engine does the transfers.
  2. **HBM sub-system:**
     - A boundary manager classifies UPAs in one cycle.
     - A Buffer Management Unit (BMU) keeps an SRAM tag directory.
     - The read buffer is split into active and prefetch regions (ping-pong).
     - On a BWB flush, prefetch is suspended ("freeze interval") and the GPU drains the read buffer.
  3. **HBF sub-system:**
     - An SRAM L2P table maps at **erase-block granularity**.
     - The DMA engine moves data both ways.
     - The HBF interface offloads "ECC, wear leveling, and NAND tR management to the HBF base die hardware".
- **Buffered Write-Back (BWB):**
  - New-token KV is accumulated **per user** into erase-block-sized entries in the HBM write buffer.
  - Each user's KV is striped across 6 planes following [6] (a 2Tb 4b/cell 6-plane ISSCC'26 part).
  - Erase block ≈ 20 MB = "16 KB (page size) × 1 (SLC) × 332 (WL) × 4 (SSL)". The per-user flush granularity is **120 MB**.
  - Tag states are accumulating, in-transit and empty.
  - Reads to accumulating or in-transit entries are redirected to the write buffer, "eliminating RAW hazards entirely".
- **Overhead:**
  - Tag directory + L2P: 17.6 MB SRAM, 3 mm² (0.021 µm²/bit).
  - Two UCIe-A D2D PHYs at 2 TB/s each: 7.6 mm².
  - CBD total 10–11 mm². Replacing Wide I/O with D2D PHYs reclaims 32 mm² (estimated from HBM4 bump count/pitch), a net margin of 21 mm².
  - Power: PHYs about **9.28 W** at peak (0.29 pJ/bit × 2 TB/s × 8 × 2); SRAM leakage 115 mW.

### 2.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| HBM | 24 GB per stack, 8 stacks, 192 GB per module | — |
| HBF | 512 GB per stack, 8 stacks, 4 TB per module | — |
| HBF tR | **15 µs** | "reflecting SLC-based NAND configuration" |
| HBF tPROG | **100 µs** | same |
| Page | 16 KB; SLC; erase block ≈ 20 MB (332 WL × 4 SSL) | [6] Thimmaiah ISSCC'26 |
| HBF per-stack bandwidth | **not stated** | — |
| D2D | UCIe-A, 2 TB/s per PHY, 0.29 pJ/bit | [8] Melek ISSCC'26 |
| Endurance (P/E) | **not stated** | — |
| Host DRAM BW / SSD BW / GPU-host | 512 GB/s; 128 GB/s (PCIe Gen5); 900 GB/s NVLink-C2C | — |

Check: 16 KiB × 332 × 4 = 20.75 MiB **[my arithmetic]**, consistent with "approximately 20 MB".

### 2.5 Evaluation methodology
- **Model:** a roofline-based analytical latency model [11]. Per-step latency is the max of compute and memory time across tiers, "conservatively accounting for cases where prefetch does not complete before the next layer begins". Both prefill and decode are modeled.
- **System:** 8 GPUs in one NVLink switch domain. The baseline is an "enhanced Blackwell-class node with **384 GB HBM**" (2× standard, "to account for the physical area overhead of the HBF pool").
- **Spill path:** both architectures spill to host DRAM and then SSD when capacity is exceeded.
- **Workload:** LLaMA-4 Maverick FP8 (~400 GB weights including experts), GQA with 8 KV heads and head dim 128. Batch 1–256, context 100K–1M, input length 1K.
- **Compared designs:**
  1. Only-HBM + host DRAM + SSD.
  2. **CBD SRAM Prefetch**, "following similar methodology to H3" with read-only HBF and the **hit rate fixed at 40%**.
  3. HBM-HBF without prefetch.
  4. Proposed (layer-wise prefetch + BWB).

### 2.6 Headline results
- **Max throughput gain 87.4×** over the conventional GPU-HBM system "in the most memory-pressured regime". TPOT is **98.6 ms/token versus 8,615 ms/token** at batch 256 and 1M context.
- Within HBM capacity, all designs except CBD SRAM Prefetch are comparable.
- CBD SRAM Prefetch throughput varies by **91% across hit rates at 100K** context and by **15% at 1M**. The proposed design is insensitive to hit rate.

### 2.7 Stated limitations / future work
- "A quantitative analysis of throughput robustness to HBF characteristics and buffer configuration remains to be addressed."
- "thermal characterization, full system power analysis, and further investigation into workload generalizability and the sensitivity to HBF and buffer parameters remain important directions for future work."

### 2.8 Critical view
- **The 87.4× is against a strawman.** The baseline spills KV to host DRAM/SSD over PCIe/C2C (SSD at 128 GB/s), so the gain mostly measures "SSD thrashing versus on-package capacity". It is not a comparison with a well-configured multi-GPU HBM system or with KV offload at NVLink-C2C.
- **The H3-like baseline is not H3.** It is an SRAM prefetch with an arbitrary fixed 40% hit rate. H3 does *not* put generated KV in HBF, and its LHB is sized for deterministic tensor-level prefetch.
- **HBF bandwidth per stack is never given.** Only the UCIe PHY rate (2 TB/s) is stated. Throughput therefore cannot be reproduced.
- **HBM bandwidth used by staging is not quantified.** HBF→HBM prefetch writes into HBM and the GPU then reads it, so HBM carries each KV byte at least twice (write + read), plus write-buffer traffic. This bandwidth cost is not analyzed.
- **Endurance is not evaluated at all**, even though the design *adds* full write traffic to HBF. There is no P/E budget, lifetime or WAF. ECC, wear leveling and tR management are "offloaded to the HBF base die" with no model.
- **120 MB per-user flush granularity.** Short-context or low-OSL users may never fill a 120 MB entry. The paper does not say how partial blocks, session termination or KV freeing (which invalidates blocks and triggers GC) are handled. GC is not discussed.
- **Freeze intervals** could create tail-latency spikes. Only average TPOT is reported, not tail latency.
- **Weights in HBM (50 GB per module)** works only because 400 GB fits across 8 modules. It does not scale to larger MoE models.
- **The "reclaims 32 mm²" claim** assumes the HBM Wide I/O is replaced by UCIe, i.e. a non-JEDEC custom HBM. Signal integrity and cost are not discussed.
- Single model, no multi-tenancy, no power/thermal figures (acknowledged).
- Interesting point for the BTP: here the HBM stays at 8 sites and HBF sits behind it, the same topology family as H3.

### 2.9 BibTeX
```bibtex
@article{park2026pooling,
  author  = {Park, Junho and An, Hyowon and Suh, Haeseok and Yoon, Youngsu and Lee, Hyuni and Kim, Joungho},
  title   = {{HBM-HBF}-Centric Memory Pooling Architecture With Custom Base Die for Terabyte-Scale {LLM} Inference},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {2},
  pages   = {259--262},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3703982}
}
```

---

## 3. Hardware-Managed Heterogeneous HBM and Flash (Atassi, Zilberman, Awad — Oxford)

### 3.1 Bibliographic
- **Title:** "Hardware-Managed Heterogeneous High-Bandwidth Memory and Flash in LLM Inference Systems".
- **Authors:** Hakam Atassi (Member, IEEE), Noa Zilberman (Senior Member, IEEE), Amro Awad (Senior Member, IEEE), all University of Oxford.
- **Venue:** **unknown.** The PDF has no journal header, volume, pages or DOI. It is IEEE two-column, 4 pages (letter-style, plausibly a CAL submission, but this is not stated). The file's download origin (macOS metadata) is Oxford's institutional repository ORA: `https://ora.ox.ac.uk/objects/uuid:a9f80041-ff2e-4714-9bc4-51c513604d36`.
- **Year:** 2026 (from the filename and the reference access dates of 2026-06-22).
- **arXiv:** none. Funding: ARIA, project code NACB-PR01-P04.

### 3.2 Problem
The paper targets single-GPU edge deployment: models such as GLM-4.7 need 661 GiB of FP16 parameters, beyond one GPU's HBM. Naively replacing HBM with HBF exposes the ~1000× higher NAND read latency, which "starv[es] GPU schedulers". Classical DRAM latency fixes (more banks, bigger page buffers, deeper FR-FCFS) do not recover performance, even when combined.

### 3.3 Mechanism / architecture
- **Placement:** HBM and HBF stacks are "integrated directly adjacent to the shoreline of the GPU", i.e. **horizontal shoreline integration**. This is the arrangement H3 rejects.
- **Configuration:** 64 GiB HBM + 1024 GiB HBF (1088 GiB total) with a **single HBF stack**. Channels: HMA has 32 HBM + 8 HBF, versus 40 for HBM-only or HBF-only.
- **Table I** compares with H3:
  - H3: 8 memory modules, ~4 TiB, ~4 TiB/s theoretical BW, higher interposer size and power/heat.
  - Ours: 4 modules, ~1 TiB, ~4 TiB/s, lower interposer size and power/heat.
- **Data placement:**
  - Static in HBM: dense parameters (embedding, output layer), KV cache, activations, and an **HBM expert prefetch buffer**.
  - In HBF: **MoE expert weights**.
- **Who manages placement:** hardware, through a **Migration Manager** at the memory side.
  1. An oracle schedules HBF reads.
  2. The response is written to HBM.
  3. On the write ack, an **HBM residency indirection cache** (page number, valid bit, cache status) is updated. Later reads that hit go to HBM and misses go to HBF.
- **Predictor:** an *ideal oracle* built from memory traces, which migrates layer i+1 during layer i. Incomplete migrations stall execution. "we assume no reuse of pages".
- **Migration page size:** 2 MiB.

### 3.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| HBM stack | 32–64 GiB at 1–2 TiB/s; 1024–2048-bit interface | [6] |
| HBF stack | 512–1024 GiB at projected 1 TiB/s; "roughly 16x" HBM | [4] (= H3) |
| NAND bitline access | ~20 µs, versus DRAM 20–50 ns | [5] Park et al. ASPLOS'21 read-retry, [8] |
| HBF read latency (sim) | **HBF Rd lat = 39,312 cycles** (≈ 27.9 µs at 1.41 GHz **[my arithmetic]**) | "recently proposed HBF timing parameters" [4],[5] |
| HBF page buffer | 16 KiB (HBM 2 KiB) | assumption |
| Banks per channel | HBF 64 (HBM 16) | assumption |
| FR-FCFS depth | 2048 entries (sim table); text: 32 entries per bank "realistic" | assumption |
| Write latency | "hundreds of microseconds but is out of scope" | — |
| Endurance, power | not given (Table I uses arrows only) | — |
| GPU | A100, 1.41 GHz, 108 SMs; PCIe 64 GiB/s, 1 µs; host DRAM 1024 GiB | — |

Note: the paper's background mentions MLC as a density enabler but does not state which cell type the simulated HBF uses.

### 3.5 Evaluation methodology
- **Simulator:** Accel-Sim (cycle-level GPU simulation) of an A100. Traces were collected with NVBit on an Ampere **A5000**.
- **Memory model:** Accel-Sim's DRAM model with modified HBF timing.
- **Models:** MiniMax-M2.5 (230B), GLM-4.7 (355B), Qwen3.5 (397B), Qwen3.5 (30B, fits in HBM), all FP16.
- **Operating point:** **batch 1**, **context limited to 256 tokens**, decode phase only.
- **Baselines:** HBF-only, CPU offload, HBM-only (ideal, infinite capacity), HMA, and HMA with 2 experts mispredicted.

### 3.6 Headline results
- Motivation sweeps on GLM-4.7:
  - banks: slowdown ~1/N, approaching a ~30× limit beyond 128 banks;
  - page buffer 2→32 KiB: 207× → 113×;
  - FR-FCFS: floor ~155× beyond depth 128;
  - composed "realistic" HBF: IPC **0.27×** of ideal HBM.
- Scheduler: issue rate 15.4% (HBM) versus 1.9% (HBF). RAW stalls 64.1% versus 89.0%.
- **HMA is 1.70–3.62× faster, geo-mean 2.79×**, versus HBF-only. CPU offload is geo-mean 2.05× over HBF-only.
- Qwen-30B: HBM-only is **<3.5%** better than HMA.
- Mispredicting 2 experts (mean misprediction rate 23.3%) costs **9.4%** geo-mean for HMA and **19.3%** for CPU offload.
- Tail latency: the p90 access latency drops ">10x", "from tens of microseconds to a few microseconds". Medians are ~1 µs for all configurations.
- Models up to **739 GiB** run on one GPU, "9.2x the 80 GiB maximum of an HBM Only A100".

### 3.7 Stated limitations / future work
- Write latency is "out of scope for this study given the read intensity of LLM inference".
- The oracle predictor and no page reuse are simplifications ("To simplify analysis").
- Future directions:
  - "explore the design space of the Migration Manager, evaluating the impact of the page migration granularity, migration scheduling/ordering, or exposure to pre-existing structures (such as translation look-aside buffers/TLBs)";
  - "novel GPU front-end architectures that are able to tolerate HBF-scale memory latencies";
  - "improved prefetching techniques".

### 3.8 Critical view
- **The operating point is unrepresentative:** batch 1 and 256 tokens. The KV cache is negligible, so the KV-in-HBM decision is never stressed. Datacenter batching, long context and prefill are not evaluated.
- **The oracle predictor is ideal.** A realistic expert predictor is modeled only as a "2 experts missed" perturbation.
- **The GPU is old (A100) and the traces come from a different GPU (A5000).** HBF is modeled by altering DRAM timing parameters (banks, page buffer, FR-FCFS), not by a NAND plane/die/channel model. There is no read-retry, ECC, or plane-conflict modeling, even though the read-retry paper [5] is cited.
- **One HBF stack = 8 of 40 channels.** HBF bandwidth is therefore about 1/5 of the aggregate. The paper does not say whether the stall-on-incomplete-migration cost is bandwidth- or latency-limited.
- **The "power, cost, interposer area" reductions in Table I are qualitative** (arrows), with no numbers.
- **Table I's characterization of H3 (~4 TiB, ~4 TiB/s) does not match H3's own numbers.** H3 reports 3 TB of HBF + 192 GB of HBM and 8 TB/s per tier.
- **No writes at all:** no endurance, and expert updates or fine-tunes are ignored.
- Shoreline integration directly contradicts H3's shoreline argument. Neither paper quantifies the shoreline-site trade-off. The BTP report addresses exactly this.

### 3.9 BibTeX (venue unknown; cite as repository preprint)
```bibtex
@misc{atassi2026hma,
  author       = {Atassi, Hakam and Zilberman, Noa and Awad, Amro},
  title        = {Hardware-Managed Heterogeneous High-Bandwidth Memory and Flash in {LLM} Inference Systems},
  year         = {2026},
  howpublished = {Oxford University Research Archive (ORA)},
  url          = {https://ora.ox.ac.uk/objects/uuid:a9f80041-ff2e-4714-9bc4-51c513604d36},
  note         = {Preprint; publication venue not stated in the manuscript}
}
```

---

## 4. Hot–Cold Tiering of HBM and HBF for Agentic LLM Serving (Baek et al., arXiv 2026)

### 4.1 Bibliographic
- **Title:** "Hot–Cold Tiering of HBM and High Bandwidth Flash for Agentic LLM Serving".
- **Authors:** Jongjin Baek, Won Ji, Seungjae Yoo, Joo-Young Kim. The affiliation is not in the extracted text; it may be on the PDF page, so check before citing an institution.
- **Venue:** arXiv preprint **arXiv:2609.25782v1 [cs.AR], 22 Sep 2026**. "Manuscript submitted to IEEE Computer Architecture Letters, 2026". The header reads "VOL. XX, NO. X", so there is no volume or pages yet.

### 4.2 Problem
Agentic multi-turn sessions idle between tool calls or user turns but must keep their context. Evicting idle KV forces recomputation or slow interconnect transfers on resume. Serving *all* KV from HBF is impractical because of high per-bit read energy and limited endurance.

### 4.3 Mechanism / architecture
- **Observation:** agentic KV is **bimodal**. A small **hot set** (the active decoding batch) is read every step. A large **cold pool** (paused sessions) is read only on resume. At N_a = 48 the hot set is 3% of resident KV capacity but draws 98% of read traffic.
- **Placement:** "each of the memory sites of GPU co-packages an HBF stack on an HBM stack" (Fig. 2). The paper adopts "the co-packaged HBM+HBF substrate proposed by [8], [14], [15]", i.e. H3, Park et al. (pooling) and FlashAccel.
  - The HBM base die has an address router and staging buffers that double-buffer inter-tier traffic.
  - "the HBF is accessible exclusively via the D2D link", i.e. behind HBM. The GPU addresses HBM directly.
- **Four data paths:**
  1. Decode: weights + active KV go from HBM to the GPU.
  2. Resume: KV moves HBF→HBM over D2D.
  3. New KV is written to HBM.
  4. Eviction: idle KV moves HBM→HBF.
- **What lives where:**
  - **HBM:** weights (all experts, to avoid MoE routing variance), active KV, recent idle KV.
  - **HBF:** demoted idle KV.
- **Write policy: write-on-evict.** Because KV is immutable, a block is written to HBF at most once. A re-admitted and re-evicted block is not rewritten.
- **Management:** runtime/scheduler level. There is LRU eviction at 1.5 MiB KV-block granularity, with no hard pinning ("Pinning may deadlock"). Resume fetches overlap with CPU-side tokenization, and decode prefetches one step ahead.
- **Lifetime model (Eq. 1):** T_life = C·N_PE / (R_W · WAF).

### 4.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| HBF per-stack read BW (background) | ~1,638 GB/s, "comparable to that of an HBM4 stack" | [12] Ma & Patterson |
| HBF per-stack capacity (background) | ~512 GB, versus ~48–64 GB for HBM4 | [12] |
| Evaluated HBF tier | **3 TiB SLC**, "bandwidth matching HBM" | — |
| P/E | **100,000** cycles | [19] Micron |
| HBF read access latency | **25 µs** | — |
| Read granularity | "page-granular (tens of KB)" | [8] H3 |
| WAF | ≈1.02 (versus 2–4 for a general SSD) | [6] Kyung, [20] HiFC; [21] |
| HBF read energy | swept **8–30 pJ/bit**, nominal 20 pJ/bit "based on the power budget in [8]" | [7] HAVEN, [8] H3 |
| HBM read energy | ≈3.5 pJ/bit | [24] RPU |
| GPU | B200: 192 GiB HBM3e, 8 TB/s, 2.25 PFLOP/s | [18] |
| Program latency | not stated | — |
| Warranty target | 5 years | [6] Kyung |

Note: 160 W / (1 TB/s × 8 bit) = 20 pJ/bit **[my arithmetic]**, which matches H3's 160 W per-cube TDP. The "20 pJ/bit" is thus derived from H3's TDP assumption, not from a device measurement.

### 4.5 Evaluation methodology
- **Simulator:** in-house, trace-driven, with analytical latency and power models.
- **Traces:** mini-swe-agent trajectories on SWE-bench Verified.
- **Model:** Qwen3-Coder-30B-A3B (MoE; 56.8 GiB BF16; 96 KiB KV per token).
- **Idle gaps:** heavy-tailed, median ~8 s and mean ~23 s. The workload is scaled to thousands of sessions by replaying trajectories with independent phases.
- **Sessions:** peak context ~16.7k tokens (1.53 GiB). The mean resident is 0.82 GiB.
- **Concurrency:** N_a (decoding sessions) swept from 8 to 128. TBT SLO is 50 ms.
- **Baselines:** all-KV-to-flash, CPU offload over PCIe 5.0 (64 GB/s), NVLink-C2C (450 GB/s), and recomputation.
- **Power model:** read power only (rate × pJ/bit).

### 4.6 Headline results
- **Capacity:** HBM-only saturates at **165 sessions**; with 3 TiB HBF it holds **3,900** ("24×").
- **Resume overhead at N_a = 48:** **HBF 0.084 ms**, NVLink-C2C 1.0 ms, PCIe 7 ms, recompute 14 ms. Recompute "over doubl[es] turn cost at N_a=128". The abstract says "≈0.1 ms of resume latency".
- **TBT:** 14 ms at N_a = 48 (72 tok/s) and 27 ms at N_a = 128 (37 tok/s), both within the 50 ms SLO. PCIe and NVLink-C2C without tiering violate the SLO.
- **Lifetime:**
  - all-KV-to-flash: ~11 years at N_a = 48 down to ~8 years at 128 ("clears the 5-year warranty");
  - write-on-evict: "effectively unbounded" below N_a = 32 and ~20 years at 48. The two policies converge at high N_a.
- **Read rate from HBF:** 6 GB/s at N_a = 16 to 38 GB/s at 128.
- **Power:** at 20 pJ/bit, write-on-evict saves about **950 W per device, i.e. 7.6 kW per 8-GPU node**, versus all-KV-to-flash. Across the swept energies the saving is **2.1–12.2 kW per node**.

### 4.7 Stated limitations / future work
- "Due to scarce public agent traces, we scaled the workload".
- "Since HBF read energy is an unstandardized value, we treat it as a parameter".
- The KV-immutability assumption "exclud[es] policies that rewrite KV in place such as in-session compression".
- Positioning: the design "suits premium latency-sensitive serving rather than throughput-maximizing batch inference".
- No explicit future-work section.

### 4.8 Critical view
- **Small model (30B-A3B, 56.8 GiB) whose weights fit in HBM.** The capacity problem that motivates the BTP (weights > HBM) is not addressed. With 405B-class weights HBM would hold no hot set, and the hierarchy would be very different.
- **The 25 µs read latency is barely relevant** because HBF reads are bulk resume transfers. The 0.084 ms resume overhead depends on bandwidth "matching HBM", i.e. 8 TB/s over the D2D link, which is unjustified for a behind-HBM topology.
- **Power model is read-only.** No program/erase energy, no idle/leakage, and no HBF TDP (H3 assumes 160 W per stack). The "7.6 kW savings" compares against a strawman (all-KV-to-flash) that no prior work literally proposes for hot decode reads. Son et al. do place all KV in HBF, but under their own SLO model.
- **Lifetime assumes WAF ≈ 1.02** because KV is appended in erase-block-aligned units. At 1.5 MiB tracking blocks versus multi-MB erase blocks, and with sessions ending at random times, invalidation creates partially valid erase blocks, so GC is *not* free. This is not modeled, and there is no static or dynamic wear-leveling analysis. The BTP report finds wear concentration can change lifetime by up to 10.7×.
- **Endurance numbers conflict with Son et al.** (see cross-paper section).
- Only average TBT at steady-state plateau is reported; no tail TBT or TTFT distribution. The arrival model is synthetic replay.
- The HBM–HBF D2D link bandwidth and the staging-buffer size are not specified.

### 4.9 BibTeX
```bibtex
@misc{baek2026hotcold,
  author        = {Baek, Jongjin and Ji, Won and Yoo, Seungjae and Kim, Joo-Young},
  title         = {Hot--Cold Tiering of {HBM} and High Bandwidth Flash for Agentic {LLM} Serving},
  year          = {2026},
  eprint        = {2609.25782},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {Submitted to IEEE Computer Architecture Letters}
}
```

---

## 5. Exploring HBF for Modern LLM Inference: Opportunities and Challenges (Son et al. — POSTECH/ETH, IEEE CAL 2026)

### 5.1 Bibliographic
- **Title:** "Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges".
- **Authors:** Dowon Son, Yonggon Park (GSM, IEEE), Hyunuk Cho, Hyungkyu Ham, Onur Mutlu (Fellow, IEEE), Sungjin Lee, Gwangsun Kim (Member, IEEE), Jisung Park (Member, IEEE). POSTECH; Mutlu is at ETH Zürich. Corresponding author: Jisung Park.
- **Venue:** IEEE Computer Architecture Letters, vol. 25, no. 2, Jul.–Dec. 2026, pp. 251–254.
- **DOI:** 10.1109/LCA.2026.3705817.
- **Dates:** received 11 May 2026; revised 9 Jun 2026; accepted 15 Jun 2026; published 22 Jun 2026.
- **Funding:** NRF RS-2025-00519994, RS-2024-00415602; IITP RS-2026-25529760, RS-2024-00437866, RS-2024-00347394.

### 5.2 Problem
This is the first systematic study of HBF as a *main* GPU memory, replacing most or all HBM stacks and serving both reads and writes. It asks under what conditions (GPU count, SLO, context length, dense versus MoE) HBF helps, and what bottlenecks remain.

### 5.3 Mechanism / architecture (configurations, Table I)
All configurations share a state-of-the-art GPU core (NVIDIA DGX Rubin NVL8 [5]; the text cites a "high-end GPU ... provides only 288 GB").
- **HBM4:** 8 HBM stacks, each 36 GB and 1.6 TB/s.
- **HBF:** 7 of the 8 HBM stacks are replaced by HBF (512 GB, 1.6 TB/s read). **One HBM stack is kept for intermediate data.**
- **HBF+:** all 8 are HBF. Intermediate data goes in a **40 MB SRAM on the logic die** (320 MB per GPU; the SRAM size is taken from H3 [8]). This supports, e.g., batch 327 on Llama 4 Maverick.
- **CONV / CONV+:** like HBF / HBF+ but with bandwidth projected from existing low-latency NAND.
- **Placement:** HBF stacks replace HBM stacks **directly on the shoreline**. There is no daisy chain.
- **What lives where:** **weights and KV cache are both in NAND** in HBF and HBF+.
- **Staging:** as in H3, each HBF stack has a **3.13 MB SRAM staging buffer** on its logic die for prefetch and double buffering.
- **Management:** static placement. The simulator picks the throughput-maximizing parallelism (DP/TP/PP/EP) under the SLO, capacity and SRAM constraints.
- Footnote: writing intermediate data to NAND "increases write traffic by up to 100× for every decode iteration".

### 5.4 Device parameters
| Parameter | HBF / HBF+ | CONV / CONV+ | Source |
|---|---|---|---|
| Capacity per stack | 512 GB (">14×" HBM4's 36 GB) | same | Sandisk projection |
| Read BW per stack | 1.6 TB/s | **0.35 TB/s** | [1] Ma et al.; projected from [9],[10] |
| Planes per die | 25 | 16 [10] | assumption |
| Page read latency | **1 µs for 4 KiB** | **3 µs** [9] (Z-NAND) | assumption |
| Page program latency | **100 µs** (all configs) | 100 µs | based on [9] |
| Power per stack | "< 80 W" versus HBM 40 W | — | [1] |
| Endurance | SLC 100K P/E | — | — |
| Warranty horizon | 3 years ("a common GPU-warranty period") | — | — |
| Existing parts | Z-NAND 87.4 GB/s; XL-Flash 262 GB/s (16 dies each) | — | [9],[10] |
| Sandisk latency target | "µs-order latency for 4-KiB reads" | — | [1] |

Note: 2 × 1.6 TB/s × 1 µs ≈ 3.2 MB **[my arithmetic]**, consistent with the 3.13 MB staging buffer via H3's Eq. (1). The exact derivation of 3.13 MB is not stated.

### 5.5 Evaluation methodology
- **Simulator:** LLMSimulator [12] (from the Duplex, MICRO'24 group), extended with:
  1. asymmetric read/write bandwidth timing;
  2. continuous batching and disaggregated prefill–decode, including the KV-write cost when prefill outputs land on decode nodes;
  3. chunked attention and pipeline parallelism.
- The authors say they "will open-source" the extended simulator.
- **Assumptions:**
  - decode nodes only ("prefill phase is inherently compute-bound and thus benefits little from HBF's large capacity");
  - continuous batching;
  - best parallelism per configuration;
  - NVLink 1,800 GB/s intra-node (≤8 GPUs) and InfiniBand 100 GB/s inter-node.
- **Models:** Llama 3 405B (dense) and Llama 4 Maverick (MoE; "746 GB").
- **Workloads (L_IN, L_OUT):**
  - Short: ⟨1,660, 373⟩ (ShareGPT);
  - Mid: ⟨5.9K, 499⟩ (LongBench);
  - Long: ⟨103.5K, 1.1K⟩ (∞Bench English summarization).
- **Scale and SLO:** 1–16 GPUs. TPOT SLO 0.1 s, plus 0.05 s and an offline (24 h) scenario. Constant arrival rate equal to the completion rate.

### 5.6 Headline results (ten observations, five takeaways)
- **Batch at 8 GPUs:** HBF **1.3–4.5×** and HBF+ **1.7–5.3×** versus HBM4.
  - HBM4 needs ≥4 GPUs to serve both LLMs. One GPU with HBF/HBF+ usually supports a larger per-GPU batch than 8-GPU HBM4.
- **CONV/CONV+:** often smaller batches than HBM4, and up to **95%** of HBF capacity unused (Llama3 Short).
- **Dedicated HBM stack (HBF) versus SRAM (HBF+):** HBF+ has a **24%** larger batch than HBF on average at 8 GPUs (Mid).
- **Context length:** HBF+ gains are 124% / 200% higher in Mid / Long than in Short.
- **Throughput:** 4-GPU HBF+ has **15%** higher per-GPU TPS than 8-GPU HBM4 (Llama4 Long).
- **KV write cost:** KV writes take **5–13.9%** of HBF+ execution time on Llama4. The share grows with more GPUs, shorter contexts and MoE.
- **SLO sensitivity:** relaxing the SLO helps HBF. For Llama4 at 16 GPUs, the offline TPS gain rises "from 4.1% to 14.8%".
- **Endurance:** per-block PEC over 3 years is "**far exceeding 100 K in most cases**".
- **Takeaways:**
  1. Benefits depend critically on read bandwidth.
  2. Throughput and flexibility improve and the minimum GPU count drops.
  3. Improving KV write performance helps.
  4. HBF is best for throughput-oriented or offline serving.
  5. "NAND flash endurance remains a critical challenge".

### 5.7 Stated limitations / future work
- "achieving HBF's target bandwidth requires significant advancements over existing low-latency NAND flash technologies". The CONV configurations were introduced for this reason.
- Power "could also cause thermal issues. However, we believe that this is primarily a power-delivery and thermal-management challenge rather than a fundamental limitation of HBF."
- Endurance: "Writing the KV cache in a retention-relaxed manner [16] could mitigate the endurance problem, but ... significant improvements in NAND flash endurance are still critical".
- Future direction: "We hope that our findings help guide future research toward efficient and practical HBF-based LLM-serving systems."

### 5.8 Critical view
- **HBF latency is optimistic:** 1 µs per 4 KiB page read. Other papers use 15–25 µs, Kyung uses 3 µs, and existing Z-NAND is 3 µs. The paper admits the target needs NAND advances, but the headline results use it.
- **No hybrid tiering:** HBF (7+1) keeps HBM only for intermediates, and KV/weights are never tiered between HBM and HBF. The one HBM stack's bandwidth "remains largely underutilized". There is no hot/cold split and no HBM caching of hot KV.
- **Endurance counts KV for every processed token** with no WAF/GC stated in the text. It does not consider write-avoiding policies (e.g. Kyung's selective placement, write-on-evict, KV kept in HBM until eviction).
- **Power:** "<80 W" is cited but no energy or throughput-per-watt result is reported. There is no thermal analysis.
- **Prefill is excluded** (decode nodes only). No MLA models. No tail latency beyond the TPOT SLO.
- **Cost:** "reduce the minimum GPU requirement" is argued, but there is no $ model.
- **Shoreline:** stacks are simply swapped 1:1, so the question of whether HBF should cost an HBM site never arises.

### 5.9 BibTeX
```bibtex
@article{son2026exploring,
  author  = {Son, Dowon and Park, Yonggon and Cho, Hyunuk and Ham, Hyungkyu and Mutlu, Onur and Lee, Sungjin and Kim, Gwangsun and Park, Jisung},
  title   = {Exploring High-Bandwidth Flash for Modern {LLM} Inference: Opportunities and Challenges},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {2},
  pages   = {251--254},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3705817}
}
```

---

## 6. High-Bandwidth Flash for KV Caches: Endurance and Performance Implications (Kyung et al. — SNU, IEEE CAL 2026)

### 6.1 Bibliographic
- **Title:** "High-Bandwidth Flash for KV Caches: Endurance and Performance Implications".
- **Authors:** Kwanhee Kyung, Yeon Ji Moon, Juhwan Cho, Jung Ho Ahn (Senior Member, IEEE). Seoul National University. Corresponding author: Jung Ho Ahn.
- **Venue:** IEEE Computer Architecture Letters, vol. 25, no. 1, Jan.–Jun. 2026, pp. 210–213.
- **DOI:** 10.1109/LCA.2026.3695938.
- **Dates:** received 30 Mar 2026; revised 18 May 2026; accepted 19 May 2026; published 22 May 2026.
- **Funding:** Samsung Advanced Institute of Technology (SAIT); MSIT/IITP RS-2024-00456287.

### 6.2 Problem
The KV cache is the capacity and bandwidth bottleneck. Offloading KV to CPU memory is limited by the interconnect (NVLink-C2C 900 GB/s versus 8 TB/s HBM). Can KV live in HBF without destroying endurance? The key quantity is the KV **read-to-write ratio** ρ_R/W.

### 6.3 Mechanism / architecture
- **Analytical model:**
  - V_write = (L_in + L_out − 1)·V_KV
  - V_read = Σ_{i=1..L_out} (L_in + i − 1)·V_KV
  - **ρ_R/W = L_out(2L_in + L_out − 1) / (2(L_in + L_out − 1))**
- **Observation 1:** for fixed context length C, ρ increases monotonically with L_out.
- **Observation 2:** ρ generally grows with context length, and L_out growth is the critical factor. Chain-of-Thought therefore raises ρ.
- **Proposed placement:** *selectively* place long-context, high-L_out requests in HBF, using L_out prediction on arrival ([5] S³).
- **Device model:** "device_HBF" employs **five 512 GB HBF stacks** (no HBM) to match B200-class 8 TB/s read. "device_HBM" has 256 GB of HBM4 plus CPU memory via NVLink-C2C. HBF is attached directly as device memory.
- **On-chip SRAM** ("sufficient") serves three roles:
  1. **double buffering** for read-latency hiding [3] (H3). Latency is exposed only when loading the first activated MoE expert;
  2. **KV write buffering**: KV for each output token is buffered and stream-written "once per decode stage";
  3. activation buffering.
  - Aggregate on-chip bandwidth is 16.8 TB/s per device. The same assumptions are applied to device_HBM.
- **Request-local KV placement:** each request's KV sits in a contiguous logical HBF region, extended in token order. Reads are request-local sequential streams and writes are "logical append-only sequential write streams". This "limits garbage-collection and wear-leveling intervention [9]", so WAF = 1.02 [4] (HiFC).
- **Lifetime:** TBW = ((V_HBF·N_device) − V_weight)·E_SLC. Lifetime is extrapolated from 1,000 decode stages.

### 6.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| HBF capacity per stack | 512 GB | [7] Ma & Patterson, [12] Sandisk |
| HBF read BW per stack | 1.6 TB/s sequential | [7],[12] |
| Cell type / endurance | SLC, 100,000 P/E | [8] Micron |
| tR | **3 µs** (Z-NAND) | [2] Cheong ISSCC'18 |
| tPROG | **100 µs** (Z-NAND) | [2] |
| Write BW per stack | BW_read × tR/tPROG = **48 GB/s** | derived |
| Cell-to-XPU latency | simplified to tR | assumption |
| Capacity ratio | "∼10×" HBM | — |
| Warranty | 5 years | [11] Samsung PM893 |
| GPU | B200-class: 8 TB/s, 2,250 TFLOPS; NVLink5 900 GB/s/dir (NVLink6 1,800 GB/s) | — |
| Power | **not modeled** | — |

Motivating number: a 512 GB SLC stack at 1.6 TB/s read and 48 GB/s write under "continuous read-write alternation would exhaust the HBF lifetime in under two weeks".

### 6.5 Evaluation methodology
- **Simulator:** LLMSimulator [1], modified for HBF.
- **System:** disaggregated serving; the decode node has 8 devices in an NVL72-like topology. Continuous batching.
- **Models:** Llama 4 Maverick (400B, GQA, 192 KB KV per token in BF16) and DeepSeek-R1 (671B, MLA, 68.6 KB per token). BF16 weights and KV.
- **MoE routing:** Zipfian token-to-expert skew 0.8. Gated experts are sharded one per device. Datasets are synthesized.
- **Lifetime study:** context ~ truncated normal (μ ± 2σ), L_out ratio ~ beta. Two dispersion settings: CV 0.1 with κ 90, and CV 0.3 with κ 30. Weights + reserved KV = 2× total HBM capacity.
- **Performance study:** 8K versus 32K context, L_in:L_out = 1:3. Weights + reserved KV = 2× (512 GB per device) or 3× (768 GB per device). TPOT SLO 200 ms (Splitwise); SLO-adjusted throughput. A 1/2-HBF sensitivity case is included.

### 6.6 Headline results
- Up to **4.9×** higher decode token-generation throughput versus HBM + CPU memory (intro).
- **Lifetime, intro version:** 32K-context requests extend effective HBF lifetime "by up to **33×** and **31×**" versus 8K (Llama 4 Maverick, DeepSeek-R1).
- **Lifetime, Section V-A version:** "increasing the context length from 8 K to 32 K extends the lifetime by approximately **4×** for Llama 4 Maverick and **3×** for DeepSeek-R1".
  - **Note the internal inconsistency.** The likely reading is that 33×/31× compares the best 32K point with the worst 8K point across L_in:L_out ratios, while 4×/3× is the typical gain along context length. The paper does not reconcile the two, so cite carefully and state which comparison you mean.
- A larger L_out can beat a longer context: L_in:L_out = 1:1 at 8K gives longer life than 15:1 at 32K.
- A higher WAF lowers absolute lifetime but not the relative trend.
- **Throughput:**
  - device_HBM with NVLink5 is always SLO-bound.
  - With NVLink6 in the 512 GB case, device_HBM meets the SLO but throughput is up to **3.0×** (Maverick) and **2.8×** (DeepSeek-R1) lower than HBF.
  - In the 768 GB case device_HBM is again SLO-bound.
  - HBF's advantage holds even at 1/2 read/write bandwidth.

### 6.7 Stated limitations / future work
- "we focus on standard full attention". The discussion says sparse attention (eviction-free) makes HBF "particularly attractive", but this is not evaluated.
- The monotonicity "is a mathematical property of (1) under fixed C", since real workloads vary.
- No explicit future-work list.

### 6.8 Critical view
- **"Sufficient on-chip SRAM" is assumed.** SRAM requirements are reported (Fig. 5), but the area/power feasibility on the GPU or base die is not closed. Writing KV "once per decode stage" needs per-token buffering across all layers.
- **device_HBF is pure HBF (five stacks) with no HBM.** Hybrid HBM+HBF with hot KV in HBM is not considered, even though that would reduce both writes and read energy.
- **TBW assumes perfect wear leveling** over all non-weight capacity (ideal wear distribution) and WAF = 1.02 from HiFC (an SSD-based KV swap system). Request-local contiguous allocation plus request completion leaves invalid regions. GC under heterogeneous lengths is not simulated, and lifetime is extrapolated from 1,000 decode stages. Weights are written once and never re-programmed, but retention and read disturb of read-hot blocks are not addressed.
- **Selective placement relies on L_out prediction**, and predictor accuracy and its effect on lifetime are not evaluated. The "where do non-selected requests go" path (presumably HBM/CPU) is not modeled in the performance study.
- **No power or energy**, despite HBF's much higher per-bit read energy (Hot–Cold uses 8–30 pJ/bit).
- **tR = 3 µs is Z-NAND.** Other papers use 15–25 µs. Latency is simplified to tR with no ECC or read-retry.
- Prefill is assumed on a separate high-performance node. The prefill KV write *into HBF* (the L_in term) is in the model, but its bandwidth cost at 48 GB/s is not isolated in the text.

### 6.9 BibTeX
```bibtex
@article{kyung2026hbfkv,
  author  = {Kyung, Kwanhee and Moon, Yeon Ji and Cho, Juhwan and Ahn, Jung Ho},
  title   = {High-Bandwidth Flash for {KV} Caches: Endurance and Performance Implications},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {1},
  pages   = {210--213},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3695938}
}
```

---

## 7. Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy (Peng et al. — Huawei, arXiv 2026)

### 7.1 Bibliographic
- **Title:** "Enabling High-Bandwidth Flash for Generative Recommendation Serving with Write-Aware KV Cache Policy".
- **Authors:** Danni Peng, Kai Wu, Tianyu Zuo, Pengfei Xia, Hui Zang (Huawei Technologies Co., Ltd.). Corresponding author: peng.danni@huawei.com.
- **Venue:** preprint, **arXiv:2609.07175v1 [cs.AR], 7 Sep 2026**. "This work has been submitted to the IEEE for possible publication" (the venue is not named).

### 7.2 Problem
Generative recommendation (GR) reuses user-level KV caches across requests.
- Capacity: for HSTU-10B with 10K users, keeping only the hottest 25% of users needs **3.62 TB**.
- Bandwidth: a 20 ms target with an 8K-interaction history needs **403 GB/s**.

HBF fits both needs, but conventional LRU couples every miss to a full KV write ("write-on-miss"). A 7-stack HBF device then lasts "only about one year".

### 7.3 Mechanism / architecture
- **Device:** 7 HBF stacks + 1 HBM4 stack, reusing Son et al.'s "HBF" configuration [7]. HBM holds frequently updated intermediate activations. **Model weights and user KV are in HBF.** HBF replaces HBM stacks directly on the device.
- **Policy:** **admission-controlled LRU-K** (O'Neil et al. 1993). A KV entry is admitted only if its K-th most recent request is more recent than the oldest K-th request among cached entries. This splits the miss path into miss-write and miss-no-write (recompute only, no persist).
- **Analytical model:**
  - T_hit = max{T_comp(inc+decode), T_read(hist), T_write(inc)}
  - T_miss = max{T_comp(hist+inc+decode), T_write(hist+inc)}
  - Average latency and QPS = 1/T̄_req
  - Lifetime L_HBF = C_KV·E_flash / (λ_wload · D̄_write,req)
- **Request model:** Poisson requests with a characteristic-time approximation. p_hit = Pr(X ≥ K), and a_K = Pr(X = K−1 | X < K).
- **Who manages:** a software caching policy in the serving system.

### 7.4 Device parameters
| Parameter | Value | Source |
|---|---|---|
| HBM4 stack | 36 GB, 1.6 TB/s read/write | [7] Son |
| HBF stack | 512 GB, **1.6 TB/s read, 48 GB/s write** | [8] Kyung |
| HBF die | 25 planes per die, 16 dies per stack; **1 µs per 4 KB page read** | following [7] |
| Staging SRAM | 3.13 MB per HBF stack on the logic die | [6] H3, [7] Son |
| Endurance | SLC, **100K P/E** | [9] Micron |
| Power | HBF "<80W" versus HBM 40 W (qualitative) | — |
| Compute | 2 PFLOPs peak for all configurations | — |
| HBM+CPU | 2 TB CPU memory over PCIe 6.0 at 128 GB/s, write-through | — |
| Warranty | "typical flash device warranty periods" [11] (Samsung PM893) | — |

### 7.5 Evaluation methodology
- **Model:** analytical (roofline).
- **GR models:** HSTU 1B, 10B, and a 100B MoE variant; OpenOneRec 1.7B, 8B, and a 100B MoE variant.
- **Traffic:** users span activity levels; hot users send >100 req/hr. Max history 8K (more active users have longer histories). Candidate size 500.
- **User scale:** model-specific, sized to retain hot users in HBM+CPU.
- **Sweep:** K from 1 to 20.

### 7.6 Headline results
- **QPS:** HBF-based is **3.8–4.7×** higher than HBM-only. HBM+CPU is 1.3–2.0× higher than HBM-only for 5 of 6 models; HSTU-1B is slightly worse because fetching from CPU is slower than recomputing.
- **Lifetime:** LRU-1 gives **about 1 year**. **LRU-10 gives more than 6 years** for all workloads.
- **HSTU-10B:** p_wr falls from **0.101 to 0.014** going from K = 1 to K = 10.
- QPS slightly *increases* with K, because hotter users with longer histories stay resident.

### 7.7 Stated limitations / future work
- Tier-specific modeling details are "omitted here for brevity".
- "Although LRU-K involves tracking access history for admission decisions, its endurance gains outweigh the added management overhead". The overhead is not quantified.
- No explicit future-work section.

### 7.8 Critical view
- **Purely analytical, with no real traces:** Poisson arrivals and the characteristic-time approximation assume independent users and stationarity. Real recommendation traffic is bursty and diurnal.
- **HBF read latency and page granularity are absent from the latency equations** (bandwidth only). Neither is the HBM→HBF write path contention with reads.
- **Incremental KV appends (D_inc, a few tokens) are small writes.** Program-page granularity (e.g. 16 KB) forces buffering or write amplification, and the model assumes no WAF. GC and invalidation on eviction are not modeled, and neither is wear leveling.
- **No power/energy evaluation.** The "<80 W" and "offset at the system level" claims are unsupported.
- Only one hardware point (7+1). No hybrid hot-KV-in-HBM tier and no multi-device scaling.

### 7.9 BibTeX
```bibtex
@misc{peng2026grhbf,
  author        = {Peng, Danni and Wu, Kai and Zuo, Tianyu and Xia, Pengfei and Zang, Hui},
  title         = {Enabling High-Bandwidth Flash for Generative Recommendation Serving with Write-Aware {KV} Cache Policy},
  year          = {2026},
  eprint        = {2609.07175},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR}
}
```

---

## 8. Cross-paper observations

### 8.1 Summary table: topology and placement
| Paper | Where HBF attaches | HBM:HBF per GPU | Weights | Shared/prefix KV | Generated/active KV | Placement manager |
|---|---|---|---|---|---|---|
| H3 (Ha) | Daisy-chained **behind HBM base die**, unified address space | 8 HBM3e (192 GB) + HBF (3 TB), 8 TB/s each | HBF | HBF | HBM | DL framework + tensor-level prefetch hints to an SRAM LHB |
| Pooling (Park) | Cascaded **behind HBM via interposer**, Custom Base Die | 8×24 GB HBM + 8×512 GB HBF | **HBM** (MoE unpredictability) | — | HBF, staged through HBM read/write buffers | **Hardware (CBD)**, transparent |
| HMA (Atassi) | **Side by side on the shoreline** | 64 GiB HBM (32 ch) + 1 × 1024 GiB HBF (8 ch) | dense in HBM; **experts in HBF** | — | HBM | **Hardware** migration manager + oracle |
| Hot–Cold (Baek) | HBF co-packaged **behind HBM** (via D2D, base-die router), adopting [H3, Park, FlashAccel] | B200 192 GiB + 3 TiB HBF | HBM | — | hot in HBM; idle/cold in HBF (write-on-evict) | runtime LRU + base-die staging |
| Exploring (Son) | **Replaces HBM stacks on the shoreline** | 7 HBF + 1 HBM4, or 8 HBF + SRAM | HBF | — | HBF | static; simulator picks parallelism |
| KV endurance (Kyung) | **Replaces HBM** (5 HBF stacks per device) | 5 × 512 GB HBF, no HBM | HBF | — | HBF (selective, high-ρ requests) | static + L_out prediction |
| GR (Peng) | Replaces HBM (7 HBF + 1 HBM4) | as Son | HBF | user KV in HBF | incremental KV in HBF | software LRU-K admission |

### 8.2 Contradictions in device assumptions
1. **HBF read latency ranges over 25×, from 1 µs to 27.9 µs:**
   - 1 µs per 4 KiB page: Son, Peng;
   - 3 µs: Kyung (Z-NAND tR);
   - 15 µs tR: Park;
   - 20 µs: H3 (commodity SLC);
   - 25 µs: Baek;
   - 39,312 cycles ≈ 27.9 µs at 1.41 GHz: Atassi **[my arithmetic]**.

   LHB and staging sizes scale linearly with latency: H3 gets 40 MB per cube at 20 µs, while Son and Peng get 3.13 MB per stack at 1 µs. Atassi's key finding (latency starves GPU schedulers) would largely disappear under Son's 1 µs assumption. Nobody reconciles these values or models the read-retry/ECC tail, even though Atassi cites the read-retry paper.
2. **HBF per-stack bandwidth:**
   - 1 TB/s: H3; Atassi ("1 TiB/s");
   - 1.6 TB/s: Son, Kyung, Peng;
   - 1,638 GB/s: Baek background (but "matching HBM", i.e. 8 TB/s aggregate, in evaluation);
   - unspecified: Park (only the 2 TB/s UCIe PHY).

   Son's CONV projection from real parts is **0.35 TB/s**, and Son shows that at this bandwidth HBF is often *worse* than HBM4. Only H3 (½ BW), Kyung (½ BW) and Son (CONV) test bandwidth sensitivity.
3. **HBF power differs by 2×:**
   - H3: **160 W per cube** (4× HBM3e's 40 W);
   - Son and Peng: **"< 80 W"** versus 40 W, citing Ma et al.;
   - Baek: 8–30 pJ/bit versus 3.5 pJ/bit. The nominal 20 pJ/bit equals H3's 160 W / 1 TB/s **[my arithmetic]**, so H3's TDP assumption propagates into Baek's power claims;
   - Kyung and Atassi: no power.

   No paper models program/erase energy, idle power, or thermals. Park and Son explicitly defer thermal analysis.
4. **Program latency / write bandwidth:** 100 µs tPROG in Son, Kyung and Park. Write bandwidth is 48 GB/s per stack (Kyung, reused by Peng); H3, Baek and Atassi give none. The 48 GB/s number is *derived* (read BW × tR/tPROG) with the aggressive 3 µs tR. With Park's 15 µs tR the same formula would give ~240 GB/s **[my arithmetic]**, i.e. the formula is not physically grounded: write BW depends on plane parallelism and tPROG, not on tR.
5. **Capacity ratio:** "16×" (H3, Atassi), ">14×" (Son: 512 vs 36 GB), "∼10×" (Kyung), "over 10×" (Peng), "orders-of-magnitude" (Baek abstract). Per-stack HBF capacity also varies: H3 implies ≈384 GB per cube (3 TB / 8, **[my arithmetic]**); others use 512 GB; Atassi uses 1024 GiB.
6. **Page/access granularity is never consistent:**
   - Son and Peng: 4 KiB read page;
   - Park and Atassi: 16 KB page (Atassi: 16 KiB page buffer; 2 MiB migration unit);
   - Baek: "tens of KB", with 1.5 MiB tracking blocks;
   - Park: ~20 MB erase block, 120 MB flush unit.

   None evaluates the mismatch between a vLLM-style KV block (e.g. 16 tokens × per-token KV) and NAND program/erase units.
7. **Endurance conclusions contradict each other:**
   - **Son:** PEC "far exceeding 100 K in most cases" over **3 years**, i.e. HBF-for-KV is endurance-infeasible without large improvements.
   - **Kyung:** HBF-for-KV meets a **5-year** warranty *if* requests are selectively placed by L_out.
   - **Baek:** even **all-KV-to-flash** lasts 8–11 years.
   - **Peng:** conventional LRU gives about 1 year, LRU-10 gives >6 years.

   The differences come from write-rate assumptions (Son: high-TPS, short-context, MoE, writes on every admitted query; Baek: TBT-capped N_a ≤ 128 with a small 30B model), from the warranty horizon (3 vs 5 years), and from WAF (1.02 assumed by Kyung and Baek, unstated by Son and Peng). No paper models GC, static wear leveling, or wear concentration. H3 and Park give no endurance numbers at all. The BTP report's finding that static wear leveling changes lifetime by up to 10.7× is not addressed anywhere in this set.
8. **MoE weight placement contradicts across papers:**
   - Park keeps *all* weights in HBM because MoE routing makes prefetch unreliable.
   - Atassi puts *experts* in HBF and shows a 2-expert misprediction costs only 9.4%.
   - Kyung puts all weights in HBF; double buffering exposes latency only on the first activated expert.
   - Son puts Llama 4 Maverick weights in HBF.
   - Baek keeps all experts in HBM "to eliminate routing-induced latency variances".

   There is no shared evaluation of expert-prediction accuracy versus HBF latency.
9. **Shoreline argument:** H3 dismisses shoreline side-by-side placement because it costs HBM sites. Atassi *chooses* the shoreline and claims lower interposer area and power than H3 (qualitative arrows). Son, Kyung and Peng simply swap HBM sites for HBF. No paper quantifies the trade (sites × bandwidth × capacity). This is the BTP report's contribution.
10. **Characterizations of H3 differ from H3:**
    - Atassi's Table I says H3 has ~4 TiB and ~4 TiB/s; H3 itself reports 3 TB HBF + 192 GB HBM at 8 TB/s per tier.
    - Park's "H3-like" baseline is an SRAM prefetch with a 40% hit rate, which is not what H3 describes.
    - Baek says "[8] restricts storage to read-only state", which is correct.
11. **Baselines inflate gains unevenly:**
    - Park's 87.4× is against SSD/host-DRAM spill.
    - Atassi's 2.79× is against *HBF-only*, not against HBM.
    - H3's 2.69× is throughput/TDP.
    - Son's gains are per-GPU batch or TPS normalized to 8-GPU HBM4.
    - Kyung's 4.9× is against HBM + NVLink-C2C CPU memory.

    The headline multipliers are therefore not comparable.
12. **Internal inconsistency in Kyung:** 33× / 31× (intro) versus ~4× / 3× (Section V-A) lifetime extension for 32K versus 8K. Cite with care.

### 8.3 Design-space points nobody in this set explores
- **The die mix inside a hybrid stack**, i.e. HBM and HBF dies in one stack or a configurable HBM:HBF die ratio. H3 lists "hybrid stacking" as future work. Everyone uses fixed stack-level ratios (8+8, 7+1, 8+0, 32:8 channels).
- **Right-sizing HBF capacity to the workload.** Every paper uses 512 GB–1 TB per stack, and none checks utilization. Son notes 95% underutilization only for the bandwidth-starved CONV. BTP finds 17–67× over-provisioning.
- **Shared-link bandwidth for behind-HBM topologies** (H3, Park, Baek). No paper models HBF traffic competing with HBM traffic on the GPU–HBM link, or the HBM bandwidth consumed by staging (Park writes prefetched KV into HBM and then reads it). Pooled versus partitioned link scheduling is unstudied.
- **GC under realistic KV lifetimes**, i.e. requests or sessions ending and invalidating blocks at mixed times. Also over-provisioning, block-allocation granularity (per-request versus shared blocks), and static wear leveling. Kyung and Baek assume WAF = 1.02, Park flushes erase-block-sized entries, and H3 claims GC is reduced. None simulates it.
- **Read disturb and retention** for read-hot data (weights and shared KV read at TB/s for years, in H3, Son, Kyung, Peng). Retention is only mentioned, in passing, as "retention-relaxed" writes (Son [16]).
- **ECC / LDPC decode latency and read-retry tails** on the HBF base die. Park offloads ECC with no model, and Atassi cites read-retry but does not model it.
- **Cell mode:** SLC is assumed everywhere (100K P/E). Nobody explores pSLC/MLC/TLC capacity-versus-endurance trade-offs, even though Park's page math cites a 4b/cell part in SLC mode.
- **Prefill-side HBF** (prefix/shared-KV caches on prefill nodes) and prefill/decode co-location. Son and Kyung restrict themselves to decode nodes. Only Park models both phases, and Baek models resume prefill.
- **Multi-tenant / multi-model serving** on a single HBF pool (model switching, LoRA adapter stores, several agents with different models). Son only hints at "two independent 4-GPU instances".
- **Tail latency.** Only Atassi reports latency CDFs (memory-access level). SLO tails for TPOT/TBT under flush freezes (Park), GC pauses, or resume bursts (Baek) are unexplored.
- **Sparse attention / KV retrieval** with random page-granular reads from HBF. Kyung mentions it qualitatively, and Baek notes sub-page reads waste bandwidth.
- **A cost model** ($/GB of HBF stacks, packaging/interposer cost, TCO per token). Everyone uses proxies (GPU count, throughput/TDP) or none.
- **Thermal/power co-design** of a 80–160 W HBF stack beside a GPU. Deferred by Park and Son, qualitative only in Atassi.
- **Write-path policies combined:** write-on-evict (Baek), selective placement by L_out (Kyung), LRU-K admission (Peng), and DRAM write-combining/BWB (Park; BTP report). These were never compared on one workload with one device model.

### 8.4 Research gaps most relevant to the BTP (HBM+HBF to avoid multi-GPU)
1. **A unified, sensitivity-first evaluation.** Every conclusion flips with HBF latency (1–28 µs), bandwidth (0.35–1.6 TB/s per stack) and power (80–160 W, 8–30 pJ/bit). A paper that sweeps these jointly, for all topologies (behind-HBM chain, shoreline side by side, full replacement, hybrid stack), would resolve the contradictions in §8.2.
2. **Shoreline-site economics.** The topology argument (H3 versus Atassi versus Son) has never been quantified. The BTP result that the chain wins by preserving HBM sites, not by flash bandwidth, fills this gap directly.
3. **Endurance beyond "WAF ≈ 1".** GC, wear concentration, static wear leveling, and allocation granularity are all unmodeled across the seven papers. BTP's 10.7× wear-leveling sensitivity and its GC-versus-die-split interaction are novel with respect to this set.
4. **Capacity right-sizing** (flash dies to HBM dies): no paper asks how much HBF is actually needed.
5. **MoE with weights beyond HBM:** the papers disagree on weight placement. A study with realistic expert predictors plus HBF latency is missing.
