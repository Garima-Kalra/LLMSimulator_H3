# E1 — Heterogeneous / tiered memory for LLM inference (non-HBF)

Papers read in full (plain-text extractions, reference lists skimmed only):
H2M2 (arXiv 2504.14893), CENT (ASPLOS'25), LIA (ISCA'25), Stratum (MICRO'25).

Context for the reader: the BTP report (BTP_Report.tex) evaluates H^3, an HBM + flash (HBF) architecture that places read-only data (weights, shared context KV) in flash. It finds the flash budget is 17–67x larger than any tested model needs, and that moving that budget to fast memory gives 1.75x concurrency at 21% lower power. It then looks at flash endurance. These notes pick out, from the four papers, the placement, cost and GPU-count ideas that bear on that work.

Convention: "[p]" means the fact appears in the paper text. "(my note)" marks my own interpretation. No numbers are invented. Where a paper's own numbers disagree with each other, I flag it.

---------------------------------------------------------------------

## 1. H2M2 — Hardware-based Heterogeneous Memory Management for Large Language Model Inference

### 1.1 Bibliographic
- **Title:** "Hardware-based Heterogeneous Memory Management for Large Language Model Inference"
- **Authors:** Soojin Hwang (KAIST), Jungwoo Kim (Stanford University), Sanghyeon Lee (KAIST), Hongbeen Kim (KAIST), Jaehyuk Huh (KAIST)
- **Venue:** arXiv preprint, arXiv:2504.14893v1 [cs.AR], 21 Apr 2025. The text names no conference, pages or DOI, so venue is **unknown**. Cite it as an arXiv preprint.
- **Pages:** 14 pages of body text plus references (the page numbers printed in the preprint). No proceedings page range.

### 1.2 Problem
- LLM inference needs large capacity (weights plus a KV cache that grows at runtime) and high bandwidth (GEMV-heavy decode). Multi-GPU HBM systems meet both, but "multiple GPUs are needed to store weights and KV cache in relative small HBMs". That raises service cost and adds TP/MP communication and synchronization overhead [p].
- Earlier alternatives fall short [p]:
  - Host-memory offloading (FlexGen, InfiniGen) is PCIe-bound.
  - LPDDR-only CXL-PNM (capacity-centric) is bandwidth-bound.
  - A strict hierarchy (HBM as a cache in front of LPDDR) fails because LLM locality is low. For GPT3-175B FP16, the weight reuse distance is at least 350 GB.
- Asked: how should **kernels and data be mapped** onto an **asymmetric** memory (HBM + LPDDR, each with its own compute), and what hardware support hides the complexity?

### 1.3 Core mechanism
**Tiers (Table 1, taken from NVIDIA Grace Hopper [32]):**

| Tier | Capacity | Bandwidth | Access latency (Table 2) |
|---|---|---|---|
| HBM3 (bandwidth-centric) | 96 GB | 3 TB/s | 32 ns |
| LPDDR5X (capacity-centric) | 512 GB | 544 GB/s | 45 ns |
| Interconnect between the two accelerators | – | 960 GB/s | – |

- **"Asymmetric", not hierarchical:** each memory has its own accelerator chip (4 cores; each core has a 128x128 weight-stationary systolic MM unit, a 32x(128x1) MV unit, a vector unit and an SFU; 1 GHz; 2x16 MB SPM per core). Compute goes to where the data lives, so data is not migrated on every access. Both chips sit on one board, connected to the host over PCIe [p].
- **Mapping granularity, "head-aware":** each sublayer is split into two parallel partitions, one per side.
  - qkv-linear and attention are split by attention head.
  - fc is split column-wise. Activations are copied to both sides, which avoids partial-sum accumulation.
  - Heads on the same side are fused into one kernel. A barrier follows each kernel.
  - Measured on GPT3-175B against an LPDDR-only baseline: naive sublayer-granularity mapping 1.27x, head-aware 1.50x (both at their best mapping) [p].
  - The paper says this generalizes to any sublayer with independent parts, for example MoE experts [p].
- **Search space:** N^3 choices (qkv x attention x fc head counts). For GPT3-175B (96 heads) that is 884,736 options, so exhaustive profiling per iteration is impractical [p].
- **Why FlexGen's LP (their Eq. 1) is suboptimal here:**
  - It is static; it does not track batch or sequence changes.
  - It groups by data type (weight / activation / KV) rather than sublayer, so it ignores that attention's GEMVs are the most bandwidth-bound.
  - Result: 1.30x vs 1.50x for "Best", i.e. 0.87x of Best [p].
- **Which sublayer gets HBM first matters most** (restricted searches, N^2 each): Q-major 1.22x, F-major 1.12x, A-major 1.40x (0.94x of Best) [p]. Attention's arithmetic intensity stays constant while its footprint grows with B x S. The intensity of qkv and fc grows with batch while their footprint is stable [p].
- **Runtime mapping algorithm (Alg. 1, "greedy" / "linear solver"):**
  - Sublayers are handled in priority order: **attention, then qkv-linear, then fc**.
  - For each, choose n heads for HBM and N−n for LPDDR such that both fit in capacity and the **peak (max) execution time of the two sides is minimized** (min-max balance).
  - Execution-time model: ideal time = ops / peak throughput, multiplied by a hyper-parameter that reflects arithmetic intensity [p].
  - Cost: 0.05 ms in single-threaded C++ on an i7-6700 [p].
  - The solver is re-run at the end of each decode iteration, triggered by a "footprint tracker".
- **Migration policy:**
  - As sequence length grows, data is evicted from HBM to LPDDR **in the order fc, qkv-linear, attention**. "Once a layer is evicted, there is no need to bring it back into HBM", so migration is small and one-directional in the growing-sequence case [p].
  - Events happen only at iteration boundaries: (1) mapping decision, (2) page allocation (free-space manager), (3) migration.
- **Who decides:**
  - A software framework on the host decides the mapping: linear solver, footprint tracker, free-space manager.
  - Hardware enforces it: a per-accelerator MMU with a 2048-entry TLB, a flat page table per side kept by the host driver, and a CUDA-event-like hardware dependency and synchronization controller.
  - The same virtual page (for example a weight) can be duplicated in both HBM and LPDDR [p].
- **Memory abstraction (hardware paging, vLLM-like but for all tensors):**
  - 2 MB huge pages; 300 ns TLB-miss latency. A flat table for 1 TB of logical space is 4 MB [p].
  - Internal fragmentation (Eq. 2): frag = (tensor_size mod page_size) x #tensors. The worst case for GPT3-175B at B=32 is 156 MB, which is 0.16% of HBM [p].
  - Remote data can be **copied** or **direct-accessed (zero-copy)**, depending on reuse [p].

### 1.4 Parameters and their sources
- Memory capacities and bandwidths: NVIDIA Grace Hopper configuration [32].
- Access latencies: from Ghose et al. [10].
- TLB of 2048 entries: from prior work [16,17].
- MM unit: modelled on Google Cloud TPU [11]. Vector unit: modelled on DFX [15], scaled up.
- Relative HBM and LPDDR energy for the static power model: from CXL-PNM [36].
- **No $/GB and no absolute power figures are given.** Precision is INT8 for all models.

### 1.5 Evaluation methodology and headline results
- **Simulator:** an in-house cycle-level simulator, cross-validated against an open-source multicore NPU simulator and a DRAM simulator [16,29].
- **Scope:** decoder layers only, **a single decode iteration** at a given (B, S). Prefill and TTFT are out of scope [p].
- **Models and batch:** GPT3-175B (B=32), Chinchilla-70B (B=64), Llama2-70B (GQA, B=128), all INT8.
- **Baseline:** an LPDDR-only capacity-centric system (CXL-PNM style [36]) with the same two accelerator chips (iso-compute).
- **Metric:** speedup = t_iter(baseline) / t_iter(H2M2). This also equals the throughput ratio and the TBT ratio [p].

Headline results [p]:
- **Speedup over LPDDR-only:**
  - GPT3-175B: 1.46x (Hierarchical 1.07x; H2M2 reaches 0.97x of Oracle).
  - Chinchilla-70B: 1.55x (Hierarchical 1.33x; 0.95x of Oracle). Hierarchical wins for S<512, where everything fits in HBM.
  - Llama2-70B: 2.94x (Hierarchical 2.75x, Oracle 3.00x). Hierarchical wins at short S because GQA shrinks the KV cache.
- **Over the strict hierarchy:** 1.36x on GPT3-175B (conclusion).
- **Overheads (Table 3):**

| Model | Abstraction | Greedy vs optimal mapping | Total |
|---|---|---|---|
| GPT3-175B | 0.80% | 2.56% | 3.36% |
| Chinchilla-70B | 1.01% | 3.76% | 4.78% |
| Llama2-70B | 1.36% | 0.60% | 1.96% |

- **Dynamic sequence lengths** (random early termination and replacement, B=32, 128 iterations, GPT3-175B): H2M2 1.48x, FlexGen 1.25x, H2M2 = 0.96x of Oracle. (The text misspells H2M2 as "H2D2" here.)
- **Sensitivity (Table 4 / Fig. 17):**
  - Most sensitive to **HBM capacity** (48 GB vs 192 GB variants).
  - Also affected by LPDDR bandwidth and LPDDR-side compute, because the LPDDR side is the critical path.
  - **HBM bandwidth barely matters** (2.25–4 TB/s variants) [p].
- **Against 8-HBM** (8 HBM devices, 768 GB total, with communication cost profiled on 8x A100):
  - 8-HBM is 2.29x over baseline, i.e. **1.57x faster than H2M2**.
  - Memory energy per token: H2M2 0.76x of baseline, 8-HBM 1.31x [p].

### 1.6 Cost / TCO / GPU-count analysis
- **There is no quantitative cost model.** The "Cost and Scalability" paragraph (Sec. 5.5) is qualitative:
  - Multi-HBM systems need more devices as capacity grows, so communication overhead grows and "cost-to-performance efficiency ... tends to degrade" (citing [54]).
  - The asymmetric design grows capacity on the cheap, scalable LPDDR side [p].
- The only quantified "multi-device" comparison is 8-HBM vs H2M2: speed (8-HBM 1.57x faster) and memory-only energy per token.
- The energy model is **static, memory-access-only**: equal read and write energy, and HBM/LPDDR relative energies taken from [36]. It includes no compute energy and no $.

### 1.7 Limitations and future work
- **No explicit limitations or future-work section.** The conclusion only restates results.
- Limitations implicit in the method (my note, but each is a stated assumption):
  - Generation phase only.
  - Decoder layers only.
  - Single-iteration measurement.
  - Fixed batch per model.
  - INT8.
  - Memory-only energy model.
  - Hierarchical wins when the working set fits in HBM (S<512 on Chinchilla; short S with GQA).
  - Needs "sufficiently large memory footprint" to pay off [p].

### 1.8 Critical view for HBM+HBF
**Transfers well:**
- **Parallel-tier, "compute-where-data-lives" framing.** Effective bandwidth is the sum of the tiers only if both tiers are read concurrently and the work is balanced. H2M2's min-max objective, `max(T_HBM, T_slow)`, is the right analytic form for an HBM+HBF stack where both feed the same GPU through separate channels. (My note: H^3 reads flash and HBM in parallel over the same interposer, so the same balance equation applies.)
- **Priority and eviction order.** Attention (KV) goes to HBM first. **fc and qkv weights are evicted to the slow tier first** and never come back. This is H^3's placement exactly: read-only weights go to flash. H2M2 gives an independent, simulation-backed justification: weights have rising arithmetic intensity with batch, so they tolerate a slower tier, while attention does not.
- **The sensitivity result (HBM capacity matters, HBM bandwidth barely does)** supports the student's finding that reallocating unused slow-tier budget to fast-tier capacity is what pays.
- **The fragmentation formula (Eq. 2)** carries over directly to flash pages or erase blocks: `(tensor_size mod page) x #tensors`, with a flash page of roughly 4–16 KB instead of 2 MB (my note; the flash sizes are not from H2M2).

**Misses for NAND:**
- Latency tolerance. LPDDR at 45 ns vs HBM at 32 ns is roughly the same class. NAND tR is in µs, so H2M2's model (ops / peak throughput x intensity factor) has **no latency or queue-depth term**. Adding one matters for HBF: small reads cannot hide a µs tR.
- **Migrations are writes.** H2M2 migrates HBM to LPDDR at no endurance cost. With HBF, evicting into flash costs program/erase cycles, so runtime rebalancing that writes to flash must be rate-limited, or avoided by only ever *dropping* HBM copies of data that already lives in flash. H2M2's "duplicate the same virtual page in both tiers" idea fits here: keep weights permanently in flash and cache the hot ones in HBM, so eviction is free.
- H2M2 puts **separate compute on each tier**. HBF has no compute near flash in H^3, so the "asymmetric" benefit (no data movement) only holds if a GPU reads flash directly over the interposer.
- No $ model, so it cannot support the GPU-count argument quantitatively.

### 1.9 BibTeX
```bibtex
@misc{hwang2025h2m2,
  title         = {Hardware-based Heterogeneous Memory Management for Large Language Model Inference},
  author        = {Hwang, Soojin and Kim, Jungwoo and Lee, Sanghyeon and Kim, Hongbeen and Huh, Jaehyuk},
  year          = {2025},
  eprint        = {2504.14893},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv preprint, v1, 21 Apr 2025}
}
```

---------------------------------------------------------------------

## 2. CENT — PIM Is All You Need: A CXL-Enabled GPU-Free System for LLM Inference

### 2.1 Bibliographic
- **Title:** "PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference"
- **Authors:**
  - Yufeng Gu* (University of Michigan)
  - Alireza Khadem* (University of Michigan)
  - Sumanth Umesh (University of Michigan)
  - Ning Liang (University of Michigan)
  - Xavier Servot (ETH Zürich)
  - Onur Mutlu (ETH Zürich)
  - Ravi Iyer† (Google; the work was done at Intel)
  - Reetuparna Das (University of Michigan)
  - (* equal contribution)
- **Venue:** Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, **Volume 2 (ASPLOS '25)**, March 30–April 3, 2025, Rotterdam, Netherlands.
- **DOI:** 10.1145/3676641.3716267. ISBN 979-8-4007-1079-7/25/03. arXiv:2502.07578v3 [cs.AR], 3 May 2025.
- **Pages:** proceedings page range **unknown** (not in text).
- **Code:** https://github.com/Yufeng98/CENT (MIT licence; also on Zenodo).

### 2.2 Problem
- Decode has low operational intensity, and the large per-request KV cache limits batch size. Together they make LLM inference memory-bandwidth-bound.
- GPUs are compute-optimized, so "users pay for expensive compute resources that are poorly utilized" [p].
- Motivation data (vLLM, 4x A100 80 GB, Llama2-70B) [p]:
  - Throughput plateaus once memory exceeds 320 GB. The saturating batch falls from 128 at 4K context to 8 at 32K. (Section 7.1 says "batch=16 at 32K"; the two passages disagree.)
  - GPU compute utilization is only 21% for Llama2-70B, against 43% for BERT and 80% for ResNet-152.
  - A decode token takes 3.4x longer than a prefill token.
- Cost anchor quoted [p]: ChatGPT inference is ~3617 HGX A100 servers, ~$694,444 per day [19].

### 2.3 Core mechanism
- **System:** a host CPU (Xeon Gold 6430L), a CXL 3.0 switch (PCIe 6.0 PHY; x16 to the host, x4 per device) and **32 CXL devices**. No GPU [p].
- **Each CXL device:**
  - CXL controller with PNM units: 32 accumulators, 32 reduction trees, 32 exponent accelerators (10th-order Taylor series), 8 BOOM-2wide RISC-V cores.
  - 64 KB Shared Buffer and 2 MB instruction buffer.
  - 16 GDDR6-PIM chips x 2 channels = 32 PIM channels.
- **Each PIM channel:**
  - 4 bank groups x 4 banks. Each bank is 32 MB with a near-bank PU (16-MAC BF16 reduction tree at 1 GHz = 32 GFLOPS) and a 2 KB global buffer.
  - All-bank ACTab/MACab/PREab commands, AiM-style [56,60] [p].
- **Division of work:** PIM does GEMV, dot products and element-wise multiply, which is more than 99% of arithmetic. PNM does softmax, sqrt, division and residual adds [p].
- **"Memory tiers":** in effect a single tier (GDDR6-PIM, 512 GB, 512 TB/s internal) plus the host. Capacity scales out over CXL, not through a heterogeneous hierarchy. PIM density penalty: AiM keeps 75% of normal capacity; UPMEM 25–50%; FIMDRAM about 25% lower on average (Table 1) [p].
- **Mapping and placement (static, set by user configuration):**
  - **PP:** each transformer block is a pipeline stage placed on the PIM channels of *one* device and never split across devices. The inter-stage transfer is a 16 KB embedding vector for Llama2-70B. There is **no batching inside a stage**, so batch = number of pipeline stages [p].
  - **TP:** FC layers are spread over all devices. The 16 KB vector is broadcast and partial results gathered; that is 135 KB of CXL traffic per block for Llama2-70B. Attention, norm and residual stay on the master device, to avoid AllReduce [p].
  - **Hybrid TP-PP:** for example TP=8, PP=4 on 32 devices, using multicast [p].
  - Within a device, a GEMV matrix is row-partitioned across 16 banks and the vector broadcast from the global buffer (Fig. 11 compile example) [p].
  - Prefill runs token by token on PIM, the same way as decode. Top-k sampling runs on the host [p].
- **Who decides:** the user picks the number of channels and pipeline stages; a Python library and an in-house compiler emit the CENT ISA (Tables 2–3). There is **no runtime migration** [p].
- **New CXL primitives:** broadcast, using a reserved H-slot code in the PBR flit and a device-ID mask; and multicast and gather, built from SEND_CXL / RECV_CXL / BCAST_CXL [p].

### 2.4 Parameters and their sources (Table 4, Methodology)

| Item | CENT (32 CXL devices) | GPU (4x A100 80 GB, NVLink 3.0) |
|---|---|---|
| Process | 1Y nm (14–16 nm) | 7 nm |
| Memory | 512 GB GDDR6 | 320 GB HBM2E |
| Compute | 512 TFLOPS PIM + 96 TFLOPS PNM | 1248 TFLOPS |
| Peak BW | 512 TB/s internal | 8 TB/s external |
| 3-yr owned TCO | $0.73/h | $1.76/h |
| 3-yr rental TCO | $1.05/h | $5.45/h |

- **GDDR6-PIM timing:** tRCD_RD=18 ns, tRAS=27 ns, tCL=25 ns, tRCD_WR=14 ns, tCCDS=1 ns, tRP=16 ns.
- **Power sources:**
  - DRAM core power: Micron DRAM Power Calculator with Samsung 8Gb GDDR6 C-die specs [97,69].
  - MAC: 3x the current of a gapless read [56].
  - Memory controller: 314.6 mW per 2 channels [108].
  - BOOM core: 250 mW [9].
  - Custom logic: RTL synthesized at TSMC 28 nm and scaled to 7 nm (Table 5 total 7.85 mm² and 1.06 W at 28 nm).
  - Controller: 19.0 mm² at 7 nm, 2.0 GHz.
- **Energy per bit:** CENT MAC_ABK uses 0.6 pJ/bit, versus 3.97 pJ/bit for an HBM2 read [78].
- **Measured power:** Llama2-70B uses 27 of the 32 devices (3 blocks per device), averaging **32.4 W per device**. One A100 draws about 8x one CENT device; the GPU runs near its 300 W TDP [p].

### 2.5 Evaluation methodology and headline results
- **Models and scaling:**
  - Llama2-7B, 13B, 70B on 1 / 2 / 4 A100 against 8 / 20 / 32 CXL devices.
  - Queries are 512 prefill + 3584 decode tokens (4K context).
  - GPU runs vLLM at batch 128.
  - CENT uses batch = PP stages, 32 / 40 / 80.
- **Simulator:**
  - Traces are generated per block and checked with a functional simulator.
  - Ramulator2 is modified to model 32 GDDR6-PIM channels.
  - CXL is an analytical model using CXL latency [61] and PCIe 6.0 bandwidth. A multicast-capable switch is modelled at half the bandwidth and double the latency of the baseline switch.
  - The GPU side is **measured on real hardware**.
- **Headline results [p]:**
  - Latency-critical (B=1, TP): **4.6x** lower end-to-end latency.
  - Throughput-critical: **2.3x** geomean throughput (1.2x on Llama2-70B, because of GQA). Prefill: GPU 2.5x better. Decode: CENT 2.5x better. Prefill is only 2% of GPU time.
  - **5.2x tokens per dollar** (Fig. 13c).
  - Energy: **2.9x tokens/J** (abstract and Sec. 7.2), but the conclusion says "2.3× less energy", which is **internally inconsistent**. Prefill: GPU 2.4x more efficient. Decode: CENT 3.2x.
  - Long context (Llama2-70B): up to **3.3x** decode throughput at 32K. The 16K and 32K runs at PP=80 need 16Gb GDDR6-PIM, i.e. 1 TB total.
  - At similar throughput, CENT has **3.4–7.6x** lower query latency (QoS). Prefill latency is 1.4x higher, decode 1.7–2.0x lower.
  - Against CXL-PNM (LPDDR5X, OPT-66B): 4.5x throughput.
  - Against GPU-PIM (GPT3-175B, power-neutral: 12 CENT devices per GPU-PIM node):

| Baseline | CENT tokens/$ | CENT raw throughput | Baseline TCO vs CENT |
|---|---|---|---|
| AttAcc | 1.8–3.7x | 0.5–1.1x | 3.5x higher |
| NeuPIM | 1.8–5.3x | 0.7–2.1x | 2.6x higher |

  - Scaling: 16 to 128 devices, **0.68K to 5.7K tokens/s** (Llama2-70B, PP then DP). Plateaus appear when 80 blocks do not divide evenly: 44 devices are run like 40, with 4 left idle. Limits are switch lanes/ports (a commercial PCIe 5.0 switch has 144 lanes, 72 ports) and server power (DGX A100 peak 6.5 kW), so **at most 64 devices per server** [p].

### 2.6 Cost / TCO / tokens-per-$ model (the most detailed of the four)
- **Metric:** tokens/$ = throughput / (3-year TCO per hour). (My note: this form is implied by "5.2× more tokens per dollar ... attributes to CENT's higher throughput and 2.5× cheaper TCO". The paper does not say whether owned or rental TCO feeds Fig. 13c. The Table 4 owned ratio is 1.76/0.73 ≈ 2.4x; the rental ratio is 5.45/1.05 ≈ 5.2x.)
- **TCO components [p]:**
  - **Owned TCO** = hardware cost + operational (electricity) cost over 3 years. Electricity is **$0.139/kWh** [79] x average power.
  - **Rental TCO:** host CPU and GPUs are priced from Microsoft Azure [1]. The CXL devices use the owned-TCO method, because no rental prices exist.
- **Hardware costs (Table 6):**

| System | Item | Cost |
|---|---|---|
| GPU | Xeon Gold 6430 | $2,128 |
| GPU | 4x A100 80 GB | $40,000 |
| GPU | **Total** | **$42,128** |
| CENT | Xeon Gold 6430 | $2,128 |
| CENT | 512 GB GDDR6-PIM | $11,873 |
| CENT | 32 CXL controllers | $381.3 |
| CENT | 96-lane 48-port switch | $490 |
| CENT | **Total** | **$14,873** |

  - The A100 is priced at **$10,000**, "conservatively deducting 50% margin" from the lowest price of about $20,000 [20].
  - **PIM module = 10x standard DRAM cost** [17,107]. HBM-PIM (for AttAcc) = 10x HBM price [98].
- **CXL controller unit cost (Fig. 12):** die + packaging + NRE.
  - Die cost comes from a 300 mm 7 nm wafer at **$9,346** with defect density **0.0015/mm²** [71] and the 19.0 mm² die.
  - 2D packaging = 29% of chip cost [59]. 2.5D packaging is computed from interposer, die placement and substrate assembly [85].
  - NRE (mask, backend labour, CAD, frontend, IP, package design, system; about $20M-scale in Fig. 12) is amortized over **3M units**. That volume is derived as: NVIDIA shipped 3.76M datacenter GPUs in 2023; 10% (about 370K) assumed for inference; each GPU draws about 8x a CENT device's power, so about 3M devices.
  - Result: about **$11.9 per controller**.
- **NPU cost** (for the NeuPIM baseline): modelled from die, 2.5D packaging and NRE [45,49,85].
- **"Number of devices" logic:**
  - Configurations are chosen **at similar average power**: 32 CXL devices vs 4 A100.
  - Llama2-70B needs 4x A100 (320 GB). CENT needs 27 devices at 3 blocks per device.
  - The paper argues that bigger models (Grok 314B, Llama3 405B, DeepSeek-V3 671B) widen CENT's cost advantage. This is asserted, not evaluated.

### 2.7 Limitations and future work
Stated or evident in the text:
- Prefill is 2.5x slower than GPU. The authors suggest **disaggregating prefill onto GPUs and decode onto CENT** for long-input tasks [p].
- No batching within a pipeline stage (buffer sizes) [p].
- Scale is limited to about 64 devices per server by switch lanes and power. Beyond that would need multi-socket hosts or two-level CXL switching [p].
- Blocks cannot be split across devices without heavy communication, which gives the plateaus and idle devices [p].
- GQA reduces CENT's advantage (1.2x on Llama2-70B) [p].
- PIM density loss [p].
- The CXL model is analytical, and the multicast switch is an assumption [p].

Future work (explicit):
- Longer contexts (128K–1M) and larger models, where "we expect CENT to provide even higher performance" [p].
- RISC-V cores for supporting new LLM operators [p].

### 2.8 Critical view for HBM+HBF
**Transfers well:**
- **The TCO methodology is directly reusable** for the student's "fewer GPUs" argument:
  - owned vs rental TCO;
  - 3-year amortization;
  - $/kWh x average power;
  - the GPU price with 50% margin deducted;
  - a memory-module cost multiplier (PIM = 10x DRAM). For HBF, an analogous multiplier on NAND $/GB would be needed; it is not in this paper.
  - die cost from wafer cost, defect density and area; packaging as a percentage of die cost; NRE over projected volume.
  - tokens/$ = throughput / TCO-per-hour.
  - **iso-power** comparison (32 devices ≈ 4 A100).
- **The Fig. 1 methodology** (throughput vs batch vs memory requirement on N GPUs, showing a plateau once memory is exhausted) is exactly the way to show that capacity, not compute, sets GPU count. It could be reproduced for H^3 against multi-GPU.
- The observation that **decode is about 98% of time** and bandwidth-bound supports putting read-only weights on a bandwidth-rich tier.

**Misses:**
- CENT removes the capacity problem by **scaling out DRAM**, not by adding a cheaper-per-bit tier. It has no hot/cold placement, no migration and no second tier, so it offers nothing on flash latency, granularity or endurance.
- Its whole cost advantage leans on assumptions the student should not copy blindly: PIM = 10x DRAM, A100 at $10K, a 3M-unit volume.
- Its sensitivity to these assumptions is not reported (my note).

### 2.9 BibTeX
```bibtex
@inproceedings{gu2025cent,
  title     = {{PIM} Is All You Need: A {CXL}-Enabled {GPU}-Free System for Large Language Model Inference},
  author    = {Gu, Yufeng and Khadem, Alireza and Umesh, Sumanth and Liang, Ning and Servot, Xavier and Mutlu, Onur and Iyer, Ravi and Das, Reetuparna},
  booktitle = {Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25)},
  year      = {2025},
  address   = {Rotterdam, Netherlands},
  publisher = {ACM},
  doi       = {10.1145/3676641.3716267},
  isbn      = {979-8-4007-1079-7},
  note      = {arXiv:2502.07578}
}
```

---------------------------------------------------------------------

## 3. LIA — A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading

### 3.1 Bibliographic
- **Title:** "LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading"
- **Authors:** Hyungyo Kim, Nachuan Wang, Qirong Xia, Jinghan Huang (all UIUC), Amir Yazdanbakhsh (Google DeepMind), Nam Sung Kim (UIUC)
- **Venue:** Proceedings of the 52nd Annual International Symposium on Computer Architecture (**ISCA '25**), June 21–25, 2025, Tokyo, Japan. ACM, New York.
- **Pages / DOI:** 15 pages; the first page is numbered 544, so **pp. 544–558**. DOI 10.1145/3695053.3731092. ISBN 979-8-4007-1261-6/25/06.

### 3.2 Problem
- A single GPU cannot hold large LLMs. H100 has up to 94 GB of HBM.
- Multi-GPU is "financially prohibitive": **OPT-175B needs at least 5 H100s, about $150,000**. Even at 4-bit, OPT-175B needs at least 2 H100s for weights alone [p].
- Offloading to host memory is PCIe-bound. H100 PCIe 5.0 is 64 GB/s, and moving OPT-175B's weights once adds about 5 s [p].
- Earlier CPU-GPU co-execution offloads only attention scoring (FlexGen, FastDecode) or needs ReLU sparsity (PowerInfer), because AVX CPUs are about 100x slower than GPUs. A fixed policy also ignores the variation with B and L [p].
- FlexGen on SPR-A100, OPT-175B [p]:
  - Parameter transfer is **more than 98%** of prefill and decode latency at B=1 and short L.
  - At B=32, KV plus activations reach 145 GB, and transfers stay above 80% of decode time.
  - Offloading attention to the CPU saves at most **10.2%** (L=1024) and *hurts* at L=64 and 128.

### 3.3 Core mechanism
**Tiers:**

| Tier | Capacity | Bandwidth / latency |
|---|---|---|
| GPU HBM | A100 40 GB HBM2 (PCIe 4.0); H100 80 GB HBM3 (PCIe 5.0) | – |
| CPU DDR5 | 512 GB, 8x DDR5-4800 | about 260 GB/s on SPR |
| CXL (Samsung Type-3 expanders) | 2 x 128 GB, DDR4-based ("repurposed from retired data center servers" is the proposal) | +140–170 ns vs DDR and up to 50% of DDR bandwidth [48]; Sec. 6 also says "latency is 2–3× higher" [p]. Each expander about 17 GB/s (Fig. 8a) |

- GNR has 12x DDR5-5600 channels, about double SPR's memory bandwidth.

**AMX characterization (Sec. 4):**
- SPR-AMX (40 cores) has a theoretical peak of 90.1 TFLOPS BF16. Measured GEMM is 4.5x AVX512, 4–11% of H100 and 7–15% of A100.
- GNR-AMX (128 cores) reaches up to 2.4x SPR. A dual-socket GNR gets another 1.8x (16% of H100).
- GEMV is memory-bound: SPR-AMX peaks at 199 GFLOPS, about the same as AVX, and 15% of H100 (in line with the bandwidth ratio). GNR is 1.7x SPR on GEMV [p].
- Abstract figures: SPR about 20 TFLOPS, GNR about 40 TFLOPS matmul [p].

**Placement / offloading algorithm (Sec. 5.1):**
- A **per-sublayer binary vector p = (p1..p6)** over QKV, Q·Kᵀ, S·V, OutProj, FC1, FC2, with **p_i = 1 meaning CPU**.
- Chosen by **exhaustive minimization of an analytical latency model**, T(p) = Σ_i [T_load + T_comp + T_store]:
  - **T_load:** activations moved over PCIe only if p_i ⊕ p_{i−1} = 1 (a device switch); weights or KV moved over PCIe if p_i = 1 (with a special case for prefill K and V, which are generated by sublayer 1); residual moved for sublayers 4 and 6 if their device differs from sublayer 1 or 4.
  - **T_comp:** (D_X + D_Y) / BW_mem + C / TH_compute, for the chosen device (roofline-additive).
  - **T_store:** D_KV / BW_PCIe if sublayer 1 runs on the GPU.
  - Data sizes and FLOPs per sublayer come from closed-form Table 1 formulas in B, L and d_m. For example, decode Q·Kᵀ has D_Y = 2BLd_m and C = 2BLd_m, so ops/byte is about 1. FC1 has D_Y = 8d_m² and C = 8Bd_m².
  - (Paper typo: Eq. 8 assigns the GPU term to "p_i = 1" and the AMX term to "p_i = 0", which contradicts the definition that p_i = 1 means CPU.)
- **Granularity is per stage** (prefill vs decode), as a function of (B, L). Inputs are the system's PCIe bandwidth, the CPU and GPU memory bandwidth, and their throughput.
- For decode the policy depends only on B, so it **does not change during generation** [p].
- **Resulting policy map for OPT-175B (Fig. 9):**
  - Prefill: all-CPU when B·L is below about 850, all-GPU above.
  - Decode: all-CPU when B is below 858. Above that, the GPU runs QKV, OutProj, FC1 and FC2, and the CPU runs attention scoring: p = (0,1,1,0,0,0).
  - Only three policies appear across OPT models: partial CPU (0,1,1,0,0,0), full CPU (1,…,1), full GPU (0,…,0).
  - For MoE the authors predict (0,1,1,0,1,1) [p].

**Optimizations:**
- **Opt-1:** use leftover GPU memory to pin *whole decoder layers*, not sublayers across all layers as FlexGen does. OPT-30B at B=1, L=2016: 62% of layers in 35 GB, against FlexGen's 58% of sublayers in 32 GB [p].
- **Opt-2:** overlap transfers with compute. Prefill uses mini-batches as in FlexGen; decode uses the **whole batch, with no mini-batching**. This gives 1.1–1.3x at B=900 [p].
- **Implementation:** Intel IPEX rebuilt against pytorch-cuda, plus changes to HF Transformers [p].

**CXL memory-offloading policy (Sec. 6):**
- **Observation 1:** CXL does not hurt CPU-to-GPU weight transfer while CXL bandwidth ≥ PCIe bandwidth. Two expanders interleaved at page granularity approach DDR-to-GPU bandwidth for transfers ≥ 300 MB [p].
- **Observation 2:** putting data in CXL hurts **CPU-side** compute.
  - Parameter sublayer 1: −11 to −70%.
  - KV sublayer 2 (ops/byte = 1): −10 to −82% [p].
- **Policy:** for throughput runs with large B, place **all parameters in CXL** (their sublayers go to the GPU, so they only cross PCIe), and **keep the KV cache in DDR** (the CPU computes attention from it). Parameters are **static** [p].

### 3.4 Parameters and their sources
- **Hardware (Table 2):** Supermicro X13DDW-A; Xeon Platinum 8460H (40C); 8x DDR5-4800, 512 GB; A100 40 GB (PCIe 4.0); H100 80 GB (PCIe 5.0); Samsung CXL Type-3 2x128 GB.
- **CXL latency and bandwidth:** from [48].
- **Pin count:** CXL needs 3x fewer pins than DDR5 [48].
- **DIMM cost:** a 256 GB DIMM costs at least 2x more per GB than a 32 GB DIMM (memory.net, Aug 2024 [4]).
- **Memory cost** (MemVerge [10]): DDR-only **$11.25/GB**; half DDR, half CXL **$5.60/GB**.
- **System costs** (footnote 7): GNR-A100 **$22,000**; DGX-A100 (8x A100 80 GB, NVLink, TP=8) **$200,000**; electricity **$0.1/kWh** ("Louisiana, the cheapest in the U.S."); power estimated at each system's **TDP**.
- **Grace CPU** (footnote 8): SVE2 at 6.91 TFLOPS (30x below GNR); 512 GB/s memory; Grace–Hopper link 900 GB/s (7x PCIe 5.0 x16).

### 3.5 Evaluation methodology and headline results
- **Setup:**
  - Real hardware for SPR-A100 and SPR-H100. GNR results appear too; the text does not say clearly whether GNR was measured or modelled, beyond the Grace-Hopper and V100 analyses being explicitly analytical.
  - An **analytical latency model** (Eq. 2 per layer x number of layers) is used for configurations that exceed 512 GB. Its **average error is 12%**. Starred bars in the figures come from the model.
  - Models: OPT-30B and OPT-175B (A100); OPT-66B and OPT-175B (H100); all BF16.
  - Scenarios: online B=1 (s/query); offline B=64 and B=900 (tokens/s).
  - Sequence lengths from Azure traces: L_in 32 to 2048 (L_max 2016 or 1792); L_out 32 (code) or 256 (conversation).
  - Energy from ipmitool average power x latency / L_out.
  - Baselines: IPEX (CPU-only AMX), FlexGen (GPU + AVX). Also PowerInfer (Llama2-70B), a DGX-A100 simulated in Vidur, and 3x V100 (analytical).
- **Latency results [p]:**
  - SPR-A100: vs IPEX 1.8–2.1x (OPT-30B) and 1.1–1.3x (OPT-175B); vs FlexGen 5.3–7.3x and 8.5–12x. Transfer reduction vs FlexGen ranges from 31x to 222,524x.
  - SPR-H100: vs IPEX 2.1–2.5x (66B) and 1.1–1.5x (175B); vs FlexGen 4.9–7.0x and 4.0–5.1x.
- **Throughput results:**
  - SPR-A100: 1.5–6.0x / 1.1–6.1x vs IPEX; 2.0–5.9x / 1.3–6.0x vs FlexGen.
  - SPR-H100: 1.3–8.3x / 1.2–10x vs IPEX; 1.2–3.3x / 1.5–3.7x vs FlexGen.
- **CXL (Table 3, OPT-30B, B=900):**
  - Up to **43.1%** of DDR usage offloaded (L_in=32, L_out=32) with less than 1% throughput change (280.38 vs 280.50 tok/s).
  - At the same DDR footprint, B rises to 1580 / 1350 / 1150 / 1050 and throughput reaches up to 406.88 tok/s, i.e. **1.45x**.
  - The abstract claims "1.5×" throughput and "1.8×" max batch (900 to 1.6K); the intro says "up to 1.76×". These are the same result, rounded differently.
- **Ablation (OPT-30B, SPR-A100):** LIA's policy is 6.2x (B=1) and 3.5x (B=64) better than FlexGen's policy. At B=900 the policy is the same but AMX gives 1.9x.
- **Energy:** 1.1–5.8x vs IPEX, 1.6–10.3x vs FlexGen.
- **GNR (Table 6):** vs FlexGen up to 13–24x online (OPT-175B, GNR-A100).
- **Other comparisons:**
  - PowerInfer: 1.4–9.0x lower latency, 1.5–15x throughput. PowerInfer runs out of memory at B=900.
  - Other models (analytical, vs FlexGen): Llama2-70B, Chinchilla-70B and Bloom-176B at 6.1–11x latency.
  - Grace-Hopper (analytical): the optimal policy is all-GPU. It gives 1.8–2.3x lower latency and 3.0–4.1x throughput vs GNR-H100. The authors conclude "**improving CPU-GPU bandwidth may be a more effective direction than increasing CPU compute power**" [p].

### 3.6 Cost / $-per-token / GPU-count analysis
- **$/million tokens (Sec. 7.8, Fig. 14; OPT-175B):**
  - cost = (system price amortized over 3 years + TDP x $0.1/kWh) / tokens produced.
  - LIA on GNR-A100 ($22K) vs DGX-A100 8-GPU TP ($200K, latency from Vidur).
  - Per-GPU throughput is used for the comparison.
  - **B=1:** LIA has 1.4–1.8x higher per-GPU throughput and **1.5–2.0x lower cost**.
  - **B=64:** LIA has 30–33% lower per-GPU throughput and **1.3–1.4x higher cost**. If AMX reached 50% of theoretical peak (analytical), that becomes 1.1–1.3x higher per-GPU throughput and 1.1–1.2x lower cost.
  - B=900 is omitted: the DGX runs out of memory.
  - "LIA requires only 10% of the system cost" (22K / 200K = 11%) [p].
- **Iso-cost alternative:** 3x V100 with a low-end CPU costs about the same as GNR-A100. LIA on GNR-A100 is 6.3–11x better on latency and 2.2–16x on throughput (analytical, ignoring inter-V100 communication). The text literally says "6.3–11× longer latency and 2.2–16× lower throughput" but means the reverse, which is a wording error [p].
- **Memory cost with CXL:** OPT-175B memory system cost falls from **$6,300 to $3,200**, which is an **8% (SPR-A100) or 9% (GNR-A100)** cut in total system cost [p].
- **GPU-vs-CPU scaling:** GNR-A100 has **1.7x lower system cost** and 1.6x higher tokens/s/W(TDP) than SPR-H100. It gives 1.4–2.0x lower online latency, but 70% of SPR-H100's throughput at B=900 [p].
- **GPU count anchor:** OPT-175B needs at least 5 H100s, $150,000 [p].

### 3.7 Limitations and future work
- **No explicit limitations section.** From the Discussion and the text:
  - Single GPU only. Multi-GPU extension via TP is discussed: communication "may reduce the scaling impact", especially over PCIe [p].
  - AMX libraries are immature, so measured utilization is low; 50% of peak is a projection [p].
  - The analytical model has 12% error and supplies many data points [p].
  - The policy is about memory-bandwidth-bound CPU work. On Grace-Hopper, CPU offload is useless [p].
  - The CXL policy helps only large-batch offline inference (B above about 860) [p].
  - The B=64 cost is *worse* than DGX [p].
  - Evaluated models are OPT. Others are analytical only [p].
- **Future directions (explicit):** MoE policy diversity (predicted but not evaluated); multi-GPU; the bandwidth-over-compute conclusion [p].

### 3.8 Critical view for HBM+HBF
**Transfers well:**
- **The Eq. 2–9 per-sublayer latency model** is a ready template for an HBM+HBF analytic model:
  - T = load + compute + store per sublayer;
  - roofline-additive compute (bytes/BW + FLOPs/throughput);
  - transfer charged only when a sublayer's operand lives on a different tier or device.
  - For HBF, replace BW_PCIe with HBF read bandwidth. Add a **per-access latency term** (tR in µs) and a **page-granularity rounding** term (bytes rounded up to the flash page) — (my note).
- **Observation 2 is the key transferable insight:**
  - Weights (ops/byte grows with B) tolerate a slow tier.
  - KV attention (ops/byte ≈ 1, constant) is badly hurt by it.
  - LIA's policy "**params in cheap/slow tier, KV in fast tier**" is the same split as H^3's "read-only weights in flash, KV in HBM", derived from measurement on real CXL hardware. That makes it good supporting evidence for the student.
- **The $/million-tokens formula and "per-GPU throughput" metric** are simple and reproducible (amortized capex over 3 years + TDP x $/kWh). They suit an HBM+HBF vs N-GPU comparison. The B=64 reversal is a warning: capacity-tier systems win on cost at small batch and lose at moderate batch unless their bandwidth is adequate. The student should report both regimes.
- The Grace-Hopper conclusion (the link bandwidth to the capacity tier dominates) supports HBF's premise: put the capacity tier on the interposer, at high bandwidth, rather than behind PCIe.

**Misses:**
- CXL-DRAM latency is roughly 100s of ns higher than DDR, not µs. Endurance is a non-issue (DRAM), and access granularity is a cache line or 4 KB NUMA page.
- The policy is static and has no hot/cold tiering inside the parameter set.
- $/GB figures are for DRAM only (DDR vs DDR+CXL). Nothing on NAND.

### 3.9 BibTeX
```bibtex
@inproceedings{kim2025lia,
  title     = {{LIA}: A Single-{GPU} {LLM} Inference Acceleration with Cooperative {AMX}-Enabled {CPU}-{GPU} Computation and {CXL} Offloading},
  author    = {Kim, Hyungyo and Wang, Nachuan and Xia, Qirong and Huang, Jinghan and Yazdanbakhsh, Amir and Kim, Nam Sung},
  booktitle = {Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA '25)},
  year      = {2025},
  pages     = {544--558},
  address   = {Tokyo, Japan},
  publisher = {ACM},
  doi       = {10.1145/3695053.3731092},
  isbn      = {979-8-4007-1261-6}
}
```

---------------------------------------------------------------------

## 4. Stratum — System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

### 4.1 Bibliographic
- **Title:** "Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving"
- **Authors:**
  - Yue Pan* (UC San Diego)
  - Zihan Xia* (UC San Diego)
  - Po-Kai Hsu (Georgia Tech)
  - Lanxiang Hu (UCSD)
  - Hyungyo Kim (UIUC)
  - Janak Sharda (Georgia Tech)
  - Minxuan Zhou (Illinois Institute of Technology)
  - Nam Sung Kim (UIUC)
  - Shimeng Yu (Georgia Tech)
  - Tajana Rosing (UCSD)
  - Mingu Kang (UCSD)
  - (* equal contribution)
- **Venue:** 58th IEEE/ACM International Symposium on Microarchitecture (**MICRO '25**), October 18–22, 2025, Seoul, Republic of Korea. ACM, New York.
- **Pages / DOI:** 17 pages; proceedings page range **unknown**. DOI 10.1145/3725843.3756043. ISBN 979-8-4007-1573-0/2025/10. arXiv:2510.05245v1 [cs.AR], 6 Oct 2025.

### 4.2 Problem
- MoE models are huge in total parameters (experts are more than 95% of Mixtral 8x7B), though sparse per token. HBM bandwidth through the interposer limits decode.
- HBM-based NMP/PIM (AttAcc, NeuPIM, Duplex) is limited by TSV count and pitch (about 10 µm) or by DRAM-process logic [p].
- Mono3D DRAM (monolithic stacked 1T1C, hybrid-bonded to a logic die at about 1 µm pitch, **about 5x denser vertical interconnect than HBM**) gives far more internal bandwidth.
- But **access latency varies with layer**: wordline staircase RC means tRCD runs from 1.11 ns at layer 1 to 22.88 ns at layer 1024 (Fig. 2). Designing for the worst case wastes bandwidth [p].

### 4.3 Core mechanism
- **System:**
  - An xPU die plus N Mono3D DRAM chips on a silicon interposer. Each chip is a DRAM die hybrid-bonded to a logic die carrying an NMP processor.
  - Configurations:
    - **Stratum-S:** RTX A6000 die + 1 chip (32 GB).
    - **Stratum-L:** H100 die + 6 chips.
    - **Stratum-XL:** 2x Stratum-L, 384 GB, cross-chip interconnect such as NVLink.
- **Tiers inside a single device ("in-memory tiering"):**
  - 1024 layers split into **8 tiers of 4 GB each** (a 32 GB chip).
  - tRCD per tier: [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88] ns. tRP = 4.77 ns; tRAS = tRCD + 27.50 ns.
  - The fast tier is **1.6x faster** than the slowest.
  - Internal bandwidth ranges from **19.01 to 30.34 TB/s** depending on tier. Table 3 prints "19.01-34.34 TB/s", which is **inconsistent** with the text [p].
  - External interface: 1024-bit, 6.4 Gbps/pin, the same as HBM3.
  - The tier is resolved in hardware by a **programmable tiering table** (row address to tier ID to tRCD) in each bank-level local memory controller. A **row-swap buffer** (8 KB RF) moves rows between tiers inside a bank without leaving the chip [p].
- **NMP:**
  - 16 PUs (one per channel), bidirectional ring at 128 GB/s per link (2.048 TB/s aggregate).
  - Each PU has 16 PEs (one per bank; each a 16x16 MAC tensor core with 64 KB psum SRAM), 1.25 MB shared memory and a 256-way SIMD special-function engine.
  - 128 TFLOPS FP16, 1 GHz, 7 nm (ASAP7), 76.63 mm² of the 82 mm² budget, 42.67 W against a 45 W cap [p].
- **Operator mapping:**
  - Experts run **one at a time with tensor parallelism across all PUs**, not expert parallelism, to avoid load imbalance made worse by tiers. W1 and W2 are split column-wise, W3 row-wise. X_t is duplicated; Z3 is reduced with reduce-scatter.
  - Attention uses head parallelism across PU groups, with the sequence dimension split inside a group. Softmax needs scalar exchange. New KV is placed round-robin.
  - Prefill runs on the xPU, decode on the NMP (as in AttAcc) [p].
- **Placement policy (Sec. 5.2, Alg. 1) — four data classes [p]:**
  - (1) Hot experts, including shared experts, go to the **fastest tiers**.
  - (2) Cold experts go to slower tiers.
  - (3) KV cache, which grows dynamically, goes to **intermediate-speed** tiers.
  - (4) **Non-NMP data** (positional embeddings, layer-norm parameters, etc., consumed by the xPU) goes to the **slowest** tier. The interposer is an order of magnitude slower than even the slowest tier's internal bandwidth, so tier latency does not matter for that data.
- **Alg. 1:**
  - Sort all K·L experts by predicted usage frequency.
  - The top τ = k·L (active experts per layer x layers) get row intervals from address 0 upward, i.e. fast.
  - The rest fill down from Φ, the reserved NMP rows. Each expert takes Δ = ⌈S_E / (N_bank·S_rb)⌉ rows.
  - Rows map to tiers uniformly (equal rows per tier) [p].
- **Prediction:**
  - Offline profiling of per-topic expert hit rates. For example, LLaMA-4 Scout shows "over 90% domain-specific expert affinity" on math and logic MMLU subsets.
  - Online, a **DistilBERT topic classifier** (67M parameters, 6 topics) tags each query on the host.
  - An **SLO-aware (TTFT) scheduler** batches same-topic queries where the SLO allows.
  - The memory mapper aggregates the batch's topic mix into a target placement. **Experts are swapped before each batch whose topic tags differ**, using near-memory row swaps [p].
- **Who decides:** the host software (classifier, scheduler, memory mapper) decides placement per batch. Hardware (tiering table, row-swap buffer) executes it. **Dynamic, per batch or topic switch.**

### 4.4 Parameters and their sources
- **Device (Table 1):** 1024 layers; 35 nm feature size; BL/WL pitch 70 nm / 1 µm; staircase pitch 500 nm; MAT 1k x 1k; 32x32 MATs per bank; 1 Gb and 0.439 mm² per bank; 32 Kb row buffer; **0.429 pJ/bit**; chip 121 mm², 32 GB; density **2.156 Gb/mm²**, 5.2x the latest 32Gb DDR5 die at 0.417 Gb/mm² [14].
- **Organization:** 16 channels x 64-bit; 16 banks per channel.
- **Device tools:** Coventor SEMulator3D for RC extraction; NeuroSim for peripherals; DDR5 timing standard; CUA at 32 nm, bonded CMOS at 7 nm.
- **Logic:** SystemVerilog, Cadence Genus, ASAP7; FinCACTI for SRAM; E_mac = 0.604 pJ.
- **Thermal:** HotSpot, assuming vapor-chamber liquid cooling (up to 200 W/cm², [53]). One DRAM die draws about 104 W at full 30.34 TB/s, so the logic die is capped at about **45 W**. Peak stack power is 144.53 W.
- **Area:** HBM3 PHY 23.94 mm²; low-voltage peripherals 14.80 mm²; power TSVs 0.21 mm² (25 µm², 36 mA each, 2:1 redundancy) [p].
- **No $/GB is given.** Cost is argued only qualitatively: no TSVs, sequential layers on one wafer, "higher density without a proportional increase in cost per bit" [p].

### 4.5 Evaluation methodology and headline results
- **Workloads (Table 2):**

| Model | Experts | GPU baseline | Stratum config |
|---|---|---|---|
| OLMoE-1B-7B | 64 choose 8 | RTX A6000 | Stratum-S |
| Mixtral 8x7B (47B) | 8 choose 2 | 2x H100 | Stratum-L |
| Qwen2.5-32B | dense | 2x H100 | Stratum-L |
| Llama-4-Scout (109B) | 1 shared + 16 choose 1 | 4x H100 | Stratum-XL |

- Each configuration is sized to hold the maximum evaluated context. GPUs run vLLM 0.8.1 in throughput mode, with energy from nvidia-smi.
- **Stratum side:** in-house simulator (cycles, communication, energy from post-synthesis netlists). A system simulator adds a Poisson request generator, the SLO-aware scheduler and the mappers. Lin = Lout, from 512 up to 8192.
- **Headline results [p]:**
  - Decode throughput vs GPU (tiering): **8.29x** (OLMoE), 5.39x (Mixtral), 6.13x (Qwen2.5), 4.48x (Llama-4) on average.
  - Tiering over no-tiering: 1.45x, 1.39x, 1.32x, 1.34x.
  - Energy efficiency: up to **7.66x**, 2.74x, 3.51x, 4.87x. (The conclusion rephrases this as "7.66× less energy consumption".)
  - Against Duplex (numbers extracted and "conservatively" scaled): up to 2.9x, 2.5x, 3.0x, 2.2x throughput and 2.7x, 1.9x, 2.9x, 2.1x energy.
  - **Accurate hot-expert prediction** gives **1.32–1.51x** throughput over uniform expert usage (Mixtral, Stratum-L). The gain is largest at short decode lengths.
  - Achieved hot-expert hit rates: **31.6% (Mixtral), 48.5% (OLMoE), 68.9% (Llama-4)**.
  - Expert-swap overhead is worst-case (B=1, L=256, the topic changes every batch). Swaps per second are 5.91 / 2.59 / 4.02. Time overhead is 0.64 ms (0.37%) / 0.90 ms (0.23%) / 0.45 ms (0.18%). Energy overhead is below 0.02–0.03%. The overhead is small because swaps stay inside a bank and use the row-swap buffer.
  - Batch scaling (Llama-4, XL): 4.7–9.8x over GPU. The advantage shrinks with larger batch.
  - Layer scaling: 1024 layers give **1.21x / 2.96x throughput per area** vs 256 / 64 layers.
  - With 512 layers, the latency spread between tiers is 1.3x, and tiering still gives 17.7–18.3% at Lin = Lout = 1024.
- **Classifier:** 94.5% on MMLU, 85.0% on Chatbot Arena (O3-mini-high: 96.2% / 91.1%). Under 10 ms on a laptop CPU with ONNX; under 2% decode-step overhead at fewer than 4 QPS. Six topics cover 93% of 33,000 LMArena queries (O3 as judge). Sec. 5.1 instead quotes "85.0% and 81.0%", which is **inconsistent** with Sec. 6.3.1 [p].

### 4.6 Cost / TCO / GPU-count analysis
- **No $ or TCO model.** The "cost-aware" metric is **throughput per Mono3D die area** (Fig. 18b).
- GPU count enters only through the baselines (1x A6000, 2x H100, 4x H100), each sized to fit the model and context. Stratum-XL (2 H100-class dies + 12 Mono3D chips) is compared against 4x H100. The implicit claim is fewer compute dies for the same capacity, but it is not quantified in $ (my note).
- "Cost-effective alternative" is asserted from the manufacturing argument (no TSVs, monolithic) [p].

### 4.7 Limitations and future work
- **No explicit limitations or future-work section.** From the text:
  - Mono3D DRAM at 1024 layers is **projected**. The industry reference is 3D NAND beyond 400 layers, with white papers suggesting 500–1000.
  - It needs high-end liquid or vapor-chamber cooling.
  - Hit rates are modest: 31.6% for Mixtral.
  - The advantage shrinks at large batch.
  - The Duplex comparison uses scaled extracted numbers.
  - The results are simulation only.
  - The classifier covers 6 coarse topics.

### 4.8 Critical view for HBM+HBF
**Transfers well:**
- **The four-class placement (hot / KV / cold / "consumed-elsewhere" data) keyed to tier speed is the most directly reusable placement idea for H^3.** Mapped onto HBM+HBF (my note):
  - Hot or shared weights (shared experts, attention projections, embeddings) go to HBM.
  - KV goes to HBM (writes).
  - Cold experts and cold dense weights go to HBF.
  - "Non-NMP" data consumed off-path goes to the slowest location.
- Stratum's rule that **data whose consumer is behind a narrower link goes to the slowest tier** generalizes: if an operand is interposer-bound anyway, tier latency is irrelevant for it.
- **Alg. 1** (sort by usage, fill fast rows from one end and slow rows from the other, uniform row-to-tier quantization) is simple, deterministic and page-friendly. It carries over to HBF pages or blocks if the unit Δ is set in flash pages.
- The sensitivity curve **throughput vs hot-hit rate (Fig. 17)** is the right experiment for an HBM-cache-over-HBF design: it shows how much prediction accuracy is worth.
- Topic-aware batching is a scheduler-level lever for improving tier hit rate without changing hardware.

**Misses for NAND:**
- Stratum's tiers differ by **1.6x in latency (2–23 ns)**. HBM vs NAND differs by roughly 1000x (ns vs µs), so hit-rate sensitivity will be far steeper. At a 31.6% hit rate, cold accesses to NAND would dominate (my note).
- **Expert swaps are writes into the slow tier.** Swaps are free in DRAM (below 0.4% time), but on HBF each swap programs flash and costs endurance. With 2.6–5.9 swaps per second on DRAM, a flash design must **never write-back**: keep all experts resident in flash and only copy hot ones into HBM, which is a read-only cache fill. (My note: this matches the student's finding that GC is absent when flash is write-once or read-mostly, while wear concentrates if placement churns.)
- Row-level (32 Kb row buffer) movement vs flash page (KB) and erase block (MB) granularity is not discussed.
- There is no TCO, so it cannot be cited for the $ argument.

### 4.9 BibTeX
```bibtex
@inproceedings{pan2025stratum,
  title     = {Stratum: System-Hardware Co-Design with Tiered Monolithic {3D}-Stackable {DRAM} for Efficient {MoE} Serving},
  author    = {Pan, Yue and Xia, Zihan and Hsu, Po-Kai and Hu, Lanxiang and Kim, Hyungyo and Sharda, Janak and Zhou, Minxuan and Kim, Nam Sung and Yu, Shimeng and Rosing, Tajana and Kang, Mingu},
  booktitle = {Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture (MICRO '25)},
  year      = {2025},
  address   = {Seoul, Republic of Korea},
  publisher = {ACM},
  doi       = {10.1145/3725843.3756043},
  isbn      = {979-8-4007-1573-0},
  note      = {arXiv:2510.05245}
}
```

---------------------------------------------------------------------

## 5. Cross-paper observations and gaps relevant to HBM+HBF

1. **All four papers agree on the split: weights to the slow tier, KV and attention to the fast tier.**
   - H2M2: attention gets HBM first; fc and qkv weights are evicted first and never return.
   - LIA: parameters in CXL, KV in DDR. Measured CPU-side slowdown is up to −82% for KV sublayers vs up to −70% for parameter sublayers.
   - Stratum: KV in mid tiers, cold weights in slow tiers.
   - Reason (H2M2, LIA): weight GEMMs gain arithmetic intensity with batch, while attention stays at about 1 op/byte.
   - This supports H^3's "read-only data in flash" from three independent angles. Caveat: none of them puts *shared-context KV* in the slow tier, which H^3 does. The student has to justify that separately. It is read-only, so the endurance argument holds, but its ops/byte is about 1, so the bandwidth argument is against it.

2. **Parallel tiers beat a strict hierarchy when locality is low.**
   - H2M2: Hierarchical gets 1.07x vs H2M2's 1.46x on GPT3-175B, and hierarchy wins only when everything fits in HBM.
   - LIA: FlexGen-style on-demand transfer is more than 98% of latency.
   - For HBM+HBF, flash should be **read directly as a parallel bandwidth source**, not used as a backing store paged into HBM. That is only viable if HBF bandwidth is high, which is HBF's premise.
   - H2M2's min-max objective `max(T_fast, T_slow)` is the right form for the student's analytic model.

3. **Fast-tier capacity dominates; fast-tier bandwidth barely matters once split.**
   - H2M2 sensitivity: HBM capacity has high impact, HBM bandwidth very little.
   - CENT Fig. 1: throughput plateaus exactly at memory exhaustion.
   - This directly backs the student's result that reallocating flash budget to HBM capacity increases concurrency.

4. **Placement granularity and dynamism:**
   - H2M2: per-head, re-solved every iteration in 0.05 ms, hardware paging at 2 MB.
   - LIA: per-sublayer, per-stage (B, L) policy, static during decode.
   - Stratum: per-expert, per-batch or topic switch, with hardware row swaps.
   - CENT: static, per-block, user-configured.
   - For HBF the natural choice is **static placement of weights at load time, plus a read-only HBM cache for hot pieces**. Every dynamic scheme here (H2M2 migration, Stratum swaps) assumes slow-tier writes are free, **which is false for NAND** because of endurance and program latency. None of the four models write cost or wear.

5. **Absent from all four:**
   - µs-class access latency (the slowest tier modelled is CXL-DRAM at +140–170 ns, per LIA);
   - page- or block-granular access;
   - read amplification;
   - ECC or read-retry latency;
   - endurance or wear levelling;
   - any NAND $/GB.
   - The analytic templates need extending with a latency term and a page-rounding term. The fragmentation term from H2M2 Eq. 2 can be reused with the flash page size.

6. **Cost methodologies worth reusing** (none covers flash):
   - **CENT**, the most complete: 3-year owned TCO (capex + $0.139/kWh x average power) and rental TCO (Azure), tokens/$ = throughput / TCO-per-hour, iso-power comparison, and chip cost from wafer $/area/defect density, packaging % and NRE/volume. Its price assumptions (A100 = $10K after deducting 50% margin; PIM = 10x DRAM) are aggressive and should not be copied without sensitivity analysis.
   - **LIA**: $/million tokens = (system price / 3 years + TDP x $0.1/kWh) / tokens; per-GPU throughput; iso-cost alternatives (3x V100 ≈ GNR-A100); memory $/GB (DDR $11.25 vs DDR+CXL $5.60) leading to an 8–9% system-cost saving.
   - **H2M2**: qualitative only.
   - **Stratum**: throughput per die area only.
   - Recommendation (my note): combine LIA's simple $/Mtoken formula with CENT's capex/opex decomposition. Report both small-batch and moderate-batch regimes, because LIA shows the cost winner flips (B=1: LIA 1.5–2.0x cheaper; B=64: 1.3–1.4x more expensive).

7. **GPU-count anchors stated in these papers** (usable in the motivation chapter):
   - OPT-175B needs at least 5 H100s, about $150K; at least 2 H100s even at 4-bit, weights only (LIA).
   - DGX-A100 8-GPU system is $200K vs $22K for single-GPU GNR-A100 (LIA).
   - Llama2-70B vLLM on 4x A100 (320 GB) saturates at batch 128 (4K context) and 8–16 (32K context) (CENT).
   - ChatGPT is about 3617 HGX A100 servers and about $694,444 per day (CENT, quoting [19]).
   - 8-HBM (768 GB) is 1.57x faster than a single HBM+LPDDR board but uses 1.72x more memory energy per token (H2M2: 1.31 vs 0.76, both relative to the LPDDR-only baseline). That ratio is my arithmetic from their numbers.
   - None of these papers computes "how many GPUs does design X replace" as a closed-form formula. CENT's iso-power sizing and LIA's per-GPU throughput are the closest. **This is a gap the student's report can fill.**

8. **Prediction-driven hot/cold placement and its sensitivity** (Stratum): throughput gain vs hit rate is 1.32–1.51x at up to 100% hit, while real hit rates are 31.6–68.9%. With a 1000x latency ratio (HBM vs NAND) instead of 1.6x, a design that relies on hit rates will be much more fragile. HBF should instead be sized to *stream* its resident weights at full bandwidth, needing no reuse (my note, consistent with H2M2's low-locality argument).

9. **The link to the capacity tier matters more than compute near it** (LIA Grace-Hopper analysis; CENT's and Stratum's use of internal bandwidth). This supports putting flash on the interposer (HBF) rather than behind PCIe or CXL. It also warns that if HBF bandwidth is well below HBM, the balanced split from H2M2's min-max drives most work back to HBM, and HBF becomes just a capacity store.

10. **Numeric inconsistencies inside the papers**, which matter if the student quotes them:
    - CENT: energy 2.9x (abstract, Sec. 7.2) vs 2.3x (conclusion); GPU saturation at 32K is batch 8 (Sec. 2) vs 16 (Sec. 7.1).
    - Stratum: internal bandwidth 19.01–30.34 TB/s (text) vs 19.01–34.34 TB/s (Table 3); classifier accuracy 85.0/81.0% (Sec. 5.1) vs 94.5/85.0% (Sec. 6.3.1).
    - LIA: Eq. 8 p_i labels swapped; "longer latency / lower throughput" wording in the V100 comparison; CXL gain given as 1.45x / 1.5x and batch gain as 1.76x / 1.8x in different places.
    - H2M2: "H2D2" typo.
    - Quote the body-text or table value and say which one you used.
