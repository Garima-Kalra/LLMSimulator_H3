# D2: SSD / near-storage offloading for LLMs: reading notes

Context: BTP report (`BTP_Report.tex`) evaluates H3 (HBM + High-Bandwidth Flash). It argues that
the constraint is capacity, not arithmetic. Read-only weights and the shared cache go in HBF. Flash is written
only when requests are deliberately over-subscribed and KV spills. The report treats wear
concentration and static wear levelling, not GC, as the factor that decides lifetime. It also recommends a DRAM write-combining buffer for
small KV appends (8x WA reduced to about 1.4x).

Sources: all four papers were read in full from the plaintext extractions in `scratchpad/txt/`.
Every number below comes from the papers. Where I derived or checked arithmetic myself, it is marked
**[my derivation]**. Where a bibliographic field is not in the text, it is marked **unknown**.

None of the four papers mentions HBF or on-package flash. All of them target PCIe-attached SSDs/CSDs.

---

## 1. HiFC (NeurIPS 2025)

### 1.1 Bibliographic
- **Title:** "HiFC: High-efficiency Flash-based KV Cache Swapping for Scaling LLM Inference"
- **Authors:** Inho Jeong (Seoul National University; SK hynix), Sunghyeon Woo (Seoul National University; NAVER Cloud), Sol Namkung (Seoul National University), Dongsuk Jeon (Seoul National University; corresponding author). Jeong and Woo share equal contribution.
- **Venue:** "39th Conference on Neural Information Processing Systems (NeurIPS 2025)".
- **Year:** 2025. **Pages / DOI / arXiv id:** unknown (not in text). The "12452" in the filename is probably a submission/paper ID, but this is unconfirmed, so it is not cited.
- Funding note: SK hynix supported Jeong's graduate studies. NAVER Cloud contributed discussions.

### 1.2 Problem
- Long-context KV caches exceed GPU HBM. vLLM swaps KV pages to host DRAM, but large DRAM pools are expensive in CapEx, power and cooling.
- SSD offloading (e.g., FlexGen) is cheap, but every flash I/O goes through host DRAM and PCIe because GPUs have no native NVMe interface. This causes a "severe bandwidth bottleneck" (they cite InstInfer [13]).
- Frequent non-sequential writes raise write amplification (WA) and wear.
- The paper lists three design challenges (§3.1):
  1. SSD throughput is lower than DRAM, especially for random access.
  2. Flash endurance is limited, and small writes increase WA.
  3. CPU DRAM staging adds PCIe transfers and copies.

### 1.3 Core mechanism
- **What is offloaded:** KV cache blocks only. This is **swapping**, not permanent offload: blocks move to flash only when HBM is exhausted during decode. A victim sequence group is chosen by vLLM's scheduler (LRU-plus-priority), swapped out, and swapped back in later. Weights stay in HBM. Compute stays entirely on the GPU; there is no in-storage compute.
- **DRAM-free data path:** GPUDirect Storage (GDS) moves KV tensors directly between HBM and NVMe SSD. It uses byte-level offsets, multi-threaded I/O (up to 16 threads), and reuses tensors as 4 KB-aligned GDS buffers. The paper claims "over 4.7 GiB/s in the pSLC region".
- **Flash Cache (FC) block allocator:** replaces vLLM's CPU block allocator. It uses fixed-size blocks (32–128 tokens) for compatibility with PagedAttention.
- **Flash-aware block management ("append policy"):** evicted blocks are allocated in physical order, and stale logical addresses are never reused. The result is sequential I/O. Fig. 5 contrasts this with a "left-append" policy, which stays sequential for about 500 swap-outs and then becomes random.
- **Byte offsets (App. B):** a precomputed per-layer lookup table.
  - block_size = H·S·D·T
  - layer_offset = l·(2·B·block_size)
  - K region first, then V region
  - KV blocks are aligned to 4 KB LBAs.
  - Example: DS-Qwen-32B (L=64, B=3200, H=8, S=128, D=128, FP16). Layer 63 starts at 98.44 GiB.
- **pSLC for wear:** HiFC selects commodity SSDs that have a pSLC region and operates "exclusively within that region". The paper says this region is 20% of capacity: a 200 GiB dynamic SLC cache on a 1 TB drive. All KV traffic is confined to a fixed 200 GiB LBA range.
- **Scheduling / latency hiding:** HiFC relies on vLLM's pipelined Running / Swap-Out / Swap-In scheduling. App. G gives a visibility criterion:
  - Stall budget: T_gap = ΔTPS·T_lat.
  - Swap time: T_swap = T_fix + N_swap·B_size·s_tok / BW_Flash.
  - A swap is visible only if T_swap > T_gap.
  - Worked example: a 29-block swap (32 MiB blocks = 128 tokens) costs 266 ms of flash latency. The budget is T_gap = 2.07 s (ΔTPS = 46 tok/s, T_lat ≈ 45 ms), so the swap is hidden.
- **Interconnect bottleneck addressed:** GDS removes the host bounce. However, the SSD link is still PCIe Gen4 x4-class, and DRAM swap bandwidth is ">2×" HiFC's (Fig. 6a). The design works only because swap latency is amortised by concurrency, not because the bandwidth matches.

### 1.4 Device / system parameters
| Item | Value | Source |
|---|---|---|
| Flash-cache SSD | "SSD, 1 TB NVMe Gen4 (pSLC cache: 200 GiB)". The model is not named; App. D calls it "SSD-A" and describes it as consumer-grade | Table 16, App. D |
| Official endurance | 750 TBW (TLC, full drive), about 3,000 P/E | App. D |
| Assumed pSLC endurance | about 30,000 P/E → "6,000 TiB" for the 200 GiB region | App. D, Eq. 8–9 |
| Measured GDS throughput, 100 GiB LBA range (inside pSLC) | SEQ WRITE avg 4.715 GiB/s (min 4.341, max 4.724); SEQ READ avg 4.987 GiB/s; RND WRITE avg 1.617 GiB/s (min 1.092, max 2.703) | Table 5 |
| Measured GDS throughput, 900 GiB LBA range (exceeds pSLC) | SEQ WRITE avg 1.689 GiB/s (min 1.416, max 1.841) | Table 5 |
| Server | Dell PowerEdge R750xa, 2× Xeon Silver 4310, 256 GiB DDR4, 2× A100 80 GiB | Table 16 |
| Software | vLLM v0.6.6, CUDA 12.3, PyTorch 2.5.1, GDS 1.8.1.2 | Table 16 |
| Cost model (Table 1, H1 2025) | DRAM 128 GiB: $433, 64 W, 3-yr $614. Enterprise TLC SSD 1.92 TiB: $270, 8.2 W, $303.6. HiFC 1 TiB: $118, 5 W, $136 | Table 1 |
| Cost model (updated H2 2025, Table 12) | DRAM $5.06/GiB, TLC SSD $0.20/GiB, HiFC $0.10/GiB. 3-yr cost: $729 / $465 / $120. PUE 1.3, $0.1/kWh. DRAM power is listed as 24 W here, vs 64 W in Table 1 | App. C |

### 1.5 Evaluation and headline results
- **Main model:** DeepSeek-R1-Distill-Qwen-32B. **Others:** DS-Llama-8B, DS-Qwen-14B, Mistral-7B. **Datasets:** Qasper (avg 3.6k), GovReport (8.7k), NarrativeQA (18.4k), LongBench gov_report_e.
- **Throughput vs DRAM swapping:**
  - The two are "near-identical" (Fig. 3).
  - Across models and datasets, HiFC is within 1–2% of DRAM (Table 3). Examples: DS-Llama-8B GovReport 182.3 vs 172.0 tok/s; DS-Qwen-14B NarrativeQA 46.6 vs 46.4.
- **Capacity:**
  - Table 2 (DS-Llama-8B, 2× A100, bs=200 GovReport): GPU+DRAM (50 GiB) caches 51 requests at 172 tok/s. GPU+SSD 1:1 (200 GiB pSLC) caches 206 requests at 182 tok/s. 2:1 gives 367 tok/s. 2:2 (400 GiB) caches 412 requests at 364 tok/s.
  - Claimed "4× more requests" and "4.5× lower memory expansion cost" (6.1× with H2 2025 prices).
- **Swap onset:** A100 with 9.1 GiB KV budget, 5k context: swaps begin beyond 8 sequences, and throughput saturates at 12 (Fig. 4).
- **Block size (Fig. 6):** DRAM swap bandwidth is more than 2× HiFC's, yet end-to-end throughput is nearly identical. 64 tokens is the "sweet spot"; 128/256 cause redundant swaps.
- **Latency under starvation (Table 19):**
  - Setup: DS-Llama-8B, 4 GiB HBM for KV.
  - GPU-only (recompute): 179 tok/s, 5.5 s/request, KV limited to 4 GiB (OOM).
  - DRAM swap (50 GiB): 172 tok/s, 5.8 s/request.
  - HiFC (200 GiB): 182 tok/s, 5.4 s/request.
- **Initialization time (Table 6):** HiFC stays at 32–34 s regardless of cache size. DRAM grows to 90 s at 100 GiB, a 2.81× difference.
- **WAF (Table 4/21):** 7,440 swap-outs × 32 MB = 232.5 GiB of host writes. SMART "Data Units Written" increased by 487,595. The paper reports WAF ≈ 1.02, "27% lower" than the SSD spec's 1.40.
- **Lifetime (Table 15):**
  - Measured: 28 swap-outs, avg 984 MiB, 26 GiB per test, 1,097 s per test.
  - Extrapolated: 78 tests/day → 1.98 TiB/day.
  - With 6,000 TiB pSLC TBW, the predicted lifetime is **8.3 years**.

### 1.6 Stated limitations and future work (§6 + checklist)
- **Limitations:**
  - Short-context or latency-sensitive workloads may see flash latency in token processing.
  - Multi-GPU sharing of a flash cache needs bandwidth scheduling to avoid contention.
  - Consistent SSD performance requires domain expertise in configuration and filesystem tuning.
  - The checklist adds: results depend on GC and TLC↔pSLC migration behaviour; real deployments may see I/O variance; nothing was validated on 100B+ models.
- **Future work:** hybrid Flash–DRAM caching schemes; sharing resources and know-how for SSD optimisation.

### 1.7 Critical view (for HBM + HBF)
Problems with the evidence:
- **The WAF measurement is not a NAND WAF.** Per the NVMe spec, SMART "Data Units Written" counts data the *host* wrote to the controller, in units of 1000×512 B. It does not count NAND program volume. So the reported ratio only compares vLLM's accounting with the NVMe host-write counter.
  - **[my derivation]** Eq. 16 evaluates to 487,595 × 512,000 / (7,440 × 32 × 1024²) = **1.00001**, not 1.02.
  - Physical WA inside the drive (SLC-cache folding, GC) was never measured.
  - For the report's endurance chapter, **do not cite HiFC as evidence of WA ≈ 1.02 at the NAND level**. It is evidence only that the host write stream is not inflated.
- **The "8×" endurance claim compares unlike things.** It sets theoretical pSLC P/E (30,000 × 200 GiB) against the vendor's *warranted* 750 TBW. HiFC's own Table 14 shows theoretical TLC full-drive at 3,000 TiB, so the raw pSLC gain is 2×, not 8×. That gain also comes from using only 20% of the drive's capacity.
  - **[my derivation]** 30,000 × 200 GiB = 5,859 TiB, not 6,000 TiB. The paper mixes units.
  - The 30,000 P/E figure is asserted without a source.
- **"pSLC region" means a consumer dynamic SLC cache, not a configured static pSLC partition.** Whether the firmware keeps a fixed 200 GiB LBA range in SLC mode indefinitely is assumed ("assuming minimal cache folding"). Table 5 shows the cliff: sequential writes fall from 4.7 to 1.69 GiB/s once the working range exceeds the SLC cache.
- **HiFC's DRAM-parity result depends on the workload being compute-bound with slack** (Fig. 4, App. G). They admit that with a faster GPU, "the overhead of KV cache swapping can emerge as a bottleneck".

What transfers to HBF:
- **Its regime is exactly the report's "deliberate over-subscription" regime:** swap only when HBM is full, at whole-sequence granularity. HiFC is therefore the closest real-system evidence that **spill-granularity, sequential, append-only writes** are what keep flash writes benign. That supports the report's "one-off spill ≈ 1.0 WA" row. HiFC does not address the per-token append population, because vLLM's swap moves whole blocks and decode appends happen in HBM after swap-in.
- **App. G's T_swap vs T_gap criterion** is a simple, citable analytical model. It could be ported to decide when an HBF spill is visible in the simulator, with BW_Flash replaced by HBF bandwidth.
- **Append-only allocation with no reuse of stale addresses** turns allocation policy into the WA determinant. This echoes the report's finding "block allocation, not the collector".
- **pSLC-only operation trades capacity for endurance and write bandwidth.** H3 already assumes SLC. The report found HBF is 17–67× over-provisioned, so operating HBF in SLC mode, or even partly as a pSLC write-staging region, costs capacity that is not needed anyway.

What HiFC does not address:
- Wear levelling between hot and cold regions. HiFC dedicates the whole drive region to KV, so there is no cold weight data.
- Near-memory compute, retention, read disturb, and sharing flash between a weight tier and a KV tier.

### 1.8 BibTeX
```bibtex
@inproceedings{jeong2025hifc,
  title     = {{HiFC}: High-efficiency Flash-based {KV} Cache Swapping for Scaling {LLM} Inference},
  author    = {Jeong, Inho and Woo, Sunghyeon and Namkung, Sol and Jeon, Dongsuk},
  booktitle = {39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025},
  note      = {Pages/DOI not stated in the paper text}
}
```

---

## 2. InstAttention (HPCA 2025)

### 2.1 Bibliographic
- **Title:** "InstAttention: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference"
- **Authors:**
  - Peking University: Xiurui Pan, Endian Li, Yingwei Luo, Xiaolin Wang, Jie Zhang
  - UESTC: Qiao Li
  - ICT, Chinese Academy of Sciences: Shengwen Liang
  - Huawei Cloud: Yizhou Shan
  - WNLO, Huazhong University of Science and Technology: Ke Zhou
- **Venue:** 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA).
- **Pages:** 1510–1525. **DOI:** 10.1109/HPCA61900.2025.00113. **ISBN:** 979-8-3315-0647-6.
- **Code:** https://github.com/ChaseLab-PKU/InstAttention
- **Name note:** the arXiv preprint of the same work is titled **"InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference"**, arXiv:2409.04992. HiFC cites it under that name (HiFC ref [13]). Cite the HPCA version.

### 2.2 Problem
- Offline (throughput-oriented), long-context, large-batch inference in resource-constrained settings (edge, small servers).
- The KV cache is huge:
  - FP16 KV size is 4·b·s·p, vs 2·p for the model.
  - OPT-13B at bs=128, 2K context: ~24 GB of weights vs ~200 GB of KV.
  - OPT-175B: 325 GB weights, 2.63 TB KV.
- Offloading to host memory or SSD is limited by PCIe:
  - FlexGen's decode time is up to 98.94% "KV Cache Access" (Fig. 5).
  - DeepSpeed loses 97.01% of throughput when it starts swapping to SSD (motivation section).

### 2.3 Core mechanism
- **What is offloaded:** the KV cache, entirely resident in the CSD for the whole inference. Also **decoding-phase attention** (Logit and Attend GeMVs). The GPU keeps prefill (including prefill attention), QKV/O projections and FFN.
- **Rationale (roofline, Fig. 6):** the CSD (Zynq7045 prototype) has at most 650 GFLOPS, which would need 650 GB/s at decode attention's 1:1 FLOP/byte. That is unreachable over PCIe (3–10 GB/s), but internal flash-channel bandwidth "tens of GB/s" narrows the gap. Only the operators that are low-intensity and need the KV cache are offloaded.
- **SparF attention:** a flash-aware version of SparQ. Algorithm 1:
  1. Take the top-r |q| channels.
  2. Load K for those channels (channel-indexed).
  3. Compute approximate scores.
  4. Take the top-k tokens.
  5. Load full K and V rows for them (token-indexed).
  6. Blend with the mean V (ᾱ weighting).
  - **Dual-step loading:** filtering first at page (group) granularity, then at entry granularity in the NAND flash controller (NFC) filters. Default compression is 1/8. First-step loading "maintains about half sparsity"; the second step reaches full sparsity.
- **Flash layout / KV-oriented FTL:**
  - **Token-indexed mapping:** 16 consecutive tokens of one head (128 × FP16 = 256 B per token) form one 4 KB page group. Groups are striped across flash channels.
  - **Channel-indexed mapping:** 2–8 channels per page, so minimum granularity is 256–1K tokens.
  - **K is stored twice** (token- and channel-indexed), trading capacity for bandwidth. SparQ itself needs 1.5× footprint.
  - Custom 32-bit logical address with fields Task/Batch/Layer/Token/Head/Channel. Two L2P tables sit in CSD DRAM. The host file system is bypassed.
- **Write handling / WA:**
  - Decode k,v vectors are buffered in the CSD's internal DRAM "group buffer" until a page fills, then flushed in the background.
  - Groups from **different attention heads are packed into the same flash block**, so writes are block-granular. Writes are append-only, which eliminates fragmentation and foreground GC.
- **GC:**
  - Host-directed. A new NVMe command `reclaim()` erases the stale KV of finished tasks, in LRU order.
  - GC runs only when the CSD is idle and the free page budget is below a threshold.
  - No valid-page copying: KV is transient, so the drive needs no persistence.
  - Other new commands: `config()` and `attend()`.
- **Interconnect:**
  - GPU↔CSD is PCIe P2P DMA. The driver rewrites NVMe DWord10 with the custom logical address. The paper stresses that this differs from GDS, which still depends on the host file system.
  - Prefill KV is transferred layer by layer, overlapped with the next layer's compute.
  - In decode only q, k, v vectors and attention outputs cross PCIe; the paper says traffic is "reduced by s/2".
  - **Multi-CSD:** attention heads are partitioned across CSDs (n_head/n heads each).
- **Endurance trick:** **retention relaxation**. KV is transient, so relaxing retention "from the typical three years to three weeks" is claimed to raise P/E endurance about 6.67×, citing prior studies. Read-retry costs are then evaluated.

### 2.4 Device / system parameters
| Item | Value | Source |
|---|---|---|
| Real CSD | Daisyplus OpenSSD: Xilinx ZU17EG MPSoC (mid-range FPGA + 4-core ARM), 2 GB DRAM, PCIe 3.0 x4. SparF engine and NFC filters on FPGA at 285 MHz; FTL in software on ARM | §V-A |
| Emulated CSD | NVMeVirt with measured OpenSSD latencies; cheaper Zynq7045 prototype; 8 flash channels × 1.4 GB/s ("align with Samsung 980pro"); external PCIe 4.0 x4 (max 7 GB/s) | §V-B |
| Emulation accuracy | about 95% (Table I). GeMV 0.32 µs, 12.7 GFLOPS real / 13.3 virtual. Softmax 164 µs. Filter 37 µs, 1.85 GB/s | Table I |
| FPGA utilisation (Zynq7045) | LUT 80.27%, FF 58.92%, BRAM 46.06%, DSP 85.33% | Table II |
| Internal bandwidth | 11.2 GB/s (vs 32 GB/s GPU↔host PCIe Gen4 x16) | §VI-C |
| Testbed | A6000 48 GB, Xeon 5320 2.2 GHz, 96 GB DDR4, Samsung 980 Pro 2 TB, GPU on PCIe Gen4 x16 | §VI-A |
| Flash endurance reference | V-NAND V6 (in the 980 Pro), 3,000 P/E | §VI-F |
| SSD background | 8–16 channels × 1–2 GB/s; 4–16 KB pages; external PCIe 3–6 GB/s (also given as 3–7 GB/s elsewhere) | §II-C, §I |
| Cost comparisons (discussion) | Samsung SmartSSD ≈ $1,500; GH chip ≈ $30,000; GH max 624 GB fast memory | §VII |

### 2.5 Evaluation and headline results
- **Baselines:** DeepSpeed-MII + ZeRO-Inference (host memory), FlexGen (SSD), FlexGen-GDS, FlexGen-SparQ (1/8), vLLM Recomp. Variants: InstA (dense) and InstA-SparF.
- **Models:** OPT-13B, OPT-30B (2× A6000), Llama-2-13B. FP16.
- **Sequence lengths:** OPT: in = out = 1024. Llama-2: in = out = 2048.
- **Datasets:** ShareGPT, WikiText-2, SQuAD, TriviaQA.
- **Accuracy (Fig. 13):** SparF ≈ SparQ, and "negligible accuracy loss" up to 1/8 compression. (HILOS later disputes this for long-context models; see §3.7.)
- **1 CSD, OPT-13B:**
  - InstA beats FlexGen by 6.85× at bs=64.
  - InstA-SparF improves on InstA by up to 2.08× at bs=256 and beats FlexGen by **up to 11.1×** (the headline).
  - InstA-SparF beats Recomp by 71.3% at bs=256.
  - InstA beats DeepSpeed's best (bs=16) by only 4.6%, because internal bandwidth (11.2 GB/s) is below host PCIe (32 GB/s).
  - FlexGen-GDS shows negligible gain over FlexGen.
- **2 CSDs:**
  - InstA (bs=256) is 10.5× FlexGen's best (bs=32).
  - InstA-SparF is 3.11× FlexGen-SparQ.
  - Traditional offloading gains little from 2 SSDs.
- **OPT-30B:** InstA / InstA-SparF reach up to 4.09× / 9.39× FlexGen. **Llama-2-13B:** 5.68× / 12.48×, and Recomp reaches 9.26× FlexGen.
  - The 12.48× exceeds the "up to 11.1×" headline. The headline is for OPT-13B.
- **Decode latency breakdown (Fig. 18/19):**
  - Dense, bs=64: KV-access share falls from 98.9% (FlexGen) to 80.7% (InstA) and 76.4% (2 CSDs).
  - Sparse: from 92.4% to 82.3% and 74.0%.
  - KV access is still the dominant cost.
- **Scaling:** 20 CSDs vs 1 give 8.99× (dense) and 7.29× (sparse) at bs=256.
- **Data migration:** reduced "by up to 94.0%".
- **I/O volumes (Table III, OPT-13B, bs=64):**
  - Prefill write 1.34 GB at 3.78 GB/s.
  - Decode read 1085.65 GB at 7.66 GB/s.
  - Decode write 1.93 GB at 2.15 GB/s.
  - Reads exceed writes by about 810× (vs prefill writes) and 560× (vs decode writes).
  - Decode writes are slightly slower because of 1-token granularity, but the effect is negligible.
- **Endurance (§VI-F):**
  - 4 CSDs, 13B model, about 0.78 MB of KV per token → "32,263,877K tokens".
  - With retention relaxation (6.67×) and 128K tokens/user/day: at least 920 extreme users over 5 years, 1,530 over 3 years, or 4,600 over 1 year.
  - Read-retry penalty after relaxation, using worst-case statistics from prior work (8K P/E, 10 days bake at 85 °C, up to 5 retries, average 0.003): at most **4.7‰** throughput degradation.
  - **[my derivation]** 0.78 MB/token matches OPT-13B: 40 layers × 5120 × 2 (K,V) × 2 B = 819,200 B = 0.781 MiB. 4 × 2 TiB × 3,000 P/E / 819,200 B ≈ 3.22 × 10¹⁰ tokens, which matches the paper's figure within 0.2%. So the paper assumes **WA = 1, no wear-levelling overhead, and full drive capacity cycled**. The user counts are consistent with ×6.67 and 128,000 tokens/day. Note that K is written twice for SparF, which the 0.78 MB/token figure does not appear to include.

### 2.6 Stated limitations and future work
- No dedicated limitations section. Limitations stated in the text:
  - CSD compute is "2–3 orders of magnitude weaker than GPUs".
  - The OpenSSD platform has expensive FPGAs, limited flash, and supports only legacy Z97 boards. For this reason most results are **emulated with NVMeVirt**.
  - KV access remains the bottleneck even with InstA.
  - The target is offline inference only.
- **Future directions:**
  - Scale to more CSDs or flash channels.
  - Combine the Grace-Hopper high-bandwidth interconnect with InstAttention ("promising solution in the future").
  - A host-side SparF engine for non-CSD systems.

### 2.7 Critical view (for HBM + HBF)
What transfers:
- **"Attention where the KV lives" is the canonical argument for putting attention logic on the HBF base die.** It pays only if the bandwidth *inside* the flash stack exceeds what crosses the link to the GPU/HBM.
  - For InstAttention: 11.2 GB/s internal vs 7 GB/s external. The gain was modest; InstA was only 4.6% over DRAM offload.
  - For H3's chained arrangement (HBF behind HBM), the HBM↔HBF link is the analogue of PCIe. Decode attention near HBF would cut traffic on that link by roughly the sequence length. The report's result that "capacity dominates wiring" suggests this link is not the main bottleneck in H3. It becomes one if shared-cache attention reads dominate.
- **Host-directed bulk reclaim of dead KV, with no valid-page copying**, is directly applicable. HBF KV spill regions can be erased per request or task, with no FTL-level GC. This matches the report's finding that GC is structurally absent. InstAttention is a real-hardware precedent for an **application-managed FTL with semantic logical addresses** (task/layer/token/head).
- **Packing heads into one erase block with DRAM-staged group writes** is the same idea as the report's DRAM write-combining buffer. InstAttention does it with in-SSD DRAM, while the report places the buffer in HBM/DRAM. This is a citable precedent.
- **Retention relaxation for endurance (6.67×)** complements H3, which already uses relaxed retention to get *latency*. The report's "retention loophole" subsection only considers refresh-induced rewrites. InstAttention adds the other side: relaxed retention *raises* P/E budget, but at the cost of read-retry.
- **Storing K in two layouts** uses spare capacity to save bandwidth. HBF has spare capacity (17–67× over-provisioning per the report), so dual layouts cost little capacity. But they **double K writes**, which matters in the spill regime.

What it does not address:
- On-package integration.
- Wear levelling between hot KV and cold data.
- GQA models: all models tested are MHA.
- Long-context accuracy of lossy sparsity. HILOS measured −3.52 to −5.73 %p on Qwen2.5-32B-32K.
- The endurance arithmetic ignores its own double-K write overhead and any WA.
- Most multi-CSD results are emulated.

### 2.8 BibTeX
```bibtex
@inproceedings{pan2025instattention,
  title     = {{InstAttention}: In-Storage Attention Offloading for Cost-Effective Long-Context {LLM} Inference},
  author    = {Pan, Xiurui and Li, Endian and Li, Qiao and Liang, Shengwen and Shan, Yizhou and Zhou, Ke and Luo, Yingwei and Wang, Xiaolin and Zhang, Jie},
  booktitle = {2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
  pages     = {1510--1525},
  year      = {2025},
  doi       = {10.1109/HPCA61900.2025.00113}
}
% Preprint of the same work (earlier title), as cited by HiFC:
@misc{pan2024instinfer,
  title        = {{InstInfer}: In-Storage Attention Offloading for Cost-Effective Long-Context {LLM} Inference},
  author       = {Pan, Xiurui and Li, Endian and Li, Qiao and Liang, Shengwen and Shan, Yizhou and Zhou, Ke and Luo, Yingwei and Wang, Xiaolin and Zhang, Jie},
  year         = {2024},
  eprint       = {2409.04992},
  archivePrefix= {arXiv}
}
```

---

## 3. HILOS (ASPLOS 2026)

### 3.1 Bibliographic
- **Title:** "A Cost-Effective Near-Storage Processing Solution for Offline Inference of Long-Context LLMs". HILOS is the system name; it does not appear in the title.
- **Authors:** Hongsun Jang, Jaeyong Song, Changmin Shin, Si Ung Noh, Jaewon Jung (Seoul National University); Jisung Park (POSTECH); Jinho Lee (Seoul National University; corresponding author).
- **Venue:** Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 22–26, 2026, Pittsburgh, PA, USA. ACM, 21 pages.
- **DOI:** 10.1145/3779212.3790119. **ISBN:** 979-8-4007-2359-9. **arXiv:** 2502.09921 (v2, 6 Feb 2026, cs.AR). **Page range:** unknown.
- **Code:** https://github.com/hongsunjang/HILOS. Artifact on Zenodo: 10.5281/zenodo.18162119.
- **Funding:** includes Samsung Memory Research Center.

### 3.2 Problem
- Offloading-based batched **offline** inference (FlexGen / DeepSpeed-style) spills weights and KV to host memory and SSD. KV I/O is ">60% of inference time" (Fig. 2, OPT-175B, A100, 512 GB host memory, 4 PCIe 4.0 SSDs).
- Multi-SSD RAID is limited by host PCIe lanes and root-port competition (Fig. 3).
- Motivating framing (§1): "equipping costly servers with multiple GPUs to meet increasing memory requirements is often cost-prohibitive". Decode is memory-bound, so GPU compute sits underused. **This is the same motivation as the BTP.**

### 3.3 Core mechanism
- **Attention Near Storage (ANS):**
  - Decode attention runs on an FPGA accelerator inside each SmartSSD. The KV cache moves only over the SmartSSD's private internal PCIe path (SSD → FPGA DRAM).
  - GPU→device traffic is new q, k, v (6h bytes). Device→host traffic is the attention output (2h bytes).
  - Baseline read traffic is 4sh bytes per step, so T_BASE / T_ANS = (4sh + 4h) / (2h + 6h) = (s + 1)/2 (Eq. 3).
  - Work is parallelised over batch × heads across devices.
- **Cooperative X-cache:**
  - For a fraction α of the batch/heads, store the pre-projection activation X instead of K and V. The GPU reloads X via GDS and recomputes K, V while the NSP computes attention on the other 1−α.
  - For MHA, X is half the size of K+V, which halves the flash reads, interconnect traffic and **writes** for that fraction.
  - Optimal α = 2B_PCI / (B_SSD + B_PCI). Measured B_SSD/B_PCI ≈ 3 gives α ≈ 50%, confirmed in Fig. 13. α is rounded to a power of two.
- **Delayed KV cache writeback:**
  - New per-token KV entries (256 B per head, far below the 4 KiB page) are staged in **host memory**. The host CPU computes **partial QKᵀ** for the buffered keys and ships only those scalars plus the new V entries to the accelerator.
  - Entries spill to SSD every c tokens. c = 16 is optimal because 16 × 256 B = 4 KiB.
  - Effect: writes leave the critical path, writes are page-sized, and WA drops.
- **Accelerator (FPGA, HLS):**
  - Temporal, block-wise design (128 tokens per block) to bound on-chip memory for long contexts.
  - **Two-pass softmax:** online-softmax style, using local block maxima.
  - **GEMV with online 128×128 block transpose of K** in on-chip buffers. This avoids storing Kᵀ, or both layouts, which would cost "redundant writes". This is explicitly the opposite choice to InstAttention.
  - **Native GQA:** K/V buffers are broadcast to d_group × 128 MACs.
  - FP16 storage with FP32 accumulate and exponent; padding mask value −10⁴.
- **Software:** PyTorch/FlexGen-based middleware (Inference Controller, Weights Prefetcher, Cache Scheduler, Writeback Manager). C++ with pybind11. Xilinx OpenCL/XRT with P2P buffers (CL_MEM_EXT_PTR_XILINX).
- **Wear:** no pSLC. Endurance is improved by (i) X-cache, which lowers writes by about α/2 %, and (ii) delayed writeback, which gives page-sized writes and less WA.

### 3.4 Device / system parameters
| Item | Value | Source |
|---|---|---|
| CSD | Samsung SmartSSD 3.84 TB NVMe + Kintex UltraScale+ KU15P FPGA + 4 GB DDR4-2400. FPGA↔SSD over an internal PCIe switch | Table 1, §6.1 |
| Accelerator | 296.05 MHz (limit about 300 MHz from the SmartSSD power envelope). 128 MAC units, which saturate DRAM bandwidth | §5.4, §6.2 |
| Accelerator resources and power (Table 3) | d_group=1: 11.9 GFLOPS, 11.25 W. d_group=4: 46.8 GFLOPS, 15.39 W. d_group=5: 56.3 GFLOPS, 16.08 W (LUT 67.4%, DSP 27.79%). 16 devices ≈ 258 W | Table 3, §6.2 |
| Kernel throughput | "far more than 3.0 GB/s", above the SSD's P2P read bandwidth | §6.4 |
| Baseline SSD | 4× Samsung PM9A3 3.84 TB, up to 6,900 MB/s read and 4,100 MB/s write, each on PCIe 4.0 x4 (16 lanes total). 13 W datasheet power | §6.1, §6.6 |
| Topology | Up to 16 SmartSSDs in an H3 Falcon 4109 chassis: host PCIe x16, 8 × x8 ports, 2 SmartSSDs per port (U.2) | §5.3, §6.2 |
| Host | A100 40 GB or H100 80 GB, Xeon Gold 6342 (24C/48T), PCIe 4.0 x16 to GPU | Table 1 |
| Bandwidth model (Fig. 18) | PCIe interconnect about 8 GB/s, CSD internal about 16 GB/s, off-chip memory about 68 GB/s. Four SmartSSDs ≈ 16 GB/s internal storage (16 PCIe 3.0 lanes), 8 GB/s system (4 PCIe 4.0 lanes), 52 GB/s internal memory (4 DDR4 channels) | §7.1 |
| Endurance spec | 7.008 PBW per 3.84 TB SmartSSD, "with a 3-month data retention" [35] | §6.6 |
| Prices | Server $15,000; A100 $7,000; H100 $30,000; PCIe expansion $10,000; SmartSSD $2,400; conventional PCIe 4.0 SSD $400. DRAM about $3/GB vs flash about $0.1/GB | §6.6, §8.2 |
| Envisioned ISP device | 16 TB NAND, 8 channels × 2,000 MT/s = 16 GB/s, LPDDR5X (4 × 16 GB channels) at 68 GB/s. ASIC version of the accelerator (d_group=1; OpenROAD on Nangate45 scaled to 8 nm; SRAM modelled with CACTI 7.0; 300 MHz): **0.47 mm², 1.13 W** on a 32K-token profile | §7.1 |

- **[my derivation]** 7.008 PBW / 3.84 TB = 1,825 full-drive writes, which is 1 DWPD over 5 years.

### 3.5 Evaluation and headline results
- **Models:** OPT-30B / 66B / 175B (MHA); Qwen2.5-32B (GQA, 8 KV heads, d_group=5); Mixtral-8×7B (GQA+MoE); GLaM-143B (MoE).
- **Settings:** contexts up to 128K; default bs=16, FP16, 64 output tokens, N=8 SmartSSDs, α from the model, c=16. Models above 100B parameters have their weights offloaded to storage.
- **Baselines:** FLEX(SSD), FLEX(DRAM), FLEX(16 PCIe 3.0 SSDs) (SmartSSDs with FPGAs disabled), DS+UVM(DRAM). All use FlashAttention for prefill and CPU attention in decode, with RAID-0 via mdadm.
- **Throughput:**
  - FLEX(16 PCIe 3.0 SSDs) reaches only 0.64–0.94× FLEX(SSD), because the PCIe link saturates.
  - HILOS with 4 SmartSSDs: 1.10–1.36× over FLEX(DRAM). With 16: 1.88–2.49×.
  - Where FLEX(DRAM) fails even at bs=1: **5.3–7.8×** over FLEX(SSD). The abstract says "up to 7.86×".
  - GQA/MoE models: 1.16–3.36×.
  - Longer outputs: up to 6.08×.
- **Ablation (Fig. 15):** ANS alone gives up to 3.39×. Adding writeback gives up to 1.32× more over ANS. Adding X-cache gives up to 1.64× more over ANS.
- **Cost-efficiency (tokens/s/$):**
  - 66B: up to 2.02× FLEX(SSD). FLEX(DRAM) is 1.53× more cost-effective when DRAM is sufficient.
  - 175B: up to 1.68×.
  - Against an H100 upgrade (1.39× speedup): HILOS gives 1.29× speedup at **2.91× the cost-efficiency**.
- **Endurance (Fig. 16b):**
  - Request classes from Azure statistics: Small I:256/O:100, Medium I:1K/O:350, Long I:8K/O:350.
  - HILOS gives **1.34–1.47× more serviceable requests** than the baseline. Raising c from 16 to 32 adds 1.02–1.05×.
  - More than 4.08 M long requests on the 175B model with 16 SmartSSDs.
- **Energy:** up to 85% reduction (Fig. 17a). GPU power measured with NVML, CPU/DRAM with RAPL, SmartSSD via the chassis controller.
- **Multi-GPU comparison:** against vLLM 0.9.1 on 2 nodes × 4 A6000 (384 GB total), with TP inside each node, PP across nodes, and InfiniBand EDR: HILOS is **1.64–1.81× faster**. The authors attribute this to small batches and inter-node communication.
- **Accuracy:** lossless vs FlashAttention. InstAttention's 1/8 SparF loses 3.52–5.73 %p F1 on Qwen2.5-32B-32K across 5 LongBench datasets (Fig. 18c).
- **Performance estimator:** Pearson correlation 0.93 with measured hardware over 4K–32K.

### 3.6 Stated limitations and future work (§7)
- A single NSP device does **not** raise internal bandwidth over a conventional system: SmartSSD internal lanes equal its external lanes. Gains come only from aggregating devices. SmartSSDs cannot fully emulate ISP devices because firmware access is restricted.
- PCIe 5.0 SSDs would need 4× accelerator throughput, meaning more than 2,000 DSPs, which exceeds SmartSSD. Softmax takes more than 50% of execution time as d_group grows.
- HLS forces a single clock domain.
- **Capacity is underused:** each SSD stores under 600 GB of its 4 TB even at peak. Host-to-CSD lanes are also underused. The authors call for a "more balanced design": less capacity, more internal bandwidth or compute, and **coarse-grained block-level FTL mapping** instead of page-level. They argue sequential KV access makes this feasible.
- Host-side buffering requires explicit XRT DMA. Throughput drops by more than 30% when c goes from 16 (4 KiB) to 64 (16 KiB), and 16 KiB NAND pages will make this worse.
- **Future work:**
  - ISP devices.
  - Dedicated exponential units.
  - Separate clock domains.
  - CXL.mem unified address space to remove explicit copies.

### 3.7 Critical view (for HBM + HBF)
Strongest ideas to borrow:
- **Split attention across tiers with softmax-statistic merging (delayed writeback + host partial QKᵀ).** HILOS keeps the newest tokens' KV in a fast tier and computes their partial scores there. The near-storage unit handles the bulk and combines both parts through the masked two-pass softmax. For HBF this maps as follows:
  - Keep the **write-combining buffer in HBM**, as the report proposes.
  - Have the GPU compute attention over the buffered tail.
  - Have the HBF-side logic compute over the flashed body.
  - Merge with online-softmax (max, sum) statistics.
  - This removes flash writes from the critical path. HILOS measured that c = 4 KiB/256 B = 16 is the sweet spot, the same page-fill logic as the report's buffer. The report sets its threshold at 75% of an *erase block*; HILOS aligns to a *page*. Both points are worth stating.
- **X-cache as a write-endurance lever.** Storing X instead of K and V lowers flash writes by about α/2.
  - **Caveat not discussed by HILOS [my observation]:** the "X is half of KV" claim holds only for MHA.
  - For GQA, K+V per token is 2 × n_kv × d. For Qwen2.5-32B that is 2 × 8 × 128 = 2,048 elements, vs X = 5,120 elements (HILOS Table 2). So X-cache would **increase** capacity, write and read volume.
  - The GQA results in the paper do not explain how α or X-cache behave there, beyond saying RoPE recomputation is negligible.
  - For modern GQA or MLA models on HBF, X-cache is unlikely to help endurance.
- **On-chip block transpose instead of dual K layouts:** HILOS avoids InstAttention's double-K writes. On a write-limited medium such as HBF, this is the right choice.
- **Base-die area and power budget:** 0.47 mm² and 1.13 W at 8 nm (scaled from 45 nm) for a lossless attention unit at about 300 MHz. This is a concrete, citable estimate for "attention on the HBF base die". It is a scaled estimate, not silicon.
- **"Capacity per device is underused; prefer more bandwidth"** (<600 GB of 4 TB used). This independently echoes the report's finding that H3 over-provisions flash by more than an order of magnitude, and its reallocation of flash to fast memory.
- **Bandwidth-gap argument:** HILOS states plainly that NSP helps only when internal bandwidth exceeds link bandwidth, or when the shared link is the bottleneck. For H3's chained HBF-behind-HBM topology the relevant question is whether the chain link, or the HBF PHY, is narrower than the HBF die array bandwidth. That is the precondition for attention-near-HBF to pay off.

What HILOS does not address:
- It relies on the SmartSSD FTL. There is no pSLC, no wear levelling and no retention control.
- Endurance is estimated from rated PBW, not measured.
- Offline only; 64-token outputs by default.
- No online latency SLOs.

### 3.8 BibTeX
```bibtex
@inproceedings{jang2026hilos,
  title     = {A Cost-Effective Near-Storage Processing Solution for Offline Inference of Long-Context {LLMs}},
  author    = {Jang, Hongsun and Song, Jaeyong and Shin, Changmin and Noh, Si Ung and Jung, Jaewon and Park, Jisung and Lee, Jinho},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26)},
  address   = {Pittsburgh, PA, USA},
  publisher = {ACM},
  year      = {2026},
  numpages  = {21},
  isbn      = {979-8-4007-2359-9},
  doi       = {10.1145/3779212.3790119},
  eprint    = {2502.09921},
  archivePrefix = {arXiv}
}
```

---

## 4. Smart-Infinity (HPCA 2024)

### 4.1 Bibliographic
- **Title:** "Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System"
- **Authors:** Hongsun Jang, Jaeyong Song, Jaewon Jung (Seoul National University); Jaeyoung Park (University of Texas at Austin, work done at Yonsei University); Youngsok Kim (Yonsei University); Jinho Lee (Seoul National University; corresponding author).
- **Venue:** 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA). Stated in the header of the arXiv text.
- **arXiv:** 2403.06664v1 [cs.AR], 11 Mar 2024. **Pages / DOI:** unknown. The extraction is the arXiv version, which carries no IEEE DOI or pages; check IEEE Xplore before submitting.
- **Code:** https://github.com/AIS-SNU/smart-infinity
- **Funding:** Samsung Electronics; environment from Samsung Memory Research Center.

### 4.2 Problem
- Storage-offloaded **training** (DeepSpeed ZeRO-Infinity) keeps FP32 optimizer states (6M: param + momentum + variance, where M is the FP16 parameter size) and gradients (2M) on NVMe.
- More than 88% of training time is storage transfer (§I). The update phase is over 80% of time (Fig. 3a).
- RAID0 scaling saturates after 4 SSDs because of shared host PCIe (Fig. 3b).
- Optimizer states "consume 75% of the total storage bandwidth".

### 4.3 Core mechanism
- **What is offloaded:** optimizer states and gradients sit in the CSDs. The **parameter update (Adam etc.) runs on the FPGA inside each SmartSSD**, fed over the SSD↔FPGA internal P2P path.
- **SmartUpdate traffic (Table I):** system-interconnect traffic falls from 6M read + 6M write (optimizer) plus 2M + 2M (gradients) under ZeRO-Infinity, to 2M read (updated params upstream) + 2M write (gradients).
- **Internal data transfer handler:**
  - Device buffers are preallocated once for the largest subgroup.
  - Two host threads with **swap overlapping**. Updated params are written back first because they are urgent for the next forward pass. Momentum and variance writeback is deferred at lower priority.
  - This overlaps SSD↔FPGA transfers without OOM on the 4 GB FPGA DRAM.
- **SmartComp:**
  - The GPU compresses gradients (Top-K magnitude; default 2% = 1% values + indices). The FPGA decompresses them (scatter only, cheap).
  - Shared-link write traffic becomes c% × 2M.
  - It is lossy but gives comparable fine-tuning accuracy.
  - Motivation: writes are slower than reads, and the gradient offload cannot be overlapped with the update because of NaN/Inf loss-scaling checks and global-norm clipping.
- **Workload distribution:** parameters are flattened and split equally across CSDs; each CSD owns and updates its shard. This is architecture-agnostic.
- **Accelerator:** updater PEs built from SIMD AXPBY units (16 per PE), generic to Adam, SGD-momentum and AdaGrad, plus a Top-K decompressor.
- **Wear / endurance: not discussed anywhere in the paper** (confirmed by search: no endurance, P/E, TBW, WAF or lifetime).

### 4.4 Device / system parameters
| Item | Value | Source |
|---|---|---|
| CSD | Up to 10× Samsung SmartSSD, 4 TB NVMe, connected to a Kintex UltraScale+ KU15P over PCIe Gen3 x4. About 522K LUTs, 984 BRAMs, 1,968 DSPs, 4 GB DDR4 | §VII-A |
| Host | Xeon Gold 6342 (2 × 48C/96T), 32 × 32 GB DDR4-3200, H3 Falcon 4109 PCIe expansion | Table II |
| GPUs | RTX A5000 24 GB (default), A100 40 GB, RTX A4000 16 GB | §VII-A |
| Software | Ubuntu 20.04, Python 3.9, PyTorch 1.12.1, CUDA 11.6.2, OpenCL 2.2, Vitis/XRT 2023.1 / 2.12.427, DeepSpeed 0.9.3 | Table II |
| Utilisation | Adam: LUT 33.66%, BRAM 27.13%, URAM 34.38%, DSP 11.03%. Adam + Top-K: LUT 34.12%, URAM 35.94% | Table III |
| Updater throughput | above 7 GB/s, higher than SSD read/write. Decompressor slightly above SSD read | §VII-H |
| Fig. 2 labels | Host memory "(16 GB/s)" behind a shared interconnect; internal "(8·#CSDs GB/s)" | Fig. 2 |
| Costs | Server about $45,000; GPU about $2,000 (A5000) / $7,000 (A100); SmartSSD about $2,400 vs $400 for same-capacity SSD (6×) | §VII-I |

### 4.5 Evaluation and headline results
- **Baseline:** ZeRO-Infinity with software RAID0 (BASE). **Models:** GPT-2 (4.0B, 8.3/8.4B, up to 33.0B), BERT, BLOOM, ViT. **Settings:** default bs=4, default 2% compression.
- **GPT-2 8.4B, 6 SSDs:** the baseline's update phase is 75.57% of time. The baseline stays flat from 6 to 10 SSDs.
- **Speedups:**
  - SU: 1.18–1.24× (6 SSDs), 1.54–1.60× (10 SSDs).
  - SU+O: 1.60–1.66× (10 SSDs).
  - SU+O+C: a further 1.22–1.31×, for **1.85–1.98×** over the baseline.
  - Best case: **up to 2.11×** on A100 with 10 CSDs.
- **Large models:** GPT-2 33.0B reaches 1.37× (6 SSDs) and 1.88× (10 SSDs).
- **Scaling:** Smart-Infinity scales nearly linearly with the number of CSDs, while the baseline stops scaling beyond 4 SSDs. With 1 CSD there is a slight slowdown, because internal and external bandwidth are equal.
- **Other optimizers / models:** SGD and AdaGrad have 3/4 of Adam's offload volume, so speedups are slightly lower. BLOOM and ViT: 1.32–1.85×.
- **Cost:** with 1–3 CSDs, GFLOPS/$ is below the baseline. Above 4 it is better, and it keeps rising with more CSDs.
- **Fine-tuning (GLUE, 6 SSDs):**
  - SU+O is bit-identical to the baseline and gives 1.10× (BERT-0.34B), 1.11× (GPT2-0.77B), 1.29× (GPT2-1.6B).
  - SU+O+C at 1–10% gives up to 1.40× / 1.44× / 1.54× with small accuracy changes. Example: GPT2-0.77B MNLI goes from 85.95/86.60 to 84.57/85.39 at 1%.
- **Congested multi-GPU topology** (1–3 A4000 in the same chassis, TP): 1.66–1.86× with ten CSDs.

### 4.6 Stated limitations and future work
- No bandwidth gain with a single CSD, plus a slight slowdown.
- The remaining bottleneck is the upstream model transfer from CSD to host, and the gradient downstream path.
- Performance depends on PCIe topology (congested scenario).
- **Future work:**
  - Use the CSD for model compression during fine-tuning (quantization with STE, pruning, low-rank), sending compressed models upstream. Open issues: per-layer quantization intervals, and an efficient GPU decompression kernel.
  - Fit into storage expansion and pooling, where "more capacity means more internal bandwidth".

### 4.7 Critical view (for HBM + HBF)
- **Least directly relevant.** It is about training, and H3/HBF targets inference with read-only data in flash. It is useful mainly as a **counter-example for endurance**. Smart-Infinity rewrites 6M bytes of optimizer state per iteration to flash and never analyses wear. Any proposal to use HBF for fine-tuning or optimizer state would hit this wall, which the report's endurance framework (hot set, wear concentration) would expose immediately.
- **Transferable principles:**
  1. Place element-wise, low-intensity operators next to the capacity tier, and shard their state so that aggregate internal bandwidth grows with device count while the shared link stays fixed. Per-stack HBF logic would do the same.
  2. **Compress on the producer, decompress near memory, to relieve a shared write link.** Applied to HBF this means compressing spilled KV (e.g., quantising) on the GPU and decompressing on the HBF base die. It cuts both link traffic and **flash write volume**, and therefore wear.
  3. **Urgency-ordered writeback:** write back what the next phase needs first and defer the rest. This is analogous to deferring non-urgent KV spill writes while prioritising reads.
- It does not address inference, KV, flash wear, or on-package integration.

### 4.8 BibTeX
```bibtex
@inproceedings{jang2024smartinfinity,
  title     = {{Smart-Infinity}: Fast Large Language Model Training using Near-Storage Processing on a Real System},
  author    = {Jang, Hongsun and Song, Jaeyong and Jung, Jaewon and Park, Jaeyoung and Kim, Youngsok and Lee, Jinho},
  booktitle = {2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  year      = {2024},
  eprint    = {2403.06664},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note      = {Pages/DOI not in the arXiv text; verify on IEEE Xplore}
}
```

---

## 5. Cross-paper observations and gaps relevant to HBM+HBF

### 5.1 Shared picture
- All four papers argue that **the problem is capacity: the alternative is buying more GPUs or DRAM**.
  - HILOS §1: "equipping costly servers with multiple GPUs ... cost-prohibitive". HILOS also beats an 8-GPU vLLM deployment by 1.64–1.81×.
  - HiFC: 4.5–6.1× lower memory-expansion cost vs DRAM.
  - InstAttention: KV "potentially exceeding even the costs associated with storing the model weights".
  - Smart-Infinity: "dozens of GPUs are required, just to keep the training data on the memory".
- This is the report's "capacity, not arithmetic" thesis, supported by four real-system papers from other groups.
- **All four sit behind PCIe, at 3–8 GB/s per device.** HBF sits on-package at near-HBM bandwidth. None of their bandwidth arguments carry over numerically, but their *write-pattern* and *endurance* arguments do.
- **Write patterns converge:** all three inference papers make KV writes append-only and page/block-aligned.
  - HiFC: append allocator, sequential LBAs.
  - InstAttention: DRAM group buffer, packing heads into a block.
  - HILOS: host buffer, spill every 16 tokens.
  - This is independent support for the report's DRAM write-combining buffer and for "block allocation, not the collector, decides collection cost".
- **The read:write asymmetry is extreme.** InstAttention measured decode reads 560–810× larger than writes (Table III). HILOS calls KV "write-once, read-many". This supports H3's premise that flash is adequate when reads dominate.

### 5.2 Ideas not yet applied to HBF (with source)
1. **Application-directed bulk erase of dead KV, with no FTL GC** (InstAttention: `reclaim()`, erase in LRU order when idle, semantic logical addresses task/layer/token/head). For HBF, spill regions could be allocated per request in whole erase blocks and erased when the request completes. This formalises the report's claim that GC is structurally absent, and gives the controller a concrete interface.
2. **Retention relaxation as an *endurance* lever, with a read-retry cost** (InstAttention: 3 years → 3 weeks, about 6.67× P/E, at most 4.7‰ slowdown from read-retry). The report treats relaxed retention only as a source of latency (H3) and of refresh writes. It could add the P/E gain to the spill-region lifetime calculation, and the read-retry penalty to the latency model. The KV spill lifetime is seconds to minutes, far shorter than 3 weeks.
3. **Split-tier attention with softmax-statistic merging** (HILOS: host partial QKᵀ over buffered tokens, masked two-pass softmax in the accelerator). The GPU computes attention over the HBM-resident tail and HBF-side logic computes over the flashed body. This makes the report's write-combining buffer free on the critical path, and bounds HBM usage per spilled request.
4. **Attention on the HBF base die, with a concrete area and power estimate** (HILOS §7.1: 0.47 mm², 1.13 W at 8 nm, 300 MHz, lossless; InstAttention: FPGA SparF engine). This is worthwhile only if HBF array bandwidth exceeds the HBF↔GPU (or chained HBF↔HBM) link bandwidth. HILOS's Eq. 3 ((s+1)/2 traffic reduction) gives the traffic model to put in the simulator.
5. **Balanced device provisioning** (HILOS §7.2: used <600 GB of 4 TB; asks for "less capacity, more internal bandwidth / compute"). This independently corroborates the report's over-provisioning result and its "reallocate flash to fast memory" recommendation. Cite it.
6. **Coarse-grained block-level mapping instead of page-level FTL** (HILOS §7.2; InstAttention's two custom L2P tables). The sequential, append-only KV and read-only weights on HBF make a block-mapped, DRAM-light controller feasible. This matters for on-package controller area.
7. **pSLC / SLC-only operation of a spill region** (HiFC). HBF has capacity to spare, so a statically configured SLC spill partition sized to the hot set (e.g., the report's 272 GB hot set) is cheap. The report's wear-concentration analysis should then use SLC P/E for the hot region only. HiFC's own data shows the danger of a *dynamic* SLC cache: throughput collapses when the working set exceeds it (4.7 → 1.69 GiB/s).
8. **Producer-side compression of spilled KV** (Smart-Infinity SmartComp, GPU compresses and FPGA decompresses; lossy KV sparsity in InstAttention). Quantising KV before spilling to HBF cuts write volume and wear proportionally. InstAttention's accuracy loss on long-context models (HILOS: −3.5 to −5.7 %p) is a warning to prefer lossless or near-lossless formats.
9. **X-cache (store activations and recompute K, V)** (HILOS). This lowers flash writes by about α/2 **for MHA only**. It should be ruled out for GQA models in the report's nine-model set: for Qwen2.5-32B, K+V = 2,048 elements per token vs X = 5,120 [my derivation from HILOS Table 2]. A quick per-model check in the simulator would show where it helps.
10. **Pipeline-slack visibility model for spill latency** (HiFC App. G: swap is hidden iff T_swap ≤ ΔTPS·T_lat). This is a simple analytic check for when HBF spill/restore latency becomes visible under the report's over-subscription policy.

### 5.3 Gaps none of the four address
- **Wear levelling between cold, read-only data (weights, shared cache) and a hot KV spill region on the same flash.** None of the four papers mix the two. HiFC and InstAttention dedicate the device to KV. HILOS puts >100B weights on storage but uses the FTL's own wear levelling. The report's 10× lifetime effect from static wear levelling is novel relative to this literature.
- **Physical NAND WAF measurement.** HiFC's "1.02" is host-level, not NAND-level (see §1.7). InstAttention and HILOS assume or infer WA from rated TBW. No paper measures NAND-level WA for KV traffic. The report's simulator-based WA curves fill this gap, but cannot be validated against these papers.
- **Endurance claims are all rated-TBW arithmetic** (HiFC 8.3 years, InstAttention 920–4,600 users, HILOS >4.08 M requests). None uses a measured wear distribution. HiFC and InstAttention implicitly assume perfect wear levelling; the report shows that assumption is worth about 10×.
- **Online, latency-SLO serving with an on-package flash tier.** InstAttention and HILOS are offline only. HiFC is online but PCIe-bound.
- **GQA/MLA-era models.** InstAttention tests only MHA. HILOS covers GQA only in its accelerator, not in X-cache sizing.
- **Read disturb** from extreme read:write ratios (560–810×) on flash storing hot shared-cache or KV data. No paper analyses read-disturb-induced refresh writes, which could be a hidden write source for H3's "read-only" HBF residency.
