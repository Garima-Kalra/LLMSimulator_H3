# E2: Memory-system and modelling papers (RoMe, InfiniGen, ZnG, HAVEN, LIFE)

Context: BTP report (`/Users/garimakalra/Desktop/BTP/BTP_Report.tex`). It evaluates H^3 (HBM + High-Bandwidth Flash, with read-only weights and a shared KV cache placed in HBF). It does this by extending **LLMSimulator (SNU SCALE lab)**, a per-operation roofline model with a single HBM tier. The report's findings: flash is over-provisioned by 17–67x; moving that budget to HBM gives 1.75x concurrent requests at 21% lower power; garbage collection is structurally absent; wear concentration sets device lifetime.

Sources: the plain-text extractions in `scratchpad/txt/`, read in full (body, tables, figure captions, discussion, conclusion and appendix; reference lists skimmed). Numbers below are quoted from the papers. Anything I derived is marked **(my derivation)**. Where a paper's text disagrees with itself, I say so and do not pick a value.

---

## 1. RoMe: Row Granularity Access Memory System for Large Language Models

### 1.1 Bibliographic
- **Title:** "RoMe: Row Granularity Access Memory System for Large Language Models"
- **Authors:** Hwayong Nam (Seoul National University), Seungmin Baek (SNU), Jumin Kim (SNU), Michael Jaemin Kim (Meta; the work was done while at SNU), Jung Ho Ahn (SNU, corresponding author). Nam and Baek contributed equally.
- **Venue/year:** arXiv:2512.01541v1 [cs.AR], 1 Dec 2025. The extracted text **does not name a venue**. "HPCA26" appears only in the local filename, so treat HPCA 2026 as unverified until you check the proceedings.
- **Pages/DOI:** unknown (not in text).

### 1.2 Problem
- HBM keeps cache-line granularity (32 B for HBM4) across generations. To raise bandwidth while holding per-bank access granularity, it added **bank groups** and **pseudo channels (PCs)**. These multiply timing parameters and memory-controller (MC) scheduling complexity.
- Per-bank bandwidth has barely moved. Bandwidth can grow only by (1) larger access granularity or (2) higher DRAM core frequency. Cache-line size caps (1), and physics caps (2).
- The C/A (command/address) pin overhead grows every generation. In HBM4, each 64-bit channel needs 10 row C/A pins plus 8 column C/A pins, and the C/A-to-DQ ratio roughly doubled from HBM1/2 to HBM3/4 (Fig. 2b).
- LLM inference streams contiguous KB-to-MB blocks (weights, KV, activations; Fig. 1). A conventional HBM system chops these into hundreds of 32 B transactions and needs elaborate scheduling to do so.

### 1.3 Core mechanism
1. **Row-level interface.** The column interface is replaced by two commands, `RD_row` and `WR_row`, plus REF. The MC access granularity (AG_MC) grows from 32 B to one row (4 KB effective).
2. **Virtual bank (VBA).** Bank groups and PCs disappear from the MC–DRAM interface. The paper explores three bank options (Fig. 7b/c/d) and two PC options (Fig. 8a/b).
   - All six combinations stay within **3.6%** of baseline performance.
   - Doubling the internal datalines (7b combined with 8a) costs up to **77%** area.
   - Chosen design: **Fig. 7(d) + Fig. 8(b)**. Two banks from different bank groups are accessed time-multiplexed, and both PCs run in lock-step as in HBM1/2 legacy mode. This needs no change to DRAM internals, and the effective row doubles.
3. **Command generator on the HBM4 logic die** (logic process). It expands each row command into a fixed sequence: ACT, then a series of RD/WR, then PRE.
   - It inserts a delay of tRRDS − tCCDS before the first bank's ACT so that the RD/WRs to the two banks interleave at tCCDS.
   - The paper weighed three placements (MC, logic die, DRAM die) and chose the logic die. Placing it in the MC gives no pin savings; placing it in the DRAM die needs one generator per channel per die.
4. **C/A pin reduction.** Column C/A pins go away and MRS moves to the row pins. The interface has 11 commands in total. The tightest spacing, REF right after `RD_row`/`WR_row`, is 2×tRRDS, and 5 pins still meet it.
   - Result: C/A pins drop from 18 to **5** per channel, **−72%**.
5. **Extra channels from freed pins.**
   - An HBM4 channel needs 120 pins; RoMe needs 107.
   - Across 32 channels that frees 416 pins, enough for 4 new channels with only 12 extra pins.
   - Channels per DRAM die go from 8 to 9, so the stack has 36 channels and **+12.5% bandwidth** (2.25 TB/s vs 2 TB/s per cube).
6. **Simplified MC** (Table IV):
   - Timing parameters: 15 → 10.
   - Bank states: 7 → 4 (Idle, Reading, Writing, Refreshing).
   - Bank FSMs: 5 (two active VBAs plus up to three in refresh).
   - No page policy, because the MC always precharges after a row.
   - Scheduling is oldest-first with VBA interleaving.
   - Request queue: HBM4 needs ≥45 entries for peak throughput; RoMe needs **2**.
7. **Refresh.** One per-bank refresh (REFpb) is issued every 2×tREFIpb, and the generator sends two REFpbs spaced tRREFD apart. Stall per VBA drops from 2×tRFCpb (2×280 ns) to tRFCpb + tRREFD (280 + 8 ns).
8. **Writes.** Writes are issued immediately rather than buffered, to avoid a big 4 KB-chunk write queue. The paper justifies this because LLM workloads are read-dominated.

### 1.4 Key parameters (Table V unless stated)
| | HBM4 baseline | RoMe |
|---|---|---|
| channels/cube (PCs) | 32 (64) | 36 |
| stacks (SIDs) | 4 | 4 |
| banks/channel | 128 | 32 |
| row size | 1 KB | 4 KB |
| data rate | 8 Gb/s | 8 Gb/s |
| bandwidth/cube | 2 TB/s | 2.25 TB/s |
| AG_MC | 32 B | 4 KB |
| timings (ns) | tRC=45, tRP=16, tRAS=29, tCL=16, tRCDRD=tRCDWR=16, tWR=16, tFAW=12, tCCDL=2, tCCDS=1, tCCDR=2, tRRD=2 | tR2RS/R=64/68, tR2WS/R=69/73, tW2RS/R=71/75, tW2WS/R=64/68, tRD_row=95, tWR_row=115 |

- **HBM4 timing source:** JEDEC had not finalised HBM4 timings, so the paper takes values from prior work: [2] Folded Banks (ISCA'25) and [51] Fine-Grained DRAM (MICRO'17).
- **Internal buses:** each bank fetches 256 bits. BK-BUS runs at 1/tCCDL (e.g., 0.5 GHz) and BG-BUS at 1/tCCDS (e.g., 1 GHz), so one bank group can use only half the bandwidth.
- **HBM4 organisation:** up to 32 channels; one SID per 4 DRAM dies, up to 4 SIDs.
- **System modelled:**
  - Accelerator target is **280 Op/B**; the paper cites 281 Op/B for B200.
  - Memory: 8 HBM4 cubes of 32 GB each (8 Gbps, 16-Hi), giving **256 GB and 16 TB/s**; BF16 throughput "scaled to 4480 TFLOPS".
  - Deployment: 8 accelerators in parallel, "each providing **560 TFLOPS** of BF16, 256 GB, 16 TB/s".
  - **Inconsistency:** 4480 TFLOPS / 16 TB/s = 280 Op/B, while 560 TFLOPS / 16 TB/s = 35 Op/B. The two statements conflict, and it is unclear which one the simulations used.

### 1.5 Evaluation
- **Simulator:** **LLMSimulator [77] (Duplex, MICRO'24)** with **Ramulator 2.0** integrated for cycle-accurate DRAM. RoMe is implemented in Ramulator 2.0 with 4 KB requests.
- **Controllers:** both MCs use FR-FCFS; the baseline uses open-page; both use per-bank refresh; address mappings are swept for both.
- **Models:** Grok 1 (GQA + MoE, top-2 of 8 experts), DeepSeek-V3 (MLA + MoE, top-8 of 256), Llama 3-405B (GQA, dense). All in BF16.
- **Parallelism:** prefill uses TP across 8 devices. Decode attention uses TP = 1 / 8 / 8 (DeepSeek-V3 / Grok 1 / Llama 3). MoE uses expert parallelism.
- **Workload:** sequence length 8K. Batch sizes are swept up to the capacity limit (DeepSeek to 1024, Grok to 512, Llama to 256).
- **Results:**
  - **TPOT −10.4% / −10.2% / −9.0%** (DeepSeek-V3 / Grok 1 / Llama 3). This is less than the 12.5% bandwidth gain because FFN layers are not always memory-bound.
  - Prefill differs by less than **0.1%** (compute-bound).
  - Channel load-balance ratio (LBR, Fig. 13) stays between roughly 0.85 and 1.0. MoE FFNs are imbalanced at small batch until all experts are selected, which happens at batch ~64 for DeepSeek-V3 and ~8 for Grok 1.
  - **Energy −1.9% / −0.7% / −0.7%**. ACT energy falls to 55.5% / 86.0% / 84.4% of baseline. The command generator uses 0.06% of energy.
  - **Area:**
    - Command generator: 4268.8 µm² for 36 channels (7 nm ASAP7), 0.003% of the logic die.
    - RoMe scheduling logic (4-entry queue) is **9.1%** of a conventional MC (64-entry queue).
    - Extra channels need 48 µbumps (≈0.14 mm², 22 µm pitch). The paper states the DRAM die grows about 12% including edge margin, the logic die grows proportionally, and the "total area overhead [is] only 0.10%". The link between the 12% and 0.10% figures is not explained in the text.

### 1.6 Stated limitations / future work (Discussion §VII)
- Fine-grained sparse attention (DeepSeek Sparse Attention picks the top-2048 tokens; gpt-oss uses sliding windows) creates irregular access, and row granularity then overfetches. The paper suggests either a hybrid RoMe + conventional HBM4 system (which risks under-utilising the HBM4 part) or column masks (which bring latency variation and complexity).
- RoMe depends on processor co-design. It suits a few large cores with big on-chip buffers and explicit data management (TPU-like).
- Other directions: larger ECC codewords at 4 KB granularity; training; other DRAM types. Non-HBM DRAM has no logic die, so the pin-reuse argument changes; adding data pins is suggested instead.
- Minor overheads from overfetch and load imbalance are said to be "negligible".

### 1.7 Relevance to HBM+HBF
- **Same simulator lineage as the BTP.** RoMe evaluates on LLMSimulator plus Ramulator 2.0, and the report extends LLMSimulator. RoMe is therefore a worked example of plugging a new memory device's timing into that stack. Ramulator models DRAM, not NAND, so an HBF tier would need its own timing model.
- **Coarse granularity costs nothing for dense LLM streams.** RoMe shows 4 KB access loses nothing for weight, KV and activation streams (§III: most weight/KV accesses exceed several hundred KB). This directly supports the HBF assumption that NAND page-sized reads (4 KB in HAVEN, 4 KB in ZnG) suit weights and shared-KV streaming.
- **The same caveat applies to HBF, more strongly.** Sparse or top-k attention (DSA, InfiniGen-style selection) reintroduces fine-grained access. RoMe calls it an overfetch risk for DRAM; for flash, with larger pages and µs latency, it matters more.
- **Logic-die command generator as a template.** A fixed-sequence command generator on the logic die is a natural place for an HBF controller: page reads, plane interleaving, FTL-lite. HAVEN's near-storage unit and ZnG's flash controller also sit at the device edge.
- **Capacity sets batch size.** RoMe's Fig. 12 caption says "maximum batch size is constrained by memory capacity", which is independent support for the BTP's central claim.

### 1.8 Critical view
- The 12.5% bandwidth gain comes almost entirely from **extra channels** (pins and area), not from row access itself. Access granularity mainly buys MC simplicity.
- Only decode TPOT at 8K is reported. There are no long-context (≥128K) or prefix-sharing cases, and no tail-latency or QoS results under mixed traffic.
- The 280 vs 35 Op/B inconsistency (above) makes the memory-boundness of the setup uncertain.
- The HBM4 timings are assumed, not JEDEC values.
- Immediate writes are argued rather than quantified. KV-cache writes during decode are small per step, but the paper does not measure their turnaround cost.
- Energy savings are small (≤1.9%). The 0.10% area figure needs a clearer derivation.
- No capacity tier is considered: the whole study is bandwidth within HBM.

### 1.9 BibTeX
```bibtex
@misc{nam2025rome,
  title         = {{RoMe}: Row Granularity Access Memory System for Large Language Models},
  author        = {Nam, Hwayong and Baek, Seungmin and Kim, Jumin and Kim, Michael Jaemin and Ahn, Jung Ho},
  year          = {2025},
  eprint        = {2512.01541},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv v1, 1 Dec 2025. Venue (HPCA 2026) inferred from local filename only -- verify before citing as conference paper}
}
```

---

## 2. InfiniGen: Efficient Generative Inference of LLMs with Dynamic KV Cache Management

### 2.1 Bibliographic
- **Title:** "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management"
- **Authors:** Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim (Seoul National University). Lee and Lee contributed equally; Sim is the corresponding author.
- **Venue/year:** arXiv:2406.19707v1 [cs.LG], 28 Jun 2024.
  - The text thanks "the anonymous reviewers and our shepherd Petros Maniatis", which indicates a peer-reviewed systems venue, but **the venue is not named in the text**.
  - The filename says OSDI'24. To my knowledge it appeared at USENIX OSDI 2024, pp. 155–172, but **I have not verified this against the extracted text**. Check the USENIX page before using the pages.
- **DOI:** unknown.

### 2.2 Problem
- The KV cache grows linearly with sequence length and batch size and can exceed the model weights. Fig. 2 shows this for OPT-30B: the KV cache is bigger than the weights at long sequences with batch 16, or at large batches with seq 2048.
- Offloading systems (FlexGen, DeepSpeed) keep the KV cache in CPU memory, and **PCIe transfer** becomes the bottleneck. Plain prefetching hides only part of it (Fig. 3). Quantization does not fix the linear scaling.
- Eviction methods such as H2O assume attention patterns persist, which fails in three ways:
  - **C1:** attention is dynamic across iterations. Tokens unimportant now can matter later (Fig. 4; Fig. 20b shows spikes after thousands of iterations).
  - **C2:** the number of KV entries needed varies by layer. Layer 0 is broad and Layer 18 is skewed (Fig. 5).
  - **C3:** it varies by query. With a 0.9 cumulative-attention target, the 500th/1000th/1500th/2000th tokens need 80/146/160/164 keys, and adjacent tokens 998–1002 need 172/164/146/154/140.

### 2.3 Core mechanism
1. **Attention inputs are similar across layers.** Because of outlier channels and LayerNorm, Tblock_in_i is dominated by Tblock_in_{i−1}. Table 1 cosine similarities with Tblock_in_{i−1} are 0.95 / 0.96 / 0.97 / 0.89 / 0.91 (OPT-6.7B / 13B / 30B, Llama-2-7B / 13B). Similarities with Attn_out and FFN_out are only about 0.27–0.37.
2. **Offline skewing.** One forward pass on a sample input; SVD of each layer's query matrix Q = UΣVᵀ; set A = V; multiply both W_Q and W_K by A.
   - This is mathematically exact: Q̃K̃ᵀ = QKᵀ (Eq. 2). It concentrates magnitude in a few columns and has no runtime cost.
3. **Prefill.** Sum the absolute values of skewed Q and K per column and take the top-k columns (**30%**). The results are a **partial query weight** and a **partial key cache** for each layer.
4. **Decode speculation.** At layer i−1, InfiniGen multiplies layer i−1's attention input by layer i's partial W_Q and partial K cache to get speculated scores.
   - It selects tokens with score > max − α. With α = 5, an excluded token has weight ≤ max/e⁵ ≈ max/148.4 after softmax.
   - The number of tokens is averaged across heads, so every head fetches the same count.
   - It then **prefetches only those K/V entries from CPU** for layer i, overlapping with layer i−1 compute. Speculation starts at Layer 1, because outliers emerge in Layer 0.
5. **KV pool manager (CPU).** An optional user-defined memory limit. Eviction is **counter-based**: prefetched entries are counted, the smallest count is evicted, and all counters halve on saturation.
   - It matches LRU accuracy and avoids LRU's locked doubly-linked list. FIFO loses badly (Table 2).
   - The victim slot is overwritten with the new token's K/V, and the partial key cache on the GPU is updated.

### 2.4 Key parameters
- **Hardware:** NVIDIA RTX A6000 (48 GB); Intel Xeon Gold 6136; 96 GB DDR4-2666; **PCIe 3.0 ×16** (no bandwidth number is given).
- **Settings:** partial-weight ratio 0.3; α = 4 (OPT) and 5 (Llama-2). This uses **<10% of the KV cache on average** across layers, with a per-layer cap of 20% of the total KV.
- **Speculation overhead:** partial W_Q is 2.5% of total parameters; partial K cache is 15% of the total KV cache (at ratio 0.3). Both are kept on the GPU in the experiments.
- **Models:** OPT-6.7B / 13B / 30B; Llama-2-7B / 13B; Llama-2-7B-32K; Llama-3-8B-1048K (analysis only).

### 2.5 Evaluation and results
- **Baselines:** CUDA UVM, UVM + H2O, FlexGen (all KV on CPU), FlexGen + INT4 (group-wise asymmetric), FlexGen + H2O (20% budget).
- **Tasks:** lm-eval-harness 5-shot (COPA, OpenBookQA, WinoGrande, PIQA, RTE); WikiText-2 and PTB perplexity; PG-19 for long sequences.
- **Accuracy:**
  - Up to **+32.6 percentage points** over prior methods. At relative KV size <10%, InfiniGen is consistently better; above 10% it matches full cache and sometimes beats it slightly.
  - Perplexity stays at full-cache level over 2048 tokens (OPT-13B) and 4096 tokens (Llama-2-13B), while H2O drifts upward.
  - The skewing ablation (OPT-6.7B, fixed 20% budget) shows a large accuracy drop without skewing, and only a small one for Llama-2.
- **Latency on OPT-13B (1920 input + 128 output):**
  - At batch 20, **1.63×–32.93×** faster than the baselines. Across batch sizes 4–20, **1.28×–34.64×**.
  - Throughput from batch 4 to 20: InfiniGen 27.36 → 41.99 tok/s; INT4 12.22 → 14.02; H2O 21.31 → 25.70.
- **Sequence length (batch 8, 512–2048 total tokens):**
  - Speedup over FlexGen reaches **5.28×**, while INT4 saturates at 1.92× and H2O at 3.40×.
  - Important tokens (score > max − 4): 37 / 60 / 66 / 73 at 512 / 1024 / 1536 / 2048. H2O loads 409 tokens at 2048.
- **Model size (batch 4):** going from 6.7B to 13B raises InfiniGen's speedup by 1.17×. On 30B, with 30% of parameters offloaded (1.7× the KV size), InfiniGen is 1.34× vs 1.18× (INT4) and 1.28× (H2O).
- **Breakdown (OPT-13B, seq 2048, batch 8):**
  - Data transfer is **96.9%** of FlexGen's block time and **91.8%** of H2O's.
  - InfiniGen is **1.52×** slower than Ideal (all on GPU); the others are 3.90×–18.55× slower.
- **Headline in the abstract:** "up to 3.00×" over prior KV-management methods.

### 2.6 Stated limitations / future work
- There is no explicit limitations section.
- The memory overhead of partial weights and keys can be reduced by storing only column indices, or by keeping the partial K on the CPU and speculating there.
- Efficient fetching assumes offloaded KV at token granularity over PCIe.
- Benefits are expected to grow with million-token contexts (Fig. 20).
- Kernel fusion is mentioned as a compatible direction.

### 2.7 Relevance to HBM+HBF
- **A lookahead window for hiding latency.** InfiniGen gives exactly one layer of lookahead: layer i's KV set is known during layer i−1. For HBF, the fetch for layer i must finish within one layer's compute and HBM time. With HBF's µs-scale latency (ZnG's Z-NAND read is 3 µs; HAVEN gives no number), this is generous compared with PCIe, but the needed bandwidth is set by the selected-token volume.
- **Cutting HBF bandwidth for a shared KV cache in flash.** If the H^3-style shared cache (the report's 540 GB at 1M tokens) sits in HBF, full-scan decode reads all of it every step. InfiniGen-style selection (<10% of KV) would cut HBF read traffic by roughly that factor. This is **the main lever for making flash bandwidth sufficient**.
- **Granularity.** InfiniGen equalises only the **number** of tokens per head ("averaging the number of tokens … across the heads"). The paper does not say that heads share the same token set, so selection may be per head.
  - Per head, one token's K (or V) is d × 2 B, i.e. 256 B for d = 128 in FP16 **(my derivation)**. Fetched from 4 KB flash pages, that would badly overfetch unless tokens are clustered. This is RoMe's DSA caveat and ZnG's 128 B-vs-4 KB problem again.
  - If heads were forced to share one token set, the per-layer chunk would be 2 × D × 2 B, e.g. 20 KB for OPT-13B with D = 5120 **(my derivation)**, which is page-friendly. This head-sharing constraint is a design choice to evaluate for HBF.
- **Writes.** The pool manager overwrites victims in place. On flash that means program/erase churn, so newly generated KV should stay in HBM (the H^3 read-only placement) or be written append-only.
- **Metadata placement.** The partial key cache (15% of KV) must itself live somewhere fast. For a 540 GB shared cache that is about 81 GB **(my derivation: 15% × 540 GB)**. That is not negligible HBM usage and eats into the batch numerator in the BTP's equation.

### 2.8 Critical view
- The evaluation uses small models (≤30B), short sequences (≤2048 for latency; the 32K results are perplexity-only), and one consumer-class GPU on PCIe 3.0. Speedups relative to FlexGen shrink as the interconnect improves (NVLink-C2C, HBF on-package).
- Latency results are OPT-13B only, and batch ≤20.
- GQA/MQA models are barely covered (Llama-2-7B/13B are MHA), and neither is MLA. Speculation accuracy for GQA group sharing is not analysed.
- Speculation reads the partial K for **all** tokens every step, which is 30% of the key columns. For very long contexts this becomes a large read stream of its own, and it lives on the GPU.
- The prefill stage still needs the full KV computed and written out. Prefill cost and TTFT with offloading are not separated.
- The paper does not model memory-device latency (it treats transfer as bandwidth-bound over PCIe) or random-access efficiency for gathering scattered tokens from host DRAM.

### 2.9 BibTeX
```bibtex
@misc{lee2024infinigen,
  title         = {{InfiniGen}: Efficient Generative Inference of Large Language Models with Dynamic {KV} Cache Management},
  author        = {Lee, Wonbeom and Lee, Jungi and Seo, Junghwan and Sim, Jaewoong},
  year          = {2024},
  eprint        = {2406.19707},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
% If citing the conference version (venue from filename; pages from my recollection -- VERIFY):
@inproceedings{lee2024infinigen_osdi,
  title     = {{InfiniGen}: Efficient Generative Inference of Large Language Models with Dynamic {KV} Cache Management},
  author    = {Lee, Wonbeom and Lee, Jungi and Seo, Junghwan and Sim, Jaewoong},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  pages     = {155--172},
  year      = {2024}
}
```

---

## 3. ZnG: Architecting GPU Multi-Processors with New Flash for Scalable Data Analysis

### 3.1 Bibliographic
- **Title:** "ZnG: Architecting GPU Multi-Processors with New Flash for Scalable Data Analysis"
- **Authors:** Jie Zhang and Myoungsoo Jung (Computer Architecture and Memory Systems Laboratory, KAIST).
- **Venue/year:** arXiv:2006.08975v1 [cs.AR], 16 Jun 2020. The **venue is not named in the text**. The filename says ISCA'20.
- **Pages/DOI:** unknown from the text. The ISCA 2020 proceedings DOI is likely in the 10.1109/ISCA45697.2020.* series, but verify it; I am not supplying a number.
- **Note:** I found no duplicated content in the extraction. The body runs from the Introduction to the Conclusion (§VII), followed by references.

### 3.2 Problem
- GPU memory capacity is small. Using an NVMe SSD as swap requires host-mediated page faults with double copies over PCIe.
- **HybridGPU** (prior work [11]) replaced GDDR with Z-NAND behind an SSD controller and a small DRAM buffer, but lags far behind GPU DRAM:
  - HybridGPU's internal DRAM buffer bandwidth is **96% lower** than the GPU memory subsystem (one package on a 32-bit bus vs 6 MCs on a 384-bit bus).
  - The flash channels and SSD controller are further bottlenecks.
  - GPU DRAM outperforms GPU-SSD by **80×** and HybridGPU by **40×** (Fig. 4c).
  - The SSD engine (FTL) is **67%** of access latency (Fig. 4d), because the SSD controller has only 2–5 embedded cores.
- **Granularity mismatch.** GPU requests are 128 B, but the minimum Z-NAND access is a 4 KB page. Direct Z-NAND access (with an idealised controller) slows workloads by up to **28×**, and **97%** of flash bandwidth is wasted.
- **Workload characteristics:**
  - Each Z-NAND page is re-read **42×** on average, so buffering pays off.
  - Each page receives **65 writes** on average ("write redundancy"), which threatens lifetime.

### 3.3 Core mechanism
1. **Replace all GPU DRAM with Z-NAND packages.**
   - The request dispatcher, SSD controller and DRAM buffer are removed.
   - Flash controllers attach directly to the GPU interconnect, and each has its own request dispatcher.
   - The flash **bus becomes a mesh network** with wider, faster links.
   - Z-NAND is not attached directly to the GPU network because (a) ONFI electrical and frequency differences and (b) the GPU network would be under-utilised.
2. **Zero-overhead FTL, split in two:**
   - **Read-only data block mapping table (DBMT).** Block-level mapping kept in the GPU MMU's two-level page table and cached in the TLB. The entries are VBN, LBN, PDBN and PLBN. The mapping is **80 KB for 1 TB of Z-NAND**, small enough for an MMU internal buffer.
   - **Log page mapping table (LPMT)** inside a **programmable row decoder** of each log block. The decoder acts as a CAM using flash cells: a two-phase search, with "protect" gates to avoid program disturb.
   - Writes go to physical log blocks taken from over-provisioned space. Several data blocks share one log block, tracked in a **log block mapping table (LBMT) in shared memory**. A register tracks the next free page, since programming must be in order.
   - **GC and wear levelling run in a GPU helper thread.** A full log block triggers a merge of the data and log blocks into free blocks chosen by wear levelling, followed by updates to the LBMT and DBMT.
3. **Read path.**
   - The L2 cache is enlarged by building it from **STT-MRAM**: 4× capacity, 6 MB → 24 MB. Its write latency is 5× the SRAM read latency, so it serves as a **read-only** cache.
   - **Dynamic read prefetch**, in three parts:
     - A PC-indexed predictor with 512 entries, tracking 5 warps per entry with 4-bit counters. It prefetches when a counter exceeds 12.
     - Accessed and prefetch bits in each L2 tag.
     - An access monitor that computes a waste ratio (unused ÷ evicted). The prefetch size **halves** when the ratio exceeds 0.3 and **grows by 1 KB** when it falls below 0.05.
4. **Write path.**
   - Writes are buffered in **Z-NAND flash registers**. There are few per plane ("i.e., 8" in §III-C; Table I lists "register 2/8 per plane").
   - Registers are grouped across planes in a package into a **fully-associative write cache**, to handle writes that are skewed across planes (Fig. 8b).
   - Three interconnect options for the registers:
     - **SWnet:** a software copy through the flash-network router, with no hardware change.
     - **FCnet:** fully connected; best performance but too many wires.
     - **Network-in-Flash (NiF):** two buses per plane group (a shared I/O path and a shared data path), plus one "data register" per group that links to other groups over a local network.
   - A **thrashing checker** pins part of the L2 for excess dirty pages ("Redirection").

### 3.4 Key parameters (Table I and text)
- **GPU:** GTX580-like in MacSim; 16 SMs at 1.2 GHz; 80 warps per core.
  - L1: 48 KB, 6-way, 64-set, 1 cycle, private.
  - L2: 6 MB, 1024-set, 8-way, 6 banks, 1 cycle. The L2 was "increased … to match … GV100".
  - STT-MRAM L2: 24 MB, read 1 cycle, write 5 cycles.
- **Z-NAND array:** 16 channels with 1 package each; 8 dies and 8 planes; 1024 blocks and 384 pages; 800 MT/s interface; SLC; 2 I/O ports per package; HW-NiF width 8 B.
- **Flash network:** mesh, 8 B wide, which is "8× higher than traditional flash channel (1B)".
- **Z-NAND device (from [12]):**
  - Read **3 µs**, program **100 µs**, which is 17× and 6× faster than TLC V-NAND.
  - Write latency is **33×** the read latency.
  - P/E endurance **100,000**, vs 3,000–10,000 for V-NAND.
  - 48 stacked layers.
  - **64×** the density of DRAM (text) and the best W/GB (Fig. 3).
- **Optane DC PMM baseline:** tRCD/tCL = 190/8.9 ns, tRP = 763 ns. The paper says its accumulated bandwidth can reach 39 GB/s; HybridGPU's is 5 GB/s.
- **Simulators:** SimpleSSD (modelling an 800 GB ZSSD) plus MacSim.
- **Workloads (Table II):** graph workloads (betw, bfs1–6, gc1–2, sssp3, deg, pr) with read ratios of 0.88–1.0, and scientific workloads (back 0.57, gaus 0.66, FDT 0.73, gram 0.75). They are co-run in read-intensive + write-intensive pairs.

### 3.5 Evaluation results
- **Platforms compared:** Hetero, HybridGPU, Optane, ZnG-base, ZnG-rdopt, ZnG-wropt, ZnG.
- **Headline:** **7.5×** higher performance than HybridGPU (abstract and conclusion; "on average" in the conclusion).
- **IPC:**
  - HybridGPU is +31% over Hetero on average on four workloads.
  - Optane is +186% over HybridGPU.
  - ZnG-base and ZnG-rdopt are **worse than HybridGPU**: unbuffered granularity waste plus blocking on writes.
  - ZnG-wropt is 2.6× over ZnG-rdopt.
  - ZnG reaches 1.9× the bandwidth of Optane.
- **Flash-array bandwidth:**
  - HybridGPU averages 4.2 GB/s.
  - ZnG-rdopt is 2.9× HybridGPU.
  - ZnG-wropt is +137% over ZnG-rdopt.
  - ZnG is +167% over ZnG-wropt.
- **Read re-accesses:** STT-MRAM cuts them by 55%; dynamic prefetch cuts a further 87%; Redirection adds 11% back.
- **Write redundancy:**
  - The baseline averages 51 (with 8 registers).
  - NiF grouping cuts it by 46%.
  - Adding L2 redirection brings it to **1.2**.
- **Register interconnect:** HW-FCnet is +19% over SWnet; NiF achieves 98% of FCnet's performance at lower cost.
- **Prefetch:**
  - Predictor accuracy is 93% on average, 87% in the worst case.
  - 1 KB and 4 KB fixed prefetch are +22% and +32% over no prefetch.
  - dyn-pref is up to +21% over predict-4KBpref.
- **Multi-app:** close to Ideal (a GPU with huge DRAM) at 4 co-running apps. At 8 apps the text says "15% and 6% of the performance improvement, compared to Ideal", which is ambiguous wording.
- **GC:** it blocks the app being collected (back −73%) but not others (betw +5%, from freed L2 lines).

### 3.6 Stated limitations / future work
- The design is for **read-intensive** data analysis. The FTL split relies on few writes.
- GC stalls the owning application heavily.
- Write-intensive workloads (e.g., gaus) thrash the registers.
- Lifetime: register merging plus Z-NAND's high P/E budget keep it acceptable, and wear-levelling policies in the helper thread could extend it further.
- Optane lasts longer but has lower density and lower accumulated bandwidth.
- There is no explicit future-work list.

### 3.7 Relevance to HBM+HBF
- **This is the closest architectural precedent for "flash as GPU main memory".** Lessons:
  1. **Put a buffer between the SMs and the flash, sized to the page.** ZnG uses an STT-MRAM L2 plus prefetch. For HBF, HBM plays this role: weights stream in page-sized chunks, so the 128 B-vs-4 KB mismatch is much smaller for LLM tensors (RoMe §III) than for graph analytics.
  2. **Read-only mapping in the MMU.** A block-level mapping small enough for the MMU (80 KB/TB) works because data is mostly read-only. That fits H^3's read-only placement of weights and shared KV exactly: no page-level FTL is needed, and a static block map suffices.
  3. **Handle writes separately.** ZnG merges writes in flash registers with an L2 overflow before any program. For HBF, per-request KV writes should go to HBM, and any rare HBF writes (for example, a new shared-cache version) should be batched to full pages or blocks.
  4. **GC in a helper thread blocks the owning tenant.** If HBF data is ever rewritten, GC stalls will reach whatever model or tenant owns those blocks. The BTP's finding that GC is structurally absent at H^3 provisioning is consistent with ZnG's view that read-dominant use avoids GC.
  5. **Flash-network bandwidth versus array bandwidth.** ZnG widened the channel from 1 B to 8 B and used a mesh. HBF's TSV/die-stack interface is the modern equivalent.
- **Device numbers usable as sensitivity points:** 3 µs read, 100 µs program, 100k P/E (SLC Z-NAND). These are Z-NAND, not HBF values; label them as such if used.

### 3.8 Critical view
- The GPU is 2010-era (GTX580 model), the workloads are graph and HPC, and there is no DNN or LLM work.
- There is no energy or power evaluation, even though the motivation cites W/GB.
- The baseline choice flatters ZnG. The 7.5× is against HybridGPU, which ZnG-base actually loses to. No comparison against a GPU with big DRAM is made except "Ideal" in the multi-app study.
- Details are inconsistent: the flash-register count (8 in the text vs "2/8" in the table), and the 6 MB baseline L2 vs "match GV100".
- The programmable row decoder (a CAM built from flash cells) has no area, latency or reliability numbers. Neither do STT-MRAM retention or endurance under L2 write traffic.
- Endurance is argued qualitatively. There is no lifetime calculation.

### 3.9 BibTeX
```bibtex
@misc{zhang2020zng,
  title         = {{ZnG}: Architecting {GPU} Multi-Processors with New Flash for Scalable Data Analysis},
  author        = {Zhang, Jie and Jung, Myoungsoo},
  year          = {2020},
  eprint        = {2006.08975},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR}
}
% Conference version (venue from filename; pages/DOI unverified -- fill from IEEE Xplore):
@inproceedings{zhang2020zng_isca,
  title     = {{ZnG}: Architecting {GPU} Multi-Processors with New Flash for Scalable Data Analysis},
  author    = {Zhang, Jie and Jung, Myoungsoo},
  booktitle = {Proceedings of the ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)},
  year      = {2020}
}
```

---

## 4. HAVEN: High-Bandwidth Flash Augmented Vector ENgine for Large-Scale ANNS Acceleration

### 4.1 Bibliographic
- **Title:** "HAVEN: High-Bandwidth Flash Augmented Vector ENgine for Large-Scale Approximate Nearest-Neighbor Search Acceleration"
- **Authors:** Po-Kai Hsu (Georgia Institute of Technology), Weihong Xu (EPFL), Qunyou Liu (EPFL), Tajana Rosing (UC San Diego), Shimeng Yu (Georgia Tech). Hsu and Xu contributed equally.
- **Venue/year:** arXiv:2603.01175v1 [cs.AR], 1 Mar 2026. The text uses an ACM template but names **no venue**.
- **Pages/DOI:** unknown.

### 4.2 Problem
- RAG retrieval with IVF-PQ needs **reranking** on full-precision vectors to reach high recall.
- Billion-scale raw vectors are 93–252 GB (Table 3), so they do not fit in GPU HBM, which is "typically 80–192 GB per device". They sit in host DDR or NVMe instead.
- Fig. 2 (measured on an A100):
  - (a) Reranking costs one to two orders of magnitude in QPS.
  - (b) Raw vectors dominate the footprint.
  - (c) Reranking dominates query time.

### 4.3 Core mechanism
1. **HBF replaces part of the HBM stacks** in the GPU package, on a 2.5D interposer (Fig. 3).
   - Coarse probing and the PQ list scan stay on the GPU with HBM.
   - Reranking moves to HBF, together with a **near-storage search unit (NSU)** on the HBF logic die (Fig. 4b).
2. **NSU components:**
   - **32 Rerank Queues**, each an 8 KB buffer holding ≤1,024 IDs (32-bit) and distances.
   - Address Generation, which issues HBF reads.
   - A Distance Computation module with **32 MACs**.
   - A Top-k unit (a **256-point Bitonic sorter**).
   - Motivation: reranking has low arithmetic intensity, so vectors are not moved to HBM.
3. **3D NAND re-architecture.** Monolithic planes become **distributed small subarrays**, each with local WL/BL drivers and sense amplifiers, similar to the fast-NAND approach of [1]. This shortens bitlines, cuts parasitics and energy, and increases parallelism.
   - Heterogeneous integration: high-voltage WL/SSL drivers in CMOS-under-array (22 nm), and the low-voltage page buffer and I/O on a bonded **7 nm** CMOS tier (Fig. 6).
4. **Modelling flow:**
   - **3D-FPIM** (MICRO'22) subarray model plus **NeuroSim** (V1.4) for peripherals, giving array-to-die projections.
   - The stack follows the HBM organisation: 8-Hi, microbump/TSV pitch **50 µm**, and TSV organisation "aligned with HBM2E specification integrated in A100".

### 4.4 Key parameters (HBF device modelling; directly useful)
- **Table 1 (3D NAND subarray):**
  - 7 nm FinFET CMOS tier at VDD1 = 0.7 V; 22 nm CMOS-under-array at VDD2 = 0.9 V.
  - VC hole diameter 145 nm; BL pitch 40 nm; VC hole pitch 248 nm; WL staircase pitch 725 nm.
  - WL layers 64 / 128 / 196 / 256; SSL = 2; sub-blocks = 2.
  - BL count (page) 1 / 2 / 4 KB; blocks 64–1024; **SLC (1 bit)**.
- **Capacity:** reaching **512 GB in an 8-Hi stack** needs at least 128 WL layers. The design uses **256 WL layers** as the baseline for margin and to follow the roadmap.
- **Energy:** calibrated from about **30 pJ/bit** read energy for contemporary 3D NAND [10], plus HBM2 data-movement energy [17]. Read energy rises quickly with page size and block count.
- **Power sets bandwidth.** The design adopts an **HBM-like 30 W envelope per stack** [25]. Bandwidth under that cap degrades sharply above 128 blocks.
- **Bandwidth cap:** a maximum of **460 GB/s per stack**, chosen for HBM2E compatibility [23].
- **Latency** grows with subarray size (RC delay and page activation). **No numeric HBF read latency is given in the text**; values appear only in figures, which are not recoverable from the extraction.
- **Chosen configuration:** **4 KB page, 64 blocks per subarray, 256 WL layers**.
- **Industry projection** (SanDisk blog [3]): HBF offers **8–16× HBM capacity per stack** and "hundreds of GB/s" read bandwidth. The paper also claims HBF "exposes wide internal parallelism and finer access granularity" than SSDs.
- **NSU (Table 2):** 22 nm synthesis at 1 GHz; **4.11 mm², 620.3 mW** in total.
  - Rerank queues: 3.84 mm², 122.6 mW.
  - MACs: 0.03 mm², 11.6 mW.
  - Bitonic sorter: 0.24 mm², 486.1 mW.

### 4.5 Evaluation
- **Real-system baselines:** AMD EPYC 7302 (16-core); 512 GB DDR4-3200; NVIDIA **A100 40 GB**; 8 TB PCIe 4.0 NVMe.
- **Index setup:** Faiss IVF-PQ with 16-byte PQ codes, cuVS enabled. Metrics are QPS and latency at recall@k = 100.
  - Recall is fixed at 0.95 for BIGANN-1B and SPACEV-1B and at 0.9 for Wiki-88M.
- **Datasets (Table 3):**

| Dataset | Vectors | Dim | Bits | Raw size | N_list |
|---|---|---|---|---|---|
| BIGANN-1B | 1B | 128 | 8 | 119 GB | 128k |
| SPACEV-1B | 1B | 100 | 8 | 93 GB | 128k |
| Wiki-88M | 88M | 768 | 32 | 252 GB | 16k |

- **Results:**
  - **Throughput:** HBF is about **3–8×** higher on BIGANN and SPACEV and **>20×** on Wiki-88M, and it scales with batch sizes 4–32.
  - **Latency:** HBF stays around **10 ms**. It is 3–6× faster than DRAM and over 10× faster than SSD on BIGANN and SPACEV, and 10–40× faster on Wiki-88M.
  - HBF Pareto-dominates the QPS–recall trade-off, and the gap widens at recall ≥0.9.
  - **Design-space exploration (Fig. 12):** more layers, blocks or page size means more capacity but less throughput. The 256L, 4 KB, 64-block point is best. Total 8-Hi capacity on the x-axis spans roughly 400–1400 GB.
  - **Table 4 (BIGANN-1B, recall ≥0.9):** HAVEN 8.1k QPS vs ANNA 4.2k (1.9×) and SmartANNS 0.9k (9×).
  - Headline: throughput up to **20×** and latency up to **40×** better.

### 4.6 Stated limitations / future work
- The only stated trade-off is that "capacity and performance scale in opposing directions, and no single design maximizes both".
- There is no explicit limitations or future-work section. The conclusion calls HBF "a promising direction for future memory-centric AI accelerators targeting retrieval-heavy workloads".

### 4.7 Relevance to HBM+HBF
- **This is the most concrete public HBF device model** found in this set. It provides:
  - an 8-Hi, 50 µm-TSV, HBM2E-aligned stack;
  - a **power-limited bandwidth** model (30 W/stack, then a sweep over page and block, capped at 460 GB/s);
  - capacity versus layers (≥128 layers for 512 GB in 8-Hi);
  - read energy calibrated at about 30 pJ/bit;
  - a 4 KB SLC page.
- These numbers can parameterise the BTP's HBF tier as a sensitivity point: 460 GB/s/stack, 512 GB+/stack, 30 W/stack. The report's own "4× power per stack" assumption should be reconciled against the HAVEN 30 W envelope and source.
- **Placement pattern matches H^3.** Latency-critical, compressed state (PQ codes) goes in HBM; bulk, read-only full-precision data goes in HBF. This is the same read-only placement principle as H^3 (weights and shared KV in HBF).
- **Near-storage compute on the HBF logic die.** It avoids moving low-arithmetic-intensity data across the interposer. For LLMs the analogue is attention over the shared KV cache near HBF, since decode attention is low-AI, like reranking. This is a design axis the BTP does not model.

### 4.8 Critical view
- **HBF performance is not measured.** DRAM and SSD baselines are measured on a real A100 system, but the text does not explain how HBF end-to-end QPS and latency were obtained (analytical model? simulation? which latency?). It gives no numeric HBF read latency, no stack count, and no split of the HBM stacks replaced.
- **Weak baselines.** Comparisons are against PCIe-attached DDR/NVMe; there is no multi-GPU HBM or CXL-memory baseline.
- **Nothing on writes:** no write, program, endurance, retention or index-update discussion, although vector databases do get updated. HBF is only argued to fit "read-intensive" workloads.
- **Textual inconsistencies:** "Figure 8(d)" is cited for both bandwidth and latency (bandwidth is 8(c) per the caption). The table's "(4.7×)" normalisation for ANNA is against SmartANNS, while the text quotes 1.9× vs ANNA; both are consistent with 4.2k/0.9k and 8.1k/4.2k, but the presentation is confusing.
- **Thermal:** the 30 W/stack envelope is borrowed from HBM thermal analysis. There is no thermal simulation of NAND stacked near a hot GPU, even though NAND retention is temperature-sensitive.
- **Granularity:** the "finer access granularity" claim is not quantified. The chosen page is still 4 KB, while a 128-dim 8-bit vector is only 128 B **(my derivation: 128 × 1 B)**, so reranking reads may overfetch unless vectors are packed. The paper does not discuss this.

### 4.9 BibTeX
```bibtex
@misc{hsu2026haven,
  title         = {{HAVEN}: High-Bandwidth Flash Augmented Vector {ENgine} for Large-Scale Approximate Nearest-Neighbor Search Acceleration},
  author        = {Hsu, Po-Kai and Xu, Weihong and Liu, Qunyou and Rosing, Tajana and Yu, Shimeng},
  year          = {2026},
  eprint        = {2603.01175},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR}
}
```

---

## 5. LIFE: Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling

### 5.1 Bibliographic
- **Title:** "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling". The framework is **LIFE (LLM Inference Forecast Engine)**.
- **Authors:** Rajeev Patwari, Ashish Sirasao, Devleena Das (Advanced Micro Devices, San Jose, California).
- **Venue/year:** arXiv:2508.00904v1 [cs.PF], 29 Jul 2025. The **venue is not named in the text** (it uses an IEEE-style template).
- **Pages/DOI:** unknown.

### 5.2 Problem
- Performance forecasting for LLMs on heterogeneous personal devices (CPU, NPU, iGPU) is hard. Existing approaches are GPU benchmarking, ML latency predictors (NeuSight) or simulators (ASTRA-sim, LLMServingSim, Vidur), and they are hardware-specific or heavy.
- Peak TOPS and bandwidth alone do not capture how efficiency changes with prompt length, KV growth and optimisations.

### 5.3 Core mechanism
1. **Operator-level analytical models** count compute ops and memory read/write bytes as functions of shape and datatype (nbytes, qbytes). No real computation is done.
   - **Foundational operators (Table 1):**

| Operator | Compute ops | Memory RD+WR (bytes, as printed) |
|---|---|---|
| Linear (GEMM+Bias) | 2mkn | ((mk)+(kn)+n)+(mn) |
| (De)Quantize | 2·num_el | num_el×nbytes + num_qparams×nbytes + num_el×qbytes |
| BMM | 2bmkn − bmn | ((bmk)+(bkn))×nbytes + (bmn)×nbytes |
| Elemw | mn | 2mn×nbytes + mn×nbytes |
| PWL non-linear | 2·num_el | (num_el + tables)×nbytes + num_el×nbytes |
| Polynomial non-linear | ((n(n+1)/2)+n)×num_el | (num_el + n)×nbytes + num_el×nbytes |
| Embedding | 1 | vocab×hidden×nbytes + hidden×nbytes |

   - **Derived operators (Table 2):** quantised Linear, LoRA Linear, inverse / inverse-sqrt, RoPE, Norm, Softmax, MLP, MHA, MLA.
   - Appendix 8.1 gives the Python code for the GEMM model: opcount = 2mkn − mn; int4 per-group dequantisation adds 2kn ops plus scale and zero reads; LoRA adds its A/B reads and the ops for (k·r·n)·2 + kn.
2. **Modelled optimisations:**
   - Operator fusion, modelled by removing intermediate memory traffic (e.g., Flash Attention).
   - Dynamic shape padding and tiling. Tiling is used "if the tensors do not fit into the on chip [memory]", which increases dispatch calls.
   - Quantisation, including the dequantisation overhead.
   - MHA, GQA, MQA and MLA.
   - KV compression: MLA, or KV at 4 or 8 bits.
   - Chunked prefill.
   - LoRA, merged inline or ahead of time.
3. **Pipeline (Fig. 2):** config file → analytical LLM → simulation over past and present sequence lengths → a **hardware-agnostic statistics database** (ops, memory RD/WR, KV RD/WR, dispatch calls) → analysis with hardware specs (TOPS, BW, efficiencies) → TTFT, TPOT and TPS.
4. **Performance equations:**
   - Eq. 1: t_c = Σ_op TOPs_op / (ec_op · TOPS) + Σ_op t_dispatch_op
   - Eq. 2: t_m = Σ_op Mem_op / (em_op · BW) + Σ_op t_dispatch_op
   - Eq. 3: TTFT = max(t_c, t_m)
   - Eqs. 4–5: TPOT = Σ_op MEM_op / (BW · em_op) + Σ t_dispatch, or MEM / (BW · em_avg) + t_dispatch_total. The extraction garbles the fraction, but the text says TPOT depends solely on t_m because t_c << t_m in decode.
   - Eq. 6: TPS = 1 / TPOT.
   - Eq. 7: t_lora = Σ_linear (W + BA).
   - The efficiencies ec and em come from operator unit tests, or are swept.

### 5.4 Key numbers
- **Models:** Llama2-7B variants (Table 3): bf16-bf16, bf16-int4, bf16-int4-fused, bf16-int4-kv4, bf16-int4-mla, bf16-int4-lora, QuaRot-w4a4kv4, fp16-fp16.
- **Prefill compute versus prompt length (Table 4, bf16):**

| Prompt | TOPs | KV (GB) | Compute mix |
|---|---|---|---|
| 256 | 3.42 | 0.1 | GEMM 99.0% |
| 2048 | 29.29 | 1.0 | GEMM 92.4% |
| 8192 | 143.87 | 4.0 | GEMM 75.2% |
| 32768 | 1002.67 | 16.0 | BMM 56.0% |
| 65536 | 3144.41 | 32.0 | BMM 71.6% |

- **Memory versus quantisation (Table 5):** TOPs barely change, but memory does. At prompt 2048:
  - bf16-bf16: MemRD 43.5 GB, MemWR 29.0 GB.
  - bf16-int4: MemRD 34.4 GB.
  - bf16-int4-kv4: MemRD 10.1 GB, MemWR 4.4 GB, KV 0.25 GB.
- **Decode (Table 7):** bf16-bf16 needs about 13.3–14.4 GOPs and 12.85–14.83 GB per token at prompts 32–2048. int4 roughly doubles GOPs (from dequantisation) but cuts memory to 3.74–5.72 GB.
- **Memory growth over 2000 generated tokens (Table 9):**
  - Prompt 128: bf16-int4 goes from 3.65 to 5.60 GB (1.53×); bf16-int4-kv4 from 3.53 to 3.90 GB (1.10×).
  - Prompt 4096: bf16-bf16 goes from 16.66 to 18.62 GB.
  - Without KV compression, TPS drops by as much as 50% for smaller prompts and 26% for longer ones. With compression it drops by at most 10%.
- **Dispatch calls per decode step (Table 8):** from Gemma2-2B-int4 at 497 up to Qwen3-32B-int4 at 1219. Llama2-7B-int4 is 611.
- **Attention memory at prompt 8192, first → 2000th token (Table 11, MB; MHA/GQA/MQA/MLA):**
  - Fused: 322/178/136/278 → 368/201/152/333.
  - Fused-KV4: 178/106/85/110.
  - Conclusion: MLA is about 50% less than MHA, and GQA with a compressed KV is comparable to MLA.
- **LoRA (Table 12):** a full-model update at r = 128 is **1670.8 GOPs**, about half of prefill-256 (3.42 TOPs). Continuous (inline) merging therefore hurts TTFT for short prompts by more than 50%.
- **Verification hardware:**
  - AMD Ryzen 9 HX 370 CPU: 326.4 GFLOPS, 240 GBps.
  - AMD Ryzen AI Max+ 395: "126 TOPS overall", 50 TOPS NPU for prefill, 256 GBps iGPU for decode.
  - NVIDIA V100: "126 TOPS", 900 GBps.
  - Software: PyTorch 2.6 with HF transformers 4.49.0, and RyzenAI hybrid-llm.
- **Prefill verification (the text calls it Table 7; the table header is Table 6):**
  - CPU efficiency (forecast at 100% ÷ measured) is 70.3% at prompt 32 but falls to 48.2% at 2048.
  - NPU: 11.3% (128) and 23.9% (1536).
  - V100: 50.3–58.6%.
  - TTFT units are not printed; presumably seconds.
- **Decode verification (Table 10):**
  - CPU: measured TPS 1.59 → 0.45 at 3–10% efficiency; forecast 1.87 → 1.62 at an assumed 10%.
  - iGPU: measured 34.5 / 32.8 at 52.5%; forecast 33.4 / 27.2 at 50%.
  - V100: measured 40.0 / 36.9 / 32.1 at 60 / 57 / 51%; forecast 32.6 / 30.3 / 26.7 at 50%.

### 5.5 Stated limitations / future work
- The study covers dense LLMs only. Extending to **VLMs, MoEs and speculative decoding** is "left for future exploration".
- Forecast accuracy depends on knowing the efficiencies. BMM efficiency in decode follows a sawtooth because of tiling and padding (Fig. 8; the ideal and tiled counts differ by up to 1000× at tile 64), which makes TPS forecasting "a challenge".
- The authors say a single roofline is insufficient because efficiencies vary with operating conditions.

### 5.6 Relevance to HBM+HBF: how to extend LIFE (or the BTP's LLMSimulator roofline) with an HBF tier
LIFE's decomposition (a hardware-agnostic byte and op count per operator, then division by BW × efficiency) is exactly the shape needed for a quick two-tier design-space exploration. A concrete extension **(my proposal, not in the paper)**:
1. **Split bytes by tier.** Tag each operator's memory reads by tensor class (weights, shared KV, per-request KV, activations) and map each class to a tier by a placement policy (H^3: weights and shared KV in HBF). The statistics database already has a separate "KV read/write" metric, so this is a small change.
2. **Two-tier memory time.** Per operator, t_m,op = max(Mem_op^HBM / (em^HBM·BW_HBM), Mem_op^HBF / (em^HBF·BW_HBF)) if the tiers are accessed concurrently, or the sum if serialised through HBM staging. Then t_op = max(t_c, t_m,op) + t_dispatch.
3. **An HBF efficiency term em^HBF(request size).** This captures page overfetch (ZnG: 97% waste at 128 B vs 4 KB) and plane parallelism. For streaming weights em is close to 1 (RoMe's argument); for sparse KV selection (InfiniGen) em drops.
4. **A latency term that LIFE lacks.** Add a per-dispatch HBF first-access latency L_HBF (µs-scale; Z-NAND reads 3 µs in ZnG). Alternatively, require prefetch depth ≥ L_HBF × BW_HBF (Little's law) and charge only unhidden latency when the lookahead window (one layer, as in InfiniGen) is too short.
5. **Capacity and batch.** LIFE has **no capacity model and appears to be single-sequence**; batch size is not stated anywhere. Add the BTP's batch equation, batch = (HBM − weights_HBM − shared_HBM − working) / KV_per_req, and multiply per-step weight reads by 1/batch amortisation. This is where HBF's benefit actually appears.
6. **Power and write accounting.** Per-tier pJ/bit (HAVEN: about 30 pJ/bit NAND read), plus a stack power cap that limits achievable BW_HBF (HAVEN: 30 W → ≤460 GB/s). Price any HBF writes with program bandwidth and P/E budget.
7. **Validation route.** LIFE's approach of fitting em per operator from unit tests maps to measuring HBF efficiency per request size once a device model or emulator exists.

### 5.7 Critical view
- The target is client devices (CPU/NPU/iGPU) at what appears to be batch 1. There is **no batching, no multi-device, no capacity constraint, and no latency or overlap modelling** (t_c and t_m are just maxed). It does not model a memory hierarchy (cache, HBM, host). The "hardware-agnostic" claim therefore rests entirely on user-supplied efficiencies.
- **Weak validation:**
  - Forecasts are made at a single assumed efficiency (10% or 50%), not predicted.
  - CPU decode TPS is off by up to 3.6× at prompt 2048 (1.62 forecast vs 0.45 measured).
  - V100 decode is under-predicted by about 17–19% (e.g., 32.6 vs 40.0).
  - The models are Llama2-7B variants only; Table 8 counts dispatch calls for other models but does not validate them.
- **Odd modelling choices:**
  - The eager, unfused prefill shows MemWR of 29–90 GB for a 7B model (Table 5), because intermediate attention tensors are written out. Prefill on the 10–100 GBps grid is then called "memory-bound", which would not hold with fused attention.
  - TPOT ignores compute entirely.
- **Typos and inconsistencies:**
  - "30 TOPS and 50 GBps" in the text vs "50TOPs, 30GBps" in the Fig. 5 caption.
  - Table numbering (the text's Table 7 is actually Table 6).
  - "7&" in Table 10.
  - Table 7 bf16-int4 at prompt 128 reads 25.60, likely a typo for about 26.6.
  - The V100 is listed at "126 TOPS", which is the same figure as the Ryzen AI Max+ overall rating.

### 5.8 BibTeX
```bibtex
@misc{patwari2025life,
  title         = {Forecasting {LLM} Inference Performance via Hardware-Agnostic Analytical Modeling},
  author        = {Patwari, Rajeev and Sirasao, Ashish and Das, Devleena},
  year          = {2025},
  eprint        = {2508.00904},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF}
}
```

---

## 6. Cross-paper observations and gaps relevant to HBM+HBF

1. **Access granularity decides whether flash bandwidth is usable.**
   - ZnG measured a slowdown of up to 28× and 97% bandwidth waste when 128 B GPU requests hit 4 KB Z-NAND pages [ZnG §III-A].
   - RoMe shows that LLM weight, KV and activation streams are KB–MB-sized, so even 4 KB rows lose nothing [RoMe §III, §VI-B].
   - HAVEN picks a 4 KB HBF page [HAVEN §3.2].
   - **Implication:** dense LLM decode (weights, full-scan shared KV) is a near-ideal HBF client. The risk is sparse or top-k attention: RoMe's DSA caveat [RoMe §VII], and InfiniGen's token selection [InfiniGen §4.3]. With per-head selection, chunks are about 256 B per K or V per head **(my derivation)**, far below a 4 KB page.
2. **Read-only placement is the recurring enabler.**
   - ZnG's FTL is small only because data is read-mostly: a read-only block map of 80 KB/TB in the MMU [ZnG §III-B, §IV-A].
   - HAVEN keeps only read-only raw vectors in HBF [HAVEN §3.1].
   - RoMe notes LLMs are read-dominated and handles writes immediately [RoMe §V-B].
   - All of this is consistent with H^3's weights-and-shared-KV-in-HBF placement and with the BTP's finding that GC is absent.
3. **Writes must be absorbed before they reach flash.**
   - ZnG needed flash-register merging, the NiF network and L2 redirection to reduce write redundancy from 51 to 1.2. Z-NAND programs are 33× slower than reads, and GC stalls the owning app by 73% [ZnG §IV-C, §V].
   - InfiniGen's pool manager overwrites KV slots in place [InfiniGen §4.4], which is flash-hostile. Per-request KV must stay in HBM.
4. **Hiding latency needs lookahead proportional to latency × bandwidth.**
   - InfiniGen gives one layer of lookahead [InfiniGen §4.3].
   - ZnG uses a PC-based prefetcher with adaptive size (93% accuracy) [ZnG §IV-B].
   - Weights have a deterministic layer order, so prefetch can be perfect.
   - No paper here sizes the HBM staging buffer for HBF. As an illustration only, 460 GB/s/stack [HAVEN] × 3 µs [ZnG Z-NAND read] ≈ 1.4 MB in flight per stack **(my derivation, mixing two sources)**.
5. **HBF bandwidth is power-limited, not only pin-limited.** HAVEN derives bandwidth from a 30 W/stack envelope and caps it at 460 GB/s [HAVEN §3.2]. The BTP's HBF power figure ("4× power per stack") should be sourced and compared with this. Tier power should appear in any perf/W claim, such as the report's "21% lower power".
6. **The logic die is the natural home for flash management and near-data compute.** RoMe puts its command generator there [RoMe §IV-C]; HAVEN puts its NSU there (4.11 mm², 620 mW) [HAVEN §3.1]; ZnG pushes flash control to per-channel controllers and the MMU [ZnG §III-B]. For LLMs, near-HBF attention over the shared KV (low arithmetic intensity, like reranking) is an unexplored option in the BTP.
7. **Capacity drives batch, and batch drives throughput.**
   - RoMe's evaluation caps batch by memory capacity [RoMe Fig. 12 caption].
   - InfiniGen shows KV exceeding weights [InfiniGen Fig. 2].
   - LIFE has no capacity model at all [LIFE §4.2], which is the biggest gap if LIFE is used for HBF design-space exploration.
   - Useful LIFE KV sizing: Llama2-7B bf16 needs about 0.5 GB of KV per 1K tokens per sequence (Table 4: 1.0 GB at 2048, 32 GB at 65536).
8. **Modelling-fidelity gap.**
   - RoMe: cycle-accurate DRAM (Ramulator 2.0 inside LLMSimulator).
   - HAVEN: circuit-level NAND (3D-FPIM + NeuroSim), but its system-level HBF performance method is not described.
   - LIFE and the BTP's LLMSimulator extension: roofline with efficiency factors.
   - ZnG: SimpleSSD + MacSim, the only one with FTL/GC timing.
   - **No paper provides a validated system-level HBF timing model for LLM inference.** Combining LIFE-style byte counting with a HAVEN-parameterised HBF tier (BW cap, pJ/bit, page size) plus ZnG-style latency and write accounting is a defensible middle ground for a BTP.
9. **Baselines tend to be weak.** InfiniGen compares against PCIe 3.0 offload [InfiniGen §5.1]; HAVEN against PCIe DDR/NVMe [HAVEN §4.1]; ZnG against HybridGPU, which its own base design loses to [ZnG §V-B]. None compares an HBF design against the real alternative the BTP cares about: **more GPUs or HBM stacks at equal cost or power**.
10. **Endurance and retention are under-treated.**
    - ZnG cites Z-NAND SLC at 100k P/E and argues lifetime only qualitatively [ZnG §VI].
    - HAVEN assumes SLC but says nothing about endurance, retention or temperature [HAVEN].
    - RoMe and LIFE do not apply.
    - The BTP's wear-concentration analysis fills a real gap.
11. **Sparse attention cuts both ways.** InfiniGen reduces the KV volume moved to <10% [InfiniGen §5.1]. That can make HBF bandwidth sufficient for a flash-resident shared cache, but its speculation needs a partial key cache equal to 15% of KV [InfiniGen §6.2], which must itself sit in fast memory. This is a direct HBM capacity cost that belongs in the BTP's batch numerator.
12. **Report errata to watch when citing.**
    - RoMe: the 4480 vs 560 TFLOPS inconsistency.
    - ZnG: register count 8 vs 2/8.
    - HAVEN: the Fig. 8(c)/(d) mix-up and no numeric latency.
    - LIFE: the 30/50 TOPS–GBps swap and table-number mismatches.
    - Venues for RoMe, InfiniGen and ZnG are **inferred from filenames**, not from the text. Verify before final BibTeX.
