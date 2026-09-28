# D1 — Prior work on NAND flash for LLM inference (mostly on-device)

Notes for BTP: HBM + High-Bandwidth Flash (HBF) for datacenter LLM inference (H^3-style: read-only weights + shared KV prefix in flash, per-request KV in HBM).

Sources read in full (plain-text extractions in `scratchpad/txt/`):
1. LLM in a Flash (Apple) — arXiv 2312.11514v3 / ACL 2024
2. AiF: Accelerator-in-Flash (SNU) — ISCA 2025
3. Lincoln (Tsinghua) — HPCA 2025
4. Cambricon-LLM (ICT-CAS / Cambricon) — arXiv 2409.15654v1 (file name says MICRO'24)

Conventions used here:
- "[p]" = stated in the paper. "[mine]" = my derivation, inference or opinion. Numbers are quoted as the papers give them. Where a text disagrees with itself, both values are shown.
- "unknown" = not in the paper's text. Anything I could not confirm is marked VERIFY.

---

## 0. One-glance comparison

| | LLM in a Flash | AiF | Lincoln | Cambricon-LLM |
|---|---|---|---|---|
| Where compute happens | Host CPU/GPU (no flash changes) | In the flash chip (GEMV PEs + ECC on the NAND die's periphery) | NPU for prefill (flash placed on the LPDDR bus); near-flash FP-Dot per plane for decode (logic die hybrid-bonded under the array) | Split: GEMV partly on-die (one shared Compute Core per die) and partly on the NPU (weights streamed through idle channel slots) |
| Flash type assumed | Commodity Apple/NVMe SSD (measured) | 3D TLC, used as fast SLC-like via LSB-only storage | SLC, array-shrunk, XL-Flash-derived, 32 planes/die | Generic NAND (text says "3D TLC" when discussing errors), tR = 30 µs |
| tR | not modelled (measured end-to-end throughput) | LSB 28 µs with (1,3,3) coding; 9.7 µs with cr-read | 3.426 µs (simulated) | 30 µs |
| Page | not stated (reads of 32 KiB+ recommended) | 16 KiB | 4 KB + 448 B ECC | 16 KB |
| ECC | handled by the SSD (not discussed) | On-chip BCH "ECCLITE": 10 bit/1 KiB at 6.4 GB/s | Near-plane BCH(9088,8192,64), 14.6 Gb/s per plane | Outlier-only on-die "ECC": majority vote over the top 1% of values, plus threshold clipping |
| Read disturb / endurance | not discussed | P/E and retention are used only to size the ECC; read disturb not discussed | Treated explicitly: read reclaim, relaxed retention, wear levelling | not discussed |
| Batch | 1 (limitation they state) | 1 ("single-batch processing characteristic") | 1, plus speculative decoding (~7 draft tokens verified as a mini-batch) | 1 |
| Headline | 2× DRAM-size models; 4–5× (CPU), 20–25× (GPU) over naive loading | 14.6× over SSD offload; 1.4× over in-memory | prefill up to 13.23×, generation up to 254.1× over an SSD baseline | 70B at 3.44 tok/s; 22–45× over flash offloading |

---

## 1. LLM in a Flash: Efficient Large Language Model Inference with Limited Memory

### 1.1 Bibliographic
- **Title:** "LLM in a flash: Efficient Large Language Model Inference with Limited Memory"
- **Authors (arXiv v3 order):** Keivan Alizadeh, Iman Mirzadeh*, Dmitry Belenko*, S. Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar (Apple; * = "Major Contribution"). The ACL version (as cited in Lincoln ref [7]) spells the second author "S. I. Mirzadeh" (Seyed Iman).
- **Venue:** arXiv:2312.11514v3 [cs.CL], 30 Jul 2024 (this is the version read). Published in *Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers)*, Bangkok, Aug 2024, pp. 12562–12584, https://aclanthology.org/2024.acl-long.678. The ACL details come from Lincoln's reference [7], not from the arXiv text itself.
- **DOI:** not in either text. 10.18653/v1/2024.acl-long.678 follows the ACL Anthology pattern — VERIFY.

### 1.2 Problem
Run LLMs whose weights are larger than available DRAM (roughly 2×) on personal devices. The weights stay in flash (SSD) and are brought into DRAM on demand for each token. The problem has two parts: flash bandwidth is roughly 10–100× below DRAM, and small random reads are much slower than large sequential ones.

### 1.3 Core mechanism
The paper uses an inference cost model with three terms: flash I/O, DRAM memory-management overhead, and compute. It pursues three goals: (i) move less data, (ii) read larger chunks, (iii) manage the DRAM copy cheaply. Flash hardware is not modified.
- **Selective persistence.** Embeddings and attention weights stay in DRAM permanently; attention is "approximately one-third of the model's size". Only FFN weights are streamed from flash.
- **Activation sparsity plus a low-rank predictor.** The method relies on ReLU sparsity in FFN layers: OPT-6.7B shows 97% sparsity in its FFN layer, relufied Falcon 7B 95%, and Llama 2 with FATReLU 90%. A per-layer low-rank predictor, fed by the *current* layer's attention output, predicts which intermediate neurons will fire. Only the matching up-projection columns and down-projection rows are loaded. Predictor ranks: OPT uses r=128 for the first 28 layers and r=1024 for the last 4. OPT predictors give about 5% false negatives and 7% false positives. Each predictor was trained on 10k C4 samples for 2 epochs, taking about 4 h on one A100. Predictor overhead is 2.75% of inference time on the M1 Max CPU and 4.8% on the RTX GPU (OPT). The predictor over-predicts by about 3× (OPT, Persimmon) and about 2× (Phi-2) relative to the neurons that actually fire.
- **Sliding window (temporal reuse).** Neurons active for any of the last k tokens stay in DRAM. Each new token loads only s_agg(k+1) − s_agg(k), the neurons not already cached. For OPT with k=4 or 5, a single token's active set would need about 10% of DRAM on average, but the incremental load per token drops to 2.4% of FFN neurons. The window itself occupies 24% of FFN. Total DRAM use is 52.1% of the model.
- **Row–column bundling (data layout).** Column i of the up-projection and row i of the down-projection are stored contiguously, because they are used together. This doubles the read chunk from d_model×bytes to 2·d_model×bytes; for OPT-6.7B fp32 the chunk is 32 KiB. Bundling neurons by co-activation ("closest friend") was a **negative result**: very active neurons are everyone's closest friend, so they got loaded many times (App. D).
- **Handling flash latency.** Reads are issued by 32 parallel threads to amortise "latency to first byte" and to use the SSD's internal parallelism. The authors find that random reads of 32 KiB or more across threads give usable throughput. Sometimes it is better to "read more than needed (but in larger chunks) and then discard".
- **DRAM management.** Each layer's matrix is preallocated at Req_i × 2·d_model. Deleting a neuron moves the last row into its slot, which costs O(c·d_model). New rows are appended at the end. No reallocation is needed because FFN neuron order does not matter.
- **Page granularity, ECC, read disturb, endurance:** not discussed. The paper works at the file-I/O level. Benchmarks run with OS caching disabled (F_NOCACHE on macOS, DirectIO on Linux, `purge` before runs), which the authors describe as a conservative lower bound on throughput.

### 1.4 Flash parameters assumed or measured
- Figure 2a (schematic): flash ~100 GB at ~1 GB/s; DRAM ~10 GB at ~100 GB/s.
- M1 Max with 1 TB SSD: more than 6 GiB/s for a 1 GiB linear uncached read; 6.1 GB/s is the dense baseline in Table 2. Sparse, scattered reads achieve 1.25 GB/s, rising to 2.25 GB/s with bundling (Table 2).
- Figure 2b: random-read throughput rises with chunk size (4–64 KB) and thread count (2–32).
- tR, page size, planes, channels, endurance: **not given** (unknown).

### 1.5 Evaluation and headline results
- **Setup:** batch 1, one sequence at a time. Prompt is the first 128 tokens of a C4 validation example; the model generates 256 tokens. About 50% of model size is available in DRAM (65% for Phi-2). Hardware: M1 Max (1 TB SSD, CPU fp32 or Metal fp16), M2 Ultra (2 TB SSD), and Linux with an RTX 4090 24 GB (bf16). Software: HF Transformers + PyTorch, with custom MLX Metal kernels for shape dynamism. Baselines are "Naive" (load everything per forward pass) and "Hybrid" (half resident, half loaded, no sparsity); both use "best theoretical possible numbers" for I/O.
- **Table 2 (OPT-6.7B, M1 Max, I/O only):** naive 13.4 GB at 6.10 GB/s = 2196 ms; hybrid 1090 ms; +predictor 738 ms; +windowing 164 ms (0.2 GB transferred); +bundling **87 ms** at 2.25 GB/s.
- **Table 3 end-to-end (ms/token):** OPT-6.7B CPU: naive 3182 vs All 669. Metal M1: 2389 vs 565. Metal M2: 2270 vs 305. GPU: 2218 vs 84. **Speculative GPU: 60.** Falcon 7B CPU: 3095 → 706. Persimmon 8B CPU: 3806 → 1041. Phi-2 CPU: 1287 → 546. Llama 2 7B CPU: 3095 → 994.
- **Headline:** models up to 2× DRAM size; "4x, 7x, and 20x" speedups over naive on CPU, Metal and NVIDIA GPU (the Discussion says 4–5× CPU and 20–25× GPU).
- **Accuracy:** zero-shot scores barely change. OPT with predictors: ArcE 66.2, ArcC 30.6, HellaSwag 49.8, vs 66.1 / 30.6 / 50.3 without. Llama 2 MMLU: 41.8 dense, 38.96 sparsified, 38.63 with predictors. Phi-2 MMLU falls from 57 to 54.3 after relufication with distillation, and to about 52 with predictors.
- **Speculative decoding (λ=4):** 1.4× speedup, close to the original 1.58×.
- **Long generation:** 1000 tokens, no thermal throttling observed. Nucleus sampling did not hurt.

### 1.6 Limitations and future work the authors state
- Power and thermal behaviour not analysed systematically. The sparse model draws less power but, because it runs longer, **uses more total energy** than a dense model of similar size; quantifying this is future work.
- **Single-batch only.** "Expanding this to include more complex scenarios like prompt processing and multi-batch inference are valuable areas for further investigation."
- Fixed 50%-memory assumption; exploring other memory sizes is future work.
- Depends on sparsified networks, so Falcon, Llama and Phi need relufication or FATReLU fine-tuning. Extending to non-sparse networks or prompt-dependent loading is future work.
- Better neuron bundling and data structures; better predictor accuracy.
- 4-bit kernels for phones are out of scope (App. F).

### 1.7 Critical view for HBM + HBF datacenter serving [mine]
- **Transfers:**
  - *Co-locate data that is consumed together.* Row–column bundling generalises to any layout that makes one HBF page or stripe hold everything a single compute tile needs.
  - *Read big or don't read.* The cost model's finding that reading extra in big chunks can beat precise small reads maps directly onto HBF page and stripe granularity.
  - *Selective persistence.* Keep attention and embeddings (small, touched every step) in HBM and stream FFN/expert weights from flash. This is essentially the H^3 split refined by tensor type.
  - *Speculative decoding helps.* It raises reuse per flash read.
- **Breaks at high batch:** the sparsity savings disappear because the union of active neurons across a batch approaches dense. The paper gives indirect evidence. Its window over only 5 tokens already needs 24% of OPT's FFN, against about 2.4% of FFN neurons loaded incrementally per token. A batch of 64–256 unrelated sequences is a union over many more contexts, so FFN-neuron sparsity reads offer little benefit [mine, extrapolated]. The sliding window is a per-sequence cache; with many independent sequences it becomes an HBM-resident cache of hot neurons.
- **Not addressed:** flash-device physics (tR, planes, ECC, read disturb), the KV cache in flash (only a footnote saying it could be held there), prefill, and multi-tenant I/O contention.

### 1.8 BibTeX
```bibtex
@inproceedings{alizadeh2024llmflash,
  title     = {{LLM} in a flash: Efficient Large Language Model Inference with Limited Memory},
  author    = {Alizadeh, Keivan and Mirzadeh, Seyed Iman and Belenko, Dmitry and Khatamifard, S. Karen and Cho, Minsik and Del Mundo, Carlo C. and Rastegari, Mohammad and Farajtabar, Mehrdad},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  editor    = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  pages     = {12562--12584},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  month     = aug,
  year      = {2024},
  url       = {https://aclanthology.org/2024.acl-long.678},
  note      = {arXiv:2312.11514}
}
```
(Lincoln's reference gives the editors only as "L.-W. Ku, A. Martins, and V. Srikumar"; I expanded the first names — VERIFY. Add `doi = {10.18653/v1/2024.acl-long.678}` only after checking it.)

---

## 2. AiF: Accelerating On-Device LLM Inference Using In-Flash Processing

### 2.1 Bibliographic
- **Title:** "AiF: Accelerating On-Device LLM Inference Using In-Flash Processing"
- **Authors:** Jaeyong Lee (SNU), Hyeunjoo Kim (SNU), Sanghun Oh (SNU), Myoungjun Chun (Soongsil Univ.), Myungsuk Kim (Kyungpook National Univ.), Jihong Kim (SNU; corresponding author).
- **Venue:** Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA '25), June 21–25, 2025, Tokyo, Japan. ACM, New York. 15 pages, **pp. 529–543**. **DOI 10.1145/3695053.3731073.** ISBN 979-8-4007-1261-6/25/06. Licence CC-BY 4.0.

### 2.2 Problem
SSD offloading for on-device LLMs is limited by read bandwidth: decode arithmetic intensity is about 1–2 ops/byte, and the whole model is read per token. SSDs give 4–8 GB/s against DRAM's 80–100 GB/s, and ≥3 tok/s for models over 30B needs "nearing 100 GB/s".
- In-storage processing in the controller is capped by flash-channel bandwidth. Up to 8 channels × 1.6–2.4 GB/s gives 12.8–19.2 GB/s.
- Naive in-flash processing is still capped by array bandwidth. A 1 TB SSD with 16 chips gives 25.6 GB/s, which means less than 0.85 tok/s for a 30B model.
- In-flash processing also needs **on-chip ECC**, because LLMs are very error-sensitive: LLaMA-3 8B INT8 loses more than 60% accuracy at RBER 1e-7, while NAND is typically above 1e-3. Strong ECC at internal bandwidth is too expensive: a BCH decoder correcting 50 bits per 1 KiB at 102.4 GB/s needs 40.12 mm² and 10.694 W, against a consumer SSD power budget of 6–8 W.

### 2.3 Core mechanism
- **In-flash GEMV.** Each "AiFChip" has product elements (INT8 multipliers plus an adder tree) and a compact ECC decoder, ECCLITE. Matrices are split into equal sub-matrices stored across all 16 chips. The controller gathers the partial output vectors and returns them to the host. Only GEMV is offloaded; norms, activations and attention (MHA) stay on the host because the KV cache lives in host DRAM and changes constantly.
- **Charge-recycling read (cr-read), the latency fix.** A conventional read is precharge + evaluate + discharge. For back-to-back reads of successive wordlines in the *same block*, the next wordline goes from V_PASS down to V_REF and the previous one from V_REF (3.5 V) back up to V_PASS (6 V). Other wordlines stay at V_PASS and the bitlines are only topped up, so both precharge and discharge are skipped. The cost is tR = tRECY + tEVAL, with the recycling phase adding 6.04 µs. Measured results: **tR drops 64%** (the modelled chip goes from 28 µs to 9.7 µs), giving **6.4 GB/s per chip, 2.8× a conventional chip**. **Read energy falls 72.1%**, from 18.278 to 5.098 pJ/bit. The condition is that reads to the same block arrive consecutively with no gap. Data layout: each matrix is placed within the same block(s) so the whole matrix can be read by consecutive cr-reads. Weights are write-once/read-many, so this layout is static. Feasibility is claimed via the existing X-decoder voltage control and flash-scheduler timer code.
- **Bias-error encoding (be-enc), the reliability and latency fix.** The TLC V_TH-state Gray coding is changed from (2,3,2) sensings per LSB/CSB/MSB to **(1,3,3)**. The LSB page then needs one sensing, is "SLC-equivalent" in speed, and errs only at V_REF4 (between P3 and P4, the most stable region). **LSB errors fall by 80%** (Fig. 13: max bits per 1 KiB at 4K P/E + 1-year retention goes from 49 to 9). The Introduction instead says "reduces bit-error rates of LSB page by 87.5%", so the paper is internally inconsistent. CSB and MSB errors rise (figure labels about 43% and 38%). Model parameters are stored **only on LSB pages**; CSB and MSB pages of the same "IFP blocks" hold ordinary data. be-enc is applied per block (IFP vs non-IFP blocks). Two supporting observations: (1) raising tR from 40 to 80 µs does not hurt user-visible SSD bandwidth, because PCIe is the bottleneck; (2) modern SSD ECC has a large margin, e.g. 2 KB parity per 16 KiB page and LDPC soft decoding.
- **Resulting ECC:** ECCLITE is a set of BCH decoders correcting **10 bits per 1 KiB at 6.4 GB/s per chip**. At equal throughput it uses 15.01× less area and 14.83× less power than the 50-bit baseline.
- **Page-buffer reuse.** A TLC page buffer has at least 4 latches. LSB-only reads need one, so two latches are reused to hold the input vector (typically 8–32 KiB) and no extra SRAM is required.
- **On-chip flow.** The controller sends a GEMV command (vector dimension, block ID, offset, page count). The input vector is loaded into the page buffers, the planes stream the matrix, ECCLITE corrects it, and the PEs multiply-accumulate. Results go to an output FIFO, which the controller polls round-robin through the status register.
- **Overhead (45 nm, 200 MHz, Design Compiler):** 0.209 mm², about 0.2% of the flash die; 51.68 mW, of which more than 87% is ECC. Breakdown: ECCLITE 0.167 mm² / 45.1 mW; PEs 0.026 mm² / 3.98 mW.
- **System integration:**
  - *Parallel host/SSD scheduling.* Head-level parallelism lets the host start MHA for head i while QKV for later heads is still being generated. Tensor-level parallelism splits the FFN matrix between host and SSD; host-resident slices are chosen during prefill and bounded by host memory.
  - *Prefill stays on the host*; only decode is offloaded.
  - *NVMe extensions.* `aif_post` stores a matrix with the in-flash layout on sequential LBAs; `aif_gemv` issues a GEMV. The host uses a raw block device or separate namespace, reached via SPDK or ioctl, with a tensor-to-LBA table.
  - *Garbage collection.* When GC runs in an IFP block, page copies preserve the original LSB page order.
- **Read disturb:** not discussed. P/E and retention are used only to size the ECC. **Endurance:** not analysed beyond the 4K P/E + 1-year retention worst case.

### 2.4 Flash parameters (Table 2 and text)
- 1 TB TLC SSD: **8 channels × 2 chips/channel × 4 planes/chip, 16 KiB page**.
- tR with (2,3,2) coding: LSB 37 µs, CSB 46 µs, MSB 37 µs. With (1,3,3): LSB 28, CSB 46, MSB 46 µs. With cr-read: **9.7 µs**. All measured on a real TLC chip, except cr-read, which is simulated.
- PCIe 4.0 ×4, 8.0 GB/s external. ONFI flash channel 2.0 GB/s.
- Per-chip internal bandwidth: 1.6 GB/s conventional (AiF−−), **6.4 GB/s** with cr-read + be-enc. SSD total: 25.6 GB/s → **102.4 GB/s**.
- Error characterisation: 160 real 3D TLC chips, 120 blocks each, 3,686,400 wordlines (11,059,200 pages), JEDEC procedures. Parity example: 2 KB per 16 KiB. The 28 µs single-sensing read and 24.22 mW read power match their SPICE model within 2.9% and 0.9%.
- cr-read validation: SPICE (Cadence Spectre) with a BSIM model fitted to real charge-trap-flash cells (8.02% drain-current mismatch), plus a fabricated 9×9 CTF cell array test that measures bitline current. That chip has no sense amplifiers.

### 2.5 Evaluation and headline results
- **Method:** NVMeVirt (SSD emulator) extended with aif_post/aif_gemv, connected to llama.cpp through a modified NVMe driver and ioctl. The virtual SSD **returns dummy output vectors** and models only the timing.
- **Systems compared:**
  - In-Memory: 128 GB DDR5, up to 86.4 GB/s (86.5 in another sentence), i9-14900KS.
  - Memory+SSD: 8 GB DRAM + 1 TB SSD.
  - AiF: 8 GB + AiFSSD.
  - AiF−−: no cr-read or be-enc, ECC assumed to work anyway.
- **Models:** all INT8. LLaMA2-7B, LLaMA3-8B, Falcon-11B, LLaMA2-13B, Mixtral-8x7B (14.7 GiB read per token of 46.2 GiB total), GPT-NeoX-20B, Falcon-40B, LLaMA3-70B.
- **Throughput (tok/s; In-Memory / Mem+SSD / AiF−− / AiF), Fig. 16:**

  | Model | In-Memory | Mem+SSD | AiF−− | AiF |
  |---|---|---|---|---|
  | LLaMA2-7B | 10.1 | 9.5 | 9.8 | 13.1 |
  | LLaMA3-8B | 9.4 | 9.2 | 9.3 | 12.9 |
  | Falcon-11B | 6.5 | 1.64 | 4.3 | 9.4 |
  | LLaMA2-13B | 5.5 | 0.7 | 2.9 | 7.7 |
  | GPT-NeoX-20B | 3.4 | 0.23 | 1.7 | 5.7 |
  | Falcon-40B | 1.8 | 0.1 | 0.74 | 2.7 |
  | Mixtral-8x7B | 5.3 | 0.33 | 1.9 | 6.2 |
  | LLaMA3-70B | 1.2 | 0.06 | 0.46 | 1.6 |

  Average speedups: **14.6× over Mem+SSD**, **1.4× over In-Memory**, 2.67× over AiF−−, and AiF−− is 4.59× over Mem+SSD.
- **Energy** (Falcon-40B, 512 tokens, normalised to In-Memory): AiF 0.93, AiF without cr-read 1.8. The In-Memory baseline is modelled with an NPU (1.4 TOPS/W) and LPDDR5 (7 pJ/bit). Other energy assumptions: flash channel 5.8 pJ/bit, PCIe 7.5 pJ/bit.
- **Capacity scaling (1 / 2 / 4 TB):** GPT-NeoX-20B 5.7 / 8.9 / 13.7; Falcon-40B 2.7 / 4.4 / 7.4; Mixtral 6.2 / 9.7 / 14.7; LLaMA3-70B 1.6 / 2.7 / 4.3. Each doubling of capacity gives only 1.35–1.68×.
- **be-enc cost** (worst case, every block an IFP block): sequential read unchanged at 7.6 GB/s; random-read IOPS down 6.8% (650K → 606K); latency up 9.3% (96 → 105 µs).

### 2.6 Limitations and future work the authors state
- Scaling is sublinear because (i) host vector operations are interleaved with GEMV and (ii) NVMe control overheads (interrupts, DMA, and loading input vectors over channels) grow with chip count. "Various scheduling optimizations on both the host and SSD sides are necessary… left for future work."
- be-enc causes some random-read loss if many blocks are IFP blocks. The authors call this "a reasonable trade-off".
- cr-read works only for consecutive same-block reads with no interval.
- The KV cache must stay in host memory because it is updated frequently, so attention stays on the host.

### 2.7 Critical view for HBM + HBF [mine]
- **Strong transfer candidates:**
  - *cr-read.* HBF streams weights and the shared prefix KV sequentially and predictably. That is exactly the case of consecutive reads within a block where cr-read applies. The −64% tR and −72.1% read energy matter directly: my report cites HBF at about 4× the power per stack, and in AiF's model read energy dominates. No one has yet applied cr-read to an HBF-class device, where the layout would be weights striped block-contiguously per plane.
  - *The ECC-throughput argument.* AiF's point that "ECC overhead scales linearly with throughput" is the key reliability constraint for HBF, which targets far more bandwidth per stack than 102 GB/s. The derived per-GB/s figures in §5.3 show that ECC placement is a first-order power and area question for HBF.
  - *be-enc idea for TLC or MLC HBF.* Store read-hot weights only on the fastest, most reliable page type and put cold or spill data in the other pages. This gives a density/latency trade-off between pure SLC and TLC.
- **On-device only:**
  - *In-flash GEMV* works at batch 1 because GEMV has a very high reduction ratio. At batch B the per-chip work becomes GEMM with B× the MACs. AiF's INT8 PEs (0.026 mm²) would need roughly B× more area and power, and the GPU already has the FLOPs. In a datacenter HBF system the flash should deliver weights to the GPU; in-flash GEMV is not the right target, except possibly for low-batch, latency-critical tenants or MoE experts with few routed tokens.
  - *Host/SSD split scheduling* is specific to a weak host with DRAM bandwidth comparable to the SSD's.
- **Not addressed:**
  - Read disturb. The workload re-reads every weight page on every token; see §5.4.
  - Temperature effects on RBER.
  - End-to-end accuracy with ECCLITE: the simulator returns dummy vectors, and only error statistics were characterised.
  - Batch > 1, prefill in flash, and KV cache in flash.

### 2.8 BibTeX
```bibtex
@inproceedings{lee2025aif,
  title     = {{AiF}: Accelerating On-Device {LLM} Inference Using In-Flash Processing},
  author    = {Lee, Jaeyong and Kim, Hyeunjoo and Oh, Sanghun and Chun, Myoungjun and Kim, Myungsuk and Kim, Jihong},
  booktitle = {Proceedings of the 52nd Annual International Symposium on Computer Architecture},
  series    = {ISCA '25},
  pages     = {529--543},
  address   = {Tokyo, Japan},
  publisher = {Association for Computing Machinery},
  year      = {2025},
  month     = jun,
  isbn      = {979-8-4007-1261-6},
  doi       = {10.1145/3695053.3731073}
}
```

---

## 3. Lincoln: Real-Time 50~100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory

### 3.1 Bibliographic
- **Title:** "Lincoln: Real-Time 50∼100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory"
- **Authors:** Weiyi Sun, Mingyu Gao, Zhaoshi Li, Aoyang Zhang, Iris Ying Chou, Jianfeng Zhu*, Shaojun Wei, Leibo Liu*. Affiliations: School of Integrated Circuits, Tsinghua; BNRist; IIIS Tsinghua; Shanghai Qi Zhi Institute; MetaX Technology. * = corresponding (Leibo Liu, Jianfeng Zhu).
- **Venue:** 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA). **pp. 1734–1750** (first page 1734; the last numbered page in the extracted text is 1750). **DOI 10.1109/HPCA61900.2025.00128.** ISBN 979-8-3315-0647-6. Month: unknown (not in text).

### 3.2 Problem
Consumer devices (DRAM ≤ 64 GB) cannot hold 50–100B+ models (over 100–200 GB), so weights are reloaded from flash on every prefill and on every generated token. On an NPU with an NVMe SSD and 2 LPDDR packages ("Base-2"), weight loading takes 94.0% of generation time (up to 41.8 s/token) and 91.4% of prefill time. Two causes: (1) the flash↔SoC interface is narrow (PCIe ×4 / UFS ~10 GB/s, versus 128+ LPDDR pins); (2) flash **internal** bandwidth is low. Density-oriented arrays take more than 50 µs (up to over 100 µs) per 16 KB page, so even 32 dies × 4 planes give at most 41 GB/s. With unlimited I/O the flash still dominates.

### 3.3 Core mechanism
**Device level: the Lincoln flash die**
- Starts from Kioxia-style XL-Flash (SLC, tR 4 µs, 16 planes, 4 KB page). The plane is shrunk about 2× along the bitline direction and the **plane count doubled to 32 per die**. Result: tR = 3.426 µs, about **1.06 GB/s per plane** and **34 GB/s per die**, or about 500 GB/s for 16 dies.
- **Hybrid bonding (Xtacking-like CMOS-under/over-array).** Page buffers, PHY and other periphery move to a logic die bonded to the array die; row decoders stay on the array. Area efficiency is 74.6%, versus about 55% for XL-Flash. Bonding parasitics add less than 0.1% latency.
- **Power delivery for all-plane parallel reads.** Power domains and pins are replicated, each copy feeding the number of planes the original design allowed. Charge-pump replication costs about 7% of area.

**Architecture level**
1. **Prefill: flash on the LPDDR bus.**
   - Physical arrangement: each LPDDR-style package holds 4 Lincoln flash dies stacked with 4 DRAM dies (two stacks per package), wire-bonded. Each 16-bit LPDDR channel has one DRAM rank and one **flash rank** (2 dies × 8 DQ). DRAM and flash form a flat address space; DRAM is not used as a flash cache. The rank bit is placed at the top address bit so flash and DRAM spaces are contiguous.
   - Meeting DRAM timing (<100 ns) when NAND takes microseconds: a near-PHY **SRAM global buffer (260 KB per die)**. The memory controller issues a prefetch command for a weight *partition*, then polls status registers (using the known tR to time the polls). The NPU reads the buffered data with normal LPDDR accesses.
   - **Double-buffered prefetch:** the next partition is fetched while the NPU computes on the current one. This fully hides flash latency because LPDDR bandwidth is more than 4× below internal flash bandwidth.
   - **ECC moved from the memory controller to near-flash** (before the global buffer). Otherwise the chunk-wise microsecond-latency ECC (1 KB chunks), which mismatches LPDDR's 32/64 B accesses, would force a redundant full-weight LPDDR pass.
   - Opportunistic reads during polling: the NPU starts reading and discards the data if the address check fails.
2. **Generation: near-flash compute.**
   - A 2-element FP dot-product unit ("FP-Dot") at 400 MHz sits in the logic block under each plane; per-plane bandwidth is only 1.06 GB/s. Table I actually lists 16 FMACs + an 8 KB register file per plane, sized to support speculative decoding.
   - Per-plane **BCH(9088,8192,64)** ECC. This is sufficient because SLC is more robust than TLC, so LDPC is unnecessary.
   - The input vector is broadcast from the global buffer; results return to it.
   - **Speculative decoding:** a ~1B draft model predicts ~7 tokens, and the target verifies them in one append-style prefill pass. FP-Dot is replicated for this small batch.
3. **Data layout: one copy of the weights serves both the NPU and near-flash compute.**
   - Capacity is too small for two layouts: 265.5 GB with 2 packages, and keeping two copies would limit models to under 71B.
   - Matrices are split into partitions that fit the aggregate global buffer. Rows are distributed **round-robin across dies**, then round-robin across planes, and stored sequentially in pages. The page address is identical across planes to satisfy multi-plane-operation rules.
   - The NPU fetches rows of a tile from all dies in parallel (full LPDDR channel parallelism). Once a partition is in SRAM, it can be accessed randomly in any order, so the NPU can choose any tiling.
   - Column-major variant: distribute columns across dies and keep rows within a die.
4. **Endurance and read disturb (§IV-F), the only paper of the four that does this:**
   - Even optimistically assuming SLC tolerates 10K reads before RBER exceeds ECC capability, "equivalent to 10K LLM execution iterations", the limit "can be easily exceeded within 2 hours". Footnote: MLC/TLC tolerate 5K–8K read cycles and regular NAND 1K–10K. Statistics for 3D SLC are "rare".
   - Fix: **read reclaim** (re-program the data; P/E resets accumulated read disturb), similar to retention refresh.
   - SLC endures up to 100K P/E, raised to **at least 4M P/E if the retention requirement is relaxed from 1 year to 3 days** with a full refresh every 3 days. That refresh itself costs 609 cycles per 5 years.
   - Wear levelling for static weights is easy (move data during reclaim and cycle through the whole space).
   - Result: with a 5-year warranty, a 5K read-disturb threshold and generation running continuously, the maximum P/E consumption is 310K. It stays below 4M even with thresholds under 1K.
   - Reclaim is spread over iterations (e.g. 1/5K of the model per iteration), costing about 0.4% runtime with a 75 µs program latency.
5. **Control system:**
   - A management layer on the SoC (OS when idle; during inference the NPU kernel maps the flash interface and does address translation, reclaim, wear levelling and GC itself).
   - An operation layer in a flash-side controller (under 1% area).
   - 64-bit transaction commands: 16-bit OP (PageRead / PageProgram / Erase / ComputePartition / MoveResult …), 16-bit buffer address, 32-bit flash address, single- or all-plane.
   - "Lincoln Access Units" in the NPU, similar to H100's TMA, plus a Flash Read Unit in the LPDDR controller. Channel interleaving and caching are disabled for flash-space accesses.
6. **Heat:** HotSpot simulation with flash energy scaled 1–3× gives peak temperature rises of 3.0 / 3.9 / 4.8 / 5.7 / 6.6 K.

### 3.4 Flash and system parameters (Table I)
- **Lincoln flash layer:** SLC, 96 wordline layers, **(4 KB + 448 B ECC) per page**, 768 pages/block, 177 blocks/plane, **32 planes/die**. 4.1484 Gb per plane, 132.75 Gb per die. **tR = 3.426 µs**, 98.0 nJ per page read. Plane area 1.702 mm²; die 72.99 mm²; 1.8 Gb/mm².
  - Sources: NVSim-based 3DFPIM simulator, compensated with XL-Flash data [50]. Real performance "at most 17% worse" because XL-Flash's 4 µs is an upper bound.
- **Logic layer (per plane):** 16 FMACs + 8 KB register file: 0.0708 mm², 6.97 mW. ECC BCH(9088, 8192, 64) at 14.6 Gb/s per plane: 0.1478 mm², 5.24 mW. Global buffer 260 KB per die: 0.4095 mm², 18.4 mW. Near-flash logic power 409.1 mW per die; logic is 7.98% of area. Scaled to 20 nm with a 2× area penalty for fewer metal layers.
- **Program latency:** 75 µs (from [50]).
- **NPU:** 32 TOPS BF16, 4.60 W; 4 MB / 2 MB output/input SRAM.
- **Memory:** LPDDR5X-8533, x8 dies, FR-FCFS. 2 or 4 packages × 4 channels × 2 DRAM dies (+2 Lincoln flash dies).
- **Baseline SSD:** TLC 2 TB, **tR = 56 µs**, 8 ch × 4 dies × 4 planes × 683 blocks × 1536 pages × 16 KB. NVMe PCIe 4.0 ×4 at 8 GB/s; 1.2 GB/s per channel.
- **Cost:** $0.52/GB at SLC density; **$0.72/GB or $1.44/GB** after the yield penalty; 2-package system $191 / $382 (5-year warranty).

### 3.5 Evaluation and headline results
- **Simulators:** ONNXim (cycle-accurate NPU) with Ramulator 2 (LPDDR) and a NAND die model; DRAMPower energy; MQSim for the NVMe SSD with SimpleSSD-style energy; HotSpot for thermals.
- **Models (draft model):** OPT-66B (OPT-125M), LLaMA-65B (NoFT-796M), LAMDA-137B (LAMDA-2B). Worst reported acceptance rates. ShareGPT dataset, 90% tail latency.
- **Configurations:** Base (NVMe SSD), BaseS (Base + speculative decoding), Lincoln; each with 2 or 4 packages.
- **Prefill:** Lincoln-2 is 11.44× / 11.46× faster than Base-2 / BaseS-2; Lincoln-4 is **13.23×** / 13.24× faster than Base-4 / BaseS-4. At 4 packages prefill becomes compute-bound. Prefill energy falls 34.5% / 35.6% (Lincoln-2) and 30.1% / 31.2% (Lincoln-4).
- **Generation:** Lincoln-2 / Lincoln-4 are 158.4× / **254.1×** faster than Base-2 / Base-4, and 47.8× / 76.6× faster than BaseS. Generation energy falls 9.11× / 3.66× (Lincoln-2 vs Base-2 / BaseS-2) and 8.46× / 3.10× (Lincoln-4).
- **Absolute latency (Table II), prefill / generation per token:**
  - OPT-66B: 1.633 s / 0.085 s (L-2); 1.315 s / 0.046 s (L-4)
  - LLaMA-65B: 1.700 / 0.163 (L-2); 1.269 / 0.087 (L-4)
  - LAMDA-137B: 3.772 / 0.264 (L-2); 2.667 / 0.142 (L-4)
  - Targets from prior work: 4.0 s prefill / 0.2 s generation, or 5.0 s prefill from industry.
- **Ablation, prefill (LLaMA-65B):**
  - Without double buffering and near-flash ECC, overheads are 56.0% (L-2) and 37.6% (L-4), dominated by ECC.
  - Double buffering alone gains only 10.98% / 7.36%.
  - Near-flash ECC alone leaves 23.2% / 15.6% overhead.
  - Both together remove almost all of it.
- **Ablation, generation (2 packages):**
  - Speculative decoding alone: 2.61× (LLaMA) and 3.226× (LAMDA), with draft-model overhead of 14.0% / 15.9%.
  - Near-flash compute alone: 3.60× / 4.05×.
  - Both: **7.99× / 10.63×**.
  - What remains is mostly attention over the KV cache.

### 3.6 Limitations and future work the authors state
- "Lincoln may also utilize dynamic sparsity for better performance, which we defer to future work."
- Performance could be up to 17% worse than simulated (tR bounded by XL-Flash).
- Read-disturb statistics for 3D SLC are "rare", so the threshold is an assumption (1K–10K).
- Near-flash logic is too weak for prefill (over 5 s on every benchmark for Lincoln-2) and "cannot perform attention due to Flash endurance issues": the KV cache is written every step. Attention stays on the NPU with DRAM.
- The heat analysis relies on simulated energy (stress-tested at 1–3×).
- No dedicated limitations section.

### 3.7 Critical view for HBM + HBF [mine]
This is the most directly relevant paper to HBF, even though it targets consumer devices.
- **Transfers almost directly:**
  - *Flash behind a DRAM-class protocol:* SRAM staging buffer, prefetch command + status polling, and double-buffered partitions. This is the blueprint for any HBF that presents an HBM-like interface to the GPU; my report's H^3 model assumes a "latency-hiding buffer" of 40 MB.
  - *Buffer sizing check.* Lincoln's numbers pass a Little's-law check: 34 GB/s × 3.426 µs ≈ 114 KiB per die, ×2 for double buffering ≈ 228 KiB, against the 260 KB they provision. The same check (buffer ≈ 2 × bandwidth × tR, plus the ECC decode latency) should be run for H^3's 40 MB at HBF bandwidth and SLC tR.
  - *ECC placement.* The ablation shows ECC, not tR, dominates overhead when decoding happens on the far side of the interface. For HBF the ECC decoder should sit on the base/logic die, before the HBM-like PHY. Otherwise the parity (448/4096 ≈ 10.9% for Lincoln's page) consumes link bandwidth and the decode lands on the GPU.
  - *Array shrinking, many planes, hybrid bonding, and power-domain replication* are exactly the device knobs HBF vendors must use to reach HBM-class bandwidth. Lincoln gives numbers: 1.06 GB/s per plane, 32 planes, 34 GB/s per die, and the power-pin issue.
  - *Round-robin row striping across dies and planes, with identical page addresses for multi-plane reads,* so one copy of the weights serves both consumers. For HBF this becomes striping across the pseudo-channels and dies of a stack.
  - *Read-disturb analysis and read reclaim.* Critical for my report; see §5.4.
- **On-device only:**
  - *Near-flash FP-Dot for decode* is valuable at batch 1 (and for ~7-token speculative verification). At datacenter batches the GPU needs the raw weights, which puts us back on Lincoln's "prefill path" (flash→processor streaming) for every step.
  - *Sharing LPDDR channels between DRAM and flash ranks* is a consumer-cost trick. In HBM + HBF the analogous design question is pooled vs partitioned links, which my report already studies.
- **Not addressed:**
  - Batch > 1 serving; multi-tenant interference between reclaim writes and reads.
  - The KV cache in flash; the shared prefix cache is explicitly not in flash.
  - How read-disturb thresholds behave at high temperature; heat was checked only as a temperature rise.
  - Whether read disturb is counted per block read or per page read; they equate one "read cycle" with one LLM iteration.

### 3.8 BibTeX
```bibtex
@inproceedings{sun2025lincoln,
  title     = {Lincoln: Real-Time 50{$\sim$}100{B} {LLM} Inference on Consumer Devices with {LPDDR}-Interfaced, Compute-Enabled Flash Memory},
  author    = {Sun, Weiyi and Gao, Mingyu and Li, Zhaoshi and Zhang, Aoyang and Chou, Iris Ying and Zhu, Jianfeng and Wei, Shaojun and Liu, Leibo},
  booktitle = {2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
  pages     = {1734--1750},
  publisher = {IEEE},
  year      = {2025},
  isbn      = {979-8-3315-0647-6},
  doi       = {10.1109/HPCA61900.2025.00128}
}
```
(Last page taken from the final page number in the extracted text, 1750 — VERIFY on IEEE Xplore.)

---

## 4. Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM

### 4.1 Bibliographic
- **Title:** "Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM"
- **Authors:** Zhongkai Yu†, Shengwen Liang†, Tianyun Ma, Yunke Cai, Ziyuan Nan, Di Huang, Xinkai Song, Yifan Hao, Jie Zhang, Tian Zhi, Yongwei Zhao, Zidong Du, Xing Hu*, Qi Guo, Tianshi Chen († equal contribution; * corresponding). Affiliations: SKL of Processors, ICT-CAS; UCAS; USTC; Peking Univ.; Shanghai Innovation Center for Processor Technologies; Cambricon Technologies.
- **Venue:** the text read is **arXiv:2409.15654v1 [cs.AR], 24 Sep 2024**. The text itself does not name the conference. The file name says MICRO'24 (57th IEEE/ACM MICRO, 2024). **Pages and DOI of the MICRO version: unknown — VERIFY on IEEE Xplore / ACM DL.**

### 4.2 Problem
Single-batch, on-device decode of 70B models. INT8 Llama-70B needs 70 GB, and decode arithmetic intensity is about 2 ops/byte: generating one token takes about 0.14 Tera-ops but more than 70 GB of memory access. UFS 4.0 at 4 GB/s allows only 0.06 tok/s for Llama2-70B INT8, while 3–10 tok/s is needed. Offloading also moves data flash→DRAM→NPU, more than 3× the transfer volume. Existing on-die ISC (OptimStore, BeaconGNN) fails for two reasons:
- **Reduction ratio.** LLM GEMV is about 4096:1 (the smallest Llama2-7B matrix is 4096×4096), roughly 100× higher than prior ISC workloads, so flash-channel utilisation falls below 10%.
- **No on-die ECC.** Unprotected flash errors cut accuracy by more than 70%.

### 4.3 Core mechanism
- **Chiplet hybrid.** The NPU has an integrated flash controller and connects to a dedicated flash chip over a die-to-die (D2D) chiplet link instead of UFS. Weights are in flash; the KV cache is in DRAM (under 700 MB for 70B at 1000 tokens). Operation mapping:
  - weight GEMVs are co-computed by NPU and flash;
  - KV-cache matrix operations run only on the NPU;
  - KV loads involve the NPU and DRAM.
  - The SFU (softmax, sin/cos, ReLU) is only on the NPU.
- **Flash die.** One **shared Compute Core per die**, time-shared by the 2 planes, containing PEs, input/output buffers and the Error Correction Unit. It is shared (not per plane) to save area and to limit heat, which raises error rates. There is also a "Compute Control" (arbitrary GEMV shapes) and a "Slice Control".
- **Read-compute request.**
  1. The input vector is sent over the channel to the Compute Core input buffer.
  2. A page of weights goes array → data register.
  3. Data register → cache register, which pipelines the next read.
  4. The PEs perform GEMV.
  5. The result returns over the channel.
  - The atomic tile equals one page.
  - Compute is matched to array speed: for tR 20 µs and a 16 KB page, 32K INT8 ops in 20 µs is 1.6 GOPS, about 2 MACs.
- **Slice Control (channel latency hiding).** Read-compute traffic uses the channel only lightly (≤6%). While one plane serves a read-compute request, the *other plane* serves a normal read that ships raw weights to the NPU. That read is **sliced into sub-page pieces** so it fits into channel bubbles without blocking later read-compute requests.
- **Hardware-aware tiling.** The channel carries W_req + channel_num × H_req, because the input vector is broadcast to all cores on a channel. With H_req × W_req = channel_num × ccore_num × page_size, the AM-GM inequality gives the optimal **H* = sqrt(ccore_num × pagesize)** and **W* = channel_num × sqrt(ccore_num × pagesize)**. The NPU/flash split is α = t_r / (t_r + t_rc), where t_rc = tR + W_req / (channel_num × bw_channel) and t_r = pagesize / ((1 − rate_rc) × bw_channel). Example: Cambricon-LLM-S has an optimal tile of 256×2048, consistent with ccore_num = 4 and a 16 KB page.
- **Outlier-oriented on-die "ECC".**
  - Per 16 KB page (16,384 INT8 values) the top 1% (163 values) are protected. Each stores a 14-bit address plus a 5-bit Hamming code on the address, and N=2 copies of the value; decoding is a bitwise majority vote with the stored value. The protected-outlier flip rate is ≈3x² (3e-8 at x = 1e-4).
  - A threshold (the smallest protected value, stored in 9 copies) triggers **clipping to zero** of any unprotected value above it, which would have to be a bit-flip.
  - Size: 8×9 + (14+5+16)×163 = **722 B**, within the 1664 B spare area of a 16 KB page.
  - It does **not** protect small or medium values. Accuracy collapses above RBER 8e-4.
- **Page granularity** is argued to be harmless: weight reads are "sequential, extensive, and predictable", and the smallest Llama2-7B INT8 matrix is 16 MB.
- **Read disturb and endurance: not discussed.** The paper says writes are irrelevant because inference "solely involve[s] reading weight data from flash without any writing". Retention errors are said to dominate, with RBER 1e-4 for new 3D TLC after hours and more than 1e-2 with aging.

### 4.4 Flash parameters (Table II)
- 2 dies per chip, 2 planes and 1 compute core per die, 1000 MT/s 8-bit channel bus (1 GB/s per channel [mine: 1000 MT/s × 8 bit]), 16 KB page, **tR = 30 µs**. INT8.
- S: 8 channels × 2 chips. M: 16 × 4. L: 32 × 8.
- NPU: 16×16 systolic array, 2 TOPS at 1 GHz. LPDDR5X at about 40 GB/s, holding only the KV cache (700 MB).
- Density claims (Table I): SK hynix 300+ layer flash 20.00 Gb/mm²; Samsung 280-layer 28.50; DDR 0.30; LPDDR 0.31. "A typical 200GB NAND flash chip occupies about 64 mm²."
- Compute Core, TSMC 65 nm (Table IV): ECC unit 496.4 µm² / 0.4 µW; PEs 562.0 µm² / 343.6 µW; I/O buffers 58755.1 µm² / 1591.7 µW; "Total Compute Core" 39813.5 µm² / 1935.6 µW; overhead **1.2% area, 4.5% power**. The table is internally inconsistent: the buffer area alone exceeds the listed total. The buffers together are 2 KB.

### 4.5 Evaluation and headline results
- **Method:** SSDsim extended with a Read-Compute command, plus a custom cycle-accurate NPU simulator in C. Accuracy: SmoothQuant INT8, with PyTorch-injected flash error models, on HellaSwag, ARC and WinoGrande.
- **Baselines:** FlexGen-SSD and FlexGen-DRAM on an AMD EPYC 7742 + A100 80 GB + Intel NVMe + 128 GB DRAM server, INT8, OPT only. MLC-LLM on a Snapdragon 8 Gen 2 with 4-bit RTN.
- **OPT decode tok/s (6.7B / 13B / 30B / 66B):**
  - S: 3.6 / 1.9 / 0.8 / 0.4
  - M: 10.96 / 4.68 / 2.50 / 1.15
  - L: **36.34** / 14.2 / 7.6 / **2.59**
  - FlexGen-SSD: 0.8 / 0.4 / 0.2 / 0.1
  - FlexGen-DRAM: 3.5 / 2.0 / 0.8 / 0.4
  - L is 22.1× FlexGen-SSD and 6.8× FlexGen-DRAM on 66B, and 44.8× / 10.3× on 6.7B. S is 8.9× FlexGen-SSD on 6.7B.
- **Llama2 (7B / 13B / 70B):**
  - S: 3.55 / 1.9 / 0.3
  - M: 10.4 / 4.7 / 1.0
  - L: 34.0 / 14.0 / **3.4** (the abstract says 3.44)
  - MLC-LLM: 7.58 / OOM / OOM
- **W4A16** vs W8A8: +85.3% (S) and +47.9% (L) on average.
- **Ablation (S configuration):**
  - Read slicing: 1.6–1.8× speedup; channel utilisation +31.6 to +41.4 points (about 50% → 79–91%).
  - Hardware-aware tiling: 1.3–1.4× speedup; utilisation +76.2 to +88.9 points (about 2–3% → 79–91%).
  - Optimal tile 256×2048 beats 128×4096 by 17.5% and 4096×128 by 24.7%.
- **ECC (OPT-6.7B):** without it, accuracy starts falling at 1e-5 and reaches about 40% of original at 2e-4. With it, 92–95% of original accuracy is kept at 2e-4 ("2.3× protection"). Accuracy is unusable above 8e-4.
- **Scalability:**
  - More chips per channel (1→128, 8 channels): gains saturate because the weights cannot be spread over all chips.
  - More channels (1→64, 4 chips/channel): steady gains, limited by chiplet geometry and the NPU buffer.
- **Data and energy per token:** S moves 9.7–11.6× less data than FlexGen-SSD and uses 67% of its data-transfer energy.
- **Cost for 70B INT8:** $43.67 (2 GB DRAM $4.87 + 80 GB flash $38.80) vs $194.68 (80 GB DRAM). Chiplet cost is estimated at under 15% of raw chip cost, not exceeding $100.

### 4.6 Limitations and future work the authors state
- ECC protects only outliers; it fails above RBER 8e-4 because medium and small values are unprotected.
- Scaling limits: too many chips per channel leaves chips idle; channel count is limited by chiplet technology, geometry and NPU buffer area.
- Extra costs are not in the cost table (on-die logic and chiplet packaging).
- No explicit future-work section. The paper says it would benefit from more aggressive 4- or 2-bit quantization.

### 4.7 Critical view for HBM + HBF [mine]
- **Transfers:**
  - *Using idle channel/link slots to stream raw weights to the main processor while in-memory compute proceeds.* Slice Control plus the α split is a general bandwidth-balancing rule between a near-data unit and a host compute unit.
  - *Sub-page slicing of transfers* to avoid head-of-line blocking on a shared channel. This is relevant if HBF pseudo-channels carry mixed traffic, e.g. weight streaming plus reclaim writes plus prefix-cache reads.
  - *The analytic tile-shape optimisation* (a function of page size, channel count and core count) is the right style of model for HBF striping.
- **On-device or questionable:**
  - *On-die GEMV at batch 1* (as with AiF).
  - *The outlier-only "ECC"* is not a real ECC. It fails at 8e-4, which is below the >1e-3 RBER AiF reports as typical for NAND. It would be unacceptable in a datacenter SLA unless paired with a real code. It also assumes INT8 value semantics, so it does not protect FP8/BF16 exponent bits or KV entries.
  - *Evaluation:* the baselines are a server A100 running FlexGen (an odd reference for edge) and a phone running MLC-LLM. The flash-side timing model is SSDsim.
- **Not addressed:** read disturb, retention refresh, wear, temperature (except as the motivation for a shared core), prefill, batch > 1, and the KV cache in flash.

### 4.8 BibTeX
```bibtex
@misc{yu2024cambriconllm,
  title         = {{Cambricon-LLM}: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70{B} {LLM}},
  author        = {Yu, Zhongkai and Liang, Shengwen and Ma, Tianyun and Cai, Yunke and Nan, Ziyuan and Huang, Di and Song, Xinkai and Hao, Yifan and Zhang, Jie and Zhi, Tian and Zhao, Yongwei and Du, Zidong and Hu, Xing and Guo, Qi and Chen, Tianshi},
  year          = {2024},
  eprint        = {2409.15654},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  note          = {arXiv:2409.15654v1}
}
% If citing the MICRO'24 version, convert to @inproceedings with
% booktitle = {2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)}
% and fill pages/doi ONLY after checking IEEE Xplore -- they are not in the text read.
```

---

## 5. Cross-paper observations and gaps relevant to HBM + HBF datacenter inference

### 5.1 What everybody assumes that HBF breaks
- **Batch 1.**
  - All four papers are built on decode arithmetic intensity of about 1–2 ops/byte [AiF §3.1; Cambricon §III-A] or on explicitly single-batch operation [LLM-in-Flash §8; Lincoln's speculative verify of ~7 tokens is the only step beyond it].
  - In datacenter serving, the whole weight set is read once per step *regardless of batch*. Flash bandwidth therefore sets the per-step floor, and batch size (limited by HBM capacity) sets throughput. My report's capacity framing is correct.
  - Consequence [mine]: the in-flash and near-flash GEMV of AiF, Lincoln-decode and Cambricon loses its purpose at batch B ≫ 1. It becomes GEMM needing B× the MACs, and the GPU has them. The part of those papers that *does* transfer is how they obtain and protect internal bandwidth, not where they compute.
- **The KV cache is always in DRAM** [AiF §5.1; Lincoln §IV-A; Cambricon §IV-A], because it is written every step. None of the papers puts a read-only *shared prefix* KV in flash. That is H^3's distinguishing move, and it has no prior-art evaluation in this set.
- **Prefill** runs on the host/NPU in every paper. Only Lincoln feeds weights to a strong processor through a DRAM-class interface (the LPDDR path), which is the closest analogue to a GPU reading HBF.

### 5.2 Flash latency hiding: techniques catalogue
| Technique | Paper | HBF relevance [mine] |
|---|---|---|
| Large contiguous reads; read-and-discard beats precise small reads; 32-thread parallel reads | LLM in a Flash §2.2, §4.1 | Striping and page-size choice for HBF; fetch whole tiles |
| Row–column bundling (co-locate up-proj column with down-proj row) | LLM in a Flash §3.2 | Layout so one HBF page or stripe = one compute tile |
| cr-read (skip precharge/discharge for consecutive same-block reads): tR −64%, energy −72.1% | AiF §4.2 | **Not applied to HBF by anyone.** Sequential weight and prefix streaming fits the condition; the energy reduction addresses HBF's power problem |
| LSB-only placement with (1,3,3) coding (SLC-speed reads from TLC) | AiF §4.3 | HBF density/latency option between SLC and TLC |
| Array shrinking + 32 planes + hybrid bonding + power-domain replication: 3.426 µs, 34 GB/s/die | Lincoln §IV-B | The device recipe HBF must follow; gives per-plane numbers for my model |
| SRAM staging buffer + prefetch command + status polling + double-buffered partitions | Lincoln §IV-C | Blueprint for H^3's latency-hiding buffer; buffer ≈ 2 × BW × (tR + ECC latency) |
| Idle-plane/idle-channel weight streaming + sub-page read slicing | Cambricon §IV-C | Mixed-traffic scheduling on HBF channels (weights + prefix + reclaim) |
| Analytic tile shape from page size, channels and cores (AM-GM) | Cambricon §V-A | Model for HBF stripe geometry |
| Head-level/tensor-level overlap of host and flash compute | AiF §5.1 | Overlap of the HBM and HBF bandwidth tiers (H^3 pooled-bandwidth model) |
| Speculative decoding to reuse each weight read over several tokens | LLM in a Flash §5.2 (1.4×); Lincoln (2.61–3.226×) | Reduces flash reads per useful token even at moderate batch |

### 5.3 ECC / reliability treatment
- **Sensitivity (the papers disagree):** AiF finds LLaMA-3 8B INT8 loses more than 60% accuracy at RBER 1e-7, injecting bit-flips only into QKV and FFN weights. Cambricon finds OPT-6.7B starts degrading at 1e-5 and reaches about 40% of original accuracy at 2e-4. Both say unprotected NAND (RBER 1e-4 to above 1e-3) is unusable [AiF Fig. 4b; Cambricon Figs. 3b, 10]. Either way: **HBF cannot hand raw NAND data to the GPU**, so a full ECC is mandatory.
- **ECC cost scales with throughput** [AiF §3.3]. Figures, the last three columns derived by me as ratios of the papers' numbers:

  | Design | Correction | Throughput | Area | Power | mm² per GB/s | mW per GB/s |
  |---|---|---|---|---|---|---|
  | AiF baseline BCH (Fig. 5) | 50 bit / 1 KiB | 102.4 GB/s aggregate | 40.12 mm² | 10.694 W | ≈0.39 | ≈104 |
  | AiF ECCLITE (45 nm) | 10 bit / 1 KiB | 6.4 GB/s per chip | 0.167 mm² | 45.1 mW | ≈0.026 | ≈7.0 |
  | Lincoln per-plane BCH(9088,8192,64) (20 nm-scaled) | 64 bit / 1 KB data | 14.6 Gb/s ≈ 1.825 GB/s | 0.1478 mm² | 5.24 mW | ≈0.081 | ≈2.9 |

  [mine] Linear extrapolation to 1 TB/s of HBF read bandwidth: roughly 3–7 W for a weak, SLC-grade BCH and roughly 100 W for a TLC-grade 50-bit BCH. The per-stack ECC power budget therefore depends strongly on cell type and retention policy. My report's power comparison (the 21% saving) should include this term, or at least state that it is excluded.
- **Parity overhead:** 448 B / 4 KB ≈ 10.9% [Lincoln Table I]; 2 KB / 16 KiB = 12.5% [AiF]; spare area 1664 B / 16 KB ≈ 10.2% [Cambricon]. If ECC is decoded on the GPU side, about 10–12.5% of HBF link bandwidth carries parity [mine]. Lincoln's ablation (§V-C) shows that off-device ECC was the dominant overhead (56.0% / 37.6% before optimisation), which argues for base-die ECC in HBF.
- **Cheaper reliability by construction:** be-enc lowers LSB errors 80% (87.5% in the intro), enabling a 10-bit BCH [AiF]. Choosing SLC allows BCH instead of LDPC [Lincoln]. Outlier-only protection is too weak (fails at 8e-4) [Cambricon].
- **Gap:** none of the papers evaluates reliability at datacenter temperatures (HBF sits beside a hot GPU), or accuracy with the *actual* ECC in the loop (AiF returns dummy vectors; Lincoln assumes the BCH is sufficient). None considers corruption of FP8/BF16 exponent bits or of KV-cache contents.

### 5.4 Read disturb: the most important gap for my report
- Only **Lincoln** analyses it. Every weight page is re-read on every decode iteration. With a read-disturb threshold of 1K–10K reads (their footnote gives 5K–8K for MLC/TLC; 10K is their optimistic SLC assumption), the limit "can be easily exceeded within 2 hours". The fix is **read reclaim**, i.e. re-programming, which **consumes P/E cycles**. Lincoln keeps this affordable only by (a) SLC with 100K P/E, (b) relaxing retention to 3 days to reach at least 4M P/E, and (c) wear levelling, and it still needs up to 310K P/E over 5 years at a 5K threshold.
- AiF, Cambricon-LLM and LLM-in-a-Flash do not mention read disturb. Cambricon says inference involves no writes at all.
- **My report says "in steady-state decode, flash is never written"** and checks only retention refresh (the 995-year figure). Read reclaim contradicts the premise. Back-of-envelope [mine], following Lincoln's convention that one decode step = one read cycle of every weight/prefix block:
  - reclaims per block per day = 86400 / (T_step × R), where T_step = decode step time and R = read-disturb threshold;
  - lifetime (years) = PE_budget × T_step × R / (86400 × 365).

  | T_step | R | reclaims/day | life @100K P/E | life @4M P/E |
  |---|---|---|---|---|
  | 20 ms | 5K | 864 | 0.32 yr | 12.7 yr |
  | 50 ms | 5K | 345.6 | 0.79 yr | 31.7 yr |
  | 50 ms | 10K | 172.8 | 1.59 yr | 63.4 yr |
  | 100 ms | 1K | 864 | 0.32 yr | 12.7 yr |

  Also [mine]: rewriting the whole static footprint once per T_step × R seconds costs write bandwidth. For the report's 165.8 GB at T_step = 50 ms and R = 5K that is 165.8 GB / 250 s ≈ 0.66 GB/s of background programming per system. This is small against read bandwidth, but it is not zero, and it creates exactly the GC-free-but-wear-concentrated traffic the report studies. Batching does not reduce it: it depends on step rate, not on tokens.
  - **Recommendation:** add a read-disturb subsection with R as a swept parameter. Note that H^3's relaxed-retention SLC helps twice: lower tR and a larger P/E budget (Lincoln's 4M figure, via its ref [48]). Also note that the per-block vs per-page counting convention changes the result by up to pages-per-block×; Lincoln counts per iteration.

### 5.5 Sparsity-aware and selective reads
- Only **LLM in a Flash** does FFN-neuron sparsity reads (predictor + window). Lincoln defers dynamic sparsity to future work. AiF and Cambricon read everything.
- [mine] At datacenter batch sizes the union of active neurons across sequences approaches dense (LLM in a Flash's own 5-token window already needs 24% of FFN), so neuron-level sparsity does not reduce HBF traffic. **Expert-level sparsity (MoE) does survive batching partially.** AiF notes MoE "aligns well" (Mixtral reads 14.7 of 46.2 GiB per token). An open direction is expert-aware placement: hot experts in HBM, cold experts in HBF, with batch-level routing statistics deciding the placement. None of the four papers studies this at batch > 1.

### 5.6 In-flash compute nobody has applied to HBF
- AiF (in-die GEMV + ECCLITE), Lincoln (hybrid-bonded logic die with per-plane FP-Dot + BCH) and Cambricon (shared per-die core + slice control) all place logic next to the array. HBF stacks will have a **base logic die**, as HBM does. [mine] At high batch, candidate operations for that die are ones that *reduce bytes* rather than do GEMV:
  - (a) ECC decode (strongly supported by Lincoln's ablation);
  - (b) dequantisation or decompression of weights, raising effective bandwidth;
  - (c) read-reclaim and wear-levelling engines that keep refresh traffic off the GPU link (Lincoln puts this on the SoC);
  - (d) low-batch or latency-critical GEMV for small tenants, i.e. Lincoln's generation path, if a hybrid policy is wanted.
  - None of the papers evaluates these for a GPU-attached stack.

### 5.7 Device-parameter anchors (for sanity-checking HBF assumptions)
| Parameter | Value | Source |
|---|---|---|
| Conventional TLC page read | 37/46/37 µs (LSB/CSB/MSB) | AiF Table 2 (measured) |
| TLC LSB with (1,3,3) coding | 28 µs | AiF Table 2 |
| cr-read tR | 9.7 µs | AiF (simulated) |
| Density-oriented array | >50 µs, up to >100 µs per 16 KB page | Lincoln §III-A |
| Baseline TLC SSD in Lincoln | 56 µs | Lincoln Table I |
| XL-Flash (SLC, small array) | 4 µs, 16 planes, 4 KB page | Lincoln §IV-B (cites [50]) |
| Lincoln shrunk SLC | 3.426 µs, 32 planes, 1.06 GB/s/plane, 34 GB/s/die | Lincoln Table I, §IV-B/D |
| Cambricon assumption | 30 µs, 16 KB page, 2 planes/die, 1000 MT/s × 8-bit channel | Cambricon Table II |
| Per-chip bandwidth, conventional vs AiF | 1.6 → 6.4 GB/s | AiF §6.1 |
| Flash channel (ONFI) | 2.0 GB/s (AiF); 1.2 GB/s (Lincoln baseline); 1.6–2.4 GB/s range (AiF §3.2.1) | AiF, Lincoln |
| SLC P/E endurance | up to 100K; ≥4M with 3-day retention | Lincoln §IV-F (cites [49], [48]) |
| TLC worst-case operating point used | 4K P/E + 1-year retention | AiF §4.3 |
| Read-disturb threshold | 1K–10K reads (MLC/TLC 5K–8K) | Lincoln §IV-F footnote |
| SLC program latency | 75 µs | Lincoln §IV-F (cites [50]) |
| Read energy | 18.278 → 5.098 pJ/bit (cr-read); 98.0 nJ per 4 KB page (Lincoln) | AiF; Lincoln Table I |
| Flash cost | $0.11/GB (YMTC 128-layer TLC with bonding) → $0.72–1.44/GB (Lincoln SLC) | Lincoln §V-E |

### 5.8 Bottom line for the BTP [mine]
1. Cite these four as the "flash for LLM inference" lineage. They all target **single-batch, on-device** use, which justifies the novelty of a datacenter HBM + HBF capacity study.
2. Borrow **Lincoln's read-disturb/read-reclaim analysis**. It is the one piece of this prior work that directly challenges a claim in the report (no writes in steady state).
3. Borrow **AiF's ECC-throughput scaling** and **Lincoln's ECC-placement ablation** to argue that ECC placement and power must appear in the HBF power model.
4. Mention **cr-read** as an unexploited HBF optimisation for sequential weight and prefix streaming (tR −64%, read energy −72.1%).
5. Treat in-flash GEMV as **not applicable at high batch**. The transferable lessons are about bandwidth delivery, latency-hiding buffers, layout and reliability, not about moving compute into the flash.
