# Literature

Papers read for this project, grouped by theme. PDFs are **not** stored in this repository because most are under publisher or author copyright; use the links below (IIT Guwahati library access for IEEE/ACM).

Detailed reading notes (mechanism, device parameters, evaluation, limitations, gaps, BibTeX) are in [`notes/`](notes/).

## HBM + HBF architectures and systems

| Paper | Venue | Link |
|---|---|---|
| H3: Hybrid Architecture Using HBM and HBF for Cost-Efficient LLM Inference | IEEE CAL 2026 | [https://doi.org/10.1109/LCA.2026.3660969](https://doi.org/10.1109/LCA.2026.3660969) |
| HBM-HBF-Centric Memory Pooling Architecture With Custom Base Die for Terabyte-Scale LLM Inference | IEEE CAL 2026 | [https://doi.org/10.1109/LCA.2026.3703982](https://doi.org/10.1109/LCA.2026.3703982) |
| Hardware-Managed Heterogeneous HBM and Flash in LLM Inference Systems (Oxford) | Preprint, 2026 | [https://ora.ox.ac.uk/objects/uuid:a9f80041-ff2e-4714-9bc4-51c513604d36](https://ora.ox.ac.uk/objects/uuid:a9f80041-ff2e-4714-9bc4-51c513604d36) |
| Hot-Cold Tiering of HBM and High Bandwidth Flash for Agentic LLM Serving | arXiv 2026 | [https://arxiv.org/abs/2609.25782](https://arxiv.org/abs/2609.25782) |
| FlashAccel: Leveraging High-Bandwidth Flash for High-Throughput LLM Inference | arXiv 2026 | [https://arxiv.org/abs/2607.10186](https://arxiv.org/abs/2607.10186) |
| FLINT: Efficiently Leveraging HBF for Capacity-Scalable LLM Inference Acceleration | arXiv 2026 | [https://arxiv.org/abs/2608.25062](https://arxiv.org/abs/2608.25062) |
| Beyond Capacity (DASH): Scalable MoE LLM Inference via HBF with Direct GPU and HBM Paths | arXiv 2026 | [https://arxiv.org/abs/2608.14333](https://arxiv.org/abs/2608.14333) |
| HBFlex: A Flexible Memory System for Fine-Grained LLM States and Coarse-Grained HBF | arXiv 2026 | [https://arxiv.org/abs/2609.18675](https://arxiv.org/abs/2609.18675) |
| SPLASH: Co-Designing Sparse Attention with High-Bandwidth Flash | arXiv 2026 | [https://arxiv.org/abs/2609.23816](https://arxiv.org/abs/2609.23816) |
| Potential Applications of HBF in LLM Serving Systems | arXiv 2026 | [https://arxiv.org/abs/2608.13127](https://arxiv.org/abs/2608.13127) |
| Trillion-Parameter MoE in a Box: Decoupling Memory Provisioning with HBF | arXiv 2026 | [https://arxiv.org/abs/2609.15636](https://arxiv.org/abs/2609.15636) |
| Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy | arXiv 2026 | [https://arxiv.org/abs/2609.07175](https://arxiv.org/abs/2609.07175) |
| MemExplorer: Heterogeneous Memory Design Space for Agentic Inference NPUs | arXiv 2026 | [https://arxiv.org/abs/2604.16007](https://arxiv.org/abs/2604.16007) |

## HBF characterisation, simulators and position papers

| Paper | Venue | Link |
|---|---|---|
| Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges | IEEE CAL 2026 | [https://doi.org/10.1109/LCA.2026.3705817](https://doi.org/10.1109/LCA.2026.3705817) · [https://arxiv.org/abs/2608.13868](https://arxiv.org/abs/2608.13868) |
| High-Bandwidth Flash for KV Caches: Endurance and Performance Implications | IEEE CAL 2026 | [https://doi.org/10.1109/LCA.2026.3695938](https://doi.org/10.1109/LCA.2026.3695938) |
| HBF Sucks? A Full-Stack Characterization of HBF for KV-Centric LLM Serving | arXiv 2026 | [https://arxiv.org/abs/2608.11668](https://arxiv.org/abs/2608.11668) |
| HBFSim: Fast and Faithful Simulation of HBF Under Real GPU Execution | arXiv 2026 | [https://arxiv.org/abs/2609.09800](https://arxiv.org/abs/2609.09800) |
| HBF-Sim: An Extensible HBF Simulator for Large-scale GPU Memory Systems | arXiv 2026 | [https://arxiv.org/abs/2609.29246](https://arxiv.org/abs/2609.29246) |
| TileLens: Efficiently Using Large-Granularity Memory Systems | arXiv 2026 | [https://arxiv.org/abs/2607.04031](https://arxiv.org/abs/2607.04031) |
| HAVEN: HBF Augmented Vector Engine for ANN Search | arXiv 2026 | [https://arxiv.org/abs/2603.01175](https://arxiv.org/abs/2603.01175) |
| Challenges and Research Directions for LLM Inference Hardware (Ma & Patterson) | arXiv 2026 | [https://arxiv.org/abs/2601.05047](https://arxiv.org/abs/2601.05047) |

## Flash for LLM inference (on-device / in-flash)

| Paper | Venue | Link |
|---|---|---|
| LLM in a Flash: Efficient LLM Inference with Limited Memory | ACL 2024 | [https://arxiv.org/abs/2312.11514](https://arxiv.org/abs/2312.11514) |
| AiF: Accelerating On-Device LLM Inference Using In-Flash Processing | ISCA 2025 | [https://doi.org/10.1145/3695053.3731073](https://doi.org/10.1145/3695053.3731073) |
| Lincoln: Real-Time 50~100B LLM Inference with Compute-Enabled Flash | HPCA 2025 | [https://doi.org/10.1109/HPCA61900.2025.00128](https://doi.org/10.1109/HPCA61900.2025.00128) |
| Cambricon-LLM: Chiplet-Based Hybrid Architecture for On-Device 70B LLM | MICRO 2024 | [https://arxiv.org/abs/2409.15654](https://arxiv.org/abs/2409.15654) |
| ZnG: Architecting GPU Multi-Processors with New Flash | ISCA 2020 | [https://arxiv.org/abs/2006.08975](https://arxiv.org/abs/2006.08975) |
| LLM Inference in a Flash! | arXiv 2026 | [https://arxiv.org/abs/2609.16161](https://arxiv.org/abs/2609.16161) |
| NVLLM: 3D NAND-Centric Architecture for Edge LLM Inference | arXiv 2026 | [https://arxiv.org/abs/2604.25699](https://arxiv.org/abs/2604.25699) |
| KVNAND: DRAM-Free In-Flash Computing for On-Device LLM | arXiv 2025 | [https://arxiv.org/abs/2512.03608](https://arxiv.org/abs/2512.03608) |
| NASiC: 3D NAND CIM for On-Device MoE Inference | arXiv 2026 | [https://arxiv.org/abs/2605.23294](https://arxiv.org/abs/2605.23294) |
| NITRO: 3D NAND Flash-Based In-Storage Computing | arXiv 2026 | [https://arxiv.org/abs/2608.11920](https://arxiv.org/abs/2608.11920) |

## SSD / storage offloading for LLMs

| Paper | Venue | Link |
|---|---|---|
| HiFC: High-efficiency Flash-based KV Cache Swapping for Scaling LLM Inference | NeurIPS 2025 | [https://openreview.net/forum?id=onhjdWCxZY](https://openreview.net/forum?id=onhjdWCxZY) |
| InstAttention: In-Storage Attention Offloading for Long-Context LLM Inference | HPCA 2025 | [https://doi.org/10.1109/HPCA61900.2025.00113](https://doi.org/10.1109/HPCA61900.2025.00113) · [https://arxiv.org/abs/2409.04992](https://arxiv.org/abs/2409.04992) |
| HILOS: Near-Storage Processing for Offline Long-Context LLM Inference | ASPLOS 2026 | [https://doi.org/10.1145/3779212.3790119](https://doi.org/10.1145/3779212.3790119) · [https://arxiv.org/abs/2502.09921](https://arxiv.org/abs/2502.09921) |
| Smart-Infinity: LLM Training using Near-Storage Processing | HPCA 2024 | [https://arxiv.org/abs/2403.06664](https://arxiv.org/abs/2403.06664) |
| SSD Offloading for MoE Weights Considered Harmful in Energy Efficiency | arXiv 2025 | [https://arxiv.org/abs/2508.06978](https://arxiv.org/abs/2508.06978) |
| MatKV: Trading Compute for Flash Storage in LLM Inference | arXiv 2025 | [https://arxiv.org/abs/2512.22195](https://arxiv.org/abs/2512.22195) |
| Tutti: SSD-Backed KV Cache for Long-Context Serving | arXiv 2026 | [https://arxiv.org/abs/2605.03375](https://arxiv.org/abs/2605.03375) |
| KVSwap: Disk-aware KV Cache Offloading On-device | arXiv 2025 | [https://arxiv.org/abs/2511.11907](https://arxiv.org/abs/2511.11907) |
| Dual-Blade: NVMe-Direct KV-Cache Offloading for Edge | arXiv 2026 | [https://arxiv.org/abs/2604.26557](https://arxiv.org/abs/2604.26557) |
| Harnessing DRAM and SSD for LLM Inference with Mixed Precision | arXiv 2024 | [https://arxiv.org/abs/2410.14740](https://arxiv.org/abs/2410.14740) |
| Where Should the KV Cache Live? Placement across GPU, CPU and SSD | arXiv 2026 | [https://arxiv.org/abs/2609.16215](https://arxiv.org/abs/2609.16215) |
| REIS: Retrieval System with In-Storage Processing | arXiv 2025 | [https://arxiv.org/abs/2506.16444](https://arxiv.org/abs/2506.16444) |

## Heterogeneous and tiered memory for LLMs

| Paper | Venue | Link |
|---|---|---|
| H2M2: Hardware-based Heterogeneous Memory Management for LLM Inference | arXiv 2025 | [https://arxiv.org/abs/2504.14893](https://arxiv.org/abs/2504.14893) |
| CENT: PIM Is All You Need, a CXL-Enabled GPU-Free System for LLM Inference | ASPLOS 2025 | [https://doi.org/10.1145/3676641.3716267](https://doi.org/10.1145/3676641.3716267) |
| LIA: Single-GPU LLM Inference with AMX CPU-GPU and CXL Offloading | ISCA 2025 | [https://doi.org/10.1145/3695053.3731092](https://doi.org/10.1145/3695053.3731092) |
| Stratum: Tiered Monolithic 3D-Stackable DRAM for MoE Serving | MICRO 2025 | [https://doi.org/10.1145/3725843.3756043](https://doi.org/10.1145/3725843.3756043) |
| RoMe: Row Granularity Access Memory System for LLMs | HPCA 2026 | [https://arxiv.org/abs/2512.01541](https://arxiv.org/abs/2512.01541) |
| InfiniGen: Dynamic KV Cache Management | OSDI 2024 | [https://arxiv.org/abs/2406.19707](https://arxiv.org/abs/2406.19707) |
| ITME: Tiered Memory Expansion with CXL-Hybrid Memories | arXiv 2026 | [https://arxiv.org/abs/2606.12556](https://arxiv.org/abs/2606.12556) |
| Beluga: CXL-Based Memory for LLM KV Cache | arXiv 2025 | [https://arxiv.org/abs/2511.20172](https://arxiv.org/abs/2511.20172) |
| A CXL Memory Rack for Multi-Turn LLM Serving | arXiv 2026 | [https://arxiv.org/abs/2607.18141](https://arxiv.org/abs/2607.18141) |
| DAK: Direct-Access GPU Memory Offloading for LLM Inference | arXiv 2026 | [https://arxiv.org/abs/2604.26074](https://arxiv.org/abs/2604.26074) |
| Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management | arXiv 2026 | [https://arxiv.org/abs/2609.26828](https://arxiv.org/abs/2609.26828) |
| BOOST: Concurrent Host Memory and HBM Access for LLM Inference | arXiv 2026 | [https://arxiv.org/abs/2609.13592](https://arxiv.org/abs/2609.13592) |
| Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy | arXiv 2026 | [https://arxiv.org/abs/2605.09490](https://arxiv.org/abs/2605.09490) |
| Scalable PNM for 1M-Token LLM Inference with CXL KV Cache | arXiv 2025 | [https://arxiv.org/abs/2511.00321](https://arxiv.org/abs/2511.00321) |
| OasisKV: KV Cache Beyond HBM with Lookahead Sparse Prefetching | arXiv 2026 | [https://arxiv.org/abs/2608.08097](https://arxiv.org/abs/2608.08097) |
| Bandwidth-Effective DRAM Cache for GPUs with Storage-Class Memory | arXiv 2024 | [https://arxiv.org/abs/2403.09358](https://arxiv.org/abs/2403.09358) |
| G10: Unified GPU Memory and Storage with Smart Tensor Migration | arXiv 2023 | [https://arxiv.org/abs/2310.09443](https://arxiv.org/abs/2310.09443) |
| HALO: Memory-Centric Heterogeneous Accelerator for Low-Batch LLM | arXiv 2025 | [https://arxiv.org/abs/2510.02675](https://arxiv.org/abs/2510.02675) |
| SuperInfer: SLO-Aware Memory Management on Superchips | arXiv 2026 | [https://arxiv.org/abs/2601.20309](https://arxiv.org/abs/2601.20309) |
| Idleness is Relative: Tool-Call Idle Windows for Offloading in Agentic Systems | arXiv 2026 | [https://arxiv.org/abs/2606.00866](https://arxiv.org/abs/2606.00866) |
| LIFE: Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling | arXiv 2025 | [https://arxiv.org/abs/2508.00904](https://arxiv.org/abs/2508.00904) |

## NAND reliability background

| Paper | Venue | Link |
|---|---|---|
| Read Disturb Errors in MLC NAND Flash (Cai et al.) | DSN 2015 | search title |
| Error Characterization, Mitigation, and Recovery in Flash-Memory SSDs (Cai et al.) | Proc. IEEE 2017 | [https://doi.org/10.1109/JPROC.2017.2713127](https://doi.org/10.1109/JPROC.2017.2713127) |
| HeatWatch: 3D NAND Reliability via Self-Recovery and Temperature Awareness (Luo et al.) | HPCA 2018 | search title |
| Optimizing NAND Flash-Based SSDs via Retention Relaxation (Liu et al.) | FAST 2012 | search title |
| OCP High Bandwidth Flash High-Level Base Die Specification v0.7.0 | OCP, Aug 2026 | [https://www.opencompute.org/ (search 'HBF')](https://www.opencompute.org/ (search 'HBF')) |

## Relevant but not on arXiv (search by title)

- Mem-Village: A Hybrid HBF-HBM Architecture on Glass for Ultra-Scale AI Inference (2025) — most relevant
- KiF: Accelerating Low-Batch LLM Inference Using In-Flash KV Cache (2026)
- AdaptiveKV: Bandwidth-Adaptive Memory Allocation for KV Cache Offloading (2026)
- Architecting a Flash-Based Storage System for Low-Cost Inference of Extreme-Scale DNNs (2022)
- FlashNeuron: SSD-Enabled Large-Batch Training (FAST 2021)
- FlashGPU: Placing New Flash Next to GPU Cores (DAC 2019)
- OptimStore: In-Storage Optimization of Large-Scale DNNs (HPCA 2023)
- An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD (2025)
- SHyLA: 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture (2026)
- BALANCE: Bit and Layer-Aware Lightweight ECC for In-Flash Computing LLM Inference (2026)
- Rowhammer Vulnerabilities in 3D NAND Flash Induced by Hot Carrier Injection (2026) — relevant to read disturb
- LLM-on-the-Palm: Mobile LLM Inference with PIM-Enhanced NAND Flash (2025)
- DIAMoND: Adaptive Edge MoE with In-NAND and Near-DRAM Compute (2026)
- REPA: Reconfigurable PIM for KV Cache Offloading and Processing (2026)
- HyMCache: KV Cache Framework with CXL-Hybrid Memory (2026)
- LongSight: Compute-Enabled Memory for Large-Context LLMs via Sparse Attention (2025)
- AutoTM: Automatic Tensor Movement in Heterogeneous Memory Systems (ASPLOS 2020)
- NVMMU: Non-volatile Memory Management Unit for GPU-SSD Architectures (PACT 2015)
- GMT: GPU Orchestrated Memory Tiering for the Big Data Era (2024)

## Notes files

| File | Papers covered |
|---|---|
| [A_core_hbm_hbf.md](notes/A_core_hbm_hbf.md) | H3, Pooling, Oxford HMA, Hot–Cold, Exploring HBF, HBF for KV caches, GR serving |
| [B_hbf_systems_1.md](notes/B_hbf_systems_1.md) | FlashAccel, FLINT, DASH |
| [C_hbf_systems_2_sims.md](notes/C_hbf_systems_2_sims.md) | HBFlex, HBF Sucks?, HBFSim, HBF-Sim, TileLens |
| [D1_flash_llm_devices.md](notes/D1_flash_llm_devices.md) | LLM in a Flash, AiF, Lincoln, Cambricon-LLM |
| [D2_storage_offload.md](notes/D2_storage_offload.md) | HiFC, InstAttention, HILOS, Smart-Infinity |
| [E1_hetero_memory.md](notes/E1_hetero_memory.md) | H2M2, CENT, LIA, Stratum |
| [E2_memsys_modelling.md](notes/E2_memsys_modelling.md) | RoMe, InfiniGen, ZnG, HAVEN, LIFE |
