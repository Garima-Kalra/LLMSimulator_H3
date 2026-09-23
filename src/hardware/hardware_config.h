#pragma once
#include <string>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"

namespace llm_system {

struct PIMHWConfig {
  ProcessorType type = ProcessorType::GPU;

  int bandwidth_x = 0;
};

class SystemConfig {

  public:
    // default initilizing with H100 config
    SystemConfig(std::string gpu_gen ="H100",
                 int num_node = 1, int num_device = 2, 
                 hw_metric node_ict_latency = 0.5 * 1000,
                 hw_metric node_ict_bandwidth = 400.0 * 1000 * 1000 * 1000,
                 hw_metric device_ict_latency = 3.0 * 1000,
                 hw_metric device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000, 
                 hw_metric compute_peak_flops = 989.4 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_capacity = 80.0 * 1024 * 1024 * 1024,\
                 int logic_x = 4,
                 hw_metric logic_op_b = 8,                 
                 int pim_x = 16,
                 hw_metric pim_op_b = 1,
                 std::vector<ProcessorType> processor_type = {},
                 bool parallel_execution = false,
                 bool hetero_subbatch = false,
                 ProcessorType high_processor_type = ProcessorType::GPU,
                 ProcessorType low_processor_type = ProcessorType::LOGIC,
                 bool communication_hiding = false,            
                 bool disagg_system = false,
                 bool use_low_unit_moe_only = false,
                 bool use_ramulator = false,
                 bool exit_out_of_memory = true,
                 bool mem_cap_limit = false,               
                 bool use_flash_mla = true,
                 bool use_flash_attention = true,
                 bool reuse_kv_cache = true,
                 hw_metric kv_cache_reuse_rate = 0.5,
                 bool prefill_mode = false,
                 bool decode_mode = false,
                 bool use_inject_rate = false,
                 int request_per_second = 10,
                 int num_cube = 5,
                 int num_logic_cube = 5,
                 bool use_hbf = false,
                 hw_metric hbf_bandwidth = 8.000 * 1000 * 1000 * 1000 * 1000,
                 hw_metric hbf_capacity = 3.0 * 1024 * 1024 * 1024 * 1024,
                 int num_hbf_cube = 8,
                 hw_metric hbf_bandwidth_scale = 1.0,
                 hw_metric gpu_tdp = 680.0,
                 hw_metric hbm_tdp_per_cube = 40.0,
                 hw_metric hbf_tdp_per_cube = 160.0,
                 // Fraction of HBM per device held back from the KV-cache
                 // pool (framework/CUDA context, comm + kernel workspace,
                 // allocator fragmentation). 0 = today's behavior: every
                 // byte not holding weights/shared-cache is usable for KV.
                 hw_metric hbm_reserve_fraction = 0.0
                )
      : gpu_gen(gpu_gen),
        num_node(num_node),
        num_device(num_device),
        node_ict_latency(node_ict_latency),
        node_ict_bandwidth(node_ict_bandwidth),
        device_ict_latency(device_ict_latency),
        device_ict_bandwidth(device_ict_bandwidth),
        compute_peak_flops(compute_peak_flops),
        memory_bandwidth(memory_bandwidth),
        memory_capacity(memory_capacity),
        logic_x(logic_x),
        logic_op_b(logic_op_b),
        pim_x(pim_x),
        pim_op_b(pim_op_b),
        processor_type(processor_type),
        parallel_execution(parallel_execution),
        hetero_subbatch(hetero_subbatch),
        high_processor_type(high_processor_type),
        low_processor_type(low_processor_type),
        communication_hiding(communication_hiding),
        disagg_system(disagg_system),
        use_low_unit_moe_only(use_low_unit_moe_only),
        use_ramulator(use_ramulator),
        exit_out_of_memory(exit_out_of_memory),
        mem_cap_limit(mem_cap_limit),
        use_flash_mla(use_flash_mla),
        use_flash_attention(use_flash_attention),
        reuse_kv_cache(reuse_kv_cache),
        kv_cache_reuse_rate(kv_cache_reuse_rate),
        prefill_mode(prefill_mode),
        decode_mode(decode_mode),
        use_inject_rate(use_inject_rate),
        request_per_second(request_per_second),
        num_cube(num_cube),
        num_logic_cube(num_logic_cube),
        use_hbf(use_hbf),
        hbf_bandwidth(hbf_bandwidth),
        hbf_capacity(hbf_capacity),
        num_hbf_cube(num_hbf_cube),
        hbf_bandwidth_scale(hbf_bandwidth_scale),
        gpu_tdp(gpu_tdp),
        hbm_tdp_per_cube(hbm_tdp_per_cube),
        hbf_tdp_per_cube(hbf_tdp_per_cube),
        hbm_reserve_fraction(hbm_reserve_fraction){
          logic_memory_bandwidth = memory_bandwidth * logic_x;
          pim_memory_bandwidth = memory_bandwidth * pim_x;
        };

    SystemConfig& operator=(const SystemConfig& rhs) = default;

  std::string gpu_gen;

  // Device number
  int num_node;
  int num_device;

  // Cluster specification
  hw_metric node_ict_latency;   // ns
  hw_metric node_ict_bandwidth; // B/s

  // Node specification
  hw_metric device_ict_latency;    // ns, 
  hw_metric device_ict_bandwidth;  // B/s

  // Device specification
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;

  hw_metric memory_capacity;

  // Logic specification
  int logic_x;
  hw_metric logic_memory_bandwidth = memory_bandwidth * logic_x;
  hw_metric logic_op_b;

  // PIM specifiaction
  int pim_x;
  hw_metric pim_memory_bandwidth = memory_bandwidth * pim_x;
  hw_metric pim_op_b;

  std::vector<ProcessorType> processor_type = {};

  bool parallel_execution = false;
  bool hetero_subbatch = false;
  ProcessorType high_processor_type = ProcessorType::GPU;
  ProcessorType low_processor_type = ProcessorType::LOGIC;

  bool communication_hiding = false;

  bool disagg_system = true;
  bool use_low_unit_moe_only = false;
  bool use_ramulator = false;
  
  bool exit_out_of_memory = false;
  bool mem_cap_limit = false;

  bool use_flash_mla = true; 
  bool use_flash_attention = true; 
  bool reuse_kv_cache = true;
  hw_metric kv_cache_reuse_rate; 
  // this rate includes, 
  // 1) how long does prompt share tokens with cached KV 
  // 2) does prompt share tokens with cached KV
  // because we select rate between [0, kv_cache_reuse_rate * 2), kv_cache_reuse_rate must be max 0.5

  bool prefill_mode = false; 
  bool decode_mode = false;

  bool use_inject_rate = false;  // injection random number of sequence
  int request_per_second;

  int num_cube; //8: for HBM3E (B100), 5 for HBM3 (H100)
  int num_logic_cube;
  // Device

  // H3: High Bandwidth Flash (HBF) specification
  // read-only data (weights, shared pre-computed KV cache) is placed here
  // when use_hbf is on; HBF is modeled as ideal/bandwidth-bound only
  // (NAND access latency is assumed hidden by the Latency Hiding Buffer,
  // per the H3 paper -- no Ramulator2/cycle-accurate path is supported).
  bool use_hbf;
  hw_metric hbf_bandwidth;       // B/s
  hw_metric hbf_capacity;        // bytes
  int num_hbf_cube;
  hw_metric hbf_bandwidth_scale; // sensitivity knob, e.g. 0.5 for halved HBF bandwidth

  // Static per-device TDP (W), used for throughput-per-power only.
  // This is independent of the DRAM per-op access-energy model in
  // dram/power.h -- cube TDP is drawn regardless of access pattern.
  hw_metric gpu_tdp;
  hw_metric hbm_tdp_per_cube;
  hw_metric hbf_tdp_per_cube;
  // See constructor comment. Applied wherever KV-cache headroom is computed
  // (Cluster::checkMemorySize / checkH3MemorySize) -- never to the raw
  // weights/shared-cache fit check, which is a hard physical limit.
  hw_metric hbm_reserve_fraction;

  // Collective model for tensor-parallel AllReduce when the device group
  // spans more than one node. false = flat ring (the paper's stated
  // assumption, Sec. IV-A.1 "ring all-reduce"): synchronous steps, so every
  // step is gated by the slowest link in the ring, i.e. the inter-node
  // fabric. true = hierarchical / NCCL-style (reduce-scatter within node ->
  // ring across nodes -> all-gather within node), which keeps most traffic
  // on NVLink. Single-node groups are unaffected either way.
  bool allreduce_hierarchical = false;

  // Achieved fraction of peak FLOPS for decode attention (eta). 1.0 = the
  // codebase's implicit assumption that attention hits 100% of peak, which
  // is optimistic: shared-prefix attention is a batched GEMM and private-KV
  // attention is GEMV-shaped, and neither reaches peak in practice. Sweeping
  // this is how the compute/memory crossover is reported as a range rather
  // than a single number resting on a perfect-peak assumption.
  hw_metric attn_compute_efficiency = 1.0;

  // sigma: fraction of the shared CAG cache actually selected/attended per
  // step (block-sparse selection). Scales BOTH the shared-KV bytes fetched
  // and the shared-attention FLOPs, since selecting 5% of blocks means
  // reading 5% of the bytes and attending to 5% of the tokens. Deliberately
  // does NOT affect stored capacity: the full cache still occupies HBF, only
  // the per-step traffic and work shrink. 1.0 = dense (no selection).
  hw_metric shared_kv_sparsity = 1.0;
  
  // Physical integration topology of the HBM/HBF tiers.
  //   "independent"  each stack has its own GPU-side link; HBM and HBF
  //                  transfers overlap (this is what max() below assumes)
  //   "cascaded"     H3's daisy chain: HBF hangs off the HBM base die, so
  //                  all traffic crosses one shared link and serializes
  //   "shared_base"  mixed dies over one base die, one link per stack
  std::string link_topology = "independent";
  // Fraction by which admission may exceed what HBM can physically hold for
  // private KV. The excess spills to HBF -- the only mechanism by which flash
  // is written under H3's placement, since weights and the shared cache are
  // written once and then read-only. 0.0 = today's behaviour (cap to fit).
  //
  // Sizing (llama3_405B, 10M, 14H2F): write bandwidth is never the limit --
  // even at 1.0 writes are <3% of a decode step. ENDURANCE is: 0.05 -> ~5.8
  // yr device life, 0.25 -> ~1.2 yr, 1.0 -> ~0.3 yr.
  hw_metric hbm_oversubscribe_fraction = 0.0;
  // Fraction of an erase block accumulated in the DRAM write-combining
  // buffer before flushing to flash. Appends arrive per (layer, sequence) at
  // 512 B against a 4096 B page, so unbuffered amplification is 8x; 0.75
  // gives 1.33x. Sweeping this is the experiment: higher threshold means
  // lower amplification but more HBM held by the buffer.
  std::string experiment_mode = "h3_extended";
  hw_metric hbf_flush_threshold = 0.75;
  // Stack sites available on the GPU shoreline. Cascaded HBF consumes none;
  // side-by-side HBF displaces an HBM stack.
  int shoreline_slots = 8;

  // Bandwidth of the shared GPU-side link, for topologies where the two
  // tiers do not have independent paths. 0 = unused (independent links).
  hw_metric shared_link_bandwidth = 0.0;
};


static SystemConfig A100 = SystemConfig(
                 "A100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 https://www.fs.com/products/161048.html?attribute=106827&id=3941024
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 3.0 * 1000,                        // device_ict_latency, nvlink 3
                 150.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 312.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.039 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system 
                 false,                             // use_low_unit_moe_only
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig H100 = SystemConfig(
                 "H100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth, nvlink 4
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 3.352 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig B100 = SystemConfig(
                  "B100",                            // gpu gen
                  1,                                 // num_node 
                  2,                                 // num_device
                  130.0,                             // node_ict_latency, connectx-7 
                  50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                  0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                  900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                  1750.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                  8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                  192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                  4,                                 // logic_x 
                  8,                                 // logic_op_b                 
                  16,                                // pim_x
                  1,                                 // pim_op_b
                  {},                                // processor_type
                  false,                             // parallel_execution
                  false,                             // hetero_subbatch
                  ProcessorType::GPU,                // high_processor_type
                  ProcessorType::LOGIC,              // low_processor_type
                  false,                             // communication_hiding
                  false,                             // disagg_system
                  false,                             // use_low_unit_moe_only 
                  false,                             // use_ramulator
                  true,                              // exit_out_of_memory
                  false,                             // mem_cap_limit
                  true,                              // use_flash_mla
                  true,                              // use_flash_attention
                  false,                             // reuse_kv_cache
                  0.0,                               // kv_cache_reuse_rate
                  false,                             // prefill_mode
                  false,                             // decode_mode
                  false,                             // use_inject_rate
                  10,                                // request_per_second
                  8,                                 // num_cube
                  8                                  // int num_logic_cube
                  );
                  
static SystemConfig B200 = SystemConfig(
                 "B200",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 2250.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );

// H3: B200 + High Bandwidth Flash.
// HBF specs per H3 paper (Sec. IV-A / Fig. 3(c)): 3TB capacity (16x
// HBM3E's 24GB/cube, 8 cubes x 384GB = 3TB), 8TB/s bandwidth (same as
// HBM3E), 160W TDP per cube (vs. 40W/cube for HBM3E). GPU TDP (680W) is
// the package TDP without HBM/HBF, per paper Sec. IV-B.3.
static SystemConfig B200_H3 = [] {
  SystemConfig config = B200;
  // NOTE: gpu_gen intentionally stays "B200" (not "B200_H3") -- it is used
  // elsewhere (e.g. Device::Device in device.cpp) as a key to select the
  // HBM3E dram_config yaml / MemoryConfig preset, which is unaffected by
  // HBF; use_hbf is the correct discriminator for H3-specific behavior.
  config.use_hbf = true;
  config.hbf_bandwidth = 8.000 * 1000 * 1000 * 1000 * 1000;
  config.hbf_capacity = 3.0 * 1024 * 1024 * 1024 * 1024;
  config.num_hbf_cube = 8;
  config.hbf_bandwidth_scale = 1.0;
  config.gpu_tdp = 680.0;
  config.hbm_tdp_per_cube = 40.0;
  config.hbf_tdp_per_cube = 160.0;
  return config;
}();

}  // namespace llm_system