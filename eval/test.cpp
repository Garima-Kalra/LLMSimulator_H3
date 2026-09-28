#include <yaml-cpp/yaml.h>
#include "validate_config.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"

using namespace llm_system;

namespace {
// Provenance for results.log: short commit hash, suffixed "-dirty" when the
// working tree differs from that commit. The hash alone is not enough --
// most rows in this project were produced from an uncommitted tree.
std::string git_revision() {
  std::string rev = "unknown";
  char buf[128];
  if (FILE *f = popen("git rev-parse --short HEAD 2>/dev/null", "r")) {
    if (fgets(buf, sizeof buf, f)) {
      rev = buf;
      rev.erase(rev.find_last_not_of(" \n\r\t") + 1);
    }
    pclose(f);
  }
  if (FILE *f = popen("git status --porcelain 2>/dev/null", "r")) {
    if (fgets(buf, sizeof buf, f)) rev += "-dirty";
    pclose(f);
  }
  return rev;
}
}  // namespace

int main(int argc, char *argv[]) {
  YAML::Node config;
  if(argc > 1){
    std::string config_path = argv[1];
    config = YAML::LoadFile(config_path);
  }
  else{
    config = YAML::LoadFile("config.yaml");
  }

  // Reject unrecognised keys and warn on silent defaults. Three separate
  // results in this project were invalidated by configs that parsed cleanly
  // but did not mean what they appeared to: hbm_reserve_fraction missing from
  // 45 of 55 files, link_topology never read at all, and a hand-written
  // hbf_stacks key that no code consumes -- which made three "different"
  // flash configurations produce byte-identical output.
  validateConfig(config);

  std::string model_name = config["model"]["model_name"].as<std::string>();
  std::string processor_type =
      config["system"]["processor_type"].as<std::string>();

  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();

  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int context_window = config["simulation"]["context_window"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();

  int max_batch_size = config["serving"]["max_batch_size"].as<int>();

  int max_process_token = config["serving"]["max_process_token"].as<int>();
  std::string output_path = config["log"]["output_directory"].as<std::string>();

  bool use_hbf = config["system"]["use_hbf"].as<bool>();
  std::string gpu_gen = config["system"]["gpu_gen"].as<std::string>();

  SystemConfig system_config;
  if(gpu_gen == "A100"){
    system_config = A100;
  }
  else if (gpu_gen == "H100"){
    system_config = H100;
  }
  else if (gpu_gen == "B100"){
    system_config = B100;
  }
  else if (gpu_gen == "B200"){
    system_config = use_hbf ? B200_H3 : B200;
  }
  else{
    fail("No GPU generation information");
  }

  if(use_hbf && gpu_gen != "B200"){
    fail("H3 (use_hbf) is currently only supported for gpu_gen: B200");
  }

  system_config.hbf_bandwidth_scale =
      config["system"]["hbf_bandwidth_scale"].as<double>();
  // Per-device HBM held back from the KV-cache pool (framework/workspace/
  // fragmentation). Default 0 = unchanged behavior. See ASSUMPTIONS.md --
  // this is CALIBRATED against the paper's Fig. 5 batch ratios, not derived.
  system_config.hbm_reserve_fraction =
      config["system"]["hbm_reserve_fraction"].as<double>(0.0);
  // Multi-node AllReduce model: false = flat ring gated by the inter-node
  // fabric (paper's stated "ring all-reduce"); true = hierarchical/NCCL-style.
  system_config.allreduce_hierarchical =
      config["system"]["allreduce_hierarchical"].as<bool>(false);
  // eta: achieved fraction of peak FLOPS for decode attention. Swept to
  // report the compute/memory crossover as a range. 1.0 = perfect peak.
  system_config.attn_compute_efficiency =
      config["system"]["attn_compute_efficiency"].as<double>(1.0);
  // sigma: shared-KV selection fraction (scales shared bytes AND shared FLOPs).
  system_config.shared_kv_sparsity =
      config["system"]["shared_kv_sparsity"].as<double>(1.0);
  system_config.experiment_mode =
    config["system"]["experiment_mode"].as<std::string>("h3_extended");
  system_config.hbm_oversubscribe_fraction =
    config["system"]["hbm_oversubscribe_fraction"].as<double>(0.0);   
  system_config.hbf_flush_threshold =
    config["system"]["hbf_flush_threshold"].as<double>(0.75); 
  // Memory-link timing model, consumed by h3MemoryDuration():
  //   independent  -> tiers overlap,  max(hbm, hbf)
  //   cascaded /
  //   shared_base  -> tiers contend over one link, (hbm+hbf)/shared_bw
  // For arch != "legacy" this is derived from the architecture further down;
  // this read covers legacy runs and allows an explicit override.
  system_config.link_topology =
      config["system"]["link_topology"].as<std::string>("independent");
  
  // === ARCHITECTURE ========================================================
  //   cascaded    : each site = HBM stack + HBF stack chained behind it.
  //                 Both tiers cross the site's link -> pooled/contended.
  //   side_by_side: sites are single-tier. HBF sites displace HBM sites.
  //                 Independent links -> bandwidth partitioned per tier.
  //   shared_base : each site = one stack, dies split HBM/HBF over one base
  //                 die. HBF dies displace HBM dies. Link pooled.
  //   hbf_only    : side_by_side with minimal HBM.
  std::string arch =
      config["system"]["architecture"].as<std::string>("legacy");
  system_config.shoreline_slots =
      config["system"]["shoreline_slots"].as<int>(8);
  int dies = config["system"]["dies_per_stack"].as<int>(8);

  if (arch != "legacy") {
    hw_metric hbm_die = system_config.memory_capacity / system_config.num_cube / dies;
    hw_metric hbf_die = system_config.hbf_capacity / system_config.num_hbf_cube / dies;
    hw_metric site_bw = system_config.memory_bandwidth / system_config.num_cube;
    hw_metric die_bw = site_bw / dies;

    int hbm_sites = config["system"]["hbm_sites"].as<int>(0);
    int hbf_sites = config["system"]["hbf_sites"].as<int>(0);
    int hbm_dies_ps = config["system"]["hbm_dies_per_site"].as<int>(dies);
    int hbf_dies_ps = config["system"]["hbf_dies_per_site"].as<int>(dies);

    long hbm_dies_tot = 0, hbf_dies_tot = 0;
    int slots = 0;
    hw_metric hbm_bw = 0, hbf_bw = 0, shared_bw = 0;

    if (arch == "cascaded" || arch == "shared_base") {
      if (arch == "shared_base") {
        assertTrue(hbm_dies_ps + hbf_dies_ps <= dies,
                   "shared_base: HBM+HBF dies exceed dies_per_stack");
      }
      slots = hbm_sites;                 // chained/mixed HBF costs no extra site
      hbm_dies_tot = (long)hbm_sites * hbm_dies_ps;
      hbf_dies_tot = (long)hbm_sites * hbf_dies_ps;
      shared_bw = hbm_sites * site_bw;   // one pooled link budget
      hbm_bw = shared_bw;
      hbf_bw = std::min((hw_metric)hbf_dies_tot * die_bw, shared_bw);
    } else {                             // side_by_side, hbf_only
      slots = hbm_sites + hbf_sites;
      hbm_dies_tot = (long)hbm_sites * hbm_dies_ps;
      hbf_dies_tot = (long)hbf_sites * hbf_dies_ps;
      hbm_bw = hbm_sites * site_bw;      // partitioned, independent paths
      hbf_bw = hbf_sites * site_bw;
      shared_bw = 0.0;
    }

    system_config.memory_capacity = hbm_die * hbm_dies_tot;
    system_config.memory_bandwidth = hbm_bw;
    system_config.hbf_capacity = hbf_die * hbf_dies_tot;
    system_config.hbf_bandwidth = hbf_bw;
    system_config.shared_link_bandwidth = shared_bw;
    // Derive the timing model from the architecture. Without this the
    // "LINK POOLED" banner below prints while h3MemoryDuration still takes
    // max(), which hands pooled designs ~50% more effective aggregate
    // bandwidth than their partitioned counterparts and makes the
    // pooled-vs-partitioned comparison unmatched. Explicit override wins.
    system_config.link_topology =
        config["system"]["link_topology"].as<std::string>(
            shared_bw > 0 ? (arch == "shared_base" ? "shared_base" : "cascaded")
                          : "independent");
    system_config.use_hbf = (hbf_dies_tot > 0);
    system_config.num_cube = (int)(hbm_dies_tot / dies);
    system_config.num_hbf_cube = (int)(hbf_dies_tot / dies);

    assertTrue(slots <= system_config.shoreline_slots,
               "architecture exceeds shoreline budget");
    assertTrue(system_config.memory_capacity > 0,
               "no HBM: activations and generated KV need a read/write tier");

    printf("\n+================================================================+\n");
    printf("| ARCHITECTURE: %-28s sites %d / %-9d |\n",
           arch.c_str(), slots, system_config.shoreline_slots);
    printf("+================================================================+\n");
    printf("|  HBM %9.0f GB  @ %5.2f TB/s   %3ld dies                    |\n",
           system_config.memory_capacity / 1073741824.0, hbm_bw / 1e12, hbm_dies_tot);
    printf("|  HBF %9.0f GB  @ %5.2f TB/s   %3ld dies                    |\n",
           system_config.hbf_capacity / 1073741824.0, hbf_bw / 1e12, hbf_dies_tot);
    printf("|  LINK %-11s %-45s|\n",
           shared_bw > 0 ? "POOLED" : "PARTITIONED",
           shared_bw > 0 ? "both tiers share one link" : "tiers have separate paths");
    printf("+================================================================+\n\n");
    use_hbf = system_config.use_hbf;   // local drives the memory-check path below
  }
    if (system_config.experiment_mode == "h3_paper") {
    auto forbid = [&](bool bad, const std::string& what) {
      if (bad) fail("experiment_mode: h3_paper -- " + what +
                    " is an extension and must be off for paper reproduction.");
    };
    forbid(system_config.hbm_oversubscribe_fraction > 0.0,
           "hbm_oversubscribe_fraction (spill / HBF writes / GC / wear)");
    forbid(arch != "legacy" && arch != "cascaded",
           "architecture '" + arch + "'");
    forbid(system_config.attn_compute_efficiency != 1.0,
           "attn_compute_efficiency");
    forbid(system_config.shared_kv_sparsity != 1.0, "shared_kv_sparsity");

    auto require = [&](bool present, const std::string& key,
                       const std::string& why) {
      if (!present) std::cerr << "[h3_paper] " << key << " not set -- " << why
                              << ". State this in the assumptions table.\n";
    };
    require((bool)config["system"]["allreduce_hierarchical"],
            "allreduce_hierarchical",
            "paper says only \"ring all-reduce\"; the two kinds differ ~2x");
    require((bool)config["simulation"]["cag_amortize_shared_compute"],
            "cag_amortize_shared_compute", "paper gives no mechanism");
    require((bool)config["system"]["hbm_reserve_fraction"],
            "hbm_reserve_fraction", "CALIBRATED, not derived");
    require((bool)config["simulation"]["kv_cache_precision_byte"],
            "kv_cache_precision_byte", "derived from the paper's 540 GB figure");
    require((bool)config["system"]["distribution"]["context_parallel_degree"],
            "context_parallel_degree", "needed to reach 32 GPUs");

    std::cout << "[experiment_mode] h3_paper -- extensions disabled\n";
  } else {
    std::cout << "[experiment_mode] h3_extended\n";
  }
  // NVLink Config // 
  if(config["system"]["nvlink_gen"].as<int>() == 4){
    system_config.device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen (H100)
    system_config.device_ict_latency = 0.8 * 1000; // ns
  }
  else if(config["system"]["nvlink_gen"].as<int>() == 5){
    system_config.device_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen (B100, B200)
    system_config.device_ict_latency = 0.8 * 1000; // ns
  }else{
    fail("Not support NVLink generation");
  }

  // InfiniBand Config // 
  if(config["system"]["infiniband_gen"].as<int>() == 400){
    system_config.node_ict_bandwidth = 50.0 * 1000 * 1000 * 1000; // B/s Infiniband NDR
    system_config.node_ict_latency = 0.13 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 800){
    system_config.node_ict_bandwidth = 100.0 * 1000 * 1000 * 1000; // B/s InfiniBand XDR
    system_config.node_ict_latency = 0.13 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 3600){
    system_config.node_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 7200){
    system_config.node_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else{
    fail("Not support InfiniBand generation");
  }
  
  system_config.num_node = num_node;
  system_config.num_device = num_device;


  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;


  system_config.parallel_execution =
      config["system"]["optimization"]["parallel_execution"].as<bool>();
  system_config.hetero_subbatch =
      config["system"]["optimization"]["hetero_subbatch"].as<bool>();
  system_config.disagg_system =
      config["system"]["optimization"]["disagg_system"].as<bool>(); 
  system_config.use_low_unit_moe_only =
      config["system"]["optimization"]["use_low_unit_moe_only"].as<bool>();      
  system_config.use_ramulator =
      config["system"]["optimization"]["use_ramulator"].as<bool>();

  system_config.use_flash_mla =
      config["system"]["optimization"]["use_flash_mla"].as<bool>();
  system_config.use_flash_attention =
      config["system"]["optimization"]["use_flash_attention"].as<bool>();

  // kv cache reuse
  system_config.reuse_kv_cache =
      config["system"]["optimization"]["reuse_kv_cache"].as<bool>();
  system_config.kv_cache_reuse_rate =
      config["system"]["optimization"]["kv_cache_reuse_rate"].as<double>();
  
  // prefill mode or decode mode
  system_config.prefill_mode =
      config["system"]["optimization"]["prefill_mode"].as<bool>();
  system_config.decode_mode =
      config["system"]["optimization"]["decode_mode"].as<bool>();
  assertTrue((system_config.prefill_mode == false) || (system_config.decode_mode == false), 
            "prefill mode and decode mode is incompatible");

  assertTrue((system_config.parallel_execution == false) || (system_config.use_low_unit_moe_only == false),
            "parallel_execution & use_low_unit_moe_only are not compatible");
            
  if(system_config.prefill_mode){
    std::cout << "[Prefill Mode] Output Length is modified into 1" << std::endl;
  }

  if(system_config.decode_mode){
    std::cout << "[Decode Mode] Current Length of sequences is modified into input_len" << std::endl;
  }

  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::GPU;
  } else if (!processor_type.compare("LOGIC")) {
    system_config.processor_type = {ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::LOGIC;
    system_config.low_processor_type = ProcessorType::LOGIC;
  } else if (!processor_type.compare("GPU+LOGIC")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::LOGIC;
    // system_config.parallel_execution = true;
  } else if (!processor_type.compare("GPU+PIM")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::PIM};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::PIM;
  }

  std::string expert_file_path;

  ModelConfig model_config;

  if (!model_name.compare("mixtral")) {
    model_config = mixtral;
  } else if (!model_name.compare("openMoE")) {
    model_config = openMoE;
  } else if (!model_name.compare("llama7bMoE")) {
    model_config = llama7bMoE;
  } else if (!model_name.compare("llama3_405B")) {
    model_config = llama3_405B;
  } else if (!model_name.compare("grok1")) {
    model_config = grok1;
  } else if (!model_name.compare("deepseekV3")) {
    model_config = deepseekV3;
  }else if (!model_name.compare("llama4_scout")) {
    model_config = llama4_scout;
  }else if (!model_name.compare("llama4_maverick")) {
    model_config = llama4_maverick;
  }  else if (!model_name.compare("glam")) {
  model_config = glam;
  }
  else {
    fail("No model configuration of " + model_name);
  }

  model_config.e_tp_dg =
      config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  model_config.ne_tp_dg =
      config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();

  // H3: context_parallel_degree > 1 splits the shared CAG cache across the
  // extra devices within a KV-head's TP group, letting ne_tp_dg exceed
  // num_kv_heads (see model_config.h and SelfAttentionParallel).
  model_config.context_parallel_degree =
      config["system"]["distribution"]["context_parallel_degree"].as<int>(1);
  assertTrue(model_config.ne_tp_dg % model_config.context_parallel_degree == 0,
            "ne_tp_dg must be divisible by context_parallel_degree");
  assertTrue(model_config.num_kv_heads %
                (model_config.ne_tp_dg / model_config.context_parallel_degree) == 0,
            "num_kv_heads must be divisible by ne_tp_dg/context_parallel_degree");

  model_config.compressed_kv =
      config["system"]["optimization"]["compressed_kv"].as<bool>();
  model_config.use_absorb =
      config["system"]["optimization"]["use_absorb"].as<bool>();
  model_config.skewness =
      config["simulation"]["skewness"].as<double>();
  
  model_config.precision_byte = config["simulation"]["precision_byte"].as<int>();
  // FP8-doubles-compute is a pre-existing (non-H3) convention, unvalidated
  // against the paper's own methodology -- kept as a sensitivity knob
  // (default on = today's exact behavior) rather than silently changed.
  // See ASSUMPTIONS.md.
  bool fp8_compute_doubling =
      config["simulation"]["fp8_compute_doubling"].as<bool>(true);
  if(model_config.precision_byte == 1 && fp8_compute_doubling){ // if FP8 or INT8
    system_config.compute_peak_flops *= 2; // system_config has FP16 peak FLOPS information
  }

  // KV cache precision defaults to precision_byte unless overridden --
  // e.g. the paper's own FP8-weight/FP16-KV-cache mix (see model_config.h).
  int kv_cache_precision_byte =
      config["simulation"]["kv_cache_precision_byte"].as<int>(0);
  model_config.kv_cache_precision_byte =
      kv_cache_precision_byte > 0 ? kv_cache_precision_byte : model_config.precision_byte;

  system_config.exit_out_of_memory = config["simulation"]["exit_out_of_memory"].as<bool>();
  system_config.mem_cap_limit = config["simulation"]["mem_cap_limit"].as<bool>();

  model_config.dataset = data_name;

  if (!data_name.compare("synthesis")) {
    expert_file_path = "none";
    model_config.input_len = input_len;
    model_config.output_len = output_len;

    // CAG: remainder of the context window beyond ISL+OSL is the shared
    // pre-computed KV cache (paper Sec. IV-A.2), read by every sequence.
    if (context_window > 0) {
      assertTrue(context_window > input_len + output_len,
                "context_window must exceed input_len + output_len");
      model_config.shared_kv_cache_len = context_window - input_len - output_len;
      // H3/CAG: charge the shared context's attention work once per batch
      // rather than per sequence (paper Sec. II-A). See ASSUMPTIONS.md.
      model_config.cag_amortize_shared_compute =
          config["simulation"]["cag_amortize_shared_compute"].as<bool>(false);
      if (model_config.context_parallel_degree > 1) {
        assertTrue(model_config.shared_kv_cache_len %
                      model_config.context_parallel_degree == 0,
                  "shared_kv_cache_len must be divisible by context_parallel_degree");
      }
    }
  } else {
    expert_file_path =
        "../expert_data/experts_" + model_name + "_" + data_name + ".csv";
  }

  if((system_config.decode_mode == true) && (model_config.output_len <= 1)){
    fail("[Decode Mode] Output length must be larger than 1");
  }

  // long max_batch_size = 128;
  if (max_process_token == 0) {
    // max_process_token = 8192 * 16;
    max_process_token = 65536 * 8;
  }
  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, expert_file_path,
                        max_batch_size, 8192, max_process_token);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  bool out_of_memory = use_hbf ? cluster->checkH3MemorySize() : cluster->checkMemorySize();
  cluster->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = iter;

  int ne_tp_dg = model_config.ne_tp_dg;
  int ne_dp_dg = system_config.num_device * num_node / ne_tp_dg;

  int precision_bytes = model_config.precision_byte;

  std::string file_name;
  if(system_config.prefill_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_prefill.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_prefill.csv";
    }
  }
  else if(system_config.decode_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_decode.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_decode.csv";
    }
  }
  else{
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + ".csv";
    }
  }
  if (out_of_memory && config["simulation"]["exit_out_of_memory"].as<bool>()) {
    std::cout << "Out of Memory: " << file_name << std::endl;
    return 0;
  }
  scheduler->getActualArrivalTime(total_iter);
  stat_list = cluster->runIteration(total_iter, file_name);
  // TopModuleGraph::Ptr top1 = cluster->get_device(8)->top_module_graph;
  std::string gantt_file_path =
      config["log"]["gantt_directory"].as<std::string>();

  if(config["log"]["export_gantt"].as<bool>()){
    cluster->exportGantt(gantt_file_path);
  }

  if(config["log"]["print_log"].as<bool>()){
    TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
    top0->print_timeboard();
  }

  // H3: static per-device TDP, used to turn the throughput already in
  // stat_list/the exported CSV (tokens/time, from Stat.process_token and
  // Stat.latency) into throughput-per-power (paper Fig. 6). TDP is a
  // constant draw independent of access pattern, so unlike the DRAM
  // access-energy model it is computed once here rather than accumulated
  // during simulation.
  double per_device_power_watts =
      system_config.gpu_tdp +
      system_config.num_cube * system_config.hbm_tdp_per_cube +
      (system_config.use_hbf
           ? system_config.num_hbf_cube * system_config.hbf_tdp_per_cube
           : 0.0);
  double total_power_watts = num_device * num_node * per_device_power_watts;
  std::cout << "H3: per-device power " << per_device_power_watts
            << " W, total system power " << total_power_watts << " W"
            << std::endl;

  // H3: aggregate throughput from stat_list and append one row to
  // results.log, so a comparison across HBM-only vs H3 runs (batch size,
  // throughput, throughput-per-power) can be reconstructed later without
  // re-parsing every run's stdout/CSV by hand. See extract_results.py.
  // Only "t2t" entries are the per-decode-step stats (one per scheduler
  // tick that actually processed a batch; runIteration's stat_list also
  // carries "e2e"/"sum" bookkeeping entries from addLatency() with
  // unrelated semantics). Stat.time is cumulative simulated time since
  // the run started, not this step's duration -- Stat.latency is the
  // actual per-step device time, so tokens/latency (summed and averaged
  // across steps) is the correct steady-state decode throughput.
  long long total_tokens = 0;
  time_ns total_latency = 0;
  for (const Stat& s : stat_list) {
    if (s.type != "t2t") {
      continue;
    }
    total_tokens += s.process_token;
    total_latency += s.latency;
  }
  double throughput_tps = total_latency > 0 ? total_tokens / (total_latency / 1.0e9) : 0.0;
  double throughput_per_power =
      total_power_watts > 0 ? throughput_tps / total_power_watts : 0.0;

  // Term-by-term attention breakdown (closed-form reconciliation aid).
  {
    auto &st = cluster->get_device(0)->status;
    double np = st.attn_dbg_calls > 0 ? st.attn_dbg_n_private / st.attn_dbg_calls : 0.0;
    std::cout << "\n[ATTN BREAKDOWN] (ms, device 0, summed over all iterations)\n"
              << "  mean n_private (private ctx tokens actually used) = " << np << "\n"
              << "  T_comp  score=" << st.attn_c_score/1e6
              << "  softmax=" << st.attn_c_softmax/1e6
              << "  ctx=" << st.attn_c_ctx/1e6
              << "  TOTAL=" << st.attn_compute_time/1e6 << "\n"
              << "  T_mem   score_priv=" << st.attn_m_score_priv/1e6
              << "  score_shared=" << st.attn_m_score_shared/1e6
              << "  ctx_priv=" << st.attn_m_ctx_priv/1e6
              << "  ctx_shared=" << st.attn_m_ctx_shared/1e6
              << "  TOTAL=" << st.attn_memory_time/1e6 << "\n\n";
  }

  // Which side of the roofline bound each linear layer (device 0). Printed so
  // a memory-model change that max() hides behind compute is visible.
  {
    auto &st = cluster->get_device(0)->status;
    std::cout << "[LINEAR BOUND] (device 0) compute_bound_ops="
              << st.linear_compute_bound_ops
              << "  memory_bound_ops=" << st.linear_memory_bound_ops << "\n\n";
  }

  // H3 Action Item 2: HBM->HBF private-KV spill traffic, aggregated across
  // devices. Simulated wall clock is total_latency below, so this block must
  // stay after the stat_list loop -- move it if you reorder.
  uint64_t hbf_wr_logical = 0, hbf_wr_physical = 0, hbf_rd = 0;
  uint64_t mig_out = 0, mig_in = 0;
  for (int d = 0; d < num_device * num_node; d++) {
    cluster->get_device(d)->kv_manager().drain_flash();
  }
  for (int d = 0; d < num_device * num_node; d++) {
    auto& s = cluster->get_device(d)->kv_manager().stats();
    // Totals, not just migrations: hbf_logical_write_bytes covers only
    // whole-sequence spills. Per-step appends are a separate population
    // (hbf_append_*) and are where write amplification actually arises,
    // since they arrive at 512 B against a 4096 B page.
    hbf_wr_logical  += s.hbf_total_logical_write_bytes();
    hbf_wr_physical += s.hbf_total_physical_write_bytes();
    hbf_rd          += s.hbf_read_bytes;
    mig_out         += s.migrations_out_count;
    mig_in          += s.migrations_in_count;
  }
  cluster->get_device(0)->kv_manager().dump("device0");
  std::cout << "[HBF WRITE] aggregate over " << (num_device * num_node)
            << " devices: logical " << hbf_wr_logical / 1073741824.0
            << " GiB, physical " << hbf_wr_physical / 1073741824.0
            << " GiB, reads " << hbf_rd / 1073741824.0
            << " GiB, migrations " << mig_out << "/" << mig_in << "\n";
  std::string results_log_path = "../results.log";
  bool write_header = !std::ifstream(results_log_path).good();
  std::ofstream results_log(results_log_path, std::ios::app);
  if (write_header) {
      results_log << "timestamp,git_rev,model,gpu_gen,use_hbf,hbf_bandwidth_scale,"
                    "architecture,hbm_gb,hbf_gb,hbm_bw_tbs,hbf_bw_tbs,shared_link,"
                    "num_node,num_device,ne_tp_dg,context_parallel_degree,"
                    "fp8_compute_doubling,cag_amortize_shared_compute,"
                    "hbm_reserve_fraction,allreduce_hierarchical,"
                    "attn_compute_efficiency,shared_kv_sparsity,attn_compute_ms,attn_memory_ms,"
                    "context_window,input_len,"
                    "output_len,requested_max_batch_size,final_max_batch_size,"
                    "throughput_tps,per_device_power_w,total_power_w,"
                    "throughput_per_power,"
                    "hbf_write_gib_logical,hbf_write_gib_physical,"
                    "hbf_read_gib,hbf_migrations_out,hbf_migrations_in,"
                    "hbf_write_gib_per_ktoken\n";
  }
  results_log << std::time(nullptr) << ","
              << git_revision() << ","
              << model_name << ","
              << gpu_gen << ","
              << use_hbf << ","
              << system_config.hbf_bandwidth_scale << ","
              << arch << ","
              << system_config.memory_capacity / 1073741824.0 << ","
              << system_config.hbf_capacity / 1073741824.0 << ","
              << system_config.memory_bandwidth / 1e12 << ","
              << system_config.hbf_bandwidth / 1e12 << ","
              << (system_config.shared_link_bandwidth > 0 ? 1 : 0) << ","
              << num_node << ","
              << num_device << ","
              << model_config.ne_tp_dg << ","
              << model_config.context_parallel_degree << ","
              << fp8_compute_doubling << ","
              << model_config.cag_amortize_shared_compute << ","
              << system_config.hbm_reserve_fraction << ","
              << system_config.allreduce_hierarchical << ","
              << system_config.attn_compute_efficiency << ","
              << system_config.shared_kv_sparsity << ","
              << cluster->get_device(0)->status.attn_compute_time / 1.0e6 << ","
              << cluster->get_device(0)->status.attn_memory_time / 1.0e6 << ","
              << context_window << ","
              << input_len << ","
              << output_len << ","
              << max_batch_size << ","
              << scheduler->total_batch_size << ","
              << throughput_tps << ","
              << per_device_power_watts << ","
              << total_power_watts << ","
              << throughput_per_power << ","
              << hbf_wr_logical / 1073741824.0 << ","
              << hbf_wr_physical / 1073741824.0 << ","
              << hbf_rd / 1073741824.0 << ","
              << mig_out << ","
              << mig_in << ","
              << (total_tokens > 0
                      ? (hbf_wr_physical / 1073741824.0) / (total_tokens / 1000.0)
                      : 0.0) << ","
              << (cluster->get_device(0)->kv_manager().flash_model_enabled()
              ? cluster->get_device(0)->kv_manager().flash_stats().waf()
              : 0.0) << "\n";
  results_log.close();

  return 0;
}