#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <iostream>

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

  std::string results_log_path = "../results.log";
  bool write_header = !std::ifstream(results_log_path).good();
  std::ofstream results_log(results_log_path, std::ios::app);
  if (write_header) {
    results_log << "timestamp,git_rev,model,gpu_gen,use_hbf,hbf_bandwidth_scale,"
                   "num_node,num_device,ne_tp_dg,context_parallel_degree,"
                   "fp8_compute_doubling,cag_amortize_shared_compute,"
                   "hbm_reserve_fraction,allreduce_hierarchical,"
                   "attn_compute_efficiency,shared_kv_sparsity,attn_compute_ms,attn_memory_ms,"
                   "context_window,input_len,"
                   "output_len,requested_max_batch_size,final_max_batch_size,"
                   "throughput_tps,per_device_power_w,total_power_w,"
                   "throughput_per_power\n";
  }
  results_log << std::time(nullptr) << ","
              << git_revision() << ","
              << model_name << ","
              << gpu_gen << ","
              << use_hbf << ","
              << system_config.hbf_bandwidth_scale << ","
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
              << throughput_per_power << "\n";
  results_log.close();

  return 0;
}
