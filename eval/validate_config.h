// eval/validate_config.h
//
// Strict configuration validation.
//
// WHY THIS EXISTS
//   YAML accepts any key, and every read in test.cpp supplies a silent
//   default. A configuration can therefore be wrong in a way that still
//   produces plausible numbers. This has cost real results three times:
//
//     hbm_reserve_fraction  omitted from 45 of 55 configs -> ran at 0.0
//                           instead of 0.0821, so the architecture sweep
//                           and the replication were calibrated
//                           differently with nothing reporting it.
//     link_topology         absent when the headline runs were made, so
//                           they silently used the overlapping model.
//     hbf_stacks            hand-written key that no code reads. Three
//                           "different" flash configurations produced
//                           byte-identical results, invalidating a
//                           published finding.
//
//   Each was found by noticing a suspicious number, not by the tool
//   complaining. This makes the tool complain.
//
// WHAT IT DOES
//   Rejects any key not in the known set (catches typos and dead keys),
//   and warns when a key that materially changes results is absent
//   (catches silent defaults).
//
// USAGE  -- one line in main(), straight after loading the YAML:
//     YAML::Node config = YAML::LoadFile(config_path);
//     validateConfig(config);          // <-- add this
//
#pragma once

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace llm_system {

inline void validateConfig(const YAML::Node& config) {
  // ---- every key the code actually reads -------------------------------
  const std::set<std::string> top = {
      "system", "model", "serving", "simulation", "log"};

  const std::set<std::string> system_keys = {
      "gpu_gen", "num_node", "num_device", "processor_type", "use_hbf",
      "hbf_bandwidth_scale", "hbm_reserve_fraction", "attn_compute_efficiency",
      "shared_kv_sparsity", "allreduce_hierarchical", "architecture",
      "shoreline_slots", "dies_per_stack", "hbm_sites", "hbf_sites",
      "hbm_dies_per_site", "hbf_dies_per_site", "link_topology","hbm_oversubscribe_fraction", "nvlink_gen",
      "infiniband_gen", "distribution", "optimization", "experiment_mode"};

  const std::set<std::string> distribution_keys = {
      "expert_tensor_degree", "none_expert_tensor_degree",
      "context_parallel_degree"};

  const std::set<std::string> optimization_keys = {
      "parallel_execution", "hetero_subbatch", "disagg_system",
      "use_low_unit_moe_only", "use_ramulator", "compressed_kv", "use_absorb",
      "use_flash_mla", "use_flash_attention", "reuse_kv_cache",
      "kv_cache_reuse_rate", "prefill_mode", "decode_mode"};

  const std::set<std::string> model_keys = {"model_name"};
  const std::set<std::string> serving_keys = {"max_batch_size",
                                              "max_process_token"};
  const std::set<std::string> simulation_keys = {
      "iter", "data", "input_len", "output_len", "precision_byte",
      "kv_cache_precision_byte", "skewness", "mem_cap_limit",
      "exit_out_of_memory", "context_window", "cag_amortize_shared_compute",
      "fp8_compute_doubling"};
  const std::set<std::string> log_keys = {"print_log", "export_gantt",
                                          "output_directory",
                                          "gantt_directory"};

  std::vector<std::string> errors;
  std::vector<std::string> warnings;

  auto check = [&](const YAML::Node& node, const std::set<std::string>& allowed,
                   const std::string& path) {
    if (!node || !node.IsMap()) return;
    for (auto it = node.begin(); it != node.end(); ++it) {
      const std::string k = it->first.as<std::string>();
      if (allowed.find(k) == allowed.end()) {
        errors.push_back(path + k + "  <-- not read by any code");
      }
    }
  };

  check(config, top, "");
  check(config["system"], system_keys, "system: ");
  check(config["system"]["distribution"], distribution_keys,
        "system: distribution: ");
  check(config["system"]["optimization"], optimization_keys,
        "system: optimization: ");
  check(config["model"], model_keys, "model: ");
  check(config["serving"], serving_keys, "serving: ");
  check(config["simulation"], simulation_keys, "simulation: ");
  check(config["log"], log_keys, "log: ");

  // ---- keys whose absence silently changes results ---------------------
  // Anything here has a default that is NOT a safe no-op: omitting it
  // produces a different machine or a different calibration, quietly.
  auto require = [&](const YAML::Node& parent, const std::string& key,
                     const std::string& path, const std::string& why) {
    if (!parent || !parent[key]) warnings.push_back(path + key + " -- " + why);
  };

  require(config["system"], "hbm_reserve_fraction", "system: ",
          "defaults to 0.0; batch sizes will not be comparable with "
          "calibrated runs");
  require(config["simulation"], "kv_cache_precision_byte", "simulation: ",
          "KV cache precision differs from weight precision in H3");
  if (config["system"]["use_hbf"] &&
      config["system"]["use_hbf"].as<bool>(false)) {
    require(config["simulation"], "cag_amortize_shared_compute", "simulation: ",
            "determines whether the machine is compute- or memory-bound");
    require(config["system"], "architecture", "system: ",
            "defaults to \"legacy\"; the two-tier model is not used");
  }

  // ---- report ----------------------------------------------------------
  if (!warnings.empty()) {
    std::cerr << "\n[CONFIG WARNING] keys absent, silent defaults in use:\n";
    for (const auto& w : warnings) std::cerr << "  " << w << "\n";
  }
  if (!errors.empty()) {
    std::cerr << "\n[CONFIG ERROR] unrecognised keys:\n";
    for (const auto& e : errors) std::cerr << "  " << e << "\n";
    std::cerr << "\nA key the code does not read has no effect. If configs "
                 "differ only in\nsuch a key they describe the SAME machine "
                 "and will produce identical\nresults. Fix the spelling or "
                 "remove the key.\n\n";
    std::exit(1);
  }
  if (warnings.empty()) std::cerr << "[config] validated, all keys explicit\n";
}

}  // namespace llm_system