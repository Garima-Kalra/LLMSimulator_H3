// module/transient_kv_manager.h
// H3: private KV residency between HBM and HBF + the write traffic migration
// generates. Tracks SEQUENCES, not Tensor objects: forward() only ever touches
// k_cache(0,0), tensors are built for the pre-cap batch, and each is sized at
// max_seq_len -- so per-tensor tracking measured nothing real.
// Must not include device.h or tensor.h (cycle); see the .cpp.
#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_system {

class Device;
class Tensor;

struct HbfParams {
  double read_bw_bytes_per_sec = 1.6e12;  // per stack
  double t_r_sec = 3e-6;
  double t_prog_sec = 100e-6;
  long page_bytes = 4096;
  long pages_per_block = 256;   // 1 MiB erase block
  double waf = 1.02;            // PLACEHOLDER: replaced by the GC model
  long pe_cycles = 100000;
  long capacity_bytes = 512L * 1024 * 1024 * 1024;
  int num_stacks = 1;

  double write_bw_bytes_per_sec() const {
    return read_bw_bytes_per_sec * (t_r_sec / t_prog_sec);
  }
  long block_bytes() const { return page_bytes * pages_per_block; }
  long physical_write_bytes(long logical_bytes) const {
    if (logical_bytes <= 0) return 0;
    long pages = (logical_bytes + page_bytes - 1) / page_bytes;
    return static_cast<long>(pages * page_bytes * waf);
  }
  double tbw_bytes(long capacity_used_by_static_data) const {
    long usable = capacity_bytes * num_stacks - capacity_used_by_static_data;
    return usable > 0 ? double(usable) * double(pe_cycles) : 0.0;
  }
};

struct SeqRecord {
  long bytes = 0;
  long len = 0;
  uint64_t last_touch_step = 0;
  bool in_hbf = false;
  uint64_t spill_count = 0;
  std::list<int>::iterator lru_it;
  bool lru_linked = false;
};

struct TransientKvStats {
  uint64_t hbf_logical_write_bytes = 0;
  uint64_t hbf_physical_write_bytes = 0;
  uint64_t hbf_read_bytes = 0;
  uint64_t migrations_out_count = 0;
  uint64_t migrations_in_count = 0;

  // Append traffic: per-step token growth of sequences already in HBF. The
  // small-write population the write-combining buffer targets; counted
  // separately because its GC behaviour is completely different.
  uint64_t hbf_append_logical_bytes = 0;
  uint64_t hbf_append_physical_bytes = 0;
  uint64_t hbf_append_count = 0;

  double hbf_write_time_sec = 0.0;
  double hbf_read_time_sec = 0.0;

  uint64_t peak_hbm_kv_bytes = 0;
  uint64_t steps_observed = 0;
  uint64_t peak_tracked_seqs = 0;

  uint64_t hbf_static_weight_bytes = 0;
  uint64_t hbf_static_shared_kv_bytes = 0;
  uint64_t hbf_static_tensor_count = 0;

  uint64_t hbf_static_bytes() const {
    return hbf_static_weight_bytes + hbf_static_shared_kv_bytes;
  }
  uint64_t hbf_total_logical_write_bytes() const {
    return hbf_logical_write_bytes + hbf_append_logical_bytes;
  }
  uint64_t hbf_total_physical_write_bytes() const {
    return hbf_physical_write_bytes + hbf_append_physical_bytes;
  }
  double bytes_per_migration_out() const {
    return migrations_out_count
               ? double(hbf_logical_write_bytes) / migrations_out_count
               : 0.0;
  }
  double bytes_per_append() const {
    return hbf_append_count
               ? double(hbf_append_logical_bytes) / hbf_append_count
               : 0.0;
  }
  double effective_waf() const {
    uint64_t l = hbf_total_logical_write_bytes();
    return l ? double(hbf_total_physical_write_bytes()) / l : 0.0;
  }
};

class TransientKvManager {
 public:
  TransientKvManager() = default;

  void configure(Device* device, long hbm_capacity_bytes,
                 long hbm_reserved_bytes);
  void set_hbf_params(const HbfParams& p) { hbf_ = p; }
  const HbfParams& hbf_params() const { return hbf_; }

  void set_kv_geometry(int num_layers, int num_kv_heads, int head_dim,
                       int precision_bytes);
  long kv_bytes_per_token() const { return kv_bytes_per_token_; }

  void advance_step();
  void account_hbf_static(const std::string& tag, long bytes);

  // MAIN ENTRY POINT: call once per step with the live batch. Sequences
  // absent from the call are treated as finished and released.
  void sync_batch(const std::vector<int>& seq_ids,
                  const std::vector<long>& seq_lens);

  // Opt-in: let spilling drive Tensor::in_hbf and thus the timing model.
  void set_perturb_cost_model(bool on) { perturb_cost_model_ = on; }

  const TransientKvStats& stats() const { return stats_; }
  long hbm_used_bytes() const { return hbm_used_bytes_; }
  long hbm_budget_for_kv_bytes() const { return hbm_budget_for_kv_bytes_; }

  double projected_lifetime_years(double wall_clock_sec,
                                  long hbf_static_bytes) const;
  void dump(const char* label) const;

  // Deprecated no-ops: keep attention.cpp compiling. Per-tensor tracking is
  // not meaningful here. Remove the call sites when convenient.
  void register_private_kv(const std::shared_ptr<Tensor>&) {}
  void on_kv_touch(const std::shared_ptr<Tensor>&) {}

 private:
  Device* device_ = nullptr;
  HbfParams hbf_;
  long hbm_budget_for_kv_bytes_ = 0;
  long hbm_used_bytes_ = 0;
  long kv_bytes_per_token_ = 0;
  uint64_t current_step_ = 0;
  bool perturb_cost_model_ = false;

  std::unordered_map<int, SeqRecord> records_;
  std::list<int> lru_order_;

  TransientKvStats stats_;

  void lru_touch(int id);
  void lru_unlink(int id);
  void make_room_if_needed();
  void spill_to_hbf(int id);
  void bring_back_to_hbm(int id);
  void release(int id);
};

}  // namespace llm_system
