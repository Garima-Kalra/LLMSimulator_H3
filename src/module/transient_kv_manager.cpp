// module/transient_kv_manager.cpp
// Only this TU needs the complete Device definition (device.h includes the
// header, so including device.h there would cycle).

#include "module/transient_kv_manager.h"

#include <algorithm>
#include <cstdio>

#include "module/tensor.h"
#include "hardware/device.h"

namespace llm_system {

void TransientKvManager::configure(Device* device, long hbm_capacity_bytes,
                                   long hbm_reserved_bytes) {
  device_ = device;
  hbm_budget_for_kv_bytes_ = (hbm_capacity_bytes > hbm_reserved_bytes)
                                 ? (hbm_capacity_bytes - hbm_reserved_bytes)
                                 : 0;
}

void TransientKvManager::set_kv_geometry(int num_layers, int num_kv_heads,
                                         int head_dim, int precision_bytes) {
  kv_bytes_per_token_ = 2L * num_layers * num_kv_heads * head_dim *
                        precision_bytes;
}

void TransientKvManager::enable_flash_model(int max_streams,
                                            double flush_threshold,
                                            long spare_capacity_bytes,
                                            double gc_low_water) {
  wbuf_.configure(max_streams, hbf_.block_bytes(), flush_threshold);
  flash_.configure(spare_capacity_bytes, hbf_.block_bytes(), hbf_.page_bytes,
                   hbf_.pe_cycles, gc_low_water, AllocPolicy::kPerSequence);
  flash_enabled_ = true;
}

void TransientKvManager::drain_flash() {
  if (!flash_enabled_) return;
  // Partial blocks still occupy a whole block each. Credit them to the
  // append stats -- omitting them made total physical < total logical, which
  // is impossible, and was the tell that they were being dropped.
  long resident = wbuf_.resident_bytes();
  long f = wbuf_.drain();
  if (f > 0) {
    long phys = flash_.write(resident, f, 0);
    stats_.hbf_append_physical_bytes += phys;
    stats_.hbf_write_time_sec += double(phys) / hbf_.write_bw_bytes_per_sec();
  }
  stats_.buffer_flushes = wbuf_.flushes();
  stats_.buffer_wasted_bytes = wbuf_.wasted_bytes();
  stats_.buffer_resident_bytes = 0;
}

void TransientKvManager::advance_step() {
  current_step_++;
  stats_.steps_observed = current_step_;
}

void TransientKvManager::account_hbf_static(const std::string& tag,
                                            long bytes) {
  if (bytes <= 0) return;
  if (tag == "weight") {
    stats_.hbf_static_weight_bytes += (uint64_t)bytes;
  } else if (tag == "cache_shared") {
    stats_.hbf_static_shared_kv_bytes += (uint64_t)bytes;
  } else {
    return;
  }
  stats_.hbf_static_tensor_count++;
}

// LRU: std::list + stored iterator -> O(1) removal. The old deque version
// linear-scanned per touch: O(n^2) overall, which hung on large batches.
void TransientKvManager::lru_unlink(int id) {
  auto it = records_.find(id);
  if (it == records_.end() || !it->second.lru_linked) return;
  lru_order_.erase(it->second.lru_it);
  it->second.lru_linked = false;
}

void TransientKvManager::lru_touch(int id) {
  auto it = records_.find(id);
  if (it == records_.end()) return;
  lru_unlink(id);
  lru_order_.push_back(id);
  it->second.lru_it = std::prev(lru_order_.end());
  it->second.lru_linked = true;
}

void TransientKvManager::release(int id) {
  auto it = records_.find(id);
  if (it == records_.end()) return;
  // A finished sequence invalidates every flash page it owns. With
  // per-sequence allocation those are whole blocks, so they become erasable
  // with ZERO copying -- the death-correlation property that makes KV the
  // friendliest possible flash workload.
  if (flash_enabled_ && it->second.in_hbf) flash_.invalidate_sequence(id);
  if (!it->second.in_hbf) hbm_used_bytes_ -= it->second.bytes;
  lru_unlink(id);
  records_.erase(it);
}

// Driven once per step from the scheduler's live batch.
void TransientKvManager::sync_batch(const std::vector<int>& seq_ids,
                                    const std::vector<long>& seq_lens) {
  if (kv_bytes_per_token_ <= 0) return;  // geometry not set; nothing to model
  const size_t n = std::min(seq_ids.size(), seq_lens.size());

  // 1. Retire sequences that left the batch.
  std::vector<int> live(seq_ids.begin(), seq_ids.begin() + n);
  std::sort(live.begin(), live.end());
  std::vector<int> gone;
  for (const auto& kv : records_) {
    if (!std::binary_search(live.begin(), live.end(), kv.first)) {
      gone.push_back(kv.first);
    }
  }
  for (int id : gone) release(id);

  // 2. Admit / grow the sequences that are present.
  for (size_t i = 0; i < n; i++) {
    const int id = seq_ids[i];
    const long want = seq_lens[i] * kv_bytes_per_token_;
    auto it = records_.find(id);

    if (it == records_.end()) {
      SeqRecord rec;
      rec.bytes = want;
      rec.len = seq_lens[i];
      rec.last_touch_step = current_step_;
      rec.in_hbf = false;
      records_.emplace(id, rec);
      hbm_used_bytes_ += want;
      lru_touch(id);
    } else {
      const long grew = want - it->second.bytes;
      if (it->second.in_hbf) {
        // Sequence lives in flash: this step's new tokens are an APPEND, a
        // small write. This is the population the write-combining buffer is
        // meant to absorb -- counted separately from whole-seq migrations.
        if (grew > 0) {
          stats_.hbf_append_logical_bytes += grew;
          stats_.hbf_append_count += 1;
          long phys;
          if (flash_enabled_) {
            // Combine per-(layer,sequence) appends until a block's worth
            // accumulates; only then does flash see a write and a block get
            // consumed. Amplification becomes a function of the flush
            // threshold instead of a hardcoded constant.
            long flushed = wbuf_.append(id, grew);
            if (flushed > 0) {
              // flushed is a whole number of blocks. The logical payload in
              // them is (blocks x threshold); the rest is flush waste, which
              // is exactly the amplification being measured.
              long blocks = flushed / wbuf_.block_bytes();
              long logical_in = blocks * wbuf_.threshold_bytes();
              phys = flash_.write(logical_in, flushed, id);
            } else {
              phys = 0;
            }
            stats_.buffer_resident_bytes = (uint64_t)wbuf_.resident_bytes();
            stats_.buffer_flushes = wbuf_.flushes();
            stats_.buffer_wasted_bytes = wbuf_.wasted_bytes();
          } else {
            phys = hbf_.physical_write_bytes(grew);  // legacy fixed WAF
          }
          stats_.hbf_append_physical_bytes += phys;
          if (phys > 0) {
            stats_.hbf_write_time_sec +=
                double(phys) / hbf_.write_bw_bytes_per_sec();
          }
        }
      } else {
        hbm_used_bytes_ += grew;
      }
      it->second.bytes = want;
      it->second.len = seq_lens[i];
      it->second.last_touch_step = current_step_;
      lru_touch(id);
    }
  }

  if (hbm_used_bytes_ > (long)stats_.peak_hbm_kv_bytes) {
    stats_.peak_hbm_kv_bytes = hbm_used_bytes_;
  }
  if (records_.size() > stats_.peak_tracked_seqs) {
    stats_.peak_tracked_seqs = records_.size();
  }

  make_room_if_needed();
}

// Deliberately dumb policy: evict least-recently-touched while over budget.
// The goal is to quantify write traffic, not to design an optimal policy.
void TransientKvManager::make_room_if_needed() {
  while (hbm_used_bytes_ > hbm_budget_for_kv_bytes_) {
    int victim = -1;
    for (int id : lru_order_) {
      auto it = records_.find(id);
      if (it != records_.end() && !it->second.in_hbf) {
        victim = id;
        break;
      }
    }
    if (victim < 0) break;  // everything resident is already spilled
    spill_to_hbf(victim);
  }
}

void TransientKvManager::spill_to_hbf(int id) {
  auto it = records_.find(id);
  if (it == records_.end() || it->second.in_hbf) return;
  SeqRecord& rec = it->second;

  rec.in_hbf = true;
  rec.spill_count++;
  hbm_used_bytes_ -= rec.bytes;

  const long logical = rec.bytes;
  // A whole-sequence migration is ~62 MB of contiguous block-aligned data
  // (126 layers x ~1010 tokens x 512 B) -- it already fills 62 whole blocks,
  // so it needs no write combining and goes straight to flash.
  const long physical = flash_enabled_ ? flash_.write(logical, id)
                                       : hbf_.physical_write_bytes(logical);
  stats_.hbf_logical_write_bytes += logical;
  stats_.hbf_physical_write_bytes += physical;
  stats_.migrations_out_count += 1;
  stats_.hbf_write_time_sec += double(physical) / hbf_.write_bw_bytes_per_sec();

  // NOT calling device_->run_ideal(kWrite,...): run_ideal() resetCounter()s,
  // so a mid-forward() call is erased by Device::execution(); and its cost
  // model is HBM channel geometry, not NAND tPROG. Accumulate analytically.

  // Move to the front so it is not immediately reconsidered.
  lru_unlink(id);
  lru_order_.push_front(id);
  rec.lru_it = lru_order_.begin();
  rec.lru_linked = true;
}

void TransientKvManager::bring_back_to_hbm(int id) {
  auto it = records_.find(id);
  if (it == records_.end() || !it->second.in_hbf) return;
  SeqRecord& rec = it->second;

  rec.in_hbf = false;
  hbm_used_bytes_ += rec.bytes;
  stats_.hbf_read_bytes += rec.bytes;
  stats_.migrations_in_count += 1;
  stats_.hbf_read_time_sec += double(rec.bytes) / hbf_.read_bw_bytes_per_sec;

  lru_touch(id);
  make_room_if_needed();
}

double TransientKvManager::projected_lifetime_years(
    double wall_clock_sec, long hbf_static_bytes) const {
  const uint64_t phys = stats_.hbf_total_physical_write_bytes();
  if (wall_clock_sec <= 0.0 || phys == 0) return -1.0;
  double rate = double(phys) / wall_clock_sec;
  double tbw = hbf_.tbw_bytes(hbf_static_bytes);
  if (rate <= 0.0 || tbw <= 0.0) return -1.0;
  return (tbw / rate) / (365.0 * 24.0 * 3600.0);
}

void TransientKvManager::dump(const char* label) const {
  const double GiB = 1024.0 * 1024.0 * 1024.0;
  std::printf("=== TransientKvManager [%s] ===\n", label ? label : "");
  std::printf("  steps                : %lu\n",
              (unsigned long)stats_.steps_observed);
  std::printf("  KV bytes / token     : %ld\n", kv_bytes_per_token_);
  std::printf("  tracked seqs (peak)  : %lu\n",
              (unsigned long)stats_.peak_tracked_seqs);
  std::printf("  HBM KV budget        : %.3f GiB\n",
              hbm_budget_for_kv_bytes_ / GiB);
  std::printf("  HBM KV peak / now    : %.3f / %.3f GiB\n",
              stats_.peak_hbm_kv_bytes / GiB, hbm_used_bytes_ / GiB);
  std::printf("  HBF static  weights  : %.3f GiB\n",
              stats_.hbf_static_weight_bytes / GiB);
  std::printf("  HBF static  sharedKV : %.3f GiB\n",
              stats_.hbf_static_shared_kv_bytes / GiB);
  std::printf("  HBF static  TOTAL    : %.3f GiB  (%lu tensors)\n",
              stats_.hbf_static_bytes() / GiB,
              (unsigned long)stats_.hbf_static_tensor_count);
  std::printf("  migrate  logical     : %.3f GiB  (%lu moves, %.1f KiB each)\n",
              stats_.hbf_logical_write_bytes / GiB,
              (unsigned long)stats_.migrations_out_count,
              stats_.bytes_per_migration_out() / 1024.0);
  std::printf("  append   logical     : %.3f GiB  (%lu appends, %.1f B each)\n",
              stats_.hbf_append_logical_bytes / GiB,
              (unsigned long)stats_.hbf_append_count,
              stats_.bytes_per_append());
  std::printf("  HBF writes  physical : %.3f GiB  (WAF %.3f)\n",
              stats_.hbf_total_physical_write_bytes() / GiB,
              stats_.effective_waf());
  if (flash_enabled_) {
    std::printf("  buffer resident      : %.3f GiB  (%lu flushes, %.3f GiB wasted)\n",
                stats_.buffer_resident_bytes / GiB,
                (unsigned long)stats_.buffer_flushes,
                stats_.buffer_wasted_bytes / GiB);
    flash_.dump(label);
  }
  std::printf("  HBF reads            : %.3f GiB\n",
              stats_.hbf_read_bytes / GiB);
  std::printf("  migrations out / in  : %lu / %lu\n",
              (unsigned long)stats_.migrations_out_count,
              (unsigned long)stats_.migrations_in_count);
  std::printf("  HBF time  wr / rd    : %.6f / %.6f s\n",
              stats_.hbf_write_time_sec, stats_.hbf_read_time_sec);
}

}  // namespace llm_system