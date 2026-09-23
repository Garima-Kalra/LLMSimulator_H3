// module/hbf_flash.h
//
// HBF write path: DRAM write-combining buffer + block-level flash with
// garbage collection. Write amplification is a MEASURED OUTPUT here, not the
// fixed HbfParams::waf constant it replaces.
//
// ---------------------------------------------------------------------------
// WHY FLASH IS WRITTEN AT ALL
// ---------------------------------------------------------------------------
// Under H3's placement flash holds weights and the shared pre-computed cache.
// Both are written once and then read-only, so in steady-state decode flash is
// never written. (Retention refresh was checked separately and is negligible:
// rewriting all 166 GB of static residency DAILY still gives ~995 years.)
//
// Flash is written only if HBM is deliberately over-subscribed and private KV
// spills. That produces two populations with completely different behaviour:
//
//   MIGRATION -- a whole cache moves at once. 126 layers x ~1010 tokens x
//     512 B = ~62 MB, i.e. 62 contiguous blocks. Block-aligned; bypasses the
//     buffer; amplification ~1.0.
//
//   APPEND -- once spilled, the request keeps generating. Each step it writes
//     2 (K+V) x kv_heads_per_device x head_dim x precision PER LAYER, which is
//     512 B against a 4096 B page. Unbuffered that is 8x. The write-combining
//     buffer exists solely for this.
//
// ---------------------------------------------------------------------------
// WHY ALLOCATION POLICY IS THE VARIABLE THAT MATTERS
// ---------------------------------------------------------------------------
// KV cache has an unusual property: a request's cache is written append-only
// and invalidated ALL AT ONCE when the request finishes. Lifetimes within a
// request are perfectly correlated.
//
//   kPerSequence -- one open block per request. A finished request invalidates
//     whole blocks, so collection copies nothing. WAF -> 1.0. Costs ~3.2 GB of
//     DRAM buffer at 4382 concurrent spilled requests.
//   kPerLayer    -- one open block per layer, requests pooled. 0.09 GB buffer,
//     but blocks mix requests with different finishing times, so they become
//     partially invalid and survivors must be relocated.
//   kGlobal      -- one open block. Minimal buffer, no death correlation.
//
// An earlier version of this model fixed the policy at kPerSequence, which is
// the most favourable case, and then observed no collection. That was
// circular: the allocator, not the workload, was deciding the answer. The
// policy is a parameter here so the comparison IS the experiment.
//
// ---------------------------------------------------------------------------
// TWO CORRECTIONS TO THE FIRST IMPLEMENTATION
// ---------------------------------------------------------------------------
//   1. collect() counted the survivor copy bytes but never actually relocated
//      them, so a victim block appeared to free its whole capacity when part
//      of it was still live. Occupancy therefore fell faster than it should
//      and collection looked cheaper than it is. Survivors are now written
//      into a fresh block, which is what a real FTL does.
//   2. maybe_collect() collected exactly one block even when still below the
//      free-block low-water mark. It now collects until the mark is met or no
//      further progress is possible.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llm_system {

// ---------------------------------------------------------------------------
// Write-combining buffer, resident in DRAM.
//
// The buffer costs no extra traffic: appended tokens are ALREADY in HBM (a
// spilled request still generates its new state there; only its older cache
// sits in flash), so buffering defers a flush rather than adding a copy. The
// sole cost is that a spilled request does not fully vacate HBM.
// ---------------------------------------------------------------------------
class WriteBuffer {
 public:
  void configure(int num_streams, long block_bytes, double flush_threshold) {
    block_bytes_ = block_bytes;
    threshold_bytes_ = (long)(block_bytes * flush_threshold);
    if (threshold_bytes_ < 1) threshold_bytes_ = 1;
    fill_.assign(num_streams > 0 ? num_streams : 1, 0);
  }

  // Returns raw bytes handed to flash (whole blocks), 0 while still filling.
  long append(int stream, long bytes) {
    if (fill_.empty() || bytes <= 0) return 0;
    int s = stream % (int)fill_.size();
    if (s < 0) s += (int)fill_.size();
    fill_[s] += bytes;
    long flushed = 0;
    while (fill_[s] >= threshold_bytes_) {
      flushed += block_bytes_;
      fill_[s] -= threshold_bytes_;
      flushes_++;
      wasted_ += block_bytes_ - threshold_bytes_;  // this IS the amplification
    }
    return flushed;
  }

  long drain() {
    long flushed = 0;
    for (auto& f : fill_) {
      if (f > 0) {
        flushed += block_bytes_;
        flushes_++;
        wasted_ += block_bytes_ - f;
        f = 0;
      }
    }
    return flushed;
  }

  uint64_t flushes() const { return flushes_; }
  uint64_t wasted_bytes() const { return wasted_; }
  long threshold_bytes() const { return threshold_bytes_; }
  long block_bytes() const { return block_bytes_; }
  long resident_bytes() const {
    long t = 0;
    for (auto f : fill_) t += f;
    return t;
  }
  long capacity_bytes() const {
    return (long)fill_.size() * threshold_bytes_;
  }

 private:
  long block_bytes_ = 1024 * 1024;
  long threshold_bytes_ = 768 * 1024;
  std::vector<long> fill_;
  uint64_t flushes_ = 0;
  uint64_t wasted_ = 0;
};

// ---------------------------------------------------------------------------
// Block-level flash with greedy garbage collection.
// ---------------------------------------------------------------------------
enum class AllocPolicy { kPerSequence, kPerLayer, kGlobal };

inline AllocPolicy parseAllocPolicy(const std::string& s) {
  if (s == "per_layer") return AllocPolicy::kPerLayer;
  if (s == "global") return AllocPolicy::kGlobal;
  return AllocPolicy::kPerSequence;
}
inline const char* allocPolicyName(AllocPolicy p) {
  switch (p) {
    case AllocPolicy::kPerLayer: return "per_layer";
    case AllocPolicy::kGlobal: return "global";
    default: return "per_sequence";
  }
}

struct FlashStats {
  uint64_t host_write_bytes = 0;       // what the workload produced
  uint64_t physical_write_bytes = 0;   // page-aligned + flush waste + GC copies
  uint64_t gc_copy_bytes = 0;
  uint64_t erase_count = 0;
  uint64_t gc_invocations = 0;
  uint64_t gc_blocks_reclaimed = 0;
  uint64_t blocks_allocated = 0;
  uint64_t device_full_events = 0;

  // Mean valid fraction of a block at erase time. The classic SSD relation is
  // WAF ~ 1/(1-u); reporting u lets that be checked rather than assumed.
  double victim_valid_frac_sum = 0.0;

  double waf() const {
    return host_write_bytes ? double(physical_write_bytes) / host_write_bytes
                            : 0.0;
  }
  double mean_victim_utilisation() const {
    return gc_invocations ? victim_valid_frac_sum / gc_invocations : 0.0;
  }
  double predicted_waf_from_utilisation() const {
    double u = mean_victim_utilisation();
    return (u < 1.0) ? 1.0 / (1.0 - u) : 0.0;
  }
};

class FlashDevice {
 public:
  void configure(long capacity_bytes, long block_bytes, long page_bytes,
                 long pe_cycles, double gc_low_water, AllocPolicy policy) {
    block_bytes_ = block_bytes;
    page_bytes_ = page_bytes;
    pe_cycles_ = pe_cycles;
    gc_low_water_ = gc_low_water;
    policy_ = policy;
    long n = capacity_bytes / block_bytes;
    if (n < 1) n = 1;
    valid_.assign(n, 0);
    written_.assign(n, 0);
    erases_.assign(n, 0);
    page_owners_.assign(n, {});
    last_writer_.assign(n, -1);
    free_.clear();
    free_.reserve(n);
    for (long i = n - 1; i >= 0; i--) free_.push_back(i);
    open_.clear();
    open_blocks_.clear();
    seq_blocks_.clear();
    scan_cursor_ = 0;
  }

  long write(long bytes, int seq_id, int layer = 0) {
    return write(bytes, bytes, seq_id, layer);
  }

  // logical_bytes = what the workload produced.
  // raw_bytes     = whole blocks the buffer hands over, incl. flush waste.
  // Recording raw_bytes as host writes would make WAF 1.0 by construction.
  long write(long logical_bytes, long raw_bytes, int seq_id, int layer = 0) {
    if (valid_.empty() || raw_bytes <= 0) return 0;
    stats_.host_write_bytes += logical_bytes;
    long physical = place(raw_bytes, seq_id, layer);
    stats_.physical_write_bytes += physical;
    return physical + maybe_collect();
  }

  // A finished request invalidates every page it owns. Under kPerSequence
  // those are whole blocks, so they become erasable with zero relocation --
  // the death-correlation property that makes KV cheap to collect.
  void invalidate_sequence(int seq_id) {
    auto it = seq_blocks_.find(seq_id);
    if (it == seq_blocks_.end()) return;
    for (long b : it->second) {
      if (b < 0 || b >= (long)valid_.size()) continue;
      auto po = page_owners_[b].find(seq_id);
      if (po == page_owners_[b].end()) continue;
      valid_[b] -= po->second;              // only THIS request's pages
      if (valid_[b] < 0) valid_[b] = 0;
      page_owners_[b].erase(po);
    }
    seq_blocks_.erase(it);
  }

  const FlashStats& stats() const { return stats_; }

  double occupancy() const {
    if (valid_.empty()) return 0.0;
    return 1.0 - double(free_.size()) / double(valid_.size());
  }

  // Endurance is limited by the WORST block, not the mean: the static region
  // (weights, shared cache) never wears at all, so a mean erase count flatters
  // the result badly.
  void erase_histogram(uint64_t& mean, uint64_t& p99, uint64_t& max) const {
    if (erases_.empty()) { mean = p99 = max = 0; return; }
    std::vector<uint64_t> e(erases_.begin(), erases_.end());
    std::sort(e.begin(), e.end());
    uint64_t sum = 0;
    for (auto x : e) sum += x;
    mean = sum / e.size();
    p99 = e[(size_t)(e.size() * 0.99) % e.size()];
    max = e.back();
  }

  double projected_life_years(double wall_clock_sec) const {
    if (wall_clock_sec <= 0) return -1.0;
    uint64_t mean, p99, mx;
    erase_histogram(mean, p99, mx);
    if (mx == 0) return -1.0;
    double spent = double(mx) / double(pe_cycles_);
    return (wall_clock_sec / spent) / (365.0 * 24.0 * 3600.0);
  }

  void dump(const char* label) const {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    uint64_t mean, p99, mx;
    erase_histogram(mean, p99, mx);
    std::printf("=== FlashDevice [%s] policy=%s ===\n", label ? label : "",
                allocPolicyName(policy_));
    std::printf("  blocks               : %zu total, %zu free (%.1f%% full)\n",
                valid_.size(), free_.size(), occupancy() * 100.0);
    std::printf("  host writes          : %.3f GiB\n",
                stats_.host_write_bytes / GiB);
    std::printf("  physical writes      : %.3f GiB\n",
                stats_.physical_write_bytes / GiB);
    std::printf("  GC copies            : %.3f GiB (%lu runs, %lu blocks)\n",
                stats_.gc_copy_bytes / GiB,
                (unsigned long)stats_.gc_invocations,
                (unsigned long)stats_.gc_blocks_reclaimed);
    std::printf("  MEASURED WAF         : %.4f\n", stats_.waf());
    std::printf("  victim utilisation u : %.4f  -> 1/(1-u) = %.4f\n",
                stats_.mean_victim_utilisation(),
                stats_.predicted_waf_from_utilisation());
    std::printf("  erases               : %lu total, mean %lu / p99 %lu / max %lu\n",
                (unsigned long)stats_.erase_count, (unsigned long)mean,
                (unsigned long)p99, (unsigned long)mx);
    if (stats_.device_full_events) {
      std::printf("  DEVICE FULL          : %lu writes could not be placed\n",
                  (unsigned long)stats_.device_full_events);
    }
  }

 private:
  long block_pages() const { return block_bytes_ / page_bytes_; }

  int alloc_key(int seq_id, int layer) const {
    switch (policy_) {
      case AllocPolicy::kPerLayer: return layer;
      case AllocPolicy::kGlobal: return 0;
      default: return seq_id;
    }
  }

  // Fill open blocks for this allocation key, taking fresh blocks as needed.
  long place(long raw_bytes, int seq_id, int layer) {
    long pages = (raw_bytes + page_bytes_ - 1) / page_bytes_;
    long physical = pages * page_bytes_;
    int key = alloc_key(seq_id, layer);
    long remaining = pages;
    while (remaining > 0) {
      long blk = open_block(key);
      if (blk < 0) { stats_.device_full_events++; break; }
      long room = block_pages() - written_[blk];
      long take = std::min(room, remaining);
      written_[blk] += take;
      valid_[blk] += take;
      page_owners_[blk][seq_id] += take;
      last_writer_[blk] = seq_id;
      seq_blocks_[seq_id].push_back(blk);
      remaining -= take;
      if (written_[blk] >= block_pages()) { open_.erase(key); open_blocks_.erase(blk); }
    }
    return physical;
  }

  long open_block(int key) {
    auto it = open_.find(key);
    if (it != open_.end() && written_[it->second] < block_pages()) {
      return it->second;
    }
    if (free_.empty()) {
      collect_until_free();
      if (free_.empty()) return -1;
    }
    long blk = free_.back();
    free_.pop_back();
    written_[blk] = 0;
    valid_[blk] = 0;
    page_owners_[blk].clear();
    last_writer_[blk] = -1;
    open_[key] = blk;
    open_blocks_.insert(blk);
    stats_.blocks_allocated++;
    return blk;
  }

  long maybe_collect() {
    if (valid_.empty()) return 0;
    double free_frac = double(free_.size()) / double(valid_.size());
    if (free_frac >= gc_low_water_) return 0;
    return collect_until_free();
  }

  // CORRECTION 2: collect until the low-water mark is met, not once.
  long collect_until_free() {
    long total = 0;
    int guard = 0;
    while (double(free_.size()) / double(valid_.size()) < gc_low_water_ &&
           guard++ < 4096) {
      long got = collect_one();
      if (got < 0) break;   // no reclaimable victim
      total += got;
    }
    return total;
  }

  // Greedy victim selection over a bounded rotating window rather than the
  // whole device. An exhaustive scan is O(blocks) per collection and O(n^2)
  // overall, which is intractable at 1.4 M blocks; real FTLs sample for the
  // same reason. The window is large enough that a fully-invalid block is
  // almost always found when one exists, which is the common case under
  // per-request allocation. Returns copy bytes, or -1 if no victim exists.
  long collect_one() {
    const long n = (long)valid_.size();
    const long window = std::min<long>(n, 4096);
    long best = -1, best_valid = block_pages() + 1;
    for (long k = 0; k < window; k++) {
      long b = (scan_cursor_ + k) % n;
      if (written_[b] == 0) continue;
      if (is_open(b)) continue;
      if (valid_[b] < best_valid) {
        best_valid = valid_[b];
        best = b;
        if (best_valid == 0) break;   // fully invalid: cannot do better
      }
    }
    scan_cursor_ = (best >= 0 ? best + 1 : scan_cursor_ + window) % n;
    if (best < 0) {
      // Nothing reclaimable in this window; sweep the rest before giving up.
      for (long b = 0; b < n; b++) {
        if (written_[b] == 0 || is_open(b)) continue;
        if (valid_[b] < best_valid) { best_valid = valid_[b]; best = b; }
      }
    }
    if (best < 0) return -1;

    long survivors = valid_[best];
    int  seq       = last_writer_[best];

    stats_.gc_invocations++;
    stats_.victim_valid_frac_sum += double(survivors) / double(block_pages());

    // Erase first so the victim is available to receive survivors if needed.
    erases_[best]++;
    stats_.erase_count++;
    written_[best] = 0;
    valid_[best] = 0;
    page_owners_[best].clear();
    last_writer_[best] = -1;
    free_.push_back(best);
    stats_.gc_blocks_reclaimed++;

    long copy_bytes = 0;
    if (survivors > 0) {
      // CORRECTION 1: survivors must be RELOCATED, not discarded. Without
      // this a victim appeared to free its whole capacity while part of it
      // was still live, so occupancy fell too fast and collection looked
      // cheaper than it is.
      copy_bytes = survivors * page_bytes_;
      stats_.gc_copy_bytes += copy_bytes;
      stats_.physical_write_bytes += copy_bytes;
      place(copy_bytes, seq >= 0 ? seq : 0, 0);
    }
    return copy_bytes;
  }

  bool is_open(long b) const { return open_blocks_.count(b) != 0; }

  long scan_cursor_ = 0;
  long block_bytes_ = 1024 * 1024;
  long page_bytes_ = 4096;
  long pe_cycles_ = 100000;
  double gc_low_water_ = 0.10;
  AllocPolicy policy_ = AllocPolicy::kPerSequence;

  std::vector<long> valid_, written_;
  std::vector<uint64_t> erases_;
  // A block may hold pages from MANY requests under per_layer / global
  // allocation, so ownership cannot be a single id. page_owners_[b] maps
  // request -> pages of block b it owns, and invalidation subtracts only that
  // request's share. An earlier version stored one int per block; invalidation
  // then silently failed for shared blocks, no block ever became reclaimable,
  // and the collector thrashed at u = 1.0 relocating fully-valid blocks.
  std::vector<std::unordered_map<int, long>> page_owners_;
  std::vector<int> last_writer_;                 // for survivor re-attribution
  std::vector<long> free_;
  std::unordered_map<int, long> open_;           // alloc key -> open block
  std::unordered_set<long> open_blocks_;         // O(1) is_open()
  std::unordered_map<int, std::vector<long>> seq_blocks_;  // O(1) invalidation

  FlashStats stats_;
};

}  // namespace llm_system