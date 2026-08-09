#pragma once
#include <memory>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"
#include "module/base.h"

namespace llm_system {

class Tensor;

struct ExecStatus {
  ProcessorType processor_type;
  time_ns time = 0.0;
  time_ns total_duration = 0.0;
  time_ns compute_duration = 0.0;
  time_ns memory_duration = 0.0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;

  hw_metric opb = 0.0;
  util compute_util = 0.0;
  util memory_util = 0.0;
  // time_ns pim_duration = 0.0;

  counter_t generic_read_cmd = 0.0;
  counter_t generic_write_cmd = 0.0;
  counter_t compute_pim_cmd = 0.0;
  counter_t move_pim_cmd = 0.0;

  counter_t act_count = 0;
  counter_t read_count = 0;
  counter_t write_count = 0;
  counter_t all_act_count = 0;
  counter_t all_read_count = 0;
  counter_t all_write_count = 0;
  counter_t ref_count = 0;

  bool parallel_execution = false;

  ExecStatus& operator+=(const ExecStatus& rhs) {
    memory_duration += rhs.memory_duration;
    generic_read_cmd += rhs.generic_read_cmd;
    compute_pim_cmd += rhs.compute_pim_cmd;
    move_pim_cmd += rhs.compute_pim_cmd;

    act_count += rhs.act_count;
    read_count += rhs.read_count;
    write_count += rhs.write_count;
    all_act_count += rhs.all_act_count;
    all_read_count += rhs.all_read_count;
    all_write_count += rhs.all_write_count;
    ref_count += rhs.ref_count;

    return *this;
  }
};

class StatusBoard {
 public:
  using Tensor_Ptr = std::shared_ptr<Tensor>;
  StatusBoard() = default;

  // accumulate
  time_ns device_time = 0;

  time_ns high_time = 0;  // time of GPU
  time_ns low_time = 0;   // time of PIM

  // Roofline instrumentation for the decode-attention crossover study:
  // accumulated attention compute vs. attention memory time, kept separate
  // so the compute/memory crossing can be located empirically instead of
  // inferred from a ratio. See ASSUMPTIONS.md.
  time_ns attn_compute_time = 0;
  time_ns attn_memory_time = 0;

  // Term-by-term breakdown for closed-form reconciliation. Sums to the two
  // totals above; dumped by eval/test.cpp so every component of T_comp and
  // T_mem can be checked against an independent hand derivation.
  time_ns attn_c_score = 0, attn_c_softmax = 0, attn_c_ctx = 0;
  time_ns attn_m_score_priv = 0, attn_m_score_shared = 0;
  time_ns attn_m_ctx_priv = 0, attn_m_ctx_shared = 0;
  double  attn_dbg_n_private = 0, attn_dbg_calls = 0;

  Tensor_Ptr tensor;

  TensorVec tensor_vec;
  // TensorVec output_tensor_vec;

  // energy_nJ act;
  // energy_nJ read;
  // energy_nJ write;
  // energy_nJ tsv;
  // energy_nJ interposer_io;
  energy_nJ act_energy;
  energy_nJ read_energy;
  energy_nJ write_energy;

  energy_nJ all_act_energy;
  energy_nJ all_read_energy;
  energy_nJ all_write_energy;

  energy_nJ act_energy_load;
  energy_nJ read_energy_load;
  energy_nJ write_energy_load;

  energy_nJ all_act_energy_load;
  energy_nJ all_read_energy_load;
  energy_nJ all_write_energy_load;

  energy_nJ mac_energy;

  util compute_util;
  util memory_util;

  // record
  time_ns start_time = 0;
  time_ns end_time = 0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;
  hw_metric opb = 0.0;
  bool isTensorVec;
  std::vector<int> input_tensor_shape;
  std::vector<int> output_tensor_shape;
  std::vector<std::vector<int>> input_tensor_vec_shape;
  std::vector<std::vector<int>> output_tensor_vec_shape;
  ProcessorType processor_type;
  bool parallel_execution = false;
};

}  // namespace llm_system