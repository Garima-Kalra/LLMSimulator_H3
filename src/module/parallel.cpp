#include "module/parallel.h"

#include "common/assert.h"
#include "module/communication.h"
// ColumnParallelLinear //

namespace llm_system {

ColumnParallelLinear::ColumnParallelLinear(std::string& prefix,
                                           std::string& name, int input_size,
                                           int output_size,
                                           std::vector<int> device_list,
                                           Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0,
             "output_size mod parallel_num != 0");

  Module::Ptr linear =
      Linear::Create(module_map_name, "Linear", input_size,
                     output_size / parallel_num, device_list, device);
  add_module(linear);
}

Tensor::Ptr ColumnParallelLinear::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr linear = get_module("Linear");

  Tensor::Ptr output_tensor = (*linear)(input, sequences_metadata);
  return output_tensor;
}

// RowParallelLinear //

RowParallelLinear::RowParallelLinear(std::string& prefix, std::string& name,
                                     int input_size, int output_size,
                                     std::vector<int> device_list,
                                     Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0,
             "output_size mod parallel_num != 0");

  // std::vector<int> shape = {input_size / parallel_num, output_size};
  Module::Ptr linear =
      Linear::Create(module_map_name, "Linear", input_size / parallel_num,
                     output_size, device_list, device);
  add_module(linear);
}

Tensor::Ptr RowParallelLinear::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr linear = get_module("Linear");

  Tensor::Ptr output_tensor = (*linear)(input, sequences_metadata);
  return output_tensor;
}

BatchedColumnParallelLinear::BatchedColumnParallelLinear(std::string& prefix,
    std::string& name, int input_size, int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  duplicated_input(duplicated_input),
  use_plain_linear(use_plain_linear) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0, "output_size mod parallel_num != 0");
  num_heads_per_device = num_heads / parallel_num;

  if(!use_plain_linear){
    Module::Ptr batched_linear =
        BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size,
          output_size / num_heads, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
  else{
    Module::Ptr batched_linear =
    BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size,
      output_size / parallel_num, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
}

TensorVec BatchedColumnParallelLinear::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
    
  Module::Ptr batched_linear = get_module("Batched_Linear");
  TensorVec output_vec = (*batched_linear)(input, sequences_metadata);

  return output_vec;
}

// BatchedRowParallelLinear //

BatchedRowParallelLinear::BatchedRowParallelLinear(std::string& prefix, std::string& name,
    int input_size, int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list, Device::Ptr device)
  : Module(prefix, name, device, device_list),
  duplicated_input(duplicated_input),
  use_plain_linear(use_plain_linear) {
  int parallel_num = device_list.size();
  assertTrue(input_size % parallel_num == 0, "input_size mod parallel_num != 0");
  num_heads_per_device = num_heads / parallel_num;

  if(!use_plain_linear){
    Module::Ptr batched_linear =
        BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size / num_heads,
          output_size, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
  else{
    Module::Ptr batched_linear =
    BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size / parallel_num,
      output_size, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
}

TensorVec BatchedRowParallelLinear::forward(
    const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
    
      
    Module::Ptr batched_linear = get_module("Batched_Linear");
    TensorVec output_vec = (*batched_linear)(input, sequences_metadata);

    return output_vec;
}

// SelfAttentionParallel //

SelfAttentionParallel::SelfAttentionParallel(std::string& prefix,
                                             std::string& name, int head_dim,
                                             int num_heads, int num_kv_heads,
                                             int max_seq_len, int batch_size, int qk_rope_head_dim,
                                             bool compressed_kv,
                                             std::vector<int> device_list,
                                             Device::Ptr device)
    : Module(prefix, name, device, device_list),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim),
      compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  // H3: context_parallel_degree > 1 lets ne_tp_dg (parallel_num) exceed
  // num_kv_heads -- head_tp_dg is the effective TP degree for whole-KV-head
  // sharding, and the remaining context_parallel_degree devices within each
  // head's group instead split the shared CAG cache length (see
  // attention.cpp/attention_gen_impl.cpp). This is a pure roofline cost
  // model with no per-device head-identity branching (every device in the
  // TP group is modeled as owning an equal share, num_kv_heads/head_tp_dg,
  // regardless of rank), so num_heads/parallel_num is unaffected: the
  // summed FLOPs total is invariant to how parallel_num factors into
  // head_tp_dg x context_parallel_degree, since
  // attention_group_size * num_kv_heads_local == num_heads_local always.
  int cp_dg = device->model_config.context_parallel_degree;
  assertTrue(parallel_num % cp_dg == 0,
             "ne_tp_dg must be divisible by context_parallel_degree");
  int head_tp_dg = parallel_num / cp_dg;

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");
  assertTrue(num_kv_heads % head_tp_dg == 0,
             "num_kv_head mod (ne_tp_dg/context_parallel_degree) == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / head_tp_dg, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = SelfAttentionSum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / head_tp_dg, max_seq_len, batch_size, qk_rope_head_dim, device_list,
  device);
  add_module(attention_sum);

  Module::Ptr attention_gen = SelfAttentionGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / head_tp_dg, max_seq_len, batch_size, qk_rope_head_dim, device_list,
  device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / head_tp_dg, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_merge);

  // H3: when the shared cache is split cp_dg ways, each device only holds a
  // partial (online-softmax) attention result over its slice and needs to
  // combine with its cp_dg peers. Reuses AllReduce's ring-combine cost model
  // verbatim. Any cp_dg-sized subset of device_list gives an identical cost
  // -- AllReduce::forward's cost depends only on group size and the
  // (uniform, non-topology-aware) device_ict_latency/bandwidth, not on
  // which specific devices are in the group -- so the first cp_dg devices
  // are used unconditionally rather than computing this device's "true"
  // peers.
  has_context_merge = cp_dg > 1;
  if (has_context_merge) {
    std::vector<int> cp_device_list(device_list.begin(),
                                    device_list.begin() + cp_dg);
    auto context_merge = AllReduce::Create(module_map_name, "context_merge",
                                           cp_device_list, device);
    add_module(context_merge);
  }

  // auto attn_sync = Sync::Create(module_map_name, "AttnSync",
  //                               {device->device_total_rank}, device);
  // add_module(attn_sync);
}

Tensor::Ptr SelfAttentionParallel::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // Module::Ptr attention_sync = get_module("AttnSync");

  TensorVec tensor_vec;
  tensor_vec.resize(0);
  tensor_vec.push_back(input);
  tensor_vec = (*attention_split)(tensor_vec, sequences_metadata);

  Tensor::Ptr sum_out = (*attention_sum)(tensor_vec[0], sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(tensor_vec[1], sequences_metadata);

  if (has_context_merge) {
    Module::Ptr context_merge = get_module("context_merge");
    gen_out = (*context_merge)(gen_out, sequences_metadata);
  }

  //(*attention_sync)(input, sequences_metadata);

  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);
  // Tensor::Ptr output_tensor_out = (*attention_sum)(input,
  // sequences_metadata); Tensor::Ptr output_tensor = (*attention_gen)(input,
  // sequences_metadata);

  return output_tensor.at(0);
}

MultiLatentAttentionParallel::MultiLatentAttentionParallel(std::string& prefix,
    std::string& name, int head_dim,
    int num_heads, int num_kv_heads,
    int max_seq_len, int batch_size, int qk_rope_head_dim,
    bool compressed_kv, bool use_flash_mla, bool use_flash_attention,
    std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = MultiLatentAttentionSum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, use_flash_attention,
  device_list, device);
  add_module(attention_sum);

  Module::Ptr attention_gen = MultiLatentAttentionGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, compressed_kv, use_flash_mla, device_list,
  device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false,  device_list,
  device);
  add_module(attention_merge);
}

TensorVec MultiLatentAttentionParallel::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // Module::Ptr attention_sync = get_module("AttnSync");
  
  TensorVec tensor_vec;
  tensor_vec.push_back(input[0]);
  tensor_vec = (*attention_split)(tensor_vec, sequences_metadata);

  Tensor::Ptr sum_out = (*attention_sum)(tensor_vec[0], sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(tensor_vec[1], sequences_metadata);

  // //(*attention_sync)(input, sequences_metadata);

  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);

  return output_tensor;
}
AbsorbMLAParallel::AbsorbMLAParallel(std::string& prefix,
    std::string& name, int head_dim,
    int num_heads, int num_kv_heads,
    int max_seq_len, int batch_size, int qk_rope_head_dim,
    int kv_lora_rank, bool compressed_kv, bool use_flash_mla,
    std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, true, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = AbsorbMLASum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, device_list,
  device);
  add_module(attention_sum);

  Module::Ptr attention_gen = AbsorbMLAGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, compressed_kv, 
  use_flash_mla, device_list, device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, true, device_list,
  device);
  add_module(attention_merge);
}

TensorVec AbsorbMLAParallel::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // TODO: device가 처리해야하는 head갯수만큼, score (input (S,512) x c_kv (512, S))

  // input: (num head per device, S, 512)
  TensorVec tensor_vec = (*attention_split)(input, sequences_metadata);
  
  int num_heads_per_device = tensor_vec.size() / 2;

  Tensor::Ptr sum_tensor = tensor_vec[0];
  Tensor::Ptr gen_tensor = tensor_vec[num_heads_per_device];
  
  Tensor::Ptr sum_out = (*attention_sum)(sum_tensor, sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(gen_tensor, sequences_metadata);
  
  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);
  
  return output_tensor;
}
}  // namespace llm_system