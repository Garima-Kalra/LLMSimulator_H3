#include "module/module.h"

#include <iostream>

#include "common/assert.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {
Tensor::Ptr Module::operator()(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr return_tensor;
  device->top_module_graph->push_module_graph(getptr(), input);
  return_tensor = forward(input, sequences_metadata);
  device->top_module_graph->pop_module_graph(return_tensor);
  return return_tensor;
};

TensorVec Module::operator()(const TensorVec input,
                             BatchedSequence::Ptr sequences_metadata) {
  std::vector<Tensor::Ptr> return_tensor;
  device->top_module_graph->push_module_graph(getptr(), input);
  return_tensor = forward(input, sequences_metadata);
  device->top_module_graph->pop_module_graph(return_tensor);
  return return_tensor;
};

void Module::set_device(Device::Ptr device) {
  for (auto iter : tensor_list) {
    iter.second->set_device(device);
  }
  for (auto iter : module_list) {
    iter.second->set_device(device);
  }
};

long Module::size(std::vector<std::string> tag) {
  long act = 0;
  long weight = 0;
  long cache = 0;
  long cache_shared = 0;

  auto size_vector = get_size();
  act = size_vector.at(0);
  weight = size_vector.at(1);
  cache = size_vector.at(2);
  cache_shared = size_vector.at(3);

  long total = 0;
  for (std::string _tag : tag) {
    if (!_tag.compare("act")) {
      total += act;
    } else if (!_tag.compare("weight")) {
      total += weight;
    } else if (!_tag.compare("cache")) {
      total += cache;
    } else if (!_tag.compare("cache_shared")) {
      total += cache_shared;
    }
  }
  return total;
}

std::vector<long> Module::get_size() {
  long act = 0;
  long weight = 0;
  long cache = 0;
  // H3/CAG: bytes of the shared pre-computed KV cache, tracked separately
  // from "cache" (private, per-sequence) since it is allocated once
  // (not per batch slot) and, when use_hbf is on, lives in HBF capacity
  // rather than HBM capacity -- see Cluster::checkH3MemorySize.
  long cache_shared = 0;
  for (auto _tensor : tensor_list) {
    Tensor::Ptr tensor = _tensor.second;
    if (tensor->tag == "act") {
      act += tensor->getSize();
    } else if (tensor->tag == "weight") {
      weight += tensor->getSize();
    } else if (tensor->tag == "cache") {
      cache += tensor->getSize();
    } else if (tensor->tag == "cache_shared") {
      cache_shared += tensor->getSize();
    }
  }

  for (auto _module : module_list) {
    Ptr module = _module.second;
    auto size_vector = module->get_size();
    act += size_vector.at(0);

    // this line is for minimizing activation size. (calculate only the size of the largest activation per module)
    // act = std::max(act, size_vector.at(0));

    weight += size_vector.at(1);
    cache += size_vector.at(2);
    cache_shared += size_vector.at(3);
  }

  std::vector<long> size_vector;
  size_vector.push_back(act);
  size_vector.push_back(weight);
  size_vector.push_back(cache);
  size_vector.push_back(cache_shared);

  return size_vector;
}

void Module::add_tensor(Tensor::Ptr tensor) {
  add_tensor(tensor->name, tensor);
}

void Module::add_tensor(std::string name, Tensor::Ptr tensor) {
  if (tensor_list.find(name) == tensor_list.end()) {
    tensor_list.emplace(name, tensor);
  } else {
    tensor_list[name] = tensor;
  }
}

void Module::unset_tensor() {
  Tensor::Ptr tensor;
  Module::Ptr module;
  for (auto _tensor : tensor_list) {
    tensor = _tensor.second;
    tensor->unset();
  }

  for (auto _module : module_list) {
    module = _module.second;
    module->unset_tensor();
  }
}

void Module::set_dependency_tensor(
    std::vector<Tensor::Ptr>& dependency_tensor_list, Tensor::Ptr tensor) {
  device->cluster->set_dependency_tensor(dependency_tensor_list, tensor,
                                         device_list);
}

Tensor::Ptr Module::get_activation(std::string name, std::vector<int> shape,
                                   bool ready) {
  if (auto tensor = tensor_list.find(name); tensor != tensor_list.end()) {
    Tensor::Ptr return_tensor = tensor->second;
    return_tensor->shape = shape;
    return_tensor->setMemoryObject();
    if (ready) {
      return_tensor->set();
    };
    return return_tensor;
  } else {
    std::cout << name <<std::endl;
    fail("Unvalid activation tensor request");
    return nullptr;
  }
}

Tensor::Ptr Module::get_cache(std::string name, int seq_idx, int kv_idx, bool compressed_kv,
                              std::vector<int> shape) {
  if(compressed_kv){
    name = name + "_" + std::to_string(seq_idx);
  }
  else{
    name = name + "_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx);
  }
  if (auto tensor = tensor_list.find(name); tensor != tensor_list.end()) {
    Tensor::Ptr return_tensor = tensor->second;
    return_tensor->setMemoryObject();
    return return_tensor;
  } else {
    std::cout << name << std::endl;
    fail("Unvalid cached tensor request");
    return nullptr;
  }
}

Tensor::Ptr Module::get_shared_cache(std::string name) {
  if (auto tensor = tensor_list.find(name); tensor != tensor_list.end()) {
    Tensor::Ptr return_tensor = tensor->second;
    return_tensor->setMemoryObject();
    return return_tensor;
  } else {
    std::cout << name << std::endl;
    fail("Unvalid shared cache tensor request");
    return nullptr;
  }
}

  void Module::set_tensor_module() {
    Tensor::Ptr tensor;
    for (auto& _tensor : tensor_list) {
      tensor = _tensor.second;
      tensor->set_module(getptr());
    }
  }

  void Module::add_module_with_name(std::string name, Module::Ptr module) {
    if (module_list.find(name) == module_list.end()) {
      module_list.emplace(name, module);
    } else {
      module_list[name] = module;
    }
    device->add_module(module->module_map_name, module);
  }

  }  // namespace llm_system