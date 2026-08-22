#include "hardware/layer_impl.h"

#include <algorithm>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "hardware/device.h"
#include "module/tensor.h"

namespace llm_system {

time_ns h3MemoryDuration(SystemConfig config, hw_metric hbm_bytes, hw_metric hbf_bytes) {
  time_ns hbm_duration = hbm_bytes / config.memory_bandwidth * 1000 * 1000 * 1000;
  if (!config.use_hbf || hbf_bytes <= 0) {
    return hbm_duration;
  }
  hw_metric hbf_bandwidth = config.hbf_bandwidth * config.hbf_bandwidth_scale;
  time_ns hbf_duration = hbf_bytes / hbf_bandwidth * 1000 * 1000 * 1000;

  // Topologies without independent paths: both tiers cross one link, so the
  // transfers serialize. HBF media bandwidth remains a second, separate cap.
  if ((config.link_topology == "cascaded" ||
       config.link_topology == "shared_base") &&
      config.shared_link_bandwidth > 0) {
    time_ns link_duration = (hbm_bytes + hbf_bytes) /
                            config.shared_link_bandwidth * 1000 * 1000 * 1000;
    return std::max(link_duration, hbf_duration);
  }
  return std::max(hbm_duration, hbf_duration);

  // "independent" (side-by-side / co-located) and "shared_base": each stack
  // has its own path to the GPU, so the transfers overlap.
  return std::max(hbm_duration, hbf_duration);
}

ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor) {
  assertFalse(tensor->in_hbf,
             "HBF is only supported in ideal/roofline mode (use_ramulator: off); "
             "Ramulator2 has no NAND-flash timing model for HBF.");
  CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type,
                                 tensor->getSize());
  ExecStatus exec_status;
  if (!device->checkExecutionCache(exec_status, key)) {
    DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
    dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
    device->run_ramulator(dram_request);
    exec_status = device->dram_interface->getExecStatus(); 
    device->addExecutionCache(exec_status, key);
  }

  return exec_status;
};

ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor) {
  
  ExecStatus exec_status;
  long total_size = tensor->getSize();
  device->run_ideal(dram_request_type, tensor);
  exec_status = device->dram_interface->getExecStatus(); 
  return exec_status;
};

}  // namespace llm_system