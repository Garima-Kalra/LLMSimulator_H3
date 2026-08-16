# LLMSimulator

## Overview
LLMSimulator simulator is a c++ based cycle-accurate simulator, which based on graph execution of Large Language Models. This simulator supports state-of-the-art LLMs such as DeepSeek, Llama, Mixtral and etc. This simulator supports not only Multi-Head Attention (MHA) mechanism, but also Grouped-Query Attention(GQA), Multi-Query Attention(MQA) and Multi-head Latent Attention (MLA). LLMSimulator equipped with simulation of Mixture of Expert (MoE). It integrates with modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) for detailed memory modeling. LLMSimulator can evaluate various type of GPU generation such as H100, B100 and B200, and also including bank-level PIM, bank-group-level PIM, and Logic-PIM.

Key features:
- Supports flexible input/output length, batch sizes, request injection rates, and multi-node hardware configurations
- Models energy consumption and performance metrics across various memory systems

## Prerequisites
- Compiler: g++ version 11.4.0
- cmake, clang++

LLMSimulator is tested under the following system.




## Getting Started
### Building LLMSimulator
1. Clone the repository
```bash
   $ git clone https://github.com/scale-snu/LLMSimulator.git
   $ cd LLMSimulator
   $ git submodule update --init --recursive
```
2. Apply patch
```bash
   $ cd src/dram/ramulator2
   $ git apply ../../../patch/ramulator2_pim.patch
   $ cd ../../../
```

3. Build executable files
```bash
   $ mkdir build && cd build
   $ cmake ..
   $ make -j 
```

### How to run
LLMSimulator has config file (config.yaml) and you can modify it with your configuration. After modifying config.yaml and saving it, you can run with command below
```bash
   $ ./run > test.log
```

## Contact
Sungmin Yun sungmin.yun@snu.ac.kr

Kwanhee Kyung kwanhee.kyung@scale.snu.ac.kr

Juhwan Cho juhwan.cho@scale.snu.ac.kr

## Note
This simulator builds upon the simulator introduced in the MICRO 2024 paper “Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.”# LLMsim
# LLMsim

## Build notes (H3 fork)
1. `git submodule update --init --recursive`
2. `cd src/dram/ramulator2 && git apply ../../../patch/ramulator2_pim.patch && cd ../../../`
3. On GCC 13+, also apply `ramulator2_local_fixes.patch` (adds missing `#include <cstdint>`)
4. `mkdir build && cd build && cmake .. && make -j`
