CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

PROJECT (llm_system)


add_library(llm_system SHARED)
add_compile_options(-fPIC)


include_directories(src)
include_directories(src/dram/ramulator2/src)
add_subdirectory(src)
add_subdirectory(eval)


if(APPLE)
  target_link_libraries(
      llm_system
      PRIVATE hardware
      PRIVATE -Wl,-all_load dram
      PRIVATE module
      PRIVATE model
      PRIVATE scheduler
  )
else()
  target_link_libraries(
      llm_system
      PRIVATE hardware
      PRIVATE -Wl,--whole-archive dram -Wl,--no-whole-archive
      #PRIVATE dram
      PRIVATE module
      PRIVATE model
      PRIVATE scheduler
  )
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

configure_file(dram_config_HBM3_80GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3_80GB.yaml COPYONLY)
configure_file(dram_config_HBM3E_192GB.yaml ${EXECUTABLE_OUTPUT_PATH}/dram_config_HBM3E_192GB.yaml COPYONLY)
configure_file(config.yaml ${EXECUTABLE_OUTPUT_PATH}/config.yaml COPYONLY)