enable_language(CUDA)
# Set gcc and nvcc flags
INFO_LOG("CMAKE_BUILD_TYPE：${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif ()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Wno-deprecated")

set(CMAKE_CUDA_STANDARD 17)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
  # #20012-D: Suppress the warning message of "diagnose_suppress" in the CUDA code of Eigen header.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20012")
endif ()

# -Wno-cpp:
#    Suppress eigen warning: "-Wcpp Please use cuda_runtime_api.h or cuda_runtime.h instead"
# -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored:
#    Suppress the nvcc warning: "__device__ anotation is ignored on a function"
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-cpp -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

execute_process(
  COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0
  RESULT_VARIABLE result
  OUTPUT_VARIABLE compute_cap
  ERROR_VARIABLE error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "." "" compute_cap ${compute_cap})

if (NOT result EQUAL 0)
  ERROR_LOG("nvidia-smi failed with error: ${error}")
else ()
  INFO_LOG("Compute capability: ${compute_cap}")
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8 AND compute_cap EQUAL 89)
    WARNING_LOG("Current CUDA does not support ${compute_cap}, please use CUDA 11.8+, we will use 52 instead.")
    set(CMAKE_CUDA_ARCHITECTURES "52")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1 AND compute_cap EQUAL 86)
    WARNING_LOG("Current CUDA does not support ${compute_cap}, please use CUDA 11.1+, we will use 52 instead.")
    set(CMAKE_CUDA_ARCHITECTURES "52")
  else ()
    set(CMAKE_CUDA_ARCHITECTURES ${compute_cap})
  endif ()
endif ()