enable_language(CUDA)
# Set gcc and nvcc flags
INFO_LOG("CMAKE_BUILD_TYPE：${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
# #20012-D: Suppress the warning message of "diagnose_suppress" in the CUDA code of Eigen header.
set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -diag-suppress=20012")

execute_process(
  COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
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
  set(CMAKE_CUDA_ARCHITECTURES ${compute_cap})
endif ()
