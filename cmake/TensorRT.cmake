## usage:
## include(cmake/TensorRT.cmake)
## CUDA_LIBRARIES, CUDNN_LIBRARY, TENSORRT_LIBRARIES

# suppress eigen warning: "-Wcpp Please use cuda_runtime_api.h or cuda_runtime.h instead"
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Wno-deprecated -Wno-cpp")
# suppress the nvcc warning: "__device__ anotation is ignored on a function"
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

# >>> CUDA >>>
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if (CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )
  include_directories(${CUDA_INCLUDE_DIRS})
  INFO_LOG("CUDA is available!")
  set(CUDA_AVAIL ON)
else ()
  ERROR_LOG("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)
INFO_LOG("CUDA Libs: ${CUDA_LIBRARIES}")
INFO_LOG("CUDA Headers: ${CUDA_INCLUDE_DIRS}")

# >>> CUDNN >>>
# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY_PATH cudnn
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/
  DOC "CUDNN library."
  )

if (CUDNN_LIBRARY_PATH)
  INFO_LOG("CUDNN is available!")
  set(CUDNN_LIBRARY ${CUDNN_LIBRARY_PATH})
  set(CUDNN_AVAIL ON)
else ()
  ERROR_LOG("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif ()
INFO_LOG("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")

# >>> TensorRT >>>
option(TRT_AVAIL "TensorRT available" OFF)

# 检查tensorrt的安装方式
execute_process(
  COMMAND dpkg -l
  COMMAND grep nv-tensorrt
  OUTPUT_VARIABLE TRT_IS_DEB)

if (NOT TRT_IS_DEB)
  INFO_LOG("TensorRT is not installed through DEB package, should specify the environment parameter TENSORRT_DIR explicitly")
  set(TENSORRT_DIR $ENV{TENSORRT_DIR})
  include_directories(${TENSORRT_DIR}/include)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${TENSORRT_DIR}/lib")
endif ()

find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_library(NVONNX_PARSER NAMES nvonnxparser)
find_library(NVCAFFE_PARSER NAMES nvcaffe_parser)

set(TENSORRT_LIBRARIES ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${NVONNX_PARSER} ${NVCAFFE_PARSER})
set(TRT_AVAIL ON)

INFO_LOG("NVINFER: ${NVINFER}")
INFO_LOG("NVPARSERS: ${NVPARSERS}")
INFO_LOG("NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
INFO_LOG("NVONNX_PARSER: ${NVONNX_PARSER}")

if (NVINFER AND NVPARSERS AND NVINFER_PLUGIN AND NVONNX_PARSER)
  INFO_LOG("TensorRT is available!")
  set(TRT_AVAIL ON)
else ()
  ERROR_LOG("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif ()

if (NOT (TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL))
  ERROR_LOG("TensorRT model won't be built, CUDA and/or CUDNN and/or TensorRT were not found.")
endif ()
