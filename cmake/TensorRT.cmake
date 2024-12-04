## usage:
## include(cmake/TensorRT.cmake)
## CUDA_LIBRARIES, CUDNN_LIBRARY, TENSORRT_LIBRARIES

# >>> CUDA >>>
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  INFO_LOG("CUDA is available!")
  set(CUDA_AVAIL ON)
else ()
  ERROR_LOG("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDAToolkit_FOUND)
INFO_LOG("CUDA Library Directory: ${CUDAToolkit_LIBRARY_DIR}")
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
  COMMAND grep libnvinfer
  OUTPUT_VARIABLE TRT_IS_DEB)

if (NOT TRT_IS_DEB)
  INFO_LOG("TensorRT is not installed through DEB package, should specify the environment parameter TENSORRT_DIR explicitly")
  set(TENSORRT_DIR $ENV{TENSORRT_DIR})
  include_directories(${TENSORRT_DIR}/include)
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${TENSORRT_DIR}/lib")
  # Attain TensorRT version
  set(NVINFER_VERSION_HEADER "${TENSORRT_DIR}/include/NvInferVersion.h")
else ()
  set(NVINFER_VERSION_HEADER "/usr/include/x86_64-linux-gnu/NvInferVersion.h")
endif ()

if (EXISTS ${NVINFER_VERSION_HEADER})
  file(STRINGS ${NVINFER_VERSION_HEADER} NVINFER_VERSION_CONTENT REGEX "NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)")
  foreach (line IN LISTS NVINFER_VERSION_CONTENT)
    if (line MATCHES "#define NV_TENSORRT_MAJOR\ +([0-9]+)")
      set(NV_TENSORRT_MAJOR "${CMAKE_MATCH_1}")
    elseif (line MATCHES "#define NV_TENSORRT_MINOR\ +([0-9]+)")
      set(NV_TENSORRT_MINOR "${CMAKE_MATCH_1}")
    elseif (line MATCHES "#define NV_TENSORRT_PATCH\ +([0-9]+)")
      set(NV_TENSORRT_PATCH "${CMAKE_MATCH_1}")
    elseif (line MATCHES "#define NV_TENSORRT_BUILD\ +([0-9]+)")
      set(NV_TENSORRT_BUILD "${CMAKE_MATCH_1}")
    endif ()
  endforeach ()
  set(TENSORRT_VERSION "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}.${NV_TENSORRT_PATCH}")
  INFO_LOG("Detected TensorRT Version: ${TENSORRT_VERSION}")
else ()
  WARNING_LOG("NvInferVersion.h not found in ${TENSORRT_INCLUDE_DIR}")
endif ()

find_library(NVINFER NAMES nvinfer)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_library(NVONNX_PARSER NAMES nvonnxparser)

set(TENSORRT_LIBRARIES ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${NVONNX_PARSER})
set(TRT_AVAIL ON)

INFO_LOG("NVINFER: ${NVINFER}")
INFO_LOG("NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
INFO_LOG("NVONNX_PARSER: ${NVONNX_PARSER}")

if (NVINFER AND NVINFER_PLUGIN AND NVONNX_PARSER)
  INFO_LOG("TensorRT is available!")
  set(TRT_AVAIL ON)
else ()
  ERROR_LOG("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif ()

if (NOT (TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL))
  ERROR_LOG("TensorRT model won't be built, CUDA and/or CUDNN and/or TensorRT were not found.")
endif ()
