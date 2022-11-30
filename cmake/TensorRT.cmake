## usage:
## include(cmake/TensorRT.cmake)
## CUDA_LIBRARIES, CUDNN_LIBRARY, TENSORRT_LIBRARIES

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Wno-deprecated -Wno-cpp")
# suppress eigen warning: "-Wcpp Please use cuda_runtime_api.h or cuda_runtime.h instead"
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

if (NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
endif ()

# >>> CUDA >>>
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if (CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "${Green}CUDA is available!${ColourReset}")
  set(CUDA_AVAIL ON)
else ()
  message(SEND_ERROR "${Red}CUDA NOT FOUND${ColourReset}")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)
message(STATUS "${Blue}CUDA Libs: ${CUDA_LIBRARIES}${ColourReset}")
message(STATUS "${Blue}CUDA Headers: ${CUDA_INCLUDE_DIRS}${ColourReset}")
# >>> CUDA >>>

# >>> CUDNN >>>
# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY_PATH cudnn
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/
  DOC "CUDNN library."
  )

if (CUDNN_LIBRARY_PATH)
  message(STATUS "${Green}[INFO] CUDNN is available!${ColourReset}")
  set(CUDNN_LIBRARY ${CUDNN_LIBRARY_PATH})
  set(CUDNN_AVAIL ON)
else ()
  message(SEND_ERROR "${Red}[ERROR] CUDNN is NOT Available${ColourReset}")
  set(CUDNN_AVAIL OFF)
endif ()
message(STATUS "${Blue}CUDNN_LIBRARY: ${CUDNN_LIBRARY}${ColourReset}")
# >>> CUDNN >>>

# >>> TensorRT >>>
option(TRT_AVAIL "TensorRT available" OFF)

# 检查tensorrt的安装方式
execute_process(
  COMMAND dpkg -l
  COMMAND grep nv-tensorrt
  OUTPUT_VARIABLE TRT_IS_DEB)

if (NOT TRT_IS_DEB)
  message(STATUS "${Red}[WARNING] Tensorrt is not installed through DEB package, should specify the LIBRARY_PATH and INCLUDE_PATH explicitly${ColourReset}")
  set(TENSORRT_DIR $ENV{HOME}/Application/TensorRT-8.4.1.5)
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

message(STATUS "${Blue}NVINFER: ${NVINFER}${ColourReset}")
message(STATUS "${Blue}NVPARSERS: ${NVPARSERS}${ColourReset}")
message(STATUS "${Blue}NVINFER_PLUGIN: ${NVINFER_PLUGIN}${ColourReset}")
message(STATUS "${Blue}NVONNX_PARSER: ${NVONNX_PARSER}${ColourReset}")

if (NVINFER AND NVPARSERS AND NVINFER_PLUGIN AND NVONNX_PARSER)
  message(STATUS "${Green}[INFO] TensorRT is available!${ColourReset}")
  set(TRT_AVAIL ON)
else ()
  message(SEND_ERROR "${Red}[ERROR] TensorRT is NOT Available${ColourReset}")
  set(TRT_AVAIL OFF)
endif ()
# >>> TensorRT >>>

if (NOT (TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL))
  message(FATAL_ERROR "[ERROR] Tensorrt model won't be built, CUDA and/or CUDNN and/or TensorRT were not found.")
endif ()
