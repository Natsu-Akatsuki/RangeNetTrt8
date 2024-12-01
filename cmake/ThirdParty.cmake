# >>> 导入第三方库 >>>

# 导入Torch库
set(Torch_DIR $ENV{Torch_DIR})
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# 导入Eigen库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# 导入yaml-cpp库
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# 导入PCL库
# This is a custom PCL_DIR for docker of Ubuntu-22.04. You could change it to your own PCL_DIR.
set(PCL_DIR "/usr/local/share/pcl-1.13")
if (DEFINED PCL_DIR)
  INFO_LOG("Using Custom PCL_DIR：${PCL_DIR}")
endif ()

find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})
INFO_LOG("PCL_VERSION is ${PCL_VERSION}")

# 导入OpenCV库
find_package(OpenCV REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

# 导入CUDA、NVCC、TensorRT库
include(cmake/TensorRT.cmake)