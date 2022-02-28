cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(rangenet_plusplus)

# ======================= COMPILE OPTION =======================
set(CMAKE_BUILD_TYPE RELEASE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations -Wno-deprecated -Wno-cpp")
# suppress eigen warning: "-Wcpp Please use cuda_runtime_api.h or cuda_runtime.h instead"
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

# suppress the nvcc warning: "__device__ anotation is ignored on a function"
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)
# ======================= COMPILE OPTION =======================

# ======================= MACRO OPTION =======================
# add_definitions(-DPERFORMANCE_LOG)
# ======================= MACRO OPTION =======================

# 导入catkin库
find_package(catkin REQUIRED
  COMPONENTS geometry_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


# 导入第三方库
set(Torch_DIR "$ENV{HOME}/application/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(yaml-cpp REQUIRED)
include_directories(${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR})

# TensorRT
include_directories($ENV{HOME}/application/TensorRT-8.2.3.0/include)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "$ENV{HOME}/application/TensorRT-8.2.3.0/lib")
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
find_library(NVONNX_PARSER NAMES nvonnxparser)
set(TENSORRT_LIBRARIES ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${NVONNX_PARSER})

# CUDA
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARIES cublas HINTS
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
include_directories(${CUDA_INCLUDE_DIRS})


include_directories(
  example
  include
  utils)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  INCLUDE_DIRS
  include
  LIBRARIES
  rangenet_lib
  CATKIN_DEPENDS
  DEPENDS
  YAML_CPP
)

add_library(pointcloud_io utils/pointcloud_io.cpp)
target_link_libraries(pointcloud_io ${PCL_LIBRARIES})

# cuda算子
include_directories(ops)
cuda_add_library(project_ops ops/project_kernel.cu ops/project.cpp)

# 后处理
include_directories(utils)
add_library(postprocess utils/postprocess.cpp)
target_link_libraries(postprocess ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(rangenet_lib src/net.cpp src/netTensorRT.cpp)

target_link_libraries(rangenet_lib
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  ${TENSORRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TORCH_LIBRARIES}
  project_ops
  ${PCL_LIBRARIES}
  )

add_executable(infer example/infer.cpp)

target_link_libraries(infer
  ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
  )

add_executable(pointcloud_semantic_segmentation src/pointcloud_semantic_segmentation.cpp)
target_link_libraries(pointcloud_semantic_segmentation ${catkin_LIBRARIES} ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess)
