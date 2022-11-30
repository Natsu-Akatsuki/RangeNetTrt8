cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(rangenet_plusplus)

# ======================= COMPILE OPTION =======================
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)
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
set(Torch_DIR "$ENV{HOME}/Application/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(yaml-cpp REQUIRED)
include_directories(${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR})

include(cmake/TensorRT.cmake)

include_directories(
  include
  src/ops
  src/utils)

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
  DEPENDS
    YAML_CPP
)

# 处理点云IO
add_library(pointcloud_io SHARED src/utils/pointcloud_io.cpp)
target_link_libraries(pointcloud_io ${PCL_LIBRARIES})

# cuda算子
cuda_add_library(project_ops src/ops/project_kernel.cu src/ops/project.cpp)

# 后处理
include_directories(src/utils)
add_library(postprocess src/utils/postprocess.cpp)
target_link_libraries(postprocess ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(rangenet_lib src/network/net.cpp src/network/netTensorRT.cpp)

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

add_executable(single_shot_demo src/single_shot_demo.cpp)

target_link_libraries(single_shot_demo
  ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
  )

add_executable(semantic_segmentation_node src/semantic_segmentation_node.cpp)
target_link_libraries(semantic_segmentation_node ${catkin_LIBRARIES} ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
  )