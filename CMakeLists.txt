cmake_minimum_required(VERSION 3.12)
project(rangenet_pp)

include(cmake/Color.cmake)

# >>> 设置宏定义 >>>
if (DEFINED PERFORMANCE_LOG)
  add_compile_definitions(PERFORMANCE_LOG)
endif ()

# >>> 编译选项 >>>
# 设置 C++标准
# note：此处 std=c++14 不能改为 std=c++17，否则会报错（与 NVCC 有关），具体原理未知
INFO_LOG("CMAKE_BUILD_TYPE：${CMAKE_BUILD_TYPE}")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14 -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -O0")
set(CMAKE_CXX_STANDARD 17)

# >>> 导入 ROS 架构下的相关代码 >>>
include(cmake/ROS.cmake)

# >>> 导入三方库 >>>
include(cmake/ThirdParty.cmake)

include_directories(
  include
  src/ops
  src/utils)

# 处理点云 IO
add_library(pointcloud_io SHARED src/utils/pointcloud_io.cpp)
target_link_libraries(pointcloud_io ${PCL_LIBRARIES})

# CUDA 算子
cuda_add_library(project_ops src/ops/project_kernel.cu src/ops/project.cpp)

# 后处理
include_directories(src/utils)
add_library(postprocess src/utils/postprocess.cpp)
target_link_libraries(postprocess ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(rangenet_lib src/network/net.cpp src/network/netTensorRT.cpp)

target_link_libraries(rangenet_lib
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  ${TENSORRT_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${PCL_LIBRARIES}
  project_ops
)

add_executable(demo src/demo.cpp)

target_link_libraries(demo
  ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
)

# 检查 ROS 版本
if (DEFINED ENV{ROS_VERSION})
  if ($ENV{ROS_VERSION} STREQUAL 1)
    include_directories(
      ${catkin_INCLUDE_DIRS}
    )
    add_executable(ros1_demo src/ros1_demo.cpp)
    target_link_libraries(ros1_demo ${catkin_LIBRARIES} ${OpenCV_LIBS}
      rangenet_lib
      pointcloud_io
      postprocess
    )
  elseif ($ENV{ROS_VERSION} STREQUAL 2)
    INFO_LOG("ROS2 is available!")
    ament_auto_add_executable(ros2_demo src/ros2_demo.cpp)
    ament_target_dependencies(ros2_demo
      OpenCV
    )
    target_link_libraries(ros2_demo
      ${OpenCV_LIBS}
      rangenet_lib
      pointcloud_io
      postprocess
    )
    ament_auto_package(
      INSTALL_TO_SHARE
      launch
      model
      data
    )
  endif ()
endif ()
