# 检查ROS版本
if (DEFINED ENV{ROS_VERSION})
  if ($ENV{ROS_VERSION} STREQUAL 1)
    INFO_LOG("ROS1 is available!")
    # 导入 catkin 库
    find_package(catkin REQUIRED
      COMPONENTS
      geometry_msgs
      sensor_msgs
      roscpp
      std_msgs
      pcl_ros
      tf)
    catkin_package()
  elseif ($ENV{ROS_VERSION} STREQUAL 2)
    INFO_LOG("ROS2 is available!")
    find_package(ament_cmake_auto REQUIRED)
  endif ()
endif ()
