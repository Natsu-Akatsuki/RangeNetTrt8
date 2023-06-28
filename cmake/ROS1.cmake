# 导入catkin库
find_package(catkin REQUIRED
  COMPONENTS geometry_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf)

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

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(ros1_demo src/ros1_demo.cpp)
target_link_libraries(ros1_demo ${catkin_LIBRARIES} ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
  )