find_package(ament_cmake_auto REQUIRED)
find_package(Boost REQUIRED)

ament_auto_package()
ament_auto_find_build_dependencies()

ament_auto_add_executable(ros2_demo src/ros2_demo.cpp)
target_link_libraries(ros2_demo
  ${OpenCV_LIBS}
  rangenet_lib
  pointcloud_io
  postprocess
  )