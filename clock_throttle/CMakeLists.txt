cmake_minimum_required(VERSION 3.8)
project(clock_throttle)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)

add_executable(clock_throttle_node src/clock_throttle_node.cpp)
ament_target_dependencies(clock_throttle_node rclcpp rosgraph_msgs)

install(TARGETS clock_throttle_node DESTINATION lib/${PROJECT_NAME})

ament_package()
