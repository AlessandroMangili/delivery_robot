cmake_minimum_required(VERSION 3.8)
project(mppi_critics)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_mppi_controller REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(dynamic_tracker_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

set(deps
  rclcpp
  rclcpp_lifecycle
  pluginlib
  nav2_mppi_controller
  nav2_costmap_2d
  dynamic_tracker_msgs
  visualization_msgs
)

add_library(${PROJECT_NAME} SHARED
  src/spatio_temporal_critic.cpp
  src/detour_critic.cpp
  src/rotate_first_critic.cpp
  src/side_choice_critic.cpp
  src/clearance_critic.cpp
  src/heading_critic.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${deps})

pluginlib_export_plugin_description_file(nav2_mppi_controller critics.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION include
)
install(FILES critics.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${deps})

ament_package()
