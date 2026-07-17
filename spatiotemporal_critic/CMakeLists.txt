cmake_minimum_required(VERSION 3.8)
project(spatiotemporal_critic)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_mppi_controller REQUIRED)
find_package(dynamic_tracker_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

set(deps
  rclcpp
  rclcpp_lifecycle
  pluginlib
  nav2_mppi_controller
  dynamic_tracker_msgs
  visualization_msgs
)

# La libreria del critic: un plugin caricato a runtime da MPPI.
add_library(${PROJECT_NAME} SHARED
  src/spatiotemporal_critic.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${deps})

# Dichiara al sistema pluginlib dove trovare il critic (base class = MPPI).
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