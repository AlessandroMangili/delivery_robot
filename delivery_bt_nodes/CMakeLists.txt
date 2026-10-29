cmake_minimum_required(VERSION 3.8)
project(delivery_bt_nodes)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(nav_msgs REQUIRED)

add_library(delivery_progress_recovery_bt_node SHARED src/progress_recovery_node.cpp)
ament_target_dependencies(delivery_progress_recovery_bt_node rclcpp behaviortree_cpp_v3 nav_msgs)
target_compile_definitions(delivery_progress_recovery_bt_node PRIVATE BT_PLUGIN_EXPORT)

install(TARGETS delivery_progress_recovery_bt_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_libraries(delivery_progress_recovery_bt_node)
ament_package()
