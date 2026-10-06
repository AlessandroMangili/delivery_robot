cmake_minimum_required(VERSION 3.8)
project(husky_a100_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # I test espandono l'xacro dalla cartella sorgente: non serve il pacchetto installato
  ament_add_pytest_test(test_a100_geometry test/test_a100_geometry.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test
    TIMEOUT 120
  )
endif()

ament_package()
