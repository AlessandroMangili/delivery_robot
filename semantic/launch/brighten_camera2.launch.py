"""Schiarisce /camera2/image_raw -> /camera2/image_bright (camera live o `ros2 bag play`).

  ros2 launch semantic brighten_camera2.launch.py
  ros2 launch semantic brighten_camera2.launch.py gamma:=2.6 use_sim_time:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory('semantic'), 'config', 'semantic.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('gamma', default_value='2.2'),
        DeclareLaunchArgument('gain', default_value='1.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='semantic', executable='image_brighten_node',
            name='image_brighten_node', output='screen',
            parameters=[params, {
                'gamma': ParameterValue(LaunchConfiguration('gamma'), value_type=float),
                'gain': ParameterValue(LaunchConfiguration('gain'), value_type=float),
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            }],
        ),
    ])
