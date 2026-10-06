"""Visualizza l'Husky A100 in RViz2.

    ros2 launch husky_a100_description display.launch.py
    ros2 launch husky_a100_description display.launch.py gui:=false
    ros2 launch husky_a100_description display.launch.py dimensions_file:=/percorso/mie_misure.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('husky_a100_description')

    model = LaunchConfiguration('model')
    dimensions_file = LaunchConfiguration('dimensions_file')
    urdf_extras = LaunchConfiguration('urdf_extras')
    gui = LaunchConfiguration('gui')
    rviz_config = LaunchConfiguration('rviz_config')

    robot_description = ParameterValue(
        Command(['xacro ', model,
                 ' dimensions_file:=', dimensions_file,
                 ' urdf_extras:=', urdf_extras]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value=os.path.join(pkg, 'urdf', 'husky_a100.urdf.xacro'),
            description='File xacro del robot'),
        DeclareLaunchArgument(
            'dimensions_file', default_value=os.path.join(pkg, 'config', 'a100_dimensions.yaml'),
            description='File YAML con le dimensioni'),
        DeclareLaunchArgument(
            'urdf_extras', default_value='',
            description='File xacro aggiuntivo con i sensori (vuoto = nessuno)'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='true = cursori per muovere le ruote (joint_state_publisher_gui)'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(pkg, 'rviz', 'a100.rviz'),
            description='Configurazione di RViz2'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}], output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             condition=IfCondition(gui), output='screen'),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             condition=UnlessCondition(gui), output='screen'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config], output='screen'),
    ])
