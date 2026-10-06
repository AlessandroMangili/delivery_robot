#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('hunter')
    default_xacro = os.path.join(pkg_share, 'description', 'hunter_sensors.urdf.xacro')
    xacro_file = LaunchConfiguration('xacro')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # launch del driver AgileX (pacchetto hunter_base)
    hunter_base_launch = os.path.join(
        get_package_share_directory('hunter_base'), 'launch', 'hunter_base.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument('xacro', default_value=default_xacro),

        DeclareLaunchArgument('use_real_base', default_value='true'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            output='screen',
        ),

        # === BASE REALE: driver Hunter su can0. Pubblica /odom + TF odom->base_footprint ===
        # base_frame DEVE essere base_footprint (root dell'URDF), NON base_link,
        # altrimenti base_link avrebbe due padri (base_footprint dall'URDF + odom dal driver).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hunter_base_launch),
            launch_arguments={
                'port_name': 'can0',
                'odom_frame': 'odom',
                'base_frame': 'base_footprint',
                'odom_topic_name': 'odom',
            }.items(),
            condition=IfCondition(LaunchConfiguration('use_real_base')),
        ),

        # === BANCO: odom -> base_footprint identita'. Attivo SOLO con use_real_base:=false ===
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='tf_odom_base',
            arguments=['--frame-id', 'odom', '--child-frame-id', 'base_footprint',
                       '--x', '0', '--y', '0', '--z', '0',
                       '--roll', '0', '--pitch', '0', '--yaw', '0'],
            output='screen',
            condition=UnlessCondition(LaunchConfiguration('use_real_base')),
        ),
    ])