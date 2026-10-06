#!/usr/bin/env python3
"""Tutti i sensori in un comando, con frame_id e calibrazioni fissati.

  ros2 launch hunter sensors.launch.py
  ros2 launch hunter sensors.launch.py cam1_dev:=/dev/video1 cam2_dev:=/dev/video3 ntrip:=false

I /dev/videoN cambiano tra un avvio e l'altro: controlla con
  v4l2-ctl --list-devices
(meglio ancora i percorsi stabili in /dev/v4l/by-path/). Il profilo DDS mette i
messaggi grandi (nuvole, camera1) in memoria condivisa.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HOME = os.path.expanduser('~')
DDS_PROFILE = os.path.join(HOME, 'Desktop', 'thesis_ws', 'dds', 'fastdds_big_msgs.xml')
CAMINFO = os.path.join(HOME, '.ros', 'camera_info')


def camera(ns, dev, frame, info_file):
    return Node(
        package='v4l2_camera', executable='v4l2_camera_node', namespace=ns,
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration(dev),
            'camera_frame_id': frame,                         # = frame ottico dell'URDF
            'camera_info_url': 'file://' + os.path.join(CAMINFO, info_file),
        }],
    )


def generate_launch_description():
    xsens = os.path.join(get_package_share_directory('xsens_mti_ros2_driver'),
                         'launch', 'xsens_mti_node.launch.py')
    ntrip = os.path.join(get_package_share_directory('ntrip'), 'launch', 'ntrip_launch.py')
    
    return LaunchDescription([
        DeclareLaunchArgument('cam1_dev', default_value='/dev/video0'),
        DeclareLaunchArgument('cam2_dev', default_value='/dev/video2'),
        DeclareLaunchArgument('ntrip', default_value='true', description='correzioni RTK (serve internet)'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', DDS_PROFILE),

        camera('/camera1', 'cam1_dev', 'camera_optical', 'camera.yaml'),
        camera('/camera2', 'cam2_dev', 'camera2_optical', 'camera2.yaml'),
        Node(package='rslidar_sdk', executable='rslidar_sdk_node', output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(xsens)),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(ntrip),
                                 condition=IfCondition(LaunchConfiguration('ntrip'))),
    ])
