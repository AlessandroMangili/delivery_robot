import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('dynamic_layer'), 'config', 'dynamic_tracker.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    single_thread = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value='detection.pt',
            description='YOLO detection weights: a file name in dynamic_layer/models, or an absolute path'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true when replaying a bag with --clock'),

        Node(
            package='dynamic_layer', executable='pointcloud_detector', name='pointcloud_detector',
            output='screen', parameters=[params, {'use_sim_time': use_sim_time}],
            additional_env=single_thread),
        Node(
            package='dynamic_layer', executable='camera_detector', name='camera_detector',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time, 'model_path': LaunchConfiguration('model')}],
            additional_env=single_thread),
        Node(
            package='dynamic_layer', executable='tracker_3d', name='tracker_3d',
            output='screen', parameters=[params, {'use_sim_time': use_sim_time}],
            additional_env=single_thread),
    ])
