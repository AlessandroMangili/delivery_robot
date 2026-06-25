import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time')

    nav2_localization_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch', 'localization_launch.py')

    localization_params_path = os.path.join(
        get_package_share_directory('localization'), 'config', 'amcl_localization.yaml')

    map_file_path = os.path.join(
        get_package_share_directory('localization'), 'maps', 'my_map.yaml')

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_localization_launch_path),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': localization_params_path,
            'map': map_file_path,
        }.items()
    )

    ld = LaunchDescription()
    ld.add_action(sim_time_arg)
    ld.add_action(localization_launch)
    return ld