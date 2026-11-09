from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('localization')
    use_gps = LaunchConfiguration('use_gps')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use the simulation clock')
    declare_use_gps = DeclareLaunchArgument(
        'use_gps', default_value='True',
        description='Start navsat_transform_node (false = no GPS input)')

    ekf_params = PathJoinSubstitution([pkg, 'config', 'localization.yaml'])

    ekf_local_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_local',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[('/odometry/filtered', '/odometry/local')],
    )

    ekf_global_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_global',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[('/odometry/filtered', '/odometry/global')],
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        condition=IfCondition(use_gps),
        parameters=[ekf_params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[
            ('imu', '/imu'),
            ('gps/fix', '/navsat'),
            ('odometry/filtered', '/odometry/global'),
            ('odometry/gps', '/odometry/gps'),
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_gps,
        ekf_local_node,
        ekf_global_node,
        navsat_transform_node,
    ])
