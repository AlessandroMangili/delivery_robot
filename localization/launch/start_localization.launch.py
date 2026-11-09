import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (AndSubstitution, EqualsSubstitution, LaunchConfiguration,
                                  NotSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_localization = get_package_share_directory('localization')
    ekf_params = os.path.join(pkg_localization, 'config', 'localization.yaml')
    fastlio_params = os.path.join(pkg_localization, 'config', 'fast_lio_vlp16.yaml')
    localizer_params = os.path.join(pkg_localization, 'config', 'lidar_localization.yaml')

    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    use_map_localization = LaunchConfiguration('use_map_localization')
    start_mode = LaunchConfiguration('start_mode')
    global_start = AndSubstitution(use_map_localization, EqualsSubstitution(start_mode, 'global'))
    initial_z = ParameterValue(LaunchConfiguration('initial_z'), value_type=float)
    initial_qz = PythonExpression(['sin(radians(', LaunchConfiguration('initial_yaw_deg'), ') / 2)'])
    initial_qw = PythonExpression(['cos(radians(', LaunchConfiguration('initial_yaw_deg'), ') / 2)'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_map_localization', default_value='false',
            description='true starts the map localizer, which publishes map -> odom'),
        DeclareLaunchArgument(
            'start_mode', default_value='fixed',
            description='fixed uses the pose given by initial_*; global reuses the last saved pose '
                        'if it still matches the scan, otherwise it searches the whole map'),
        DeclareLaunchArgument('initial_x', default_value='0.0', description='start x on the map [m]'),
        DeclareLaunchArgument('initial_y', default_value='0.0', description='start y on the map [m]'),
        DeclareLaunchArgument('initial_z', default_value='-0.57',
                              description='height of base_footprint in the map frame [m]'),
        DeclareLaunchArgument('initial_yaw_deg', default_value='0.0',
                              description='start heading on the map [deg]'),
        DeclareLaunchArgument(
            'map_path', default_value=os.path.join(pkg_localization, 'maps', 'map.pcd'),
            description='prior point-cloud map'),
        DeclareLaunchArgument(
            'occupancy_yaml', default_value=os.path.join(pkg_localization, 'maps', 'grid.yaml'),
            description='2D grid of the map used by the global search'),
        DeclareLaunchArgument(
            'pose_state_path',
            default_value=os.path.join(os.path.expanduser('~'), '.ros', 'localizer_pose.json'),
            description='file where the last pose on the map is saved'),
        DeclareLaunchArgument(
            'use_fastlio', default_value='true',
            description='false does not start FAST-LIO, for bags that already contain /Odometry'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true when replaying a bag with --clock'),

        Node(
            package='fast_lio', executable='fastlio_mapping', name='laser_mapping', output='screen',
            parameters=[fastlio_params, sim_time],
            condition=IfCondition(LaunchConfiguration('use_fastlio'))),
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='odom_to_camera_init',
            arguments=['--x', '-0.12', '--y', '0', '--z', '0.35', '--frame-id', 'odom',
                       '--child-frame-id', 'camera_init'],
            parameters=[sim_time]),
        Node(
            package='robot_localization', executable='ekf_node', name='ekf_local', output='screen',
            parameters=[ekf_params, sim_time],
            remappings=[('/odometry/filtered', '/odometry/local')]),
        Node(
            package='lidar_localization_ros2', executable='lidar_localization_node',
            name='lidar_localization', output='screen',
            parameters=[localizer_params, sim_time,
                        {'map_path': LaunchConfiguration('map_path'),
                         'set_initial_pose': NotSubstitution(EqualsSubstitution(start_mode, 'global')),
                         'initial_pose_x': ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
                         'initial_pose_y': ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
                         'initial_pose_z': initial_z,
                         'initial_pose_qz': ParameterValue(initial_qz, value_type=float),
                         'initial_pose_qw': ParameterValue(initial_qw, value_type=float)}],
            remappings=[('/cloud', '/velodyne_points')],
            condition=IfCondition(use_map_localization)),
        Node(
            package='lidar_localization_ros2', executable='start_lifecycle_node.py',
            arguments=['lidar_localization'], output='screen',
            condition=IfCondition(use_map_localization)),
        Node(
            package='lidar_localization_ros2', executable='global_localization_node.py',
            name='global_localization_node', output='screen',
            parameters=[sim_time,
                        {'occupancy_yaml': LaunchConfiguration('occupancy_yaml'),
                         'map_path': LaunchConfiguration('map_path'),
                         'cloud_topic': '/velodyne_points',
                         'global_frame_id': 'map',
                         'base_frame_id': 'base_footprint',
                         'odom_frame_id': 'odom',
                         'use_cpp_backend': True,
                         'enable_registration_scoring': True,
                         'registration_score_gate': 6.0,
                         'registration_refine_candidates': False,
                         'registration_seed_z_m': initial_z,
                         'seed_z_m': initial_z,
                         'max_scan_points': 256,
                         'angular_resolution_deg': 5.0,
                         'max_candidates': 8,
                         'nms_radius_m': 3.0,
                         'startup_status_topic': '/startup_initialization/status'}],
            condition=IfCondition(global_start)),
        Node(
            package='lidar_localization_ros2', executable='startup_initialization_node.py',
            name='startup_initialization', output='screen',
            parameters=[sim_time,
                        {'map_path': LaunchConfiguration('map_path'),
                         'pose_state_path': LaunchConfiguration('pose_state_path'),
                         'restore_saved_pose': True,
                         'initial_pose_preconfigured': False,
                         'enable_global_initialization': True,
                         'require_global_registration_scoring': True,
                         'cloud_topic': '/velodyne_points',
                         'pose_topic': '/pcl_pose',
                         'global_frame_id': 'map',
                         'odom_frame_id': 'odom',
                         'base_frame_id': 'base_footprint',
                         'default_z_m': initial_z}],
            condition=IfCondition(global_start)),
    ])
