import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_semantic = get_package_share_directory('semantic')
    params = os.path.join(pkg_semantic, 'config', 'semantic.yaml')

    replay = LaunchConfiguration('replay')
    sim_time = {'use_sim_time': replay}
    map_params = {'target_frame': LaunchConfiguration('frame'),
                  'map_name': LaunchConfiguration('map_name'),
                  'update': LaunchConfiguration('update_maps')}
    single_thread_numpy = {'OPENBLAS_NUM_THREADS': '1'}
    cores = ['taskset -c ', LaunchConfiguration('cpu_cores')]

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_name', default_value='',
            description='Maps to load at start and save on Ctrl-C: semantic_<name>.npz and '
                        'elevation_<name>.npz in semantic/maps. Empty starts from blank maps and saves nothing'),
        DeclareLaunchArgument(
            'update_maps', default_value='true',
            description='false freezes the maps: only the loaded maps are published, without segmentation or updates'),
        DeclareLaunchArgument(
            'model', default_value='semantic.pt',
            description='Segmentation weights: a file name in semantic/models, or an absolute path'),
        DeclareLaunchArgument(
            'step_lethal_radius', default_value='0.0',
            description='Distance from a height jump within which a step is lethal [m]; '
                        '0 extends it up to 0.6 m past the curb'),
        DeclareLaunchArgument(
            'semantic_ray_clearing', default_value='true',
            description='Removes lethal semantic cells that the LiDAR sees through close to the ground'),
        DeclareLaunchArgument(
            'frame', default_value='map',
            description='Frame of the maps: map with the localizer, odom to map without it'),
        DeclareLaunchArgument(
            'cpu_cores', default_value='4,5',
            description='CPU cores of the map nodes, as accepted by taskset'),
        DeclareLaunchArgument(
            'replay', default_value='false',
            description='true when replaying a bag with --clock: simulated time, decompressed images, overlay'),

        Node(
            package='image_transport', executable='republish', name='c920_decompress',
            prefix=cores, arguments=['compressed', 'raw'],
            remappings=[('in/compressed', '/c920/image_raw/compressed'), ('out', '/c920/image_raw')],
            parameters=[sim_time],
            condition=IfCondition(replay)),
        Node(
            package='semantic', executable='seg_infer_node', name='seg_infer_node',
            prefix=cores, output='screen',
            parameters=[params, sim_time, {'model_path': LaunchConfiguration('model')}],
            condition=IfCondition(LaunchConfiguration('update_maps'))),
        Node(
            package='semantic', executable='semantic_costmap_node', name='semantic_costmap_node',
            prefix=cores, output='screen',
            parameters=[params, sim_time, map_params,
                        {'ray_clear_enable': LaunchConfiguration('semantic_ray_clearing')}],
            additional_env=single_thread_numpy),
        Node(
            package='semantic', executable='elevation_costmap_node', name='elevation_costmap_node',
            prefix=cores, output='screen',
            parameters=[params, sim_time, map_params,
                        {'step_lethal_radius_m': LaunchConfiguration('step_lethal_radius')}],
            additional_env=single_thread_numpy),
        Node(
            package='semantic', executable='semantic_overlay_node', name='semantic_overlay_node',
            prefix=cores, output='screen',
            parameters=[params, sim_time],
            additional_env=single_thread_numpy,
            condition=IfCondition(replay)),
        Node(
            package='semantic', executable='ssrl_combiner', name='ssrl_combiner',
            prefix=cores, output='screen',
            parameters=[params, sim_time],
            additional_env=single_thread_numpy),
    ])
