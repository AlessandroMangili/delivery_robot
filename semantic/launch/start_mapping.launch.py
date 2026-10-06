import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('semantic'), 'config', 'semantic.yaml')
    use_sim_time = {'use_sim_time': False}

    seg_infer = Node(
        package='semantic', executable='seg_infer_node',
        name='seg_infer_node', output='screen',
        parameters=[params, use_sim_time],
    )

    costmap = Node(
        package='semantic', executable='semantic_costmap_node',
        name='semantic_costmap_node', output='screen',
        parameters=[params, use_sim_time],
    )

    elevation = Node(
        package='semantic', executable='elevation_costmap_node',
        name='elevation_costmap_node', output='screen',
        parameters=[params, use_sim_time],
    )
    overlay = Node(
        package='semantic', executable='semantic_overlay_node',
        name='semantic_overlay_node', output='screen',
        parameters=[params, use_sim_time],
    )

    combiner = Node(
        package='semantic', executable='ssrl_combiner',
        name='ssrl_combiner', output='screen',
        parameters=[params, use_sim_time],
    )

    return LaunchDescription([
        seg_infer,
        costmap,
        elevation,
        overlay,
        combiner,
    ])