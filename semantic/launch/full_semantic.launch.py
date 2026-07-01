import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('semantic'), 'config', 'params.yaml')
    use_sim_time = {'use_sim_time': True}

    # --- nodi gia' presenti nel launch originale ---
    gt_relay = Node(
        package='semantic', executable='gt_segmentation_node',
        name='gt_segmentation_node', output='screen',
        parameters=[params, use_sim_time],
    )
    costmap = Node(
        package='semantic', executable='semantic_costmap_node',
        name='semantic_costmap_node', output='screen',
        parameters=[params, use_sim_time],
    )
    overlay = Node(
        package='semantic', executable='semantic_overlay_node',
        name='semantic_overlay_node', output='screen',
        parameters=[use_sim_time],
    )

    # --- nodi aggiunti: confinamento + combinatore SSRL ---
    confinement = Node(
        package='semantic', executable='confinement_layer',
        name='confinement_layer', output='screen',
        parameters=[params, use_sim_time],
    )
    combiner = Node(
        package='semantic', executable='ssrl_combiner',
        name='ssrl_combiner', output='screen',
        parameters=[params, use_sim_time],
    )

    return LaunchDescription([
        gt_relay,
        costmap,
        overlay,
        confinement,
        combiner,
    ])