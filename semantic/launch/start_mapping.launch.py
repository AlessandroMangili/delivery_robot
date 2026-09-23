import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('semantic'), 'config', 'semantic.yaml')
    use_sim_time = {'use_sim_time': True}
    # /clock_throttled (100 Hz, da clock_throttle nel launch della simulazione)
    # invece di /clock (~870 msg/s): ogni messaggio sveglia l'executor Python.
    # Con un bag riprodotto (ros2 bag play --clock) passare clock_topic:=/clock.
    clock_remap = [('/clock', LaunchConfiguration('clock_topic'))]

    gt_relay = Node(
        package='semantic', executable='gt_segmentation_node',
        name='gt_segmentation_node', output='screen',
        parameters=[params, use_sim_time], remappings=clock_remap,
    )
    costmap = Node(
        package='semantic', executable='semantic_costmap_node',
        name='semantic_costmap_node', output='screen',
        parameters=[params, use_sim_time], remappings=clock_remap,
    )

    elevation = Node(
        package='semantic', executable='elevation_costmap_node',
        name='elevation_costmap_node', output='screen',
        parameters=[params, use_sim_time], remappings=clock_remap,
    )
    overlay = Node(
        package='semantic', executable='semantic_overlay_node',
        name='semantic_overlay_node', output='screen',
        parameters=[use_sim_time], remappings=clock_remap,
    )

    combiner = Node(
        package='semantic', executable='ssrl_combiner',
        name='ssrl_combiner', output='screen',
        parameters=[params, use_sim_time], remappings=clock_remap,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'clock_topic', default_value='/clock_throttled',
            description='Topic del tempo simulato per i nodi Python'),
        # Senza limite OpenBLAS/OpenMP creano un thread per core (48) anche per
        # operazioni numpy piccole: semantic_costmap_node usava ~29 core in
        # attesa attiva e rallentava la simulazione (real-time factor 0.65).
        # Con 4: ~4 core, real-time factor 0.86.
        SetEnvironmentVariable('OMP_NUM_THREADS', '4'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '4'),
        gt_relay,
        costmap,
        elevation,
        overlay,
        combiner,
    ])
