import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
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
    # semantic_overlay_node (/semantic/overlay_rgb, camera + segmentazione per
    # RViz) spento di default: ~35% di CPU e nessun nodo lo usa. overlay:=true
    # per riaccenderlo quando serve vederlo.
    overlay = LaunchConfiguration('overlay')

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
    overlay_node = Node(
        package='semantic', executable='semantic_overlay_node',
        name='semantic_overlay_node', output='screen',
        parameters=[use_sim_time], remappings=clock_remap,
        condition=IfCondition(overlay),
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
        DeclareLaunchArgument(
            'overlay', default_value='false',
            description='Pubblica /semantic/overlay_rgb (camera + segmentazione, per RViz)'),
        # 1 thread BLAS: le operazioni sono piccole (es. 90k punti x matrice
        # 3x3) e i thread extra di OpenBLAS restano in attesa ATTIVA fra una
        # chiamata e l'altra. Misurato sulla trasformazione della nuvola a
        # 17 Hz: 4 thread = 310% CPU (5.8 ms/frame), 1 thread = 16% (9.5 ms).
        # Senza limite (48 thread) semantic_costmap_node usava ~29 core.
        SetEnvironmentVariable('OMP_NUM_THREADS', '1'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
        gt_relay,
        costmap,
        elevation,
        overlay_node,
        combiner,
    ])
