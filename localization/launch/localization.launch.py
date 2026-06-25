import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def find_source_maps_dir(pkg='localization'):
    """Trova (o crea) la cartella maps/ nel SORGENTE del package 'localization',
    cercandola ovunque sotto <ws>/src (riconosciuta dal package.xml).
    Ripiega sulla share installata se non trova il sorgente."""
    share = get_package_share_directory(pkg)
    ws = share
    for _ in range(8):
        ws = os.path.dirname(ws)
        src = os.path.join(ws, 'src')
        if os.path.isdir(src):
            for root, dirs, files in os.walk(src):
                if os.path.basename(root) == pkg and 'package.xml' in files:
                    d = os.path.join(root, 'maps')
                    os.makedirs(d, exist_ok=True)
                    return d
            break
    d = os.path.join(share, 'maps')
    os.makedirs(d, exist_ok=True)
    return d


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='True')

    pkg_loc = get_package_share_directory('localization')
    loc_yaml = os.path.join(pkg_loc, 'config', 'slam_toolbox_localization.yaml')

    # percorso mappa risolto automaticamente nella cartella del package (senza estensione)
    maps_dir = find_source_maps_dir('localization')
    map_file = os.path.join(maps_dir, 'sidewalk_map')

    slam_node = Node(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            loc_yaml,
            {
                'use_sim_time': use_sim_time,
                'mode': 'localization',
                'map_file_name': map_file,          # carica .posegraph/.data all'avvio
                'map_start_pose': [0.0, 0.0, 0.0],  # o usa "2D Pose Estimate" in RViz
            },
        ],
    )

    ld = LaunchDescription()
    ld.add_action(sim_arg)
    ld.add_action(slam_node)
    return ld