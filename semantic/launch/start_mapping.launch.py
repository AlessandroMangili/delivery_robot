import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_semantic = get_package_share_directory('semantic')

    replay_arg = DeclareLaunchArgument(
        'replay', default_value='false',
        description='true = rosbag con --clock: tempo simulato e immagini decompresse'
    )

    frame_arg = DeclareLaunchArgument(
        'frame', default_value='map',
        description='map = mappe sulla mappa del localizzatore; odom = giro di mappatura senza localizzatore'
    )

    map_name_arg = DeclareLaunchArgument(
        'map_name', default_value='',
        description='nome delle mappe da caricare all avvio e salvare alla chiusura (vuoto = niente)'
    )

    elevazione_360_arg = DeclareLaunchArgument(
        'elevazione_360', default_value='true',
        description='true = elevazione a 360 gradi; false = solo nel campo della camera'
    )

    modello_arg = DeclareLaunchArgument(
        'modello', default_value='yolo26n-sem_eliza.pt',
        description='pesi della segmentazione: nome in semantic/models oppure percorso assoluto'
    )

    brighten_arg = DeclareLaunchArgument(
        'brighten', default_value='false',
        description='true = schiarisce le immagini prima della segmentazione (sera)'
    )

    semantic_params_path = os.path.join(pkg_semantic, 'config', 'semantic.yaml')

    sim_time = {'use_sim_time': LaunchConfiguration('replay')}
    map_params = {'target_frame': LaunchConfiguration('frame'),
                  'map_name': LaunchConfiguration('map_name')}
    single_thread_numpy = {'OPENBLAS_NUM_THREADS': '1'}

    decompress_node = Node(
        package='image_transport',
        executable='republish',
        name='c920_decompress',
        arguments=['compressed', 'raw'],
        remappings=[('in/compressed', '/c920/image_raw/compressed'), ('out', '/c920/image_raw')],
        parameters=[sim_time],
        condition=IfCondition(LaunchConfiguration('replay'))
    )

    brighten_node = Node(
        package='semantic',
        executable='image_brighten_node',
        name='image_brighten_node',
        output='screen',
        parameters=[semantic_params_path, sim_time],
        condition=IfCondition(LaunchConfiguration('brighten'))
    )

    seg_infer_node = Node(
        package='semantic',
        executable='seg_infer_node',
        name='seg_infer_node',
        output='screen',
        parameters=[semantic_params_path, sim_time,
                    {'model_path': LaunchConfiguration('modello')}],
        condition=UnlessCondition(LaunchConfiguration('brighten'))
    )

    seg_infer_bright_node = Node(
        package='semantic',
        executable='seg_infer_node',
        name='seg_infer_node',
        output='screen',
        parameters=[semantic_params_path, sim_time,
                    {'model_path': LaunchConfiguration('modello'), 'image_topic': '/c920/image_bright'}],
        condition=IfCondition(LaunchConfiguration('brighten'))
    )

    semantic_costmap_node = Node(
        package='semantic',
        executable='semantic_costmap_node',
        name='semantic_costmap_node',
        output='screen',
        parameters=[semantic_params_path, sim_time, map_params],
        additional_env=single_thread_numpy
    )

    elevation_costmap_node = Node(
        package='semantic',
        executable='elevation_costmap_node',
        name='elevation_costmap_node',
        output='screen',
        parameters=[semantic_params_path, sim_time, map_params,
                    {'elevazione_360': LaunchConfiguration('elevazione_360')}],
        additional_env=single_thread_numpy
    )

    overlay_node = Node(
        package='semantic',
        executable='semantic_overlay_node',
        name='semantic_overlay_node',
        output='screen',
        parameters=[semantic_params_path, sim_time],
        additional_env=single_thread_numpy,
        condition=UnlessCondition(LaunchConfiguration('brighten'))
    )

    overlay_bright_node = Node(
        package='semantic',
        executable='semantic_overlay_node',
        name='semantic_overlay_node',
        output='screen',
        parameters=[semantic_params_path, sim_time, {'rgb_topic': '/c920/image_bright'}],
        additional_env=single_thread_numpy,
        condition=IfCondition(LaunchConfiguration('brighten'))
    )

    ssrl_combiner_node = Node(
        package='semantic',
        executable='ssrl_combiner',
        name='ssrl_combiner',
        output='screen',
        parameters=[semantic_params_path, sim_time],
        additional_env=single_thread_numpy
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(replay_arg)
    launchDescriptionObject.add_action(frame_arg)
    launchDescriptionObject.add_action(map_name_arg)
    launchDescriptionObject.add_action(elevazione_360_arg)
    launchDescriptionObject.add_action(modello_arg)
    launchDescriptionObject.add_action(brighten_arg)
    launchDescriptionObject.add_action(decompress_node)
    launchDescriptionObject.add_action(brighten_node)
    launchDescriptionObject.add_action(seg_infer_node)
    launchDescriptionObject.add_action(seg_infer_bright_node)
    launchDescriptionObject.add_action(semantic_costmap_node)
    launchDescriptionObject.add_action(elevation_costmap_node)
    launchDescriptionObject.add_action(overlay_node)
    launchDescriptionObject.add_action(overlay_bright_node)
    launchDescriptionObject.add_action(ssrl_combiner_node)

    return launchDescriptionObject
