import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, AndSubstitution, NotSubstitution
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

    core_mappe_arg = DeclareLaunchArgument(
        'core_mappe', default_value='4,5',
        description='core su cui girano i nodi delle mappe (taskset); 0-5 = tutti, come prima'
    )

    pulizia_raggi_arg = DeclareLaunchArgument(
        'pulizia_raggi', default_value='true',
        description='true = la semantica toglie le celle letali che i raggi del LiDAR attraversano vicino al suolo (ray_clear_mode in semantic.yaml)'
    )

    aggiorna_mappe_arg = DeclareLaunchArgument(
        'aggiorna_mappe', default_value='true',
        description='false = mappe ferme: pubblica solo le mappe caricate con map_name, senza segmentazione ne aggiornamento'
    )

    raggio_gradino_arg = DeclareLaunchArgument(
        'raggio_gradino_letale', default_value='0.0',
        description='0 = gradino letale anche oltre l ombra fino a 0,6 m (come prima); per esempio 0.2 = letale solo entro 20 cm dal salto'
    )

    semantic_params_path = os.path.join(pkg_semantic, 'config', 'semantic.yaml')

    sim_time = {'use_sim_time': LaunchConfiguration('replay')}
    map_params = {'target_frame': LaunchConfiguration('frame'),
                  'map_name': LaunchConfiguration('map_name')}
    single_thread_numpy = {'OPENBLAS_NUM_THREADS': '1'}
    core_mappe = ['taskset -c ', LaunchConfiguration('core_mappe')]

    decompress_node = Node(
        package='image_transport',
        executable='republish',
        name='c920_decompress',
        prefix=core_mappe,
        arguments=['compressed', 'raw'],
        remappings=[('in/compressed', '/c920/image_raw/compressed'), ('out', '/c920/image_raw')],
        parameters=[sim_time],
        condition=IfCondition(LaunchConfiguration('replay'))
    )

    brighten_node = Node(
        package='semantic',
        executable='image_brighten_node',
        name='image_brighten_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time],
        condition=IfCondition(AndSubstitution(LaunchConfiguration('aggiorna_mappe'), LaunchConfiguration('brighten')))
    )

    seg_infer_node = Node(
        package='semantic',
        executable='seg_infer_node',
        name='seg_infer_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time,
                    {'model_path': LaunchConfiguration('modello')}],
        condition=IfCondition(AndSubstitution(LaunchConfiguration('aggiorna_mappe'), NotSubstitution(LaunchConfiguration('brighten'))))
    )

    seg_infer_bright_node = Node(
        package='semantic',
        executable='seg_infer_node',
        name='seg_infer_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time,
                    {'model_path': LaunchConfiguration('modello'), 'image_topic': '/c920/image_bright'}],
        condition=IfCondition(AndSubstitution(LaunchConfiguration('aggiorna_mappe'), LaunchConfiguration('brighten')))
    )

    semantic_costmap_node = Node(
        package='semantic',
        executable='semantic_costmap_node',
        name='semantic_costmap_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time, map_params,
                    {'ray_clear_enable': LaunchConfiguration('pulizia_raggi'),
                     'aggiorna': LaunchConfiguration('aggiorna_mappe')}],
        additional_env=single_thread_numpy
    )

    elevation_costmap_node = Node(
        package='semantic',
        executable='elevation_costmap_node',
        name='elevation_costmap_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time, map_params,
                    {'elevazione_360': LaunchConfiguration('elevazione_360'),
                     'aggiorna': LaunchConfiguration('aggiorna_mappe'),
                     'step_lethal_radius_m': LaunchConfiguration('raggio_gradino_letale')}],
        additional_env=single_thread_numpy
    )

    overlay_node = Node(
        package='semantic',
        executable='semantic_overlay_node',
        name='semantic_overlay_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time],
        additional_env=single_thread_numpy,
        condition=IfCondition(AndSubstitution(LaunchConfiguration('replay'), NotSubstitution(LaunchConfiguration('brighten'))))
    )

    overlay_bright_node = Node(
        package='semantic',
        executable='semantic_overlay_node',
        name='semantic_overlay_node',
        prefix=core_mappe,
        output='screen',
        parameters=[semantic_params_path, sim_time, {'rgb_topic': '/c920/image_bright'}],
        additional_env=single_thread_numpy,
        condition=IfCondition(AndSubstitution(LaunchConfiguration('replay'), LaunchConfiguration('brighten')))
    )

    ssrl_combiner_node = Node(
        package='semantic',
        executable='ssrl_combiner',
        name='ssrl_combiner',
        prefix=core_mappe,
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
    launchDescriptionObject.add_action(core_mappe_arg)
    launchDescriptionObject.add_action(pulizia_raggi_arg)
    launchDescriptionObject.add_action(aggiorna_mappe_arg)
    launchDescriptionObject.add_action(raggio_gradino_arg)
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
