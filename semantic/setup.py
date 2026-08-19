from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alessandro',
    maintainer_email='alessandro.mangili1@studenti.unimi.it',
    description='Map semantic to keep the robot upon the sidewalk',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'semantic_overlay_node = semantic.semantic_overlay_node:main',
            'semantic_costmap_node = semantic.semantic_costmap_node:main',
            'gt_segmentation_node = semantic.gt_segmentation_relay:main',
            'confinement_layer = semantic.confinement_layer:main',
            'ssrl_combiner = semantic.ssrl_combiner:main',
            'elevation_costmap_node = semantic.elevation_costmap_node:main',
        ],
    },
)
