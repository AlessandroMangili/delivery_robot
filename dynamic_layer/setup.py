from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dynamic_layer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alessandro',
    maintainer_email='alessandro.mangili1@studenti.unimi.it',
    description='People detection from camera and LiDAR, and tracking',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pointcloud_detector = dynamic_layer.pointcloud_detector:main',
            'tracker_3d = dynamic_layer.tracker_3d:main',
            'camera_detector = dynamic_layer.camera_detector:main',
        ],
    },
)
