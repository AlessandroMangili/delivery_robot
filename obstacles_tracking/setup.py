from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'obstacles_tracking'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alessandro',
    maintainer_email='alessandro.mangili1@studenti.unimi.it',
    description=(
        'Detection (PointPillars, YOLO) e tracking dinamico EagerMOT: fusione, '
        'tracker a due stadi, ciclo di vita nascita/conferma/morte.'
    ),
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eagermot_node = obstacles_tracking.eagermot_node:main',
            'pointpillars_detector = obstacles_tracking.pointpillars_detector:main',
            'yolo_detect_node = obstacles_tracking.yolo_detect_node:main',
        ],
    },
)
