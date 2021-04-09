from setuptools import setup


package_name = 'quad_gaits'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Andrew Euredjian, Ankur Gupta, Revant Mahajan',
    maintainer_email='ageuredjian@wpi.edu, agupta4@wpi.edu, rmahajan@wpi.edu',
    description='quad walking controller',
    license='MIT',
    entry_points={
        'console_scripts': [
            'trot = quad_gaits.trot:trot'
            'calibrate = quad_gaits.trot:calibrate'
        ],
    },
)