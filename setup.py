import os
from glob import glob
from setuptools import setup
import setuptools.command.install
import setuptools.command.develop

package_name = 'vision_grasp'


def _create_lib_symlinks(install_base):
    """Create symlinks in lib/<package_name>/ pointing to bin/ executables."""
    bin_dir = os.path.join(install_base, 'bin')
    lib_pkg_dir = os.path.join(install_base, 'lib', package_name)
    if os.path.isdir(bin_dir):
        os.makedirs(lib_pkg_dir, exist_ok=True)
        for fname in os.listdir(bin_dir):
            src = os.path.join(bin_dir, fname)
            dst = os.path.join(lib_pkg_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst)


class CustomInstall(setuptools.command.install.install):
    """Custom install command that creates lib/<package_name>/ symlinks for ros2 run."""

    def run(self):
        super().run()
        _create_lib_symlinks(self.install_base)


class CustomDevelop(setuptools.command.develop.develop):
    """Custom develop command that creates lib/<package_name>/ symlinks for ros2 run."""

    def run(self):
        super().run()
        # In develop mode, install_dir is the site-packages directory.
        # Derive the package install prefix from COLCON_PREFIX_PATH.
        install_base = None
        colcon_prefix = os.environ.get('COLCON_PREFIX_PATH', '')
        if colcon_prefix:
            # COLCON_PREFIX_PATH may contain multiple paths; use the first one.
            prefix = colcon_prefix.split(':')[0]
            install_base = os.path.join(prefix, package_name)
        if install_base and os.path.isdir(install_base):
            _create_lib_symlinks(install_base)


setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Alkaid',
    maintainer_email='alkaid@example.com',
    description='Vision-based grasping for AGX Arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_picker = vision_grasp.vlm_picker_node:main',
            'grasp_executor = vision_grasp.grasp_executor:main',
            'test_move_to = vision_grasp.test_move_to:main',
        ],
    },
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
)
