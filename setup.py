from setuptools import setup, find_packages

setup(
    name='quantum_electron',
    version='0.1.0',
    packages=find_packages(include=['quantum_electron']), 
    install_requires=['shapely', 'scikit-image', 'pyvista']
)