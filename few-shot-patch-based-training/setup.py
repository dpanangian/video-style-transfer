from setuptools import setup, find_packages
import pathlib, os


setup(
    name='few-shot-patch-based-training',
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=[
        'numpy==1.19.1',
        'opencv-python==4.4.0.40',
        'Pillow==7.2.0',
        'PyYAML==5.3.1',
        'scikit-image==0.17.2',
        'scipy==1.5.2',
        'tensorflow==1.15.3',
        'torch==1.6.0',
        'torchvision==0.7.0',
        ],
    
)