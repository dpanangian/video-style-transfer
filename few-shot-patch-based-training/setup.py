from setuptools import setup, find_packages
import pathlib, os


setup(
    name='few-shot-patch-based-training',
    version='0.0.1',
    description='Implements VQGAN+CLIP for image and video generation, and style transfers, based on text and image prompts. Emphasis on ease-of-use, documentation, and smooth video creation.',
    long_description_content_type='text/markdown',
    url='https://github.com/rkhamilton/vqgan-clip-generator',
    author='Ryan Hamilton',
    author_email='ryan.hamilton@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Environment :: GPU :: NVIDIA CUDA',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='VQGAN, VQGAN+CLIP, deep dream, neural network, pytorch',  # Optional
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