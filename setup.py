# This defines the python module installation.

import os

from setuptools import setup, Extension

# Parse the requirements
with open(os.path.join('cufinufftpy', 'requirements.txt'), 'r') as fh:
    requirements = [item.strip() for item in fh.readlines()]

# Add sanity check that we can find the cuda libraries...


# Python Package Setup
setup(
    name='cufinufftpy',
    version='0.1',
    author='abc efg',
    author_email='asb@123.com',
    url='http://github.com/flatironinstitute/cufinufft',
    description='python interface to cufinufft',
    long_description='python interface to cufinufft (CUDA Flatiron Institute Nonuniform Fast Fourier Transform) library.',
    license="Apache 2",
    packages=['cufinufftpy'],
    install_requires=requirements,
    zip_safe=False,
)
