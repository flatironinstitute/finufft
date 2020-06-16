# This defines the python module installation.

import os
import ctypes

from setuptools import setup, Extension

# Parse the requirements
with open(os.path.join('cufinufftpy', 'requirements.txt'), 'r') as fh:
    requirements = [item.strip() for item in fh.readlines()]

# Sanity check that we can find the CUDA cufinufft libraries before we get too far.
for lib in ['libcufinufftc.so', 'libcufinufftcf.so']:
    # One day can use find_library instead, but today many (Python) versions of this have bugs.
    try:
        lib = ctypes.cdll.LoadLibrary(lib)
    except Exception as e:
        print(lib, 'CUDA shared libraries not found in library path.'
              '  Please refer to installation documentation at http://github.com/flatironinstitute/cufinufft'
              ' and ensure CUDA installation is successful first before attempting to install the python wrappers.')
        raise(e)
else:
    print('cufinufft CUDA shared libraries found, continuing...')


# Python Package Setup
setup(
    name='cufinufftpy',
    version='1.0',
    author='python interfaces by: Melody Shih, Joakim Anden, Garrett Wright',
    author_email='yoyoshih13@gmail.com',
    url='http://github.com/flatironinstitute/cufinufft',
    description='python interface to cufinufft',
    long_description='python interface to cufinufft (CUDA Flatiron Institute Nonuniform Fast Fourier Transform) library.',
    license="Apache 2",
    packages=['cufinufftpy'],
    install_requires=requirements,
    # If you'd like to build or alter the docs you may additionally require these.
    extras_require={
        'docs': ['sphinx', 'sphinx_rtd_theme']
    },
    zip_safe=False,
)
