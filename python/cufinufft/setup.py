# This defines the Python module installation.

import os
import ctypes
from pathlib import Path
import shutil

from setuptools import setup

# Description
DESCRIPTION = "Non-uniform fast Fourier transforms on the GPU"

with open('README.md', encoding='utf8') as fh:
    LONG_DESCRIPTION = fh.read()

# Parse the requirements
with open('requirements.txt', 'r') as fh:
    requirements = [item.strip() for item in fh.readlines()]

cufinufft_dir = os.environ.get('CUFINUFFT_DIR')
if not cufinufft_dir:
    cufinufft_dir = Path(__file__).resolve().parents[2]

shared_obj = os.path.join(cufinufft_dir, 'build', 'libcufinufft.so')

# Sanity check that we can find and load the CUDA cufinufft libraries before we get too far.
try:
    lib = ctypes.cdll.LoadLibrary(shared_obj)
except Exception as e:
    print('CUDA shared libraries not found in library path.'
           '  Please refer to installation documentation at '
           'https://finufft.readthedocs.io/en/latest/install_gpu.html '
           ' and ensure CUDA installation is successful first before '
           'attempting to install the Python wrappers.')
    raise(e)
print('cufinufft CUDA shared libraries found, continuing...')

shutil.copy(shared_obj, os.path.join(os.getcwd(), 'cufinufft'))

# Python Package Setup
setup(
    name='cufinufft',
    version='2.2.0.dev0',
    author='Yu-shuan Melody Shih, Garrett Wright, Joakim Anden, Johannes Blaschke, Alex Barnett',
    author_email='janden-vscholar@flatironinstitute.org',
    url='https://github.com/flatironinstitute/finufft',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=['cufinufft'],
    package_dir={'': '.'},
    package_data={'cufinufft': ['libcufinufft.so']},
    install_requires=requirements,
    # If you'd like to build or alter the docs you may additionally require these.
    extras_require={
        'docs': ['sphinx', 'sphinx_rtd_theme']
    },
    classifiers=['Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Environment :: GPU',
        'Topic :: Scientific/Engineering :: Mathematics'],
    python_requires='>=3.6',
    zip_safe=False,
)
