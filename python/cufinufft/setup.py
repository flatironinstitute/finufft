# This defines the Python module installation.

import os
import ctypes
from pathlib import Path

from tempfile import mkstemp

from setuptools import setup, Extension

# Description
DESCRIPTION = "Non-uniform fast Fourier transforms on the GPU"

with open('README.md', encoding='utf8') as fh:
    LONG_DESCRIPTION = fh.read()

# Parse the requirements
with open('requirements.txt', 'r') as fh:
    requirements = [item.strip() for item in fh.readlines()]

cufinufft_dir = os.environ.get('CUFINUFFT_DIR')

if cufinufft_dir == None or cufinufft_dir == '':
    cufinufft_dir = Path(__file__).resolve().parents[2]

include_dir = os.path.join(cufinufft_dir, "include")
library_dir = os.path.join(cufinufft_dir, "build")

# Sanity check that we can find the CUDA cufinufft libraries before we get too far.
try:
    lib = ctypes.cdll.LoadLibrary(os.path.join(library_dir, 'libcufinufft.so'))
except Exception as e:
    print('CUDA shared libraries not found in library path.'
           '  Please refer to installation documentation at '
           'https://finufft.readthedocs.io/en/latest/install_gpu.html '
           ' and ensure CUDA installation is successful first before '
           'attempting to install the Python wrappers.')
    raise(e)
print('cufinufft CUDA shared libraries found, continuing...')

# For certain platforms (e.g. Ubuntu 20.04), we need to create a dummy source
# that calls one of the functions in the FINUFFT dynamic library. The reason
# is that these platforms override the default --no-as-needed flag for ld,
# which means that the linker will only link to those dynamic libraries for
# which there are unresolved symbols in the object files. Since we do not have
# a real source, the result is that no dynamic libraries are linked. To
# prevent this, we create a dummy source so that the library will link as
# expected.
fd, source_filename = mkstemp(suffix='.c', text=True)

with open(fd, 'w') as f:
    f.write( \
"""
#include <cufinufft.h>

void PyInit_cufinufftc(void) {
    cufinufft_opts opt;

    cufinufft_default_opts(&opt);
}
""")


# Python Package Setup
setup(
    name='cufinufft',
    version='2.2.0.dev0',
    author='Yu-shuan Melody Shih, Garrett Wright, Joakim Anden, Johannes Blaschke, Alex Barnett',
    author_email='janden-vscholar@flatironinstitute.org',
    url='https://github.com/flatironinstitute/cufinufft',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=['cufinufft'],
    package_dir={'': '.'},
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
    # This explicitly tells the wheel systems that we're platform specific.
    #   Addiitonally, will create a new cPython library with a decorated name
    #   that is rpath linked to CUDA library, also decorated (by auditwheel).
    #   Most importantly, pip will manage to install all this stuff in
    #   in places Python can find it (with a little help).
    py_modules=['cufinufftc'],
    ext_modules=[
        Extension(name='cufinufftc',
                  sources=[source_filename],
                  libraries=['cufinufft'],
                  include_dirs=[include_dir],
                  library_dirs=[library_dir],
                  runtime_library_dirs=[library_dir],
                  )
        ]
)

os.unlink(source_filename)
