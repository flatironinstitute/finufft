# This defines the python module installation.

# Barnett 3/1/18. Updates by Yu-Hsuan Shih, June 2018.
# win32 mingw patch by Vineet Bansal, Feb 2019.
# attempt ../make.inc reading (failed) and default finufftdir. 2/25/20
# Barnett trying to get sphinx.ext.autodoc to work w/ this, 10/5/20

# Max OSX users: please edit as per below comments, and docs/install.rst

__version__ = '2.0.1'

import os
import ctypes

from setuptools import setup, Extension

# Sanity check that we can find the finufft library before we get too far.
try:
    lib = ctypes.cdll.LoadLibrary('libfinufft.so')
except Exception as e:
    print('FINUFFT shared libraries not found in library path.')
    raise(e)
print('FINUFFT shared libraries found, continuing...')


finufft_dir = os.environ.get('FINUFFT_DIR')

lib_dir = os.path.join(finufft_dir, 'lib')

########## SETUP ###########
setup(
    name='finufft',
    version=__version__,
    author='Python interfaces by: Jeremy Magland, Daniel Foreman-Mackey, Joakim Anden, Libin Lu, and Alex Barnett',
    author_email='abarnett@flatironinstitute.org',
    url='https://github.com/flatironinstitute/finufft',
    description='Python interface to FINUFFT',
    long_description='Python interface to FINUFFT (Flatiron Institute Nonuniform Fast Fourier Transform) library.',
    license="Apache 2",
    packages=['finufft'],
    install_requires=['numpy'],
    zip_safe=False,
    py_modules=['finufft/finufftc'],
    ext_modules=[
        Extension(name='finufft/finufftc',
                  sources=[],
                  libraries=['finufft'],
                  library_dirs=[lib_dir])
        ]
)

