# This defines the Python module installation.

# Barnett 3/1/18. Updates by Yu-Hsuan Shih, June 2018.
# win32 mingw patch by Vineet Bansal, Feb 2019.
# attempt ../make.inc reading (failed) and default finufftdir. 2/25/20
# Barnett trying to get sphinx.ext.autodoc to work w/ this, 10/5/20

__version__ = '2.0.3.post1'

from setuptools import setup, Extension
import os
import platform

from tempfile import mkstemp

finufft_dir = os.environ.get('FINUFFT_DIR')

# Note: This will not work if run through pip install since setup.py is copied
# to a different location.
if finufft_dir == None or finufft_dir == '':
    current_path = os.path.abspath(__file__)
    finufft_dir = os.path.dirname(os.path.dirname(current_path))

# Set include and library paths relative to FINUFFT root directory.
inc_dir = os.path.join(finufft_dir, 'include')
lib_dir = os.path.join(finufft_dir, 'lib')

# Read in long description from README.md.
with open(os.path.join(finufft_dir, 'python', 'README.md'), 'r') as f:
        long_description = f.read()

# We specifically link to the dynamic library here through its absolute path
# (that is not through -lfinufft) to ensure that the absolute path of the
# library is encoded in the DT_NEEDED tag. This way, we won't need to have
# libfinufft.so in the LD_LIBRARY_PATH at runtime. The risk with this is that
# if the libfinufft.so is deleted or moved, the Python module will break
# unless LD_LIBRARY_PATH is updated.
if platform.system() == 'Windows':
    finufft_dlib = 'finufft'
else:
    finufft_dlib = os.path.join(lib_dir, 'finufft')

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
#include <finufft.h>

void PyInit_finufftc(void) {
    nufft_opts opt;

    finufft_default_opts(&opt);
}
""")


########## SETUP ###########
setup(
    name='finufft',
    version=__version__,
    author='Python interfaces by: Jeremy Magland, Daniel Foreman-Mackey, Joakim Anden, Libin Lu, and Alex Barnett',
    author_email='abarnett@flatironinstitute.org',
    url='https://github.com/flatironinstitute/finufft',
    description='Python interface to FINUFFT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="Apache 2",
    packages=['finufft'],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],
    install_requires=['numpy>=1.12.0'],
    python_requires='>=3.6',
    zip_safe=False,
    py_modules=['finufft.finufftc'],
    ext_modules=[
        Extension(name='finufft.finufftc',
                  sources=[source_filename],
                  include_dirs=[inc_dir],
                  library_dirs=[lib_dir],
                  libraries=[finufft_dlib])
        ]
)

os.unlink(source_filename)
