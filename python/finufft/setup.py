# This defines the Python module installation.

# Barnett 3/1/18. Updates by Yu-Hsuan Shih, June 2018.
# win32 mingw patch by Vineet Bansal, Feb 2019.
# attempt ../make.inc reading (failed) and default finufftdir. 2/25/20
# Barnett trying to get sphinx.ext.autodoc to work w/ this, 10/5/20

__version__ = '2.2.0b1'

from setuptools import setup, Extension
from distutils.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import os
import platform
from pathlib import Path

from tempfile import mkstemp

finufft_dir = os.environ.get('FINUFFT_DIR')

# Note: This will not work if run through pip install since setup.py is copied
# to a different location.
if finufft_dir == None or finufft_dir == '':
    finufft_dir = Path(__file__).resolve().parents[2]

# Set include and library paths relative to FINUFFT root directory.
inc_dir = os.path.join(finufft_dir, 'include')
lib_dir = os.path.join(finufft_dir, 'lib')
lib_dir_cmake = os.path.join(finufft_dir, 'build')   # lib may be only here

# Read in long description from README.md.
with open(os.path.join(finufft_dir, 'python', 'finufft', 'README.md'), 'r') as f:
        long_description = f.read()

finufft_dlib = 'finufft'

# Windows does not have the concept of rpath and as a result, MSVC crashes if
# supplied with one.
if platform.system() != "Windows":
    runtime_library_dirs = [lib_dir, lib_dir_cmake]
else:
    runtime_library_dirs = []

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
    finufft_opts opt;

    finufft_default_opts(&opt);
}
""")


########## CTYPES CROSS-PLATFORM HACK ##########


class CTypesExtension(Extension):
    pass


class build_ext(_build_ext):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            # importlib can not recognize .so on Windows
            if platform.system() != "Windows":
                return ext_name + ".so"
            else:
                return ext_name + ".pyd"
        return super().get_ext_filename(ext_name)


class bdist_wheel_abi_none(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        return "py3", "none", plat


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
        CTypesExtension(name='finufft.finufftc',
                  sources=[source_filename],
                  include_dirs=[inc_dir, '/usr/local/include'],
                  library_dirs=[lib_dir, lib_dir_cmake, '/usr/local/lib'],
                  libraries=[finufft_dlib],
                  runtime_library_dirs=runtime_library_dirs)
        ],
    cmdclass={"build_ext": build_ext, "bdist_wheel": bdist_wheel_abi_none}
)

os.unlink(source_filename)
