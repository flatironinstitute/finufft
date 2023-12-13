# This defines the Python module installation.

# Barnett 3/1/18. Updates by Yu-Hsuan Shih, June 2018.
# win32 mingw patch by Vineet Bansal, Feb 2019.
# attempt ../make.inc reading (failed) and default finufftdir. 2/25/20
# Barnett trying to get sphinx.ext.autodoc to work w/ this, 10/5/20

__version__ = '2.2.0.dev0'

from setuptools import setup
import os
import platform
from pathlib import Path
import shutil
import glob
import itertools
from wheel.bdist_wheel import bdist_wheel
from distutils.util import get_platform

finufft_dir = os.environ.get('FINUFFT_DIR')

# Note: This will not work if run through pip install since setup.py is copied
# to a different location.
if finufft_dir == None or finufft_dir == '':
    finufft_dir = Path(__file__).resolve().parents[2]

# Set include and library paths relative to FINUFFT root directory.

# Read in long description from README.md.
with open(os.path.join(finufft_dir, 'python', 'finufft', 'README.md'), 'r') as f:
    long_description = f.read()

def get_libname():
    extensions = ("dll", "lib", "so")
    lib_dirs = (os.path.join(finufft_dir, 'lib'), os.path.join(finufft_dir, 'build'))
    for directory, ext in itertools.product(lib_dirs, extensions):
        path = os.path.join(directory, 'libfinufft.' + ext)
        if os.path.isfile(path):
            return directory, ext

    raise FileNotFoundError("Unable to find suitable finufft library")


class finufft_bdist(bdist_wheel):
    def finalize_options(self, *args, **kwargs):
        bdist_wheel.finalize_options(self, *args, **kwargs)

    def get_tag(self, *args, **kwargs):
        return "py3", "none", get_platform().replace('-', '_').replace('.', '_')


lib_dir, ext = get_libname()
ext_glob = "lib[!c]*" + ext

# setuptools will only copy files from inside package, so put them there
# use glob to copy supporting libraries in windows
libfiles = glob.glob(os.path.join(lib_dir, ext_glob))
for src in libfiles:
    shutil.copy2(src, 'finufft')


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
    package_dir={'': '.'},
    package_data={'finufft': [ext_glob]},
    cmdclass={'bdist_wheel': finufft_bdist},
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
)
