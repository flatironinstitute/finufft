# This defines the python module installation.
# Only for double-prec, multi-threaded for now.

# Barnett 3/1/18. Updates by Yu-Hsuan Shih, June 2018.
# win32 mingw patch by Vineet Bansal, Feb 2019.
# attempt ../make.inc reading (failed) and default finufftdir. 2/25/20

# Max OSX users: please edit as per below comments, and docs/install.rst

__version__ = '2.0.0'

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import ctypes

# libin to change to python-dotenv or whatever's simplest:
import dotenv   # is this part of standard python? (install_requires fails) ?

# since people might not set it, set to the parent of this script's dir...
finufftdir = os.environ.get('FINUFFT_DIR')
if finufftdir==None or finufftdir=='':
    finufftdir = os.path.dirname(os.path.dirname(__file__))

# need this on ubuntu with python3.8
os.system("export LDFLAGS=-wl,--no-as-needed")

# alex debugging - please remove when debugged:
print("finufftdir: ", finufftdir)
print("pathname of __file__: ",os.path.dirname(__file__))   # ok, fails why?

# default compiler choice (note g++ = clang in mac-osx):
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

# attempt override compiler choice using ../make.inc to match your C++ build
makeinc = finufftdir+"/make.inc"
dotenv.load_dotenv(makeinc, override=True)   # modifies os.environ

# debug and remove:
print(os.environ['CXX'])  # check - doesn't read correctly from ../make.inc  :(

# in the end avoided code from https://stackoverflow.com/questions/3503719/emulating-bash-source-in-python
#if os.path.isfile(makeinc):
#    command = 'env -i bash -c "source %s"' % (makeinc)
#    for line in subprocess.getoutput(command).split("\n"):
#        if line!='':
#            key, value = line.split("=")
#            print(key, value)
#            os.environ[key] = value

inc_dir = finufftdir+"/include"
src_dir = finufftdir+"/src"
lib_dir = finufftdir+"/lib"
finufft_dlib = finufftdir+"/lib/finufft"
finufft_lib = finufftdir+"/lib-static/finufft"

########## SETUP ###########
setup(
    name='finufft',
    version=__version__,
    author='python interfaces by: Jeremy Magland, Daniel Foreman-Mackey, Alex Barnett',
    author_email='abarnett@flatironinstitute.org',
    url='http://github.com/ahbarnett/finufft',
    description='python interface to FINUFFT',
    long_description='python interface to FINUFFT (Flatiron Institute Nonuniform Fast Fourier Transform) library.',
    license="Apache 2",
    packages=['finufft'],
    install_requires=['numpy','python-dotenv'],
    zip_safe=False,
    py_modules=['finufft/finufftc'],
    ext_modules=[
        Extension(name='finufft/finufftc',
                  sources=[],
                  libraries=[finufft_dlib])
        ]
)

