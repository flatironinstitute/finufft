Installation
============

Quick linux install instructions
--------------------------------

In brief, go to the github page https://github.com/flatironinstitute/finufft and
follow instructions to download the source (eg see the green button).
Make sure you have packages ``fftw3`` and ``fftw3-devel`` installed.
Then ``cd`` into your FINUFFT directory and ``make test``.
This should compile the static
library in ``lib-static/``, some C++ test drivers in ``test/``, then run them,
printing some terminal output ending in::

  0 crashes out of 5 tests done

If this fails see the more detailed instructions below.
If it succeeds, run ``make lib`` and proceed to link to the library.
Alternatively, try one of our `precompiled linux and OSX binaries <http://users.flatironinstitute.org/~ahb/codes/finufft-binaries>`_.
Type ``make`` to see a list of other aspects to build (language
interfaces, etc). Consider installing ``numdiff`` as below to allow
``make test`` to perform a better accuracy check.
Please read :ref:`Usage <usage>` and look in ``examples/`` and ``test/``
for other usage examples.

Dependencies
------------

This library is fully supported for unix/linux and almost fully on
Mac OSX.  We have also heard that it can be compiled under Windows
using MinGW; we also suggest trying within the Windows Subsystem for
Linux (WSL).

For the basic libraries you need

* C++ compiler, such as ``g++`` packaged with GCC, or ``clang`` with OSX
* FFTW3
* GNU make

Optional:

* ``numdiff`` (preferred but not essential; enables better pass-fail accuracy validation)
* for Fortran wrappers: compiler such as ``gfortran``
* for matlab/octave wrappers: MATLAB, or octave and its development libraries
* for the python wrappers you will need ``python`` and ``pip`` (if you are stuck on python v2), or ``python3`` and ``pip3`` (for the standard python v3). You will also need ``pybind11``
* for rebuilding new matlab/octave wrappers (experts only): ``mwrap``


Tips for installing dependencies on linux
-----------------------------------------

On a Fedora/CentOS linux system, dependencies can be installed as follows::

  sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp octave octave-devel

.. note::

   we are not exactly sure how to install python3 and pip3 using yum

Alternatively, on Ubuntu linux (assuming python3 as opposed to python)::

  sudo apt-get install make build-essential libfftw3-dev gfortran numdiff python3 python3-pip octave liboctave-dev

For any linux flavor see below for the optional ``numdiff`` (and very optional ``mwrap``). You should then compile via the various ``make`` tasks.

.. note::

   GCC versions on linux.  Rather than using the default GCC which may be as
   old as 4.8 or 5.4 on current linux systems, we **strongly** recommend you
   compile with a recent GCC version such as GCC 7.3 (which we used
   benchmarks in our SISC paper), or GCC 9.2.1. We do not recommend
   GCC versions prior to 7. We also **do not recommend GCC8** since
   its auto vectorization has worsened, and its kernel evaluation rate
   using the default looped piecewise-polynomial Horner code drops to
   less than 150 Meval/s/core on an i7. This contrasts 400-700
   Meval/s/core achievable with GCC7 or GCC9 on i7. If you wish to
   test these raw kernel evaluation rates, do into ``devel\``, compile
   ``test_ker_ppval.cpp`` and run ``fig_speed_ker_ppval.m`` in MATLAB. We are
   unsure if GCC8 is poor in Mac OSX (see below).


Tips for installing dependencies and compiling on Mac OSX
---------------------------------------------------------

.. note::

   Improved Mac OSX instructions, and possibly a brew package, will come shortly. Stay tuned. The below has been tested on 10.14 (Mojave) with both clang and gcc-8.

First you'll want to set up Homebrew, as follows.
If you don't have Xcode, install Command Line Tools
(this is only around 130 MB in contrast to the full 6 GB size of Xcode),
by opening a terminal (from ``/Applications/Utilities/``) and typing::

  xcode-select --install
   
You will be asked for an administrator password.
Then, also as an administrator,
install Homebrew by pasting the installation command from
https://brew.sh

Then do::

  brew install libomp fftw

This happens to also install the latest GCC, which is 8.2.0 in our tests.

.. note::
   
   There are two options for compilers: 1) the native ``clang`` which
   works with octave but will *not*
   so far allow you to link against fortran applications, or 2) GCC, which
   will allow fortran linking with ``gfortran``, but currently fails with
   octave.

First the **clang route**, which is the default.
Once you have downloaded FINUFFT, to set up for this, do::

  cp make.inc.macosx_clang make.inc

This gives you compile flags that should work with ``make test`` and other tasks. Optionally, install ``numdiff`` as below. Then
for python (note that pip is not installed with the default python v2)::

  brew install python3
  pip3 install numpy pybind11
  make python3
  
This should generate the ``finufftpy`` module (and ``finufftpy_cpp`` which it depends on).
However, we have found that it may fail with an error about ``-lstdc++``,
in which case you should try setting an environment variable::

  export MACOSX_DEPLOYMENT_TARGET=10.14

We have also found that running::

  pip3 install .

in the command line can work even when ``make python3`` does not (probably
to do with environment variables).
Octave interfaces work out of the box::

  brew install octave
  make octave

Look in ``make.inc.macosx_*``, and see below,
for ideas for building MATLAB MEX interfaces.

Alternatively, here's the **GCC route**, which we have also tested on Movaje::

  cp make.inc.macosx_gcc-8 make.inc

You must now by hand edit ``setup.py``, changing ``gcc`` to ``gcc-8`` and ``g++`` to ``g++-8``. Then proceed as above with python3. ``make fortran`` in addition to the above (apart from octave) should now work.

.. note::

   Choosing GCC-8 in OSX there is a
   problem with octave MEX compilation. Please help if you can!

   
General notes about compilation and tests
-----------------------------------------

We first describe compilation for default options (double precision, openmp) via GCC.
If you have a nonstandard unix environment (eg a Mac) or want to change the compiler,
then place your compiler and linking options in a new file ``make.inc``.
For example such files see ``make.inc.*``. See the text of ``makefile`` for discussion of what can be overridden.

Compile and do a rapid (less than 1-second) test of FINUFFT via::

  make test

This should compile the main libraries then run tests which should report zero crashes and zero fails. (If numdiff is absent, it instead produces output only about crashes; you will have to check by eye that accuracy is as expected.)
Note that the very first test run is ``test/finufft1d_basicpassfail`` which
does include a low-accuracy math test, producing the exit code 0 if success,
nonzero if fail. You can check the exit code thus::
  
  test/finufft1d_basicpassfail; echo $?

Use ``make perftest`` for larger spread/interpolation and NUFFT tests taking 10-20 seconds. This writes into ``test/results/`` where you will be able to compare to results from standard CPUs.

Run ``make`` without arguments for full list of possible make tasks.

``make examples`` to compile and run the examples for calling from C++ and from C.

The ``examples`` and ``test`` directories are good places to see usage examples.

``make fortran`` to compile and run the fortran wrappers and examples.

Note that the library includes fortran interfaces
defined in ``fortran/finufft_f.h``.

If there is an error in testing on a standard set-up,
please file a bug report as a New Issue at https://github.com/flatironinstitute/finufft/issues

Custom library compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to make the library for other data types. Currently
library names are distinct for single precision (``libfinufftf``) vs
double (``libfinufft``). However, single-threaded vs multithreaded are
built with the same name, so you will have to move them to other
locations, or build a 2nd copy of the repo, if you want to keep both
versions.

You *must* do at least ``make objclean`` before changing precision or openmp options.

**Single precision**: append ``PREC=SINGLE`` to the make task.
Single-precision saves half the RAM, and increases
speed slightly (<20%). The C++, C, and fortran demos are all tested in
single precision. However, it will break matlab, octave, python interfaces.

**Single-threaded**: append ``OMP=OFF`` to the make task.


Building MATLAB/octave wrappers, including in Mac OSX
-----------------------------------------------------

``make matlab`` to build the MEX interface to matlab.

``make octave`` to build the MEX-like interface to octave.

We have had success in Mac OSX Mojave compiling the octave wrapper out of the box.
For MATLAB, the MEX settings may need to be
overridden: edit the file ``mex_C++_maci64.xml`` in the MATLAB distro,
to read, for instance::

  CC="gcc-8"
  CXX="g++-8"
  CFLAGS="-ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread"
  CXXFLAGS="-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread"

These settings are copied from the ``glnxa64`` case. Here you will want to replace the compilers by whatever version of GCC you have installed, eg via brew,
  or the default gcc/g++ that are aliased to clang.
For pre-2016 MATLAB Mac OSX versions you'll instead want to edit the ``maci64``
section of ``mexopts.sh``.


Building the python wrappers
----------------------------

First make sure you have python3 and pip3 (or python and pip) installed and that you can already compile the C++ library (eg via ``make lib``).
Python links to this compiled library. You will get an error unless you first
compile the static library.
Next make sure you have NumPy and pybind11 installed::
  
  pip3 install numpy pybind11

You may then do ``make python3`` which calls
pip3 for the install then runs some tests. An additional test you could do is::

  python3 python_tests/run_speed_tests.py

In all the above, the suffix "3" should be omitted if you either want to work with python v2, or you are using a virtual python environment (see below).

See also Dan Foreman-Mackey's earlier repo that also wraps finufft, and from which we have drawn code: `python-finufft <https://github.com/dfm/python-finufft>`_

A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it (after installing python-virtualenv)::

  Open a terminal
  virtualenv -p /usr/bin/python3 env1
  . env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the env1 directory. (You can get out of the environment by typing ``deactivate``). Also see documentation for ``conda``. You then should use ``make python`` instead of ``make python3`` in the above.


Tips for installing optional dependencies
-----------------------------------------

Installing numdiff
~~~~~~~~~~~~~~~~~~

`numdiff <http://www.nongnu.org/numdiff>`_ by Ivano Primi extends ``diff`` to assess errors in floating-point outputs. It is an optional dependency that provides a better pass-fail test; in particular it allows the accuracy check message
``0 fails out of 5 tests done`` when ``make test`` is done for FINUFFT.
To install ``numdiff`` on linux,
download the latest version from
http://gnu.mirrors.pair.com/savannah/savannah/numdiff/
un-tar the package, cd into it, then build via ``./configure; make; sudo make install``.

This compilation fails on Mac OSX, for which we found the following was needed
in Mojave. Assume you un-tarred into ``/usr/local/numdiff-5.9.0``. Then::

  brew install gettext
  ./configure 'CFLAGS=-I/usr/local/opt/gettext/include' 'LDFLAGS=-L/usr/local/opt/gettext/lib'
  make
  sudo ln /usr/local/numdiff-5.9.0/numdiff /usr/local/bin

You should now be able to run ``make test`` in FINUFFT and get the second
message about zero fails.

Installing MWrap
~~~~~~~~~~~~~~~~

This is not needed for most users.
`MWrap <http://www.cs.cornell.edu/~bindel/sw/mwrap>`_
is a very useful MEX interface generator by Dave Bindel.
Make sure you have ``flex`` and ``bison`` installed.
Download version 0.33 or later from http://www.cs.cornell.edu/~bindel/sw/mwrap, un-tar the package, cd into it, then::
  
  make
  sudo cp mwrap /usr/local/bin/



