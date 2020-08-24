.. _install:

Installation
============

Quick linux install instructions
--------------------------------

In brief, go to the github page https://github.com/flatironinstitute/finufft and
follow instructions to download the source (eg see the green button).
Make sure you have packages ``fftw3`` and ``fftw3-devel`` installed.
Then ``cd`` into your FINUFFT directory and ``make test``, or ``make test -j8`` for a faster build.
This should compile the static
library in ``lib-static/``, some C++ test drivers in ``test/``, then run them,
printing some terminal output ending in::

  0 fails out of 8 tests done

If this fails see the more detailed instructions below.
If it succeeds, run ``make lib`` and proceed to link to the library.
Alternatively, try one of our `precompiled linux and OSX binaries <http://users.flatironinstitute.org/~ahb/codes/finufft-binaries>`_.
Type ``make`` to see a list of other aspects to build (examples, language
interfaces, etc).
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
* FFTW3 including its development libraries
* GNU ``make`` and other standard unix/POSIX tools such as ``bash``

Optional:

* for Fortran wrappers: compiler such as ``gfortran``
* for MATLAB/octave wrappers: MATLAB, or octave and its development libraries
* for the python wrappers you will need ``python`` (it is assumed you have python v3; v2 is unsupported). You will also need the python module ``pybind11``
* for rebuilding new matlab/octave wrappers (experts only): ``mwrap`` version>=0.33.10


Tips for installing dependencies on linux
-----------------------------------------

On a Fedora/CentOS linux system, dependencies can be installed as follows::

  sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp octave octave-devel

.. note::

   we are not exactly sure how to install python3 and pip3 using yum

Alternatively, on Ubuntu linux::

  sudo apt-get install make build-essential libfftw3-dev gfortran python3 python3-pip octave liboctave-dev

You should then compile via the various ``make`` tasks, eg ``make test -j8``
then checking you got ``0 fails``.  

.. note::

   GCC versions on linux.  Rather than using the default GCC which may be as
   old as 4.8 or 5.4 on current linux systems, we **strongly** recommend you
   compile with a recent GCC version such as GCC 7.3 (which we used
   benchmarks in our SISC paper), or GCC 9+. We do not recommend
   GCC versions prior to 7. We also **do not recommend GCC8** since
   its auto vectorization has worsened, and its kernel evaluation rate
   using the default looped piecewise-polynomial Horner code drops to
   less than 150 Meval/s/core on an i7. This contrasts 400-700
   Meval/s/core achievable with GCC7 or GCC9 on i7. If you wish to
   test these raw kernel evaluation rates, do into ``devel/``, compile
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

This gives you compile flags that should work with ``make test`` and other tasks. Please try ``make test`` at this point, and check for ``0 fails``. Then for python (note that pip is not installed with the default python v2)::

  brew install python3
  pip3 install numpy pybind11
  make python
  
This should generate the ``finufft`` module.
However, we have found that it may fail with an error about ``-lstdc++``,
in which case you should try setting an environment variable::

  export MACOSX_DEPLOYMENT_TARGET=10.14

We have also found that running::

  pip3 install .

in the command line can work even when ``make python`` does not (probably
to do with environment variables).
Octave interfaces work out of the box::

  brew install octave
  make octave

Look in ``make.inc.macosx_*``, and see below,
for ideas for building MATLAB MEX interfaces.

Alternatively, here's the **GCC route**, which we have also tested on Movaje::

  cp make.inc.macosx_gcc-8 make.inc

You must now by hand edit ``python/setup.py``, changing ``gcc`` to ``gcc-8`` and ``g++`` to ``g++-8``. Then proceed as above with python3. ``make fortran`` in addition to the above (apart from octave) should now work.

.. note::

   Choosing GCC-8 in OSX there is a
   problem with octave MEX compilation. Please help if you can!

   
Details about compilation and tests
-----------------------------------------

The make tasks (eg ``make lib``) compiles double and single precision functions,
which live simultaneously in ``libfinufft``, with distinct function names.

The only selectable option at compile time is
multithreaded (default, using OpenMP) vs single-threaded
(to achieve this append ``OMP=OFF`` to the make tasks).
Since you may always set ``opts.nthreads=1`` when calling the multithreaded
library,
the point of having a single-threaded library is
mostly for small repeated problems to avoid any OpenMP overhead, or
for debugging purposes.
You *must* do at least ``make objclean`` before changing this threading
option.

.. note::

   By default, neither the multithreaded or single-threaded library (e.g. made by ``make lib OMP=OFF``) are thread-safe, due to the FFTW3 plan stage. However, see below for the compiler option to fix this if you have a recent FFTW3 version.

If you have a nonstandard unix environment (eg a Mac) or want to change the compiler or its flags,
then place your compiler and linking options in a new file ``make.inc``.
For example such files see ``make.inc.*``. See the text of ``makefile`` for discussion of what can be overridden.

Compile and do a rapid (few seconds duration) test of FINUFFT via::

  make test

This should compile the main libraries then run double- and single-precision tests which should report zero segfaults and zero fails.
Its initial test is ``test/basicpassfail`` which is the most basic smoke test,
producing the exit code 0 if success, nonzero if fail.
You can check the exit code thus::
  
  test/basicpassfail; echo $?

The make task also runs ``(cd test; ./check_finufft.sh)`` which is the main
validation of the library in double precision, and
``(cd test; ./check_finufft.sh SINGLE)`` which does it in single precision.
Text (and stderr) outputs are written into ``test/results/*.out``.

Use ``make perftest`` for larger spread/interpolation and NUFFT tests taking 10-20 seconds. This writes log files into ``test/results/`` where you will be able to compare to results from standard CPUs.

Run ``make`` without arguments for full list of possible make tasks.

``make examples`` to compile and run the examples for calling from C++ and from C.

``make fortran`` to compile and run the fortran wrappers and examples.

Here are all the **compile flags** that the FINUFFT source responds to.
Active them by adding a line of the form ``CFLAGS+=-DMYFLAG`` in your ``make.inc``:

* ``-DFFTW_PLAN_SAFE``: This makes FINUFFT call ``fftw_make_planner_thread_safe()`` as part of its FFTW3 planner stage; see http://www.fftw.org/fftw3_doc/Thread-safety.html. This makes FINUFFT thread-safe. This is only available in FFTW version >=3.3.5; for this reason it is not the default.

* ``-DSINGLE``: This is internally used by our build process to switch
  (via preprocessor macros) the source from double to single precision.
  You should not need to use this flag yourself.


If there is an error in testing on a standard set-up,
please file a bug report as a New Issue at https://github.com/flatironinstitute/finufft/issues





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

.. _install-python:

Building the python wrappers
----------------------------

First make sure you have python3 and pip3 (or python and pip) installed, and that you can already compile the C++ library (eg via ``make test``).
Next make sure you have NumPy and pybind11 installed::
  
  pip install numpy pybind11

You may then do ``make python`` which calls
``pip`` for the install then runs some tests.
An additional test you could do is::

  python python/run_speed_tests.py

See also Dan Foreman-Mackey's earlier repo that also wraps finufft, and from which we have drawn code: `python-finufft <https://github.com/dfm/python-finufft>`_

A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it (after installing python-virtualenv)::

  Open a terminal
  virtualenv -p /usr/bin/python3 env1
  . env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the env1 directory. (You can get out of the environment by typing ``deactivate``). Also see documentation for ``conda``. In both cases ``python`` will call the version of python you set up, which these days should be v3.
