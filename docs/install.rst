.. _install:

Installation
============

.. note::
   
   If the below fails in any operating system, try the relevant version of our `precompiled linux, OSX, and Windows binaries <http://users.flatironinstitute.org/~ahb/codes/finufft-binaries>`_, place it (or them) in your linking path, and try ``make test``. We will be adding to these as needed; please email us to contribute or request one.



Quick linux install instructions
--------------------------------

In brief, go to the github page https://github.com/flatironinstitute/finufft and
follow instructions to download the source (eg see the green button).
Make sure you have packages ``fftw3`` and ``fftw3-devel`` installed.
Then ``cd`` into your FINUFFT directory and ``make test``, or ``make test -j`` for a faster build.
This should compile the static
library in ``lib-static/``, some C++ test drivers in ``test/``, then run them,
printing some terminal output ending in::

  0 segfaults out of 8 tests done
  0 fails out of 8 tests done

If this fails, see the more detailed instructions below.
If it succeeds, run ``make lib`` and proceed to link to the library.
Please look in ``examples/``, ``test/``, and the rest of this manual,
for examples of how to call and link to the library.
Type ``make`` to see a list of other aspects to build (examples, language
interfaces, etc).


Dependencies
------------

This library is fully supported for unix/linux and almost fully on
Mac OSX.  We have also heard that it can be compiled under Windows
using MinGW; we also suggest trying within the Windows Subsystem for
Linux (WSL).

For the basic libraries you need

* C++ compiler supporting C++14, such ``g++`` in GCC (version >=5.0), or ``clang`` (version >=3.4)
* FFTW3 including its development libraries
* GNU ``make`` and other standard unix/POSIX tools such as ``bash``

Optional:

* for Fortran wrappers: compiler such as ``gfortran`` in GCC
* for MATLAB wrappers: MATLAB (versions at least R2016b up to current work)
* for octave wrappers: recent octave version at least 4.4, and its development libraries
* for the python wrappers you will need ``python`` (it is assumed you have python v3; v2 is unsupported). You will also need the python module ``pybind11``


Tips for installing dependencies on linux
-----------------------------------------

On a Fedora/CentOS linux system, dependencies can be installed as follows::

  sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp octave octave-devel

.. note::

  We are not exactly sure how to install python3 and pip3 using yum. You may prefer to use conda to set up a python environment (see below).

Alternatively, on Ubuntu linux::

  sudo apt-get install make build-essential libfftw3-dev gfortran python3 python3-pip octave liboctave-dev

In older distro you may have to compile octave from source to get a >=4.4 version.

You should then compile via the various ``make`` tasks, eg ``make test -j8``
then checking you got ``0 fails``.

.. note::

   GCC versions on linux: long-term linux distros ship old GCC versions
   that may not be C++14 compatible. We recommend that you
   compile with a recent GCC, at least GCC 7.3 (which we used
   for benchmarks in 2018 in our SISC paper), or GCC 9+. We do not recommend
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

   A brew package will come shortly; stay tuned. The below has been tested on 10.14 (Mojave) with both clang and gcc-8, and 10.15 (Catalina) with clang.

First you'll want to set up Homebrew, as follows.
If you don't have Xcode, install Command Line Tools
(which is a few hundred MB download, much smaller than the now
10 GB size of Xcode),
by opening a terminal (from ``/Applications/Utilities/``) and typing::

  xcode-select --install
   
You will be asked for an administrator password.
Then, also as an administrator,
install Homebrew by pasting the installation command from
https://brew.sh

Then do::

  brew install libomp fftw

This happens to also install the latest GCC (which was 8.2.0 in Mojave,
and 10.2.0 in Catalina, in our tests).

.. note::
   
   There are two options for compilers: 1) the native ``clang`` which
   works with octave but will *not*
   so far allow you to link against fortran applications, or 2) GCC, which
   will allow fortran linking with ``gfortran``, but currently fails with
   octave.

First the **clang route**, which is our default.
Once you have downloaded FINUFFT from github, go to its top directory.
You now need to decide if you will be wanting to call FINUFFT from
MATLAB (and currently have MATLAB installed). If so, do::

  cp make.inc.macosx_clang_matlab make.inc

Else if you don't have MATLAB, do::

  cp make.inc.macosx_clang make.inc

.. note::

  The difference here is the version of OpenMP linked: MATLAB crashes when ``gomp`` is linked, so for MATLAB users the OpenMP version used by MATLAB must be linked against (``iomp5``), not ``gomp``.

Whichever you picked, now try ``make test -j``, and clang should compile and you should get ``0 fails``.

**clang MATLAB setup**. Assuming you chose the MATLAB clang variant above,
you should now ``make matlab``. To test, open MATLAB, ``addpath matlab``,
``cd matlab/test``, and ``check_finufft``, which should complete in around 5 seconds.

.. note::

   Unfortunately OSX+MATLAB+mex is notoriously poorly supported, and you may need to search the web for help on that, then `check you are able to compile a simple mex file first <https://www.mathworks.com/help/matlab/matlab_external/getting-started.html>`_.
   For instance, on Catalina (10.15.6), ``make matlab`` fails with a warning involving Xcode ``license has not been accepted``, and then an error with ``no supported compiler was found``. Eventually `this property file hack worked <https://www.mathworks.com/matlabcentral/answers/307362-mex-on-macosx-without-xcode>`_, which simply requires typing ``/usr/libexec/PlistBuddy -c 'Add :IDEXcodeVersionForAgreedToGMLicense string 10.0' ~/Library/Preferences/com.apple.dt.Xcode.plist``
   Please also read our https://github.com/flatironinstitute/finufft/issues and if you *are* able to mex compile, but ``make matlab`` fails, post a new Issue.
   
Octave interfaces work out of the box (this also runs a self-test)::

  brew install octave
  make octave

Then for python (note that pip is not installed with the default python v2)::

  brew install python3
  pip3 install numpy pybind11 python-dotenv
  make python
  
This should generate the ``finufft`` module and run some python test outputs.

.. note::

   If trouble with python with clang: 1) we have found that the above may fail with an error about ``-lstdc++``, in which case you should try setting an environment variable like::

     export MACOSX_DEPLOYMENT_TARGET=10.14

  We have also found that running::

    pip3 install .

  in the command line can work even when ``make python`` does not (probably to do with environment variables).

Alternatively, here's the **GCC route**, which is less recommended, unless you want to link from gfortan. We have also tested on Movaje::

  cp make.inc.macosx_gcc-8 make.inc

You must now by hand edit ``python/setup.py``, changing ``gcc`` to ``gcc-8`` and ``g++`` to ``g++-8``. Then proceed as above with python3. ``make fortran`` in addition to the above (apart from octave) should now work.
In Catalina you'll probably need to replace with ``g++-10``.

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

``make matlab`` to compile the MEX interface to matlab.

``make octave`` to compile the MEX-like interface to octave.

We have had success in Mac OSX Mojave compiling the octave wrapper out of the box.
For MATLAB, the MEX settings may need to be
overridden: edit the file ``mex_C++_maci64.xml`` in the MATLAB distro,
to read, for instance::

  CC="gcc-8"
  CXX="g++-8"
  CFLAGS="-ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread"
  CXXFLAGS="-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread"

These settings are copied from the ``glnxa64`` case. Here you will want to replace the compilers by whatever version of GCC you have installed, eg via brew, or the default gcc/g++ that are aliased to clang.
For pre-2016 MATLAB Mac OSX versions you'll instead want to edit the ``maci64``
section of ``mexopts.sh``.

.. _install-python:

Building the python wrappers
----------------------------

First make sure you have python3 and pip3 (or python and pip) installed, and that you can already compile the C++ library (eg via ``make test``).
Next make sure you have NumPy and pybind11 installed::
  
  pip install numpy pybind11

You may then do ``make python`` which calls
``pip`` for the install then runs some tests and examples.
An additional performance test you could then do is::

  python python/test/run_speed_tests.py

Note that our new (v2.0) python interface is quite different from the Dan Foreman-Mackey's original repo that wrapped finufft: `python-finufft <https://github.com/dfm/python-finufft>`_. We now use `ctypes`.
  

A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it from a shell (after installing ``python-virtualenv``)::

  virtualenv -p /usr/bin/python3 env1
  . env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the ``env1`` directory. (You can get out of the environment by typing ``deactivate``). Also see documentation for ``conda``. In both cases ``python`` will call the version of python you set up, which these days should be v3.
