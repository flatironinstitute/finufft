.. _install:

Installation
============

.. note::
   
   If the below instructions fail in any operating system, try the relevant version of our `precompiled linux, OSX, and Windows binaries <http://users.flatironinstitute.org/~ahb/codes/finufft-binaries>`_, place it (or them) in your linking path, and try ``make test``. We will be adding to these as needed; please email us to contribute or request one.

.. note::

   Python-only users can simply install via ``pip install finufft`` which downloads a generic binary from PyPI. Only if you prefer a custom compilation, see :ref:`below<install-python>`.

Below we deal with the three standard OSes in order: 1) **linux**, 2) **Mac OSX**, 3) **Windows**.
We have some users contributing settings for other OSes, for instance
PowerPC. The general procedure to download, then compile for such a special setup is, illustrating with the PowerPC case::

  git clone https://github.com/flatironinstitute/finufft.git
  cd finufft
  cp make.inc.powerpc make.inc
  make test -j

Have a look for ``make.inc.*`` to see what is available, and/or edit your ``make.inc`` based on looking in the ``makefile`` and quirks of your local setup. As of 2021, we have continuous integration which tests the default (linux) settings in this ``makefile``, plus those in three OS-specific setup files::

  make.inc.macosx_clang
  make.inc.macosx_gcc-10
  make.inc.windows_msys
  
If there is an error in testing on what you consider a standard set-up,
please file a detailed bug report as a New Issue at https://github.com/flatironinstitute/finufft/issues

  
Quick linux install instructions
--------------------------------

Make sure you have packages ``fftw3`` and ``fftw3-dev`` (or their
equivalent on your distro) installed.
Then ``cd`` into your FINUFFT directory and do ``make test -j``.
This should compile the static
library in ``lib-static/``, some C++ test drivers in ``test/``, then run them,
printing some terminal output ending in::

  0 segfaults out of 8 tests done
  0 fails out of 8 tests done

This output repeats for double then single precision (hence, scroll up to check the double also gave no fails).
If this fails, see the more detailed instructions below.
If it succeeds,
please look in ``examples/``, ``test/``, and the rest of this manual,
for examples of how to call and link to the library.
Type ``make`` to see a list of other aspects the user can build
(examples, language interfaces, etc).


Dependencies
------------

This library is fully supported for unix/linux, and partially for
Mac OSX for Windows (eg under MSYS or WSL using MinGW compilers).

For the basic libraries you need

* C++ compiler supporting C++14, such ``g++`` in GCC (version >=5.0), or ``clang`` (version >=3.4)
* FFTW3 including its development libraries
* GNU ``make`` and other standard unix/POSIX tools such as ``bash``

Optional:

* for Fortran wrappers: compiler such as ``gfortran`` in GCC
* for MATLAB wrappers: MATLAB (versions at least R2016b up to current work)
* for Octave wrappers: recent Octave version at least 4.4, and its development libraries
* for the python wrappers you will need ``python`` version at least 3.6 (python 2 is unsupported), with ``numpy``.


1) Linux: tips for installing dependencies and compiling
-------------------------------------------------------------------

On a Fedora/CentOS linux system, the base dependencies can be installed by::

  sudo yum install make gcc gcc-c++ fftw-devel libgomp
  
To add Fortran and Octave language interfaces also do::

  sudo yum install gcc-gfortran octave octave-devel

.. note::

  We are not exactly sure how to install ``python3`` and ``pip3`` using ``yum``. You may prefer to use ``conda`` or ``virtualenv`` to set up a python environment anyway (see bottom).

Alternatively, on Ubuntu linux, base dependencies are::

  sudo apt-get install make build-essential libfftw3-dev

and for Fortran, Python, and Octave language interfaces also do::

  sudo apt-get gfortran python3 python3-pip octave liboctave-dev

In older distros you may have to compile ``octave`` from source to get the needed >=4.4 version.

You should then compile and test the library via various ``make`` tasks, eg::

  make test -j
  
then checking you got ``0 fails``.
This compiles the main libraries then runs double- and single-precision tests, each of which should report zero segfaults and zero fails.

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
   unsure if GCC8 is so poor in Mac OSX (see below).

The make tasks (eg ``make lib``) compiles double and single precision functions,
which live simultaneously in ``libfinufft``, with distinct function names.

The only selectable option at compile time is
multithreaded (default, using OpenMP) vs single-threaded
(to achieve this append ``OMP=OFF`` to the make tasks).
Since you may always set ``opts.nthreads=1`` when calling the multithreaded
library, the point of having a single-threaded library is
mostly for small repeated problems to avoid *any* OpenMP overhead, or
for debugging purposes.
You *must* do at least ``make objclean`` before changing this threading
option.

.. note::

   By default, neither the multithreaded or single-threaded library (e.g. made by ``make lib OMP=OFF``) are thread-safe, due to the FFTW3 plan stage. However, keep reading for the compiler option to fix this if you have a recent FFTW3 version.

**Testing**. The initial test is ``test/basicpassfail`` which is the most basic double-precision smoke test,
producing the exit code 0 if success, nonzero if fail.
You can check the exit code thus::
  
  test/basicpassfail; echo $?

The single-precision version is ``test/basicpassfailf``.
The make task also runs ``(cd test; OMP_NUM_THREADS=4 ./check_finufft.sh)`` which is the main
validation of the library in double precision, and
``(cd test; OMP_NUM_THREADS=4 ./check_finufft.sh SINGLE)`` which does it in single precision.
Since these call many tiny problem sizes, they will (due to openmp and fftw thread-wise overheads)
run much faster with less than the full thread count, explaining our use of 4 threads.
Text (and stderr) outputs are written into ``test/results/*.out``.

Use ``make perftest`` for larger spread/interpolation and NUFFT tests taking 10-20 seconds. This writes log files into ``test/results/`` where you will be able to compare to results from standard CPUs.

Run ``make`` without arguments for full list of possible make tasks.

``make examples`` to compile and run the examples for calling from C++ and from C.

``make fortran`` to compile and run the fortran wrappers and examples.

**High-level interfaces**.
See :ref:`below<install-python>` for python compilation.

``make matlab`` to compile the MEX interface to matlab,
then within MATLAB add the ``matlab`` directory to your path,
cd to ``matlab/test`` and run ``check_finufft`` which should run for 5 secs
and print a bunch of errors around ``1e-6``.

.. note::

   If this MATLAB test crashes, it is most likely to do with incompatible versions of OpemMP. Thus, you will want to make (or add to) a file ``make.inc`` the line::

      OMPLIBS=/usr/local/MATLAB/R2020a/sys/os/glnxa64/libiomp5.so

   or appropriate to your MATLAB version. You'll want to check this shared
   object exists. Then ``make clean`` and ``make test -j``, finally
   ``make matlab`` again.
  
``make octave`` to compile and test the MEX-like interface to Octave.



Compilation flags and make.inc settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is for experts.
Here are all the flags that the FINUFFT source responds to.
Activate them by adding a line of the form ``CXXFLAGS+=-DMYFLAG`` in your ``make.inc``:

* ``-DFFTW_PLAN_SAFE``: This makes FINUFFT call ``fftw_make_planner_thread_safe()`` as part of its FFTW3 planner stage; see http://www.fftw.org/fftw3_doc/Thread-safety.html. This makes FINUFFT thread-safe. See ``examples/threadsafe1d1.cpp``. This is only available in FFTW version >=3.3.6; for this reason it is not yet the default.

* ``-DSINGLE``: This is internally used by our build process to switch
  (via preprocessor macros) the source from double to single precision.
  You should not need to use this flag yourself.

Here are some other settings that you may need to adjust in ``make.inc``:


* Switching to linking tests, examples, etc, with PTHREADS instead of the default OMP version of FFTW, is achieved by inserting into ``make.inc`` the line
``FFTWOMPSUFFIX = threads``.




  
2) Mac OSX: tips for installing dependencies and compiling
-----------------------------------------------------------

.. note::

   A brew package will come shortly; stay tuned. However, the below has been tested on 10.14 (Mojave) with both clang and gcc-8, and 10.15 (Catalina) with clang.

First you'll want to set up Homebrew, as follows. We assume a fresh OSX machine.
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

If you are python-only, use::

     brew install python3
     pip3 install finufft
     
Or, for experts to compile python interfaces locally using either clang or gcc,
see :ref:`below<install-python>`.

Now to compiling the library for C++/C/fortran/MATLAB/octave use.
There are now two options for compilers: 1) the native ``clang`` which
works with octave but will *not*
so far allow you to link against fortran applications, or 2) GCC, which
will allow fortran linking with ``gfortran``, but currently fails with
octave.

The clang route (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

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

The GCC route
~~~~~~~~~~~~~~

This is less recommended, unless you need to link from ``gfortran``, when it
appears to be essential. We have tested on Movaje::

  cp make.inc.macosx_gcc-8 make.inc
  make test -j
  make fortran

which also compiles and tests the fortran interfaces.
In Catalina you'll probably need to edit to ``g++-10`` in your ``make.inc``.
We find python may be built as :ref:`below<install-python>`.
We found that octave interfaces do not work with GCC; please help.
For MATLAB, the MEX settings may need to be
overridden: edit the file ``mex_C++_maci64.xml`` in the MATLAB distro,
to read, for instance::

  CC="gcc-8"
  CXX="g++-8"
  CFLAGS="-ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread"
  CXXFLAGS="-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread"

These settings are copied from the ``glnxa64`` case. Here you will want to replace the compilers by whatever version of GCC you have installed, eg via brew.
For pre-2016 MATLAB Mac OSX versions you'll instead want to edit the ``maci64``
section of ``mexopts.sh``.

.. note::

   GCC with OSX is only partially supported. Please help us if you can!



3) Windows: tips for compiling
-------------------------------   
   
We have users who have adjusted the makefile to work - at least to some extent - on Windows 10. If you are only interested in calling from Octave (which already comes with MinGW-w64 and FFTW), then we have been told this can be done very simply: from within Octave, go to the ``finufft`` directory and do ``system('make octave')``. You may have to tweak ``OCTAVE`` in your ``make.inc`` in a similar fashion to below.

More generally, please make sure to have a recent version of Mingw at hand, preferably with a 64bit version of gnu-make like the WinLibs standalone build of GCC and MinGW-w64 for Windows. Note that most MinGW-w64 distributions, such as TDM-GCC, do not feature the 64bit gnu-make. Fortunately, this limitation is only relevant to run the tests. To prepare the build of the static and dynamic libraries run::

  copy make.inc.windows_mingw make.inc

Subsequently, open this ``make.inc`` file with the text editor of your choice and assign the parent directories of the FFTW header file to ``FFTW_H_DIR``, of the FFTW libraries to ``FFTW_LIB_DIR``, and of the GCC OpenMP library lgomp.dll to ``LGOMP_DIR``. Note that you need the last-mentioned only if you plan to build the MEX-interface for MATLAB. Now, you should be able to run::

  make lib 

If the command ``make`` cannot be found and the MinGW binaries are part of your system PATH: Keep in mind that the MinGW installation contains only a file called mingw32-make.exe, not make.exe. Create a copy of this file, call it make.exe, and make sure the corresponding parent folder is part of your system PATH. If the library is compiled successfully, you can try to run the tests. Note that your system has to fulfill the following prerequisites to this end: A Linux distribution set up via WSL (has been tested with Ubuntu 20.04 LTS from the Windows Store) and the 64bit gnu-make mentioned before. Further, make sure that the directory containing the FFTW-DLLs is part of your system PATH. Otherwise the executables built will not run. As soon as you have everything set up, run the following command::

  make test

In a similar fashion, the examples can now be build with ``make examples``. This rule of the makefile does neither require WSL nor the 64bit gnu-make and should hopefully work out-of-the-box. Finally, it is also possible to build the MEX file needed to call FINUFFT from MATLAB. Since the MinGW support of MATLAB is somewhat limited, you will probably have to define the environment variable ``MW_MINGW64_LOC`` and assign the path of your MinGW installation. Hint to avoid misunderstandings: The last-mentioned directory contains folders named ``bin``, ``include``, and ``lib`` among others. Then, the following command should generate the required MEX-file::

  make matlab

For users who work with Windows using MSYS and MinGW compilers. Please
try::

  cp make.inc.windows_msys make.inc
  make test -j

We seek help with Windows support. Also see https://github.com/flatironinstitute/finufft/issues




.. _install-python:

Building a python interface to a locally compiled library
-----------------------------------------------------------------------

Recall that the basic user may simply ``pip install finufft``,
then check it worked via::

  python3 python/test/run_accuracy_tests.py

However, a user or developer may want to build a python wrapper to their locally
compiled FINUFFT library, perhaps for more speed. We now describe this,
for all OSes.
We assume python3 (hence pip3).
First make sure you have pip
installed, and that you can already compile the C++ library (eg via ``make test``).
Next make sure you have the required python packages::

  pip3 install numpy

You may then::

  make python

which builds the ``finufft`` module,
installs (in editable mode) via pip, then runs some tests and examples.
You will see that the ``finufftc`` shared object appears in the ``python/finufft`` directory.
An additional performance test you could then do is::

  python3 python/test/run_speed_tests.py

.. note::

   On OSX, if trouble with python with clang: we have found that the above may fail with an error about ``-lstdc++``, in which case you should try setting an environment variable like::

     export MACOSX_DEPLOYMENT_TARGET=10.14

   where you should replace 10.14 by your OSX number. We have also in the past found that running::

     pip3 install ./python

   in the command line can work even when ``make python`` does not (probably to do with environment variables).

.. note::

   Our new (v2.0.1) python interface is quite different from Dan Foreman-Mackey's original repo that wrapped finufft: `python-finufft <https://github.com/dfm/python-finufft>`_, or Jeremy Magland's. The interface is simpler, and the existing library is linked to. Under the hood we now use ``ctypes`` instead of ``pybind11``.
  

A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it from a shell in the FINUFFT top directory (after installing ``python-virtualenv``)::

  virtualenv -p /usr/bin/python3 env1
  source env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the ``env1`` directory. (You can get out of the environment by typing ``deactivate``). Also see documentation for ``conda``. In both cases ``python`` will call the version of python you set up. To get the packages FINUFFT needs::

  pip install -r python/requirements.txt

Then ``pip install finufft`` or build as above.
