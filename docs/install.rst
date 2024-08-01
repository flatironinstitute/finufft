.. _install:

Installation
============

There are two main ways to compile this library from source:
via CMake (the recommended modern way, being more platform-independent),
or via a GNU ``makefile`` (which has various settings for linux, OSX, Windows).
We currently support both, and detail them in that order in the text below.
If you cannot get FINUFFT to compile, as a last resort you might find
a precompiled binary for your platform under Assets for various
`releases <https://github.com/flatironinstitute/finufft/releases>`_.
Please post an `Issue <https://github.com/flatironinstitute/finufft/issues>`_
to document your installation problem.
When using CMake, finufft requires no external dependencies except (c/c++ compilers supporting OpenMP).
However GNU ``makefile`` assumes that FFTW (single/double) are installed.

.. note::
    finufft builds with no issues on Linux and MacOS using any compiler, in our experience GCC-13 gives best performance.
    On Windows MSVC works fine. The llvm toolchain included in Visual Studio does not seem to have OpenMP, it is possible to build single-threaded FINUFFT.
    The official windows LLVM distribution builds finufft with no issues but debug builds using sanitizers break.

.. note::

    On windows finufft built with MSVC requires ``VCOMP140D.DLL`` which is part of the `Microsoft Visual C++ Redistributable <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_.
    It is likely to be already installed in your system.
    If the library is built with LLVM it requires ``libomp140.x86.64.dll``, more information `here <https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/>`_.
    Python-only users can simply install via ``pip install finufft`` which downloads a generic binary from PyPI. Only if you prefer a custom compilation, see :ref:`below<install-python>`.

CMake CPM Based Installation
----------------------------

This is the easiest way to install ``finufft`` if you are using CMake in your own project.
First include `CPM <https://github.com/cpm-cmake/CPM.cmake>`_ to your project.

The easiest way is to follow the `instructions <https://github.com/cpm-cmake/CPM.cmake/wiki/Downloading-CPM.cmake-in-CMake>`_ to automatically add CPM to cmake.

Then add the following to your ``CMakeLists.txt``:

.. code-block:: cmake

  # short version
  CPMAddPackage("gh:flatironinstitute/finufft@2.3")

  # alternative in case custom options are needed
  CPMAddPackage(
    NAME             Finufft
    GIT_REPOSITORY   https://github.com/flatironinstitute/finufft.git
    GIT_TAG          2.3
    GIT_SHALLOW      Yes
    GIT_PROGRESS     Yes
    EXCLUDE_FROM_ALL Yes
    SYSTEM
  )

  target_link_library(your_executable [PUBLIC|PRIVATE|INTERFACE] finufft)

Then cmake will automatically download the library and link it to your executable.

CMake FetchContent Based Installation
----------------------------

Another way to include finufft in the project is to use FetchContent
which is provided directly by cmake.
To do so add the following to your ``CMakeLists.txt``:

.. code-block:: cmake

    include(FetchContent)

    # Define the finufft library
    FetchContent_Declare(
      finufft
      GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
      GIT_TAG 2.3
    )

    # Make the content available
    FetchContent_MakeAvailable(finufft)

    # Optionally, link the finufft library to your target
    target_link_libraries(your_executable [PUBLIC|PRIVATE|INTERFACE] finufft)

Then cmake will automatically download the library and link it to your executable.

CMake Based Installation
------------------------

These instructions are in draft form.
Make sure you have ``cmake`` version at least 3.19.
The basic quick download, building, and test is then:

.. code-block:: bash

  git clone https://github.com/flatironinstitute/finufft.git
  cd finufft
  cmake -S . -B build -DFINUFFT_BUILD_TESTS=ON --install-prefix /path/to/install
  cmake --build build
  ctest
  cmake --install .

.. note::

   If you don't supply `--install-prefix`, it will default to ``/usr/local`` on most systems. If you don't have root access, you must supply a prefix you can write to such as ``$HOME/local``.

In ``build``, this creates ``libfinufft.a`` or ``libfinufft.so``, and runs a test that should take a
few seconds and report ``100% tests passed, 0 tests failed out of 17``.  To use the library, link against
either the static or dynamic library in ``build`` or your installed version
(i.e. ``/path/to/install/lib64/libfinufft.so`` or ``/path/to/install/lib/libfinufft.so``). If you install
anywhere other than standard system wide locations (``/usr/local``), building/linking requires you specify the
location of the library. If you link the shared library, you should also tell your compiled binary to store
the location of that library in its ``RPATH``. Let's say you installed with the prefix ``$HOME/local``, your
system prefers the ``lib64`` library directory, and you're still in the build directory. Then...

.. code-block:: bash

  g++ -o simple1d1 ../examples/simple1d1.cpp -I$HOME/local/include -L$HOME/local/lib64 -Wl,-rpath $HOME/local/lib64 -lfinufft -O2


will manually build the ``simple1d1`` example and drop it in the current directory.

Here are all our build options, showing name, explanatory text, and default value, straight from the ``CMakeLists.txt`` file:

.. literalinclude:: ../CMakeLists.txt
   :language: cmake
   :start-after: @cmake_opts_start
   :end-before: @cmake_opts_end

.. note::
    It is possible to choose between ``FFTW`` and `DUCC fft <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_, Ducc fft is from the same author as `pocket fft <https://gitlab.mpcdf.mpg.de/mtr/pocketfft>`_.
    Pocket fft is the fft used by `scipy <https://scipy.org/>`_.
    An idea about ducc performance can be found in `this discussion <https://github.com/flatironinstitute/finufft/pull/463#issuecomment-2223988300>`_. We encourage the power user can try switching to ducc0 to see if it improves performance.

For convenience we also provide a number of `cmake presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_
for various options and compilers, in ``CMakePresets.json`` (this will grow to replace the old ``make.inc.*`` site files).
For example, to configure, build and test the development preset (which builds tests and examples), from ``build`` do:

.. code-block:: bash

  cmake -S . -B build --preset dev # dev is the preset name
  cmake --build build
  ctest

.. warning::

  Intel compilers (unlike GPU compilers) currently engage ``fastmath`` behavior with ``-O2`` or ``-O3``. This may interfere with our use of ``std::isfinite`` in our test codes. For this reason in the Intel presets ``icx`` and ``icc`` have set ``-fp-model=strict``. You may get more speed if you remove this flag, or try ``-fno-finite-math-only``.

From other CMake projects, to use ``finufft`` as a library, simply add this repository as a subdirectory using
``add_subdirectory``, and use ``target_link_library(your_executable finufft)``.

.. note::

   CMake compiling on linux at Flatiron Institute (Rusty cluster). We have had a report that if you want to use LLVM, you need to ``module load llvm/16.0.3`` otherwise the default ``llvm/14.0.6`` does not find ``OpenMP_CXX``.


Classic GNU make based route
----------------------------

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


Quick linux GNU MAKE install instructions
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
* FFTW3 (version at least 3.3.6) including its development libraries
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
   that may not be C++17 compatible. We recommend that you
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

* ``-DSINGLE``: This is internally used by our build process to switch
  (via preprocessor macros) the source from double to single precision.
  You should not need to use this flag yourself.

Here are some other settings that you may need to adjust in ``make.inc``:

* Switching to linking tests, examples, etc, with PTHREADS instead of the default OMP version of FFTW, is achieved by inserting into ``make.inc`` the line ``FFTWOMPSUFFIX = threads``.





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
you should now ``make matlab``. You may need to do ``make matlab -j``; see
https://github.com/flatironinstitute/finufft/issues/157 which needs attention.
To test, open MATLAB, ``addpath matlab``,
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
appears to be essential. The basic idea is::

  cp make.inc.macosx_gcc-12 make.inc
  make test -j
  make fortran

which also compiles and tests the fortran interfaces.
You may need to edit to ``g++-11``, or whatever your GCC version is,
in your ``make.inc``.

.. note::

   A problem between GCC and the new XCode 15 requires a workaround to add ``LDFLAGS+=-ld64`` to force the old linker to be used. See the above file ``make.inc.macosx_gcc-12``.

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



3) Windows GNU MAKE: tips for compiling
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
then check it worked via either (if you have ``pytest`` installed)::

  pytest python/finufft/test

or the older-style eyeball check with::

  python3 python/finufft/test/run_accuracy_tests.py

which should report errors around ``1e-6`` and throughputs around 1-10 million points/sec.

However, a user or developer may want to build a python wrapper to their locally
compiled FINUFFT library, perhaps for more speed. We now describe this,
for all OSes.
We assume ``python3`` (hence ``pip3``; make sure you have that installed).

First, compile the shared C++ library, via, eg ``make lib -j`` (using the old-style ``makefile``),
or via::

  make -p build
  cd build
  cmake ..
  cmake --build . -j
  cd ..

You may then run::

  pip3 install -e python/finufft

which builds the ``finufft`` Python module, linking to the ``.so``,
and installs (in editable mode) via pip.
You will see that the ``finufftc.*.so`` shared object appears in the ``python/finufft/finufft/`` directory.
You should then run the above tests. You could also run tests and examples via ``make python``.

An additional performance test you could then do is::

  python3 python/finufft/test/run_speed_tests.py

.. note::

   On OSX, if trouble with python with clang: we have found that the above may fail with an error about ``-lstdc++``, in which case you should try setting an environment variable like::

     export MACOSX_DEPLOYMENT_TARGET=10.14

   where you should replace 10.14 by your OSX number.

.. note::

   As of v2.0.1, our python interface is quite different from Dan Foreman-Mackey's original repo that wrapped finufft: `python-finufft <https://github.com/dfm/python-finufft>`_, or Jeremy Magland's wrapper. The interface is simpler, and the existing shared binary is linked to (no recompilation). Under the hood we achieve this via ``ctypes`` instead of ``pybind11``.


A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it from a shell in the FINUFFT top directory (after installing ``python-virtualenv``)::

  virtualenv -p /usr/bin/python3 env1
  source env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the ``env1`` directory. (You can get out of the environment by typing ``deactivate``). Also see documentation for ``conda``. In both cases ``python`` will call the version of python you set up. To get the packages FINUFFT needs::

  pip install -r python/requirements.txt

Then ``pip install finufft`` or build as above.
