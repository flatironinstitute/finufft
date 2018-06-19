Installation
============

Obtaining FINUFFT
*****************

Go to the github page https://github.com/ahbarnett/finufft and
follow instructions (eg see the green button).


Dependencies
************

This library is currently supported for unix/linux
and also tested on Mac OSX. We have heard that it can be compiled
on Windows too.

For the basic libraries

* C++ compiler, such as ``g++`` packaged with GCC
* FFTW3
* GNU make

Optional:

* ``numdiff`` (preferred but not essential; enables pass-fail math validation)
* for Fortran wrappers: compiler such as ``gfortran``
* for matlab/octave wrappers: MATLAB, or octave and its development libraries
* for building new matlab/octave wrappers (experts only): ``mwrap``
* for the python wrappers you will need ``python`` and ``pip`` (if you prefer python v2), or ``python3`` and ``pip3`` (for python v3). You will also need ``pybind11``


Tips for installing dependencies on various operating systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a Fedora/CentOS linux system, these dependencies can be installed as follows::

  sudo yum install make gcc gcc-c++ gcc-gfortran fftw3 fftw3-devel libgomp octave octave-devel

then see below for ``numdiff`` and ``mwrap``.

.. note::

   we are not exactly sure how to install python3 and pip3 using yum

then download the latest ``numdiff`` from http://gnu.mirrors.pair.com/savannah/savannah/numdiff/ and set it up via ``./configure; make; sudo make install``

On Ubuntu linux (assuming python3 as opposed to python)::

  sudo apt-get install make build-essential libfftw3-dev gfortran numdiff python3 python3-pip octave liboctave-dev

On Mac OSX:

Make sure you have ``make`` installed, eg via XCode.

Install gcc, for instance using pre-compiled binaries from
http://hpc.sourceforge.net/

Install homebrew from http://brew.sh::

  brew install fftw

Install ``numdiff`` as below.

(Note: we are not exactly sure how to install python3 and pip3 on mac)

Currently in Mac OSX, ``make lib`` fails to make the shared object library (.so);
however the static (.a) library is of reasonable size and works fine.


Installing numdiff
------------------

`numdiff <http://www.nongnu.org/numdiff>`_ by Ivano Primi extends ``diff`` to assess errors in floating-point outputs.
Download the latest ``numdiff`` from the above URL, un-tar the package, cd into it, then build via ``./configure; make; sudo make install``

Installing MWrap
----------------

This is not needed for most users.
`MWrap <http://www.cs.cornell.edu/~bindel/sw/mwrap>`_
is a very useful MEX interface generator by Dave Bindel.
Make sure you have ``flex`` and ``bison`` installed.
Download version 0.33 or later from http://www.cs.cornell.edu/~bindel/sw/mwrap, un-tar the package, cd into it, then::
  
  make
  sudo cp mwrap /usr/local/bin/

Compilation
***********

We first describe compilation for default options (double precision, openmp) via GCC.
If you have a nonstandard unix environment (eg a Mac) or want to change the compiler,
then place your compiler and linking options in a new file ``make.inc``.
For example such files see ``make.inc.*``. See ``makefile`` for what can be overridden.

Compile and do a rapid (less than 1-second) test of FINUFFT via::

  make test

This should compile the main libraries then run tests which should report zero crashes and zero fails. (If numdiff was not installed, it instead produces output that you will have to check by eye matches the requested accuracy.)

Use ``make perftest`` for larger spreader and NUFFT tests taking 15-30 seconds.

Run ``make`` without arguments for full list of possible make tasks.

Note that the library includes the C and fortran interfaces
defined in ``src/finufft_c.h`` and ``fortran/finufft_f.h`` respectively.
If there is an error in testing on a standard set-up,
please file a bug report as a New Issue at https://github.com/ahbarnett/finufft/issues

Custom library compilation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to make the library for other data types. Currently
library names are distinct for single precision (libfinufftf) vs
double (libfinufft). However, single-threaded vs multithreaded are
built with the same name, so you will have to move them to other
locations, or build a 2nd copy of the repo, if you want to keep both
versions.

You *must* do at least ``make objclean`` before changing precision or openmp options.

**Single precision**: append ``PREC=SINGLE`` to the make task.
Single-precision saves half the RAM, and increases
speed slightly (<20%). The  C++, C, and fortran demos are all tested in
single precision. However, it will break matlab, octave, python interfaces.

**Single-threaded**: append ``OMP=OFF`` to the make task.


Building examples and wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make examples`` to compile and run the examples for calling from C++ and from C.

The ``examples`` and ``test`` directories are good places to see usage examples.

``make fortran`` to compile and run the fortran wrappers and examples.

``make matlab`` to build the MEX interface to matlab.

``make octave`` to build the MEX-like interface to octave.

On Mac OSX, we have found that the MATLAB MEX settings need to be
overridden: edit the file ``mex_C++_maci64.xml`` in the MATLAB distro,
to read, for instance::

  CC="gcc-8"
  CXX="g++-8"
  CFLAGS="-ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -pthread"
  CXXFLAGS="-ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread"

These settings are copied from the ``glnxa64`` case. Here you will want to replace the compilers by whatever version of GCC you have installed.
For pre-2016 MATLAB Mac OSX versions you'll instead want to edit the ``maci64``
section of ``mexopts.sh``.


Building the python wrappers
****************************

First make sure you have python3 and pip3 (or python and pip) installed and that you have already compiled the C++ library (eg via ``make lib``).
Python links to this compiled library.
Next make sure you have NumPy and pybind11 installed::
  
  pip3 install numpy pybind11

You may then do ``make python3`` which calls
pip3 for the install then runs some tests. An additional test you could do is::

  python3 run_speed_tests.py

In all the above the "3" can be omitted if you want to work with python v2.

See also Dan Foreman-Mackey's earlier repo that also wraps finufft, and from which we have drawn code: `python-finufft <https://github.com/dfm/python-finufft>`_


A few words about python environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There can be confusion and conflicts between various versions of python and installed packages. It is therefore a very good idea to use virtual environments. Here's a simple way to do it (after installing python-virtualenv)::

  Open a terminal
  virtualenv -p /usr/bin/python3 env1
  . env1/bin/activate

Now you are in a virtual environment that starts from scratch. All pip installed packages will go inside the env1 directory. (You can get out of the environment by typing ``deactivate``)
