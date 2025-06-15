Directories in this package
===========================

When you ``git clone https://github.com/flatironinstitute/finufft``, or unpack
a tar ball, you will get the following. (Please see :ref:`installation <install>` instructions)

Main library source:

- ``makefile`` : the single GNU makefile (there are no makefiles in subdirectories)
- ``make-platforms/`` : OS/platform specific setting files to use as your ``make.inc``
- ``CMakeLists.txt`` : top-level CMake file
- ``cmake/`` : CMake specific helper files
- ``src/`` : main library C++ CPU sources
- ``src/cuda/`` : main library CUDA GPU sources
- ``include/`` : public library API header files
- ``include/{cu}finufft`` : private header files
- ``lib/`` : dynamic (``.so``) library will be built here by GNU make
- ``lib-static/`` : static (``.a``) library will be built here by GNU make

Examples, tutorials, and docs:

- ``examples/`` : simple example codes for calling the library from C++ and C
- ``tutorial/`` : application demo codes (various languages), supporting ``docs/tutorial/``
- ``docs/`` : source files for documentation (``.rst`` files are human-readable, kinda)
- ``README.md`` : github-facing (and human text-only reader) welcome message
- ``LICENSE`` : how you may use this software
- ``CHANGELOG`` : list of changes, release notes
- ``devel/`` : scratch space for development, ideas docs, code snippets

Testing:

- ``test/`` : main validation tests (C++/bash), including:

  - ``test/basicpassfail{f}`` simple smoke test with exit code
  - ``test/check_finufft.sh`` is the main pass-fail validation bash script
  - ``test/results/`` : some rather old output text files
  - ``test/cuda/`` : GPU tests

- ``perftest/`` : main performance and developer tests (C++/bash), including:

  - ``perftest/spreadtestnd.sh``, etc : Please see ``perftest/README``
  - ``perftest/cuda/`` : GPU performance tests

- ``.github/workflows/`` and ``Jenkinsfile`` : for continuous integration (CI)

Language interfaces and packaging:

- ``fortran/`` : wrappers and example drivers for Fortran (see ``fortran/README``)
- ``matlab/`` : MATLAB/octave wrappers (CPU), tests, and examples
- ``python/`` : python wrappers (CPU and GPU), examples, and tests
- ``tools/`` : tools for building python wheels, docker
