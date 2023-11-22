.. _devnotes:

Developer notes
===============

* Developers needing to update/regenerate documentation in general, including our readthedocs website, see ``docs/README``. Developers changing MATLAB/octave interfaces or docs, also see ``matlab/README``.

* To update the version number, this needs to be done by hand in the following places:

  - ``CMakeLists.txt`` for cmake
  - ``docs/conf.py`` for sphinx
  - ``python/finufft/setup.py`` for the python pkg version
  - ``python/finufft/__init__.py`` ditto
  - ``python/cufinufft/setup.py`` for the GPU python pkg version
  - ``python/cufinufft/__init__.py`` ditto
  - ``include/finufft/defs.h`` for the debug>0 output
  - ``matlab/Contents.m`` for the MATLAB/Octave help
  - ``CHANGELOG``: don't forget to describe the new features and changes, folding lines at 80 chars.

* There are some sphinx tags in the source code, indicated by @ in comments. Please leave these alone since they are needed by the doc generation.

* If you add a new option field (recall it must be plain C style only, no special types) to ``include/finufft_opts.h``, don't forget to add it to ``include/finufft.fh``, ``include/finufft_mod.f90``, ``matlab/finufft.mw``, ``python/finufft/_finufft.py``, and the Julia interface, as well a paragraph describing its use in the docs. Also to set its default value in ``src/finufft.cpp``. You will then need to regenerate the docs as in ``docs/README``.

* For testing and performance measuring routines see ``test/README`` and ``perftest/README``. We need more of the latter, eg, something making performance graphs that enable rapid eyeball comparison of various settings/machines.

* Continuous Integration (CI). See files for this in ``.github/workflows/``. It currently tests the default ``makefile`` settings in linux, and three other ``make.inc.*`` files covering OSX and Windows (MinGW). CI does not test build the variant OMP=OFF. The dev should test these locally. Likewise, the Julia wrapper is separate and thus not tested in CI. We have added ``JenkinsFile`` for the GPU CI via python wrappers.

* **Installing MWrap**. This is needed only for experts to rebuild the matlab/octave interfaces.
  `MWrap <https://github.com/zgimbutas/mwrap>`_
  is a very useful MEX interface generator by Dave Bindel, now maintained
  and expanded by Zydrunas Gimbutas.
  Make sure you have ``flex`` and ``bison`` installed to build it.
  As of FINUFFT v2.0 you will need a recent (>=0.33.10) version of MWrap.
  Make sure to override the location of MWrap by adding a line such as::

    MWRAP = your-path-to-mwrap-executable
  
  to your ``make.inc``, and then you can use the ``make mex`` task.

* The cufinufft Python wheels are generated using Docker based on the manylinux2014 image. For instructions, see ``tools/cufinufft/distribution_helper.sh``. These are binary wheels that are built using CUDA 11 (or optionally CUDA 12, but these are not distributed on PyPI) and bundled with the necessary libraries.


