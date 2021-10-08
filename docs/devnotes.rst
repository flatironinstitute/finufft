.. _devnotes:

Developer notes
===============

* Developers needing to update documentation in general, including our readthedocs website, see ``docs/README``.

* To update the version number, this needs to be done by hand in the following places:

  - ``docs/conf.py`` for sphinx
  - ``python/setup.py`` for the python pkg version
  - ``include/defs.h``
  - ``CHANGELOG``: don't forget to describe the new features and changes, folding lines at 80 chars.

  You will then need to update the docs as in ``docs/README``.

* There are some sphinx tags in the source code, indicated by @ in comments. Please leave these alone since they are needed by the doc generation.

* If you add a new option field (recall it must be plain C style only, no special types) to ``include/nufft_opts.h``, don't forget to add it to ``include/finufft.fh``, ``matlab/finufft.mw``, ``python/finufft/_finufft.py``, and the julia interface, as well a paragraph describing its use in the docs. Also to set its default value in ``src/finufft.cpp``. You will then need to update the docs as in ``docs/README``.

* Developers changing MATLAB/octave interfaces or docs, also see ``matlab/README``.

* For testing and performance measuring routines see ``test/README`` and ``perftest/README``. We need more of the latter, eg, something making performance graphs that enable rapid eyeball comparison of various settings/machines.

* Continuous Integration (CI). See files for this in ``.github/workflows/``. It currently tests the default ``makefile`` settings in linux, and three other ``make.inc.*`` files covering OSX and Windows (MinGW).

* **Installing MWrap**. This is needed only to rebuild the matlab/octave interfaces.
  `MWrap <https://github.com/zgimbutas/mwrap>`_
  is a very useful MEX interface generator by Dave Bindel, now maintained
  and expanded by Zydrunas Gimbutas.
  Make sure you have ``flex`` and ``bison`` installed to build it.
  As of FINUFFT v.2.0 you will need a recent (>=0.33.10) version of MWrap.
  Make sure to override the location of MWrap by adding a line such as::

    MWRAP = your-path-to-mwrap-executable
  
  to your ``make.inc``, and then you can use the ``make mex`` task.


