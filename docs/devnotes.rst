.. _devnotes:

Developer notes
===============

* To update the version number, this needs to be done by hand in the following places:
  - ``docs/conf.py`` for sphinx
  - ``python/setup.py`` for the python pkg version
  - ``include/defs.h``
  - ``CHANGELOG``: don't forget to describe the new features and changes.

* There are some sphinx tags in the source code, indicated by @ in comments. Please leave these alone since they are needed by the doc generation.

* Developers changing MATLAB/octave interfaces or docs, see ``matlab/README``

* Developers changing overall web docs, see ``docs/README``
      
* For testing and performance measuring routines see ``test/REAME`` and ``perftest/README``


