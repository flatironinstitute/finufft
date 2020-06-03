.. _error:

Error (status) codes
====================

In all FINUFFT interfaces, the returned value ``ier`` is a status indicator.
It is 0 if successful, otherwise the error code
has the following meanings (see ``include/defs.h``):

::

  1  requested tolerance epsilon too small to achieve
  2  attemped to allocate internal arrays larger than MAX_NF (defined in defs.h)
  3  spreader: fine grid too small compared to spread (kernel) width
  4  spreader: if chkbnds=1, a nonuniform point coordinate is out of input range [-3pi,3pi]^d
  5  spreader: array allocation error
  6  spreader: illegal direction (should be 1 or 2)
  7  upsampfac too small (should be >1)
  8  upsampfac not a value with known Horner poly eval rule (currently 2.0 or 1.25 only)
  9  ntrans not valid in "many" (vectorized) or guru interface (should be >= 1)
  10 transform type invalid
  11 general allocation failure
  12 dimension invalid

When ``ier=1`` the transform(s) is/are still completed, at the smallest epsilon achievable, so the answer should be usable. For any other nonzero values of ``ier`` the transform may not have been performed and the output should not be trusted. However, we hope that the value of ``ier`` will help narrow down the problem.

FINUFFT sometimes also sends error reports to ``stderr`` if it detects faulty input parameters.

If you are getting error codes, please reread the documentation
for your language, then see our :ref:`troubleshooting advice <trouble>`.

