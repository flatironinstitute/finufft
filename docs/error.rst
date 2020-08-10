.. _error:

Error (status) codes
====================

In all FINUFFT interfaces, the returned value ``ier`` is a status indicator.
It is ``0`` if successful, otherwise the error code
has the following meanings (see ``include/defs.h``):

::

  1  requested tolerance epsilon too small to achieve (warning only)
  2  attemped to allocate internal array larger than MAX_NF (defined in defs.h)
  3  spreader: fine grid too small compared to spread (kernel) width
  4  spreader: if chkbnds=1, a nonuniform point coordinate is out of input range [-3pi,3pi]^d
  5  spreader: array allocation error
  6  spreader: illegal direction (should be 1 or 2)
  7  upsampfac too small (should be >1.0)
  8  upsampfac not a value with known Horner poly eval rule (currently 2.0 or 1.25 only)
  9  ntrans not valid in "many" (vectorized) or guru interface (should be >= 1)
  10 transform type invalid
  11 general allocation failure
  12 dimension invalid
  13 spread_thread option invalid
  
When ``ier=1`` (warning only) the transform(s) is/are still completed, at the smallest epsilon achievable, so, with that caveat, the answer should still be usable.

For any other nonzero values of ``ier`` the transform may not have been performed and the output should not be trusted. However, we hope that the value of ``ier`` will help to narrow down the problem.

FINUFFT sometimes also sends error text to ``stderr`` if it detects faulty input parameters.

If you are getting error codes, please reread the documentation
for your language, then see our :ref:`troubleshooting advice <trouble>`.


Large internal arrays
-----------------------

In case your input parameters demand the allocation of very large arrays, an
internal check is done to see if their size exceeds a rather generous internal
limit, set in ``defs.h`` as ``MAX_NF``. The current value in the source code is
``1e11``, which corresponds to about 1TB for double precision.
Allocations beyond this cause a graceful exit with error code ``2`` as above.
Such a large allocation can be due to enormous ``N`` (in types 1 or 2), or ``M``,
but also large values of the space-bandwidth product (loosely, range of :math:`\mathbf{x}_j` points times range of :math:`\mathbf{k}_j` points) for type 3 transforms; see Remark 5 in :ref:`reference FIN <refs>`.
Note that mallocs smaller than this, but which still exceed available RAM, may cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc or STL vector creation in the code, and neither is this recommended in modern style guides.
If you have a large-RAM machine and want to exceed the above hard-coded limit, you will need
to edit ``defs.h`` and recompile.

