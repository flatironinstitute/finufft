.. _error:

Error (status) codes
====================

In all FINUFFT interfaces, the returned value ``ier`` is a status indicator.
It is ``0`` if successful, otherwise the error code
has the following meanings (see codes in ``include/finufft_errors.h``):

::

  1  requested tolerance epsilon too small to achieve (warning only)
  2  stopped due to needing internal array size >MAX_NF (defined in defs.h)
  3  spreader: fine grid too small compared to spread (kernel) width
  4  spreader: [DEPRECATED]
  5  spreader: array allocation error
  6  spreader: illegal direction (should be 1 or 2)
  7  upsampfac too small (should be >1.0)
  8  upsampfac not a value with known Horner poly eval rule (currently 2.0 or 1.25 only)
  9  ntrans not valid in "many" (vectorized) or guru interface (should be >= 1)
  10 transform type invalid
  11 general internal allocation failure
  12 dimension invalid
  13 spread_thread option invalid
  14 invalid mode array (more than ~2^31 modes, dimension with 0 modes, etc)
  15 CUDA failure (failure to call any cuda function/kernel, malloc/memset, etc))
  16 attempt to destroy an uninitialized plan
  17 invalid spread/interp method for dim (attempt to blockgather in 1D, e.g.)
  18 size of bins for subprob/blockgather invalid
  19 GPU shmem too small for subprob/blockgather parameters
  20 invalid number of nonuniform points: nj or nk negative, or too big (see defs.h)

When ``ier=1`` (warning only) the transform(s) is/are still completed, at the smallest epsilon achievable, so, with that caveat, the answer should still be usable.

For any other nonzero values of ``ier`` the transform may not have been performed and the output should not be trusted. However, we hope that the value of ``ier`` will help to narrow down the problem.

FINUFFT sometimes also sends error text to ``stderr`` if it detects faulty input parameters. Please check your terminal output.

If you are getting error codes, please reread the documentation
for your language, then see our :ref:`troubleshooting advice <trouble>`.


Large internal arrays
-----------------------

In case your input parameters demand the allocation of very large arrays, an
internal check is done to see if their size exceeds a rather generous internal
limit, set in ``defs.h`` as ``MAX_NF``. The current value in the source code is
``1e12``, which corresponds to about 10TB for double precision.
Allocations beyond this cause a graceful exit with error code ``2`` as above.
Such a large allocation can be due to enormous ``N`` (in types 1 or 2), or ``M``,
but also large values of the space-bandwidth product (loosely, range of :math:`\mathbf{x}_j` points times range of :math:`\mathbf{k}_j` points) for type 3 transforms; see Remark 5 in :ref:`reference FIN <refs>`.
Note that mallocs smaller than this, but which still exceed available RAM, may cause segfaults as usual. For simplicity of code, we do not do error checking on every malloc or STL vector creation in the code, and neither is this recommended in modern style guides.
If you have a large-RAM machine and want to exceed the above hard-coded limit, you will need
to edit ``defs.h`` and recompile.

Similar sanity checks are done on the numbers of nonuniform points, and it is
(barely) conceivable that you could want to
increase ``MAX_NU_PTS`` beyond its current value
of ``1e14`` in ``defs.h``, and recompile.
