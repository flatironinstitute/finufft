.. _cufinufft_migration:

Migration to cuFINUFFT v2.2
===========================

Between versions 1.3 and 2.2 of cuFINUFFT, the API was updated to align more closely with that of FINUFFT.
This was done for both the C and C++ interface as well as for the Python interface.
A summary of the API changes will be found below.

C and C++ interface
-------------------

The following function signatures were updated during the API change:

- ``cufinufft_makeplan``, which previously had the signature

    .. code-block:: c

      int cufinufft_makeplan(int type, int dim, int *n_modes, int iflag,
          int ntransf, double tol, int maxbatchsize, cufinufft_plan *plan,
          cufinufft_opts *opts);

  and now has the signature

    .. code-block:: c

      int cufinufft_makeplan(int type, int dim, const int64_t *n_modes,
          int iflag, int ntr, double eps, cufinufft_plan *d_plan_ptr,
          cufinufft_opts *opts);


  In other words, the ``n_modes`` argument now takes the type ``int64_t`` to accomodate larger arrays and the ``maxbatchsize`` argument has been removed (and can now be found as part of ``cufinufft_opts``).
  The ``tol`` and ``ntransf`` arguments have also been renamed to ``eps`` and ``ntr``, respectively.

- ``cufinufft_setpts``, which had the signature

    .. code-block:: c

      int cufinufft_setpts(int M, double* h_kx, double* h_ky, double* h_kz,
          int N, double *h_s, double *h_t, double *h_u, cufinufft_plan d_plan);


  and now has the signature

    .. code-block:: c

      int cufinufft_setpts(cufinufft_plan d_plan, int M, double *d_x,
          double *d_y, double *d_z, int N, double *d_s, double *d_t, double *d_u);


  Aside from name changes, main difference here is that the ``plan`` is now the first argument, not the last.

- ``cufinufft_execute``, which had the signature

    .. code-block:: c

      int cufinufft_execute(cuDoubleComplex* h_c, cuDoubleComplex* h_fk, cufinufft_plan d_plan);

  and now has the signature

    .. code-block:: c

      int cufinufft_execute(cufinufft_plan d_plan, cuDoubleComplex *d_c, cuDoubleComplex *d_fk);

  Again, the names have changed slightly and the plan argument is moved to be the first argument.

- ``cufinufft_destroy`` has not changed its signature.

- ``cufinufft_default_opts`` used to have the signature

    .. code-block:: c

      int cufinufft_default_opts(int type, int dim, cufinufft_opts *opts);

  but now has the signature

    .. code-block:: c

      void cufinufft_default_opts(cufinufft_opts *opts);

  Consequently, you no longer need to specify the type and dimension when filling out the default options structure.

Note that the above function signature are given for the double-precision API.
For single precision, replace the ``cufinufft_`` with ``cufinufftf_`` and occcurences of ``double`` with ``float``.

Python interface
----------------

One big difference in the Python interface is that ``cufinufft.cufinufft`` has been renamed ``cufinufft.Plan``.
Its methods have the following updates

- The constructor ``Plan.__init__`` now defaults to ``dtype="complex64"`` instead of ``dtype="float32"``.
  The effect is the same (single-precision computations) but now makes explicit that we are dealing with complex (not real) transforms.

- The ``set_pts`` method is now called ``setpts``.

- The ``execute`` method now takes a ``data`` argument and returns its output instead of using ``c`` and ``fk`` as input/output arguments. An optional ``out`` argument is also used to specify an output array.

The new API also includes simple interfaces ``cufinufft.nufft*d*`` in the style of ``finufft.nufft*d*``.
