The "many" interface
====================

2D transforms
~~~~~~~~~~~~~

::

  int finufft2d1many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                     FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

  Type-1 2D complex nonuniform FFT for multiple data.

                    nj
    f[k1,k2,d] =   SUM  c[j,d] exp(+-i (k1 x[j] + k2 y[j]))
                   j=1

    for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, d = 0, ..., ndata-1

    The output array is in increasing k1 ordering (fast), then increasing
    k2 ordering (slow), then increasing d (slowest). If iflag>0 the + sign
    is used, otherwise the - sign is used, in the exponential.
  Inputs:
    ndata  number of data
    nj     number of sources
    xj,yj  x,y locations of sources on 2D domain [-pi,pi]^2.
    c      a size nj*ndata complex FLT array of source strengths,
           increasing fast in nj then slow in ndata.
    iflag  if >=0, uses + sign in exponential, otherwise - sign.
    eps    precision requested (>1e-16)
    ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2]
    opts   struct controlling options (see finufft.h)
  Outputs:
    fk     complex FLT array of Fourier transform values
           (size ms*mt*ndata, increasing fast in ms then slow in mt then in ndata
           ie Fortran ordering).
    returned value - 0 if success, else see ../docs/usage.rst

  The type 1 NUFFT proceeds in three main steps (see [GL]):
  1) spread data to oversampled regular mesh using kernel.
  2) compute FFT on uniform mesh
  3) deconvolve by division of each Fourier mode independently by the
     Fourier series coefficient of the kernel.
  The kernel coeffs are precomputed in what is called step 0 in the code.

::

  int finufft2d2many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                     FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

  Type-2 2D complex nonuniform FFT for multiple data.

    cj[j,d] =  SUM   fk[k1,k2,d] exp(+/-i (k1 xj[j] + k2 yj[j]))
             k1,k2
    for j = 0,...,nj-1, d = 0,...,ndata-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2

  Inputs:
    ndata  number of data
    nj     number of sources
    xj,yj  x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
    fk     FLT complex array of Fourier transform values (size ms*mt*ndata,
           increasing fast in ms then slow in mt then in ndata, ie Fortran
           ordering). Along each dimension the ordering is set by opts.modeord.
    iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
    eps    precision requested (>1e-16)
    ms,mt  numbers of Fourier modes given in x and y
           each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
    opts   struct controlling options (see finufft.h)
  Outputs:
    cj     size-nj*ndata complex FLT array of target values, (ie, stored as
           2*nj*ndata FLTs interleaving Re, Im), increasing fast in nj then
           slow in ndata.
    returned value - 0 if success, else see ../docs/usage.rst

  The type 2 algorithm proceeds in three main steps (see [GL]).
  1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
  2) compute inverse FFT on uniform fine grid
  3) spread (dir=2, ie interpolate) data to regular mesh
  The kernel coeffs are precomputed in what is called step 0 in the code.

Design notes
~~~~~~~~~~~~
The ``many_seq`` option controls the algorithm to use when we run the code with multiple
threads. When running with 1 thread, the only difference between the two
algorithms comes from the fftw library: if ``many_seq=1``, we use the basic interface,
``fftw_plan_dft_2d``; if ``many_seq=0``, we use an advanced one, ``fftw_plan_many_dft``,
but asking it to compute FFT for one data each time.

When running with nth threads, if ``many_seq=0``, nth of data are processed
simultaneously. Taking 2D type 1 transform as an example, the algorithm proceeds
in following three steps:
  1) Each thread calls a single-threaded spreader.
  2) Apply FFT on nths of data using the many interface providing by fftw.
  3) Each thread calls a single-threaded deconvolve function.

If ``many_seq=1``, then all the data are processed sequentially. We simply add a
big for loop looping through all the data around the original code and move the
memory pointer correspondingly for each iteration.

For both algorithms, we reuse the plan and sorted index of the source points
for all the data, i.e. the plan and the sorting function are only called
once at the very beginning of the many interface.
