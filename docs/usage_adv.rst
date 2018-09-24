.. _advinterface:

Advanced interfaces for many vectors with same nonuniform points
================================================================

It is common to need repeated NUFFTs with a fixed set of
nonuniform points, but different strength or mode coefficient vectors.
For large problems, performing sequential plain calls is efficient
(although there would be a slight benefit to sorting only once),
but when the problem size is smaller, certain start-up costs cause
repeated calls to the plain interface to be slower than necessary.
In particular, we note that FFTW takes around 0.1 ms per thread to
look up stored wisdom, which for small problems (of order 10000
or less input and output data) can, sadly, dominate the runtime.
Thus we include interfaces, described here, for multiple stacked strength
or coefficient vectors with the same nonuniform points.

These have only been implemented for the 2d1 and 2d2 types so far,
for which there are applications in cryo-EM.

For data types in the below, please see :ref:`data types <datatypes>`.


2D transforms
~~~~~~~~~~~~~

::

  int finufft2d1many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                     FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

  Type-1 2D complex nonuniform FFT for multiple strength vectors, same NU pts.

                    nj
    f[k1,k2,d] =   SUM  c[j,d] exp(+-i (k1 x[j] + k2 y[j]))
                   j=1

    for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, d = 0,...,ndata-1

    The output array is in increasing k1 ordering (fast), then increasing
    k2 ordering (slow), then increasing d (slowest). If iflag>0 the + sign
    is used, otherwise the - sign is used, in the exponential.
  Inputs:
    ndata  number of data
    nj     number of sources (int64, aka BIGINT)
    xj,yj  x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
    c      a size nj*ndata complex FLT array of source strengths,
           increasing fast in nj then slow in ndata.
    iflag  if >=0, uses + sign in exponential, otherwise - sign.
    eps    precision requested (>1e-16)
    ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
     opts  struct controlling options (see finufft.h)
  Outputs:
    fk     complex FLT array of Fourier transform values
           (size ms*mt*ndata, increasing fast in ms then slow in mt then in ndata
           ie Fortran ordering).
    returned value - 0 if success, else see ../docs/usage.rst

    Note: nthreads times the RAM is needed, so this is good only for small problems.


  
  int finufft2d2many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                     FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

  Type-2 2D complex nonuniform FFT for multiple coeff vectors, same NU pts.

    cj[j,d] =  SUM   fk[k1,k2,d] exp(+/-i (k1 xj[j] + k2 yj[j]))
             k1,k2
    for j = 0,...,nj-1,  d = 0,...,ndata-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2

  Inputs:
    ndata  number of mode coefficient vectors
    nj     number of targets (int64, aka BIGINT)
    xj,yj  x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
    fk     FLT complex array of Fourier transform values (size ms*mt*ndata,
           increasing fast in ms then slow in mt then in ndata, ie Fortran
           ordering). Along each dimension the ordering is set by opts.modeord.
    iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
    eps    precision requested (>1e-16)
    ms,mt  numbers of Fourier modes given in x and y
           each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
    opts   struct controlling options (see finufft.h)
  Outputs:
    cj     size-nj*ndata complex FLT array of target values, (ie, stored as
           2*nj*ndata FLTs interleaving Re, Im), increasing fast in nj then
           slow in ndata.
    returned value - 0 if success, else see ../docs/usage.rst

    Note: nthreads times the RAM is needed, so this is good only for small problems.

Design notes
~~~~~~~~~~~~

After extensive timing tests, we settled on blocking up
the ndata vectors into blocks of size nthreads (the available thread number).
Each block is handled together via FFTW and OpenMP parallelism.
For instance, for type-1:

#. Each thread calls a single-threaded spreader, reusing a precomputed sorted index list.
#. Apply FFT on nthreads vectors of data using FFTW's "many dft" interface.
#. Each thread calls a single-threaded deconvolve function.

This requires ndata times the RAM overhead than the plain interface.

It would also be possible to call multi-threaded spreading, sequentially
on each data vector; we found this slower in all cases, and so close to
repeated calls to the plain interface as to not be useful.

For repeated small problems where the nonuniform points and strengths
or coefficients change, but the mode grid is fixed, reusing the FFTW
plan may still be beneficial; this would require a three-call "plan,
execute, destroy" interface which we have not considered worth
building yet.
