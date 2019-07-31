#include <finufft_legacy.h>
#include <invokeGuru.h>
#include <common.h>
#include <utils.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>



int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
 /*  Type-1 2D complex nonuniform FFT.

                  nj-1
     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                  j=0

     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is k1 (fast), then k2 (slow), with each dimension
     determined by opts.modeord.
     If iflag>0 the + sign is used, otherwise the - sign is used,
     in the exponential.

   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj     x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  number of Fourier modes requested in x and y (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values
            (size ms*mt, fast in ms then slow in mt,
            ie Fortran ordering).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps:
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.

 */
{


  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = 1;
  int n_dims = 2;
  int n_transf = 1;
  finufft_type type = type1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL,fk, opts);

  
  return ier; 
  
}


int finufft2d1many(int n_transf, BIGINT nj, FLT* xj, FLT *yj, CPX* c,
		   int iflag, FLT eps, BIGINT ms, BIGINT mt, CPX* fk,
		   nufft_opts opts)
/*
  Type-1 2D complex nonuniform FFT for multiple strength vectors, same NU pts.

                    nj
    f[k1,k2,d] =   SUM  c[j,d] exp(+-i (k1 x[j] + k2 y[j]))
                   j=1

    for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, d = 0,...,n_transf-1

    The output array is in increasing k1 ordering (fast), then increasing
    k2 ordering (slow), then increasing d (slowest). If iflag>0 the + sign
    is used, otherwise the - sign is used, in the exponential.
  Inputs:
    n_transf  number of strength vectors
    nj     number of sources (int64, aka BIGINT)
    xj,yj  x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
    c      a size nj*n_transf complex FLT array of source strengths,
           increasing fast in nj then slow in n_transf.
    iflag  if >=0, uses + sign in exponential, otherwise - sign.
    eps    precision requested (>1e-16)
    ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
    opts   struct controlling options (see finufft.h)
  Outputs:
    fk     complex FLT array of Fourier transform values
           (size ms*mt*n_transf, increasing fast in ms then slow in mt then in n_transf
           ie Fortran ordering).
    returned value - 0 if success, else see ../docs/usage.rst

  Note: nthreads times the RAM is needed, so this is good only for small problems.

 */
{

  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = 1;
  int n_dims = 2;
  finufft_type type = type1;
  
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj,NULL, c, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);


  return ier; 
}


int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

 /*  Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

   Inputs:
     nj     number of targets (int64, aka BIGINT)
     xj,yj     x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier transform values (size ms*mt,
            increasing fast in ms then slow in mt, ie Fortran ordering).
            Along each dimension the ordering is set by opts.modeord.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  numbers of Fourier modes given in x and y (int64)
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of target values
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps:
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

 */
{
  
  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = 0;
  int n_dims = 2;
  int n_transf = 1;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);
 
  
  return ier;
}


int finufft2d2many(int n_transf, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
		   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
/*
  Type-2 2D complex nonuniform FFT for multiple coeff vectors, same NU pts.

	     cj[j,d] =  SUM   fk[k1,k2,d] exp(+/-i (k1 xj[j] + k2 yj[j]))
	               k1,k2
	     for j = 0,...,nj-1,  d = 0,...,n_transf-1
	     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

  Inputs:
    n_transf  number of mode coefficient vectors
    nj     number of targets (int64, aka BIGINT)
    xj,yj  x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
    fk     FLT complex array of Fourier transform values (size ms*mt*n_transf,
           increasing fast in ms then slow in mt then in n_transf, ie Fortran
           ordering). Along each dimension the ordering is set by opts.modeord.
    iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
    eps    precision requested (>1e-16)
    ms,mt  numbers of Fourier modes given in x and y (int64)
           each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
    opts   struct controlling options (see finufft.h)
  Outputs:
    cj     size-nj*n_transf complex FLT array of target values, (ie, stored as
           2*nj*n_transf FLTs interleaving Re, Im), increasing fast in nj then
           slow in n_transf.
    returned value - 0 if success, else see ../docs/usage.rst

  Note: nthreads times the RAM is needed, so this is good only for small problems.

*/
{

  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = 1;
  int n_dims = 2;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, c, iflag,
				eps, n_modes, 0, NULL, NULL, NULL, fk, opts);

  return ier; 
}


int finufft2d3(BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts opts)
 /*  Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj  x,y location of sources in the plane R^2 (each size-nj FLT array)
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points (int64)
     s,t    (k_x,k_y) frequency locations of targets in R^2.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex FLT Fourier transform values at the
            target frequencies sk
     returned value - 0 if success, else see ../docs/usage.rst

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1.
     Beyond this, the new twists are:
     i) number of upsampled points for the type-1 in each dim, depends on the
       product of interval widths containing input and output points (X*S), for
       that dim.
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf, in each dim.

   No references to FFTW are needed here. Some CPX arithmetic is used.
 */
{

  int n_dims = 2;
  finufft_type type = type3;
  int n_transf = 1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj,iflag, eps, NULL, nk, s,t,NULL, fk, opts);
  return ier;
  
}


int finufft2d3many(int n_transf, BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts opts){

  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  int n_dims = 2;
  finufft_type type = type3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj,iflag, eps, NULL, nk, s,t,NULL, fk, opts);
  return ier;
}
