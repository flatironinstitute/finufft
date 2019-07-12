#include <finufft_legacy.h>
#include <invokeGuru.h>
#include <common.h>
#include <utils.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk,
	       nufft_opts opts)
 /*  Type-1 3D complex nonuniform FFT.

                     nj-1
     f[k1,k2,k3] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                     j=0

	for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
            -mu/2 <= k3 <= (mu-1)/2.

     The output array is as in opt.modeord in each dimension.
     k1 changes is fastest, k2 middle,
     and k3 slowest, ie Fortran ordering. If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj,zj   x,y,z locations of sources (each size-nj FLT array) in [-3pi,3pi]
     cj     size-nj complex FLT array of source strengths, 
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested
     ms,mt,mu  number of Fourier modes requested in x,y,z (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values (size ms*mt*mu,
            changing fast in ms to slowest in mu, ie Fortran ordering).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps:
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays.
 */
{
  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = mu;
  int n_dims = 3;
  int n_transf = 1;
  finufft_type type = type1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
		      eps, n_modes, NULL, NULL, NULL, fk, opts);


  return ier;
}


int finufft3d1many(int n_transf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,
	       FLT eps, BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk,
	       nufft_opts opts)
{

  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }
    
  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = mu;
  int n_dims = 3;
  finufft_type type = type1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
		      eps, n_modes, NULL, NULL, NULL, fk, opts);


  return ier;


}

int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
	       int iflag,FLT eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       CPX* fk, nufft_opts opts)

 /*  Type-2 3D complex nonuniform FFT.

     cj[j] =    SUM   fk[k1,k2,k3] exp(+/-i (k1 xj[j] + k2 yj[j] + k3 zj[j]))
             k1,k2,k3
      for j = 0,...,nj-1
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 
                       -mu/2 <= k3 <= (mu-1)/2

   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj,zj     x,y,z locations of targets (each size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier series values (size ms*mt*mu,
            increasing fastest in ms to slowest in mu, ie Fortran ordering).
            (ie, stored as alternating Re & Im parts, 2*ms*mt*mu FLTs)
	    Along each dimension, opts.modeord sets the ordering.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested
     ms,mt,mu  numbers of Fourier modes given in x,y,z (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of target values,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps:
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. 
 */
{
  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = mu;
  int n_dims = 3;
  int n_transf = 1;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
		      eps, n_modes,NULL, NULL, NULL, fk, opts);



  return ier;
}

int finufft3d2many(int n_transf, BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,
	       int iflag,FLT eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       CPX* fk, nufft_opts opts)
{

  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  
  BIGINT n_modes[3];
  n_modes[0] = ms;
  n_modes[1] = mt;
  n_modes[2] = mu;
  int n_dims = 3;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
		      eps, n_modes,NULL, NULL, NULL, fk, opts);

  return ier;
  
}




int finufft3d3(BIGINT nj,FLT* xj,FLT* yj,FLT *zj, CPX* cj,
	       int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t,
	       FLT *u, CPX* fk, nufft_opts opts)
 /*  Type-3 3D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j] + u[k] zj[j]),
               j=0
                          for k=0,...,nk-1
   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj,zj   x,y,z location of sources in R^3 (each size-nj FLT array)
     cj     size-nj complex FLT array of source strengths
            (ie, interleaving Re & Im parts)
     nk     number of frequency target points (int64)
     s,t,u      (k_x,k_y,k_z) frequency locations of targets in R^3.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (FLT)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex FLT array of Fourier transform values at the
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
   Barnett 2/17/17, 6/12/17
 */
{
  BIGINT n_modes[3];
  n_modes[0] = nk;
  n_modes[1] = 1;
  n_modes[2] = 1;
  int n_dims = 3;
  int n_transf = 1;
  finufft_type type = type3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes,s ,t ,u, fk, opts);



  return ier;
}


int finufft3d3many(int n_transf, BIGINT nj,FLT* xj,FLT* yj,FLT *zj, CPX* cj,
	       int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t,
	       FLT *u, CPX* fk, nufft_opts opts)
{


  if (n_transf<1) {
    fprintf(stderr,"n_transf should be at least 1 (n_transf=%d)\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  
  BIGINT n_modes[3];
  n_modes[0] = nk;
  n_modes[1] = 1;
  n_modes[2] = 1;
  int n_dims = 3;
  finufft_type type = type3;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, zj, cj, iflag,
				eps, n_modes,s ,t ,u, fk, opts);



  return ier;


  
}
