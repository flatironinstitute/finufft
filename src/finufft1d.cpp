#include "finufft1d.h"

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,int ms,
	       double* fk)
{
 /*  Type-1 1D NUFFT.

     if (iflag>0) then

               1  nj
     fk(k1) = -- SUM cj(j) exp(+i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=1                            

     else
               1  nj
     fk(k1) = -- SUM cj(j) exp(-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=1                            

   Inputs:
     nj     number of sources
     xj     location of sources on interval [-pi,pi].
     cj     strengths of sources (complex *16)
     iflag  determines sign of FFT (see above)
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
   Outputs:
     fk     Fourier transform values (size ms, mode numbers as above).
            Complex stored as alternating real and imag double reals.
     returned value - error return code: as returned by cnufftspread, but with
            following extra cases:
	    5 - ...

     The type 1 NUFFT proceeds in three steps (see [GL]).

     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.

   Written in C style.
 */
  
  double params[4];
  spread_opts opts;
  int ier_set = get_kernel_params_for_eps(params,eps);
  
  BIGINT nf1 = 2*ms;  // adjust
  double *fw = (double *)malloc(sizeof(double)*2*nf1);  // since complex
  double *fwker = (double *)malloc(sizeof(double)*2*nf1);  // since complex
  int dir = 1;
  int ier_spread = twopispread1d(nf1,fw,nj,xj,cj,dir,params);

  // call 1D FFT

  // ..

  free(fw);
  free(fwker);
  return ier_spread;
}
