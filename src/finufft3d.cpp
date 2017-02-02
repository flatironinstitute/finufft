#include "finufft.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft3d1(BIGINT nj,double* xj,double *yj,double *zj,double* cj,int iflag,
	       double eps, BIGINT ms, BIGINT mt, BIGINT mu, double* fk,
	       nufft_opts opts)
 /*  Type-1 3D complex nonuniform FFT.

                     1  nj-1
     f[k1,k2,k3] =  --  SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j]))
                    nj  j=0

	for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
            -mu/2 <= k3 <= (mu-1)/2.

     The output array is in increasing k orderings. k1 is fastest, k2 middle,
     and k3 slowest, ie Fortran ordering. If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj,zj   x,y,z locations of sources on 3D domain [-pi,pi]^3.
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt,mu  number of Fourier modes requested in x,y,z;
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex Fourier transform values (size ms*mt*mu, increasing fast
            in ms to slowest in mu, ie Fortran ordering),
            stored as alternating Re & Im parts (2*ms*mt*mu doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 2/2/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  BIGINT nf2 = set_nf(mt,opts,spopts);
  BIGINT nf3 = set_nf(mu,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("3d1: (ms,mt,mu)=(%ld,%ld,%ld) (nf1,nf2,nf3)=(%ld,%ld,%ld) nj=%d ...\n",ms,mt,mu,nf1,nf2,nf3,nj); 

  // STEP 0: get DCT of half of spread kernel in each dim, since real symm:
  CNTime timer; timer.start();
  double *fwkerhalf1 = fftw_alloc_real(nf1/2+1);
  double *fwkerhalf2 = fftw_alloc_real(nf2/2+1);
  double *fwkerhalf3 = fftw_alloc_real(nf3/2+1);
  double prefac_unused_dims;
  onedim_dct_kernel(nf1, fwkerhalf1, prefac_unused_dims, spopts);
  onedim_dct_kernel(nf2, fwkerhalf2, prefac_unused_dims, spopts);
  onedim_dct_kernel(nf3, fwkerhalf3, prefac_unused_dims, spopts); //prefacs same
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel dct (ns=%d):\t %.3g s\n", spopts.nspread,t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1*nf2*nf3);  // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_3d(nf3,nf2,nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  int ier_spread = twopispread3d(nf1,nf2,nf3,(double*)fw,nj,xj,yj,zj,cj,1,params,opts.spread_debug);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  double prefac = 1.0/nj;    // 1/nj norm
  deconvolveshuffle3d(1,prefac,fwkerhalf1,fwkerhalf2,fwkerhalf3,ms,mt,mu,fk,nf1,nf2,nf3,fw);
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());

  fftw_free(fw); fftw_free(fwkerhalf1); fftw_free(fwkerhalf2); fftw_free(fwkerhalf3);
  if (opts.debug) printf("freed\n");
  return 0;
}

int finufft3d2(BIGINT nj,double* xj,double *yj,double *zj,double* cj,
	       int iflag,double eps, BIGINT ms, BIGINT mt, BIGINT mu,
	       double* fk, nufft_opts opts)

 /*  Type-2 3D complex nonuniform FFT.

     cj[j] =    SUM   fk[k1,k2,k3] exp(+/-i (k1 xj[j] + k2 yj[j] + k3 zj[j]))
             k1,k2,k3
      for j = 0,...,nj-1
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 
                       -mu/2 <= k3 <= (mu-1)/2

   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj,zj     x,y,z locations of sources on 3D domain [-pi,pi]^3.
     fk     complex Fourier transform values (size ms*mt*mu, increasing fastest
            in ms to slowest in mu, ie Fortran ordering),
            stored as alternating Re & Im parts (2*ms*mt*mu doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt,mu  numbers of Fourier modes given in x,y,z; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 2/2/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  BIGINT nf2 = set_nf(mt,opts,spopts);
  BIGINT nf3 = set_nf(mu,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("3d2: (ms,mt,mu)=(%ld,%ld,%ld) (nf1,nf2,nf3)=(%ld,%ld,%ld) nj=%d ...\n",ms,mt,mu,nf1,nf2,nf3,nj);

  // STEP 0: get DCT of half of spread kernel in each dim, since real symm:
  CNTime timer; timer.start();
  double *fwkerhalf1 = fftw_alloc_real(nf1/2+1);
  double *fwkerhalf2 = fftw_alloc_real(nf2/2+1);
  double *fwkerhalf3 = fftw_alloc_real(nf3/2+1);
  double prefac_unused_dims;
  onedim_dct_kernel(nf1, fwkerhalf1, prefac_unused_dims, spopts);
  onedim_dct_kernel(nf2, fwkerhalf2, prefac_unused_dims, spopts);
  onedim_dct_kernel(nf3, fwkerhalf3, prefac_unused_dims, spopts); //prefacs same
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel dct (ns=%d):\t %.3g s\n", spopts.nspread,t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1*nf2*nf3); // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_3d(nf3,nf2,nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  double prefac = 1.0;
  deconvolveshuffle3d(2,prefac,fwkerhalf1,fwkerhalf2,fwkerhalf3,ms,mt,mu,fk,nf1,nf2,nf3,fw);
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n",timer.elapsedsec());

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n",nth,timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  int ier_spread = twopispread3d(nf1,nf2,nf3,(double*)fw,nj,xj,yj,zj,cj,2,params,opts.spread_debug);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());

  fftw_free(fw);
  fftw_free(fwkerhalf1); fftw_free(fwkerhalf2); fftw_free(fwkerhalf3);
  if (opts.debug) printf("freed\n");
  return ier_spread;
}
