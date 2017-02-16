#include "finufft.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft2d1(BIGINT nj,double* xj,double *yj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, double* fk, nufft_opts opts)
 /*  Type-1 2D complex nonuniform FFT.

                  1  nj-1
     f[k1,k2] =  --  SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                 nj  j=0

     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is in increasing k1 ordering (fast), then increasing
     k2 ordering (slow). If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex Fourier transform values (size ms*mt, increasing fast in ms
            then slow in mt, ie Fortran ordering),
            stored as alternating Re & Im parts (2*ms*mt doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 2/1/17
 */
{
  spread_opts spopts;
  double params[4];
  int ier_set = setup_kernel(spopts,params,eps);
  BIGINT nf1 = set_nf(ms,opts,spopts);
  BIGINT nf2 = set_nf(mt,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d1: (ms,mt)=(%d,%d) (nf1,nf2)=(%d,%d) nj=%d ...\n",ms,mt,nf1,nf2,nj); 

  // STEP 0: get DCT of half of spread kernel in each dim, since real symm:
  CNTime timer; timer.start();
  double *fwkerhalf1 = fftw_alloc_real(nf1/2+1);
  double *fwkerhalf2 = fftw_alloc_real(nf2/2+1);
  onedim_fseries_kernel(nf1, fwkerhalf1, spopts);
  onedim_fseries_kernel(nf2, fwkerhalf2, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1*nf2);  // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_2d(nf2,nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  int ier_spread = twopispread2d(nf1,nf2,(double*)fw,nj,xj,yj,cj,1,params,opts.spread_debug);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  deconvolveshuffle2d(1,1.0/nj,fwkerhalf1,fwkerhalf2,ms,mt,fk,nf1,nf2,fw);  // 1/nj prefac
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());

  fftw_free(fw); fftw_free(fwkerhalf1); fftw_free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return 0;
}

int finufft2d2(BIGINT nj,double* xj,double *yj,double* cj,int iflag,double eps,
	       BIGINT ms, BIGINT mt, double* fk, nufft_opts opts)

 /*  Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2 
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 

   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     fk     complex Fourier transform values (size ms*mt, increasing fast in ms
            then slow in mt, ie Fortran ordering),
            stored as alternating Re & Im parts (2*ms*mt doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms,mt  numbers of Fourier modes given in x and y; each may be even or odd;
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

   Written with FFTW style complex arrays. Barnett 2/1/17
 */
{
  spread_opts spopts;
  double params[4];
  int ier_set = setup_kernel(spopts,params,eps);
  BIGINT nf1 = set_nf(ms,opts,spopts);
  BIGINT nf2 = set_nf(mt,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d2: (ms,mt)=(%ld,%ld) (nf1,nf2)=(%ld,%ld) nj=%d ...\n",ms,mt,nf1,nf2,nj); 

  // STEP 0: get DCT of half of spread kernel in each dim, since real symm:
  CNTime timer; timer.start();
  double *fwkerhalf1 = fftw_alloc_real(nf1/2+1);
  double *fwkerhalf2 = fftw_alloc_real(nf2/2+1);
  onedim_fseries_kernel(nf1, fwkerhalf1, spopts);
  onedim_fseries_kernel(nf2, fwkerhalf2, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1*nf2);  // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_2d(nf2,nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  deconvolveshuffle2d(2,1.0,fwkerhalf1,fwkerhalf2,ms,mt,fk,nf1,nf2,fw);
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n",timer.elapsedsec());
  //cout<<"fw:\n"; for (int j=0;j<nf1*nf2;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n",nth,timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  int ier_spread = twopispread2d(nf1,nf2,(double*)fw,nj,xj,yj,cj,2,params,opts.spread_debug);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());

  fftw_free(fw); fftw_free(fwkerhalf1); fftw_free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return ier_spread;
}
