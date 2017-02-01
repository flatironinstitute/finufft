#include "finufft.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts)
 /*  Type-1 1D complex nonuniform FFT.

               1 nj-1
     fk(k1) = -- SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=0                            
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 1/22/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d1: ms=%d nf1=%d nj=%d ...\n",ms,nf1,nj); 

  // STEP 0: get DCT of half of spreading kernel, since it's real symmetric
  CNTime timer; timer.start();
  double *fwkerhalf = fftw_alloc_real(nf1/2+1);
  double prefac_unused_dims;
  onedim_dct_kernel(nf1, fwkerhalf, prefac_unused_dims, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("onedim_dct_kernel:\t %.3g s\n", t);
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,1,params,opts.spread_debug);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d thr):\t\t %.3g s\n", nth, timer.elapsedsec());
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims*nj);  // 1/nj norm
  deconvolveshuffle1d(1,prefac,fwkerhalf,ms,fk,nf1,fw);
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());
  //for (int j=0;j<ms;++j) cout<<fk[2*j]<<"\t"<<fk[2*j+1]<<endl;

  fftw_free(fw); fftw_free(fwkerhalf); if (opts.debug) printf("freed\n");
  return 0;
}


int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts)
 /*  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
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

   Written with FFTW style complex arrays. Barnett 1/25/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d2: ms=%d nf1=%d nj=%d ...\n",ms,nf1,nj); 

  // STEP 0: get DCT of half of spreading kernel, since it's real symmetric
  CNTime timer; timer.start();
  double *fwkerhalf = fftw_alloc_real(nf1/2+1);
  double prefac_unused_dims;
  onedim_dct_kernel(nf1, fwkerhalf, prefac_unused_dims, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("onedim_dct_kernel:\t %.3g s\n", t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE); // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims);
  deconvolveshuffle1d(2,prefac,fwkerhalf,ms,fk,nf1,fw);
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n",timer.elapsedsec());
  //cout<<"fw:\n"; for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d thr):\t\t %.3g s\n",nth,timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,2,params,opts.spread_debug);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());

  fftw_free(fw); fftw_free(fwkerhalf); if (opts.debug) printf("freed\n");
  return ier_spread;
}
